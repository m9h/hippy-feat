"""GLMdenoise — adaptive PCA-based noise regressor extraction.

Reference:
  Kay KN, Rokem A, Winawer J, Dougherty RF, Wandell BA (2013). GLMdenoise:
  a fast, automated technique for denoising task-based fMRI data. Frontiers
  in Neuroscience 7:247.

Core idea:
  1. Fit a baseline GLM, get per-voxel cross-validated R².
  2. Identify a "noise pool" = voxels with consistently LOW R² across runs
     (definitely not task-responsive).
  3. PCA on the noise-pool timeseries → top-K principal components capture
     systematic noise (drift, motion residuals, physiological).
  4. Refit the GLM with those K components as nuisance regressors.
  5. Choose K via leave-one-run-out cross-validation on R² of the held-out
     task variance.

Stage 2 of the GLMsingle pipeline. Task 2.1 finding: this stage and stage 3
(fracridge) are where most of GLMsingle's win comes from — NOT the per-voxel
HRF library.

Real-time use: K-fold CV is offline. The PRODUCED noise components, however,
can be streamed: precompute components from a calibration block (recognition
run), then add as fixed regressors during feedback runs.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


def per_voxel_r2(Y: np.ndarray, X: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """Per-voxel coefficient of determination R² = 1 - RSS / TSS.

    Args:
        Y:     (V, T)   voxel timeseries.
        X:     (T, P)   design matrix used for the fit.
        betas: (V, P)   estimated regression coefficients.

    Returns:
        (V,) per-voxel R².
    """
    pred = betas @ X.T
    rss = ((Y - pred) ** 2).sum(axis=1)
    tss = ((Y - Y.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return 1.0 - rss / (tss + 1e-12)


def select_noise_pool(per_run_r2: np.ndarray, threshold: float = 0.0,
                      max_pool_frac: float = 0.5) -> np.ndarray:
    """Identify voxels in the noise pool by cross-run R² consistency.

    Args:
        per_run_r2: (n_runs, V) — R² of the baseline GLM per run, per voxel.
        threshold: voxels with mean R² below this threshold are pool candidates.
        max_pool_frac: cap pool size to this fraction of total voxels.

    Returns:
        (V,) boolean mask: True = in noise pool.

    A voxel must have LOW R² in EVERY run to qualify (otherwise it's likely
    task-responsive in at least one run). The threshold defaults to 0; pool
    is the bottom `max_pool_frac` of voxels by mean R² subject to the
    threshold constraint.
    """
    n_runs, V = per_run_r2.shape
    mean_r2 = per_run_r2.mean(axis=0)
    below = mean_r2 < threshold
    # Also require per-run consistency: max R² across runs must be low
    max_r2 = per_run_r2.max(axis=0)
    eligible = below & (max_r2 < max(threshold, 0.0) + 0.05)
    if eligible.sum() == 0:
        # Fallback: just take the bottom-K by mean R²
        cutoff = int(np.floor(V * max_pool_frac))
        order = np.argsort(mean_r2)
        mask = np.zeros(V, dtype=bool)
        mask[order[:cutoff]] = True
        return mask
    # Keep the bottom max_pool_frac of eligible voxels
    n_keep = min(int(eligible.sum()), int(np.floor(V * max_pool_frac)))
    eligible_idx = np.where(eligible)[0]
    sorted_idx = eligible_idx[np.argsort(mean_r2[eligible_idx])]
    pool = np.zeros(V, dtype=bool)
    pool[sorted_idx[:n_keep]] = True
    return pool


def extract_noise_components(noise_pool_ts: np.ndarray, max_K: int = 30
                              ) -> tuple[np.ndarray, np.ndarray]:
    """PCA on the noise-pool timeseries → top-`max_K` temporal components.

    Args:
        noise_pool_ts: (V_pool, T) noise-pool voxel timeseries (mean-removed
            per voxel; we re-center inside).
        max_K: cap on number of components to return.

    Returns:
        components: (T, K) top-K temporal principal components, unit-norm.
        explained_var: (K,) singular-value-squared / total variance.
    """
    Y = noise_pool_ts - noise_pool_ts.mean(axis=1, keepdims=True)
    # SVD: Y = U Σ V^T → temporal components are rows of V^T
    # Y is (V_pool, T) so temporal modes are columns of V (full SVD form)
    U, S, Vt = np.linalg.svd(Y, full_matrices=False)
    K = min(max_K, len(S))
    components = Vt[:K].T                                  # (T, K) unit-norm
    explained_var = (S[:K] ** 2) / (S ** 2).sum()
    return components.astype(np.float32), explained_var.astype(np.float32)


class GLMdenoiseResult(NamedTuple):
    """Bundle of artifacts produced by `glmdenoise_fit`."""
    K_chosen: int
    noise_components: np.ndarray            # (T, K_chosen)
    noise_pool_mask: np.ndarray             # (V,)
    augmented_design: np.ndarray            # (T, P + K_chosen)
    cv_curve: np.ndarray                    # (max_K + 1,) median R² over voxels
    explained_variance: np.ndarray          # (max_K,)


def glmdenoise_fit(Y_per_run: list[np.ndarray],
                    X_per_run: list[np.ndarray],
                    fit_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    max_K: int = 30,
                    noise_pool_threshold: float = 0.0,
                    ) -> GLMdenoiseResult:
    """Run GLMdenoise over a list of runs.

    `fit_fn(X, Y) -> betas` is a callable that fits the GLM (caller-provided
    so this works for OLS, ridge, AR(1) prewhitened, etc.).

    Args:
        Y_per_run:  list of (V, T_r) for r runs.
        X_per_run:  list of (T_r, P) design matrices.
        fit_fn:     fitting function — `fit_fn(X, Y) -> betas` of shape (V, P).
        max_K:      maximum number of noise components to consider.
        noise_pool_threshold: R² threshold for noise-pool eligibility.

    Returns:
        GLMdenoiseResult with chosen K, noise components, augmented design.

    Note: input timeseries should ALREADY be voxelwise z-scored or
    detrended to your taste; this routine doesn't apply additional drift
    regressors beyond what's in `X_per_run`.
    """
    n_runs = len(Y_per_run)
    if n_runs < 2:
        raise ValueError("GLMdenoise needs ≥ 2 runs for cross-validation")

    V = Y_per_run[0].shape[0]

    # Step 1: baseline GLM per run → per-run R²
    print(f"[glmdenoise] step 1/4: baseline GLM on {n_runs} runs")
    per_run_betas = []
    per_run_r2 = np.zeros((n_runs, V), dtype=np.float32)
    for r, (Y, X) in enumerate(zip(Y_per_run, X_per_run)):
        betas = fit_fn(X, Y)
        per_run_betas.append(betas)
        per_run_r2[r] = per_voxel_r2(Y, X, betas)
    print(f"  per-run R² mean: {per_run_r2.mean():.3f}, "
          f"min: {per_run_r2.min():.3f}, max: {per_run_r2.max():.3f}")

    # Step 2: noise pool selection
    print(f"[glmdenoise] step 2/4: identify noise pool")
    pool_mask = select_noise_pool(per_run_r2, threshold=noise_pool_threshold)
    n_pool = int(pool_mask.sum())
    print(f"  noise pool: {n_pool}/{V} voxels ({100 * n_pool / V:.1f}%)")
    if n_pool < 10:
        raise RuntimeError(
            f"Noise pool too small ({n_pool} voxels) — adjust threshold "
            f"or check that runs have variable activation patterns"
        )

    # Step 3: PCA on noise-pool timeseries to get K components per run
    # Then concatenate runs to extract a common set of components
    print(f"[glmdenoise] step 3/4: PCA on noise pool (max K = {max_K})")
    pool_ts = np.concatenate(
        [Y[pool_mask] for Y in Y_per_run], axis=1
    )  # (n_pool, sum(T_r))
    components_concat, explained_var = extract_noise_components(pool_ts, max_K)
    print(f"  top component explained variance: "
          f"{explained_var[:5].round(3).tolist()}")

    # Step 4: leave-one-run-out CV on K
    # For each candidate K, augment the design with K components, refit per
    # run, score on held-out runs.
    print(f"[glmdenoise] step 4/4: LOO-CV over K ∈ [0, {max_K}]")
    # Per-run components: split components_concat back into per-run
    starts = np.cumsum([0] + [Y.shape[1] for Y in Y_per_run])
    cv_curve = []
    for K in range(0, max_K + 1):
        if K == 0:
            r2_held = per_run_r2.mean()
            cv_curve.append(r2_held)
            continue
        held_r2 = []
        for held in range(n_runs):
            train_runs = [r for r in range(n_runs) if r != held]
            # Augment held-out run's design
            X_h = X_per_run[held]
            comps_h = components_concat[starts[held]:starts[held + 1], :K]
            X_h_aug = np.concatenate([X_h, comps_h], axis=1)
            beta_h = fit_fn(X_h_aug, Y_per_run[held])
            held_r2.append(per_voxel_r2(Y_per_run[held], X_h_aug, beta_h).mean())
        cv_curve.append(float(np.mean(held_r2)))
    cv_curve = np.asarray(cv_curve)
    K_chosen = int(np.argmax(cv_curve))
    print(f"  CV-optimal K = {K_chosen}  (R² @ K=0: {cv_curve[0]:.4f}, "
          f"@ K={K_chosen}: {cv_curve[K_chosen]:.4f})")

    # Build the final augmented design (per-run; caller stitches together).
    # We return the FIRST run's augmented design as a reference template;
    # caller should reconstruct per-run if needed.
    X_aug_first = np.concatenate(
        [X_per_run[0], components_concat[:starts[1], :K_chosen]], axis=1,
    )
    return GLMdenoiseResult(
        K_chosen=K_chosen,
        noise_components=components_concat[:, :K_chosen],
        noise_pool_mask=pool_mask,
        augmented_design=X_aug_first,
        cv_curve=cv_curve,
        explained_variance=explained_var,
    )


__all__ = [
    "GLMdenoiseResult",
    "extract_noise_components",
    "glmdenoise_fit",
    "per_voxel_r2",
    "select_noise_pool",
]
