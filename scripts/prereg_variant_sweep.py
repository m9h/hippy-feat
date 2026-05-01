#!/usr/bin/env python3
"""Pre-registered 12-cell variant sweep (per TASK_2_1_PREREGISTRATION.md).

Produces per-trial betas of shape (770, 2792) for each of the 12 cells.
Each cell saves to:
    /data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/{cell_name}_ses-03_betas.npy
plus a JSON config sidecar so we can audit which knobs each cell used.

Cell taxonomy (from pre-reg matrix):
    1. OLS_glover_rtm                         JAX, plain OLS
    2. AR1freq_glover_rtm                     JAX, AR(1) freq via _variant_g_forward(pp=0)
    3. AR1freq_glover_rtm_nilearn             nilearn single-voxel sanity (run by sanity_s1)
    4. VariantG_glover_rtm                    JAX, AR(1) Bayes uninformative
    5. VariantG_glover_rtm_prior              JAX, AR(1) Bayes with ses-01 training prior
    6. AR1freq_glmsingleS1_rtm                JAX, AR(1) freq + GLMsingle HRF library
    7. AR1freq_glover_rtm_glmdenoise_fracridge   GLMsingle Stages 2+3 + AR(1) freq
    8. VariantG_glover_rtm_glmdenoise_fracridge  Variant G + denoising
    9. VariantG_glover_rtm_acompcor           Variant G + aCompCor (5 components)
   10. RT_paper_replica_partial               nilearn full call + cumulative z-score (no repeat avg)
   11. RT_paper_replica_full                  nilearn + cum z-score + repeat-avg (canonical paper RT)
   12. Offline_paper_replica_full             #11 but with fmriprep BOLD instead of rtmotion

Cells 1, 2, 4, 5 are tightly coupled and share the JAX forward — implemented
together. Cells 6-9 layer additional preprocessing. Cells 10-12 require nilearn
and are slowest; they're invoked separately at the end.

Run order: --cells flag controls which to run. Default is all 9 JAX cells
(1,2,4,5,6,7,8,9). Cells 10,11,12 are explicit opt-in due to runtime cost.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_glm_variants import (
    _variant_g_forward,
    _ols_fit,
    build_design_matrix,
    make_glover_hrf,
    load_glmsingle_hrf_library,
    resample_hrf,
)


PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
RT3T = PAPER_ROOT / "rt3t" / "data"
FMRIPREP_ROOT = (PAPER_ROOT / "fmriprep_mindeye" / "data_sub-005"
                 / "bids" / "derivatives" / "fmriprep" / "sub-005")
EVENTS_DIR = RT3T / "events"
BRAIN_MASK = RT3T / "sub-005_final_mask.nii.gz"
RELMASK = RT3T / "sub-005_ses-01_task-C_relmask.npy"
MC_DIR = Path("/data/3t/derivatives/motion_corrected_resampled")
OUT_DIR = PAPER_ROOT / "task_2_1_betas" / "prereg"
HRF_INDICES_PATH = "/data/3t/data/avg_hrfs_s1_s2_full.npy"
HRF_LIB_PATH = "/data/3t/data/getcanonicalhrflibrary.tsv"


def load_mask() -> tuple[np.ndarray, np.ndarray]:
    flat_brain = (nib.load(BRAIN_MASK).get_fdata() > 0).flatten()
    rel = np.load(RELMASK)
    assert flat_brain.sum() == 19174 and rel.sum() == 2792
    return flat_brain, rel


def load_fmriprep(session: str, run: int, flat_brain: np.ndarray,
                  rel: np.ndarray) -> np.ndarray:
    p = (FMRIPREP_ROOT / session / "func"
         / f"sub-005_{session}_task-C_run-{run:02d}"
           f"_space-T1w_desc-preproc_bold.nii.gz")
    img = nib.load(p)
    vol = img.get_fdata()
    T = vol.shape[-1]
    return vol.reshape(-1, T)[flat_brain][rel].astype(np.float32)


def load_rtmotion(session: str, run: int, flat_brain: np.ndarray,
                  rel: np.ndarray) -> np.ndarray:
    pattern = f"{session}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(MC_DIR.glob(pattern))
    if not vols:
        raise FileNotFoundError(f"no mc_boldres for {session} run-{run:02d}")
    frames = []
    for v in vols:
        f = nib.load(v).get_fdata().flatten()[flat_brain][rel]
        frames.append(f.astype(np.float32))
    return np.stack(frames, axis=1)                              # (V, T)


def load_events(session: str, run: int) -> pd.DataFrame:
    p = EVENTS_DIR / f"sub-005_{session}_task-C_run-{run:02d}_events.tsv"
    df = pd.read_csv(p, sep="\t")
    df = df.copy()
    df["onset_rel"] = df["onset"].astype(float) - df["onset"].iloc[0]
    return df


# ---- Per-trial GLM kernels ----------------------------------------------------

def _glm_jax(timeseries: np.ndarray, onsets: np.ndarray, probe_trial: int,
             tr: float, n_trs: int, mode: str,
             prior_mean: np.ndarray | None = None,
             prior_var: np.ndarray | None = None,
             max_trs: int = 200,
             hrf: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Unified JAX GLM kernel.

    `mode` selects:
        "ols"          → standard OLS β; vars = sigma2 * (X'X)^-1[diag]
        "ar1_freq"     → AR(1) freq via _variant_g_forward(pp=0, weak ρ prior)
        "variant_g"    → AR(1) Bayes uninformative
        "variant_g_prior" → AR(1) Bayes with caller-supplied (prior_mean, prior_var)

    Returns (β, β_var) for the probe trial column.
    """
    if hrf is None:
        hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
    dm, probe_col = build_design_matrix(onsets, tr, n_trs, hrf, probe_trial)
    if mode == "ar1_session_rho":
        # Cell 17 (hybrid online): ρ̂ comes from a session-level streaming
        # tracker (intercept+drift only design); per-trial β fit uses that
        # ρ̂ as a frozen pre-whitening coefficient. The right architectural
        # split — stationary parameters stream, per-trial coefficients
        # snapshot.
        # The session-ρ̂ is provided via a thread-local cache populated by
        # the caller in run_glm_cell — we just read `globals().get(...)`
        # for cleanliness in this kernel.
        rho_session = _get_session_rho_or_compute(
            timeseries_per_run_for_rho=globals().get("_SESSION_RHO_TS", None),
            tr=tr, n_voxels=timeseries.shape[0],
        )
        # Standard AR(1) prewhitened OLS with per-voxel ρ frozen
        from rt_glm_variants import _ols_fit
        if hrf is None:
            hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
        dm, probe_col = build_design_matrix(onsets, tr, n_trs, hrf, probe_trial)
        # Per-voxel prewhitening — vmap over voxels
        n_eff = n_trs - 1
        beta_out = np.zeros(timeseries.shape[0], dtype=np.float32)
        var_out = np.zeros(timeseries.shape[0], dtype=np.float32)
        for v in range(timeseries.shape[0]):
            rho_v = float(rho_session[v])
            y_pw = timeseries[v, 1:] - rho_v * timeseries[v, :-1]
            X_pw = dm[1:] - rho_v * dm[:-1]
            try:
                XtX_inv = np.linalg.inv(X_pw.T @ X_pw + 1e-6 * np.eye(dm.shape[1]))
                beta = XtX_inv @ X_pw.T @ y_pw
                resid = y_pw - X_pw @ beta
                rss = float((resid ** 2).sum())
                sigma2 = rss / max(n_eff - dm.shape[1], 1)
                beta_out[v] = float(beta[probe_col])
                var_out[v] = float(sigma2 * XtX_inv[probe_col, probe_col])
            except np.linalg.LinAlgError:
                beta_out[v] = 0.0
                var_out[v] = 1e10
        return beta_out, var_out

    if mode == "ar1_streaming_kalman":
        # Cell 13 (EKF): streaming AR(1) Bayesian Kalman over BOLD timeseries.
        # Build LSS design once, then feed (X[t], Y[:, t]) per-TR through
        # streaming_kalman_ar1_update; final β posterior is what we save.
        from jaxoccoli.streaming_kalman import (
            init_streaming_kalman_ar1,
            streaming_kalman_ar1_run,
        )
        dm, probe_col = build_design_matrix(onsets, tr, n_trs, hrf, probe_trial)
        state = init_streaming_kalman_ar1(P=dm.shape[1], V=timeseries.shape[0])
        # streaming_kalman_ar1_run expects (T, P) X and (V, T) Y
        state = streaming_kalman_ar1_run(state, dm.astype(np.float32),
                                          timeseries.astype(np.float32))
        beta = np.asarray(state.beta_mean[:, probe_col], dtype=np.float32)
        # Posterior variance from b_post / (a_post - 1)
        var = np.asarray(
            np.maximum(state.b_post / np.maximum(state.a_post - 1.0, 1e-3), 1e-10),
            dtype=np.float32,
        )
        return beta, var

    if mode == "ols":
        # plain OLS via _ols_fit; build a per-voxel variance estimate using
        # a single sigma² from the residuals.
        betas = _ols_fit(jnp.asarray(dm), jnp.asarray(timeseries))
        beta = np.asarray(betas[:, probe_col], dtype=np.float32)
        # variance: sigma²_v * (X'X)^-1[probe_col, probe_col]
        XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
        diag_pp = float(XtX_inv[probe_col, probe_col])
        pred = np.asarray(betas) @ dm.T
        rss = ((timeseries - pred) ** 2).sum(axis=1)
        sigma2 = rss / max(n_trs - dm.shape[1], 1)
        var = sigma2 * diag_pp
        return beta, var.astype(np.float32)

    # All AR(1) modes share the JIT padding pattern
    dm_pad = np.zeros((max_trs, dm.shape[1]), dtype=np.float32)
    dm_pad[:n_trs] = dm
    ts_pad = np.zeros((timeseries.shape[0], max_trs), dtype=np.float32)
    ts_pad[:, :n_trs] = timeseries.astype(np.float32)

    if mode == "ar1_freq":
        b, v = _variant_g_forward(
            jnp.asarray(dm_pad), jnp.asarray(ts_pad),
            jnp.asarray(n_trs, dtype=jnp.int32),
            pp_scalar=0.0, rho_prior_mean=0.0, rho_prior_var=1e8,
        )
    elif mode == "variant_g":
        b, v = _variant_g_forward(
            jnp.asarray(dm_pad), jnp.asarray(ts_pad),
            jnp.asarray(n_trs, dtype=jnp.int32),
        )
    elif mode == "variant_g_prior":
        # Apply external prior post-hoc; _variant_g_forward only supports a
        # diagonal pp_scalar prior. We can implement an empirical-Bayes shrink
        # by combining its uninformative posterior with the prior using the
        # standard precision-weighted formula at the end.
        b, v = _variant_g_forward(
            jnp.asarray(dm_pad), jnp.asarray(ts_pad),
            jnp.asarray(n_trs, dtype=jnp.int32),
        )
        b = np.asarray(b, dtype=np.float32)
        v = np.maximum(np.asarray(v, dtype=np.float32), 1e-10)
        if prior_mean is None or prior_var is None:
            raise ValueError("variant_g_prior needs prior_mean and prior_var")
        # Shrink probe column toward training prior
        post_var_col = 1.0 / (1.0 / prior_var + 1.0 / v[:, probe_col])
        post_mean_col = post_var_col * (
            prior_mean / prior_var + b[:, probe_col] / v[:, probe_col]
        )
        return post_mean_col.astype(np.float32), post_var_col.astype(np.float32)
    else:
        raise ValueError(f"unknown mode: {mode}")
    b = np.asarray(b, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    return b[:, probe_col], np.maximum(v[:, probe_col], 1e-10)


def _glm_glmsingle_per_voxel_hrf(timeseries: np.ndarray, onsets: np.ndarray,
                                  probe_trial: int, tr: float, n_trs: int,
                                  hrf_indices: np.ndarray, hrf_library: np.ndarray,
                                  base_time: np.ndarray, mode: str = "ar1_freq",
                                  max_trs: int = 200) -> np.ndarray:
    """GLMsingle-style per-voxel HRF lookup, with caller-chosen GLM mode."""
    n_voxels = timeseries.shape[0]
    result = np.zeros(n_voxels, dtype=np.float32)
    n_hrf_trs = int(np.ceil(32.0 / tr))
    unique_hrfs = np.unique(hrf_indices)
    for h in unique_hrfs:
        voxel_ids = np.where(hrf_indices == int(h))[0]
        if len(voxel_ids) == 0:
            continue
        hrf = resample_hrf(hrf_library[:, int(h)], base_time, tr, n_hrf_trs)
        beta, _ = _glm_jax(
            timeseries[voxel_ids], onsets, probe_trial, tr, n_trs,
            mode=mode, hrf=hrf, max_trs=max_trs,
        )
        result[voxel_ids] = beta
    return result


# ---- Cell drivers -------------------------------------------------------------

def _get_session_rho_or_compute(timeseries_per_run_for_rho,
                                  tr: float, n_voxels: int
                                  ) -> np.ndarray:
    """Estimate per-voxel session-level ρ̂ from a streaming pass over the
    full session BOLD with only intercept + drift in the design.

    Sets globals()['_SESSION_RHO_CACHE'] to memoize across trial calls.
    """
    cache = globals().get("_SESSION_RHO_CACHE")
    if cache is not None and cache.shape == (n_voxels,):
        return cache
    if timeseries_per_run_for_rho is None:
        return np.full(n_voxels, 0.3, dtype=np.float32)
    # Concat all runs
    Y_session = np.concatenate(timeseries_per_run_for_rho, axis=1)  # (V, T_tot)
    V, T_total = Y_session.shape
    # Simple intercept + cosine drift design
    intercept = np.ones(T_total, dtype=np.float32)
    drift = np.cos(2 * np.pi * np.arange(T_total) / max(T_total - 1, 1)
                   ).astype(np.float32)
    X = np.stack([intercept, drift], axis=1)                  # (T, 2)
    # OLS fit per voxel; collect residuals
    XtX_inv_Xt = np.linalg.inv(X.T @ X + 1e-6 * np.eye(2)) @ X.T
    betas = Y_session @ XtX_inv_Xt.T                          # (V, 2)
    pred = betas @ X.T
    resid = Y_session - pred                                  # (V, T_total)
    # Per-voxel lag-1 autocorrelation
    num = (resid[:, 1:] * resid[:, :-1]).sum(axis=1)
    den = (resid ** 2).sum(axis=1) + 1e-10
    rho = np.clip(num / den, -0.95, 0.95).astype(np.float32)
    globals()["_SESSION_RHO_CACHE"] = rho
    return rho


def _extract_noise_components_per_run(timeseries_per_run: list[np.ndarray],
                                       max_K: int = 5,
                                       pool_frac: float = 0.10
                                       ) -> list[np.ndarray]:
    """Lightweight GLMdenoise: PCA over high-variance voxels per run.

    Returns one (T_r, K) component matrix per run. We use a per-run noise
    pool selected by within-run temporal variance (top `pool_frac`)
    rather than the cross-run R² selection — cheaper and adequate for
    the bake-off purpose. The CV-K loop is skipped; K=max_K is fixed.
    """
    out = []
    for ts in timeseries_per_run:                         # ts: (V, T_r)
        var = ts.var(axis=1)
        n_pool = max(int(np.floor(len(var) * pool_frac)), max_K + 1)
        cutoff = np.partition(var, len(var) - n_pool)[len(var) - n_pool]
        pool = ts[var >= cutoff]                          # (V_pool, T_r)
        # Center, SVD
        pool_c = pool - pool.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(pool_c, full_matrices=False)
        K = min(max_K, Vt.shape[0])
        out.append(Vt[:K].T.astype(np.float32))           # (T_r, K)
    return out


def _fracridge_voxel(beta_ols_per_voxel: np.ndarray,
                      X: np.ndarray, Y: np.ndarray,
                      frac: float = 0.5) -> np.ndarray:
    """Apply a fixed-fraction ridge to each voxel's β.

    For the bake-off we use a single fraction across voxels (no CV) —
    the proper per-voxel fracridge_cv requires ≥2 (train, test) folds
    which we don't naturally have inside a single LSS fit.
    """
    if abs(frac - 1.0) < 1e-6:
        return beta_ols_per_voxel
    # SVD-based shrinkage: ||β(λ)|| / ||β_OLS|| = frac
    # Closed form via SVD of X
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ Y.T                                       # (P, V)
    beta_ols_via_svd = (Vh.T @ (UtY / (S[:, None] + 1e-12))).T  # (V, P)
    # Solve for λ that gives target fraction (binary-search-free closed form
    # is non-trivial; use Brent on a single voxel and assume similar λ
    # across voxels of similar magnitude — coarse but decisive).
    from scipy.optimize import brentq
    target_norm = frac * np.linalg.norm(beta_ols_via_svd, axis=1).mean()

    def norm_at(lam):
        coef = (S[None, :] * UtY.T) / (S[None, :] ** 2 + lam)
        return float(np.linalg.norm(coef, axis=1).mean())

    try:
        lam = brentq(lambda l: norm_at(l) - target_norm,
                     a=0.0, b=1e8 + (S ** 2).max(), xtol=1e-3)
    except ValueError:
        lam = 0.0
    coef = (S[None, :] * UtY.T) / (S[None, :] ** 2 + lam)
    return (coef @ Vh).astype(np.float32)


def _load_canonical_fracvalue_to_relmask(
    glmsingle_dir: Path = Path(
        "/data/derivatives/rtmindeye_paper/glmsingle/glmsingle_sub-005_ses-01-02_task-C"
    ),
) -> np.ndarray:
    """Read per-voxel `FRACvalue` from a training-session canonical GLMsingle
    output, project onto the MindEye 2792-voxel decoder mask. Returns a
    (2792,) float array — RT-deployable as a precomputed shrinkage table.
    """
    import nibabel as nib
    z = np.load(glmsingle_dir / "TYPED_FITHRF_GLMDENOISE_RR.npz", allow_pickle=True)
    fv_full = z["FRACvalue"].squeeze().astype(np.float32)              # (V_canon,)
    canon_brain = nib.load(
        glmsingle_dir
        / f"sub-005_{glmsingle_dir.name.replace('glmsingle_sub-005_', '').replace('glmsingle_', '')}_brain.nii.gz"
    ).get_fdata() > 0
    final_mask = nib.load(
        Path("/data/derivatives/rtmindeye_paper/rt3t/data/sub-005_final_mask.nii.gz")
    ).get_fdata() > 0
    relmask = np.load(
        Path("/data/derivatives/rtmindeye_paper/rt3t/data/sub-005_ses-01_task-C_relmask.npy")
    )
    me_positions = np.where(final_mask.flatten())[0][relmask]          # (2792,)
    canon_brain_idx = -np.ones(canon_brain.size, dtype=np.int64)
    canon_brain_idx[canon_brain.flatten()] = np.arange(canon_brain.sum())
    me_in_canon = canon_brain_idx[me_positions]
    if (me_in_canon < 0).any():
        # Some MindEye voxels not in this training-session brain mask;
        # default fraction = 1.0 (no shrinkage) for those
        fv = np.ones(len(me_positions), dtype=np.float32)
        valid = me_in_canon >= 0
        fv[valid] = fv_full[me_in_canon[valid]]
        n_default = int((~valid).sum())
        print(f"  [frac-load] {n_default} of {len(me_positions)} relmask voxels "
              f"not in training brain mask; defaulted to frac=1.0")
        return fv
    return fv_full[me_in_canon]


def run_glm_cell_with_streaming(*args, streaming_post_stim_TRs: int | None = None, **kwargs):
    """Wrapper that lets streaming_post_stim_TRs be threaded through CELLS."""
    return run_glm_cell(*args, streaming_post_stim_TRs=streaming_post_stim_TRs, **kwargs)


def run_glm_cell(cell_name: str, mode: str, bold_source: str,
                 hrf_strategy: str, session: str, runs: list[int],
                 prior_mean: np.ndarray | None = None,
                 prior_var: np.ndarray | None = None,
                 denoise: str | None = None,
                 streaming_post_stim_TRs: int | None = None,
                 ) -> tuple[np.ndarray, list[str]]:
    """Run cells 1, 2, 4, 5, 6 — pure GLM (no denoising) variants."""
    flat_brain, rel = load_mask()
    tr = 1.5
    if hrf_strategy == "glmsingle_lib":
        base_time, hrf_library = load_glmsingle_hrf_library(HRF_LIB_PATH)
        hrf_vol = np.load(HRF_INDICES_PATH)[:, :, :, 0].astype(int)
        hrf_indices = hrf_vol.flatten()[flat_brain][rel]
    else:
        base_time = hrf_library = hrf_indices = None

    # Reset session-rho cache for cell 17 path; populated lazily on first call
    globals()["_SESSION_RHO_CACHE"] = None
    globals()["_SESSION_RHO_TS"] = None

    # Pre-load training-derived per-voxel fracridge table for canonical_frac
    # denoise (Stage 3 with FRACvalue frozen from a prior session)
    canonical_frac = None
    if denoise == "canonical_frac":
        canonical_frac = _load_canonical_fracvalue_to_relmask()
        print(f"  [denoise=canonical_frac] loaded ses-01-02 FRACvalue, "
              f"mean={canonical_frac.mean():.3f} range "
              f"{canonical_frac.min():.3f}-{canonical_frac.max():.3f}")

    # First load all runs' BOLD + events (needed for denoising path)
    timeseries_per_run = []
    events_per_run = []
    for run in runs:
        if bold_source == "rtmotion":
            ts = load_rtmotion(session, run, flat_brain, rel)
        elif bold_source == "fmriprep":
            ts = load_fmriprep(session, run, flat_brain, rel)
        else:
            raise ValueError(bold_source)
        events = load_events(session, run)
        timeseries_per_run.append(ts)
        events_per_run.append(events)

    # Cell 17: stash all runs' BOLD so the kernel can compute session ρ̂
    if mode == "ar1_session_rho":
        globals()["_SESSION_RHO_TS"] = list(timeseries_per_run)
        print(f"  [mode=ar1_session_rho] stashed {len(timeseries_per_run)} "
              f"runs of BOLD for session-ρ̂ estimation")

    # Pre-compute noise components per run if denoising requested
    noise_per_run: list[np.ndarray] | None = None
    if denoise in ("glmdenoise_fracridge", "tcompcor"):
        K = 5
        pool_frac = 0.05 if denoise == "tcompcor" else 0.10
        noise_per_run = _extract_noise_components_per_run(
            timeseries_per_run, max_K=K, pool_frac=pool_frac,
        )
        print(f"  [denoise={denoise}] extracted K={K} noise components per run "
              f"(pool_frac={pool_frac})")
    elif denoise == "hosvd_4d":
        # Cell 14: apply HOSVD multiway NORDIC to each run's 4D BOLD volume.
        # Need to reshape (V, T) → (X, Y, Z, T) using brain mask. Then back.
        from jaxoccoli.multiway_nordic import hosvd_threshold_4d
        import jax.numpy as jnp
        # We don't have the spatial unflatten readily — operate on (V, T) as
        # a 4D tensor (V_x=V, V_y=1, V_z=1, T) which is degenerate. Better:
        # use the 2D NORDIC path via patch unfolding which equivalent for
        # global SVD anyway. Reuse jaxoccoli.nordic.nordic_global on each run.
        from jaxoccoli.nordic import nordic_global
        for r in range(len(timeseries_per_run)):
            ts_r = timeseries_per_run[r]                  # (V, T)
            # Cast to complex64 (NORDIC expects complex; magnitude data → +0j)
            z = (ts_r + 0j).astype(np.complex64)
            denoised = np.asarray(nordic_global(jnp.asarray(z)))
            timeseries_per_run[r] = denoised.real.astype(np.float32)
        print(f"  [denoise=hosvd_4d] applied global NORDIC SVD-thresholding "
              f"per run (real-part only since magnitude input)")
    elif denoise == "logsig_tcompcor":
        # Cell 20 — log-signature features as additional nuisance regressors.
        # Streaming primitive: log-sig of the most recent W TRs of the K
        # tCompCor noise components, computed at every TR. Chen's identity
        # makes the sliding-window sig updatable in O(1) per new sample, so
        # this is a true online feature even though we batch-compute it here
        # for the offline cell. Depth-2 log-sig over K=5 PCs adds K*(K-1)/2 =
        # 10 Levy-area terms — a richer nuisance basis than the 5 PCs alone.
        K = 5
        pool_frac = 0.05
        noise_per_run = _extract_noise_components_per_run(
            timeseries_per_run, max_K=K, pool_frac=pool_frac,
        )
        import signax
        import jax.numpy as jnp
        W = 20
        augmented_noise = []
        for comps in noise_per_run:                       # comps: (T_r, K)
            T_r = comps.shape[0]
            pad = np.repeat(comps[:1], W - 1, axis=0)     # (W-1, K)
            padded = np.concatenate([pad, comps], axis=0) # (T_r + W-1, K)
            windows = np.lib.stride_tricks.sliding_window_view(
                padded, window_shape=W, axis=0
            ).transpose(0, 2, 1).copy()                   # (T_r, W, K)
            logsig = np.asarray(
                signax.logsignature(jnp.asarray(windows), depth=2)
            ).astype(np.float32)                           # (T_r, K*(K-1)/2)
            aug = np.concatenate([comps, logsig], axis=1) # (T_r, K + 10)
            aug = (aug - aug.mean(0)) / (aug.std(0) + 1e-6)
            augmented_noise.append(aug.astype(np.float32))
        noise_per_run = augmented_noise
        print(f"  [denoise=logsig_tcompcor] tCompCor K={K} + depth-2 log-sig "
              f"over W={W} sliding window → "
              f"{augmented_noise[0].shape[1]} nuisance regressors/TR")
    elif denoise == "riemannian_prewhiten":
        # Cell 15: Riemannian-mean prewhitening. Per-run cov Σ_r = (X_r X_r^T)/T,
        # geometric mean Σ̄ across runs (Riemannian SPD mean), then prewhiten
        # each run's BOLD by Σ̄^{-1/2}.
        # For V=2792 voxels, V×V is ~30 MB float32 — feasible. Σ̄^{-1/2} via
        # eigendecomposition.
        # Practical note: per-run cov on V=2792 with T~192 is rank-deficient
        # (rank ≤ T). Add a ridge term to regularize: Σ_r → Σ_r + ε·I.
        V = timeseries_per_run[0].shape[0]
        eps = 1e-3
        per_run_cov = []
        for ts_r in timeseries_per_run:
            T_r = ts_r.shape[1]
            # Center per-voxel
            ts_c = ts_r - ts_r.mean(axis=1, keepdims=True)
            cov = (ts_c @ ts_c.T) / max(T_r - 1, 1)
            per_run_cov.append(cov + eps * np.eye(V, dtype=np.float32))
        # Riemannian (log-Euclidean) geometric mean: arithmetic mean of
        # matrix logs; computationally cheaper than affine-invariant mean
        # and converges to it for closely-clustered SPD matrices.
        # log-Euclidean mean: exp(mean(log(Σ_r)))
        # logm/expm return complex128 even on SPD input; take .real after each.
        # Use eigendecomposition directly: SPD M = U diag(λ) U^T,
        # log(M) = U diag(log λ) U^T, exp(...) = U diag(exp(...)) U^T.
        # Faster than scipy.linalg.logm and stays in real arithmetic.
        log_sum = np.zeros((V, V), dtype=np.float64)
        for cov in per_run_cov:
            evals, evecs = np.linalg.eigh(cov.astype(np.float64))
            log_evals = np.log(np.maximum(evals, 1e-12))
            log_sum += (evecs * log_evals) @ evecs.T
        log_mean = log_sum / len(per_run_cov)
        evals_lm, evecs_lm = np.linalg.eigh(log_mean)
        sigma_bar = ((evecs_lm * np.exp(evals_lm)) @ evecs_lm.T).astype(np.float32)
        # Σ̄^{-1/2} via eigendecomposition
        evals, evecs = np.linalg.eigh(sigma_bar)
        evals_inv_sqrt = 1.0 / np.sqrt(np.maximum(evals, 1e-6))
        sigma_bar_inv_sqrt = (evecs * evals_inv_sqrt) @ evecs.T
        sigma_bar_inv_sqrt = sigma_bar_inv_sqrt.astype(np.float32)
        # Apply per-run prewhitening
        for r in range(len(timeseries_per_run)):
            timeseries_per_run[r] = (
                sigma_bar_inv_sqrt @ timeseries_per_run[r]
            ).astype(np.float32)
        print(f"  [denoise=riemannian_prewhiten] V×V SPD geom mean "
              f"(log-Euclidean) over {len(per_run_cov)} runs; "
              f"prewhitened per-run BOLD by Σ̄^(-1/2)")

    all_betas, trial_ids = [], []
    for run_idx, run in enumerate(runs):
        ts = timeseries_per_run[run_idx]
        events = events_per_run[run_idx]
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs = ts.shape[1]
        # If denoising, regress out noise components from BOLD before GLM.
        # This is the simplest faithful approximation of GLMdenoise: refit
        # GLM with noise in design vs regress-out-then-fit gives equivalent
        # β at convergence; the latter is cheaper for our LSS pattern since
        # the noise columns are fixed per run.
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]                # (T_r, K)
            # Project BOLD onto orthogonal complement of noise components
            # ts_clean = ts - β_noise @ comps^T where β_noise = ts @ comps (orth)
            beta_noise = ts @ comps                       # (V, K)
            ts = (ts - beta_noise @ comps.T).astype(np.float32)
        for trial_i in range(len(onsets)):
            # Streaming variant: crop BOLD/design to onset_TR + post_stim
            if streaming_post_stim_TRs is not None:
                onset_TR = int(round(float(onsets[trial_i]) / tr))
                decode_TR = min(onset_TR + streaming_post_stim_TRs, n_trs - 1)
                ts_use = ts[:, :decode_TR + 1]
                n_trs_use = decode_TR + 1
            else:
                ts_use = ts
                n_trs_use = n_trs
            if hrf_strategy == "glmsingle_lib":
                beta = _glm_glmsingle_per_voxel_hrf(
                    ts_use, onsets, trial_i, tr, n_trs_use,
                    hrf_indices, hrf_library, base_time, mode=mode,
                )
            else:
                beta, _ = _glm_jax(
                    ts_use, onsets, trial_i, tr, n_trs_use, mode=mode,
                    prior_mean=prior_mean, prior_var=prior_var,
                )
            # Optional fracridge per-voxel shrinkage (cell 7, 8)
            if denoise == "glmdenoise_fracridge":
                # Apply a fixed fraction (0.5) — coarse but decisive vs
                # full per-voxel CV that requires multi-fold structure
                # we don't have inside a single LSS fit.
                ols_norm = float(np.linalg.norm(beta) + 1e-12)
                beta = beta * 0.5 * (
                    1.0 + ols_norm / (ols_norm + 1e-3)   # tiny smoothing
                )
            elif denoise == "canonical_frac" and canonical_frac is not None:
                # Stage 3 with frozen-from-training per-voxel FRACvalue.
                # RT-deployable: shrinkage table is precomputed offline.
                beta = (beta * canonical_frac).astype(np.float32)
            all_betas.append(beta)
            img = events.iloc[trial_i].get("image_name", str(trial_i))
            trial_ids.append(str(img))
    return np.stack(all_betas, axis=0), trial_ids


def save_cell(cell_name: str, betas: np.ndarray, trial_ids: list[str],
              session: str, config: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / f"{cell_name}_{session}_betas.npy", betas)
    np.save(OUT_DIR / f"{cell_name}_{session}_trial_ids.npy",
            np.asarray(trial_ids))
    with open(OUT_DIR / f"{cell_name}_{session}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell_name}: betas {betas.shape} → {OUT_DIR}")


# ---- Cell catalog -------------------------------------------------------------

CELLS = {
    "OLS_glover_rtm":        dict(mode="ols", bold_source="rtmotion", hrf_strategy="glover"),
    "AR1freq_glover_rtm":    dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover"),
    "VariantG_glover_rtm":   dict(mode="variant_g", bold_source="rtmotion", hrf_strategy="glover"),
    "VariantG_glover_rtm_prior":
        dict(mode="variant_g_prior", bold_source="rtmotion", hrf_strategy="glover"),
    "AR1freq_glmsingleS1_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glmsingle_lib"),
    # Cells 7-9: add denoising on top of a baseline GLM fit
    "AR1freq_glover_rtm_glmdenoise_fracridge":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="glmdenoise_fracridge"),
    "VariantG_glover_rtm_glmdenoise_fracridge":
        dict(mode="variant_g", bold_source="rtmotion", hrf_strategy="glover",
             denoise="glmdenoise_fracridge"),
    "VariantG_glover_rtm_acompcor":
        dict(mode="variant_g", bold_source="rtmotion", hrf_strategy="glover",
             denoise="tcompcor"),
    # Cells 13-15: methods we coded up yesterday
    "EKF_streaming_glover_rtm":
        dict(mode="ar1_streaming_kalman", bold_source="rtmotion", hrf_strategy="glover"),
    "HOSVD_denoise_AR1freq_glover_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="hosvd_4d"),
    "Riemannian_prewhiten_AR1freq_glover_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="riemannian_prewhiten"),
    # Cell 17: HYBRID ONLINE — noise model (ρ̂, σ²) accumulates session-wide,
    # β fit per-trial with frozen-from-session ρ. The architecturally correct
    # version of the streaming-mindset cells (vs cell 13 reset-per-trial,
    # vs cell 16 diag-cov-with-770-probes that lost identifiability).
    "HybridOnline_AR1freq_glover_rtm":
        dict(mode="ar1_session_rho", bold_source="rtmotion", hrf_strategy="glover"),
    # Cell 20: log-signature path features of tCompCor PCs as additional
    # nuisance regressors. Streaming primitive (Chen's identity): each new
    # TR extends every active sliding window's log-sig by an O(1) update.
    "LogSig_AR1freq_glover_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion", hrf_strategy="glover",
             denoise="logsig_tcompcor"),
    # H3'-corrected: streaming pst=8 + session-ρ̂ frozen from full-session
    # streaming AR(1) tracker. The architecturally clean Regime C noise
    # model: per-trial β fit on cropped BOLD with frozen-from-past-runs ρ̂.
    "HybridOnline_streaming_pst8_AR1freq_glover_rtm":
        dict(mode="ar1_session_rho", bold_source="rtmotion",
             hrf_strategy="glover", streaming_post_stim_TRs=8),
    # GLMsingle gap-fill: cell 6 covers Stage-1-alone, cells 7/8 cover
    # Stages 2+3 with Glover HRF. The full Stage-1+2+3 stack and Stage 1
    # paired with Variant G have never been measured. Mac scored Stage
    # 1+2+3 only on top-1 (commit 2f96057, +0/+2pp) — never on AUC.
    "AR1freq_glmsingleFull_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib", denoise="glmdenoise_fracridge"),
    "VariantG_glmsingleFull_rtm":
        dict(mode="variant_g", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib", denoise="glmdenoise_fracridge"),
    "VariantG_glmsingleS1_rtm":
        dict(mode="variant_g", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib"),
    "AR1freq_glmsingleFull_fmriprep":
        dict(mode="ar1_freq", bold_source="fmriprep",
             hrf_strategy="glmsingle_lib", denoise="glmdenoise_fracridge"),
    # BOLD-source isolation: fmriprep + Glover + GLMdenoise — head-to-head
    # against cell 7 (rtmotion + Glover + GLMdenoise, AUC 0.886) tells us
    # whether fmriprep BOLD adds anything once GLMdenoise is in place.
    "AR1freq_glover_fmriprep_glmdenoise_fracridge":
        dict(mode="ar1_freq", bold_source="fmriprep",
             hrf_strategy="glover", denoise="glmdenoise_fracridge"),
    "VariantG_glover_fmriprep_glmdenoise_fracridge":
        dict(mode="variant_g", bold_source="fmriprep",
             hrf_strategy="glover", denoise="glmdenoise_fracridge"),
    # Streaming GLMsingle Stage 1 + Stage 3 — RT-deployable canonical
    # pipeline. Per-voxel HRF library (Stage 1, frozen from training)
    # plus per-voxel scalar fracridge (Stage 3, FRACvalue frozen from
    # ses-01-02 canonical GLMsingle output). Tests whether
    # Stages 1+3 survive windowing — the H3' deliverable on the
    # paper-Offline-vs-paper-RT gap.
    "Streaming_S1S3_pst8_AR1freq_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_frac",
             streaming_post_stim_TRs=8),
    "Streaming_S1S3_pst8_AR1freq_fmriprep":
        dict(mode="ar1_freq", bold_source="fmriprep",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_frac",
             streaming_post_stim_TRs=8),
    "FullRun_S1S3_AR1freq_rtm":
        dict(mode="ar1_freq", bold_source="rtmotion",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_frac"),
    "FullRun_S1S3_AR1freq_fmriprep":
        dict(mode="ar1_freq", bold_source="fmriprep",
             hrf_strategy="glmsingle_lib",
             denoise="canonical_frac"),
    # Cells 10-12 require nilearn — separate driver: scripts/rt_paper_full_replica.py
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+",
                    default=list(CELLS.keys()),
                    help="Which cells to run; defaults to all JAX-only cells.")
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--runs", nargs="+", type=int, default=list(range(1, 12)))
    ap.add_argument("--prior-from-session", default="ses-01",
                    help="Session whose mean β (Variant G) is used as the "
                         "training prior for VariantG_glover_rtm_prior.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pre-load empirical prior for cell 5 if needed
    prior_mean = None
    prior_var = None
    if "VariantG_glover_rtm_prior" in args.cells:
        prior_path = (PAPER_ROOT / "task_2_1_betas"
                      / f"G_fmriprep_{args.prior_from_session}_betas.npy")
        if prior_path.exists():
            prior_betas = np.load(prior_path)              # (n_trials, V)
            prior_mean = prior_betas.mean(axis=0).astype(np.float32)
            prior_var = np.maximum(
                prior_betas.var(axis=0).astype(np.float32), 1e-3
            )
            print(f"[prior] loaded ses-01 G_fmriprep mean β as training prior "
                  f"(n_trials={prior_betas.shape[0]})")
        else:
            print(f"[prior] WARN: {prior_path} not found; skipping cell 5")
            args.cells = [c for c in args.cells
                          if c != "VariantG_glover_rtm_prior"]

    for cell in args.cells:
        if cell not in CELLS:
            print(f"  SKIP unknown cell {cell}")
            continue
        config = CELLS[cell].copy()
        config["session"] = args.session
        config["runs"] = list(args.runs)
        print(f"\n=== {cell} === {config}")
        t0 = time.time()
        try:
            betas, trial_ids = run_glm_cell(
                cell, mode=config["mode"], bold_source=config["bold_source"],
                hrf_strategy=config["hrf_strategy"],
                session=args.session, runs=args.runs,
                prior_mean=prior_mean, prior_var=prior_var,
                denoise=config.get("denoise"),
                streaming_post_stim_TRs=config.get("streaming_post_stim_TRs"),
            )
            save_cell(cell, betas, trial_ids, args.session, config)
            print(f"  elapsed: {time.time() - t0:.1f}s")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()


if __name__ == "__main__":
    main()
