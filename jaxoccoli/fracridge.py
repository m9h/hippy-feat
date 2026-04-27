"""Fractional ridge regression.

Reference:
  Rokem A, Kay KN (2020). Fractional ridge regression: a fast, interpretable
  reparameterization of ridge regression. GigaScience 9(12):giaa133.

Core idea: standard ridge regression
    β_λ = (X'X + λI)^{-1} X' y
parameterizes shrinkage by λ, which is hard to pick consistently across
voxels and across X with different SNRs. fracridge re-parameterizes by
the FRACTION of the OLS solution length retained:
    f = ||β_λ|| / ||β_OLS||  ∈ (0, 1]
f=1 ⇒ β = β_OLS (no shrinkage); f→0 ⇒ β → 0 (full shrinkage). The mapping
is monotone and can be precomputed via an SVD of X.

Implementation: SVD-based — compute the singular values of X once, then
for each desired f, solve a 1-D root-finding problem to get λ. This is
the algorithm in `fracridge` (the official Python package).

Stage 3 of GLMsingle. Combined with GLMdenoise (stage 2) and per-voxel
HRF (stage 1), this is what gives GLMsingle its measured advantage —
NOT the HRF library swap, per Task 2.1's bake-off finding.

Real-time use: training-time CV to choose f per voxel is offline. The
chosen-f ridge solution is a one-shot linear projection — fast at inference.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq


def _ols_norm(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute ||β_OLS|| and the SVD of X for fast subsequent ridge solves.

    Returns:
        beta_ols_norm: scalar — the L2 norm of the OLS solution.
        UtY: (P,) — U^T y from SVD; cached for ridge-path computations.
        S: (P,) — singular values of X.
    """
    U, S, Vh = np.linalg.svd(X, full_matrices=False)
    UtY = U.T @ y
    beta_ols = Vh.T @ (UtY / (S + 1e-12))
    return float(np.linalg.norm(beta_ols)), UtY, S, Vh


def fracridge_solve(X: np.ndarray, y: np.ndarray, fracs: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Solve fracridge for one regression problem at multiple fractions.

    Args:
        X: (T, P) design matrix.
        y: (T,) target.
        fracs: (n_frac,) array of fraction values in (0, 1].

    Returns:
        betas: (n_frac, P) — β at each fraction.
        lambdas: (n_frac,) — corresponding ridge λ values (for reference).

    Notes:
        - fracs=1 gives β_OLS exactly (λ=0).
        - The mapping f → λ is monotone decreasing — bigger λ means smaller f.
    """
    T, P = X.shape
    fracs = np.asarray(fracs, dtype=np.float64)
    if (fracs <= 0).any() or (fracs > 1).any():
        raise ValueError("fracs must be in (0, 1]")

    beta_ols_norm, UtY, S, Vh = _ols_norm(X, y)
    if beta_ols_norm < 1e-12:
        # Degenerate case: all-zero target → return zeros at every frac
        return (np.zeros((len(fracs), P), dtype=np.float32),
                np.zeros(len(fracs), dtype=np.float32))

    def beta_norm_at_lambda(lam: float) -> float:
        coef = UtY * S / (S ** 2 + lam)                   # (P,)
        return float(np.linalg.norm(coef))

    lambdas = np.zeros_like(fracs)
    betas = np.zeros((len(fracs), P), dtype=np.float32)
    for i, f in enumerate(fracs):
        target_norm = f * beta_ols_norm
        if abs(f - 1.0) < 1e-9:
            lam = 0.0
        else:
            # Find λ such that ||β_λ|| = target_norm via Brent's method
            try:
                lam = brentq(
                    lambda l: beta_norm_at_lambda(l) - target_norm,
                    a=0.0, b=1e8 + (S ** 2).max(),
                    xtol=1e-6,
                )
            except ValueError:
                # Fall back: very large λ → β ≈ 0
                lam = 1e8
        lambdas[i] = lam
        # Compute β at this λ:  β = V (S / (S² + λ)) U^T y
        coef = (S * UtY) / (S ** 2 + lam)
        betas[i] = (Vh.T @ coef).astype(np.float32)
    return betas, lambdas.astype(np.float32)


def fracridge_cv(Y_train: list[np.ndarray], X_train: list[np.ndarray],
                  Y_test: list[np.ndarray], X_test: list[np.ndarray],
                  fracs: np.ndarray | None = None,
                  metric: str = "r2"
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel cross-validated fraction selection.

    For each voxel, choose the fraction that maximizes a CV metric (default
    held-out R²). One fraction per voxel.

    Args:
        Y_train, X_train: per-fold training data; lists of length n_folds.
        Y_test, X_test:   per-fold held-out data; lists of length n_folds.
        fracs: candidate fractions; default `np.linspace(0.05, 1.0, 20)`.
        metric: 'r2' (default) or 'mse'.

    Returns:
        f_chosen: (V,) — best fraction per voxel.
        cv_score: (V,) — CV metric at chosen fraction.
    """
    if fracs is None:
        fracs = np.linspace(0.05, 1.0, 20).astype(np.float64)
    n_folds = len(Y_train)
    if n_folds < 1:
        raise ValueError("fracridge_cv needs ≥ 1 (train, test) split")
    if len(X_train) != n_folds or len(Y_test) != n_folds or len(X_test) != n_folds:
        raise ValueError("Y_train/X_train/Y_test/X_test must be same length")
    V = Y_train[0].shape[0]

    # Per-fold per-voxel score grid: (V, n_fracs, n_folds)
    score_grid = np.zeros((V, len(fracs), n_folds), dtype=np.float32)
    for k in range(n_folds):
        Xtr, Ytr = X_train[k], Y_train[k]
        Xte, Yte = X_test[k], Y_test[k]
        # For each voxel, solve at all fracs in one call
        for v in range(V):
            betas, _ = fracridge_solve(Xtr, Ytr[v], fracs)         # (F, P)
            preds = betas @ Xte.T                                  # (F, T_te)
            if metric == "r2":
                resid = Yte[v][None, :] - preds                    # (F, T_te)
                rss = (resid ** 2).sum(axis=1)
                tss = ((Yte[v] - Yte[v].mean()) ** 2).sum() + 1e-12
                score_grid[v, :, k] = 1.0 - rss / tss
            elif metric == "mse":
                resid = Yte[v][None, :] - preds
                score_grid[v, :, k] = -(resid ** 2).mean(axis=1)
            else:
                raise ValueError(metric)

    # Average over folds, pick best fraction per voxel
    score_mean = score_grid.mean(axis=2)                            # (V, F)
    best_idx = score_mean.argmax(axis=1)
    f_chosen = fracs[best_idx].astype(np.float32)
    cv_score = score_mean[np.arange(V), best_idx].astype(np.float32)
    return f_chosen, cv_score


__all__ = ["fracridge_solve", "fracridge_cv"]
