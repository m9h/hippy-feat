"""Frequentist voxelwise test statistics for the GLM.

Provides pure-functional implementations of voxelwise t- and
F-statistics that operate on pre-fitted GLM outputs (betas, residuals,
``(X'X)^{-1}``).  These are the building blocks consumed by
:mod:`jaxoccoli.permutation` for non-parametric inference and by
:mod:`jaxoccoli.glm` ``GeneralLinearModel.compute_stats``.

Key functions:
    - ``compute_t_stat`` -- univariate contrast t-statistic.
    - ``compute_f_stat`` -- multivariate (matrix) contrast F-statistic
      via the extra-sum-of-squares principle.
    - ``t_to_p`` -- two-tailed p-value from a t-statistic using
      ``jax.scipy.stats.t.cdf``.

Mathematical detail:
    t = c' beta_hat / sqrt(sigma_hat^2 * c' (X'X)^{-1} c)
    F = (C beta_hat)' [C (X'X)^{-1} C']^{-1} (C beta_hat) / rank(C)
        / (RSS / df)

where ``df = T - P`` (timepoints minus regressors).

For Bayesian alternatives with variance propagation, see
:mod:`jaxoccoli.bayesian_beta`.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats

def compute_t_stat(betas, residuals, XtX_inv, contrast, df):
    """Voxelwise t-statistic for a univariate contrast.

    Computes t = c' beta / sqrt(sigma_hat^2 * c' (X'X)^{-1} c) for
    every voxel in parallel via broadcasting.

    Args:
        betas: (..., P) estimated regression coefficients.
        residuals: (..., T) model residuals.
        XtX_inv: (P, P) precomputed (X'X)^{-1}.
        contrast: (P,) contrast weight vector.
        df: Scalar residual degrees of freedom (T - P).

    Returns:
        (...,) t-statistic map (same leading shape as *betas*).
    """
    rss = jnp.sum(residuals**2, axis=-1)
    sigma2 = rss / df
    c_var = contrast @ XtX_inv @ contrast.T
    se = jnp.sqrt(sigma2 * c_var)
    effect = jnp.tensordot(betas, contrast, axes=(-1, 0))
    t_stat = effect / se
    return t_stat

def t_to_p(t_stat, df):
    """Two-tailed p-value from a t-statistic via the Student-t CDF.

    Args:
        t_stat: (...,) t-statistic values.
        df: Scalar residual degrees of freedom.

    Returns:
        (...,) two-tailed p-values.
    """
    return 2 * (1 - stats.t.cdf(jnp.abs(t_stat), df))

def compute_f_stat(betas, residuals, XtX_inv, contrast_matrix, df, N_timepoints):
    """Voxelwise F-statistic for a matrix (multi-row) contrast.

    Uses the extra-sum-of-squares principle:
        F = (C beta)' [C (X'X)^{-1} C']^{-1} (C beta) / rank(C)
            / (RSS / df)

    The matrix ``[C (X'X)^{-1} C']^{-1}`` is constant across voxels
    and inverted once.

    Args:
        betas: (..., P) estimated regression coefficients.
        residuals: (..., T) model residuals.
        XtX_inv: (P, P) precomputed (X'X)^{-1}.
        contrast_matrix: (R, P) contrast matrix with R <= P rows.
        df: Scalar residual degrees of freedom (T - P).
        N_timepoints: Number of timepoints T (used for reference).

    Returns:
        (...,) F-statistic map.
    """
    # RSS_full
    rss = jnp.sum(residuals**2, axis=-1)
    
    # Calculate ESS (Explained Sum of Squares) for the contrast
    # ESS = (C beta)' (C (X'X)^-1 C')^-1 (C beta)
    
    # This involves matrix inversion of (Rank, Rank) per voxel if betas vary?
    # No, C (X'X)^-1 C' is constant across voxels! It only depends on Design.
    
    C = contrast_matrix
    inv_CXtXinvCt = jnp.linalg.inv(C @ XtX_inv @ C.T)
    
    # C_beta: (..., Rank)
    # betas: (..., Regressors)
    # C: (Rank, Regressors)
    C_beta = jnp.tensordot(betas, C, axes=(-1, 1)) # -> (..., Rank)
    
    # ESS calculation
    # We need bilinear form xM x' per voxel
    # M is (Rank, Rank). x is (..., Rank)
    
    # Expanded: sum(x_i * M_ij * x_j)
    # tensordot C_beta with inv_... 
    temp = jnp.tensordot(C_beta, inv_CXtXinvCt, axes=(-1, 1)) # (..., Rank)
    ess = jnp.sum(temp * C_beta, axis=-1) # (...,)
    
    rank = C.shape[0]
    
    # F = (ESS / rank) / (RSS / df)
    f_stat = (ess / rank) / (rss / df)
    
    return f_stat
