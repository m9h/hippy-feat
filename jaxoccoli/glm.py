"""General Linear Model for massively parallel voxelwise fMRI analysis.

Provides a class-based OLS GLM (``GeneralLinearModel``) that precomputes
the pseudo-inverse ``(X'X)^{-1} X'`` once and then fits all voxels in a
single ``jnp.tensordot`` call, making it efficient for whole-brain
analysis on GPU.

For Bayesian extensions with variance propagation (conjugate, AR(1),
NUTS), see :mod:`jaxoccoli.bayesian_beta`.  The point-estimate betas
from this module feed into :mod:`jaxoccoli.stats` for frequentist
inference and into :mod:`jaxoccoli.permutation` for non-parametric
testing.

Key class:
    ``GeneralLinearModel`` -- Cholesky-accelerated OLS with contrast-level
    t-statistic computation via ``compute_stats``.

Mathematical model:
    y = X beta + epsilon,  epsilon ~ N(0, sigma^2 I)
    beta_hat = (X'X)^{-1} X' y  (OLS)
    t = c' beta_hat / sqrt(sigma_hat^2 * c' (X'X)^{-1} c)
"""

import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union
from functools import partial

class GeneralLinearModel:
    """Cholesky-accelerated OLS General Linear Model for whole-brain analysis.

    Precomputes the pseudo-inverse ``(X'X)^{-1} X'`` at construction time
    so that fitting all voxels reduces to a single ``jnp.tensordot`` call.
    Contrast-level t-statistics are computed via ``compute_stats``.

    For Bayesian single-trial beta estimation with variance propagation,
    see :mod:`jaxoccoli.bayesian_beta`.

    Args:
        design_matrix: (T, P) design matrix where T is the number of
            timepoints and P is the number of regressors.
    """

    def __init__(self, design_matrix: jnp.ndarray):
        """Initialise the GLM and precompute ``(X'X)^{-1} X'``.

        Args:
            design_matrix: (T, P) design matrix.
        """
        self.X = design_matrix
        # Precompute (X'X)^-1 X' for OLS
        # We use Cholesky for stability and speed on positive definite X'X
        self.XtX = self.X.T @ self.X
        # Use Cholesky-based inversion for stability and speed
        from .matrix import cholesky_invert
        self.XtX_inv = cholesky_invert(self.XtX + 1e-6 * jnp.eye(self.XtX.shape[0])) 
        self.pinv = self.XtX_inv @ self.X.T

    @partial(jax.jit, static_argnums=(0,))
    def fit(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Fit the OLS GLM to *data* in a single tensordot call.

        Args:
            data: (..., T) array where the last axis is time.
                Typical shapes: (V, T) for V voxels or (X, Y, Z, T).

        Returns:
            Tuple of (betas, residuals) where *betas* has shape
            (..., P) and *residuals* has shape (..., T).
        """
        # Ensure data is last dimension aligned with time
        # Betas = (X'X)^-1 X' Y = pinv @ Y.T
        # We assume data is (Voxels, Time) or (X, Y, Z, Time)
        # We want to contract the last dimension of data with the last dimension of pinv's columns
        
        # pinv shape: (Regressors, Time)
        # data shape: (..., Time)
        # result: (..., Regressors)
        
        betas = jnp.tensordot(data, self.pinv, axes=(-1, 1))
        
        # Calculate fitted values: Y_hat = X @ betas.T
        # But betas is (..., Regressors) and X is (Time, Regressors)
        # We want (..., Time)
        
        predicted = jnp.tensordot(betas, self.X, axes=(-1, 1))
        residuals = data - predicted
        
        return betas, residuals
    
    @partial(jax.jit, static_argnums=(0,))
    def compute_stats(self, betas: jnp.ndarray, residuals: jnp.ndarray, contrast: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute voxelwise t-statistics for a univariate contrast.

        t = c' beta / sqrt(sigma_hat^2 * c' (X'X)^{-1} c)

        Args:
            betas: (..., P) regression coefficients from :meth:`fit`.
            residuals: (..., T) model residuals from :meth:`fit`.
            contrast: (P,) contrast weight vector.

        Returns:
            (...,) t-statistic map.
        """
        # RSS: Sum of squared residuals
        rss = jnp.sum(residuals**2, axis=-1)
        
        # Degrees of freedom: Timepoints - Regressors
        df = self.X.shape[0] - self.X.shape[1]
        
        # Variance estimate: sigma_hat^2 = RSS / df
        sigma2 = rss / df
        
        # Contrast variance: c' (X'X)^-1 c
        c_var = contrast @ self.XtX_inv @ contrast.T
        
        # Standard error of the contrast = sqrt(sigma2 * c_var)
        se = jnp.sqrt(sigma2 * c_var)
        
        # Effect size = c' beta
        effect = jnp.tensordot(betas, contrast, axes=(-1, 0))
        
        # T-statistic = Effect / SE
        t_stat = effect / se
        
        return t_stat

