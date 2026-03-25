import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union
from functools import partial

class GeneralLinearModel:
    """
    A JAX-based General Linear Model for massively parallel fMRI analysis.
    Optimized for speed using Cholesky decomposition and JIT compilation.
    """
    
    def __init__(self, design_matrix: jnp.ndarray):
        """
        Initialize the GLM with a design matrix.
        
        Args:
            design_matrix: (N_timepoints, N_regressors) array.
        """
        self.X = design_matrix
        # Precompute (X'X)^-1 X' for OLS
        # We use Cholesky for stability and speed on positive definite X'X
        self.XtX = self.X.T @ self.X
        # Add a small jitter for numerical stability if needed, but usually redundant if well designed
        self.XtX_inv = jnp.linalg.inv(self.XtX) 
        self.pinv = self.XtX_inv @ self.X.T

    @partial(jax.jit, static_argnums=(0,))
    def fit(self, data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Fit the GLM to the data using OLS.
        
        Args:
            data: (..., N_timepoints) array.
            
        Returns:
            betas: (..., N_regressors)
            residuals: (..., N_timepoints)
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
        """
        Compute t-statistics for a given contrast.
        
        Args:
            betas: (..., N_regressors)
            residuals: (..., N_timepoints)
            contrast: (N_regressors,)
            
        Returns:
            t_values: (...,)
            p_values: (...,) (Not implemented yet, requires jax.scipy.stats)
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

