import jax
import jax.numpy as jnp
from jax.scipy import stats

def compute_t_stat(betas, residuals, XtX_inv, contrast, df):
    """
    Functional t-statistic computation.
    """
    rss = jnp.sum(residuals**2, axis=-1)
    sigma2 = rss / df
    c_var = contrast @ XtX_inv @ contrast.T
    se = jnp.sqrt(sigma2 * c_var)
    effect = jnp.tensordot(betas, contrast, axes=(-1, 0))
    t_stat = effect / se
    return t_stat

def t_to_p(t_stat, df):
    """
    Two-tailed p-value from t-statistic.
    """
    return 2 * (1 - stats.t.cdf(jnp.abs(t_stat), df))

def compute_f_stat(betas, residuals, XtX_inv, contrast_matrix, df, N_timepoints):
    """
    F-statistic for matrix contrasts.
    contrast_matrix: (Rank, Regressors)
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
