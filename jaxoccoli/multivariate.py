"""Multivariate analysis tools for neuroimaging.

Provides JIT-compiled implementations of:
- Principal Component Analysis (PCA) with variance-based truncation.
- Canonical Correlation Analysis (CCA) between two multivariate blocks.
- Multivariate Conditional Mutual Information (mvCMI) core logic.

These tools form the foundation for individual-level brain mapping
and high-dimensional connectivity analysis.

References:
    Sundaram et al. (2020) "Individual Resting-State Brain Networks Enabled
    by Massive Multivariate Conditional Mutual Information" -- IEEE TMI.
"""

import jax
import jax.numpy as jnp
from typing import Tuple, Optional


def pca(data, n_components: Optional[float] = 0.95, max_k: int = 30) -> Tuple[jnp.ndarray, int]:
    """Principal Component Analysis with variance-based truncation.

    Retains the top K components accounting for at least *n_components*
    of the spatial variance.

    Args:
        data: (T, V) time series for a single parcel (T timepoints, V voxels).
        n_components: Float (0-1) for variance threshold, or Int for fixed K.
            Defaults to 0.95 (95% variance).
        max_k: Maximum number of components to return (for static shapes).

    Returns:
        projected: (T, max_k) projected time series, zero-padded if K < max_k.
        k: The actual number of components retained.
    """
    T, V = data.shape
    # Center the data
    mean = jnp.mean(data, axis=0)
    centered = data - mean

    # Use the smaller of (T, V) for the covariance trick
    if V > T:
        # Dual covariance: (T, T)
        C = (centered @ centered.T) / (V - 1)
        evals, evecs_dual = jnp.linalg.eigh(C)
        # Recover primal eigenvectors: evecs = centered.T @ evecs_dual
        # (V, T) = (V, T) @ (T, T)
        evecs = centered.T @ evecs_dual
        # Normalize primal eigenvectors
        evecs = evecs / jnp.linalg.norm(evecs, axis=0)
    else:
        # Standard covariance: (V, V)
        C = (centered.T @ centered) / (T - 1)
        evals, evecs = jnp.linalg.eigh(C)

    # Sort descending
    idx = jnp.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

    # Find K
    if isinstance(n_components, float):
        cum_var = jnp.cumsum(jnp.maximum(evals, 0.0))
        cum_var /= jnp.sum(evals)
        k = jnp.argmax(cum_var >= n_components) + 1
    else:
        k = int(n_components)

    # Limit to max_k and ensure k > 0
    k = jnp.clip(k, 1, max_k)

    # Project
    projected_full = centered @ evecs[:, :max_k]  # (T, max_k)
    
    # Mask components beyond k (to maintain static shapes for JIT)
    mask = jnp.arange(max_k) < k
    projected = projected_full * mask[None, :]
    
    return projected, k


def cca(x, y) -> jnp.ndarray:
    """Canonical Correlation Analysis between two multivariate blocks.

    Computes the canonical correlation coefficients (singular values)
    between two sets of variables.

    Args:
        x: (T, dx) first multivariate time series.
        y: (T, dy) second multivariate time series.

    Returns:
        rho: (min(dx, dy),) canonical correlation coefficients.
    """
    # 1. Z-score normalization
    x = (x - jnp.mean(x, axis=0)) / jnp.std(x, axis=0)
    y = (y - jnp.mean(y, axis=0)) / jnp.std(y, axis=0)
    
    # 2. QR decomposition for whitening
    qx, rx = jnp.linalg.qr(x)
    qy, ry = jnp.linalg.qr(y)
    
    # 3. SVD of the cross-product of orthogonal bases
    # (dx, dy)
    m = qx.T @ qy
    rho = jnp.linalg.svd(m, compute_uv=False)
    
    return jnp.clip(rho, 0.0, 1.0)


def mv_cmi(x, y, z, reg: float = 1e-4) -> jnp.ndarray:
    """Multivariate Conditional Mutual Information.

    I(X; Y | Z) = I(X; Y, Z) - I(X; Z)
    
    Calculated via the log-determinant of the partial correlation matrix.

    Args:
        x: (T, dx) first multivariate block.
        y: (T, dy) second multivariate block.
        z: (T, dz) conditioning block (rest of brain).
        reg: L2 regularization for matrix inversion.

    Returns:
        Scalar mutual information (in nats).
    """
    # This is a direct implementation. For "massive" all-pairs CMI, 
    # use the precision matrix trick in jaxoccoli.connectivity.
    
    # Combine [x, y, z] and compute joint covariance
    joint = jnp.concatenate([x, y, z], axis=1)
    cov = jnp.cov(joint, rowvar=False)
    
    # Precision matrix (inverse covariance)
    prec = jnp.linalg.inv(cov + reg * jnp.eye(cov.shape[0]))
    
    dx, dy, dz = x.shape[1], y.shape[1], z.shape[1]
    
    # Extract blocks of the precision matrix
    # Q = [ Qxx Qxy Qxz ]
    #     [ Qyx Qyy Qyz ]
    #     [ Qzx Qzy Qzz ]
    
    qxx = prec[:dx, :dx]
    qyy = prec[dx:dx+dy, dx:dx+dy]
    qxy = prec[:dx, dx:dx+dy]
    qyx = prec[dx:dx+dy, :dx]
    
    # The joint block for (X, Y)
    q_joint = prec[:dx+dy, :dx+dy]
    
    # I(X; Y | Z) = 0.5 * [ log|Qxx| + log|Qyy| - log|Q_joint| ]
    _, logdet_x = jnp.linalg.slogdet(qxx)
    _, logdet_y = jnp.linalg.slogdet(qyy)
    _, logdet_xy = jnp.linalg.slogdet(q_joint)
    
    return 0.5 * (logdet_x + logdet_y - logdet_xy)
