"""Sliding-window and dynamic functional connectivity estimation.

Provides both a stateful class (``SlidingWindowConnectivity``) for real-time
streaming use and pure-functional helpers for offline analysis.  All
correlation/covariance estimation delegates to :mod:`jaxoccoli.covariance`
so that variance-aware extensions (weighted, disattenuated, posterior)
compose naturally.

Key functions:
    - ``sliding_window_corr`` -- pure-functional sliding-window Pearson
      correlation over a (C, T) time series.
    - ``dynamic_connectivity`` -- generalised version accepting any
      estimator ('corr' or 'cov') with extra keyword arguments.
    - ``sample_nonoverlapping_windows`` / ``sample_overlapping_windows``
      -- stochastic window sampling for bootstrap or null-model analyses.

The ``SlidingWindowConnectivity`` class wraps the same logic in a
stateful API suitable for real-time neurofeedback loops where a new
TR arrives each iteration and the buffer is shifted in-place.

References:
    Allen et al. (2014) "Tracking whole-brain connectivity dynamics in
    the resting state" -- sliding-window dFC methodology.
"""

import jax
import jax.numpy as jnp
from functools import partial
from typing import NamedTuple, List

from .covariance import corr, cov


class MVCMIParams(NamedTuple):
    """Parameters and bookkeeping for Massive mvCMI.

    Args:
        projections: (T, total_components) concatenated PCA projections.
            Assumes each parcel is padded to *max_k*.
        n_parcels: Total number of parcels.
        max_k: Fixed number of components per parcel (for static shapes).
    """
    projections: jnp.ndarray
    n_parcels: int
    max_k: int


class SlidingWindowConnectivity:
    """Stateful sliding-window Pearson correlation for real-time streaming.

    Maintains a fixed-length buffer of shape ``(n_rois, window_size)``
    and exposes JIT-compiled methods to compute the correlation matrix
    for the current window or to shift the buffer by one TR and
    recompute in a single call.

    For offline (whole-session) analysis prefer the pure-functional
    :func:`sliding_window_corr` or :func:`dynamic_connectivity`.

    Args:
        n_rois: Number of ROIs / channels (C).
        window_size: Number of timepoints in the sliding window.
    """

    def __init__(self, n_rois, window_size):
        self.n_rois = n_rois
        self.window_size = window_size
        
    @partial(jax.jit, static_argnums=(0,))
    def compute_correlation(self, data_window):
        """Compute the Pearson correlation matrix for a data window.

        Z-scores each ROI along the time axis and computes the
        normalised cross-product matrix.

        Args:
            data_window: (n_rois, window_size) time series block.

        Returns:
            (n_rois, n_rois) Pearson correlation matrix.
        """
        # 1. Z-score normalization along time (axis 1)
        mean = jnp.mean(data_window, axis=1, keepdims=True)
        std = jnp.std(data_window, axis=1, keepdims=True)
        
        # Avoid division by zero
        std = jnp.where(std == 0, 1.0, std)
        
        normalized = (data_window - mean) / std
        
        # 2. Correlation = (X @ X.T) / (N - 1)
        # normalized is (N_ROIs, Window_Size)
        corr_matrix = (normalized @ normalized.T) / (self.window_size - 1)
        
        return corr_matrix

    @partial(jax.jit, static_argnums=(0,))
    def update_and_compute(self, current_buffer, new_data):
        """Shift the buffer by one TR and recompute the correlation matrix.

        Drops the oldest timepoint, appends *new_data*, and returns both
        the updated buffer and the new correlation matrix.

        Args:
            current_buffer: (n_rois, window_size) current sliding window.
            new_data: (n_rois,) or (n_rois, 1) new TR observation.

        Returns:
            Tuple of (updated_buffer, corr_matrix) where
            *updated_buffer* has shape (n_rois, window_size) and
            *corr_matrix* has shape (n_rois, n_rois).
        """
        # Shift
        # Concatenate: buffer[:, 1:] + new_data
        # Ensure new_data is (N_ROIs, 1)
        if new_data.ndim == 1:
            new_data = new_data[:, None]
            
        updated_buffer = jnp.concatenate([current_buffer[:, 1:], new_data], axis=1)
        
        # Compute
        corr = self.compute_correlation(updated_buffer)
        
        return updated_buffer, corr


# ---------------------------------------------------------------------------
# Functional (stateless) connectivity operations
# ---------------------------------------------------------------------------

def sliding_window_corr(data, window_size):
    """Pure functional sliding window correlation.

    Args:
        data: (C, T) time series.  C = channels/ROIs, T = timepoints.
        window_size: Number of timepoints per window.

    Returns:
        (n_windows, C, C) correlation matrices per window.
    """
    C, T = data.shape
    n_windows = T - window_size + 1

    def _compute_window(i):
        window = jax.lax.dynamic_slice(data, (0, i), (C, window_size))
        return corr(window)

    return jax.vmap(_compute_window)(jnp.arange(n_windows))


def sample_nonoverlapping_windows(data, window_size, num_windows, *, key):
    """Randomly sample non-overlapping temporal windows.

    Args:
        data: (C, T) time series.
        window_size: Samples per window.
        num_windows: Number of windows to sample.
        key: JAX PRNG key.

    Returns:
        (num_windows, C, window_size) sampled windows.
    """
    C, T = data.shape
    max_windows = T // window_size
    num_windows = min(num_windows, max_windows)

    # Random permutation of possible start indices
    starts = jnp.arange(max_windows) * window_size
    perm = jax.random.permutation(key, starts)[:num_windows]
    perm = jnp.sort(perm)

    def _extract(start):
        return jax.lax.dynamic_slice(data, (0, start), (C, window_size))

    return jax.vmap(_extract)(perm)


def sample_overlapping_windows(data, window_size, num_windows, *, key):
    """Randomly sample overlapping temporal windows.

    Args:
        data: (C, T) time series.
        window_size: Samples per window.
        num_windows: Number of windows to sample.
        key: JAX PRNG key.

    Returns:
        (num_windows, C, window_size) sampled windows.
    """
    C, T = data.shape
    max_start = T - window_size
    starts = jax.random.randint(key, (num_windows,), 0, max_start + 1)
    starts = jnp.sort(starts)

    def _extract(start):
        return jax.lax.dynamic_slice(data, (0, start), (C, window_size))

    return jax.vmap(_extract)(starts)


def dynamic_connectivity(data, window_size, estimator='corr', **kw):
    """Dynamic functional connectivity using arbitrary estimator.

    Args:
        data: (C, T) time series.
        window_size: Samples per window.
        estimator: 'corr' (Pearson) or 'cov' (covariance).
        **kw: Extra arguments passed to the estimator.

    Returns:
        (n_windows, C, C) connectivity matrices per window.
    """
    C, T = data.shape
    n_windows = T - window_size + 1
    est_fn = corr if estimator == 'corr' else cov

    def _compute_window(i):
        window = jax.lax.dynamic_slice(data, (0, i), (C, window_size))
        return est_fn(window, **kw)

    return jax.vmap(_compute_window)(jnp.arange(n_windows))


def massive_mv_cmi(params: MVCMIParams, reg: float = 1e-4) -> jnp.ndarray:
    """Computes all-pairs Multivariate Conditional Mutual Information.

    Uses the Precision Matrix block-determinant formula:
    I(X; Y | Rest) = 0.5 * log(|Q_ii| * |Q_jj| / |Q_ij_block|)
    where Q is the global precision matrix of all parcel projections.

    This implementation assumes all parcels are padded to *max_k* components
    to ensure static shapes for JIT/vmap.

    Args:
        params: MVCMIParams containing (T, n_parcels * max_k) projections.
        reg: L2 regularization for the global matrix inversion.

    Returns:
        (n_parcels, n_parcels) CMI matrix in nats.
    """
    T, M = params.projections.shape
    Q = jnp.linalg.inv(jnp.cov(params.projections, rowvar=False) + reg * jnp.eye(M))
    max_k = params.max_k
    
    def pair_cmi(i, j):
        si = i * max_k
        sj = j * max_k
        
        # Block determinants via dynamic_slice
        # (max_k, max_k)
        q_i = jax.lax.dynamic_slice(Q, (si, si), (max_k, max_k))
        q_j = jax.lax.dynamic_slice(Q, (sj, sj), (max_k, max_k))
        
        # Combined block Q_ij Schur term: I - inv(Q_j) @ Q_ji @ inv(Q_i) @ Q_ij
        q_ij_cross = jax.lax.dynamic_slice(Q, (si, sj), (max_k, max_k))
        q_ji_cross = jax.lax.dynamic_slice(Q, (sj, si), (max_k, max_k))
        
        # Add regularization to sub-blocks before inversion
        sub_reg = reg * jnp.eye(max_k)
        term = q_ji_cross @ jnp.linalg.inv(q_i + sub_reg) @ q_ij_cross
        core = jnp.eye(max_k) - jnp.linalg.inv(q_j + sub_reg) @ term
        
        _, logdet = jnp.linalg.slogdet(core)
        
        # Diagonal (i==j) results in logdet of 0 (identity - identity = 0, but 
        # slogdet might be unstable). Return 0.0 explicitly for diagonal.
        is_diag = (i == j)
        return jnp.where(is_diag, 0.0, -0.5 * logdet)

    # Double vmap: (n_parcels, n_parcels)
    indices = jnp.arange(params.n_parcels)
    cmi_matrix = jax.vmap(lambda i: jax.vmap(lambda j: pair_cmi(i, j))(indices))(indices)
    
    return cmi_matrix
