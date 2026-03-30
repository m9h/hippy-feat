import jax
import jax.numpy as jnp
from functools import partial

from .covariance import corr, cov


class SlidingWindowConnectivity:
    """
    Real-time sliding window connectivity (Correlation Matrix).
    Optimized for JAX.
    """
    def __init__(self, n_rois, window_size):
        self.n_rois = n_rois
        self.window_size = window_size
        
    @partial(jax.jit, static_argnums=(0,))
    def compute_correlation(self, data_window):
        """
        Compute Pearson correlation matrix for the given window.
        data_window: (N_ROIs, Window_Size)
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
        """
        Updates the buffer (shift left) and calculates connectivity.
        current_buffer: (N_ROIs, Window_Size)
        new_data: (N_ROIs,) or (N_ROIs, 1)
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
