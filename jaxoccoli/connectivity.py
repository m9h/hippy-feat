import jax
import jax.numpy as jnp
from functools import partial

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
