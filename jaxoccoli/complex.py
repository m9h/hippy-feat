"""Complex-valued fMRI utilities — primitives for NORDIC, phase regression,
and complex-domain GLMs.

Most fMRI pipelines discard phase and operate on magnitude only. This module
preserves phase information so downstream methods (NORDIC denoising, phase
regression for physiological cleanup, Rowe-style complex GLM) can use it.

Convention: complex data is JAX/NumPy complex64. Convert at boundaries:
  - From magnitude + phase: `from_mag_phase(mag, phase)`
  - From paired-real arrays: `from_real_imag(re, im)`
  - To magnitude + phase: `to_mag_phase(z)`

Real-time use: all functions are JIT-compatible and shape-stable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def from_mag_phase(magnitude: jnp.ndarray, phase: jnp.ndarray) -> jnp.ndarray:
    """Build complex array from magnitude and phase.

    Args:
        magnitude: real-valued, shape (..., T) or (..., X, Y, Z, T).
        phase: same shape, in radians.

    Returns:
        complex64 array.
    """
    return (magnitude * jnp.exp(1j * phase)).astype(jnp.complex64)


def from_real_imag(re: jnp.ndarray, im: jnp.ndarray) -> jnp.ndarray:
    """Build complex array from paired real / imaginary arrays."""
    return (re + 1j * im).astype(jnp.complex64)


def to_mag_phase(z: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Decompose complex array into (magnitude, phase) in radians."""
    return jnp.abs(z), jnp.angle(z)


def unwrap_phase_temporal(phase: jnp.ndarray, axis: int = -1) -> jnp.ndarray:
    """Unwrap phase along the time axis (handles ±π discontinuities).

    Phase signals from BOLD are usually small modulations on a slowly drifting
    DC; unwrapping along time removes the spurious jumps. This function uses
    JAX's analogue of `numpy.unwrap`.
    """
    return jnp.unwrap(phase, axis=axis)


def detrend_phase_voxelwise(phase: jnp.ndarray, deg: int = 2) -> jnp.ndarray:
    """Per-voxel polynomial detrending of phase.

    Phase has a slow B0 drift across a session — fit and remove a polynomial
    of degree `deg` per voxel before using phase as a regressor.

    Args:
        phase: (V, T) phase timeseries, voxels x TRs.
        deg: polynomial degree (default cubic).

    Returns:
        (V, T) detrended phase, mean-zeroed per voxel.
    """
    V, T = phase.shape
    t = jnp.arange(T, dtype=jnp.float32) / max(T - 1, 1)
    cols = [jnp.ones(T)] + [t ** k for k in range(1, deg + 1)]
    X = jnp.stack(cols, axis=1)                          # (T, deg+1)
    # Closed-form OLS, identical for every voxel
    Xt = X.T
    XtX_inv_Xt = jnp.linalg.inv(Xt @ X) @ Xt              # (deg+1, T)
    coefs = phase @ XtX_inv_Xt.T                          # (V, deg+1)
    fit = coefs @ X.T                                     # (V, T)
    return phase - fit


def voxelwise_zscore_complex(z: jnp.ndarray, axis: int = 1
                              ) -> jnp.ndarray:
    """Voxelwise z-score for complex data.

    Treats real and imaginary parts independently; preserves the
    cross-component covariance structure in the original signal.
    """
    re_z = (z.real - z.real.mean(axis=axis, keepdims=True)) / (
        z.real.std(axis=axis, keepdims=True) + 1e-8
    )
    im_z = (z.imag - z.imag.mean(axis=axis, keepdims=True)) / (
        z.imag.std(axis=axis, keepdims=True) + 1e-8
    )
    return (re_z + 1j * im_z).astype(jnp.complex64)
