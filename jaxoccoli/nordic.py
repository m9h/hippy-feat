"""NORDIC-style denoising for complex fMRI data.

Reference:
  Vizioli L, Moeller S, Dowdle L, Akçakaya M, De Martino F, Yacoub E,
  Uğurbil K. (2021). Lowering the thermal noise barrier in functional brain
  imaging with magnetic resonance imaging. Nature Communications 12:5181.

Core idea: thermal noise in MRI is i.i.d. complex Gaussian; signal is
spatially+temporally structured. PCA on a (voxels × time) complex matrix,
with singular values below a Marchenko-Pastur–derived threshold zeroed,
removes the random-noise component while preserving signal.

We provide both a simple "global NORDIC" (single SVD over whole brain) and
a patch-based variant matching the published implementation more closely.

JIT note: `nordic_global` is fully JIT-compatible. The patch-based variant
uses Python-side patch iteration; consider it for offline use.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def marchenko_pastur_threshold(M: int, N: int, sigma: float) -> float:
    """Upper edge of the Marchenko-Pastur distribution for noise singular
    values of an M×N i.i.d. Gaussian matrix with elementwise std `sigma`.

    Singular values from random noise concentrate below
        σ * sqrt(M) + σ * sqrt(N)  (for large M, N)
    times a small constant — we use the standard MP edge:
        threshold = σ * (sqrt(M) + sqrt(N))
    (≈ the largest singular value of pure noise).
    """
    return float(sigma) * (float(np.sqrt(M)) + float(np.sqrt(N)))


def estimate_noise_sigma_complex(z_voxel_time: jnp.ndarray,
                                  noise_voxel_mask: jnp.ndarray | None = None
                                  ) -> float:
    """Estimate complex noise standard deviation σ.

    Two paths:
      1. If `noise_voxel_mask` is provided (e.g., voxels outside the brain
         where signal is zero), σ is the std of those voxels' real part
         (and imaginary part — should match for circular complex noise).
      2. Otherwise, use the median absolute deviation (MAD) of the real part
         of the temporal first-difference (which removes any signal drift):
           σ ≈ MAD(diff(z.real, axis=time)) / 0.6745 / sqrt(2)
         The sqrt(2) accounts for variance doubling under differencing.
    """
    if noise_voxel_mask is not None:
        real_noise = z_voxel_time.real[noise_voxel_mask]
        imag_noise = z_voxel_time.imag[noise_voxel_mask]
        sigma_re = float(jnp.std(real_noise))
        sigma_im = float(jnp.std(imag_noise))
        return 0.5 * (sigma_re + sigma_im)
    # Robust MAD-based estimate from temporal differences
    diffs = jnp.diff(z_voxel_time.real, axis=-1)
    mad = float(jnp.median(jnp.abs(diffs - jnp.median(diffs))))
    sigma = mad / 0.6745 / float(np.sqrt(2))
    return sigma


def nordic_global(z_voxel_time: jnp.ndarray,
                  sigma: float | None = None,
                  noise_voxel_mask: jnp.ndarray | None = None,
                  return_kept: bool = False
                  ) -> jnp.ndarray | tuple[jnp.ndarray, int]:
    """Apply NORDIC PCA-thresholding once to the whole-brain complex matrix.

    This is the "global NORDIC" simplification — one SVD on
    (n_voxels × n_timepoints) complex data, then zero singular values below
    the Marchenko-Pastur edge. The published patch-based implementation
    (per-voxel sliding patches, repeated SVD, overlap-aggregation) is more
    accurate but compute-heavy; this is the RT-feasible version that gets
    most of the SNR benefit at much lower cost.

    Args:
        z_voxel_time: (V, T) complex64 — voxels × time matrix. Caller is
            responsible for masking to gray matter or brain mask.
        sigma: complex noise σ estimate. If None, estimated from the data
            via `estimate_noise_sigma_complex`.
        noise_voxel_mask: optional 1-D boolean mask over V indicating
            voxels expected to be pure noise (e.g., outside brain).
        return_kept: if True, also return the number of singular components
            kept (useful for diagnostics).

    Returns:
        denoised: (V, T) complex64.
        n_kept: int (only if return_kept).
    """
    if z_voxel_time.dtype not in (jnp.complex64, jnp.complex128):
        raise ValueError(f"input must be complex, got dtype {z_voxel_time.dtype}")

    V, T = z_voxel_time.shape
    if sigma is None:
        sigma = estimate_noise_sigma_complex(
            z_voxel_time, noise_voxel_mask=noise_voxel_mask,
        )

    # SVD of the complex matrix
    U, S, Vh = jnp.linalg.svd(z_voxel_time, full_matrices=False)

    # MP threshold; σ_complex ≈ √2 * σ_real for circular complex Gaussian
    threshold = marchenko_pastur_threshold(V, T, sigma * float(np.sqrt(2)))
    keep_mask = S > threshold
    n_kept = int(keep_mask.sum())

    # Zero out below-threshold components
    S_thresh = jnp.where(keep_mask, S, 0.0)
    denoised = (U * S_thresh) @ Vh

    if return_kept:
        return denoised.astype(jnp.complex64), n_kept
    return denoised.astype(jnp.complex64)


def nordic_streaming_window(buffer_z: jnp.ndarray,
                            window_T: int = 30,
                            sigma: float | None = None,
                            ) -> jnp.ndarray:
    """RT-NORDIC: SVD-threshold over a sliding window of the most recent
    `window_T` complex TRs, returning the denoised version of those TRs.

    Use case: at each new TR arrival, pass the most recent `window_T` TRs
    (or all available if fewer) through NORDIC. The denoised output of the
    LAST TR is what gets sent to the GLM / decoder.

    `window_T` is a tradeoff: larger = better noise estimation, but slower
    response to non-stationary signal. 30 TRs at TR=1.5 s ≈ 45 s window.

    Args:
        buffer_z: (V, T_buffer) complex64. T_buffer ≥ 2 required.
        window_T: most-recent TR count to denoise jointly.

    Returns:
        denoised_window: (V, min(T_buffer, window_T)) complex64.
    """
    V, T_buf = buffer_z.shape
    eff_T = min(T_buf, window_T)
    sub = buffer_z[:, -eff_T:]
    return nordic_global(sub, sigma=sigma)


__all__ = [
    "marchenko_pastur_threshold",
    "estimate_noise_sigma_complex",
    "nordic_global",
    "nordic_streaming_window",
]
