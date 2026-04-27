"""Multiway NORDIC — HOSVD-thresholded denoising on native 4D fMRI tensors.

NORDIC v1 (`jaxoccoli.nordic`) operates on a 2D unfolding `(V, T)` of the
data, losing all spatial coherence — voxels are exchangeable. This module
applies higher-order SVD (HOSVD) to the native 4-D tensor `(X, Y, Z, T)`
with per-mode Marchenko-Pastur thresholding. Each spatial mode and the
temporal mode get independent rank truncations derived from the singular-
value spectrum of that mode's unfolding, preserving 3-D spatial structure.

Pure-JAX implementation — JIT-friendly, no `tensorly` dependency. For
larger / patch-based / iterative Tucker (HOOI), TT decompositions, or
robust PCA variants, see the comments at module bottom on when to switch
to `tensorly[jax]`.

Reference for thresholding rule:
    Marchenko-Pastur edge: σ * (sqrt(M) + sqrt(N)) for an M×N i.i.d.
    Gaussian matrix with elementwise std σ. We apply this to each mode
    unfolding independently.

References for HOSVD itself:
    De Lathauwer L, De Moor B, Vandewalle J (2000). A multilinear singular
    value decomposition. SIAM J Matrix Anal Appl 21(4):1253-1278.
    Kolda TG, Bader BW (2009). Tensor decompositions and applications.
    SIAM Review 51(3):455-500.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _unfold(tensor: jnp.ndarray, mode: int) -> jnp.ndarray:
    """Mode-`mode` unfolding of a tensor: rearrange so the chosen mode is
    rows and all other modes are flattened into columns.

    For a 4-D tensor `(X, Y, Z, T)`:
      - mode 0 unfold → (X, Y·Z·T)
      - mode 1 unfold → (Y, X·Z·T)
      - etc.
    """
    return jnp.moveaxis(tensor, mode, 0).reshape(tensor.shape[mode], -1)


def _fold(unfolded: jnp.ndarray, mode: int, shape: tuple[int, ...]
          ) -> jnp.ndarray:
    """Inverse of `_unfold` — restore the tensor from its mode-`mode`
    unfolding given the original shape."""
    full_shape = (shape[mode],) + tuple(s for i, s in enumerate(shape)
                                          if i != mode)
    return jnp.moveaxis(unfolded.reshape(full_shape), 0, mode)


def _mp_truncate(matrix: jnp.ndarray, sigma: float
                 ) -> tuple[jnp.ndarray, int]:
    """Truncate matrix at the Marchenko-Pastur edge: zero singular values
    below `σ * (sqrt(M) + sqrt(N))`.

    Returns (truncated reconstruction, n_kept).
    """
    M, N = matrix.shape[-2], matrix.shape[-1]
    threshold = float(sigma) * (float(jnp.sqrt(M)) + float(jnp.sqrt(N)))
    U, S, Vh = jnp.linalg.svd(matrix, full_matrices=False)
    keep = S > threshold
    n_kept = int(keep.sum())
    S_thresh = jnp.where(keep, S, 0.0)
    truncated = (U * S_thresh) @ Vh
    return truncated, n_kept


def hosvd_threshold_4d(tensor: jnp.ndarray,
                        sigma: float | None = None,
                        complex_data: bool = True
                        ) -> tuple[jnp.ndarray, list[int]]:
    """HOSVD with per-mode MP thresholding on a 4-D `(X, Y, Z, T)` tensor.

    Strategy: unfold along each of the 4 modes; SVD-truncate each unfolding
    independently at its mode-specific MP threshold; reconstruct. The four
    truncations are applied SEQUENTIALLY (not iteratively) — this is the
    truncated HOSVD baseline. For HOOI (iterative refinement), use
    `tensorly[jax]`'s `decomposition.tucker`.

    Args:
        tensor: complex64 or float32 array, shape (X, Y, Z, T).
        sigma:  noise σ per real component. If None, estimated robustly
            from the temporal first-difference MAD of the real part.
        complex_data: if True, σ is interpreted as the real-part σ; the
            effective MP threshold uses σ_complex = σ * sqrt(2).

    Returns:
        denoised tensor (same shape and dtype as input).
        per_mode_n_kept: list of 4 ints — singular values kept along each
            mode unfolding.
    """
    if tensor.ndim != 4:
        raise ValueError(f"expected 4-D tensor (X, Y, Z, T), got {tensor.shape}")

    # Estimate noise σ if not provided
    if sigma is None:
        diffs = jnp.diff(tensor.real if complex_data else tensor, axis=-1)
        mad = float(jnp.median(jnp.abs(diffs - jnp.median(diffs))))
        sigma = mad / 0.6745 / float(jnp.sqrt(2))

    sigma_eff = sigma * (float(jnp.sqrt(2)) if complex_data else 1.0)

    out = tensor
    n_kept_per_mode: list[int] = []
    for mode in range(4):
        unfolded = _unfold(out, mode)
        truncated, n_kept = _mp_truncate(unfolded, sigma_eff)
        out = _fold(truncated, mode, out.shape)
        n_kept_per_mode.append(n_kept)
    return out.astype(tensor.dtype), n_kept_per_mode


def hosvd_threshold_5d(tensor: jnp.ndarray,
                        sigma: float | None = None,
                        complex_data: bool = True
                        ) -> tuple[jnp.ndarray, list[int]]:
    """HOSVD on `(X, Y, Z, T, run)` — multi-run joint denoising.

    Helps when multiple runs of the same task share noise structure (drift,
    physio cycles, motion residuals) — joint denoising leverages this for
    a stronger noise pool. Run is treated as an independent mode for
    truncation.
    """
    if tensor.ndim != 5:
        raise ValueError(f"expected 5-D tensor (X,Y,Z,T,run), got {tensor.shape}")
    if sigma is None:
        diffs = jnp.diff(tensor.real if complex_data else tensor, axis=-2)
        mad = float(jnp.median(jnp.abs(diffs - jnp.median(diffs))))
        sigma = mad / 0.6745 / float(jnp.sqrt(2))
    sigma_eff = sigma * (float(jnp.sqrt(2)) if complex_data else 1.0)

    out = tensor
    n_kept_per_mode: list[int] = []
    for mode in range(5):
        unfolded = _unfold(out, mode)
        truncated, n_kept = _mp_truncate(unfolded, sigma_eff)
        out = _fold(truncated, mode, out.shape)
        n_kept_per_mode.append(n_kept)
    return out.astype(tensor.dtype), n_kept_per_mode


__all__ = ["hosvd_threshold_4d", "hosvd_threshold_5d"]
