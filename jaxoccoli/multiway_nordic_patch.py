"""Patch-based Tucker NORDIC — closer to Vizioli 2021's published algorithm.

Where `multiway_nordic.hosvd_threshold_4d` does ONE global HOSVD on the
whole `(X, Y, Z, T)` tensor, this module sweeps small spatial patches
(e.g., 4×4×4×T or 5×5×5×T), runs HOSVD per patch with per-mode MP
thresholding, and aggregates overlapping patches via average.

Why patches matter: globally low-rank assumptions are too strong on real
fMRI — different brain regions have very different signal structure.
Locally, a small patch is much closer to genuinely low-rank, so
truncation throws away less signal.

Reference:
    Vizioli L, Moeller S, Dowdle L, et al. (2021) Lowering the thermal
    noise barrier in functional brain imaging with magnetic resonance
    imaging. Nat Commun 12:5181.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from .nordic import nordic_global


def patch_tucker_threshold_4d(tensor,
                               patch_size: int = 4,
                               stride: int = 2,
                               sigma: float | None = None,
                               complex_data: bool = True
                               ) -> tuple[np.ndarray, int]:
    """Sweep spatial patches; HOSVD-threshold each; average overlaps.

    Args:
        tensor: (X, Y, Z, T) array — float32 or complex64.
        patch_size: spatial side length of each patch (uniform along X, Y, Z).
        stride: step between patch starts. `stride < patch_size` gives
            overlapping coverage; `stride == patch_size` gives a tiling.
        sigma: noise σ per real component; if None, robust MAD estimate
            from temporal first-difference of the real part.
        complex_data: True for complex input, False for real.

    Returns:
        cleaned tensor (same shape and dtype as input).
        n_patches: number of HOSVD calls executed.

    Notes:
      - Patches that extend past the FOV are clipped (smaller actual size).
        Aggregation accounts for variable patch counts per voxel.
      - For real-time use, this is offline-only — `n_patches` HOSVDs per
        run is too slow per-TR. The streaming variant (sliding window
        over T only, single global HOSVD per window) lives in
        `multiway_nordic.nordic_streaming_window`.
    """
    if tensor.ndim != 4:
        raise ValueError(f"expected 4-D, got {tensor.shape}")
    arr = np.asarray(tensor)
    X, Y, Z, T = arr.shape
    out = np.zeros_like(arr)
    count = np.zeros((X, Y, Z), dtype=np.float32)

    n_patches = 0
    for px in range(0, X, stride):
        for py in range(0, Y, stride):
            for pz in range(0, Z, stride):
                ex = min(px + patch_size, X)
                ey = min(py + patch_size, Y)
                ez = min(pz + patch_size, Z)
                if (ex - px) < 2 or (ey - py) < 2 or (ez - pz) < 2:
                    continue
                # Per Vizioli 2021: unfold the patch to (V_patch, T) and do
                # ONE SVD with MP threshold. This is the actual published
                # NORDIC patch algorithm — full HOSVD on small patches
                # over-truncates spatial modes whose unfoldings are tiny.
                patch = arr[px:ex, py:ey, pz:ez, :]
                Px, Py, Pz, Tp = patch.shape
                patch_2d = patch.reshape(Px * Py * Pz, Tp)
                cleaned_2d = nordic_global(
                    jnp.asarray(patch_2d), sigma=sigma,
                )
                cleaned = np.asarray(cleaned_2d).reshape(Px, Py, Pz, Tp)
                out[px:ex, py:ey, pz:ez, :] += cleaned
                count[px:ex, py:ey, pz:ez] += 1.0
                n_patches += 1

    # Avoid division by zero for any spatial voxel that was never covered
    safe_count = np.maximum(count, 1.0)
    out = out / safe_count[..., None]
    out = out.astype(arr.dtype)
    return out, n_patches


__all__ = ["patch_tucker_threshold_4d"]
