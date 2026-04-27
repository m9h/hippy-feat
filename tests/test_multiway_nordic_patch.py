"""Red-green TDD for the next iteration of multiway NORDIC: patch-based Tucker.

Building on the working `hosvd_threshold_4d` (passes 5 tests). The next
upgrade is patch-based Tucker that matches Vizioli 2021's published
implementation: spatial patches × T temporal × per-patch Tucker × overlap-
aggregate. Better preserves local spatial structure than global HOSVD.
"""
from __future__ import annotations

import numpy as np
import pytest

# Patch-based path doesn't exist yet — this is the RED gate.
pytest.importorskip("jaxoccoli.multiway_nordic_patch",
                     reason="awaiting patch-based Tucker implementation")

from jaxoccoli.multiway_nordic_patch import patch_tucker_threshold_4d


def test_patch_tucker_shape_preserved():
    rng = np.random.default_rng(0)
    shape = (16, 16, 16, 30)
    z = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    out, n_patches = patch_tucker_threshold_4d(z, patch_size=4, stride=2)
    assert out.shape == shape
    assert n_patches > 0


def test_patch_tucker_recovers_local_low_rank():
    """Each spatial patch should be locally low-rank under additive noise."""
    rng = np.random.default_rng(0)
    shape = (12, 12, 12, 40)
    # Build low-rank signal locally
    signal = np.zeros(shape, dtype=np.complex64)
    for px in range(0, shape[0], 4):
        for py in range(0, shape[1], 4):
            for pz in range(0, shape[2], 4):
                # Shared temporal pattern across this patch
                t_pat = (rng.normal(size=shape[-1])
                         + 1j * rng.normal(size=shape[-1])).astype(np.complex64)
                signal[px:px+4, py:py+4, pz:pz+4, :] = t_pat[None, None, None, :]
    noise = (0.1 * (rng.normal(size=shape) + 1j * rng.normal(size=shape))
             ).astype(np.complex64)
    z_noisy = signal + noise
    out, _ = patch_tucker_threshold_4d(z_noisy, patch_size=4, stride=2)
    err_noisy = np.linalg.norm(z_noisy - signal)
    err_clean = np.linalg.norm(out - signal)
    assert err_clean < err_noisy
