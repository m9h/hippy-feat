"""Red-green TDD for jaxoccoli/multiway_nordic.py."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.multiway_nordic import hosvd_threshold_4d, hosvd_threshold_5d


def test_hosvd_4d_shape_preserved():
    rng = np.random.default_rng(0)
    shape = (8, 8, 8, 30)
    z = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    out, n_kept = hosvd_threshold_4d(jnp.asarray(z))
    assert out.shape == shape
    assert out.dtype == np.complex64
    assert len(n_kept) == 4


def test_hosvd_4d_recovers_low_rank_signal():
    """Low-rank tensor + small noise — HOSVD should recover the low-rank
    component well."""
    rng = np.random.default_rng(0)
    X, Y, Z, T = 6, 6, 6, 40
    # Build a rank-(2, 2, 2, 3) tensor by Tucker construction
    core = rng.normal(size=(2, 2, 2, 3)).astype(np.complex64)
    Ux = rng.normal(size=(X, 2)).astype(np.complex64)
    Uy = rng.normal(size=(Y, 2)).astype(np.complex64)
    Uz = rng.normal(size=(Z, 2)).astype(np.complex64)
    Ut = rng.normal(size=(T, 3)).astype(np.complex64)
    # Tensor product
    signal = np.einsum("abcd,Xa,Yb,Zc,Td->XYZT", core, Ux, Uy, Uz, Ut)
    noise = (0.05 * (rng.normal(size=signal.shape)
                     + 1j * rng.normal(size=signal.shape))).astype(np.complex64)
    z_noisy = signal + noise
    z_clean, n_kept = hosvd_threshold_4d(jnp.asarray(z_noisy))
    z_clean = np.asarray(z_clean)
    err_noisy = np.linalg.norm(z_noisy - signal)
    err_clean = np.linalg.norm(z_clean - signal)
    assert err_clean < err_noisy, (
        f"HOSVD did not reduce reconstruction error: noisy={err_noisy:.3f}, "
        f"clean={err_clean:.3f}"
    )


def test_hosvd_4d_drops_high_frequency_noise():
    """Pure-noise tensor: HOSVD should keep very few components (most σ_i
    fall below the MP edge)."""
    rng = np.random.default_rng(0)
    shape = (10, 10, 10, 50)
    sigma = 0.5
    z = sigma * (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    _, n_kept = hosvd_threshold_4d(jnp.asarray(z), sigma=sigma)
    # In pure noise, MP edge cuts almost everything
    for mode_keep in n_kept:
        # Allow up to a few stragglers; certainly not "all"
        assert mode_keep < min(shape) * 0.5, (
            f"HOSVD kept {mode_keep} singular values from pure noise "
            f"(should be ≪ {min(shape)})"
        )


def test_hosvd_4d_real_data_dtype():
    """Real-valued (float32) input handled correctly."""
    rng = np.random.default_rng(0)
    z = rng.normal(size=(6, 6, 6, 20)).astype(np.float32)
    out, _ = hosvd_threshold_4d(jnp.asarray(z), complex_data=False, sigma=0.5)
    assert out.dtype == np.float32


def test_hosvd_5d_shape_preserved():
    rng = np.random.default_rng(0)
    shape = (5, 5, 5, 20, 4)            # X, Y, Z, T, run
    z = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    out, n_kept = hosvd_threshold_5d(jnp.asarray(z))
    assert out.shape == shape
    assert len(n_kept) == 5
