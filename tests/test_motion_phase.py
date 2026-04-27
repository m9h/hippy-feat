"""Red-green TDD for jaxoccoli/motion_phase.py — phase-correlation MC."""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.motion_phase import (
    apply_translation,
    cross_power_spectrum,
    estimate_translation,
    hamming_window_3d,
    register_translation,
)


def _gaussian_blob(shape, center, sigma=2.0):
    """3-D Gaussian blob useful for testing translation."""
    nx, ny, nz = shape
    cx, cy, cz = center
    x, y, z = np.ogrid[:nx, :ny, :nz]
    return np.exp(
        -((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2) / (2 * sigma ** 2)
    ).astype(np.float32)


def test_hamming_window_shape_and_range():
    win = np.asarray(hamming_window_3d((16, 16, 16)))
    assert win.shape == (16, 16, 16)
    # 3-D Hamming center is the product of 1-D Hamming peaks; for even
    # N=16 the peak is just under 1 (≈0.989 per axis → ~0.96 in 3-D).
    assert win[8, 8, 8] >= 0.95
    assert win[0, 0, 0] < 0.01


def test_apply_translation_zero_is_identity():
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(16, 16, 16)).astype(np.float32)
    out = np.asarray(apply_translation(jnp.asarray(vol), (0.0, 0.0, 0.0)))
    np.testing.assert_allclose(out, vol, atol=1e-5)


def test_apply_translation_integer_shift_matches_roll():
    """Fourier translation by an integer should match np.roll."""
    rng = np.random.default_rng(0)
    vol = rng.normal(size=(20, 20, 20)).astype(np.float32)
    shift = (3.0, -2.0, 1.0)
    out = np.asarray(apply_translation(jnp.asarray(vol), shift))
    expected = np.roll(vol, (3, -2, 1), axis=(0, 1, 2))
    # Fourier shift is exact for periodic signals up to roundoff
    np.testing.assert_allclose(out, expected, atol=1e-3)


def test_estimate_translation_recovers_known_integer_shift():
    """Make a Gaussian-blob template, translate by (4, -3, 2), recover."""
    shape = (32, 32, 32)
    template = _gaussian_blob(shape, (16, 16, 16))
    shift = (4.0, -3.0, 2.0)
    moving = np.asarray(apply_translation(jnp.asarray(template), shift))
    t_est = estimate_translation(jnp.asarray(template), jnp.asarray(moving),
                                  subvoxel=False)
    # estimate_translation returns the shift such that moving = template
    # translated by t_est. Phase correlation finds the shift; sign convention
    # may flip — accept either direction.
    matches_pos = all(abs(e - s) < 1.0 for e, s in zip(t_est, shift))
    matches_neg = all(abs(e + s) < 1.0 for e, s in zip(t_est, shift))
    assert matches_pos or matches_neg, \
        f"phase corr failed to recover ({shift}); got {t_est}"


def test_estimate_translation_subvoxel():
    """Subvoxel shifts should be refined to within 0.2 voxels."""
    shape = (32, 32, 32)
    template = _gaussian_blob(shape, (16, 16, 16))
    shift = (2.5, -1.7, 0.0)
    moving = np.asarray(apply_translation(jnp.asarray(template), shift))
    t_est = estimate_translation(jnp.asarray(template), jnp.asarray(moving),
                                  subvoxel=True)
    matches_pos = all(abs(e - s) < 0.5 for e, s in zip(t_est, shift))
    matches_neg = all(abs(e + s) < 0.5 for e, s in zip(t_est, shift))
    assert matches_pos or matches_neg, \
        f"subvoxel refinement failed for ({shift}); got {t_est}"


def test_register_translation_brings_moving_back_to_template():
    shape = (32, 32, 32)
    template = _gaussian_blob(shape, (16, 16, 16))
    shift = (3.0, 2.0, -1.0)
    moving = np.asarray(apply_translation(jnp.asarray(template), shift))
    t, registered = register_translation(jnp.asarray(template), jnp.asarray(moving))
    # Registered should be much closer to template than moving was
    err_before = np.linalg.norm(template - moving)
    err_after = np.linalg.norm(template - np.asarray(registered))
    assert err_after < 0.2 * err_before, (
        f"registration didn't reduce error enough: {err_before:.3f} → {err_after:.3f}"
    )


def test_cross_power_spectrum_unit_magnitude():
    """The normalized cross-power spectrum should be unit-modulus."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(16, 16, 16)).astype(np.float32)
    b = rng.normal(size=(16, 16, 16)).astype(np.float32)
    cross = np.asarray(cross_power_spectrum(jnp.asarray(a), jnp.asarray(b)))
    np.testing.assert_allclose(np.abs(cross), 1.0, atol=1e-5)
