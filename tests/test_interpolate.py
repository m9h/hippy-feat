"""Tests for jaxoccoli.interpolate module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.interpolate import (
    hybrid_interpolate,
    linear_interpolate,
    spectral_interpolate,
)

KEY = jax.random.PRNGKey(55)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_data():
    """Smooth sine signal with known values at all timepoints."""
    t = jnp.linspace(0, 2 * jnp.pi, 100)
    return jnp.sin(t), t


@pytest.fixture
def multichannel_data():
    """(3 channels, 100 timepoints) smooth signals."""
    t = jnp.linspace(0, 2 * jnp.pi, 100)
    return jnp.stack([jnp.sin(t), jnp.cos(t), jnp.sin(2 * t)]), t


# ---------------------------------------------------------------------------
# Linear interpolation
# ---------------------------------------------------------------------------

class TestLinearInterpolate:
    def test_no_censoring_unchanged(self, sine_data):
        data, _ = sine_data
        mask = jnp.ones(100, dtype=bool)
        result = linear_interpolate(data, mask)
        np.testing.assert_allclose(result, data, atol=1e-6)

    def test_single_gap(self, sine_data):
        data, t = sine_data
        mask = jnp.ones(100, dtype=bool).at[50].set(False)
        result = linear_interpolate(data, mask)
        # Interpolated value should be between neighbours
        expected = (data[49] + data[51]) / 2
        np.testing.assert_allclose(result[50], expected, atol=1e-4)

    def test_valid_frames_preserved(self, sine_data):
        data, _ = sine_data
        mask = jnp.ones(100, dtype=bool).at[30:35].set(False)
        result = linear_interpolate(data, mask)
        # Valid frames should be exactly preserved
        valid_idx = jnp.where(mask)[0]
        np.testing.assert_allclose(result[valid_idx], data[valid_idx], atol=1e-7)

    def test_multichannel(self, multichannel_data):
        data, _ = multichannel_data
        mask = jnp.ones(100, dtype=bool).at[40:45].set(False)
        result = linear_interpolate(data, mask)
        assert result.shape == data.shape

    def test_edge_censoring(self, sine_data):
        """Censoring at the start should use nearest valid value."""
        data, _ = sine_data
        mask = jnp.ones(100, dtype=bool).at[:3].set(False)
        result = linear_interpolate(data, mask)
        # First valid is index 3, so censored frames should take its value
        np.testing.assert_allclose(result[0], data[3], atol=1e-5)


# ---------------------------------------------------------------------------
# Spectral interpolation
# ---------------------------------------------------------------------------

class TestSpectralInterpolate:
    def test_no_censoring_unchanged(self, sine_data):
        data, _ = sine_data
        mask = jnp.ones(100, dtype=bool)
        result = spectral_interpolate(data, mask, sampling_period=0.1)
        np.testing.assert_allclose(result, data, atol=0.1)

    def test_recovers_sine(self, sine_data):
        """Should recover a smooth sine from sparse valid points."""
        data, _ = sine_data
        # Censor 10 consecutive frames in the middle
        mask = jnp.ones(100, dtype=bool).at[45:55].set(False)
        result = spectral_interpolate(data, mask, sampling_period=0.1)
        # Reconstructed censored region should be close to true
        np.testing.assert_allclose(result[45:55], data[45:55], atol=0.3)

    def test_valid_frames_close(self, sine_data):
        data, _ = sine_data
        mask = jnp.ones(100, dtype=bool).at[45:55].set(False)
        result = spectral_interpolate(data, mask, sampling_period=0.1)
        valid_idx = jnp.where(mask)[0]
        np.testing.assert_allclose(result[valid_idx], data[valid_idx], atol=0.15)

    def test_shape(self, multichannel_data):
        data, _ = multichannel_data
        mask = jnp.ones(100, dtype=bool).at[40:50].set(False)
        result = spectral_interpolate(data, mask, sampling_period=0.1)
        assert result.shape == data.shape


# ---------------------------------------------------------------------------
# Hybrid interpolation
# ---------------------------------------------------------------------------

class TestHybridInterpolate:
    def test_no_censoring_unchanged(self, sine_data):
        data, _ = sine_data
        mask = jnp.ones(100, dtype=bool)
        result = hybrid_interpolate(data, mask, sampling_period=0.1)
        np.testing.assert_allclose(result, data, atol=1e-6)

    def test_short_gap_uses_linear(self, sine_data):
        """Short gaps should produce linear interpolation results."""
        data, _ = sine_data
        mask = jnp.ones(100, dtype=bool).at[50].set(False)  # 1 frame gap
        result_hybrid = hybrid_interpolate(data, mask, max_consecutive_linear=3,
                                            sampling_period=0.1)
        result_linear = linear_interpolate(data, mask)
        np.testing.assert_allclose(result_hybrid[50], result_linear[50], atol=1e-5)

    def test_valid_preserved(self, sine_data):
        data, _ = sine_data
        mask = jnp.ones(100, dtype=bool).at[30:40].set(False)
        result = hybrid_interpolate(data, mask, sampling_period=0.1)
        valid_idx = jnp.where(mask)[0]
        np.testing.assert_allclose(result[valid_idx], data[valid_idx], atol=1e-6)

    def test_shape(self, multichannel_data):
        data, _ = multichannel_data
        mask = jnp.ones(100, dtype=bool).at[40:50].set(False)
        result = hybrid_interpolate(data, mask, sampling_period=0.1)
        assert result.shape == data.shape
