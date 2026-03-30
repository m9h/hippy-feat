"""Tests for jaxoccoli.fourier module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.fourier import (
    analytic_signal,
    envelope,
    hilbert_transform,
    instantaneous_frequency,
    instantaneous_phase,
    product_filter,
    product_filtfilt,
    unwrap,
)

KEY = jax.random.PRNGKey(99)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sine_signal():
    """Pure 5 Hz sine at 100 Hz sampling rate, 2 seconds."""
    t = jnp.linspace(0, 2.0, 200, endpoint=False)
    return jnp.sin(2 * jnp.pi * 5.0 * t), t


@pytest.fixture
def multi_channel():
    """(3 channels, 200 samples) multi-channel signal."""
    t = jnp.linspace(0, 2.0, 200, endpoint=False)
    freqs = jnp.array([3.0, 5.0, 10.0])
    return jnp.sin(2 * jnp.pi * freqs[:, None] * t[None, :]), t


# ---------------------------------------------------------------------------
# Analytic signal / Hilbert
# ---------------------------------------------------------------------------

class TestAnalyticSignal:
    def test_real_part_preserved(self, sine_signal):
        x, _ = sine_signal
        z = analytic_signal(x)
        np.testing.assert_allclose(jnp.real(z), x, atol=1e-5)

    def test_output_complex(self, sine_signal):
        x, _ = sine_signal
        z = analytic_signal(x)
        assert jnp.iscomplexobj(z)

    def test_shape_preserved(self, multi_channel):
        X, _ = multi_channel
        Z = analytic_signal(X, axis=-1)
        assert Z.shape == X.shape

    def test_jit(self, sine_signal):
        x, _ = sine_signal
        z = jax.jit(analytic_signal)(x)
        np.testing.assert_allclose(jnp.real(z), x, atol=1e-5)


class TestHilbertTransform:
    def test_sine_to_cosine(self, sine_signal):
        """Hilbert transform of sin should be -cos (up to edge effects)."""
        x, t = sine_signal
        h = hilbert_transform(x)
        expected = -jnp.cos(2 * jnp.pi * 5.0 * t)
        # Ignore edges (edge effects from finite length)
        np.testing.assert_allclose(h[20:-20], expected[20:-20], atol=0.05)

    def test_shape(self, multi_channel):
        X, _ = multi_channel
        H = hilbert_transform(X, axis=-1)
        assert H.shape == X.shape


class TestEnvelope:
    def test_constant_for_sine(self, sine_signal):
        """Envelope of a pure sine should be ~1.0 (constant amplitude)."""
        x, _ = sine_signal
        env = envelope(x)
        # Ignore edges
        np.testing.assert_allclose(env[20:-20], 1.0, atol=0.05)

    def test_nonnegative(self, sine_signal):
        x, _ = sine_signal
        env = envelope(x)
        assert jnp.all(env >= -1e-6)

    def test_am_signal(self):
        """Envelope of AM signal should recover the modulator."""
        t = jnp.linspace(0, 2.0, 1000, endpoint=False)
        modulator = 1.0 + 0.5 * jnp.sin(2 * jnp.pi * 2.0 * t)
        carrier = jnp.sin(2 * jnp.pi * 50.0 * t)
        am = modulator * carrier
        env = envelope(am)
        # Envelope should track the modulator (ignore edges)
        np.testing.assert_allclose(env[100:-100], modulator[100:-100], atol=0.15)


class TestInstantaneousPhase:
    def test_linear_for_sine(self, sine_signal):
        """Phase of a pure sine should increase linearly."""
        x, t = sine_signal
        phase = instantaneous_phase(x)
        # Unwrap and check linearity
        phase_unwrapped = np.unwrap(np.array(phase))
        # Linear fit
        coeffs = np.polyfit(np.array(t[20:-20]), phase_unwrapped[20:-20], 1)
        expected_slope = 2 * np.pi * 5.0
        np.testing.assert_allclose(coeffs[0], expected_slope, rtol=0.05)


class TestInstantaneousFrequency:
    def test_constant_for_sine(self, sine_signal):
        x, _ = sine_signal
        freq = instantaneous_frequency(x, fs=100.0)
        # Should be ~5 Hz (ignoring edges and the fact we lose one sample)
        np.testing.assert_allclose(freq[30:-30], 5.0, atol=0.5)


# ---------------------------------------------------------------------------
# Frequency-domain filtering
# ---------------------------------------------------------------------------

class TestProductFilter:
    def test_identity_filter(self, sine_signal):
        x, _ = sine_signal
        n = len(x)
        w = jnp.ones(n // 2 + 1)
        y = product_filter(x, w)
        np.testing.assert_allclose(y, x, atol=1e-5)

    def test_zero_filter(self, sine_signal):
        x, _ = sine_signal
        n = len(x)
        w = jnp.zeros(n // 2 + 1)
        y = product_filter(x, w)
        np.testing.assert_allclose(y, 0.0, atol=1e-5)

    def test_lowpass(self):
        """Low-pass filter should remove high frequency component."""
        t = jnp.linspace(0, 1.0, 200, endpoint=False)
        low = jnp.sin(2 * jnp.pi * 5.0 * t)
        high = jnp.sin(2 * jnp.pi * 40.0 * t)
        x = low + high
        # Create lowpass: pass below bin 20, block above
        n_freq = 101  # rfft bins for n=200
        w = jnp.where(jnp.arange(n_freq) < 15, 1.0, 0.0)
        y = product_filter(x, w)
        # Result should mostly be the low-frequency component
        np.testing.assert_allclose(y[20:-20], low[20:-20], atol=0.3)

    def test_shape_multichannel(self, multi_channel):
        X, _ = multi_channel
        n = X.shape[-1]
        w = jnp.ones(n // 2 + 1)
        Y = product_filter(X, w, axis=-1)
        assert Y.shape == X.shape

    def test_jit(self, sine_signal):
        x, _ = sine_signal
        w = jnp.ones(len(x) // 2 + 1)
        y = jax.jit(lambda x, w: product_filter(x, w))(x, w)
        np.testing.assert_allclose(y, x, atol=1e-5)

    def test_grad(self):
        x = jax.random.normal(KEY, (50,))
        w = jnp.ones(26)  # rfft of 50

        def loss(w):
            return jnp.sum(product_filter(x, w) ** 2)

        g = jax.grad(loss)(w)
        assert g.shape == w.shape
        assert jnp.all(jnp.isfinite(g))


class TestProductFiltfilt:
    def test_squared_magnitude(self, sine_signal):
        x, _ = sine_signal
        n = len(x)
        w = 0.5 * jnp.ones(n // 2 + 1)
        y1 = product_filtfilt(x, w)
        y2 = product_filter(x, w * jnp.conj(w))
        np.testing.assert_allclose(y1, y2, atol=1e-5)


# ---------------------------------------------------------------------------
# Phase unwrapping
# ---------------------------------------------------------------------------

class TestUnwrap:
    def test_basic(self):
        """Unwrap a linearly increasing phase with wraps."""
        phase = jnp.linspace(0, 6 * jnp.pi, 100)
        wrapped = jnp.mod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi
        unwrapped = unwrap(wrapped)
        # Should recover original linear trend (up to constant offset)
        diff = unwrapped - phase
        np.testing.assert_allclose(diff - diff[0], 0.0, atol=0.1)

    def test_no_change_smooth(self):
        """Smooth phase should not be modified."""
        phase = jnp.linspace(0, 1.0, 50)
        result = unwrap(phase)
        np.testing.assert_allclose(result, phase, atol=1e-7)

    def test_multidimensional(self):
        phase = jnp.linspace(0, 6 * jnp.pi, 100)
        wrapped = jnp.mod(phase + jnp.pi, 2 * jnp.pi) - jnp.pi
        batch = jnp.stack([wrapped, wrapped * 0.5])
        result = unwrap(batch, axis=-1)
        assert result.shape == batch.shape
