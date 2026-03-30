"""Frequency-domain signal processing operations.

Ports essential algorithms from hypercoil functional/fourier.py in vbjax
style (pure functions, JIT/vmap/grad compatible, no Equinox).
"""

import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Analytic signal and derived quantities
# ---------------------------------------------------------------------------

def analytic_signal(X, axis=-1, n=None):
    """Compute the analytic signal via FFT-based Hilbert transform.

    Args:
        X: Real-valued input array.
        axis: Axis along which to compute (default: last).
        n: FFT length.  If None, uses X.shape[axis].

    Returns:
        Complex analytic signal: X + j * hilbert(X).
    """
    N = X.shape[axis] if n is None else n
    Xf = jnp.fft.fft(X, n=N, axis=axis)

    # Build the doubling filter h: h[0]=1, h[1:N//2]=2, h[N//2]=1, rest=0
    h = jnp.zeros(N)
    if N % 2 == 0:
        h = h.at[0].set(1.0)
        h = h.at[1:N // 2].set(2.0)
        h = h.at[N // 2].set(1.0)
    else:
        h = h.at[0].set(1.0)
        h = h.at[1:(N + 1) // 2].set(2.0)

    # Broadcast h to match Xf shape along the transform axis
    shape = [1] * X.ndim
    shape[axis] = N
    h = h.reshape(shape)

    return jnp.fft.ifft(Xf * h, axis=axis)


def hilbert_transform(X, axis=-1, n=None):
    """Hilbert transform (imaginary part of the analytic signal).

    Args:
        X: Real-valued input array.
        axis: Axis along which to compute.
        n: FFT length.

    Returns:
        Hilbert transform of X (real-valued).
    """
    return jnp.imag(analytic_signal(X, axis=axis, n=n))


def envelope(X, axis=-1, n=None):
    """Instantaneous amplitude (envelope) of the signal.

    Args:
        X: Real-valued input array.
        axis: Axis along which to compute.
        n: FFT length.

    Returns:
        Signal envelope |analytic_signal(X)|.
    """
    return jnp.abs(analytic_signal(X, axis=axis, n=n))


def instantaneous_phase(X, axis=-1, n=None):
    """Instantaneous phase of the signal.

    Args:
        X: Real-valued input array.
        axis: Axis along which to compute.
        n: FFT length.

    Returns:
        Phase in radians, wrapped to [-pi, pi].
    """
    return jnp.angle(analytic_signal(X, axis=axis, n=n))


def instantaneous_frequency(X, axis=-1, n=None, fs=1.0):
    """Instantaneous frequency via phase derivative.

    Args:
        X: Real-valued input array.
        axis: Axis along which to compute.
        n: FFT length.
        fs: Sampling frequency.

    Returns:
        Instantaneous frequency in Hz (same units as fs).
    """
    phase = instantaneous_phase(X, axis=axis, n=n)
    phase = unwrap(phase, axis=axis)
    dphase = jnp.diff(phase, axis=axis)
    return dphase * fs / (2.0 * jnp.pi)


# ---------------------------------------------------------------------------
# Frequency-domain filtering
# ---------------------------------------------------------------------------

def product_filter(X, weight, axis=-1):
    """Frequency-domain filtering via pointwise multiplication.

    Args:
        X: Input signal (..., T) along axis.
        weight: (T,) or (T//2+1,) transfer function.
            If real-valued and length T//2+1, applied to rfft.
            If complex or length T, applied to full fft.
        axis: Axis along which to filter.

    Returns:
        Filtered signal (same shape as X).
    """
    n = X.shape[axis]
    use_rfft = weight.shape[-1] == n // 2 + 1 and jnp.isrealobj(weight)

    if use_rfft:
        Xf = jnp.fft.rfft(X, axis=axis)
        # Broadcast weight
        shape = [1] * X.ndim
        shape[axis] = weight.shape[-1]
        w = weight.reshape(shape)
        return jnp.fft.irfft(Xf * w, n=n, axis=axis)
    else:
        Xf = jnp.fft.fft(X, axis=axis)
        shape = [1] * X.ndim
        shape[axis] = weight.shape[-1]
        w = weight.reshape(shape)
        return jnp.real(jnp.fft.ifft(Xf * w, axis=axis))


def product_filtfilt(X, weight, axis=-1):
    """Zero-phase frequency-domain filtering (forward + backward).

    Equivalent to applying the squared magnitude of the transfer function.

    Args:
        X: Input signal.
        weight: Transfer function (see product_filter).
        axis: Axis along which to filter.

    Returns:
        Zero-phase filtered signal.
    """
    return product_filter(X, weight * jnp.conj(weight), axis=axis)


# ---------------------------------------------------------------------------
# Phase unwrapping
# ---------------------------------------------------------------------------

def unwrap(phase, axis=-1, discont=None, period=2.0 * jnp.pi):
    """Unwrap phase angles by changing absolute jumps > discont to their
    period-complement.

    JAX-compatible reimplementation of numpy.unwrap.

    Args:
        phase: Array of phase angles.
        axis: Axis along which to unwrap.
        discont: Maximum discontinuity (default: period/2).
        period: Phase wrapping period (default: 2*pi).

    Returns:
        Unwrapped phase.
    """
    if discont is None:
        discont = period / 2.0

    diff = jnp.diff(phase, axis=axis)
    # Wrap differences to [-period/2, period/2)
    diff_mod = jnp.mod(diff + period / 2.0, period) - period / 2.0
    # Correct the boundary case
    diff_mod = jnp.where(
        (diff_mod == -period / 2.0) & (diff > 0),
        period / 2.0,
        diff_mod,
    )
    # Only correct where discontinuity exceeds threshold
    correction = diff_mod - diff
    correction = jnp.where(jnp.abs(diff) < discont, 0.0, correction)

    return phase.at[tuple(
        slice(1, None) if i == (axis % phase.ndim) else slice(None)
        for i in range(phase.ndim)
    )].add(jnp.cumsum(correction, axis=axis))
