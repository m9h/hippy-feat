"""Temporal interpolation for censored/scrubbed fMRI frames.

Ports essential algorithms from hypercoil functional/interpolate.py in
vbjax style (pure functions, JIT/vmap/grad compatible, no Equinox).
"""

import jax
import jax.numpy as jnp


def linear_interpolate(data, mask):
    """Proximity-weighted linear interpolation of missing frames.

    For each masked (censored) timepoint, interpolates from the nearest
    valid timepoints on each side using inverse-distance weighting.

    Args:
        data: (..., T) time series.
        mask: (T,) boolean mask.  True = valid, False = censored.

    Returns:
        (..., T) interpolated time series (valid frames unchanged).
    """
    T = data.shape[-1]
    t = jnp.arange(T)
    valid = mask.astype(jnp.float32)

    # Distance to nearest valid timepoint on the left (cumulative min trick)
    # Use scan to find left/right nearest valid indices
    valid_idx = jnp.where(mask, t, -1)

    # Forward fill: nearest valid index to the left
    def _forward_fill(carry, x):
        last_valid = carry
        last_valid = jnp.where(x >= 0, x, last_valid)
        return last_valid, last_valid

    _, left_idx = jax.lax.scan(_forward_fill, jnp.int32(-1), valid_idx)

    # Backward fill: nearest valid index to the right
    def _backward_fill(carry, x):
        next_valid = carry
        next_valid = jnp.where(x >= 0, x, next_valid)
        return next_valid, next_valid

    _, right_idx = jax.lax.scan(
        _backward_fill, jnp.int32(T), jnp.flip(valid_idx)
    )
    right_idx = jnp.flip(right_idx)

    # Compute weights (inverse distance)
    dist_left = (t - left_idx).astype(jnp.float32)
    dist_right = (right_idx - t).astype(jnp.float32)
    total_dist = dist_left + dist_right
    total_dist = jnp.where(total_dist == 0, 1.0, total_dist)

    w_left = 1.0 - dist_left / total_dist
    w_right = 1.0 - dist_right / total_dist

    # Handle boundary cases (no valid left or right)
    has_left = left_idx >= 0
    has_right = right_idx < T
    has_both = has_left & has_right

    # Clamp indices to valid range for indexing
    left_safe = jnp.clip(left_idx, 0, T - 1)
    right_safe = jnp.clip(right_idx, 0, T - 1)

    left_vals = data[..., left_safe]
    right_vals = data[..., right_safe]

    interpolated = jnp.where(
        has_both,
        w_left * left_vals + w_right * right_vals,
        jnp.where(has_left, left_vals, jnp.where(has_right, right_vals, 0.0)),
    )

    # Keep original data where mask is True
    return jnp.where(mask, data, interpolated)


def spectral_interpolate(data, mask, oversampling_frequency=8,
                         maximum_frequency=1.0, sampling_period=1.0):
    """Spectral interpolation via basis function projection.

    Fits a truncated Fourier basis to the valid timepoints and reconstructs
    the full time series including censored frames.

    Args:
        data: (..., T) time series.
        mask: (T,) boolean mask.  True = valid, False = censored.
        oversampling_frequency: Oversampling factor for the basis.
        maximum_frequency: Maximum frequency in the basis (relative to Nyquist).
        sampling_period: Sampling period (TR) in seconds.

    Returns:
        (..., T) spectrally interpolated time series.
    """
    T = data.shape[-1]
    t = jnp.arange(T, dtype=jnp.float32)

    # Construct truncated Fourier basis
    nyquist = 0.5 / sampling_period
    max_freq = maximum_frequency * nyquist
    n_basis = int(2 * max_freq * T * sampling_period * oversampling_frequency) + 1
    n_basis = max(n_basis, 3)  # at least DC + one sine/cosine pair

    freqs = jnp.arange(1, (n_basis + 1) // 2) / (T * sampling_period)
    # Basis: [1, cos(2*pi*f1*t), sin(2*pi*f1*t), cos(2*pi*f2*t), ...]
    basis = [jnp.ones((T,))]
    for f in freqs:
        basis.append(jnp.cos(2 * jnp.pi * f * t * sampling_period))
        basis.append(jnp.sin(2 * jnp.pi * f * t * sampling_period))
    B = jnp.stack(basis, axis=-1)  # (T, n_basis)

    # Mask the basis and data: only fit to valid timepoints
    mask_float = mask.astype(jnp.float32)
    B_masked = B * mask_float[:, None]  # (T, n_basis) with zeros at censored
    data_masked = data * mask_float  # (..., T)

    # Solve least squares: B_masked^T B_masked @ coeffs = B_masked^T @ data_masked^T
    BtB = B_masked.T @ B_masked  # (n_basis, n_basis)
    # Regularise for numerical stability
    BtB = BtB + 1e-8 * jnp.eye(BtB.shape[0])

    # (..., T) @ (T, n_basis) -> (..., n_basis)
    Btd = jnp.einsum('...t,tk->...k', data_masked, B_masked)

    # Solve for coefficients: BtB @ coeffs = Btd
    # Use lstsq for robustness (handles rank-deficient cases)
    # Reshape Btd to (n_basis, M) where M = product of batch dims
    batch_shape = Btd.shape[:-1]
    n_b = Btd.shape[-1]
    if len(batch_shape) == 0:
        # 1D data: Btd is (n_basis,)
        coeffs, _, _, _ = jnp.linalg.lstsq(BtB, Btd)
    else:
        Btd_flat = Btd.reshape(-1, n_b)  # (M, n_basis)
        coeffs_flat, _, _, _ = jnp.linalg.lstsq(BtB, Btd_flat.T)  # (n_basis, M)
        coeffs = coeffs_flat.T.reshape(*batch_shape, n_b)

    # Reconstruct: (..., n_basis) @ (n_basis, T) -> (..., T)
    reconstructed = jnp.einsum('...k,tk->...t', coeffs, B)

    # Keep original data where valid
    return jnp.where(mask, data, reconstructed)


def hybrid_interpolate(data, mask, max_consecutive_linear=3,
                       oversampling_frequency=8, maximum_frequency=1.0,
                       sampling_period=1.0):
    """Hybrid linear + spectral interpolation.

    Uses linear interpolation for short gaps (<= max_consecutive_linear)
    and spectral interpolation for longer gaps.

    Args:
        data: (..., T) time series.
        mask: (T,) boolean mask.  True = valid, False = censored.
        max_consecutive_linear: Maximum consecutive censored frames to
            handle with linear interpolation.
        oversampling_frequency: Passed to spectral_interpolate.
        maximum_frequency: Passed to spectral_interpolate.
        sampling_period: Passed to spectral_interpolate.

    Returns:
        (..., T) interpolated time series.
    """
    T = data.shape[-1]
    censored = ~mask

    # Compute consecutive censored run lengths at each position
    def _run_length(carry, x):
        count = jnp.where(x, carry + 1, 0)
        return count, count
    _, forward_runs = jax.lax.scan(_run_length, jnp.int32(0), censored)

    # Also backward to get total run length at each censored position
    _, backward_runs = jax.lax.scan(
        _run_length, jnp.int32(0), jnp.flip(censored)
    )
    backward_runs = jnp.flip(backward_runs)

    # Total run length = forward_run + backward_run - 1 at each censored point
    total_run = forward_runs + backward_runs - 1
    total_run = jnp.where(censored, total_run, 0)

    # Short gaps: use linear; long gaps: use spectral
    use_linear = censored & (total_run <= max_consecutive_linear)
    use_spectral = censored & (total_run > max_consecutive_linear)

    # Linear interpolation for all censored
    linear_result = linear_interpolate(data, mask)

    # Spectral interpolation for all censored
    spectral_result = spectral_interpolate(
        data, mask,
        oversampling_frequency=oversampling_frequency,
        maximum_frequency=maximum_frequency,
        sampling_period=sampling_period,
    )

    # Combine: original where valid, linear for short gaps, spectral for long
    result = jnp.where(mask, data,
                       jnp.where(use_linear, linear_result, spectral_result))
    return result
