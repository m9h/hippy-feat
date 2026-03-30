"""Learnable components via vbjax-style factory functions.

Replaces hypercoil nn/atlas.py, nn/cov.py, nn/freqfilter.py, and
init/mapparam.py using the make_*() -> (params, forward_fn) pattern.
No Equinox, no Flax — plain JAX + NamedTuples.
"""

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Tuple

from .covariance import cov, corr


# ---------------------------------------------------------------------------
# Fisher-Rao natural gradient (adapted from hgx/_info_geometry.py)
# ---------------------------------------------------------------------------

_FR_EPS = 1e-12


def fisher_rao_metric(probs):
    """Fisher information matrix for categorical distributions.

    For categorical probabilities p_1, ..., p_K, the Fisher information
    is diagonal with g_{kk} = 1/p_k.

    Args:
        probs: (..., K) probability vectors.

    Returns:
        (..., K, K) diagonal metric tensors.
    """
    safe_p = jnp.clip(probs, _FR_EPS, None)
    return jnp.apply_along_axis(jnp.diag, -1, 1.0 / safe_p)


def natural_gradient(grad, probs):
    """Compute natural gradient: G^{-1} @ grad.

    For categorical Fisher metric F = diag(1/p), the inverse is diag(p),
    so the natural gradient is simply p * grad (element-wise).

    Args:
        grad: (..., K) Euclidean gradient.
        probs: (..., K) probability vectors.

    Returns:
        (..., K) natural gradient.
    """
    safe_p = jnp.clip(probs, _FR_EPS, None)
    return safe_p * grad


def natural_gradient_step(loss_fn, p, lr=0.01):
    """Single natural gradient descent step on the probability simplex.

    Args:
        loss_fn: Scalar loss function mapping probabilities to scalar.
        p: (..., K) current probability distributions.
        lr: Learning rate.

    Returns:
        (..., K) updated distributions (projected back to simplex).
    """
    euclidean_grad = jax.grad(loss_fn)(p)
    nat_grad = natural_gradient(euclidean_grad, p)
    log_p = jnp.log(jnp.clip(p, _FR_EPS, None))
    log_p_new = log_p - lr * nat_grad
    return jax.nn.softmax(log_p_new, axis=-1)


# ---------------------------------------------------------------------------
# Parameter constraint factories
# ---------------------------------------------------------------------------

class ConstraintFns(NamedTuple):
    """Pair of functions for constrained optimisation."""
    project: Callable    # unconstrained -> constrained domain
    unproject: Callable  # constrained -> unconstrained (approximate inverse)


def make_simplex_constraint(temperature=1.0):
    """Probability simplex constraint via softmax.

    Args:
        temperature: Softmax temperature (lower = sharper).

    Returns:
        ConstraintFns(project=softmax, unproject=log).
    """
    def project(x):
        return jax.nn.softmax(x / temperature, axis=-1)

    def unproject(p):
        return jnp.log(jnp.clip(p, 1e-8, 1.0)) * temperature

    return ConstraintFns(project=project, unproject=unproject)


def make_spd_constraint(eps=1e-6):
    """SPD cone constraint via eigenvalue clamping.

    Args:
        eps: Minimum eigenvalue.

    Returns:
        ConstraintFns(project=clamp_eigenvalues, unproject=identity).
    """
    def project(X):
        X = (X + jnp.swapaxes(X, -2, -1)) / 2
        eigvals, eigvecs = jnp.linalg.eigh(X)
        eigvals = jnp.maximum(eigvals, eps)
        return (eigvecs * eigvals[..., None, :]) @ eigvecs.swapaxes(-2, -1)

    def unproject(X):
        return X

    return ConstraintFns(project=project, unproject=unproject)


def make_orthogonal_constraint():
    """Orthogonality constraint via QR decomposition.

    Returns:
        ConstraintFns(project=QR_extract_Q, unproject=identity).
    """
    def project(X):
        Q, _ = jnp.linalg.qr(X)
        return Q

    def unproject(Q):
        return Q

    return ConstraintFns(project=project, unproject=unproject)


# ---------------------------------------------------------------------------
# Differentiable atlas (voxel -> parcel mapping)
# ---------------------------------------------------------------------------

class AtlasParams(NamedTuple):
    """Parameters for differentiable atlas mapping."""
    weight: jnp.ndarray   # (n_parcels, n_voxels) raw logits


def make_atlas_linear(n_voxels, n_parcels, *, key, normalisation='mean'):
    """Factory for differentiable voxel-to-parcel mapping.

    Args:
        n_voxels: Number of input voxels/vertices.
        n_parcels: Number of output parcels/labels.
        key: JAX PRNG key.
        normalisation: 'mean' (weighted mean) or 'sum' (weighted sum).

    Returns:
        params: AtlasParams with Dirichlet-initialised weights.
        forward_fn: (params, data) -> parcellated_data.
            data: (n_voxels, T) or (batch, n_voxels, T)
            returns: (n_parcels, T) or (batch, n_parcels, T)
    """
    raw = jax.random.normal(key, (n_parcels, n_voxels)) * 0.1
    params = AtlasParams(weight=raw)

    def forward(params, data):
        w = jax.nn.softmax(params.weight, axis=-1)  # (parcels, voxels)
        if normalisation == 'mean':
            return jnp.einsum('pv,...vt->...pt', w, data)
        elif normalisation == 'sum':
            w_unnorm = jnp.exp(params.weight)
            return jnp.einsum('pv,...vt->...pt', w_unnorm, data)
        else:
            return jnp.einsum('pv,...vt->...pt', w, data)

    return params, forward


def make_atlas_linear_uncertain(n_voxels, n_parcels, *, key, normalisation='mean'):
    """Factory for variance-aware voxel-to-parcel mapping.

    Propagates both mean and variance through the linear parcellation.
    Addresses the Rissman/Mumford beta series variance gap.

    Args:
        n_voxels: Number of input voxels.
        n_parcels: Number of output parcels.
        key: JAX PRNG key.
        normalisation: 'mean' or 'sum'.

    Returns:
        params: AtlasParams.
        forward_fn: (params, beta_mean, beta_var) -> (parc_mean, parc_var).
            beta_mean: (..., n_voxels, T) posterior means.
            beta_var: (..., n_voxels, T) posterior variances (diagonal).
            parc_mean: (..., n_parcels, T) parcellated means.
            parc_var: (..., n_parcels, T) parcellated variances.
    """
    raw = jax.random.normal(key, (n_parcels, n_voxels)) * 0.1
    params = AtlasParams(weight=raw)

    def forward(params, beta_mean, beta_var):
        w = jax.nn.softmax(params.weight, axis=-1)  # (P, V)
        # Mean: E[W @ beta] = W @ E[beta]
        parc_mean = jnp.einsum('pv,...vt->...pt', w, beta_mean)
        # Variance: Var[W @ beta] = W^2 @ Var[beta]  (diagonal case)
        w2 = w ** 2
        parc_var = jnp.einsum('pv,...vt->...pt', w2, beta_var)
        return parc_mean, parc_var

    return params, forward


def make_atlas_natural_grad(n_voxels, n_parcels, *, key, lr=0.01):
    """Factory for atlas with Fisher-Rao natural gradient updates.

    Uses the natural gradient on the probability simplex for geometry-aware
    optimization of parcellation weights.  Faster convergence and
    better-conditioned updates than Euclidean gradients.

    Adapted from hgx/_info_geometry.py.

    Args:
        n_voxels: Number of input voxels.
        n_parcels: Number of output parcels.
        key: JAX PRNG key.
        lr: Natural gradient step size.

    Returns:
        params: AtlasParams.
        forward_fn: (params, data) -> parcellated_data.
        update_fn: (params, loss_fn) -> updated params.
            Performs one natural gradient step on each parcel's weight row.
    """
    raw = jax.random.normal(key, (n_parcels, n_voxels)) * 0.1
    params = AtlasParams(weight=raw)

    def forward(params, data):
        w = jax.nn.softmax(params.weight, axis=-1)
        return jnp.einsum('pv,...vt->...pt', w, data)

    def update(params, loss_fn):
        """One natural gradient step on the atlas weights.

        Args:
            params: Current AtlasParams.
            loss_fn: Callable(AtlasParams) -> scalar loss.

        Returns:
            Updated AtlasParams after one natural gradient step.
        """
        euclidean_grad = jax.grad(loss_fn)(params)
        # Softmax weights live on the simplex per parcel row
        w = jax.nn.softmax(params.weight, axis=-1)  # (P, V)
        # Natural gradient: G^{-1} @ grad = p * grad
        nat_grad_w = natural_gradient(euclidean_grad.weight, w)
        # Update in log-space and project back
        log_w = jnp.log(jnp.clip(w, _FR_EPS, None))
        log_w_new = log_w - lr * nat_grad_w
        new_weight = log_w_new  # softmax will normalise in forward
        return AtlasParams(weight=new_weight)

    return params, forward, update


# ---------------------------------------------------------------------------
# Learnable covariance estimation
# ---------------------------------------------------------------------------

class CovParams(NamedTuple):
    """Parameters for learnable covariance estimation."""
    weight: jnp.ndarray   # (T,) observation weights (diagonal) or (T,T) full


def make_learnable_cov(dim, time_dim, *, key, estimator='corr',
                       weight_type='diagonal', l2=0.0):
    """Factory for learnable covariance/correlation estimation.

    Args:
        dim: Number of variables (channels/parcels).
        time_dim: Number of time points.
        key: JAX PRNG key.
        estimator: 'cov' or 'corr'.
        weight_type: 'diagonal' (per-observation weights) or 'none'.
        l2: L2 regularisation.

    Returns:
        params: CovParams.
        forward_fn: (params, data) -> covariance_or_correlation_matrix.
            data: (dim, time_dim) -> returns (dim, dim).
    """
    if weight_type == 'diagonal':
        raw_weight = jnp.zeros(time_dim)  # initialise at uniform
    else:
        raw_weight = jnp.zeros(1)  # placeholder
    params = CovParams(weight=raw_weight)

    est_fn = corr if estimator == 'corr' else cov

    def forward(params, data):
        if weight_type == 'diagonal':
            w = jax.nn.softmax(params.weight) * time_dim  # normalised, sum=T
            return est_fn(data, weight=w, l2=l2)
        else:
            return est_fn(data, l2=l2)

    return params, forward


# ---------------------------------------------------------------------------
# Learnable frequency-domain filter
# ---------------------------------------------------------------------------

class FreqFilterParams(NamedTuple):
    """Parameters for learnable frequency-domain filter."""
    transfer_fn: jnp.ndarray   # (n_filters, n_freq) transfer function


def make_freq_filter(n_freq, *, key, n_filters=1, init_fn=None):
    """Factory for learnable frequency-domain filter bank.

    Args:
        n_freq: Number of frequency bins (typically T//2+1 for rfft).
        key: JAX PRNG key.
        n_filters: Number of parallel filters.
        init_fn: Optional function (n_freq,) -> (n_freq,) for initialisation.
            If None, initialises as all-pass (ones).

    Returns:
        params: FreqFilterParams.
        forward_fn: (params, data) -> filtered_data.
            data: (..., T) -> returns (..., n_filters, T) if n_filters > 1
            else (..., T).
    """
    if init_fn is not None:
        tf = jnp.stack([init_fn(n_freq) for _ in range(n_filters)])
    else:
        tf = jnp.ones((n_filters, n_freq))
    params = FreqFilterParams(transfer_fn=tf)

    def forward(params, data):
        T = data.shape[-1]
        Xf = jnp.fft.rfft(data, axis=-1)  # (..., n_freq)
        # Apply each filter: (..., n_freq) * (n_filters, n_freq)
        filtered_f = Xf[..., None, :] * params.transfer_fn  # (..., n_filters, n_freq)
        filtered = jnp.fft.irfft(filtered_f, n=T, axis=-1)  # (..., n_filters, T)
        if n_filters == 1:
            return filtered[..., 0, :]
        return filtered

    return params, forward


# ---------------------------------------------------------------------------
# Filter initialisation helpers
# ---------------------------------------------------------------------------

def init_ideal_spectrum(n_freq, low=None, high=None, fs=1.0):
    """Ideal (brick-wall) bandpass filter transfer function.

    Args:
        n_freq: Number of frequency bins.
        low: Low cutoff frequency in Hz (None = no highpass).
        high: High cutoff frequency in Hz (None = no lowpass).
        fs: Sampling frequency.

    Returns:
        (n_freq,) transfer function.
    """
    freqs = jnp.linspace(0, fs / 2, n_freq)
    tf = jnp.ones(n_freq)
    if low is not None:
        tf = jnp.where(freqs >= low, tf, 0.0)
    if high is not None:
        tf = jnp.where(freqs <= high, tf, 0.0)
    return tf


def init_butterworth_spectrum(n_freq, order, low=None, high=None, fs=1.0):
    """Butterworth filter magnitude response.

    Args:
        n_freq: Number of frequency bins.
        order: Filter order.
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.
        fs: Sampling frequency.

    Returns:
        (n_freq,) magnitude response.
    """
    freqs = jnp.linspace(0, fs / 2, n_freq)
    tf = jnp.ones(n_freq)

    if high is not None:
        # Lowpass component
        ratio = freqs / high
        tf = tf * 1.0 / jnp.sqrt(1.0 + ratio ** (2 * order))

    if low is not None:
        # Highpass component
        ratio = jnp.where(freqs > 0, low / freqs, jnp.inf)
        hp = 1.0 / jnp.sqrt(1.0 + ratio ** (2 * order))
        tf = tf * hp

    return tf
