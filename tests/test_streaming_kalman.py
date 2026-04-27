"""Red-green TDD for the Phase 4 streaming Kalman/EKF Bayesian Variant G.

Pre-implementation hypotheses (RED):

1. After T TRs of streaming updates, the posterior β should match what the
   batch `_variant_g_forward` would produce on the same X[:T], Y[:T].
2. Posterior covariance should decrease (monotonically in trace) as more
   TRs arrive — adding evidence sharpens the posterior.
3. The streaming filter should be JIT-compatible and produce stable output
   under different update orderings of the same data.

Once the module is built, these tests pin the recursive math against the
batch closed-form. Implementation lives in jaxoccoli/streaming_kalman.py
(planned).
"""
from __future__ import annotations

import numpy as np
import pytest

# This import will fail until streaming_kalman lands — that's the RED:
pytest.importorskip("jaxoccoli.streaming_kalman",
                     reason="awaiting Phase 4 implementation")

from jaxoccoli.streaming_kalman import (
    StreamingKalmanState,
    init_streaming_kalman,
    streaming_kalman_run,
    streaming_kalman_update,
)


def _build_synth(T=80, P=4, V=20, sigma=0.3, rho=0.3, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(T, P)).astype(np.float32)
    true_beta = rng.normal(size=(V, P)).astype(np.float32)
    eps = np.zeros((V, T), dtype=np.float32)
    for t in range(1, T):
        eps[:, t] = rho * eps[:, t - 1] + sigma * rng.normal(size=V).astype(np.float32)
    Y = true_beta @ X.T + eps
    return X, Y, true_beta


def test_streaming_matches_batch_ols_after_full_run():
    """v1 streaming Kalman uses white noise (no AR(1)). Compare against
    batch OLS — they should agree at convergence.

    AR(1)-prewhitened streaming Kalman is v2; we'll add a separate test
    against `_variant_g_forward` once that variant lands.
    """
    X, Y, _ = _build_synth(T=80, P=4, V=10, rho=0.0)   # white noise synth
    T, P = X.shape
    V = Y.shape[0]

    # Streaming
    state = init_streaming_kalman(P=P, V=V, prior_var_scale=1e6)
    state = streaming_kalman_run(state, X, Y)
    beta_stream = np.asarray(state.beta_mean)               # (V, P)

    # Batch OLS
    XtX_inv = np.linalg.inv(X.T @ X + 1e-6 * np.eye(P))
    beta_ols = (Y @ X) @ XtX_inv.T

    diff = np.abs(beta_stream - beta_ols).mean()
    assert diff < 0.05, (
        f"streaming β posterior diverged from batch OLS ({diff:.4f})"
    )


def test_posterior_covariance_shrinks_as_data_arrives():
    """Trace of posterior covariance should be monotonically decreasing
    (or non-increasing) as more TRs arrive."""
    X, Y, _ = _build_synth(T=60, P=4, V=5)
    state = init_streaming_kalman(P=X.shape[1], V=Y.shape[0])
    traces = []
    for t in range(X.shape[0]):
        state = streaming_kalman_update(state, X[t], Y[:, t])
        traces.append(float(np.asarray(state.beta_cov).trace(axis1=-2, axis2=-1).mean()))
    # Allow occasional small bumps (numerical) but overall trend must be down
    assert traces[-1] < 0.5 * traces[10], (
        f"posterior covariance not shrinking: trace[10]={traces[10]:.3e}, "
        f"trace[-1]={traces[-1]:.3e}"
    )


def test_streaming_kalman_state_is_jit_compatible():
    """The state PyTree should round-trip cleanly through jax.jit."""
    import jax
    state = init_streaming_kalman(P=3, V=8)

    @jax.jit
    def _identity(s):
        return s
    s2 = _identity(state)
    # Same shapes, same field names
    assert s2.beta_mean.shape == state.beta_mean.shape
    assert s2.beta_cov.shape == state.beta_cov.shape


def test_v2_ar1_streaming_matches_batch_variant_g():
    """v2: AR(1)-prewhitened streaming Kalman.

    Synthetic AR(1) noise. Streaming filter with AR(1) update should
    converge to the batch _variant_g_forward result (which itself does
    AR(1) prewhitening + Bayesian conjugate).
    """
    import jax.numpy as jnp
    pytest.importorskip("jaxoccoli.streaming_kalman",
                         reason="streaming_kalman v2 must be implemented")
    from jaxoccoli.streaming_kalman import (
        init_streaming_kalman_ar1,
        streaming_kalman_ar1_run,
    )
    from jaxoccoli.realtime import _variant_g_forward_jit

    X, Y, _ = _build_synth(T=120, P=4, V=8, sigma=0.3, rho=0.4, seed=42)
    T, P = X.shape
    V = Y.shape[0]

    state = init_streaming_kalman_ar1(P=P, V=V)
    state = streaming_kalman_ar1_run(state, X, Y)
    beta_stream = np.asarray(state.beta_mean)

    n_pad = T + 8
    X_pad = np.zeros((n_pad, P), dtype=np.float32)
    X_pad[:T] = X
    Y_pad = np.zeros((V, n_pad), dtype=np.float32)
    Y_pad[:, :T] = Y
    beta_batch, _ = _variant_g_forward_jit(
        jnp.asarray(X_pad), jnp.asarray(Y_pad),
        jnp.asarray(T, dtype=jnp.int32),
    )
    beta_batch = np.asarray(beta_batch)

    diff = np.abs(beta_stream - beta_batch).mean()
    assert diff < 0.1, (
        f"AR(1) streaming β diverged from batch Variant G ({diff:.4f})"
    )
