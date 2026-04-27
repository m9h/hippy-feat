"""Phase 4 streaming Bayesian Variant G — recursive Kalman filter.

The batch `_variant_g_forward` (in `realtime.py`) recomputes the full
conjugate posterior every TR — O(T²) per TR because the design matrix
grows. The recursive Kalman version maintains running `(β_mean, β_cov,
σ²_post)` per voxel and updates each in O(P²) per TR, independent of T.

v1 (this module): white-noise Kalman (no AR(1)). At convergence, β_mean
matches batch OLS within numerical tolerance. AR(1)-prewhitened version
is v2.

State design:
    StreamingKalmanState (NamedTuple):
        beta_mean: (V, P) running posterior mean
        beta_cov:  (V, P, P) running posterior covariance
        a_post:    (V,) InverseGamma shape parameter for σ²
        b_post:    (V,) InverseGamma scale parameter for σ²
        n_obs:     scalar — TRs ingested so far

The factory is `init_streaming_kalman(P, V)` which returns the initial
state with weak priors. The update is `streaming_kalman_update(state,
x_row, y_obs)` which is a pure function — no in-place mutation, returns
a new state.

Both factory and update are JIT-compatible. State is a JAX PyTree.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class StreamingKalmanState(NamedTuple):
    """Running Kalman state for Bayesian per-voxel β posterior."""
    beta_mean: jnp.ndarray   # (V, P)
    beta_cov:  jnp.ndarray   # (V, P, P)
    a_post:    jnp.ndarray   # (V,) InverseGamma shape
    b_post:    jnp.ndarray   # (V,) InverseGamma scale
    n_obs:     jnp.ndarray   # scalar int32 — TR count


def init_streaming_kalman(P: int, V: int,
                            prior_var_scale: float = 1e3,
                            a0: float = 0.01, b0: float = 0.01
                            ) -> StreamingKalmanState:
    """Initialize the running posterior with a weak prior.

    Args:
        P: number of regression coefficients (design columns).
        V: number of voxels.
        prior_var_scale: diagonal scale on the prior covariance — larger
            means weaker prior.
        a0, b0: InverseGamma prior on σ². Default (0.01, 0.01) is weakly
            informative (the standard NIG conjugate weak prior).
    """
    beta_mean = jnp.zeros((V, P), dtype=jnp.float32)
    beta_cov = jnp.broadcast_to(
        prior_var_scale * jnp.eye(P, dtype=jnp.float32),
        (V, P, P),
    )
    return StreamingKalmanState(
        beta_mean=beta_mean,
        beta_cov=beta_cov,
        a_post=jnp.full((V,), a0, dtype=jnp.float32),
        b_post=jnp.full((V,), b0, dtype=jnp.float32),
        n_obs=jnp.asarray(0, dtype=jnp.int32),
    )


@jax.jit
def streaming_kalman_update(state: StreamingKalmanState,
                              x_row: jnp.ndarray,
                              y_obs: jnp.ndarray) -> StreamingKalmanState:
    """Process a new TR: update (β_mean, β_cov, σ²_post) per voxel.

    Uses **Recursive Least Squares (RLS)** form for the β posterior — σ²
    does NOT enter the Kalman gain, so the β posterior converges exactly
    to OLS as more data arrives, regardless of σ² estimation. σ² is
    tracked separately via a Normal-InverseGamma conjugate update for
    posterior-variance reporting.

    Math (RLS form, equivalent to Bayesian linear regression with weak
    Gaussian prior on β and conjugate NIG on σ²):

        S         = 1 + x^T Σ x                                 (V,)
        K         = Σ x / S                                     (V, P)
        innov     = y - x^T μ                                   (V,)
        μ_new     = μ + K · innov                               (V, P)
        Σ_new     = Σ - K (x^T Σ)                               (V, P, P)
        a_post   += 0.5
        b_post   += 0.5 · innov² / S          (Schur complement of σ² posterior)

    Note: the `1 +` in S replaces a `σ̂²` term that would appear if we
    were using a different parameterization. With this normalization,
    the β update is OLS-recursive: at convergence, μ_T = (X^T X)^{-1} X^T y.
    """
    Sx = state.beta_cov @ x_row                                # (V, P)
    xSx = jnp.einsum("p,vp->v", x_row, Sx)                     # (V,)
    S = 1.0 + xSx                                              # (V,)
    K = Sx / S[:, None]                                        # (V, P)

    innovation = y_obs - jnp.einsum("p,vp->v", x_row, state.beta_mean)  # (V,)
    beta_mean_new = state.beta_mean + K * innovation[:, None]  # (V, P)

    # Σ_new = Σ - K (x^T Σ) — rank-1 update
    KSx_outer = K[:, :, None] * Sx[:, None, :]                 # (V, P, P)
    beta_cov_new = state.beta_cov - KSx_outer

    # σ² update via Schur-complement contribution to RSS
    rss_inc = innovation ** 2 / S                              # (V,)
    a_new = state.a_post + 0.5
    b_new = state.b_post + 0.5 * rss_inc

    return StreamingKalmanState(
        beta_mean=beta_mean_new,
        beta_cov=beta_cov_new,
        a_post=a_new,
        b_post=b_new,
        n_obs=state.n_obs + 1,
    )


def streaming_kalman_run(initial_state: StreamingKalmanState,
                          X: np.ndarray, Y: np.ndarray
                          ) -> StreamingKalmanState:
    """Convenience: feed an entire (T, P) design / (V, T) observation
    sequence through the streaming filter, return final state.

    For RT use, the rt-cloud bridge calls `streaming_kalman_update`
    once per arriving DICOM and keeps the state externally; this helper
    is for offline benchmarking + tests.
    """
    state = initial_state
    T = X.shape[0]
    for t in range(T):
        state = streaming_kalman_update(state, jnp.asarray(X[t]),
                                         jnp.asarray(Y[:, t]))
    return state


# ---------------------------------------------------------------------------
# v2 — AR(1)-prewhitened streaming Kalman
# ---------------------------------------------------------------------------

class StreamingKalmanAR1State(NamedTuple):
    """Running state for AR(1)-prewhitened streaming Bayesian Variant G.

    Adds two fields beyond the white-noise version:
        prev_y:  (V,) last observation per voxel — used for prewhitening
        prev_x:  (P,) last design row — used for prewhitening
        rho_acc_num: (V,) running numerator of ρ estimator (Σ resid[t-1]·resid[t])
        rho_acc_den: (V,) running denominator (Σ resid[t-1]²)
    """
    beta_mean: jnp.ndarray
    beta_cov:  jnp.ndarray
    a_post:    jnp.ndarray
    b_post:    jnp.ndarray
    n_obs:     jnp.ndarray
    prev_y:    jnp.ndarray   # (V,)
    prev_x:    jnp.ndarray   # (P,)
    rho_num:   jnp.ndarray   # (V,)
    rho_den:   jnp.ndarray   # (V,)


def init_streaming_kalman_ar1(P: int, V: int,
                                prior_var_scale: float = 1e3,
                                a0: float = 0.01, b0: float = 0.01
                                ) -> StreamingKalmanAR1State:
    base = init_streaming_kalman(P, V, prior_var_scale, a0, b0)
    return StreamingKalmanAR1State(
        beta_mean=base.beta_mean,
        beta_cov=base.beta_cov,
        a_post=base.a_post,
        b_post=base.b_post,
        n_obs=base.n_obs,
        prev_y=jnp.zeros((V,), dtype=jnp.float32),
        prev_x=jnp.zeros((P,), dtype=jnp.float32),
        rho_num=jnp.zeros((V,), dtype=jnp.float32),
        rho_den=jnp.full((V,), 1e-6, dtype=jnp.float32),
    )


@jax.jit
def streaming_kalman_ar1_update(state: StreamingKalmanAR1State,
                                  x_row: jnp.ndarray,
                                  y_obs: jnp.ndarray
                                  ) -> StreamingKalmanAR1State:
    """AR(1)-prewhitened recursive Bayesian update.

    Strategy: maintain running ρ̂_v per voxel from accumulated lag-1
    autocorrelation of OLS-residuals; prewhiten the new (x, y) pair with
    the previous step's estimates; do a standard RLS update on the
    prewhitened observation.

        rho_v        = rho_num_v / rho_den_v   (clipped to [-0.99, 0.99])
        y_pw_v       = y_obs_v - rho_v · prev_y_v
        x_pw         = x_row   - rho_v · prev_x   (per-voxel — broadcast over V)
        ... then standard RLS update on (x_pw, y_pw)
        ... track residuals to update rho_num / rho_den

    On the first TR (n_obs=0), prev_y / prev_x are zero by construction
    so prewhitening is a no-op and the step degrades to v1's white-noise
    update — correct behaviour.
    """
    rho = jnp.clip(state.rho_num / jnp.maximum(state.rho_den, 1e-10),
                   -0.99, 0.99)                                       # (V,)
    # Prewhiten — note x_pw is per-voxel because rho varies; we form the
    # effective (V, P) prewhitened design row.
    x_pw = x_row[None, :] - rho[:, None] * state.prev_x[None, :]      # (V, P)
    y_pw = y_obs - rho * state.prev_y                                  # (V,)

    # Standard RLS update with per-voxel x_pw row
    Sx = jnp.einsum("vpq,vq->vp", state.beta_cov, x_pw)               # (V, P)
    xSx = jnp.einsum("vp,vp->v", x_pw, Sx)                            # (V,)
    S = 1.0 + xSx                                                      # (V,)
    K = Sx / S[:, None]                                                # (V, P)

    innovation = y_pw - jnp.einsum("vp,vp->v", x_pw, state.beta_mean)  # (V,)
    beta_mean_new = state.beta_mean + K * innovation[:, None]
    KSx_outer = K[:, :, None] * Sx[:, None, :]                         # (V, P, P)
    beta_cov_new = state.beta_cov - KSx_outer

    # σ² update
    rss_inc = innovation ** 2 / S
    a_new = state.a_post + 0.5
    b_new = state.b_post + 0.5 * rss_inc

    # Update ρ accumulators using the OLS-residual at the unprewhitened
    # (x_row, y_obs) pair — i.e., r_t = y_obs - x_row · beta_mean (post-update)
    resid_t = y_obs - jnp.einsum("p,vp->v", x_row, beta_mean_new)
    # ρ accumulator update; uses lag-1 product with the previous residual.
    # The "previous residual" is implicit: we track prev_y / prev_x and
    # reconstruct prev_resid = prev_y - prev_x · beta_mean (using current β).
    prev_resid = state.prev_y - jnp.einsum("p,vp->v", state.prev_x, beta_mean_new)
    rho_num_new = state.rho_num + prev_resid * resid_t
    rho_den_new = state.rho_den + prev_resid ** 2

    return StreamingKalmanAR1State(
        beta_mean=beta_mean_new,
        beta_cov=beta_cov_new,
        a_post=a_new,
        b_post=b_new,
        n_obs=state.n_obs + 1,
        prev_y=y_obs,
        prev_x=x_row,
        rho_num=rho_num_new,
        rho_den=rho_den_new,
    )


def streaming_kalman_ar1_run(state: StreamingKalmanAR1State,
                              X: np.ndarray, Y: np.ndarray
                              ) -> StreamingKalmanAR1State:
    T = X.shape[0]
    for t in range(T):
        state = streaming_kalman_ar1_update(state, jnp.asarray(X[t]),
                                              jnp.asarray(Y[:, t]))
    return state


__all__ = [
    "StreamingKalmanState",
    "StreamingKalmanAR1State",
    "init_streaming_kalman",
    "init_streaming_kalman_ar1",
    "streaming_kalman_update",
    "streaming_kalman_ar1_update",
    "streaming_kalman_run",
    "streaming_kalman_ar1_run",
]
