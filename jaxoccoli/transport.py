"""Optimal transport for connectivity comparison.

Adapted from hgx/_ot.py.  Provides log-domain Sinkhorn, Wasserstein distance,
and Gromov-Wasserstein distance for comparing functional connectivity matrices
across subjects without parcellation alignment.

All pure JAX, fully differentiable, no external OT libraries required.
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Log-domain Sinkhorn
# ---------------------------------------------------------------------------

def sinkhorn(C, a=None, b=None, epsilon=0.1, max_iters=100):
    """Entropy-regularised optimal transport via log-domain Sinkhorn.

    Solves: min_{T>=0} <C, T> + epsilon * KL(T || a b^T)
            s.t. T @ 1 = a,  T^T @ 1 = b

    Args:
        C: (N, M) cost matrix.
        a: (N,) source marginal. If None, uses uniform.
        b: (M,) target marginal. If None, uses uniform.
        epsilon: Entropic regularisation strength.
        max_iters: Number of Sinkhorn iterations.

    Returns:
        T: (N, M) optimal transport plan.
    """
    N, M = C.shape
    if a is None:
        a = jnp.ones(N) / N
    if b is None:
        b = jnp.ones(M) / M

    log_a = jnp.log(jnp.maximum(a, 1e-30))
    log_b = jnp.log(jnp.maximum(b, 1e-30))
    log_K = -C / epsilon

    f = jnp.zeros(N)
    g = jnp.zeros(M)

    def _step(carry, _):
        f, g = carry
        f = log_a - jax.nn.logsumexp(log_K + g[None, :], axis=1)
        g = log_b - jax.nn.logsumexp(log_K + f[:, None], axis=0)
        return (f, g), None

    (f, g), _ = jax.lax.scan(_step, (f, g), None, length=max_iters)

    log_T = f[:, None] + log_K + g[None, :]
    return jnp.exp(log_T)


# ---------------------------------------------------------------------------
# Cost matrices
# ---------------------------------------------------------------------------

def euclidean_cost(X, Y=None):
    """Squared Euclidean pairwise cost matrix.

    Args:
        X: (N, D) point cloud.
        Y: (M, D) point cloud. If None, computes self-distance.

    Returns:
        (N, M) squared distance matrix.
    """
    if Y is None:
        Y = X
    diff = X[:, None, :] - Y[None, :, :]
    return jnp.sum(diff ** 2, axis=-1)


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------

def wasserstein_distance(X, Y, a=None, b=None, epsilon=0.1, max_iters=100):
    """Entropy-regularised Wasserstein distance between point clouds.

    Args:
        X: (N, D) source features.
        Y: (M, D) target features.
        a: (N,) source marginal. If None, uniform.
        b: (M,) target marginal. If None, uniform.
        epsilon: Regularisation strength.
        max_iters: Sinkhorn iterations.

    Returns:
        Scalar Wasserstein distance.
    """
    C = euclidean_cost(X, Y)
    T = sinkhorn(C, a, b, epsilon=epsilon, max_iters=max_iters)
    return jnp.sum(T * C)


def wasserstein_fc_distance(fc1, fc2, epsilon=0.1, max_iters=100):
    """Wasserstein distance between two FC matrices (treated as distributions).

    Vectorises the upper triangles and computes OT distance between them.

    Args:
        fc1: (N, N) functional connectivity matrix.
        fc2: (N, N) functional connectivity matrix.
        epsilon: Regularisation strength.
        max_iters: Sinkhorn iterations.

    Returns:
        Scalar Wasserstein distance between FC matrices.
    """
    # Extract upper triangles as feature vectors
    N = fc1.shape[0]
    rows, cols = jnp.triu_indices(N, k=1)
    v1 = fc1[rows, cols]
    v2 = fc2[rows, cols]
    # Treat as 1D distributions; compute 1-Wasserstein via sorted CDF
    # (closed form, exact for 1D)
    s1 = jnp.sort(v1)
    s2 = jnp.sort(v2)
    return jnp.mean(jnp.abs(s1 - s2))


# ---------------------------------------------------------------------------
# Gromov-Wasserstein distance
# ---------------------------------------------------------------------------

def gromov_wasserstein(D1, D2, a=None, b=None, epsilon=0.1,
                       max_iters=50, gw_iters=10):
    """Gromov-Wasserstein distance for comparing metric spaces.

    Compares two distance/similarity matrices without node correspondence.
    Useful for cross-subject FC comparison with different parcellations.

    Solves: min_T sum_{i,j,k,l} |D1[i,k] - D2[j,l]|^2 * T[i,j] * T[k,l]
            + epsilon * KL(T || a b^T)

    Uses the iterative linearisation approach.

    Args:
        D1: (N, N) distance/similarity matrix for space 1.
        D2: (M, M) distance/similarity matrix for space 2.
        a: (N,) source marginal. If None, uniform.
        b: (M,) target marginal. If None, uniform.
        epsilon: Entropic regularisation.
        max_iters: Sinkhorn iterations per GW step.
        gw_iters: Number of GW linearisation iterations.

    Returns:
        T: (N, M) transport plan.
        cost: Scalar Gromov-Wasserstein cost.
    """
    N = D1.shape[0]
    M = D2.shape[0]
    if a is None:
        a = jnp.ones(N) / N
    if b is None:
        b = jnp.ones(M) / M

    # Initialise transport plan as outer product of marginals
    T = a[:, None] * b[None, :]

    # Precompute: D1^2 @ a and D2^2 @ b for the quadratic cost
    D1_sq = D1 ** 2
    D2_sq = D2 ** 2

    def _gw_step(T, _):
        # Linearised cost: C_lin = D1^2 @ a @ 1^T + 1 @ b^T @ D2^2 - 2 * D1 @ T @ D2
        term1 = (D1_sq @ a)[:, None]  # (N, 1)
        term2 = (D2_sq @ b)[None, :]  # (1, M)
        term3 = 2.0 * D1 @ T @ D2.T  # (N, M)
        C_lin = term1 + term2 - term3
        # Solve entropic OT with linearised cost
        T_new = sinkhorn(C_lin, a, b, epsilon=epsilon, max_iters=max_iters)
        return T_new, None

    T, _ = jax.lax.scan(_gw_step, T, None, length=gw_iters)

    # Compute final GW cost
    cost = jnp.sum((D1_sq @ a)[:, None] * T) + jnp.sum(T * (D2_sq @ b)[None, :]) \
           - 2.0 * jnp.sum(D1 @ T @ D2.T * T)

    return T, cost


def gromov_wasserstein_fc(fc1, fc2, epsilon=0.1, max_iters=50, gw_iters=10):
    """Gromov-Wasserstein distance between two FC matrices.

    Compares the structural similarity of two connectivity matrices
    without requiring node correspondence. Ideal for comparing subjects
    with different parcellations or vertex counts.

    Args:
        fc1: (N, N) functional connectivity matrix.
        fc2: (M, M) functional connectivity matrix.
        epsilon: Regularisation strength.
        max_iters: Sinkhorn iterations.
        gw_iters: GW linearisation iterations.

    Returns:
        T: (N, M) transport plan (soft alignment).
        cost: Scalar GW distance.
    """
    # Convert correlation to distance
    D1 = 1.0 - jnp.abs(fc1)
    D2 = 1.0 - jnp.abs(fc2)
    return gromov_wasserstein(D1, D2, epsilon=epsilon,
                              max_iters=max_iters, gw_iters=gw_iters)
