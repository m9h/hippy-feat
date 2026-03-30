"""Matrix utilities and SPD manifold operations.

Ports essential algorithms from hypercoil functional/matrix.py and
functional/semidefinite.py in vbjax style (pure functions, JIT/vmap/grad
compatible, no Equinox).
"""

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

def symmetric(X, skew=False):
    """Impose symmetry (or skew-symmetry) on a matrix.

    Args:
        X: (..., N, N) matrix.
        skew: If True, return (X - X^T)/2 instead of (X + X^T)/2.

    Returns:
        (..., N, N) symmetric (or skew-symmetric) matrix.
    """
    Xt = jnp.swapaxes(X, -2, -1)
    if skew:
        return (X - Xt) / 2.0
    return (X + Xt) / 2.0


def ensure_spd(X, eps=1e-6):
    """Project matrix onto the SPD cone via eigenvalue clamping.

    Args:
        X: (..., N, N) matrix (should be approximately symmetric).
        eps: Minimum eigenvalue after clamping.

    Returns:
        (..., N, N) symmetric positive definite matrix.
    """
    X = symmetric(X)
    eigvals, eigvecs = jnp.linalg.eigh(X)
    eigvals = jnp.maximum(eigvals, eps)
    return (eigvecs * eigvals[..., None, :]) @ eigvecs.swapaxes(-2, -1)


def cholesky_invert(X):
    """Invert an SPD matrix via Cholesky decomposition.

    More numerically stable and faster than direct inversion for SPD matrices.

    Args:
        X: (..., N, N) symmetric positive definite matrix.

    Returns:
        (..., N, N) inverse of X.
    """
    L = jnp.linalg.cholesky(X)
    eye = jnp.eye(X.shape[-1])
    return jax.scipy.linalg.cho_solve((L, True), eye)


def sym2vec(S, offset=1):
    """Extract the upper triangle of a symmetric matrix as a vector.

    Args:
        S: (..., N, N) symmetric matrix.
        offset: Diagonal offset.  1 = strict upper triangle (default),
                0 = includes diagonal.

    Returns:
        (..., K) vector where K = N*(N-1)/2 (offset=1) or N*(N+1)/2 (offset=0).
    """
    n = S.shape[-1]
    rows, cols = jnp.triu_indices(n, k=offset)
    return S[..., rows, cols]


def vec2sym(v, offset=1):
    """Reconstruct a symmetric matrix from its upper-triangle vector.

    Inverse of sym2vec.

    Args:
        v: (..., K) vector from sym2vec.
        offset: Diagonal offset used in sym2vec.

    Returns:
        (..., N, N) symmetric matrix.
    """
    # Solve for n from K = n*(n-1)/2 (offset=1) or n*(n+1)/2 (offset=0)
    k = v.shape[-1]
    if offset == 1:
        # k = n*(n-1)/2 => n^2 - n - 2k = 0 => n = (1 + sqrt(1+8k))/2
        n = int((1 + jnp.sqrt(1 + 8 * k)) / 2)
    else:
        # k = n*(n+1)/2 => n = (-1 + sqrt(1+8k))/2
        n = int((-1 + jnp.sqrt(1 + 8 * k)) / 2)

    rows, cols = jnp.triu_indices(n, k=offset)
    batch_shape = v.shape[:-1]
    S = jnp.zeros((*batch_shape, n, n), dtype=v.dtype)
    S = S.at[..., rows, cols].set(v)
    S = S + jnp.swapaxes(S, -2, -1)

    if offset == 0:
        # Diagonal was counted twice; halve it
        diag = jnp.diagonal(S, axis1=-2, axis2=-1)
        S = S - jnp.einsum('...i,ij->...ij', diag / 2, jnp.eye(n))

    return S


def toeplitz(c, r=None):
    """Construct a Toeplitz matrix.

    Args:
        c: (N,) first column.
        r: (N,) first row.  If None, uses conjugate of c.

    Returns:
        (N, N) Toeplitz matrix.
    """
    if r is None:
        r = jnp.conj(c)
    n = len(c)
    vals = jnp.concatenate([jnp.flip(r[1:]), c])
    idx = jnp.arange(n)
    indices = idx[:, None] - idx[None, :] + (n - 1)
    return vals[indices]


# ---------------------------------------------------------------------------
# SPD manifold operations
# ---------------------------------------------------------------------------

def _matrix_power(X, p):
    """Matrix power via eigendecomposition.  Works for fractional p."""
    eigvals, eigvecs = jnp.linalg.eigh(X)
    eigvals = jnp.maximum(eigvals, 1e-12)
    return (eigvecs * (eigvals ** p)[..., None, :]) @ eigvecs.swapaxes(-2, -1)


def _logm_spd(X):
    """Matrix logarithm of an SPD matrix via eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eigh(X)
    eigvals = jnp.maximum(eigvals, 1e-12)
    return (eigvecs * jnp.log(eigvals)[..., None, :]) @ eigvecs.swapaxes(-2, -1)


def _expm_spd(X):
    """Matrix exponential of a symmetric matrix via eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eigh(X)
    return (eigvecs * jnp.exp(eigvals)[..., None, :]) @ eigvecs.swapaxes(-2, -1)


def tangent_project_spd(X, reference):
    """Project from SPD cone to tangent space at a reference point.

    T = ref^{-1/2} @ log(ref^{-1/2} @ X @ ref^{-1/2}) @ ref^{-1/2}

    Simplified (symmetric) form:
    T = log(ref^{-1/2} @ X @ ref^{-1/2})

    Args:
        X: (..., N, N) SPD matrix to project.
        reference: (..., N, N) SPD reference point (tangent base).

    Returns:
        (..., N, N) symmetric matrix in tangent space.
    """
    ref_inv_sqrt = _matrix_power(reference, -0.5)
    inner = ref_inv_sqrt @ X @ ref_inv_sqrt
    return _logm_spd(inner)


def cone_project_spd(X, reference):
    """Project from tangent space back to SPD cone.

    S = ref^{1/2} @ exp(X) @ ref^{1/2}

    Inverse of tangent_project_spd.

    Args:
        X: (..., N, N) symmetric matrix in tangent space.
        reference: (..., N, N) SPD reference point.

    Returns:
        (..., N, N) SPD matrix.
    """
    ref_sqrt = _matrix_power(reference, 0.5)
    return ref_sqrt @ _expm_spd(X) @ ref_sqrt


def mean_logeuc_spd(inputs, axis=0):
    """Log-Euclidean mean of SPD matrices.

    mean = exp( mean_i( log(S_i) ) )

    Fast closed-form approximation to the Frechet mean.

    Args:
        inputs: (K, ..., N, N) stack of SPD matrices along axis.
        axis: Axis along which to average.

    Returns:
        (..., N, N) log-Euclidean mean.
    """
    log_inputs = jax.vmap(_logm_spd)(inputs) if axis == 0 else _logm_spd(inputs)
    mean_log = jnp.mean(log_inputs, axis=axis)
    return _expm_spd(mean_log)


def mean_geom_spd(inputs, axis=0, max_iter=10, eps=1e-8):
    """Geometric (Frechet) mean of SPD matrices via iterative tangent averaging.

    Iteratively projects to tangent space at current estimate, averages,
    and maps back.

    Args:
        inputs: (K, ..., N, N) stack of SPD matrices.
        axis: Axis along which to average.
        max_iter: Maximum iterations.
        eps: Convergence threshold (Frobenius norm of tangent update).

    Returns:
        (..., N, N) Frechet mean.
    """
    # Initialise with log-Euclidean mean
    mean = mean_logeuc_spd(inputs, axis=axis)

    def _step(carry, _):
        mean = carry
        mean_inv_sqrt = _matrix_power(mean, -0.5)

        def _project_one(S):
            return _logm_spd(mean_inv_sqrt @ S @ mean_inv_sqrt)

        tangent_vecs = jax.vmap(_project_one)(inputs)
        avg_tangent = jnp.mean(tangent_vecs, axis=0)

        mean_sqrt = _matrix_power(mean, 0.5)
        new_mean = mean_sqrt @ _expm_spd(avg_tangent) @ mean_sqrt
        return new_mean, None

    mean, _ = jax.lax.scan(_step, mean, None, length=max_iter)
    return mean
