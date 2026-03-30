"""Graph primitives, spectral embedding, and sparse message passing.

Ports essential algorithms from hypercoil functional/graph.py and
functional/connectopy.py, plus Chebyshev spectral filtering and sparse
ops from hgx.  All vbjax style (pure functions, JIT/vmap/grad compatible,
no Equinox).
"""

import jax
import jax.numpy as jnp

from .matrix import symmetric


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

def degree(W):
    """Node degree (strength) from a weighted adjacency matrix.

    Args:
        W: (..., N, N) weighted adjacency matrix.

    Returns:
        (..., N) degree vector.
    """
    return jnp.sum(W, axis=-1)


def graph_laplacian(W, normalise=True):
    """Graph Laplacian from a weighted adjacency matrix.

    Args:
        W: (..., N, N) weighted adjacency matrix (non-negative).
        normalise: If True, return the symmetric normalised Laplacian
            L_sym = I - D^{-1/2} W D^{-1/2}.
            If False, return the combinatorial Laplacian L = D - W.

    Returns:
        (..., N, N) Laplacian matrix.
    """
    d = degree(W)

    if normalise:
        d_inv_sqrt = jnp.where(d > 0, 1.0 / jnp.sqrt(d), 0.0)
        # D^{-1/2} W D^{-1/2}
        L_rw = d_inv_sqrt[..., :, None] * W * d_inv_sqrt[..., None, :]
        n = W.shape[-1]
        return jnp.eye(n) - L_rw
    else:
        D = jnp.zeros_like(W)
        idx = jnp.arange(W.shape[-1])
        D = D.at[..., idx, idx].set(d)
        return D - W


def girvan_newman_null(A):
    """Girvan-Newman null model for modularity.

    P_ij = k_i * k_j / (2m)

    Args:
        A: (..., N, N) adjacency matrix.

    Returns:
        (..., N, N) null model matrix.
    """
    k = degree(A)
    m2 = jnp.sum(A, axis=(-2, -1))  # 2m
    m2 = jnp.where(m2 == 0, 1.0, m2)
    return k[..., :, None] * k[..., None, :] / m2[..., None, None]


def modularity_matrix(A, gamma=1.0):
    """Modularity matrix B = A - gamma * P.

    Args:
        A: (..., N, N) adjacency matrix.
        gamma: Resolution parameter (default 1.0).

    Returns:
        (..., N, N) modularity matrix.
    """
    P = girvan_newman_null(A)
    return A - gamma * P


def relaxed_modularity(A, C, gamma=1.0, exclude_diag=True):
    """Differentiable relaxed modularity Q = tr(C^T B C) / (2m).

    Soft community assignment allows gradient-based optimisation.

    Args:
        A: (..., N, N) adjacency matrix.
        C: (..., N, K) soft community assignment matrix.
            Each row sums to ~1 (e.g. softmax output).
        gamma: Resolution parameter.
        exclude_diag: If True, zero out self-loops in B.

    Returns:
        Scalar modularity value.
    """
    B = modularity_matrix(A, gamma=gamma)
    if exclude_diag:
        n = B.shape[-1]
        B = B * (1.0 - jnp.eye(n))
    m2 = jnp.sum(A, axis=(-2, -1))
    m2 = jnp.where(m2 == 0, 1.0, m2)
    # Q = tr(C^T B C) / (2m)
    return jnp.trace(C.swapaxes(-2, -1) @ B @ C) / m2


# ---------------------------------------------------------------------------
# Spectral embedding
# ---------------------------------------------------------------------------

def laplacian_eigenmaps(W, k=10, normalise=True):
    """Connectopic coordinates via Laplacian eigenmaps.

    Computes the k smallest non-trivial eigenvectors of the graph Laplacian
    as low-dimensional embedding coordinates.

    Args:
        W: (N, N) weighted adjacency matrix (symmetric, non-negative).
        k: Number of embedding dimensions.
        normalise: If True, use normalised Laplacian.

    Returns:
        eigvals: (k,) smallest non-trivial eigenvalues.
        eigvecs: (N, k) corresponding eigenvectors (embedding coordinates).
    """
    L = graph_laplacian(W, normalise=normalise)
    # eigh returns eigenvalues in ascending order
    all_eigvals, all_eigvecs = jnp.linalg.eigh(L)
    # Skip the first eigenvector (constant, eigenvalue ~0)
    eigvals = all_eigvals[1:k + 1]
    eigvecs = all_eigvecs[:, 1:k + 1]
    # Enforce sign consistency: largest absolute element is positive
    signs = jnp.sign(eigvecs[jnp.argmax(jnp.abs(eigvecs), axis=0), jnp.arange(k)])
    eigvecs = eigvecs * signs[None, :]
    return eigvals, eigvecs


def diffusion_mapping(W, k=10, alpha=0.5, diffusion_time=0):
    """Diffusion map embedding.

    Computes an anisotropic diffusion on the graph and returns the top-k
    eigenvectors weighted by eigenvalues.

    Args:
        W: (N, N) weighted affinity matrix (symmetric, non-negative).
        k: Number of embedding dimensions.
        alpha: Anisotropy parameter in [0, 1].
            0 = classical Laplacian normalisation.
            0.5 = Fokker-Planck (default, Lafon & Lee 2006).
            1.0 = Laplace-Beltrami normalisation.
        diffusion_time: If > 0, scale eigenvectors by lambda^t.
            If 0, use default lambda / (1 - lambda) scaling.

    Returns:
        eigvals: (k,) diffusion eigenvalues.
        eigvecs: (N, k) diffusion coordinates.
    """
    # Anisotropic kernel normalisation
    d_alpha = jnp.sum(W, axis=-1) ** alpha
    d_alpha = jnp.where(d_alpha > 0, d_alpha, 1.0)
    K = W / (d_alpha[:, None] * d_alpha[None, :])

    # Row-normalise to get the diffusion operator
    row_sums = jnp.sum(K, axis=-1)
    row_sums = jnp.where(row_sums > 0, row_sums, 1.0)
    P = K / row_sums[:, None]

    # Symmetrise for eigendecomposition: P_sym = D^{1/2} P D^{-1/2}
    d_sqrt = jnp.sqrt(row_sums)
    d_inv_sqrt = 1.0 / d_sqrt
    P_sym = d_sqrt[:, None] * P * d_inv_sqrt[None, :]
    P_sym = symmetric(P_sym)

    # Eigendecompose (largest eigenvalues = slowest diffusion modes)
    all_eigvals, all_eigvecs = jnp.linalg.eigh(P_sym)
    # Reverse to get descending order; skip first (trivial, eigenvalue ~1)
    all_eigvals = jnp.flip(all_eigvals)
    all_eigvecs = jnp.flip(all_eigvecs, axis=-1)

    eigvals = all_eigvals[1:k + 1]
    eigvecs = all_eigvecs[:, 1:k + 1]

    # Map back from symmetrised eigenvectors
    eigvecs = eigvecs / d_sqrt[:, None]

    # Apply diffusion time scaling
    if diffusion_time > 0:
        eigvecs = eigvecs * (eigvals ** diffusion_time)[None, :]
    else:
        # Default: lambda / (1 - lambda) scaling
        scale = eigvals / jnp.maximum(1.0 - eigvals, 1e-10)
        eigvecs = eigvecs * scale[None, :]

    # Sign consistency
    signs = jnp.sign(eigvecs[jnp.argmax(jnp.abs(eigvecs), axis=0), jnp.arange(k)])
    eigvecs = eigvecs * signs[None, :]

    return eigvals, eigvecs


# ---------------------------------------------------------------------------
# Chebyshev spectral filtering (adapted from hgx/_wavelets.py)
# ---------------------------------------------------------------------------

def chebyshev_filter(L, x, coeffs):
    """Apply a Chebyshev polynomial spectral filter without eigendecomposition.

    Computes h(L) @ x where h(lambda) = sum_k a_k T_k(L_tilde), using
    the Chebyshev recurrence T_{k+1} = 2*L_tilde*T_k - T_{k-1}.

    O(K * nnz) instead of O(N^3) for eigendecomposition.  Essential for
    vertex-wise analysis on cortical surfaces (32k+ nodes).

    Args:
        L: (N, N) normalised Laplacian (eigenvalues in [0, 2]).
        x: (N, D) node features / signals.
        coeffs: (K,) Chebyshev polynomial coefficients.

    Returns:
        (N, D) filtered features.
    """
    K = coeffs.shape[0]
    n = L.shape[0]

    # Rescale L to [-1, 1]: L_tilde = L - I
    L_tilde = L - jnp.eye(n)

    # T_0(L_tilde) @ x = x
    T_prev = x
    out = coeffs[0] * T_prev

    if K == 1:
        return out

    # T_1(L_tilde) @ x = L_tilde @ x
    T_curr = L_tilde @ x
    out = out + coeffs[1] * T_curr

    def _step(carry, k):
        T_prev, T_curr, out = carry
        T_next = 2.0 * (L_tilde @ T_curr) - T_prev
        out = out + coeffs[k] * T_next
        return (T_curr, T_next, out), None

    if K > 2:
        (_, _, out), _ = jax.lax.scan(
            _step, (T_prev, T_curr, out), jnp.arange(2, K),
        )

    return out


def make_chebyshev_filter(L, K=5, *, key):
    """Factory for learnable Chebyshev spectral filter (vbjax style).

    Args:
        L: (N, N) normalised Laplacian.
        K: Polynomial order (number of Chebyshev terms).
        key: JAX PRNG key.

    Returns:
        coeffs: (K,) initial Chebyshev coefficients.
        forward_fn: (coeffs, x) -> filtered features.
    """
    coeffs = jax.random.normal(key, (K,)) * 0.1

    def forward(coeffs, x):
        return chebyshev_filter(L, x, coeffs)

    return coeffs, forward


# ---------------------------------------------------------------------------
# Spectral features (adapted from hgx/_wavelets.py)
# ---------------------------------------------------------------------------

def spectral_features(W, k=10, normalise=True):
    """Extract spectral summary features from a connectivity matrix.

    Returns the first k eigenvalues of the Laplacian plus:
    - Spectral gap (lambda_2 - lambda_1)
    - Algebraic connectivity (lambda_2)
    - Spectral radius (largest eigenvalue)

    Args:
        W: (N, N) weighted adjacency / connectivity matrix.
        k: Number of smallest eigenvalues to include.
        normalise: If True, use normalised Laplacian.

    Returns:
        features: (k + 3,) feature vector.
    """
    L = graph_laplacian(W, normalise=normalise)
    eigenvalues = jnp.linalg.eigvalsh(L)
    eigenvalues = jnp.sort(eigenvalues)

    n = eigenvalues.shape[0]
    padded = jnp.zeros(k)
    num_to_take = jnp.minimum(n, k)
    padded = padded.at[:num_to_take].set(eigenvalues[:num_to_take])

    lambda_2 = jnp.where(n >= 2, eigenvalues[1], 0.0)
    spectral_gap = lambda_2
    algebraic_connectivity = lambda_2
    spectral_radius = eigenvalues[-1]

    return jnp.concatenate([
        padded,
        jnp.array([spectral_gap, algebraic_connectivity, spectral_radius]),
    ])


# ---------------------------------------------------------------------------
# Sparse message passing (adapted from hgx/_sparse.py)
# ---------------------------------------------------------------------------

def sparse_degree(indices, num_nodes):
    """Compute node degree from edge index pairs.

    Args:
        indices: (E, 2) edge index pairs (source, target).
        num_nodes: Total number of nodes.

    Returns:
        (num_nodes,) degree vector.
    """
    return jax.ops.segment_sum(
        jnp.ones(indices.shape[0]), indices[:, 0], num_segments=num_nodes
    )


def sparse_aggregate(x, source_idx, target_idx, num_targets):
    """Scatter-add features from source nodes to target nodes.

    O(E * D) where E = number of edges, D = feature dim.

    Args:
        x: (N, D) source node features.
        source_idx: (E,) indices of source nodes.
        target_idx: (E,) indices of target nodes.
        num_targets: Total number of target nodes.

    Returns:
        (num_targets, D) aggregated features.
    """
    vals = x[source_idx]
    return jax.ops.segment_sum(vals, target_idx, num_segments=num_targets)


def sparse_graph_conv(x, source_idx, target_idx, num_nodes, weights=None):
    """Sparse graph convolution via message passing.

    Computes: h_i = (1/d_i) * sum_{j in N(i)} w_{ji} * x_j

    Args:
        x: (N, D) node features.
        source_idx: (E,) source node indices.
        target_idx: (E,) target node indices.
        num_nodes: Total number of nodes.
        weights: Optional (E,) edge weights. If None, uses 1.0.

    Returns:
        (N, D) convolved features.
    """
    vals = x[source_idx]
    if weights is not None:
        vals = vals * weights[:, None]
    agg = jax.ops.segment_sum(vals, target_idx, num_segments=num_nodes)
    d = jax.ops.segment_sum(
        jnp.ones(source_idx.shape[0]) if weights is None else weights,
        target_idx, num_segments=num_nodes,
    )
    d = jnp.where(d > 0, d, 1.0)
    return agg / d[:, None]


def adjacency_to_edge_index(W, threshold=0.0):
    """Convert dense adjacency matrix to sparse edge index format.

    Args:
        W: (N, N) adjacency matrix.
        threshold: Minimum weight to include an edge.

    Returns:
        source_idx: (E,) source node indices.
        target_idx: (E,) target node indices.
        weights: (E,) edge weights.
    """
    N = W.shape[0]
    max_edges = N * N
    rows, cols = jnp.nonzero(W > threshold, size=max_edges, fill_value=0)
    vals = W[rows, cols]
    valid = jnp.arange(max_edges) < jnp.sum(W > threshold)
    # Zero out padding entries
    rows = jnp.where(valid, rows, 0)
    cols = jnp.where(valid, cols, 0)
    vals = jnp.where(valid, vals, 0.0)
    return rows, cols, vals
