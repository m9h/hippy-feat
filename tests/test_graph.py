"""Tests for jaxoccoli.graph module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.graph import (
    degree,
    diffusion_mapping,
    girvan_newman_null,
    graph_laplacian,
    laplacian_eigenmaps,
    modularity_matrix,
    relaxed_modularity,
)

KEY = jax.random.PRNGKey(21)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def block_adjacency():
    """Two-block adjacency matrix (10 nodes, 2 communities of 5)."""
    A = jnp.zeros((10, 10))
    # Block 1: nodes 0-4 fully connected
    A = A.at[:5, :5].set(1.0)
    # Block 2: nodes 5-9 fully connected
    A = A.at[5:, 5:].set(1.0)
    # Remove self-loops
    A = A * (1 - jnp.eye(10))
    # Weak inter-block connections
    A = A.at[0, 5].set(0.1)
    A = A.at[5, 0].set(0.1)
    return A


@pytest.fixture
def random_adjacency():
    """Random symmetric non-negative adjacency (20 nodes)."""
    A = jnp.abs(jax.random.normal(KEY, (20, 20)))
    A = (A + A.T) / 2
    A = A * (1 - jnp.eye(20))
    return A


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

class TestDegree:
    def test_shape(self, block_adjacency):
        d = degree(block_adjacency)
        assert d.shape == (10,)

    def test_values(self, block_adjacency):
        d = degree(block_adjacency)
        # Node 0: connected to 1,2,3,4 (weight 1 each) + node 5 (0.1)
        np.testing.assert_allclose(d[0], 4.1, atol=1e-6)
        # Node 1: connected to 0,2,3,4 (weight 1 each)
        np.testing.assert_allclose(d[1], 4.0, atol=1e-6)


class TestGraphLaplacian:
    def test_unnormalised_row_sum_zero(self, block_adjacency):
        L = graph_laplacian(block_adjacency, normalise=False)
        row_sums = jnp.sum(L, axis=-1)
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-6)

    def test_normalised_shape(self, block_adjacency):
        L = graph_laplacian(block_adjacency, normalise=True)
        assert L.shape == (10, 10)

    def test_normalised_eigenvalues_bounded(self, block_adjacency):
        L = graph_laplacian(block_adjacency, normalise=True)
        eigvals = jnp.linalg.eigvalsh(L)
        assert jnp.all(eigvals >= -1e-6)
        assert jnp.all(eigvals <= 2.0 + 1e-6)

    def test_smallest_eigenvalue_zero(self, block_adjacency):
        L = graph_laplacian(block_adjacency, normalise=False)
        eigvals = jnp.linalg.eigvalsh(L)
        np.testing.assert_allclose(eigvals[0], 0.0, atol=1e-5)

    def test_symmetric(self, random_adjacency):
        L = graph_laplacian(random_adjacency, normalise=True)
        np.testing.assert_allclose(L, L.T, atol=1e-6)


class TestModularity:
    def test_girvan_newman_null_row_sums(self, block_adjacency):
        P = girvan_newman_null(block_adjacency)
        # Each row sum of P should equal degree
        d = degree(block_adjacency)
        np.testing.assert_allclose(jnp.sum(P, axis=-1), d, atol=1e-5)

    def test_modularity_matrix_shape(self, block_adjacency):
        B = modularity_matrix(block_adjacency)
        assert B.shape == (10, 10)

    def test_modularity_matrix_row_sum_zero(self, block_adjacency):
        B = modularity_matrix(block_adjacency)
        np.testing.assert_allclose(jnp.sum(B, axis=-1), 0.0, atol=1e-5)


class TestRelaxedModularity:
    def test_positive_for_correct_partition(self, block_adjacency):
        # Perfect partition: nodes 0-4 in community 0, 5-9 in community 1
        C = jnp.zeros((10, 2))
        C = C.at[:5, 0].set(1.0)
        C = C.at[5:, 1].set(1.0)
        Q = relaxed_modularity(block_adjacency, C)
        assert Q > 0

    def test_lower_for_wrong_partition(self, block_adjacency):
        # Correct partition
        C_good = jnp.zeros((10, 2))
        C_good = C_good.at[:5, 0].set(1.0)
        C_good = C_good.at[5:, 1].set(1.0)
        # Random partition
        C_bad = jax.nn.softmax(jax.random.normal(KEY, (10, 2)), axis=-1)
        Q_good = relaxed_modularity(block_adjacency, C_good)
        Q_bad = relaxed_modularity(block_adjacency, C_bad)
        assert Q_good > Q_bad

    def test_grad(self, block_adjacency):
        def loss(C):
            return -relaxed_modularity(block_adjacency, jax.nn.softmax(C, axis=-1))
        C = jax.random.normal(KEY, (10, 2))
        g = jax.grad(loss)(C)
        assert g.shape == (10, 2)
        assert jnp.all(jnp.isfinite(g))

    def test_jit(self, block_adjacency):
        C = jax.nn.softmax(jax.random.normal(KEY, (10, 2)), axis=-1)
        Q = jax.jit(relaxed_modularity)(block_adjacency, C)
        assert jnp.isfinite(Q)


# ---------------------------------------------------------------------------
# Spectral embedding
# ---------------------------------------------------------------------------

class TestLaplacianEigenmaps:
    def test_shape(self, random_adjacency):
        eigvals, eigvecs = laplacian_eigenmaps(random_adjacency, k=5)
        assert eigvals.shape == (5,)
        assert eigvecs.shape == (20, 5)

    def test_eigenvalues_positive(self, random_adjacency):
        eigvals, _ = laplacian_eigenmaps(random_adjacency, k=5)
        assert jnp.all(eigvals > -1e-6)

    def test_two_blocks_separated(self, block_adjacency):
        """First eigenvector should separate the two communities."""
        _, eigvecs = laplacian_eigenmaps(block_adjacency, k=2)
        # First embedding dimension should have opposite signs for the two blocks
        v1 = eigvecs[:, 0]
        block1_sign = jnp.sign(jnp.mean(v1[:5]))
        block2_sign = jnp.sign(jnp.mean(v1[5:]))
        assert block1_sign != block2_sign

    def test_eigenvalues_ascending(self, random_adjacency):
        eigvals, _ = laplacian_eigenmaps(random_adjacency, k=5)
        diffs = jnp.diff(eigvals)
        assert jnp.all(diffs >= -1e-6)


class TestDiffusionMapping:
    def test_shape(self, random_adjacency):
        eigvals, eigvecs = diffusion_mapping(random_adjacency, k=5)
        assert eigvals.shape == (5,)
        assert eigvecs.shape == (20, 5)

    def test_eigenvalues_bounded(self, random_adjacency):
        eigvals, _ = diffusion_mapping(random_adjacency, k=5)
        # Diffusion eigenvalues should be in (0, 1)
        assert jnp.all(eigvals > -1e-6)
        assert jnp.all(eigvals <= 1.0 + 1e-6)

    def test_two_blocks_separated(self, block_adjacency):
        _, eigvecs = diffusion_mapping(block_adjacency, k=2)
        v1 = eigvecs[:, 0]
        block1_sign = jnp.sign(jnp.mean(v1[:5]))
        block2_sign = jnp.sign(jnp.mean(v1[5:]))
        assert block1_sign != block2_sign

    def test_alpha_parameter(self, random_adjacency):
        """Different alpha values should produce different embeddings."""
        _, v1 = diffusion_mapping(random_adjacency, k=3, alpha=0.0)
        _, v2 = diffusion_mapping(random_adjacency, k=3, alpha=1.0)
        # They should be different (not identical)
        assert not jnp.allclose(v1, v2, atol=1e-3)
