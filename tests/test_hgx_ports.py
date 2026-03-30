"""Tests for hgx-ported functionality: Chebyshev filter, spectral features,
sparse ops, transport, and Fisher-Rao natural gradient."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.graph import (
    adjacency_to_edge_index,
    chebyshev_filter,
    graph_laplacian,
    make_chebyshev_filter,
    sparse_aggregate,
    sparse_degree,
    sparse_graph_conv,
    spectral_features,
)
from jaxoccoli.transport import (
    euclidean_cost,
    gromov_wasserstein,
    gromov_wasserstein_fc,
    sinkhorn,
    wasserstein_distance,
    wasserstein_fc_distance,
)
from jaxoccoli.learnable import (
    fisher_rao_metric,
    make_atlas_natural_grad,
    natural_gradient,
    natural_gradient_step,
)

KEY = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def block_adjacency():
    A = jnp.zeros((10, 10))
    A = A.at[:5, :5].set(1.0).at[5:, 5:].set(1.0)
    A = A * (1 - jnp.eye(10))
    return A


@pytest.fixture
def random_adjacency():
    A = jnp.abs(jax.random.normal(KEY, (20, 20)))
    A = (A + A.T) / 2
    A = A * (1 - jnp.eye(20))
    return A


# ---------------------------------------------------------------------------
# Chebyshev spectral filter
# ---------------------------------------------------------------------------

class TestChebyshevFilter:
    def test_identity_coeffs(self, random_adjacency):
        """coeffs = [1, 0, 0, ...] should return identity (T_0 = I)."""
        L = graph_laplacian(random_adjacency)
        x = jax.random.normal(KEY, (20, 3))
        coeffs = jnp.array([1.0, 0.0, 0.0, 0.0, 0.0])
        out = chebyshev_filter(L, x, coeffs)
        np.testing.assert_allclose(out, x, atol=1e-5)

    def test_output_shape(self, random_adjacency):
        L = graph_laplacian(random_adjacency)
        x = jax.random.normal(KEY, (20, 5))
        coeffs = jnp.ones(4)
        out = chebyshev_filter(L, x, coeffs)
        assert out.shape == (20, 5)

    def test_single_coeff(self, random_adjacency):
        L = graph_laplacian(random_adjacency)
        x = jax.random.normal(KEY, (20, 3))
        coeffs = jnp.array([2.0])
        out = chebyshev_filter(L, x, coeffs)
        np.testing.assert_allclose(out, 2.0 * x, atol=1e-5)

    def test_grad(self, random_adjacency):
        L = graph_laplacian(random_adjacency)
        x = jax.random.normal(KEY, (20, 3))

        def loss(coeffs):
            return jnp.sum(chebyshev_filter(L, x, coeffs) ** 2)

        g = jax.grad(loss)(jnp.ones(5))
        assert g.shape == (5,)
        assert jnp.all(jnp.isfinite(g))

    def test_jit(self, random_adjacency):
        L = graph_laplacian(random_adjacency)
        x = jax.random.normal(KEY, (20, 3))
        coeffs = jnp.ones(5)
        out = jax.jit(lambda c: chebyshev_filter(L, x, c))(coeffs)
        assert out.shape == (20, 3)


class TestMakeChebyshevFilter:
    def test_factory(self, random_adjacency):
        L = graph_laplacian(random_adjacency)
        coeffs, fwd = make_chebyshev_filter(L, K=5, key=KEY)
        x = jax.random.normal(KEY, (20, 3))
        out = fwd(coeffs, x)
        assert out.shape == (20, 3)

    def test_grad_through_factory(self, random_adjacency):
        L = graph_laplacian(random_adjacency)
        coeffs, fwd = make_chebyshev_filter(L, K=5, key=KEY)
        x = jax.random.normal(KEY, (20, 3))

        def loss(c):
            return jnp.sum(fwd(c, x) ** 2)

        g = jax.grad(loss)(coeffs)
        assert g.shape == (5,)
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Spectral features
# ---------------------------------------------------------------------------

class TestSpectralFeatures:
    def test_shape(self, random_adjacency):
        feats = spectral_features(random_adjacency, k=5)
        assert feats.shape == (8,)  # 5 eigenvalues + 3 summary stats

    def test_algebraic_connectivity_positive(self, block_adjacency):
        feats = spectral_features(block_adjacency, k=5)
        alg_conn = feats[-2]  # algebraic connectivity
        assert alg_conn > 0  # connected graph

    def test_spectral_radius_bounded(self, random_adjacency):
        feats = spectral_features(random_adjacency, k=5)
        spec_radius = feats[-1]
        assert spec_radius <= 2.0 + 1e-5  # normalised Laplacian bound


# ---------------------------------------------------------------------------
# Sparse message passing
# ---------------------------------------------------------------------------

class TestSparseOps:
    def test_sparse_degree(self):
        # Simple triangle: edges (0,1), (1,2), (2,0)
        indices = jnp.array([[0, 1], [1, 2], [2, 0]])
        d = sparse_degree(indices, 3)
        np.testing.assert_allclose(d, jnp.array([1.0, 1.0, 1.0]))

    def test_sparse_aggregate(self):
        x = jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        # Aggregate nodes 0,1 -> target 0; node 2 -> target 1
        source = jnp.array([0, 1, 2])
        target = jnp.array([0, 0, 1])
        agg = sparse_aggregate(x, source, target, 2)
        np.testing.assert_allclose(agg[0], jnp.array([1.0, 1.0]))
        np.testing.assert_allclose(agg[1], jnp.array([1.0, 1.0]))

    def test_sparse_graph_conv_shape(self, random_adjacency):
        src, tgt, w = adjacency_to_edge_index(random_adjacency, threshold=0.5)
        x = jax.random.normal(KEY, (20, 4))
        out = sparse_graph_conv(x, src, tgt, 20, weights=w)
        assert out.shape == (20, 4)

    def test_adjacency_to_edge_index(self, block_adjacency):
        src, tgt, w = adjacency_to_edge_index(block_adjacency)
        # All weights should be 1.0 (binary adjacency)
        assert jnp.all((w == 0.0) | (w == 1.0))


# ---------------------------------------------------------------------------
# Optimal transport
# ---------------------------------------------------------------------------

class TestSinkhorn:
    def test_marginals(self):
        N, M = 5, 7
        C = jnp.abs(jax.random.normal(KEY, (N, M)))
        a = jnp.ones(N) / N
        b = jnp.ones(M) / M
        T = sinkhorn(C, a, b, epsilon=0.1, max_iters=200)
        np.testing.assert_allclose(jnp.sum(T, axis=1), a, atol=1e-3)
        np.testing.assert_allclose(jnp.sum(T, axis=0), b, atol=1e-3)

    def test_nonnegative(self):
        C = jnp.abs(jax.random.normal(KEY, (5, 5)))
        T = sinkhorn(C)
        assert jnp.all(T >= -1e-8)

    def test_jit(self):
        C = jnp.abs(jax.random.normal(KEY, (5, 5)))
        T = jax.jit(sinkhorn)(C)
        assert T.shape == (5, 5)

    def test_grad(self):
        def loss(C):
            T = sinkhorn(C, epsilon=0.5, max_iters=50)
            return jnp.sum(T * C)

        C = jnp.abs(jax.random.normal(KEY, (5, 5))) + 0.1
        g = jax.grad(loss)(C)
        assert g.shape == C.shape
        assert jnp.all(jnp.isfinite(g))


class TestWassersteinDistance:
    def test_zero_for_identical(self):
        X = jax.random.normal(KEY, (10, 3))
        d = wasserstein_distance(X, X, epsilon=0.1, max_iters=200)
        np.testing.assert_allclose(d, 0.0, atol=0.1)

    def test_positive(self):
        k1, k2 = jax.random.split(KEY)
        X = jax.random.normal(k1, (10, 3))
        Y = jax.random.normal(k2, (10, 3)) + 5.0
        d = wasserstein_distance(X, Y, epsilon=0.1)
        assert d > 0

    def test_symmetric(self):
        k1, k2 = jax.random.split(KEY)
        X = jax.random.normal(k1, (8, 2))
        Y = jax.random.normal(k2, (8, 2))
        d1 = wasserstein_distance(X, Y, epsilon=0.5, max_iters=100)
        d2 = wasserstein_distance(Y, X, epsilon=0.5, max_iters=100)
        np.testing.assert_allclose(d1, d2, atol=0.1)


class TestWassersteinFC:
    def test_zero_for_identical(self):
        fc = jax.random.normal(KEY, (10, 10))
        fc = (fc + fc.T) / 2
        d = wasserstein_fc_distance(fc, fc)
        np.testing.assert_allclose(d, 0.0, atol=1e-6)

    def test_positive_for_different(self):
        k1, k2 = jax.random.split(KEY)
        fc1 = jax.random.normal(k1, (10, 10))
        fc2 = jax.random.normal(k2, (10, 10))
        fc1 = (fc1 + fc1.T) / 2
        fc2 = (fc2 + fc2.T) / 2
        d = wasserstein_fc_distance(fc1, fc2)
        assert d > 0


class TestGromovWasserstein:
    def test_output_shapes(self):
        k1, k2 = jax.random.split(KEY)
        D1 = jnp.abs(jax.random.normal(k1, (8, 8)))
        D2 = jnp.abs(jax.random.normal(k2, (10, 10)))
        D1 = (D1 + D1.T) / 2
        D2 = (D2 + D2.T) / 2
        T, cost = gromov_wasserstein(D1, D2, epsilon=1.0, gw_iters=5)
        assert T.shape == (8, 10)
        assert cost.shape == ()

    def test_zero_for_identical(self):
        D = jnp.abs(jax.random.normal(KEY, (8, 8)))
        D = (D + D.T) / 2
        _, cost = gromov_wasserstein(D, D, epsilon=1.0, gw_iters=10)
        np.testing.assert_allclose(cost, 0.0, atol=0.5)

    def test_fc_wrapper(self):
        k1, k2 = jax.random.split(KEY)
        fc1 = jax.random.normal(k1, (8, 8))
        fc2 = jax.random.normal(k2, (8, 8))
        fc1 = (fc1 + fc1.T) / 2
        fc2 = (fc2 + fc2.T) / 2
        T, cost = gromov_wasserstein_fc(fc1, fc2, epsilon=1.0, gw_iters=5)
        assert T.shape == (8, 8)
        assert jnp.isfinite(cost)


# ---------------------------------------------------------------------------
# Fisher-Rao natural gradient
# ---------------------------------------------------------------------------

class TestFisherRao:
    def test_metric_shape(self):
        p = jax.nn.softmax(jax.random.normal(KEY, (5, 3)), axis=-1)
        G = fisher_rao_metric(p)
        assert G.shape == (5, 3, 3)

    def test_metric_diagonal(self):
        p = jax.nn.softmax(jax.random.normal(KEY, (3,)), axis=-1)
        G = fisher_rao_metric(p[None, :])[0]
        # Should be diagonal with 1/p_k
        np.testing.assert_allclose(jnp.diag(G), 1.0 / p, atol=1e-5)

    def test_natural_gradient_scaling(self):
        p = jax.nn.softmax(jax.random.normal(KEY, (5,)), axis=-1)
        grad = jnp.ones(5)
        ng = natural_gradient(grad, p)
        # Natural gradient = p * grad = p * 1 = p
        np.testing.assert_allclose(ng, p, atol=1e-6)

    def test_natural_gradient_step_on_simplex(self):
        p = jax.nn.softmax(jax.random.normal(KEY, (5,)), axis=-1)

        def loss(p):
            target = jnp.ones(5) / 5.0
            return jnp.sum((p - target) ** 2)

        p_new = natural_gradient_step(loss, p, lr=0.1)
        np.testing.assert_allclose(jnp.sum(p_new), 1.0, atol=1e-5)
        assert jnp.all(p_new > 0)


class TestAtlasNaturalGrad:
    def test_factory_outputs(self):
        params, fwd, update = make_atlas_natural_grad(50, 5, key=KEY)
        data = jax.random.normal(KEY, (50, 30))
        out = fwd(params, data)
        assert out.shape == (5, 30)

    def test_update_changes_params(self):
        params, fwd, update = make_atlas_natural_grad(50, 5, key=KEY)
        data = jax.random.normal(KEY, (50, 30))

        def loss(p):
            return jnp.sum(fwd(p, data) ** 2)

        new_params = update(params, loss)
        assert not jnp.allclose(new_params.weight, params.weight)

    def test_update_reduces_loss(self):
        params, fwd, update = make_atlas_natural_grad(50, 5, key=KEY, lr=0.01)
        data = jax.random.normal(KEY, (50, 30))

        def loss(p):
            return jnp.sum(fwd(p, data) ** 2)

        loss_before = loss(params)
        new_params = update(params, loss)
        loss_after = loss(new_params)
        assert loss_after < loss_before

    def test_grad_flows(self):
        params, fwd, update = make_atlas_natural_grad(50, 5, key=KEY)
        data = jax.random.normal(KEY, (50, 30))

        def loss(p):
            return jnp.sum(fwd(p, data) ** 2)

        g = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(g.weight))
