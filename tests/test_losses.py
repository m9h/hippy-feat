"""Tests for jaxoccoli.losses module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.losses import (
    compactness_loss,
    connectopy_loss,
    dispersion_loss,
    eigenmaps_loss,
    entropy,
    equilibrium_loss,
    js_divergence,
    kl_divergence,
    modularity_loss,
    multivariate_kurtosis,
    qcfc_loss,
    reference_tether_loss,
    reliability_weighted_loss,
    smoothness_loss,
)

KEY = jax.random.PRNGKey(77)


# ---------------------------------------------------------------------------
# Information-theoretic
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_maximum(self):
        p = jnp.ones(10) / 10
        h = entropy(p)
        expected = jnp.log(10.0)
        np.testing.assert_allclose(h, expected, atol=1e-5)

    def test_deterministic_zero(self):
        p = jnp.zeros(10).at[0].set(1.0)
        h = entropy(p)
        np.testing.assert_allclose(h, 0.0, atol=1e-4)

    def test_nonnegative(self):
        p = jax.nn.softmax(jax.random.normal(KEY, (20,)))
        h = entropy(p)
        assert h >= -1e-6


class TestKLDivergence:
    def test_zero_for_identical(self):
        p = jax.nn.softmax(jax.random.normal(KEY, (10,)))
        kl = kl_divergence(p, p)
        np.testing.assert_allclose(kl, 0.0, atol=1e-5)

    def test_nonnegative(self):
        k1, k2 = jax.random.split(KEY)
        p = jax.nn.softmax(jax.random.normal(k1, (10,)))
        q = jax.nn.softmax(jax.random.normal(k2, (10,)))
        kl = kl_divergence(p, q)
        assert kl >= -1e-6

    def test_asymmetric(self):
        k1, k2 = jax.random.split(KEY)
        p = jax.nn.softmax(jax.random.normal(k1, (10,)))
        q = jax.nn.softmax(jax.random.normal(k2, (10,)))
        assert not jnp.allclose(kl_divergence(p, q), kl_divergence(q, p))


class TestJSDivergence:
    def test_zero_for_identical(self):
        p = jax.nn.softmax(jax.random.normal(KEY, (10,)))
        js = js_divergence(p, p)
        np.testing.assert_allclose(js, 0.0, atol=1e-5)

    def test_symmetric(self):
        k1, k2 = jax.random.split(KEY)
        p = jax.nn.softmax(jax.random.normal(k1, (10,)))
        q = jax.nn.softmax(jax.random.normal(k2, (10,)))
        np.testing.assert_allclose(js_divergence(p, q), js_divergence(q, p), atol=1e-6)

    def test_bounded(self):
        k1, k2 = jax.random.split(KEY)
        p = jax.nn.softmax(jax.random.normal(k1, (10,)))
        q = jax.nn.softmax(jax.random.normal(k2, (10,)))
        js = js_divergence(p, q)
        assert js >= -1e-6
        assert js <= jnp.log(2.0) + 1e-6


# ---------------------------------------------------------------------------
# Network losses
# ---------------------------------------------------------------------------

class TestModularityLoss:
    def test_correct_partition_negative(self):
        """Correct partition should produce negative loss (high modularity)."""
        A = jnp.zeros((10, 10))
        A = A.at[:5, :5].set(1.0).at[5:, 5:].set(1.0)
        A = A * (1 - jnp.eye(10))
        C = jnp.zeros((10, 2))
        C = C.at[:5, 0].set(1.0).at[5:, 1].set(1.0)
        loss = modularity_loss(A, C)
        assert loss < 0

    def test_grad(self):
        A = jnp.abs(jax.random.normal(KEY, (10, 10)))
        A = (A + A.T) / 2

        def loss(C_raw):
            C = jax.nn.softmax(C_raw, axis=-1)
            return modularity_loss(A, C)

        g = jax.grad(loss)(jax.random.normal(KEY, (10, 3)))
        assert jnp.all(jnp.isfinite(g))


class TestConnectopyLoss:
    def test_zero_for_constant_embedding(self):
        Q = jnp.ones((10, 2))
        A = jnp.ones((10, 10))
        loss = connectopy_loss(Q, A)
        np.testing.assert_allclose(loss, 0.0, atol=1e-6)

    def test_grad(self):
        A = jnp.abs(jax.random.normal(KEY, (10, 10)))

        def loss(Q):
            return connectopy_loss(Q, A)

        g = jax.grad(loss)(jax.random.normal(KEY, (10, 3)))
        assert jnp.all(jnp.isfinite(g))


class TestEigenmapsLoss:
    def test_grad(self):
        A = jnp.abs(jax.random.normal(KEY, (10, 10)))
        A = (A + A.T) / 2

        def loss(Q):
            return eigenmaps_loss(Q, A)

        g = jax.grad(loss)(jax.random.normal(KEY, (10, 3)))
        assert jnp.all(jnp.isfinite(g))


# ---------------------------------------------------------------------------
# Spatial constraint losses
# ---------------------------------------------------------------------------

class TestCompactnessLoss:
    def test_nonnegative(self):
        assignment = jax.nn.softmax(jax.random.normal(KEY, (20, 3)), axis=-1)
        coor = jax.random.normal(KEY, (20, 3))
        loss = compactness_loss(assignment, coor)
        assert loss >= -1e-6

    def test_grad(self):
        coor = jax.random.normal(KEY, (20, 3))

        def loss(raw):
            return compactness_loss(jax.nn.softmax(raw, axis=-1), coor)

        g = jax.grad(loss)(jax.random.normal(KEY, (20, 3)))
        assert jnp.all(jnp.isfinite(g))


class TestDispersionLoss:
    def test_negative(self):
        assignment = jax.nn.softmax(jax.random.normal(KEY, (20, 3)), axis=-1)
        coor = jax.random.normal(KEY, (20, 3))
        loss = dispersion_loss(assignment, coor)
        assert loss < 0  # negative = encourages spread


class TestReferenceTetherLoss:
    def test_zero_at_reference(self):
        # If COM matches reference, loss should be ~0
        assignment = jnp.eye(5)  # 5 nodes, 5 parcels (identity)
        coor = jax.random.normal(KEY, (5, 2))
        ref = coor  # reference = actual coordinates
        loss = reference_tether_loss(assignment, coor, ref)
        np.testing.assert_allclose(loss, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Regularisation
# ---------------------------------------------------------------------------

class TestSmoothnessLoss:
    def test_constant_zero(self):
        x = jnp.ones(50)
        loss = smoothness_loss(x)
        np.testing.assert_allclose(loss, 0.0, atol=1e-7)

    def test_linear_first_order_zero(self):
        x = jnp.linspace(0, 1, 50)
        loss = smoothness_loss(x, n=2)
        np.testing.assert_allclose(loss, 0.0, atol=1e-4)


class TestEquilibriumLoss:
    def test_uniform_minimum(self):
        X = jnp.ones((20, 5))
        loss_uniform = equilibrium_loss(X)
        X_imbalanced = jnp.zeros((20, 5)).at[:, 0].set(1.0)
        loss_imbalanced = equilibrium_loss(X_imbalanced)
        assert loss_uniform < loss_imbalanced


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------

class TestQcfcLoss:
    def test_shape(self):
        fc = jax.random.normal(KEY, (30, 45))  # 30 subjects, 45 edges
        qc = jax.random.uniform(KEY, (30,))
        loss = qcfc_loss(fc, qc)
        assert loss.shape == ()

    def test_nonnegative(self):
        fc = jax.random.normal(KEY, (30, 45))
        qc = jax.random.uniform(KEY, (30,))
        loss = qcfc_loss(fc, qc)
        assert loss >= -1e-6


class TestMultivariateKurtosis:
    def test_shape(self):
        ts = jax.random.normal(KEY, (5, 100))
        k = multivariate_kurtosis(ts)
        assert k.shape == ()

    def test_gaussian_near_expected(self):
        # For multivariate Gaussian, E[kurtosis] ≈ p(p+2) where p = dimension
        ts = jax.random.normal(KEY, (5, 10000))
        k = multivariate_kurtosis(ts)
        expected = 5.0 * 7.0  # p*(p+2) = 35
        np.testing.assert_allclose(k, expected, rtol=0.15)


# ---------------------------------------------------------------------------
# Variance-aware losses
# ---------------------------------------------------------------------------

class TestReliabilityWeightedLoss:
    def test_uniform_weights_equals_mean(self):
        losses = jax.random.normal(KEY, (20,)) ** 2
        weights = jnp.ones(20)
        result = reliability_weighted_loss(losses, weights)
        np.testing.assert_allclose(result, jnp.mean(losses), atol=1e-5)

    def test_zero_weight_ignored(self):
        losses = jnp.array([10.0, 0.1, 0.1])
        weights = jnp.array([0.0, 1.0, 1.0])
        result = reliability_weighted_loss(losses, weights)
        np.testing.assert_allclose(result, 0.1, atol=1e-5)
