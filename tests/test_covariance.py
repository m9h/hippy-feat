"""Tests for jaxoccoli.covariance module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.covariance import (
    attenuated_corr,
    conditional_cov,
    corr,
    cov,
    paired_cov,
    partial_corr,
    posterior_corr,
    precision,
    weighted_corr,
)

KEY = jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_data():
    """(5 variables, 100 observations) random data."""
    return jax.random.normal(KEY, (5, 100))


@pytest.fixture
def two_sets():
    """Two correlated variable sets for cross-covariance tests."""
    k1, k2 = jax.random.split(KEY)
    X = jax.random.normal(k1, (3, 80))
    Y = X[:2] + 0.1 * jax.random.normal(k2, (2, 80))
    return X, Y


# ---------------------------------------------------------------------------
# Core covariance
# ---------------------------------------------------------------------------

class TestCov:
    def test_shape(self, random_data):
        S = cov(random_data)
        assert S.shape == (5, 5)

    def test_symmetry(self, random_data):
        S = cov(random_data)
        np.testing.assert_allclose(S, S.T, atol=1e-6)

    def test_matches_numpy(self, random_data):
        S = cov(random_data)
        S_np = np.cov(np.array(random_data))
        np.testing.assert_allclose(S, S_np, atol=1e-5)

    def test_bias(self, random_data):
        S = cov(random_data, bias=True)
        S_np = np.cov(np.array(random_data), bias=True)
        np.testing.assert_allclose(S, S_np, atol=1e-5)

    def test_l2_regularisation(self, random_data):
        S0 = cov(random_data)
        S_reg = cov(random_data, l2=1.0)
        diag_diff = jnp.diagonal(S_reg) - jnp.diagonal(S0)
        np.testing.assert_allclose(diag_diff, 1.0, atol=1e-5)

    def test_rowvar_false(self, random_data):
        X_T = random_data.T  # (100, 5)
        S1 = cov(random_data, rowvar=True)
        S2 = cov(X_T, rowvar=False)
        np.testing.assert_allclose(S1, S2, atol=1e-5)

    def test_weighted(self, random_data):
        w = jnp.ones(100)
        S1 = cov(random_data)
        S2 = cov(random_data, weight=w)
        np.testing.assert_allclose(S1, S2, atol=1e-4)

    def test_jit(self, random_data):
        S = jax.jit(cov)(random_data)
        assert S.shape == (5, 5)

    def test_grad(self, random_data):
        def loss(X):
            return jnp.sum(cov(X) ** 2)
        g = jax.grad(loss)(random_data)
        assert g.shape == random_data.shape
        assert jnp.all(jnp.isfinite(g))

    def test_vmap(self):
        batch = jax.random.normal(KEY, (10, 5, 50))
        S = jax.vmap(cov)(batch)
        assert S.shape == (10, 5, 5)


class TestCorr:
    def test_diagonal_ones(self, random_data):
        R = corr(random_data)
        np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-6)

    def test_bounded(self, random_data):
        R = corr(random_data)
        assert jnp.all(R >= -1.0 - 1e-6)
        assert jnp.all(R <= 1.0 + 1e-6)

    def test_matches_numpy(self, random_data):
        R = corr(random_data)
        R_np = np.corrcoef(np.array(random_data))
        np.testing.assert_allclose(R, R_np, atol=1e-5)


class TestPrecision:
    def test_inverse_of_cov(self, random_data):
        S = cov(random_data, l2=0.01)
        P = precision(random_data, l2=0.01)
        eye = jnp.eye(5)
        np.testing.assert_allclose(S @ P, eye, atol=1e-4)


class TestPartialCorr:
    def test_diagonal_ones(self, random_data):
        pc = partial_corr(random_data)
        np.testing.assert_allclose(jnp.diagonal(pc), 1.0, atol=1e-5)

    def test_bounded(self, random_data):
        pc = partial_corr(random_data)
        assert jnp.all(pc >= -1.0 - 1e-5)
        assert jnp.all(pc <= 1.0 + 1e-5)

    def test_shape(self, random_data):
        pc = partial_corr(random_data)
        assert pc.shape == (5, 5)


# ---------------------------------------------------------------------------
# Cross-covariance and conditional
# ---------------------------------------------------------------------------

class TestPairedCov:
    def test_shape(self, two_sets):
        X, Y = two_sets
        S = paired_cov(X, Y)
        assert S.shape == (3, 2)

    def test_self_paired_equals_cov(self, random_data):
        S1 = cov(random_data)
        S2 = paired_cov(random_data, random_data)
        np.testing.assert_allclose(S1, S2, atol=1e-5)


class TestConditionalCov:
    def test_shape(self, two_sets):
        X, Y = two_sets
        Sc = conditional_cov(X, Y)
        assert Sc.shape == (3, 3)

    def test_smaller_than_marginal(self, two_sets):
        X, Y = two_sets
        Sxx = cov(X)
        Sc = conditional_cov(X, Y)
        # Conditional variance <= marginal variance
        assert jnp.all(jnp.diagonal(Sc) <= jnp.diagonal(Sxx) + 1e-5)


# ---------------------------------------------------------------------------
# Variance-aware extensions
# ---------------------------------------------------------------------------

class TestWeightedCorr:
    def test_uniform_weights_match_corr(self, random_data):
        w = jnp.ones(100)
        R1 = corr(random_data)
        R2 = weighted_corr(random_data, w)
        np.testing.assert_allclose(R1, R2, atol=1e-4)

    def test_shape(self, random_data):
        w = jax.random.uniform(KEY, (100,)) + 0.1
        R = weighted_corr(random_data, w)
        assert R.shape == (5, 5)


class TestAttenuatedCorr:
    def test_perfect_reliability_unchanged(self, random_data):
        rel = jnp.ones(5)
        R1 = corr(random_data)
        R2 = attenuated_corr(random_data, rel)
        np.testing.assert_allclose(R1, R2, atol=1e-6)

    def test_correction_inflates(self, random_data):
        rel = 0.5 * jnp.ones(5)
        R_obs = corr(random_data)
        R_corr = attenuated_corr(random_data, rel)
        # Off-diagonal magnitudes should be larger after correction
        mask = 1 - jnp.eye(5)
        assert jnp.mean(jnp.abs(R_corr) * mask) >= jnp.mean(jnp.abs(R_obs) * mask) - 1e-6

    def test_bounded(self, random_data):
        rel = 0.3 * jnp.ones(5)
        R = attenuated_corr(random_data, rel)
        assert jnp.all(R >= -1.0 - 1e-6)
        assert jnp.all(R <= 1.0 + 1e-6)


class TestPosteriorCorr:
    def test_shape(self):
        k1, k2 = jax.random.split(KEY)
        beta_mean = jax.random.normal(k1, (5, 60))
        beta_var = jnp.abs(jax.random.normal(k2, (5, 60))) + 0.01
        R = posterior_corr(beta_mean, beta_var)
        assert R.shape == (5, 5)

    def test_diagonal_ones(self):
        k1, k2 = jax.random.split(KEY)
        beta_mean = jax.random.normal(k1, (5, 60))
        beta_var = jnp.abs(jax.random.normal(k2, (5, 60))) + 0.01
        R = posterior_corr(beta_mean, beta_var)
        np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-5)

    def test_zero_variance_matches_corr(self):
        beta_mean = jax.random.normal(KEY, (5, 60))
        beta_var = jnp.zeros((5, 60))
        R1 = corr(beta_mean)
        R2 = posterior_corr(beta_mean, beta_var)
        np.testing.assert_allclose(R1, R2, atol=1e-5)

    def test_high_variance_inflates(self):
        k1, k2 = jax.random.split(KEY)
        beta_mean = jax.random.normal(k1, (5, 60))
        beta_var_low = 0.01 * jnp.ones((5, 60))
        beta_var_high = 10.0 * jnp.ones((5, 60))
        R_low = posterior_corr(beta_mean, beta_var_low)
        R_high = posterior_corr(beta_mean, beta_var_high)
        # High variance should produce larger correction (larger magnitudes)
        mask = 1 - jnp.eye(5)
        assert jnp.mean(jnp.abs(R_high) * mask) >= jnp.mean(jnp.abs(R_low) * mask) - 1e-4

    def test_jit(self):
        k1, k2 = jax.random.split(KEY)
        bm = jax.random.normal(k1, (5, 60))
        bv = jnp.abs(jax.random.normal(k2, (5, 60))) + 0.01
        R = jax.jit(posterior_corr)(bm, bv)
        assert R.shape == (5, 5)

    def test_grad(self):
        k1, k2 = jax.random.split(KEY)
        bm = jax.random.normal(k1, (5, 60))
        bv = jnp.abs(jax.random.normal(k2, (5, 60))) + 0.01

        def loss(bm, bv):
            return jnp.sum(posterior_corr(bm, bv) ** 2)

        g_mean, g_var = jax.grad(loss, argnums=(0, 1))(bm, bv)
        assert g_mean.shape == bm.shape
        assert g_var.shape == bv.shape
        assert jnp.all(jnp.isfinite(g_mean))
        assert jnp.all(jnp.isfinite(g_var))
