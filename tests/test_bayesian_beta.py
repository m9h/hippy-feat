"""Tests for jaxoccoli.bayesian_beta module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.bayesian_beta import (
    make_ar1_conjugate_glm,
    make_conjugate_glm,
    make_conjugate_glm_vmap,
)

KEY = jax.random.PRNGKey(31)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_glm():
    """Simple GLM: 100 timepoints, 3 regressors, known betas."""
    T, P = 100, 3
    k1, k2, k3 = jax.random.split(KEY, 3)
    X = jax.random.normal(k1, (T, P))
    true_beta = jnp.array([2.0, -1.0, 0.5])
    sigma = 0.5
    noise = sigma * jax.random.normal(k2, (T,))
    y = X @ true_beta + noise
    return X, y, true_beta, sigma


@pytest.fixture
def multi_voxel_data():
    """50 voxels, 100 timepoints, 3 regressors."""
    T, P, V = 100, 3, 50
    k1, k2, k3 = jax.random.split(KEY, 3)
    X = jax.random.normal(k1, (T, P))
    betas = jax.random.normal(k2, (V, P))
    noise = 0.5 * jax.random.normal(k3, (V, T))
    Y = jax.vmap(lambda b: X @ b)(betas) + noise
    return X, Y, betas


# ---------------------------------------------------------------------------
# Conjugate GLM
# ---------------------------------------------------------------------------

class TestConjugateGLM:
    def test_output_shapes(self, simple_glm):
        X, y, _, _ = simple_glm
        params, fwd = make_conjugate_glm(X)
        beta_mean, beta_var, sigma2 = fwd(params, y)
        assert beta_mean.shape == (3,)
        assert beta_var.shape == (3,)
        assert sigma2.shape == ()

    def test_recovers_true_beta(self, simple_glm):
        X, y, true_beta, _ = simple_glm
        params, fwd = make_conjugate_glm(X)
        beta_mean, _, _ = fwd(params, y)
        # Should be close to true beta (100 obs, low noise)
        np.testing.assert_allclose(beta_mean, true_beta, atol=0.3)

    def test_variance_positive(self, simple_glm):
        X, y, _, _ = simple_glm
        params, fwd = make_conjugate_glm(X)
        _, beta_var, sigma2 = fwd(params, y)
        assert jnp.all(beta_var > 0)
        assert sigma2 > 0

    def test_sigma2_reasonable(self, simple_glm):
        X, y, _, true_sigma = simple_glm
        params, fwd = make_conjugate_glm(X)
        _, _, sigma2 = fwd(params, y)
        # Should be in the right ballpark of true_sigma^2 = 0.25
        np.testing.assert_allclose(sigma2, true_sigma ** 2, rtol=1.0)

    def test_strong_prior_shrinks(self, simple_glm):
        X, y, _, _ = simple_glm
        # Weak prior
        params_weak, fwd_weak = make_conjugate_glm(X, prior_precision=0.01 * jnp.eye(3))
        beta_weak, _, _ = fwd_weak(params_weak, y)
        # Strong prior toward zero
        params_strong, fwd_strong = make_conjugate_glm(
            X, prior_precision=100.0 * jnp.eye(3), prior_mean=jnp.zeros(3)
        )
        beta_strong, _, _ = fwd_strong(params_strong, y)
        # Strong prior should produce betas closer to zero
        assert jnp.sum(beta_strong ** 2) < jnp.sum(beta_weak ** 2)

    def test_jit(self, simple_glm):
        X, y, _, _ = simple_glm
        params, fwd = make_conjugate_glm(X)
        beta_mean, beta_var, sigma2 = jax.jit(fwd)(params, y)
        assert jnp.all(jnp.isfinite(beta_mean))
        assert jnp.all(jnp.isfinite(beta_var))

    def test_grad(self, simple_glm):
        """Verify gradient flows through the conjugate GLM.
        (Useful for learning the prior or design matrix.)"""
        X, y, _, _ = simple_glm
        params, fwd = make_conjugate_glm(X)

        def loss(y):
            bm, bv, _ = fwd(params, y)
            return jnp.sum(bm ** 2)

        g = jax.grad(loss)(y)
        assert g.shape == y.shape
        assert jnp.all(jnp.isfinite(g))


class TestConjugateGLMVmap:
    def test_output_shapes(self, multi_voxel_data):
        X, Y, _ = multi_voxel_data
        params, fwd = make_conjugate_glm_vmap(X)
        bm, bv, s2 = fwd(params, Y)
        assert bm.shape == (50, 3)
        assert bv.shape == (50, 3)
        assert s2.shape == (50,)

    def test_matches_single_voxel(self, multi_voxel_data):
        X, Y, _ = multi_voxel_data
        params_v, fwd_v = make_conjugate_glm_vmap(X)
        params_s, fwd_s = make_conjugate_glm(X)

        bm_v, bv_v, s2_v = fwd_v(params_v, Y)
        bm_s, bv_s, s2_s = fwd_s(params_s, Y[0])
        np.testing.assert_allclose(bm_v[0], bm_s, atol=1e-5)
        np.testing.assert_allclose(bv_v[0], bv_s, atol=1e-5)

    def test_jit(self, multi_voxel_data):
        X, Y, _ = multi_voxel_data
        params, fwd = make_conjugate_glm_vmap(X)
        bm, _, _ = jax.jit(fwd)(params, Y)
        assert jnp.all(jnp.isfinite(bm))


# ---------------------------------------------------------------------------
# AR(1) prewhitened conjugate GLM
# ---------------------------------------------------------------------------

class TestAR1ConjugateGLM:
    def test_output_shapes(self, simple_glm):
        X, y, _, _ = simple_glm
        params, fwd = make_ar1_conjugate_glm(X)
        beta_mean, beta_var, sigma2, rho = fwd(params, y)
        assert beta_mean.shape == (3,)
        assert beta_var.shape == (3,)
        assert sigma2.shape == ()
        assert rho.shape == ()

    def test_rho_bounded(self, simple_glm):
        X, y, _, _ = simple_glm
        params, fwd = make_ar1_conjugate_glm(X)
        _, _, _, rho = fwd(params, y)
        assert jnp.abs(rho) < 1.0

    def test_recovers_beta_with_ar_noise(self):
        """GLM with AR(1) noise should still recover betas."""
        T, P = 200, 2
        k1, k2, k3 = jax.random.split(KEY, 3)
        X = jax.random.normal(k1, (T, P))
        true_beta = jnp.array([3.0, -2.0])
        true_rho = 0.7
        # Generate AR(1) noise
        innovations = 0.5 * jax.random.normal(k2, (T,))
        noise = jnp.zeros(T)
        def _ar_step(carry, inn):
            prev = carry
            curr = true_rho * prev + inn
            return curr, curr
        _, noise = jax.lax.scan(_ar_step, 0.0, innovations)
        y = X @ true_beta + noise

        params, fwd = make_ar1_conjugate_glm(X)
        beta_mean, _, _, rho_est = fwd(params, y)
        np.testing.assert_allclose(beta_mean, true_beta, atol=0.5)
        # rho should be roughly correct
        np.testing.assert_allclose(rho_est, true_rho, atol=0.3)

    def test_jit(self, simple_glm):
        X, y, _, _ = simple_glm
        params, fwd = make_ar1_conjugate_glm(X)
        bm, bv, s2, rho = jax.jit(fwd)(params, y)
        assert jnp.all(jnp.isfinite(bm))


# ---------------------------------------------------------------------------
# End-to-end variance propagation
# ---------------------------------------------------------------------------

class TestVariancePropagation:
    """Integration test: conjugate GLM -> uncertain atlas -> posterior_corr."""

    def test_full_pipeline(self, multi_voxel_data):
        from jaxoccoli.learnable import make_atlas_linear_uncertain
        from jaxoccoli.covariance import posterior_corr

        X, Y, _ = multi_voxel_data  # Y: (50 voxels, 100 timepoints)

        # Step 1: Bayesian beta estimation
        params_glm, fwd_glm = make_conjugate_glm_vmap(X)
        beta_means, beta_vars, _ = fwd_glm(params_glm, Y)
        # beta_means: (50, 3), beta_vars: (50, 3)
        # Atlas expects (n_voxels, T) where T = n_params here
        V, P = beta_means.shape

        # Step 2: Variance-aware parcellation (50 voxels -> 5 parcels)
        params_atlas, fwd_atlas = make_atlas_linear_uncertain(V, 5, key=KEY)
        parc_mean, parc_var = fwd_atlas(params_atlas, beta_means, beta_vars)
        # parc_mean: (5, 3), parc_var: (5, 3)
        assert parc_mean.shape == (5, P)
        assert parc_var.shape == (5, P)

        # Step 3: Posterior correlation
        fc = posterior_corr(parc_mean, parc_var)
        assert fc.shape == (5, 5)
        np.testing.assert_allclose(jnp.diagonal(fc), 1.0, atol=1e-5)
        assert jnp.all(jnp.isfinite(fc))

    def test_gradient_flows_end_to_end(self, multi_voxel_data):
        """Verify gradients propagate through the full pipeline."""
        from jaxoccoli.learnable import make_atlas_linear_uncertain
        from jaxoccoli.covariance import posterior_corr

        X, Y, _ = multi_voxel_data
        params_glm, fwd_glm = make_conjugate_glm_vmap(X)
        params_atlas, fwd_atlas = make_atlas_linear_uncertain(50, 5, key=KEY)

        def pipeline_loss(atlas_params):
            bm, bv, _ = fwd_glm(params_glm, Y)
            pm, pv = fwd_atlas(atlas_params, bm, bv)
            fc = posterior_corr(pm, pv)
            return jnp.sum(fc ** 2)

        g = jax.grad(pipeline_loss)(params_atlas)
        assert g.weight.shape == params_atlas.weight.shape
        assert jnp.all(jnp.isfinite(g.weight))
