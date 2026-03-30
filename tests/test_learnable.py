"""Tests for jaxoccoli.learnable module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.learnable import (
    init_butterworth_spectrum,
    init_ideal_spectrum,
    make_atlas_linear,
    make_atlas_linear_uncertain,
    make_freq_filter,
    make_learnable_cov,
    make_orthogonal_constraint,
    make_simplex_constraint,
    make_spd_constraint,
)

KEY = jax.random.PRNGKey(13)


# ---------------------------------------------------------------------------
# Constraint factories
# ---------------------------------------------------------------------------

class TestSimplexConstraint:
    def test_sums_to_one(self):
        c = make_simplex_constraint()
        x = jax.random.normal(KEY, (10,))
        p = c.project(x)
        np.testing.assert_allclose(jnp.sum(p), 1.0, atol=1e-6)

    def test_roundtrip(self):
        c = make_simplex_constraint()
        x = jax.random.normal(KEY, (10,))
        p = c.project(x)
        x2 = c.unproject(p)
        p2 = c.project(x2)
        np.testing.assert_allclose(p, p2, atol=1e-5)

    def test_nonnegative(self):
        c = make_simplex_constraint()
        x = jax.random.normal(KEY, (10,))
        p = c.project(x)
        assert jnp.all(p >= 0)

    def test_temperature(self):
        c_hot = make_simplex_constraint(temperature=10.0)
        c_cold = make_simplex_constraint(temperature=0.1)
        x = jax.random.normal(KEY, (10,))
        p_hot = c_hot.project(x)
        p_cold = c_cold.project(x)
        # Hot: more uniform; Cold: more peaked
        assert jnp.max(p_hot) < jnp.max(p_cold)


class TestSpdConstraint:
    def test_positive_eigenvalues(self):
        c = make_spd_constraint()
        X = jax.random.normal(KEY, (5, 5))
        S = c.project(X)
        eigvals = jnp.linalg.eigvalsh(S)
        assert jnp.all(eigvals > 0)

    def test_symmetric(self):
        c = make_spd_constraint()
        X = jax.random.normal(KEY, (5, 5))
        S = c.project(X)
        np.testing.assert_allclose(S, S.T, atol=1e-6)


class TestOrthogonalConstraint:
    def test_orthogonal(self):
        c = make_orthogonal_constraint()
        X = jax.random.normal(KEY, (5, 3))
        Q = c.project(X)
        np.testing.assert_allclose(Q.T @ Q, jnp.eye(3), atol=1e-5)

    def test_shape(self):
        c = make_orthogonal_constraint()
        X = jax.random.normal(KEY, (10, 4))
        Q = c.project(X)
        assert Q.shape == (10, 4)


# ---------------------------------------------------------------------------
# Atlas factories
# ---------------------------------------------------------------------------

class TestAtlasLinear:
    def test_output_shape(self):
        params, fwd = make_atlas_linear(100, 10, key=KEY)
        data = jax.random.normal(KEY, (100, 50))
        out = fwd(params, data)
        assert out.shape == (10, 50)

    def test_batch(self):
        params, fwd = make_atlas_linear(100, 10, key=KEY)
        data = jax.random.normal(KEY, (8, 100, 50))
        out = fwd(params, data)
        assert out.shape == (8, 10, 50)

    def test_grad(self):
        params, fwd = make_atlas_linear(50, 5, key=KEY)
        data = jax.random.normal(KEY, (50, 30))

        def loss(params):
            return jnp.sum(fwd(params, data) ** 2)

        g = jax.grad(loss)(params)
        assert g.weight.shape == params.weight.shape
        assert jnp.all(jnp.isfinite(g.weight))

    def test_jit(self):
        params, fwd = make_atlas_linear(50, 5, key=KEY)
        data = jax.random.normal(KEY, (50, 30))
        out = jax.jit(fwd)(params, data)
        assert out.shape == (5, 30)

    def test_weights_sum_to_one(self):
        params, _ = make_atlas_linear(50, 5, key=KEY)
        w = jax.nn.softmax(params.weight, axis=-1)
        np.testing.assert_allclose(jnp.sum(w, axis=-1), 1.0, atol=1e-5)


class TestAtlasLinearUncertain:
    def test_output_shapes(self):
        params, fwd = make_atlas_linear_uncertain(100, 10, key=KEY)
        k1, k2 = jax.random.split(KEY)
        bm = jax.random.normal(k1, (100, 50))
        bv = jnp.abs(jax.random.normal(k2, (100, 50))) + 0.01
        pm, pv = fwd(params, bm, bv)
        assert pm.shape == (10, 50)
        assert pv.shape == (10, 50)

    def test_variance_nonnegative(self):
        params, fwd = make_atlas_linear_uncertain(100, 10, key=KEY)
        k1, k2 = jax.random.split(KEY)
        bm = jax.random.normal(k1, (100, 50))
        bv = jnp.abs(jax.random.normal(k2, (100, 50))) + 0.01
        _, pv = fwd(params, bm, bv)
        assert jnp.all(pv >= 0)

    def test_zero_variance_zero_output_var(self):
        params, fwd = make_atlas_linear_uncertain(50, 5, key=KEY)
        bm = jax.random.normal(KEY, (50, 30))
        bv = jnp.zeros((50, 30))
        _, pv = fwd(params, bm, bv)
        np.testing.assert_allclose(pv, 0.0, atol=1e-7)

    def test_mean_matches_standard(self):
        k = KEY
        params_u, fwd_u = make_atlas_linear_uncertain(50, 5, key=k)
        params_s, fwd_s = make_atlas_linear(50, 5, key=k)
        bm = jax.random.normal(KEY, (50, 30))
        bv = jnp.abs(jax.random.normal(KEY, (50, 30))) + 0.01
        pm_u, _ = fwd_u(params_u, bm, bv)
        pm_s = fwd_s(params_s, bm)
        np.testing.assert_allclose(pm_u, pm_s, atol=1e-5)

    def test_grad(self):
        params, fwd = make_atlas_linear_uncertain(50, 5, key=KEY)
        k1, k2 = jax.random.split(KEY)
        bm = jax.random.normal(k1, (50, 30))
        bv = jnp.abs(jax.random.normal(k2, (50, 30))) + 0.01

        def loss(params):
            pm, pv = fwd(params, bm, bv)
            return jnp.sum(pm ** 2) + jnp.sum(pv)

        g = jax.grad(loss)(params)
        assert jnp.all(jnp.isfinite(g.weight))


# ---------------------------------------------------------------------------
# Learnable covariance
# ---------------------------------------------------------------------------

class TestLearnableCov:
    def test_output_shape(self):
        params, fwd = make_learnable_cov(10, 50, key=KEY)
        data = jax.random.normal(KEY, (10, 50))
        out = fwd(params, data)
        assert out.shape == (10, 10)

    def test_symmetric(self):
        params, fwd = make_learnable_cov(10, 50, key=KEY, estimator='corr')
        data = jax.random.normal(KEY, (10, 50))
        out = fwd(params, data)
        np.testing.assert_allclose(out, out.T, atol=1e-5)

    def test_grad(self):
        params, fwd = make_learnable_cov(10, 50, key=KEY)
        data = jax.random.normal(KEY, (10, 50))

        def loss(params):
            return jnp.sum(fwd(params, data) ** 2)

        g = jax.grad(loss)(params)
        assert g.weight.shape == params.weight.shape
        assert jnp.all(jnp.isfinite(g.weight))


# ---------------------------------------------------------------------------
# Frequency filter
# ---------------------------------------------------------------------------

class TestFreqFilter:
    def test_allpass_identity(self):
        params, fwd = make_freq_filter(51, key=KEY)
        data = jax.random.normal(KEY, (100,))
        out = fwd(params, data)
        np.testing.assert_allclose(out, data, atol=1e-5)

    def test_output_shape_single(self):
        params, fwd = make_freq_filter(51, key=KEY, n_filters=1)
        data = jax.random.normal(KEY, (100,))
        out = fwd(params, data)
        assert out.shape == (100,)

    def test_output_shape_multi(self):
        params, fwd = make_freq_filter(51, key=KEY, n_filters=3)
        data = jax.random.normal(KEY, (100,))
        out = fwd(params, data)
        assert out.shape == (3, 100)

    def test_grad(self):
        params, fwd = make_freq_filter(51, key=KEY)
        data = jax.random.normal(KEY, (100,))

        def loss(params):
            return jnp.sum(fwd(params, data) ** 2)

        g = jax.grad(loss)(params)
        assert g.transfer_fn.shape == params.transfer_fn.shape
        assert jnp.all(jnp.isfinite(g.transfer_fn))

    def test_jit(self):
        params, fwd = make_freq_filter(51, key=KEY)
        data = jax.random.normal(KEY, (100,))
        out = jax.jit(fwd)(params, data)
        assert out.shape == (100,)


# ---------------------------------------------------------------------------
# Filter initialisation
# ---------------------------------------------------------------------------

class TestFilterInit:
    def test_ideal_bandpass(self):
        tf = init_ideal_spectrum(51, low=0.01, high=0.1, fs=1.0)
        assert tf.shape == (51,)
        assert tf[0] == 0.0  # DC blocked by highpass
        # Some frequencies should pass
        assert jnp.sum(tf) > 0

    def test_ideal_lowpass(self):
        tf = init_ideal_spectrum(51, high=0.1, fs=1.0)
        assert tf[0] == 1.0  # DC passes

    def test_butterworth(self):
        tf = init_butterworth_spectrum(51, order=4, low=0.01, high=0.1, fs=1.0)
        assert tf.shape == (51,)
        # Should be smooth, not brick-wall
        assert jnp.all(tf >= 0)
        assert jnp.all(tf <= 1.0 + 1e-6)
