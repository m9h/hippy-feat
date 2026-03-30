"""Tests for jaxoccoli.matrix module."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.matrix import (
    cholesky_invert,
    cone_project_spd,
    ensure_spd,
    mean_geom_spd,
    mean_logeuc_spd,
    sym2vec,
    symmetric,
    tangent_project_spd,
    toeplitz,
    vec2sym,
)

KEY = jax.random.PRNGKey(7)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def random_spd():
    """5x5 random SPD matrix."""
    A = jax.random.normal(KEY, (5, 5))
    return A @ A.T + 0.1 * jnp.eye(5)


@pytest.fixture
def batch_spd():
    """Batch of 8 random 4x4 SPD matrices."""
    keys = jax.random.split(KEY, 8)
    def _make(k):
        A = jax.random.normal(k, (4, 4))
        return A @ A.T + 0.1 * jnp.eye(4)
    return jax.vmap(_make)(keys)


# ---------------------------------------------------------------------------
# Matrix utilities
# ---------------------------------------------------------------------------

class TestSymmetric:
    def test_symmetric(self):
        A = jax.random.normal(KEY, (4, 4))
        S = symmetric(A)
        np.testing.assert_allclose(S, S.T, atol=1e-7)

    def test_skew_symmetric(self):
        A = jax.random.normal(KEY, (4, 4))
        S = symmetric(A, skew=True)
        np.testing.assert_allclose(S, -S.T, atol=1e-7)

    def test_already_symmetric(self, random_spd):
        S = symmetric(random_spd)
        np.testing.assert_allclose(S, random_spd, atol=1e-6)


class TestEnsureSpd:
    def test_positive_eigenvalues(self):
        A = jax.random.normal(KEY, (5, 5))
        A = A + A.T  # symmetric but not necessarily PD
        S = ensure_spd(A)
        eigvals = jnp.linalg.eigvalsh(S)
        assert jnp.all(eigvals > 0)

    def test_already_spd(self, random_spd):
        S = ensure_spd(random_spd)
        np.testing.assert_allclose(S, random_spd, atol=1e-5)

    def test_jit(self):
        A = jax.random.normal(KEY, (4, 4))
        A = A + A.T
        S = jax.jit(ensure_spd)(A)
        eigvals = jnp.linalg.eigvalsh(S)
        assert jnp.all(eigvals > 0)


class TestCholeskyInvert:
    def test_inverse(self, random_spd):
        inv = cholesky_invert(random_spd)
        eye = jnp.eye(5)
        np.testing.assert_allclose(random_spd @ inv, eye, atol=1e-4)

    def test_matches_linalg_inv(self, random_spd):
        inv1 = cholesky_invert(random_spd)
        inv2 = jnp.linalg.inv(random_spd)
        np.testing.assert_allclose(inv1, inv2, atol=1e-4)

    def test_grad(self, random_spd):
        def loss(X):
            return jnp.sum(cholesky_invert(X))
        g = jax.grad(loss)(random_spd)
        assert g.shape == random_spd.shape
        assert jnp.all(jnp.isfinite(g))


class TestSym2VecVec2Sym:
    def test_roundtrip_offset1(self, random_spd):
        S = symmetric(random_spd)
        v = sym2vec(S, offset=1)
        S2 = vec2sym(v, offset=1)
        # Off-diagonal should match; diagonal is zero for offset=1
        mask = 1 - jnp.eye(5)
        np.testing.assert_allclose(S2 * mask, S * mask, atol=1e-6)

    def test_roundtrip_offset0(self, random_spd):
        S = symmetric(random_spd)
        v = sym2vec(S, offset=0)
        S2 = vec2sym(v, offset=0)
        np.testing.assert_allclose(S2, S, atol=1e-6)

    def test_vector_length(self, random_spd):
        v1 = sym2vec(random_spd, offset=1)
        assert v1.shape == (10,)  # 5*4/2
        v0 = sym2vec(random_spd, offset=0)
        assert v0.shape == (15,)  # 5*6/2


class TestToeplitz:
    def test_basic(self):
        c = jnp.array([1.0, 2.0, 3.0])
        T = toeplitz(c)
        expected = jnp.array([[1, 2, 3], [2, 1, 2], [3, 2, 1]], dtype=float)
        np.testing.assert_allclose(T, expected, atol=1e-7)

    def test_asymmetric(self):
        c = jnp.array([1.0, 2.0, 3.0])
        r = jnp.array([1.0, 4.0, 5.0])
        T = toeplitz(c, r)
        assert T[0, 0] == 1.0
        assert T[1, 0] == 2.0
        assert T[0, 1] == 4.0


# ---------------------------------------------------------------------------
# SPD manifold operations
# ---------------------------------------------------------------------------

class TestTangentProject:
    def test_roundtrip(self, random_spd):
        ref = jnp.eye(5)
        T = tangent_project_spd(random_spd, ref)
        S = cone_project_spd(T, ref)
        np.testing.assert_allclose(S, random_spd, atol=1e-4)

    def test_identity_reference(self, random_spd):
        ref = jnp.eye(5)
        T = tangent_project_spd(random_spd, ref)
        # At identity, tangent projection = logm(X)
        eigvals = jnp.linalg.eigvalsh(random_spd)
        # Trace should equal sum of log eigenvalues
        np.testing.assert_allclose(
            jnp.trace(T), jnp.sum(jnp.log(eigvals)), atol=1e-4
        )

    def test_tangent_is_symmetric(self, random_spd):
        ref = jnp.eye(5)
        T = tangent_project_spd(random_spd, ref)
        np.testing.assert_allclose(T, T.T, atol=1e-5)

    def test_jit(self, random_spd):
        ref = jnp.eye(5)
        T = jax.jit(tangent_project_spd)(random_spd, ref)
        assert T.shape == (5, 5)

    def test_grad(self, random_spd):
        ref = jnp.eye(5)
        def loss(X):
            return jnp.sum(tangent_project_spd(X, ref) ** 2)
        g = jax.grad(loss)(random_spd)
        assert jnp.all(jnp.isfinite(g))


class TestMeanLogEuc:
    def test_single_matrix(self, random_spd):
        inputs = random_spd[None, ...]  # (1, 5, 5)
        m = mean_logeuc_spd(inputs)
        np.testing.assert_allclose(m, random_spd, atol=1e-4)

    def test_identical_matrices(self, random_spd):
        inputs = jnp.stack([random_spd] * 4)
        m = mean_logeuc_spd(inputs)
        np.testing.assert_allclose(m, random_spd, atol=1e-4)

    def test_spd_output(self, batch_spd):
        m = mean_logeuc_spd(batch_spd)
        eigvals = jnp.linalg.eigvalsh(m)
        assert jnp.all(eigvals > 0)

    def test_shape(self, batch_spd):
        m = mean_logeuc_spd(batch_spd)
        assert m.shape == (4, 4)


class TestMeanGeom:
    def test_identical_matrices(self, random_spd):
        inputs = jnp.stack([random_spd] * 4)
        m = mean_geom_spd(inputs)
        np.testing.assert_allclose(m, random_spd, atol=1e-3)

    def test_spd_output(self, batch_spd):
        m = mean_geom_spd(batch_spd, max_iter=15)
        eigvals = jnp.linalg.eigvalsh(m)
        assert jnp.all(eigvals > 0)

    def test_shape(self, batch_spd):
        m = mean_geom_spd(batch_spd)
        assert m.shape == (4, 4)

    def test_two_matrices(self):
        """Geometric mean of two SPD matrices: M = A^{1/2} (A^{-1/2} B A^{-1/2})^{1/2} A^{1/2}."""
        k1, k2 = jax.random.split(KEY)
        A_raw = jax.random.normal(k1, (3, 3))
        B_raw = jax.random.normal(k2, (3, 3))
        A = A_raw @ A_raw.T + 0.1 * jnp.eye(3)
        B = B_raw @ B_raw.T + 0.1 * jnp.eye(3)
        inputs = jnp.stack([A, B])
        m = mean_geom_spd(inputs, max_iter=20)
        # Check symmetry
        np.testing.assert_allclose(m, m.T, atol=1e-4)
        # Check SPD
        eigvals = jnp.linalg.eigvalsh(m)
        assert jnp.all(eigvals > 0)
