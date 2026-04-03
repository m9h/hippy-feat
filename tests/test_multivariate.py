"""Tests for jaxoccoli.multivariate and massive_mv_cmi."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxoccoli.multivariate import pca, cca, mv_cmi
from jaxoccoli.connectivity import massive_mv_cmi, MVCMIParams

KEY = jax.random.PRNGKey(42)

def test_pca_shapes():
    T, V = 100, 50
    data = jax.random.normal(KEY, (T, V))
    # 95% variance
    projected, k = pca(data, n_components=0.95, max_k=30)
    assert projected.shape == (100, 30)
    assert k > 0
    assert k <= 30
    # Check that components beyond k are zeroed out
    if k < 30:
        assert jnp.all(projected[:, k:] == 0)

def test_pca_variance_retention():
    # Create data with 5 strong components
    T, V = 200, 100
    k1, k2 = jax.random.split(KEY)
    latent = jax.random.normal(k1, (T, 5))
    weights = jax.random.normal(k2, (5, V))
    data = latent @ weights + 0.1 * jax.random.normal(KEY, (T, V))
    
    projected, k = pca(data, n_components=0.9, max_k=20)
    # Should retain roughly 5 components
    assert 4 <= k <= 7

def test_cca_recovers_correlation():
    T = 500
    k1, k2, k3 = jax.random.split(KEY, 3)
    # Latent shared variable
    z = jax.random.normal(k1, (T, 1))
    x = z + 0.1 * jax.random.normal(k2, (T, 1))
    y = z + 0.1 * jax.random.normal(k3, (T, 1))
    
    rho = cca(x, y)
    assert rho.shape == (1,)
    assert rho[0] > 0.9

def test_mv_cmi_simple():
    T = 1000
    k1, k2, k3, k4 = jax.random.split(KEY, 4)
    # Z is the common driver
    z = jax.random.normal(k1, (T, 2))
    # X and Y are driven by Z but independent given Z
    x = z @ jax.random.normal(k2, (2, 2)) + 0.1 * jax.random.normal(k3, (T, 2))
    y = z @ jax.random.normal(k4, (2, 2)) + 0.1 * jax.random.normal(KEY, (T, 2))
    
    # Conditional MI should be low because Z explains the relationship
    cmi = mv_cmi(x, y, z)
    assert cmi < 0.1
    
    # Marginal MI (without Z) should be high
    # We can test this by providing a zero Z or random Z
    z_rand = jax.random.normal(KEY, (T, 2))
    cmi_high = mv_cmi(x, y, z_rand)
    assert cmi_high > 1.0

def test_massive_mv_cmi_consistency():
    """Test that massive_mv_cmi matches the direct mv_cmi implementation."""
    T = 200
    n_parcels = 5
    max_k = 3

    # Create concatenated projections (T, n_parcels * max_k)
    total_dim = n_parcels * max_k
    projections = jax.random.normal(KEY, (T, total_dim))

    params = MVCMIParams(
        projections=projections,
        n_parcels=n_parcels,
        max_k=max_k
    )

    # Compute massive CMI matrix
    cmi_matrix = massive_mv_cmi(params)
    assert cmi_matrix.shape == (5, 5)
    assert jnp.all(jnp.diag(cmi_matrix) == 0.0)

    # Check one off-diagonal element (0, 1) conditioning on (2, 3, 4)
    # Direct CMI:
    x = projections[:, 0:3]
    y = projections[:, 3:6]
    rest = projections[:, 6:]

    direct_val = mv_cmi(x, y, rest)

    # Use a small tolerance due to regularization in sub-blocks
    np.testing.assert_allclose(cmi_matrix[0, 1], direct_val, atol=1e-2)
    # Symmetry
    np.testing.assert_allclose(cmi_matrix[0, 1], cmi_matrix[1, 0], rtol=1e-5)

