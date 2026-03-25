import pytest
import jax
import jax.numpy as jnp
import numpy as np
from neurojax.geometry import spd

# Set to 64-bit to match analytical precision if needed, but 32 is standard JAX
# jax.config.update("jax_enable_x64", True)

def test_metrics_scalar():
    """Verify metrics on scaled identity matrices against analytical formulas."""
    n = 3
    a, b = 2.0, 8.0
    A = a * jnp.eye(n)
    B = b * jnp.eye(n)
    
    # Analytical distance
    expected = jnp.sqrt(n * jnp.log(b/a)**2)
    
    dist_r = spd.distance_riemann(A, B)
    dist_le = spd.distance_logeuclid(A, B)
    
    assert jnp.allclose(dist_r, expected, atol=1e-5), f"Riemann: {dist_r} vs {expected}"
    assert jnp.allclose(dist_le, expected, atol=1e-5), f"LogEuclid: {dist_le} vs {expected}"

def test_mappings_roundtrip():
    """Verify Tangent Space roundtrip C -> T -> C."""
    key = jax.random.PRNGKey(1)
    n = 3
    
    def rand_spd(k):
        tmp = jax.random.normal(k, (n, n))
        return tmp @ tmp.T + 0.1 * jnp.eye(n)
        
    k1, k2 = jax.random.split(key)
    C_ref = rand_spd(k1)
    C_target = rand_spd(k2)
    
    # Tangent space (returns vector)
    vec = spd.tangent_space(C_target, C_ref)
    
    expected_dim = n * (n + 1) // 2
    assert vec.shape == (expected_dim,)
    
    # Untangent (returns matrix)
    C_recon = spd.untangent_space(vec, C_ref)
    
    assert jnp.allclose(C_target, C_recon, atol=1e-5)

def test_means_geometric():
    """Verify means on scalar matrices match geometric mean."""
    n = 3
    k = 2.0
    A = k * jnp.eye(n)
    B = (k**3) * jnp.eye(n) # 8.0
    # Geometric mean: sqrt(2 * 8) = 4
    expected = (k**2) * jnp.eye(n) # 4.0
    
    covs = jnp.stack([A, B])
    
    mean_r = spd.mean_riemann(covs)
    mean_le = spd.mean_logeuclid(covs)
    
    assert jnp.allclose(mean_r, expected, atol=1e-5), f"Riemann Mean failed"
    # LogEuclidean mean of scaled identities is also geometric mean
    assert jnp.allclose(mean_le, expected, atol=1e-5), f"LogEuclid Mean failed"

def test_jit_vmap_compatibility():
    """Verify functions can be JIT compiled and VMAPped."""
    n = 3
    n_matrices = 10
    key = jax.random.PRNGKey(2)
    
    def rand_spd(k):
        tmp = jax.random.normal(k, (n, n))
        return tmp @ tmp.T + jnp.eye(n)
        
    keys = jax.random.split(key, n_matrices)
    covs = jax.vmap(rand_spd)(keys)
    C_ref = jnp.eye(n)
    
    # Test 1: vmap distance
    # Distance from each to Ref
    @jax.jit
    def batch_dist(cs, r):
        return jax.vmap(lambda c: spd.distance_riemann(c, r))(cs)
        
    dists = batch_dist(covs, C_ref)
    assert dists.shape == (n_matrices,)
    assert jnp.all(dists > 0)
    
    # Test 2: vmap tangent space is already built-in, check jit
    tangent_op = jax.jit(lambda c, r: spd.tangent_space(c, r))
    vecs = tangent_op(covs, C_ref)
    assert vecs.shape == (n_matrices, n * (n+1)//2)
    
    # Test 3: Mean (which uses while_loop) under JIT
    calculate_mean = jax.jit(spd.mean_riemann)
    m = calculate_mean(covs)
    assert m.shape == (n, n)
    # Check it is SPD (eigenvalues > 0)
    w, _ = jnp.linalg.eigh(m)
    assert jnp.all(w > 0)
    
def test_grad_compatibility():
    """Verify gradients propagate through geometric operations."""
    n = 3
    key = jax.random.PRNGKey(3)
    tmp = jax.random.normal(key, (n, n))
    A_base = tmp @ tmp.T + jnp.eye(n)
    
    # Define a scalar loss function: distance to Identity
    def loss(A):
        return spd.distance_riemann(A, jnp.eye(n))
        
    # Grad of loss
    grad_fn = jax.jit(jax.grad(loss))
    
    g = grad_fn(A_base)
    assert g.shape == (n, n)
    assert not jnp.any(jnp.isnan(g))
