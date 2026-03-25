import jax
import jax.numpy as jnp
import numpy as np
import pytest
from neurojax.ml.riemannian.classifiers import MDM, TangentSpaceLR

# Enable double precision if needed
# jax.config.update("jax_enable_x64", True)

def generate_synthetic_data(n_per_class=20, size=3):
    """
    Generate synthetic SPD data for 2 classes.
    Class 0: Centered at Identity
    Class 1: Centered at 2*Identity
    """
    key = jax.random.PRNGKey(42)
    k1, k2, k3 = jax.random.split(key, 3)
    
    def sample_spd(key, center_scale):
        # A = U S U^T, randomize U
        # Simple generation: A A^T
        A = jax.random.normal(key, (size, size))
        cov = A @ A.T + 0.1 * jnp.eye(size)
        # Shift mean
        return cov * center_scale

    # Class 0
    X0 = jax.vmap(lambda k: sample_spd(k, 1.0))(jax.random.split(k1, n_per_class))
    y0 = jnp.zeros(n_per_class, dtype=int)
    
    # Class 1 (Scaled up)
    X1 = jax.vmap(lambda k: sample_spd(k, 5.0))(jax.random.split(k2, n_per_class))
    y1 = jnp.ones(n_per_class, dtype=int)
    
    X = jnp.concatenate([X0, X1])
    y = jnp.concatenate([y0, y1])
    return X, y

def test_mdm_classification():
    X, y = generate_synthetic_data(n_per_class=20)
    
    mdm = MDM(metric='riemann')
    mdm.fit(X, y)
    
    # Check fitted attributes
    assert hasattr(mdm, 'classes_')
    assert len(mdm.classes_) == 2
    assert mdm.covmeans_.shape == (2, 3, 3)
    
    # Predict
    preds = mdm.predict(X)
    acc = jnp.mean(preds == y)
    assert acc > 0.8 # Should be easily separable

def test_mdm_logeuclid():
    X, y = generate_synthetic_data()
    mdm = MDM(metric='logeuclid')
    mdm.fit(X, y)
    preds = mdm.predict(X)
    assert preds.shape == y.shape

def test_tslr_classification():
    X, y = generate_synthetic_data(n_per_class=20)
    
    tslr = TangentSpaceLR(metric='riemann')
    tslr.fit(X, y)
    
    assert hasattr(tslr, 'reference_mean_')
    assert tslr.reference_mean_.shape == (3, 3)
    
    preds = tslr.predict(X)
    acc = jnp.mean(preds == y)
    assert acc > 0.8

def test_sklearn_pipeline_compatibility():
    """Verify it can be part of a pipeline (conceptually)."""
    from sklearn.pipeline import Pipeline
    X, y = generate_synthetic_data()
    
    # Convert to numpy to simulate typical sklearn input
    X_np = np.array(X)
    y_np = np.array(y)
    
    pipe = Pipeline([
        ('mdm', MDM())
    ])
    
    pipe.fit(X_np, y_np)
    preds = pipe.predict(X_np)
    assert len(preds) == len(y_np)
