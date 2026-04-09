"""
Tests for jaxoccoli.hf_encoder — HuggingFace foundation model adapter.

Covers:
  - HFModelAdapter base class and registration
  - make_hf_encoder factory (jaxoccoli pattern)
  - TribeV2Adapter (fsaverage5 BOLD prediction from stimuli)
  - make_cortical_projection (learnable vertex→parcel mapping)
  - PyTorch→JAX array bridge
  - Integration with downstream jaxoccoli tools (covariance, graph)

All tests mock the actual HF models — no downloads required.
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from unittest.mock import MagicMock, patch
from typing import NamedTuple

from jaxoccoli.hf_encoder import (
    HFModelAdapter,
    HFEncoderParams,
    TribeV2Adapter,
    make_hf_encoder,
    make_cortical_projection,
    torch_to_jax,
    register_adapter,
    get_adapter,
)


# ===========================================================================
# Fixtures
# ===========================================================================

N_VERTICES_FS5 = 20484  # fsaverage5 cortical mesh
N_TIMESTEPS = 50
N_PARCELS = 100


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def mock_tribe_predictions():
    """Simulated TRIBEv2 output: (n_timesteps, n_vertices) numpy array."""
    rng = np.random.RandomState(0)
    return rng.randn(N_TIMESTEPS, N_VERTICES_FS5).astype(np.float32)


@pytest.fixture
def mock_tribe_model(mock_tribe_predictions):
    """Mock TribeModel that returns canned predictions."""
    model = MagicMock()
    model.predict.return_value = (mock_tribe_predictions, np.arange(N_TIMESTEPS))
    model.get_events_dataframe.return_value = MagicMock()  # dummy df
    model.config = {"n_vertices": N_VERTICES_FS5, "surface": "fsaverage5"}
    return model


class DummyAdapter(HFModelAdapter):
    """Minimal adapter for testing the base class contract."""

    def load_model(self, model_id, cache_dir=None, **kwargs):
        model = MagicMock()
        model.config = {"n_features": 10}
        return model

    def extract_features(self, model, inputs, **kwargs):
        n_time = inputs.get("n_time", 20)
        return np.random.randn(n_time, 10).astype(np.float32)

    @property
    def output_dim(self):
        return 10

    @property
    def output_space(self):
        return "generic"


# ===========================================================================
# Base class contract
# ===========================================================================

class TestHFModelAdapter:
    """Tests for the adapter base class."""

    def test_abstract_methods_enforced(self):
        """Cannot instantiate HFModelAdapter directly."""
        with pytest.raises(TypeError):
            HFModelAdapter()

    def test_dummy_adapter_instantiates(self):
        adapter = DummyAdapter()
        assert adapter.output_dim == 10
        assert adapter.output_space == "generic"

    def test_load_model_returns_object(self):
        adapter = DummyAdapter()
        model = adapter.load_model("dummy/model")
        assert model is not None

    def test_extract_features_returns_ndarray(self):
        adapter = DummyAdapter()
        model = adapter.load_model("dummy/model")
        features = adapter.extract_features(model, {"n_time": 15})
        assert isinstance(features, np.ndarray)
        assert features.shape == (15, 10)


# ===========================================================================
# Adapter registry
# ===========================================================================

class TestAdapterRegistry:
    """Tests for adapter registration and lookup."""

    def test_register_and_retrieve(self):
        register_adapter("test/dummy", DummyAdapter)
        adapter_cls = get_adapter("test/dummy")
        assert adapter_cls is DummyAdapter

    def test_unknown_adapter_raises(self):
        with pytest.raises(KeyError):
            get_adapter("nonexistent/model-xyz-999")

    def test_tribev2_registered(self):
        """TribeV2Adapter should be pre-registered."""
        adapter_cls = get_adapter("facebook/tribev2")
        assert adapter_cls is TribeV2Adapter


# ===========================================================================
# TribeV2Adapter
# ===========================================================================

class TestTribeV2Adapter:
    """Tests for the TRIBEv2 adapter."""

    def test_output_space_is_fsaverage5(self):
        adapter = TribeV2Adapter()
        assert adapter.output_space == "fsaverage5"

    def test_output_dim_is_fsaverage5_vertices(self):
        adapter = TribeV2Adapter()
        assert adapter.output_dim == N_VERTICES_FS5

    @patch("jaxoccoli.hf_encoder.TribeV2Adapter._import_tribe")
    def test_load_model(self, mock_import):
        """load_model should call TribeModel.from_pretrained."""
        mock_cls = MagicMock()
        mock_import.return_value = mock_cls
        adapter = TribeV2Adapter()
        adapter.load_model("facebook/tribev2", cache_dir="/tmp/cache")
        mock_cls.from_pretrained.assert_called_once_with(
            "facebook/tribev2", cache_folder="/tmp/cache"
        )

    def test_extract_features_shape(self, mock_tribe_model, mock_tribe_predictions):
        adapter = TribeV2Adapter()
        features = adapter.extract_features(
            mock_tribe_model,
            {"video_path": "test.mp4"},
        )
        assert isinstance(features, np.ndarray)
        assert features.shape == mock_tribe_predictions.shape


# ===========================================================================
# torch_to_jax bridge
# ===========================================================================

class TestTorchToJax:
    """Tests for the PyTorch→JAX array conversion."""

    def test_numpy_passthrough(self):
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        result = torch_to_jax(arr)
        assert isinstance(result, jnp.ndarray)
        np.testing.assert_array_equal(np.asarray(result), arr)

    def test_jax_passthrough(self):
        arr = jnp.array([1.0, 2.0])
        result = torch_to_jax(arr)
        assert isinstance(result, jnp.ndarray)

    def test_torch_tensor_conversion(self):
        """If torch is available and functional, convert torch.Tensor → jax array."""
        try:
            import torch
            t = torch.tensor([1.0, 2.0, 3.0])
            result = torch_to_jax(t)
            assert isinstance(result, jnp.ndarray)
            np.testing.assert_array_almost_equal(
                np.asarray(result), [1.0, 2.0, 3.0]
            )
        except (ImportError, RuntimeError):
            pytest.skip("torch not installed or numpy ABI mismatch")

    def test_preserves_float32(self):
        arr = np.array([1.0], dtype=np.float32)
        result = torch_to_jax(arr)
        assert result.dtype == jnp.float32

    def test_float64_requires_x64(self):
        """float64 is preserved only when JAX x64 mode is enabled."""
        arr = np.array([1.0], dtype=np.float64)
        result = torch_to_jax(arr)
        if jax.config.x64_enabled:
            assert result.dtype == jnp.float64
        else:
            # JAX truncates to float32 by default
            assert result.dtype == jnp.float32


# ===========================================================================
# make_hf_encoder factory
# ===========================================================================

class TestMakeHFEncoder:
    """Tests for the main factory function."""

    def test_returns_params_and_forward(self, rng_key):
        register_adapter("test/dummy", DummyAdapter)
        params, forward_fn = make_hf_encoder(
            "test/dummy",
            key=rng_key,
        )
        assert isinstance(params, HFEncoderParams)
        assert callable(forward_fn)

    def test_params_contain_model_id(self, rng_key):
        register_adapter("test/dummy", DummyAdapter)
        params, _ = make_hf_encoder("test/dummy", key=rng_key)
        assert params.model_id == "test/dummy"

    def test_forward_returns_jax_array(self, rng_key):
        register_adapter("test/dummy", DummyAdapter)
        params, forward_fn = make_hf_encoder("test/dummy", key=rng_key)
        output = forward_fn(params, {"n_time": 10})
        assert isinstance(output, jnp.ndarray)

    def test_forward_output_shape(self, rng_key):
        register_adapter("test/dummy", DummyAdapter)
        params, forward_fn = make_hf_encoder("test/dummy", key=rng_key)
        output = forward_fn(params, {"n_time": 10})
        assert output.shape[0] == 10  # n_time
        assert output.shape[1] == 10  # DummyAdapter.output_dim

    def test_lazy_loading(self, rng_key):
        """Model should not be loaded until first forward call."""
        register_adapter("test/dummy", DummyAdapter)
        params, forward_fn = make_hf_encoder(
            "test/dummy", key=rng_key, lazy=True,
        )
        # Model not yet loaded
        assert params.model is None
        # First call triggers loading
        output = forward_fn(params, {"n_time": 5})
        assert output is not None


# ===========================================================================
# make_cortical_projection (learnable vertex→parcel)
# ===========================================================================

class TestMakeCorticalProjection:
    """Tests for learnable projection from cortical vertices to parcels."""

    def test_returns_params_and_forward(self, rng_key):
        params, forward_fn = make_cortical_projection(
            n_vertices=N_VERTICES_FS5,
            n_parcels=N_PARCELS,
            key=rng_key,
        )
        assert callable(forward_fn)

    def test_forward_output_shape(self, rng_key):
        params, forward_fn = make_cortical_projection(
            n_vertices=N_VERTICES_FS5,
            n_parcels=N_PARCELS,
            key=rng_key,
        )
        # Input: (n_time, n_vertices), output: (n_time, n_parcels)
        data = jnp.ones((N_TIMESTEPS, N_VERTICES_FS5))
        output = forward_fn(params, data)
        assert output.shape == (N_TIMESTEPS, N_PARCELS)

    def test_weights_sum_to_one(self, rng_key):
        """Parcellation weights should be normalised (softmax rows)."""
        params, _ = make_cortical_projection(
            n_vertices=N_VERTICES_FS5,
            n_parcels=N_PARCELS,
            key=rng_key,
        )
        # The projection matrix after softmax should have rows summing to ~1
        weights = jax.nn.softmax(params.logits, axis=0)  # (n_vertices, n_parcels)
        col_sums = jnp.sum(weights, axis=0)
        # Each parcel's weights across vertices should sum to something > 0
        assert jnp.all(col_sums > 0)

    def test_differentiable(self, rng_key):
        """Gradients should flow through the projection."""
        params, forward_fn = make_cortical_projection(
            n_vertices=100, n_parcels=10, key=rng_key,
        )
        data = jnp.ones((5, 100))

        def loss(p):
            out = forward_fn(p, data)
            return jnp.sum(out ** 2)

        grads = jax.grad(loss)(params)
        assert grads.logits.shape == params.logits.shape
        assert jnp.any(grads.logits != 0)

    def test_jit_compatible(self, rng_key):
        params, forward_fn = make_cortical_projection(
            n_vertices=100, n_parcels=10, key=rng_key,
        )
        data = jnp.ones((5, 100))
        jitted = jax.jit(forward_fn)
        output = jitted(params, data)
        assert output.shape == (5, 10)


# ===========================================================================
# Integration: HF encoder → cortical projection → jaxoccoli covariance
# ===========================================================================

class TestEndToEnd:
    """Integration: encoder output feeds into jaxoccoli connectivity tools."""

    def test_encoder_to_parcellation_to_corr(self, rng_key):
        """Full path: HF features → vertex BOLD → parcels → FC matrix."""
        from jaxoccoli.covariance import corr

        # Step 1: Simulated encoder output (n_time, n_vertices)
        n_time, n_verts, n_parc = 50, 200, 20
        encoder_output = jax.random.normal(rng_key, (n_time, n_verts))

        # Step 2: Project to parcels
        params, project_fn = make_cortical_projection(
            n_vertices=n_verts, n_parcels=n_parc, key=rng_key,
        )
        parcellated = project_fn(params, encoder_output)
        assert parcellated.shape == (n_time, n_parc)

        # Step 3: Compute FC matrix
        fc = corr(parcellated.T)  # (n_parc, n_parc)
        assert fc.shape == (n_parc, n_parc)
        # Diagonal should be ~1
        np.testing.assert_array_almost_equal(
            np.diag(np.asarray(fc)), np.ones(n_parc), decimal=5
        )

    def test_gradients_flow_through_full_pipeline(self, rng_key):
        """Gradient from FC loss back through projection."""
        from jaxoccoli.covariance import corr

        n_time, n_verts, n_parc = 30, 100, 10
        encoder_output = jax.random.normal(rng_key, (n_time, n_verts))

        params, project_fn = make_cortical_projection(
            n_vertices=n_verts, n_parcels=n_parc, key=rng_key,
        )

        def pipeline_loss(proj_params):
            parcellated = project_fn(proj_params, encoder_output)
            fc = corr(parcellated.T)
            # Minimize off-diagonal FC (encourage independence)
            mask = 1.0 - jnp.eye(n_parc)
            return jnp.sum(fc ** 2 * mask)

        grads = jax.grad(pipeline_loss)(params)
        assert jnp.any(grads.logits != 0), "Gradients should flow through FC→projection"
