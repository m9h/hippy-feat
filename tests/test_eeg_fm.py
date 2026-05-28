"""
Tests for jaxoccoli.eeg_fm — REVE + LaBraM HuggingFace adapters.

All tests mock the actual HF / braindecode model classes — no downloads,
no torch GPU work, no gated-checkpoint access. Coverage:

  - Adapter registration (REVE + LaBraM IDs).
  - Layer-hook plumbing: hooked block's output is what extract_features returns.
  - output_dim / output_space contracts (including error before load).
  - Inputs-dict validation.
  - Compatibility with make_hf_encoder factory (lazy path).
"""
import sys
import types
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from jaxoccoli.hf_encoder import get_adapter, make_hf_encoder
from jaxoccoli.eeg_fm import (
    REVEAdapter,
    LaBraMAdapter,
    REVE_BASE_ID,
    LABRAM_DEFAULT_ID,
    _resolve_dotted,
)


# ===========================================================================
# Mock torch / transformers / braindecode so adapters can load
# ===========================================================================

class _FakeTensor:
    """Stand-in for a torch.Tensor sufficient for adapter testing."""
    def __init__(self, arr):
        self._arr = np.asarray(arr)
    def to(self, device): return self
    def detach(self): return self
    def cpu(self): return self
    def float(self): return self  # autocast upcast no-op on the mock
    def numpy(self): return self._arr
    def size(self, dim): return self._arr.shape[dim]
    def expand(self, *shape): return self  # positions broadcast


def _make_fake_torch():
    """Build a minimal fake torch module with cuda + no_grad + autocast +
    from_numpy enough to drive the adapter test path on the CPU branch."""
    fake = types.ModuleType("torch")
    fake.cuda = types.SimpleNamespace(is_available=lambda: False)

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): return False
    fake.no_grad = _NoGrad

    class _Autocast:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
    fake.autocast = _Autocast
    fake.bfloat16 = "bf16"

    def _from_numpy(arr):
        return _FakeTensor(arr)
    fake.from_numpy = _from_numpy
    fake.Tensor = _FakeTensor

    def _load(path, map_location=None):
        return {}
    fake.load = _load
    return fake


# ===========================================================================
# Registry
# ===========================================================================

class TestRegistration:

    def test_reve_registered(self):
        assert get_adapter(REVE_BASE_ID) is REVEAdapter

    def test_labram_registered(self):
        assert get_adapter(LABRAM_DEFAULT_ID) is LaBraMAdapter


# ===========================================================================
# _resolve_dotted helper
# ===========================================================================

class TestResolveDotted:

    def test_single_attr(self):
        obj = types.SimpleNamespace(layers=[1, 2, 3])
        assert _resolve_dotted(obj, "layers") == [1, 2, 3]

    def test_nested_attr(self):
        obj = types.SimpleNamespace(encoder=types.SimpleNamespace(layers=[0, 1]))
        assert _resolve_dotted(obj, "encoder.layers") == [0, 1]

    def test_missing_raises_with_helpful_msg(self):
        obj = types.SimpleNamespace()
        with pytest.raises(AttributeError, match="missing attribute"):
            _resolve_dotted(obj, "encoder.layers")


# ===========================================================================
# REVE adapter
# ===========================================================================

class _MockBlock:
    """A fake transformer block with a register_forward_hook stub. Only the
    LaBraM mock still uses this — REVE switched to ``return_output=True``."""
    def __init__(self, output):
        self._output = output
        self._hooks = []
    def register_forward_hook(self, fn):
        self._hooks.append(fn)
        return MagicMock()
    def _fire(self):
        for h in self._hooks:
            h(self, None, self._output)


class _MockREVEModel:
    """Fake REVE model honoring the real ``return_output=True`` contract:
    when called with that flag, returns the full per-layer activation list
    ``[x_initial, x_after_block_0, …, x_after_block_N]``."""
    def __init__(self, n_layers=4, d_model=384, n_patches=10):
        self._out_layers = [
            _FakeTensor(np.random.randn(2, n_patches, d_model).astype(np.float32))
            for _ in range(n_layers + 1)  # +1 for the pre-block embedding
        ]
        # transformer.layers is what introspection falls back on when config.depth is absent
        self.transformer = types.SimpleNamespace(
            layers=[types.SimpleNamespace() for _ in range(n_layers)]
        )
        self.config = types.SimpleNamespace(embed_dim=d_model, depth=n_layers)
        self.d_model = d_model
    def to(self, device): return self
    def eval(self): return self
    def __call__(self, eeg, positions, return_output=False):
        if return_output:
            return self._out_layers
        return self._out_layers[-1]  # final-block output, post-Identity


class _MockPosBank:
    def to(self, device): return self
    def eval(self): return self
    def __call__(self, names):
        return _FakeTensor(np.zeros((len(names), 3)))


@pytest.fixture
def patched_reve(monkeypatch):
    """Install fake torch + a fake transformers.AutoModel that returns
    mocked REVE / positions models."""
    fake_torch = _make_fake_torch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    pos = _MockPosBank()
    mdl = _MockREVEModel()

    fake_transformers = types.ModuleType("transformers")
    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return pos if "positions" in model_id else mdl
    fake_transformers.AutoModel = _AutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return pos, mdl


class TestREVEAdapter:

    def test_init_defaults(self):
        a = REVEAdapter()
        assert a.layer == -1

    def test_output_dim_before_load_raises(self):
        with pytest.raises(RuntimeError, match="only known after load_model"):
            REVEAdapter().output_dim

    def test_output_space(self):
        assert REVEAdapter().output_space == "reve-hidden-states"

    def test_load_model_sets_d_model(self, patched_reve):
        a = REVEAdapter()
        a.load_model(REVE_BASE_ID)
        assert a.output_dim == 384

    def test_load_model_reads_n_blocks(self, patched_reve):
        a = REVEAdapter()
        a.load_model(REVE_BASE_ID)
        assert a._n_blocks == 4  # from _MockREVEModel default

    def test_extract_features_returns_requested_block(self, patched_reve):
        _, mdl = patched_reve
        a = REVEAdapter(layer=2)
        loaded = a.load_model(REVE_BASE_ID)
        # Block index 2 maps to out_layers[3] (out_layers[0] is pre-block)
        expected = mdl._out_layers[3]._arr
        out = a.extract_features(
            loaded,
            {"eeg": np.zeros((2, 16, 200), dtype=np.float32),
             "electrode_names": [f"E{i}" for i in range(16)]},
        )
        np.testing.assert_array_equal(out, expected)

    def test_negative_layer_indexes_from_end(self, patched_reve):
        _, mdl = patched_reve
        a = REVEAdapter(layer=-1)
        loaded = a.load_model(REVE_BASE_ID)
        out = a.extract_features(
            loaded,
            {"eeg": np.zeros((1, 4, 200), dtype=np.float32),
             "electrode_names": ["E0", "E1", "E2", "E3"]},
        )
        # layer=-1 → final block → out_layers[-1] (= out_layers[n_blocks])
        np.testing.assert_array_equal(out, mdl._out_layers[-1]._arr)

    def test_embedding_layer_returns_pre_block(self, patched_reve):
        _, mdl = patched_reve
        a = REVEAdapter(layer="embedding")
        loaded = a.load_model(REVE_BASE_ID)
        out = a.extract_features(
            loaded,
            {"eeg": np.zeros((1, 4, 200), dtype=np.float32),
             "electrode_names": ["E0", "E1", "E2", "E3"]},
        )
        np.testing.assert_array_equal(out, mdl._out_layers[0]._arr)

    def test_missing_electrode_names_raises(self, patched_reve):
        a = REVEAdapter()
        loaded = a.load_model(REVE_BASE_ID)
        with pytest.raises(ValueError, match="electrode_names"):
            a.extract_features(
                loaded,
                {"eeg": np.zeros((1, 4, 200), dtype=np.float32)},
            )

    def test_missing_eeg_raises(self, patched_reve):
        a = REVEAdapter()
        loaded = a.load_model(REVE_BASE_ID)
        with pytest.raises(ValueError, match="'eeg'"):
            a.extract_features(loaded, {"electrode_names": ["E0"]})

    def test_layer_out_of_range_raises(self, patched_reve):
        a = REVEAdapter(layer=999)
        loaded = a.load_model(REVE_BASE_ID)
        with pytest.raises(IndexError, match="out of range"):
            a.extract_features(
                loaded,
                {"eeg": np.zeros((1, 4, 200), dtype=np.float32),
                 "electrode_names": ["E0", "E1", "E2", "E3"]},
            )


# ===========================================================================
# LaBraM adapter
# ===========================================================================

class _MockLaBraMModel:
    def __init__(self, n_blocks=3, d_model=200, n_patches=8):
        self.embed_dim = d_model
        block_outputs = [
            _FakeTensor(np.random.randn(2, n_patches, d_model).astype(np.float32))
            for _ in range(n_blocks)
        ]
        self.blocks = [_MockBlock(o) for o in block_outputs]
    def to(self, device): return self
    def eval(self): return self
    def load_state_dict(self, state, strict=False): pass
    def __call__(self, eeg):
        for b in self.blocks:
            b._fire()
        return MagicMock()


@pytest.fixture
def patched_labram(monkeypatch):
    fake_torch = _make_fake_torch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    mdl = _MockLaBraMModel()
    fake_bd = types.ModuleType("braindecode")
    fake_bd_models = types.ModuleType("braindecode.models")
    fake_bd_models.Labram = lambda n_chans, n_times: mdl
    fake_bd.models = fake_bd_models
    monkeypatch.setitem(sys.modules, "braindecode", fake_bd)
    monkeypatch.setitem(sys.modules, "braindecode.models", fake_bd_models)
    return mdl


class TestLaBraMAdapter:

    def test_init_defaults(self):
        a = LaBraMAdapter()
        assert a.layer == -1
        assert a.hook_path == "blocks"
        assert a.n_channels == 64
        assert a.n_times == 200

    def test_output_space(self):
        assert LaBraMAdapter().output_space == "labram-hidden-states"

    def test_load_model_sets_d_model_from_embed_dim(self, patched_labram):
        a = LaBraMAdapter()
        a.load_model(LABRAM_DEFAULT_ID)
        assert a.output_dim == 200

    def test_extract_features_returns_hooked_layer_output(self, patched_labram):
        mdl = patched_labram
        a = LaBraMAdapter(layer=1)
        loaded = a.load_model(LABRAM_DEFAULT_ID)
        expected = mdl.blocks[1]._output._arr
        out = a.extract_features(
            loaded, {"eeg": np.zeros((2, 64, 200), dtype=np.float32)}
        )
        np.testing.assert_array_equal(out, expected)

    def test_missing_eeg_raises(self, patched_labram):
        a = LaBraMAdapter()
        loaded = a.load_model(LABRAM_DEFAULT_ID)
        with pytest.raises(ValueError, match="'eeg'"):
            a.extract_features(loaded, {})


# ===========================================================================
# make_hf_encoder integration with lazy loading
# ===========================================================================

class TestMakeHFEncoderWithEEGAdapters:

    def test_reve_through_factory_lazy(self, patched_reve):
        """make_hf_encoder(REVE_BASE_ID, lazy=True) should defer load."""
        a = REVEAdapter()
        params, forward_fn = make_hf_encoder(
            REVE_BASE_ID, adapter=a, lazy=True,
        )
        assert params.model is None
        out = forward_fn(
            params,
            {"eeg": np.zeros((1, 4, 200), dtype=np.float32),
             "electrode_names": ["E0", "E1", "E2", "E3"]},
        )
        # forward_fn returns a jax array; .shape should be the hooked tensor's
        # shape (2, n_patches, d_model). The mocked tensor uses batch=2.
        assert out.shape[-1] == 384  # d_model
