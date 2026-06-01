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
    ZunaAdapter,
    REVE_BASE_ID,
    LABRAM_DEFAULT_ID,
    ZUNA_BASE_ID,
    _resolve_dotted,
    _zuna_discretize_chan_pos,
    _zuna_tokenize,
    _zuna_strip_registers_and_pool,
    _zuna_ensure_rope_capacity,
    _zuna_precompute_freqs_cis,
    ZUNA_FINE_TIME_PTS,
    ZUNA_POS_NUM_BINS,
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
    def __getitem__(self, idx): return self  # [None] unsqueeze on the mock


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

    fake.tensor = lambda data, device=None: _FakeTensor(np.asarray(data))
    return fake


# ===========================================================================
# Registry
# ===========================================================================

class TestRegistration:

    def test_reve_registered(self):
        assert get_adapter(REVE_BASE_ID) is REVEAdapter

    def test_labram_registered(self):
        assert get_adapter(LABRAM_DEFAULT_ID) is LaBraMAdapter

    def test_zuna_registered(self):
        assert get_adapter(ZUNA_BASE_ID) is ZunaAdapter


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
# ZUNA tokeniser helpers (pure numpy — no model needed)
# ===========================================================================

class TestZunaHelpers:

    def test_discretize_centre_and_extremes(self):
        # Centre of the [-0.13, 0.13] cube -> middle bin; corners -> 0 / max.
        pos = np.array([[0.0, 0.0, 0.0],
                        [-0.13, -0.13, -0.13],
                        [0.13, 0.13, 0.13]])
        disc = _zuna_discretize_chan_pos(pos)
        assert disc.shape == (3, 3)
        np.testing.assert_array_equal(disc[0], [50, 50, 50])
        np.testing.assert_array_equal(disc[1], [0, 0, 0])
        # Upper extreme clamps to num_bins-1.
        np.testing.assert_array_equal(disc[2], [ZUNA_POS_NUM_BINS - 1] * 3)

    def test_discretize_clamps_out_of_bounds(self):
        pos = np.array([[10.0, -10.0, 0.0]])
        disc = _zuna_discretize_chan_pos(pos)
        assert disc[0, 0] == ZUNA_POS_NUM_BINS - 1
        assert disc[0, 1] == 0

    def test_tokenize_shapes_and_channel_major_order(self):
        C, tc, tf = 4, 3, ZUNA_FINE_TIME_PTS
        T = tc * tf
        # Distinct constant per channel so we can verify token ordering.
        eeg = np.repeat(np.arange(C).reshape(C, 1), T, axis=1).astype(np.float32)
        cpd = np.zeros((C, 3), dtype=np.int64)
        enc_in, t_coarse, cpd_r, tc_out = _zuna_tokenize(eeg, cpd, tf=tf)
        assert tc_out == tc
        assert enc_in.shape == (C * tc, tf)
        assert t_coarse.shape == (C * tc, 1)
        assert cpd_r.shape == (C * tc, 3)
        # Channel-major: first tc tokens all belong to channel 0, etc.
        for ch in range(C):
            block = enc_in[ch * tc:(ch + 1) * tc]
            assert np.allclose(block, ch)
        # Coarse-time index cycles 0..tc-1 within each channel.
        np.testing.assert_array_equal(
            t_coarse.ravel(), np.tile(np.arange(tc), C)
        )

    def test_tokenize_rejects_indivisible_T(self):
        eeg = np.zeros((2, ZUNA_FINE_TIME_PTS + 1), dtype=np.float32)
        with pytest.raises(ValueError, match="divisible"):
            _zuna_tokenize(eeg, np.zeros((2, 3)))

    def test_strip_registers_and_pool(self):
        # df=1 -> token layout [reg0, real0, reg1, real1, ...].
        C, tc, D = 2, 3, 5
        seqlen = C * tc
        L = 2 * seqlen
        h = np.zeros((1, L, D), dtype=np.float32)
        # Mark registers with -1 (must be dropped) and reals with a value
        # encoding (channel, coarse_time) so we can check pooling.
        for g in range(seqlen):
            h[0, 2 * g] = -1.0                  # register
            ch, t = divmod(g, tc)               # channel-major real token
            h[0, 2 * g + 1] = ch * 10 + t       # real token value
        pooled = _zuna_strip_registers_and_pool(h, C, tc, df=1)
        assert pooled.shape == (1, C, D)
        # Channel ch pools mean over t=0..tc-1 of (ch*10 + t).
        for ch in range(C):
            expected = np.mean([ch * 10 + t for t in range(tc)])
            assert np.allclose(pooled[0, ch], expected)
        # Registers (-1) must not leak into the pooled result.
        assert (pooled != -1.0).all()


class TestZunaRopeCapacity:
    """The 4D-RoPE table must cover position bins 0..99; the checkpoint ships
    a 50-row table (max_seqlen=50), which previously asserted on the GPU."""

    class _StubRope:
        def __init__(self, freqs_cis, head_dim=64, rope_dim=4, theta=10000.0):
            self.freqs_cis = freqs_cis
            self.head_dim = head_dim
            self.rope_dim = rope_dim
            self.theta = theta
            self.max_seqlen = freqs_cis.shape[0]

        def register_buffer(self, name, value, persistent=True):
            setattr(self, name, value)

    class _StubTransformer:
        """Mirrors ZUNA's BaseTransformer: owns rope_embeddings + max_seqlen,
        the latter being the value that slices the table before the gather."""
        def __init__(self, rope, max_seqlen):
            self.rope_embeddings = rope
            self.max_seqlen = max_seqlen

    def test_noop_when_already_large_enough(self):
        # Pure path: no torch needed when capacity already suffices.
        big = np.zeros((200, 8, 2, 2))
        rope = self._StubRope(big)
        tr = self._StubTransformer(rope, max_seqlen=200)
        _zuna_ensure_rope_capacity(tr, 100)
        assert rope.freqs_cis is big          # untouched
        assert tr.max_seqlen == 200

    def test_grows_buffer_and_transformer_max_seqlen(self):
        torch = pytest.importorskip("torch")
        head_dim, rope_dim, theta = 64, 4, 10000.0
        dim = head_dim // rope_dim
        small = _zuna_precompute_freqs_cis(dim, 50, theta)
        rope = self._StubRope(small, head_dim, rope_dim, theta)
        tr = self._StubTransformer(rope, max_seqlen=50)
        _zuna_ensure_rope_capacity(tr, 100)
        assert rope.freqs_cis.shape[0] == 100
        # Both the table and the slicing bound must grow, or the table is
        # re-truncated to 50 before the position gather (the jobs-1515/1516 bug).
        assert tr.max_seqlen == 100
        assert rope.max_seqlen == 100
        # Enlarging is loss-free: original rows are bit-identical.
        assert torch.equal(rope.freqs_cis[:50], small)
        full = _zuna_precompute_freqs_cis(dim, 100, theta)
        assert torch.equal(rope.freqs_cis, full)


# ===========================================================================
# ZUNA adapter
# ===========================================================================

class _MockZunaBlock:
    def __init__(self):
        self._hooks = []
    def register_forward_hook(self, fn):
        self._hooks.append(fn)
        return MagicMock()
    def _fire(self, output):
        for h in self._hooks:
            h(self, None, output)


class _MockZunaModel:
    """Honors the ZunaModel attribute tree (``model.encoder.layers``) and the
    ``encode(encoder_input, seq_lens, ...)`` API; firing the hooked encoder
    block with a deterministic register-interleaved output."""
    def __init__(self, n_blocks=16, dim=8):
        self.blocks = [_MockZunaBlock() for _ in range(n_blocks)]
        encoder = types.SimpleNamespace(layers=self.blocks)
        self.model = types.SimpleNamespace(encoder=encoder)
        self.config = types.SimpleNamespace(dim=dim, n_layers=n_blocks)
        self._dim = dim
        self.last_block_output = None
    def to(self, device): return self
    def eval(self): return self
    def encode(self, encoder_input, seq_lens, t_coarse=None,
               chan_pos_discrete=None, tok_idx=None):
        seqlen = encoder_input.size(0)
        L = 2 * seqlen  # df=1 register interleave
        arr = np.arange(L * self._dim, dtype=np.float32).reshape(1, L, self._dim)
        out = _FakeTensor(arr)
        self.last_block_output = arr
        for b in self.blocks:
            if b._hooks:
                b._fire(out)
        return _FakeTensor(np.zeros((1, seqlen, 32), dtype=np.float32))


@pytest.fixture
def patched_zuna(monkeypatch):
    fake_torch = _make_fake_torch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    mdl = _MockZunaModel()
    fake_transformers = types.ModuleType("transformers")
    class _AutoModel:
        @staticmethod
        def from_pretrained(model_id, **kwargs):
            return mdl
    fake_transformers.AutoModel = _AutoModel
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    return mdl


class TestZunaAdapter:

    def test_init_defaults(self):
        a = ZunaAdapter()
        assert a.layer == -1
        assert a.hook_path == "model.encoder.layers"

    def test_output_space(self):
        assert ZunaAdapter().output_space == "zuna-encoder-hidden-states"

    def test_output_dim_before_load_raises(self):
        with pytest.raises(RuntimeError, match="only known after load_model"):
            ZunaAdapter().output_dim

    def test_load_model_sets_d_model_and_blocks(self, patched_zuna):
        a = ZunaAdapter()
        a.load_model(ZUNA_BASE_ID)
        assert a.output_dim == 8       # _MockZunaModel dim
        assert a._n_blocks == 16

    def test_extract_features_pools_to_per_channel(self, patched_zuna):
        mdl = patched_zuna
        C, tc, tf = 3, 2, ZUNA_FINE_TIME_PTS
        T = tc * tf
        a = ZunaAdapter()
        loaded = a.load_model(ZUNA_BASE_ID)
        ch_names = [f"E{i+1}" for i in range(C)]
        # Pre-seed the position cache so MNE isn't needed in the unit test.
        a._pos_cache[tuple(ch_names)] = np.zeros((C, 3), dtype=np.int64)

        eeg = np.random.randn(2, C, T).astype(np.float32)
        out = a.extract_features(
            loaded, {"eeg": eeg, "electrode_names": ch_names}
        )
        assert out.shape == (2, C, 8)   # (B, n_chans, dim)
        # Must match the pure-helper reduction of the fired block output.
        expected = _zuna_strip_registers_and_pool(
            mdl.last_block_output, C, tc, df=1
        )[0]
        np.testing.assert_allclose(out[0], expected, rtol=1e-5)

    def test_missing_eeg_raises(self, patched_zuna):
        a = ZunaAdapter()
        loaded = a.load_model(ZUNA_BASE_ID)
        with pytest.raises(ValueError, match="'eeg'"):
            a.extract_features(loaded, {"electrode_names": ["E1"]})

    def test_missing_ch_names_raises(self, patched_zuna):
        a = ZunaAdapter()
        loaded = a.load_model(ZUNA_BASE_ID)
        with pytest.raises(ValueError, match="electrode_names"):
            a.extract_features(
                loaded, {"eeg": np.zeros((1, 3, ZUNA_FINE_TIME_PTS),
                                         dtype=np.float32)}
            )

    def test_positions_from_gsn_montage(self):
        # Real MNE montage resolution (skips if mne unavailable). Verifies our
        # name lookup returns distinct, in-range coordinates for GSN labels.
        pytest.importorskip("mne")
        from jaxoccoli.eeg_fm import _zuna_chan_positions
        pos = _zuna_chan_positions(["E1", "E50", "E101"])
        assert pos.shape == (3, 3)
        # Distinct electrodes -> distinct positions, all within head radius.
        assert not np.allclose(pos[0], pos[1])
        assert np.abs(pos).max() < 0.13


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
