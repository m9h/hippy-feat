"""HuggingFace EEG foundation-model adapters for jaxoccoli.

Extends the ``hf_encoder.HFModelAdapter`` pattern to **EEG** foundation
models, with one wrinkle: for mechanistic-interpretability work the
caller doesn't want the model's task head, they want the activations
**at a specific transformer block**. Each adapter therefore:

  1. Loads the frozen pretrained encoder.
  2. Registers a PyTorch forward hook on a chosen block.
  3. Runs a forward pass; ``extract_features`` returns the **hooked
     activations** (not the final output) as a numpy array.

Two adapters ship:

  * ``REVEAdapter`` — wraps ``brain-bzh/reve-base`` (REVE: 4D positional
    encoding, 200 Hz, arbitrary montage). Companion ``brain-bzh/reve-positions``
    is loaded automatically. **Both checkpoints are gated** — accept the
    Responsible Use Agreement on HF before first run.
  * ``LaBraMAdapter`` — wraps ``braindecode.models.Labram`` with weights
    from the upstream ``935963004/LaBraM`` release. 200 Hz, 1-s channel
    patches.

Inputs dict (REVE)::

    {
        "eeg":             np.ndarray (B, C, T),  float32, 200 Hz,
        "electrode_names": list[str]  (length C),
    }

Inputs dict (LaBraM)::

    {
        "eeg":             np.ndarray (B, C, T),  float32, 200 Hz,
        "ch_names":        list[str]  (length C),  # mapped to LaBraM's montage
    }

Output: numpy ``(B, n_patches, d_model)`` — the activations of the hooked
block. SAE training flattens (B, n_patches) into the leading axis.

This module does NOT hard-import torch / transformers / braindecode at
top level — those land at ``load_model`` time so the rest of jaxoccoli
stays importable on machines without the GPU stack.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from jaxoccoli.hf_encoder import HFModelAdapter, register_adapter


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_dotted(obj: Any, path: str):
    """Resolve a dotted attribute path on ``obj``.

    ``_resolve_dotted(model, "encoder.layers")`` returns ``model.encoder.layers``.
    Used to locate the transformer block list inside an HF model whose
    exact module layout we don't want to hard-code.
    """
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            raise AttributeError(
                f"Could not resolve '{path}' on {type(obj).__name__}: "
                f"missing attribute '{part}'. Inspect the loaded model with "
                f"`print(model)` to find the right hook path."
            )
        cur = getattr(cur, part)
    return cur


# ---------------------------------------------------------------------------
# REVE adapter
# ---------------------------------------------------------------------------

REVE_BASE_ID = "brain-bzh/reve-base"
REVE_POS_ID = "brain-bzh/reve-positions"


class REVEAdapter(HFModelAdapter):
    """Adapter for the REVE EEG foundation model (brain-bzh/reve-base).

    Extracts the per-block activations of REVE's ``TransformerBackbone``
    by calling the model with ``return_output=True``, which already returns
    the full ``[x_initial, x_after_block0, …, x_after_blockN]`` list. This
    avoids ``register_forward_hook`` entirely — REVE's blocks are stored as
    ``transformer.layers`` where each entry is ``ModuleList([attn, ff])``
    (no callable forward), so hooks would never fire on them anyway, and
    hooking ``attn``/``ff`` separately would capture pre-residual values.

    Parameters
    ----------
    layer  : block index. ``-1`` = final block (post-attn + post-ff with
             residuals applied). ``0`` = first block. Negative indexing
             is supported. (Use ``layer="embedding"`` to get the pre-block
             token sequence — useful as a control.)
    device : "cuda" or "cpu". Defaults to cuda if available.
    """

    def __init__(self, *, layer: int | str = -1,
                 device: str | None = None):
        self.layer = layer
        self._device = device
        self._d_model: int | None = None
        self._n_blocks: int | None = None

    # ---- HFModelAdapter API ------------------------------------------------

    def load_model(self, model_id: str, cache_dir: str | None = None,
                   **kwargs) -> dict:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "REVEAdapter requires torch + transformers. "
                "Install in the GPU container (NGC PyTorch SIF) or with: "
                "pip install 'transformers>=4.40' torch"
            ) from e

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        pos_bank = AutoModel.from_pretrained(
            REVE_POS_ID, trust_remote_code=True, cache_dir=cache_dir,
        ).to(self._device).eval()
        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir,
        ).to(self._device).eval()

        # Validate that this REVE checkpoint exposes the per-layer return
        # path we depend on, and read hidden size + depth from its config.
        cfg = getattr(model, "config", None)
        self._d_model = (
            getattr(cfg, "embed_dim", None)
            or getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", None)
        )
        self._n_blocks = getattr(cfg, "depth", None)
        if self._n_blocks is None:
            # Fall back to introspection
            backbone = getattr(model, "transformer", None)
            layers = getattr(backbone, "layers", None) if backbone else None
            if layers is not None:
                self._n_blocks = len(layers)

        return {"model": model, "pos_bank": pos_bank}

    def extract_features(self, model_dict: dict, inputs: dict,
                         **kwargs) -> np.ndarray:
        try:
            import torch
        except ImportError as e:  # pragma: no cover — caught at load_model
            raise ImportError("torch not available") from e

        if "eeg" not in inputs:
            raise ValueError("REVE inputs dict must contain 'eeg' (B,C,T)")
        if "electrode_names" not in inputs:
            raise ValueError(
                "REVE inputs dict must contain 'electrode_names' (list of "
                "channel labels) so reve-positions can map to 3D coords"
            )

        model = model_dict["model"]
        pos_bank = model_dict["pos_bank"]

        eeg_np = np.asarray(inputs["eeg"], dtype=np.float32)
        eeg = torch.from_numpy(eeg_np).to(self._device)
        electrode_names = list(inputs["electrode_names"])

        # REVE's internal Attention dispatches to FlashAttention which only
        # accepts fp16/bf16. The model itself is fp32-loaded and REVE.forward
        # does `eeg = eeg.float()` early on, so we have to autocast at the op
        # level to get bf16 into the attention kernel. Activations come back
        # as bf16; we upcast to fp32 before NumPy conversion.
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self._device == "cuda"
            else torch.autocast(device_type="cpu", dtype=torch.bfloat16,
                                enabled=False)
        )
        with torch.no_grad(), autocast_ctx:
            positions = pos_bank(electrode_names)
            positions = positions.expand(eeg.size(0), -1, -1)
            # return_output=True bypasses the final_layer Identity and gives
            # us the full per-layer list out of TransformerBackbone.
            out_layers = model(eeg, positions, return_output=True)

        # ``out_layers`` is a Python list:
        #   [x_initial, x_after_block_0, x_after_block_1, ..., x_after_block_N]
        # Length = n_blocks + 1.
        n_blocks = len(out_layers) - 1
        if isinstance(self.layer, str):
            if self.layer == "embedding":
                tensor = out_layers[0]
            else:
                raise ValueError(
                    f"unknown layer specifier {self.layer!r}; expected an "
                    f"int block index or 'embedding'"
                )
        else:
            # Map block index k -> out_layers[k+1] (out_layers[0] is pre-block).
            k = int(self.layer)
            if k < 0:
                k += n_blocks
            if not 0 <= k < n_blocks:
                raise IndexError(
                    f"layer={self.layer} out of range; REVE has {n_blocks} "
                    f"blocks (valid 0..{n_blocks - 1} or -{n_blocks}..-1)"
                )
            tensor = out_layers[k + 1]

        try:
            # Upcast bf16 → fp32 before numpy (numpy has no bf16 dtype).
            return tensor.detach().float().cpu().numpy()
        except AttributeError:
            return np.asarray(tensor)

    # ---- HFModelAdapter introspection -------------------------------------

    @property
    def output_dim(self) -> int:
        if self._d_model is None:
            raise RuntimeError(
                "output_dim is only known after load_model — REVE's hidden "
                "size is read from the model config at load time."
            )
        return int(self._d_model)

    @property
    def output_space(self) -> str:
        return "reve-hidden-states"


# ---------------------------------------------------------------------------
# LaBraM adapter
# ---------------------------------------------------------------------------

LABRAM_DEFAULT_ID = "labram-base"  # virtual ID — braindecode-hosted
LABRAM_DEFAULT_HOOK_PATH = "blocks"


class LaBraMAdapter(HFModelAdapter):
    """Adapter for LaBraM (Large Brain Model for EEG, ICLR'24 spotlight).

    Uses ``braindecode.models.Labram`` (preferred — its weight loader handles
    the upstream ``935963004/LaBraM`` ``.pth`` checkpoint format), with a
    forward hook on the requested transformer block.

    Parameters
    ----------
    layer       : block index (default -1).
    hook_path   : dotted path to block list (``blocks`` for braindecode's port).
    n_channels  : montage channel count. Required by braindecode's Labram
                  constructor.
    n_times     : window length in samples (LaBraM default 1 s @ 200 Hz = 200).
    weights     : path to a local ``.pth`` checkpoint (LaBraM weights are
                  released as ``.pth`` files, not on HF).
    device      : "cuda" or "cpu". Defaults to cuda if available.
    """

    def __init__(self, *, layer: int = -1,
                 hook_path: str = LABRAM_DEFAULT_HOOK_PATH,
                 n_channels: int = 64, n_times: int = 200,
                 weights: str | None = None,
                 device: str | None = None):
        self.layer = layer
        self.hook_path = hook_path
        self.n_channels = n_channels
        self.n_times = n_times
        self.weights = weights
        self._device = device
        self._d_model: int | None = None
        self._captured: np.ndarray | None = None
        self._hook_handle = None

    def load_model(self, model_id: str, cache_dir: str | None = None,
                   **kwargs) -> Any:
        try:
            import torch
            from braindecode.models import Labram
        except ImportError as e:
            raise ImportError(
                "LaBraMAdapter requires torch + braindecode>=0.8. "
                "Install with: pip install 'braindecode>=0.8' torch"
            ) from e

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        model = Labram(n_chans=self.n_channels, n_times=self.n_times)
        if self.weights is not None:
            state = torch.load(self.weights, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
        model = model.to(self._device).eval()

        try:
            block_list = _resolve_dotted(model, self.hook_path)
            target = block_list[self.layer]
        except (AttributeError, IndexError) as e:
            raise RuntimeError(
                f"Could not hook layer {self.layer} at '{self.hook_path}'. "
                f"Inspect LaBraM's module layout with `print(model)` and pass "
                f"a different `hook_path` to LaBraMAdapter."
            ) from e

        self._hook_handle = target.register_forward_hook(self._capture_hook)

        # braindecode Labram exposes embedding dim as `embed_dim` on the model
        self._d_model = getattr(model, "embed_dim", None)
        if self._d_model is None:
            # Fall back to inspecting the patch projector's output channels
            patch = getattr(model, "patch_embed", None)
            if patch is not None:
                self._d_model = getattr(patch, "embed_dim", None) \
                    or getattr(patch, "out_chans", None)

        return model

    def extract_features(self, model, inputs: dict, **kwargs) -> np.ndarray:
        try:
            import torch
        except ImportError as e:  # pragma: no cover
            raise ImportError("torch not available") from e

        if "eeg" not in inputs:
            raise ValueError("LaBraM inputs dict must contain 'eeg' (B,C,T)")

        eeg_np = np.asarray(inputs["eeg"], dtype=np.float32)
        eeg = torch.from_numpy(eeg_np).to(self._device)

        with torch.no_grad():
            _ = model(eeg)

        if self._captured is None:
            raise RuntimeError(
                "LaBraM forward hook did not fire — check `hook_path`."
            )
        out = self._captured
        self._captured = None
        return out

    @property
    def output_dim(self) -> int:
        if self._d_model is None:
            raise RuntimeError(
                "output_dim is only known after load_model — LaBraM's hidden "
                "size is read from the model at load time."
            )
        return int(self._d_model)

    @property
    def output_space(self) -> str:
        return "labram-hidden-states"

    def _capture_hook(self, module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        try:
            arr = output.detach().cpu().numpy()
        except AttributeError:
            arr = np.asarray(output)
        self._captured = arr


# ---------------------------------------------------------------------------
# ZUNA adapter
# ---------------------------------------------------------------------------

ZUNA_BASE_ID = "mhough/zuna-base"

# ZUNA (Zyphra) hyper-parameters baked into the public ``zuna-base`` config.
# Kept as module constants so the tokenizer helpers below stay pure/testable
# without instantiating the (heavyweight, trust_remote_code) HF model.
ZUNA_INPUT_DIM = 32          # config.input_dim == num_fine_time_pts (tf)
ZUNA_FINE_TIME_PTS = 32      # samples per token along fine-time axis
ZUNA_DOWNSAMPLE_FACTOR = 1   # config.encoder_latent_downsample_factor
ZUNA_DIM = 1024              # config.dim — the residual-stream width we hook
ZUNA_SFREQ = 256             # ZUNA was pretrained at 256 Hz
# Discrete-position scheme from BCIDatasetArgs (the "v5"/"thirteens" dataset):
# bin each xyz coordinate (metres) into ``num_bins`` cells over [-extreme, +extreme].
ZUNA_POS_NUM_BINS = 100
ZUNA_POS_EXTREME = 0.13


def _zuna_discretize_chan_pos(chan_pos: np.ndarray,
                              num_bins: int = ZUNA_POS_NUM_BINS,
                              extreme: float = ZUNA_POS_EXTREME) -> np.ndarray:
    """Discretise continuous (x,y,z) electrode positions (metres) into integer
    RoPE bins, replicating ``eeg_data.discretize_chan_pos`` with the v5
    "thirteens" extremes ([-0.13, 0.13] on every axis, ``num_bins`` cells).
    """
    chan_pos = np.asarray(chan_pos, dtype=np.float64)
    norm = (chan_pos - (-extreme)) / (extreme - (-extreme))
    disc = np.floor(norm * num_bins).astype(np.int64)
    return np.clip(disc, 0, num_bins - 1)


def _zuna_chan_positions(ch_names: list[str]) -> np.ndarray:
    """3D positions (metres) for HBN's GSN-HydroCel-128 montage, by channel name.

    HBN was recorded on an EGI/Philips GSN-HydroCel-128 net; we source the
    canonical electrode geometry from MNE's standard montage. Channels we
    can't resolve (rare label mismatches) fall back to the head centre
    (which discretises to the middle bin), matching ZUNA's all-zero
    dummy-position convention for unknown sensors.
    """
    try:
        import mne
    except ImportError as e:  # pragma: no cover — present alongside braindecode
        raise ImportError(
            "ZunaAdapter needs MNE to resolve GSN-HydroCel-128 electrode "
            "positions. It ships with braindecode; install 'mne'."
        ) from e

    montage = mne.channels.make_standard_montage("GSN-HydroCel-128")
    ch_pos = montage.get_positions()["ch_pos"]  # name -> (3,) metres
    # Build a case-insensitive lookup so "e12"/"E12"/"EEG E12" all resolve.
    lower = {k.lower(): v for k, v in ch_pos.items()}

    out = np.zeros((len(ch_names), 3), dtype=np.float64)
    for i, name in enumerate(ch_names):
        key = str(name).strip().lower()
        if key in lower:
            out[i] = lower[key]
        elif key.replace("eeg ", "") in lower:
            out[i] = lower[key.replace("eeg ", "")]
        # else: leave at (0,0,0) -> centre bin
    return out


def _zuna_tokenize(eeg: np.ndarray, chan_pos_discrete: np.ndarray,
                   tf: int = ZUNA_FINE_TIME_PTS):
    """Chop one (C, T) window into ZUNA's channel-together token layout.

    Mirrors ``eeg_data.chop_and_reshape_signals`` with ``use_coarse_time="B"``
    (the public checkpoint's layout): each (channel, coarse-time) pair becomes
    one token whose feature axis is ``tf`` fine-time samples. Tokens are
    ordered channel-major: [ch0·tc0, ch0·tc1, …, ch0·tc(K-1), ch1·tc0, …].

    Returns
    -------
    encoder_input    : (seqlen, tf)  float32
    t_coarse         : (seqlen, 1)   int64   — coarse-time index per token
    chan_pos_discr   : (seqlen, 3)   int64   — per-token discrete xyz bins
    tc               : int                    — coarse-time steps (= T // tf)
    """
    C, T = eeg.shape
    if T % tf != 0:
        raise ValueError(
            f"ZUNA tokeniser needs T divisible by {tf} fine-time samples; got "
            f"T={T}. At 256 Hz a 5 s window is 1280 samples (=40·{tf}). Pass "
            f"--target-sfreq {ZUNA_SFREQ} to the extraction pipeline."
        )
    tc = T // tf
    encoder_input = eeg.reshape(C, tc, tf).reshape(C * tc, tf)
    t_coarse = np.tile(np.arange(tc), C).reshape(-1, 1)
    chan_pos_discr = np.repeat(chan_pos_discrete, tc, axis=0)
    return (encoder_input.astype(np.float32), t_coarse,
            chan_pos_discr.astype(np.int64), tc)


def _zuna_strip_registers_and_pool(h: np.ndarray, n_chans: int, tc: int,
                                   df: int = ZUNA_DOWNSAMPLE_FACTOR) -> np.ndarray:
    """Reduce a hooked encoder-block output to per-channel vectors.

    ZUNA's encoder interleaves one *register* token in front of every group
    of ``df`` real tokens (``EncoderTransformer._interleave_registers``), so a
    block's residual stream has ``num_groups·(df+1)`` tokens. We drop the
    registers, trim padding back to ``n_chans·tc`` real tokens, then mean-pool
    over the ``tc`` coarse-time steps within each channel.

    Parameters
    ----------
    h  : (B, L, D) block output, L == num_groups·(df+1).

    Returns
    -------
    (B, n_chans, D) — one pooled vector per channel.
    """
    h = np.asarray(h)
    if h.ndim == 2:
        h = h[None]
    B, L, D = h.shape
    group = df + 1
    num_groups = L // group
    h = h[:, :num_groups * group, :].reshape(B, num_groups, group, D)
    real = h[:, :, 1:, :].reshape(B, num_groups * df, D)   # drop register at idx 0
    seqlen = n_chans * tc
    real = real[:, :seqlen, :].reshape(B, n_chans, tc, D)
    return real.mean(axis=2)


class ZunaAdapter(HFModelAdapter):
    """Adapter for ZUNA (Zyphra ``mhough/zuna-base``) — a 382 M masked-diffusion
    EEG autoencoder. We expose its **encoder** as an SAE feature source.

    Why the encoder block residual (``dim=1024``) and not ``ZunaModel.encode``?
    ``encode`` returns the 32-d bottleneck latent — too narrow for dictionary
    learning. WeightWatcher analysis (``analyze_zuna_weightwatcher.py``) found
    every one of the 16 encoder blocks healthy (α≈2.5–3.2, 1.8 % under-trained),
    so the rich 1024-d residual stream is the right hook target. A forward hook
    on ``model.encoder.layers[layer]`` captures it.

    Tokenisation (replicated from ``zuna…/eeg_data.py``): each window is chopped
    into ``tf=32``-sample fine-time tokens in channel-together order; 4D-RoPE is
    driven by discrete electrode positions (GSN-HydroCel-128, binned over
    [-0.13, 0.13] m) plus a coarse-time index. The encoder also interleaves
    register tokens, which we strip before pooling.

    Output: ``(B, n_chans, 1024)`` — the block residual mean-pooled over
    coarse-time within each channel. ``n_per_window`` therefore equals the
    montage channel count.

    Caveats (deviations from ZUNA's exact training recipe):

      * **Normalisation.** The extraction pipeline z-scores per channel and
        clamps ±15 before calling us; ZUNA trained on ``signal / data_norm``
        (a fixed global scale) + clamp. Per-channel z-score is a reasonable,
        in-distribution-ish standardisation but is a known deviation.
      * **Positions.** We use the *nominal* GSN-HydroCel-128 geometry, not
        per-subject digitised coordinates.
      * **Batch.** ZUNA's ``encode`` assumes B=1 packed sequences
        (``do_idx.squeeze(0)``), so we loop over the batch and run one window
        at a time.

    Parameters
    ----------
    layer     : encoder block index to hook (default -1 = last block).
    hook_path : dotted path to the block list on the ZunaModel
                (default ``model.encoder.layers``).
    device    : "cuda" or "cpu". Defaults to cuda if available.
    """

    def __init__(self, *, layer: int = -1,
                 hook_path: str = "model.encoder.layers",
                 device: str | None = None):
        self.layer = layer
        self.hook_path = hook_path
        self._device = device
        self._d_model: int | None = None
        self._n_blocks: int | None = None
        self._captured = None
        self._hook_handle = None
        self._pos_cache: dict[tuple, np.ndarray] = {}

    def load_model(self, model_id: str, cache_dir: str | None = None,
                   **kwargs) -> Any:
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:
            raise ImportError(
                "ZunaAdapter requires torch + transformers, plus the 'zuna' "
                "package and its lingua deps on PYTHONPATH (trust_remote_code "
                "custom modules). See scripts/extract_eeg_fm_acts.sbatch."
            ) from e

        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        model = AutoModel.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir,
        ).to(self._device).eval()

        cfg = getattr(model, "config", None)
        self._d_model = getattr(cfg, "dim", None) or ZUNA_DIM
        self._n_blocks = getattr(cfg, "n_layers", None)

        try:
            block_list = _resolve_dotted(model, self.hook_path)
            target = block_list[self.layer]
            if self._n_blocks is None:
                self._n_blocks = len(block_list)
        except (AttributeError, IndexError) as e:
            raise RuntimeError(
                f"Could not hook layer {self.layer} at '{self.hook_path}'. "
                f"Inspect ZUNA's module layout with `print(model)` and pass a "
                f"different `hook_path` to ZunaAdapter."
            ) from e

        self._hook_handle = target.register_forward_hook(self._capture_hook)
        return model

    def _capture_hook(self, module, inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        try:
            self._captured = output.detach().float().cpu().numpy()
        except AttributeError:
            self._captured = np.asarray(output)

    def extract_features(self, model, inputs: dict, **kwargs) -> np.ndarray:
        try:
            import torch
        except ImportError as e:  # pragma: no cover — caught at load_model
            raise ImportError("torch not available") from e

        if "eeg" not in inputs:
            raise ValueError("ZUNA inputs dict must contain 'eeg' (B,C,T)")
        ch_names = inputs.get("electrode_names") or inputs.get("ch_names")
        if ch_names is None:
            raise ValueError(
                "ZUNA inputs dict must contain 'electrode_names' (or "
                "'ch_names') so 4D-RoPE positions can be resolved"
            )
        ch_names = list(ch_names)

        eeg_np = np.asarray(inputs["eeg"], dtype=np.float32)
        if eeg_np.ndim == 2:
            eeg_np = eeg_np[None]
        B, C, T = eeg_np.shape

        key = tuple(ch_names)
        if key not in self._pos_cache:
            self._pos_cache[key] = _zuna_discretize_chan_pos(
                _zuna_chan_positions(ch_names)
            )
        chan_pos_discrete = self._pos_cache[key]

        pooled = []
        for b in range(B):
            enc_in, t_coarse, cpd, tc = _zuna_tokenize(eeg_np[b], chan_pos_discrete)
            seqlen = enc_in.shape[0]
            enc_in_t = torch.from_numpy(enc_in).to(self._device)
            t_coarse_t = torch.from_numpy(t_coarse).to(self._device)[None]      # (1,seqlen,1)
            cpd_t = torch.from_numpy(cpd).to(self._device)[None]                # (1,seqlen,3)
            seq_lens_t = torch.tensor([seqlen], device=self._device)

            self._captured = None
            with torch.no_grad():
                model.encode(
                    encoder_input=enc_in_t,
                    seq_lens=seq_lens_t,
                    t_coarse=t_coarse_t,
                    chan_pos_discrete=cpd_t,
                )
            if self._captured is None:
                raise RuntimeError(
                    f"ZUNA forward hook at '{self.hook_path}'[{self.layer}] did "
                    f"not fire — check the hook path against `print(model)`."
                )
            pooled.append(
                _zuna_strip_registers_and_pool(self._captured, C, tc)[0]
            )

        return np.stack(pooled, axis=0).astype(np.float32)   # (B, C, dim)

    @property
    def output_dim(self) -> int:
        if self._d_model is None:
            raise RuntimeError(
                "output_dim is only known after load_model — ZUNA's hidden "
                "size is read from the model config at load time."
            )
        return int(self._d_model)

    @property
    def output_space(self) -> str:
        return "zuna-encoder-hidden-states"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------
# IDs are registered without instantiating the adapter — callers customise
# layer / hook_path / weights when they pass an explicit ``adapter=...``
# to ``make_hf_encoder``. ``register_adapter`` stores the class, not an
# instance, so the customisation path is preserved.

register_adapter(REVE_BASE_ID, REVEAdapter)
register_adapter(LABRAM_DEFAULT_ID, LaBraMAdapter)
register_adapter(ZUNA_BASE_ID, ZunaAdapter)
