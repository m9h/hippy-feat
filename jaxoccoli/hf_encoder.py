"""HuggingFace foundation model adapter for jaxoccoli.

Provides a reusable pattern for integrating pretrained HuggingFace models
(PyTorch or JAX) into jaxoccoli's pure-JAX analysis pipeline:

    HF model (frozen) → numpy features → JAX arrays → learnable projection
                                                       → jaxoccoli tools

The adapter pattern separates:
    1. **Model loading** — handled by model-specific ``HFModelAdapter`` subclasses
    2. **Feature extraction** — adapter calls the HF model, returns numpy
    3. **Projection** — ``make_cortical_projection`` maps features to parcels
       via a learnable softmax-normalised weight matrix (differentiable)

Factory:
    ``make_hf_encoder(model_id, ...)`` → ``(HFEncoderParams, forward_fn)``

Concrete adapters:
    - ``TribeV2Adapter`` — Meta's TRI-modal Brain Encoder (video/audio/text → fsaverage5 BOLD)
    - Extensible via ``register_adapter(model_id, adapter_cls)``

Example::

    params, forward_fn = make_hf_encoder("facebook/tribev2", key=key)
    bold = forward_fn(params, {"video_path": "stimulus.mp4"})  # (T, 20484)

    proj_params, project = make_cortical_projection(20484, 100, key=key)
    parcellated = project(proj_params, bold)                    # (T, 100)
    fc = jaxoccoli.corr(parcellated.T)                          # (100, 100)

References
----------
d'Ascoli S et al. (2026) A foundation model of vision, audition, and
    language for in-silico neuroscience. ArXiv.
"""

from __future__ import annotations

import abc
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# PyTorch → JAX bridge
# ---------------------------------------------------------------------------

def torch_to_jax(tensor) -> jnp.ndarray:
    """Convert a PyTorch tensor, numpy array, or JAX array to a JAX array.

    Parameters
    ----------
    tensor : torch.Tensor, np.ndarray, or jnp.ndarray

    Returns
    -------
    jnp.ndarray
    """
    if isinstance(tensor, jnp.ndarray):
        return tensor
    if isinstance(tensor, np.ndarray):
        return jnp.array(tensor)
    # Try torch
    try:
        import torch
        if isinstance(tensor, torch.Tensor):
            return jnp.array(tensor.detach().cpu().numpy())
    except ImportError:
        pass
    # Fallback: try converting via numpy
    return jnp.array(np.asarray(tensor))


# ---------------------------------------------------------------------------
# Adapter registry
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type[HFModelAdapter]] = {}


def register_adapter(model_id: str, adapter_cls: type[HFModelAdapter]) -> None:
    """Register an adapter class for a HuggingFace model ID.

    Parameters
    ----------
    model_id : HuggingFace model identifier (e.g. "facebook/tribev2")
    adapter_cls : subclass of HFModelAdapter
    """
    _ADAPTER_REGISTRY[model_id] = adapter_cls


def get_adapter(model_id: str) -> type[HFModelAdapter]:
    """Look up the adapter class for a model ID.

    Parameters
    ----------
    model_id : HuggingFace model identifier

    Returns
    -------
    Adapter class

    Raises
    ------
    KeyError : if no adapter is registered for the model ID
    """
    if model_id not in _ADAPTER_REGISTRY:
        raise KeyError(
            f"No adapter registered for '{model_id}'. "
            f"Available: {list(_ADAPTER_REGISTRY.keys())}. "
            f"Register one with register_adapter(model_id, adapter_cls)."
        )
    return _ADAPTER_REGISTRY[model_id]


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------

class HFModelAdapter(abc.ABC):
    """Abstract base class for HuggingFace model adapters.

    Subclass this to integrate a new HF model into jaxoccoli.
    Each adapter must implement:
        - ``load_model`` — download/load the pretrained model
        - ``extract_features`` — run inference and return numpy arrays
        - ``output_dim`` — dimensionality of the feature output
        - ``output_space`` — description of the output space
    """

    @abc.abstractmethod
    def load_model(self, model_id: str, cache_dir: str | None = None, **kwargs) -> Any:
        """Load the pretrained model.

        Parameters
        ----------
        model_id : HuggingFace model identifier
        cache_dir : optional local cache directory

        Returns
        -------
        The loaded model object.
        """

    @abc.abstractmethod
    def extract_features(self, model: Any, inputs: dict, **kwargs) -> np.ndarray:
        """Run the model and return features as a numpy array.

        Parameters
        ----------
        model : loaded model object (from load_model)
        inputs : dict of input data (model-specific keys)

        Returns
        -------
        np.ndarray of shape (n_timesteps, output_dim) or similar.
        """

    @property
    @abc.abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the feature output."""

    @property
    @abc.abstractmethod
    def output_space(self) -> str:
        """Description of the output coordinate space (e.g. 'fsaverage5')."""


# ---------------------------------------------------------------------------
# TRIBEv2 adapter
# ---------------------------------------------------------------------------

N_VERTICES_FS5 = 20484  # fsaverage5 cortical mesh vertex count


class TribeV2Adapter(HFModelAdapter):
    """Adapter for Meta's TRIBEv2 (TRI-modal Brain Encoder v2).

    Wraps ``tribev2.TribeModel`` to produce fMRI predictions on the
    fsaverage5 cortical surface from video, audio, or text stimuli.

    Inputs dict keys (at least one required):
        - ``video_path`` : path to video file
        - ``audio_path`` : path to audio file
        - ``text_path`` : path to text file
        - ``events`` : pre-built events dataframe (optional, built from paths if absent)
    """

    @staticmethod
    def _import_tribe():
        """Import tribev2 module, raising a clear error if not installed."""
        try:
            from tribev2 import TribeModel
            return TribeModel
        except ImportError:
            raise ImportError(
                "TRIBEv2 is not installed. Install with: pip install tribev2\n"
                "Model card: https://huggingface.co/facebook/tribev2"
            )

    def load_model(self, model_id: str, cache_dir: str | None = None, **kwargs) -> Any:
        TribeModel = self._import_tribe()
        return TribeModel.from_pretrained(
            model_id, cache_folder=cache_dir,
        )

    def extract_features(self, model: Any, inputs: dict, **kwargs) -> np.ndarray:
        # Build events dataframe if not provided
        events = inputs.get("events")
        if events is None:
            event_kwargs = {}
            for key in ("video_path", "audio_path", "text_path"):
                if key in inputs:
                    event_kwargs[key] = inputs[key]
            events = model.get_events_dataframe(**event_kwargs)

        preds, segments = model.predict(events=events)
        return np.asarray(preds, dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return N_VERTICES_FS5

    @property
    def output_space(self) -> str:
        return "fsaverage5"


# Register TRIBEv2
register_adapter("facebook/tribev2", TribeV2Adapter)


# ---------------------------------------------------------------------------
# Raramuri adapter (accelerated TRIBEv2 via HTTP service)
# ---------------------------------------------------------------------------
# Raramuri (https://github.com/chrisvoncsefalvay/raramuri) runs the same
# TRIBEv2 stack inside a Docker image with BF16/FP8 precision tuning,
# parallel extractors, and Parakeet transcription. Output is byte-identical
# to TRIBEv2 within BF16 rounding (~0.11% NRMSE). It is not a Python
# library — the container exposes an HTTP server on port 8765 with an
# /infer endpoint, so this adapter is an HTTP client rather than an
# in-process model wrapper.

class RaramuriAdapter(HFModelAdapter):
    """Adapter for the Raramuri accelerated TRIBEv2 inference service.

    Unlike TribeV2Adapter (which imports ``tribev2`` in-process), Raramuri
    runs as a separate container with a REST API. ``load_model`` here just
    verifies the service is reachable and warmed; ``extract_features``
    POSTs the input spec to ``/infer`` and reshapes the response to the
    (T, 20484) array the downstream JAX pipeline expects.

    Inputs dict keys (at least one of video_path / video_url required):
        - ``video_path``  : path to local video file
        - ``video_url``   : yt-dlp-supported remote URL
        - ``start_time``  : optional "HH:MM:SS" or seconds
        - ``end_time``    : optional "HH:MM:SS" or seconds
        - ``server_url``  : override base URL (default http://localhost:8765)
        - ``timeout``     : optional per-request timeout (default 900s)
    """

    @staticmethod
    def _import_requests():
        try:
            import requests
            return requests
        except ImportError:
            raise ImportError(
                "Raramuri adapter needs `requests`. Install with: pip install requests"
            )

    def load_model(self, model_id: str, cache_dir: str | None = None,
                   server_url: str = "http://localhost:8765",
                   ready_timeout: float = 600.0, **kwargs) -> Any:
        """Wait for the Raramuri server to be ready; return a client handle."""
        import time
        requests = self._import_requests()

        t0 = time.time()
        while time.time() - t0 < ready_timeout:
            try:
                r = requests.get(f"{server_url}/ready", timeout=5)
                if r.status_code == 200:
                    return {"base_url": server_url}
            except Exception:
                pass
            time.sleep(5)
        raise RuntimeError(
            f"Raramuri server at {server_url} did not become ready within "
            f"{ready_timeout}s. Check the server job logs."
        )

    def extract_features(self, model: Any, inputs: dict, **kwargs) -> np.ndarray:
        requests = self._import_requests()
        base_url = inputs.get("server_url") or model["base_url"]

        payload = {}
        for k in ("video_path", "video_url", "start_time", "end_time", "output"):
            if k in inputs and inputs[k] is not None:
                payload[k] = inputs[k]
        if "video_path" not in payload and "video_url" not in payload:
            raise ValueError("Raramuri requires video_path or video_url in inputs")

        timeout = inputs.get("timeout", 900)
        r = requests.post(f"{base_url}/infer", json=payload, timeout=timeout)
        r.raise_for_status()
        result = r.json()

        # Response shape: {"predictions": [[...]], "shape": [T, 20484], ...}
        preds = result.get("predictions")
        if preds is None and "result" in result:
            preds = result["result"].get("predictions")
        if preds is None:
            raise RuntimeError(f"Raramuri response missing 'predictions': keys={list(result.keys())}")
        return np.asarray(preds, dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return N_VERTICES_FS5

    @property
    def output_space(self) -> str:
        return "fsaverage5"


# Register Raramuri
register_adapter("raramuri", RaramuriAdapter)


# ---------------------------------------------------------------------------
# HFEncoderParams (jaxoccoli NamedTuple pattern)
# ---------------------------------------------------------------------------

class HFEncoderParams(NamedTuple):
    """Parameters for a HuggingFace encoder in the jaxoccoli factory pattern.

    Attributes
    ----------
    model_id : HuggingFace model identifier
    model : the loaded model object (None if lazy and not yet loaded)
    adapter : instantiated HFModelAdapter
    """
    model_id: str
    model: Any
    adapter: HFModelAdapter


# ---------------------------------------------------------------------------
# make_hf_encoder factory
# ---------------------------------------------------------------------------

def make_hf_encoder(
    model_id: str,
    *,
    key: jax.Array | None = None,
    cache_dir: str | None = None,
    lazy: bool = False,
    adapter: HFModelAdapter | None = None,
    **load_kwargs,
) -> tuple[HFEncoderParams, callable]:
    """Create a HuggingFace encoder following jaxoccoli's factory pattern.

    Parameters
    ----------
    model_id : HuggingFace model identifier (e.g. "facebook/tribev2")
    key : JAX PRNG key (unused by frozen models, kept for API consistency)
    cache_dir : local directory for model weights cache
    lazy : if True, defer model loading until first forward call
    adapter : optional pre-instantiated adapter (auto-detected from registry if None)
    **load_kwargs : passed to adapter.load_model

    Returns
    -------
    (HFEncoderParams, forward_fn)
        forward_fn(params, inputs) → jnp.ndarray
    """
    if adapter is None:
        adapter_cls = get_adapter(model_id)
        adapter = adapter_cls()

    if lazy:
        loaded_model = None
    else:
        loaded_model = adapter.load_model(
            model_id, cache_dir=cache_dir, **load_kwargs,
        )

    params = HFEncoderParams(
        model_id=model_id,
        model=loaded_model,
        adapter=adapter,
    )

    def forward_fn(params: HFEncoderParams, inputs: dict) -> jnp.ndarray:
        model = params.model
        if model is None:
            # Lazy loading on first call
            model = params.adapter.load_model(
                params.model_id, cache_dir=cache_dir, **load_kwargs,
            )
        features = params.adapter.extract_features(model, inputs)
        return torch_to_jax(features)

    return params, forward_fn


# ---------------------------------------------------------------------------
# Cortical projection (learnable vertex→parcel mapping)
# ---------------------------------------------------------------------------

class CorticalProjectionParams(NamedTuple):
    """Learnable parcellation weights (logit space).

    Attributes
    ----------
    logits : (n_vertices, n_parcels) raw logits, softmax-normalised at forward time
    """
    logits: jnp.ndarray


def make_cortical_projection(
    n_vertices: int,
    n_parcels: int,
    *,
    key: jax.Array,
    init: str = "random",
) -> tuple[CorticalProjectionParams, callable]:
    """Create a learnable vertex→parcel projection layer.

    The projection matrix is softmax-normalised along the vertex axis
    so that each parcel's weights sum to 1. This is differentiable
    and compatible with jax.grad/jit/vmap.

    Parameters
    ----------
    n_vertices : number of cortical vertices (e.g. 20484 for fsaverage5)
    n_parcels : number of output parcels
    key : JAX PRNG key for weight initialisation
    init : initialisation strategy:
        "random" — small normal noise (uniform softmax, learnable from scratch)
        "block" — block-diagonal seeding so each parcel starts with a
                  contiguous vertex subset (preserves spatial structure)

    Returns
    -------
    (CorticalProjectionParams, forward_fn)
        forward_fn(params, data) → parcellated data
        data shape: (n_time, n_vertices) → (n_time, n_parcels)
    """
    if init == "block":
        # Block-diagonal: each parcel gets a contiguous vertex range
        # with a strong logit bias, plus small noise for differentiability
        logits = jax.random.normal(key, (n_vertices, n_parcels)) * 0.01
        block_size = n_vertices // n_parcels
        for p in range(n_parcels):
            v_start = p * block_size
            v_end = min((p + 1) * block_size, n_vertices)
            logits = logits.at[v_start:v_end, p].add(5.0)
    else:
        logits = jax.random.normal(key, (n_vertices, n_parcels)) * 0.01

    params = CorticalProjectionParams(logits=logits)

    def forward_fn(
        params: CorticalProjectionParams,
        data: jnp.ndarray,
    ) -> jnp.ndarray:
        # Softmax along vertex axis → each parcel gets normalised weights
        weights = jax.nn.softmax(params.logits, axis=0)  # (n_vertices, n_parcels)
        # data: (n_time, n_vertices) @ weights: (n_vertices, n_parcels)
        return data @ weights

    return params, forward_fn
