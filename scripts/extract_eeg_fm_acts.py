#!/usr/bin/env python
"""Extract block-level activations from an EEG foundation model into an .npz.

Pipeline
--------
    eegdash dataset → braindecode windows @ 200 Hz → REVE / LaBraM forward
                    → forward-hook captures block-k output
                    → flatten (B, n_patches, d_model) into (N, d_model)
                    → save activations + per-window subject metadata

Default target is the HBN-mini release (20 subjects, RestingState task) so
the smoke run completes inside a single GPU job. ``--full`` switches to
the full HBN release for the real SAE training corpus.

Run under the PyTorch NGC SIF (``/data/derivatives/containers/pytorch_26.03.sif``)
since the encoders are PyTorch. JAX is NOT needed here — SAE training is
a separate sbatch.

Output schema (single ``.npz``)
-------------------------------
    activations  : (N, d_model)  float32 — the hooked block's output, flattened
    layer        : ()            int     — which block index produced them
    d_model      : ()            int     — hidden size
    n_per_window : ()            int     — tokens per window (so SAE
                                            training can re-group if it wants)
    subject_id   : (N,)          object  — repeated subject ID per window
    age          : (N,)          float32 — subject age in years
    sex          : (N,)          object  — 'M' / 'F' / 'NA'
    p_factor / internalizing / externalizing / attention
                 : (N,)          float32 — HBN bifactor psychopathology dims
                                            (NaN if dataset lacks them)
    window_idx   : (N,)          int     — which window within the recording
"""
from __future__ import annotations

# Stub torchaudio BEFORE any braindecode import. braindecode imports many
# torchaudio symbols when its package-level __init__ registers model classes
# — including `class X(torchaudio.transforms.Spectrogram)` subclassing, so
# leaf attribute accesses must return a real *class* (subclassable), while
# `torchaudio.transforms` etc. must remain a real *module* (so the importer
# accepts `from torchaudio.transforms import Resample`).
#
# Design:
#   _AnyClass: permissive class — instantiable, callable, subclassable, and
#     auto-creates subclasses on .X attribute access.
#   _StubModule(ModuleType): __getattr__ returns either a submodule (for
#     a fixed known set) or a new _AnyClass subclass (for everything else).
import sys as _sys
import types as _types
from importlib.machinery import ModuleSpec as _Spec

_TA_SUBMODULES = {'functional', 'transforms', 'io', 'models',
                  'pipelines', 'datasets', '_extension'}

class _AnyClass:
    """Permissive class stub: instantiable, callable, has any attribute."""
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        sub = type(name, (_AnyClass,), {})
        # cache so identity checks work
        object.__setattr__(self, name, sub)
        return sub


class _StubModule(_types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        # Submodule? route to sys.modules.
        if self.__name__ == 'torchaudio' and name in _TA_SUBMODULES:
            full = f"torchaudio.{name}"
            if full in _sys.modules:
                setattr(self, name, _sys.modules[full])
                return _sys.modules[full]
        # Otherwise: synthesise a class.
        cls = type(name, (_AnyClass,), {})
        setattr(self, name, cls)
        return cls

def _install_torchaudio_stub():
    if 'torchaudio' in _sys.modules:
        return
    for name in ('torchaudio',) + tuple(f'torchaudio.{s}' for s in _TA_SUBMODULES):
        m = _StubModule(name)
        m.__spec__ = _Spec(name=name, loader=None)
        m.__spec__.submodule_search_locations = []
        m.__path__ = []
        m.__version__ = 'stub'
        _sys.modules[name] = m

_install_torchaudio_stub()

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_eegdash(release: int, mini: bool, cache_dir: str,
                 task: str | None) -> tuple:
    """Load an HBN release through ``EEGDashDataset`` with OpenNeuro routing.

    Why not the typed classes (``EEG2025R1MINI`` etc.) or the
    ``EEGChallengeDataset`` helper? In eegdash 0.7.2 those route via the
    NEMAR git-annex backend by constructing URLs like
    ``raw.githubusercontent.com/NEMARDatasets/EEG2025r1mini/...`` — repos
    that do not exist (the EEG 2025 challenge releases are mirrored to
    NEMAR's S3 bucket but the GitHub annex pointer tree was never pushed).
    ``RELEASE_TO_OPENNEURO_DATASET_MAP`` in the same eegdash module maps
    each release to its OpenNeuro dataset (R1 → ds005505 …) and OpenNeuro
    *does* expose its S3 bucket. So we go direct: ``EEGDashDataset`` with
    ``query={"dataset": "ds005505", "task": ..., "subject": {"$in": [...]}}``.
    """
    try:
        from eegdash import EEGDashDataset
        from eegdash.const import (
            RELEASE_TO_OPENNEURO_DATASET_MAP,
            SUBJECT_MINI_RELEASE_MAP,
        )
    except ImportError as e:
        raise ImportError(
            "eegdash not installed. Run: pip install 'eegdash>=0.2' "
            "(inside the PyTorch SIF or in your venv)."
        ) from e

    release_key = f"R{release}"
    if release_key not in RELEASE_TO_OPENNEURO_DATASET_MAP:
        raise ValueError(
            f"Unknown HBN release {release_key!r}. "
            f"Known: {list(RELEASE_TO_OPENNEURO_DATASET_MAP.keys())}"
        )
    openneuro_id = RELEASE_TO_OPENNEURO_DATASET_MAP[release_key]

    query: dict = {"dataset": openneuro_id}
    if task:
        query["task"] = task
    if mini:
        if release_key not in SUBJECT_MINI_RELEASE_MAP:
            raise ValueError(
                f"No mini-subject list for {release_key}. Pass --full to "
                f"download all subjects of {openneuro_id}."
            )
        query["subject"] = {"$in": list(SUBJECT_MINI_RELEASE_MAP[release_key])}

    n_subjects = len(query["subject"]["$in"]) if mini else "all"
    print(f"[eegdash] EEGDashDataset(cache_dir={cache_dir!r}, query="
          f"{{dataset: {openneuro_id!r}, task: {task!r}, "
          f"mini_subjects: {n_subjects}}})", flush=True)
    ds = EEGDashDataset(cache_dir=cache_dir, query=query)
    desc = getattr(ds, "description", None)
    return ds, desc


def _drop_broken_subjects(concat_ds, min_duration_s: float = 6.0):
    """Filter out recordings braindecode preprocess() would crash on.

    Two failure modes we've hit across HBN releases:

      1. ``DataIntegrityError`` — eegdash registry advertises a RestingState
         file but the .set isn't on the OpenNeuro mirror (R3 had one).
      2. ``ValueError: picks (NoneNone, ...) yielded no channels`` — MNE
         can't find any data-typed channels in ``raw.info`` when filter()
         tries to default-pick (R10 had at least one).

    Both kill the whole joblib preprocess pass. We probe each recording
    upfront with ``_ensure_raw()`` then ``mne.pick_types(eeg=True)``,
    drop the failures, and release the loaded Raw to keep peak memory
    bounded to ~1 Raw at a time across the validation loop.
    """
    # Validate by trying ``_ensure_raw()`` — catches DataIntegrityError
    # (missing .set on S3, e.g. one R3 subject) and FileNotFoundError.
    # We deliberately do NOT inspect channel types here. eegdash's
    # ``EEGDashRaw`` types HBN R10 channels as "misc" instead of "eeg"
    # (same .set file loaded via plain mne.io.read_raw_eeglab is typed
    # "eeg" — likely a channels.tsv interpretation difference), and any
    # `pick_types(eeg=True)` / `_picks_to_idx("data_or_ica")` check would
    # drop all 479 R10 subjects spuriously. The downstream `filter` step
    # is run with `picks="all"` instead, which handles "misc" channels.
    valid_idx = []
    dropped = []
    for i, ds_rec in enumerate(concat_ds.datasets):
        try:
            ds_rec._ensure_raw()
            raw = ds_rec.raw
            duration = raw.n_times / raw.info["sfreq"]
            if duration < min_duration_s:
                dropped.append((i, f"too short: {duration:.2f}s "
                                 f"< {min_duration_s}s window+margin"))
                ds_rec._raw = None
                continue
            valid_idx.append(i)
        except Exception as e:
            msg = str(e).splitlines()[0][:120]
            dropped.append((i, msg))
            try:
                ds_rec._raw = None
            except Exception:
                pass
    if dropped:
        print(f"  [drop] {len(dropped)} broken recording(s):", flush=True)
        for i, msg in dropped[:5]:
            print(f"    rec {i}: {msg}", flush=True)
        if len(dropped) > 5:
            print(f"    … and {len(dropped) - 5} more", flush=True)
        # braindecode's split returns a dict keyed by str
        concat_ds = concat_ds.split(valid_idx)["0"]
    return concat_ds


def make_windows(concat_ds, *, win_seconds: float, target_sfreq: float):
    """Resample → bandpass → slice into fixed-length windows.

    Task filtering is upstream — the eegdash class is built with a
    ``query={"task": ...}`` constructor kwarg, so ``concat_ds`` here
    already contains only the requested task's recordings.

    The braindecode preprocessing chain is pure CPU; we keep it minimal
    because REVE/LaBraM expect raw-ish EEG at 200 Hz, 0.5–50 Hz bandpass.
    """
    from braindecode.preprocessing import (
        Preprocessor, preprocess, create_fixed_length_windows,
    )

    # Require enough samples for at least one window plus a bit of margin.
    concat_ds = _drop_broken_subjects(concat_ds, min_duration_s=win_seconds * 1.2)

    # Match REVE's pretraining recipe (NeuralBench reve.yaml): filter
    # [0.5, 99.5] Hz, not [0.5, 50] — REVE was trained on the gamma band.
    # StandardScaler + clamp 15 happen at batch-time in extract_all() so
    # we can vectorise across the GPU and don't need a braindecode
    # custom-preprocessor wrapper for the per-channel z-score.
    # ``picks="all"`` so eegdash's misc-typed channels (R10) get filtered
    # too — filter()'s default picks="data_or_ica" excludes them.
    preprocs = [
        Preprocessor("resample", sfreq=target_sfreq),
        Preprocessor("filter", l_freq=0.5, h_freq=99.5, picks="all"),
    ]
    preprocess(concat_ds, preprocs)

    win_samples = int(win_seconds * target_sfreq)
    return create_fixed_length_windows(
        concat_ds,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=win_samples,
        window_stride_samples=win_samples,
        drop_last_window=True,
        preload=False,
    )


# ---------------------------------------------------------------------------
# Encoder wiring
# ---------------------------------------------------------------------------

def _shim_jaxoccoli_package():
    """Pre-register ``jaxoccoli`` as a bare namespace package so submodule
    imports don't trigger ``jaxoccoli/__init__.py`` (which eagerly imports
    every JAX/nibabel-dependent module in the library). The PyTorch SIF
    has neither of those by default and we only need the two adapter
    submodules — both pure-PyTorch — so this shim avoids installing
    transitive deps that this script never uses.
    """
    import sys
    import os
    import types
    if "jaxoccoli" in sys.modules:
        return
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_dir = os.path.join(project_root, "jaxoccoli")
    shim = types.ModuleType("jaxoccoli")
    shim.__path__ = [pkg_dir]  # makes submodule lookup work
    shim.__file__ = os.path.join(pkg_dir, "__init__.py")
    sys.modules["jaxoccoli"] = shim


def build_adapter(model_id: str, layer: int):
    """Return a loaded ``HFModelAdapter`` and its loaded model handle."""
    _shim_jaxoccoli_package()
    from jaxoccoli.eeg_fm import REVEAdapter, LaBraMAdapter, REVE_BASE_ID

    if model_id == REVE_BASE_ID:
        adapter = REVEAdapter(layer=layer)
    elif model_id.startswith("labram"):
        adapter = LaBraMAdapter(layer=layer)
    else:
        raise ValueError(
            f"Unknown EEG FM model_id={model_id!r}. Supported: "
            f"'{REVE_BASE_ID}', 'labram-base'"
        )
    loaded = adapter.load_model(model_id)
    return adapter, loaded


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

HBN_FIELDS = ("age", "sex", "p_factor", "internalizing",
              "externalizing", "attention")


def subject_metadata(desc, sub_id) -> dict:
    """Pull HBN-style metadata for a given subject_id from the description df."""
    out = {f: np.nan for f in HBN_FIELDS}
    out["sex"] = "NA"
    if desc is None:
        return out
    try:
        row = desc[desc["subject"].astype(str) == str(sub_id)].iloc[0]
    except (KeyError, IndexError, AttributeError):
        return out
    for f in HBN_FIELDS:
        if f in row.index:
            v = row[f]
            if f == "sex":
                out[f] = str(v) if v is not None else "NA"
            else:
                try:
                    out[f] = float(v)
                except (TypeError, ValueError):
                    out[f] = np.nan
    return out


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def extract_all(windows_ds, adapter, loaded_model, *,
                desc, batch_size: int, max_windows: int | None) -> dict:
    """Iterate windows, run encoder, write activations + metadata into
    pre-allocated arrays.

    Previous version accumulated all batches into a Python list and
    ``np.concatenate``-d at the end — that transiently doubles peak memory
    (input chunks AND output array alive simultaneously). For R4 / R5
    with ~17 M tokens × 512 floats = ~34 GB activations, the peak hit
    ~70 GB and killed the job. We now pre-allocate after the first
    batch tells us ``n_per_window`` and write each batch slice in place.
    Single allocation; no concat doubling.
    """
    n_windows = len(windows_ds)
    if max_windows is not None:
        n_windows = min(n_windows, max_windows)
    print(f"[extract] {n_windows} windows total, batch={batch_size}",
          flush=True)

    acts_full = None         # (N_total_tokens, d_model) float32 — pre-allocated
    meta_arrays = None       # dict of pre-allocated metadata arrays
    write_idx = 0
    n_per_window = None
    d_model = None

    buf_x, buf_ch_names, buf_meta = [], None, []
    t0 = time.time()
    for i in range(n_windows):
        x, _y, ind = windows_ds[i]                                  # (C, T)
        try:
            sub_id = windows_ds.datasets[ind[0]].description["subject"]
        except Exception:
            sub_id = f"sub-{ind[0]:04d}"
        if buf_ch_names is None:
            ds_record = windows_ds.datasets[ind[0]]
            raw = getattr(ds_record, "raw", None)
            if raw is None:
                raise AttributeError(
                    f"could not find a .raw attribute on "
                    f"{type(ds_record).__name__} — braindecode API may have "
                    f"shifted again"
                )
            buf_ch_names = list(raw.ch_names)
        buf_x.append(np.asarray(x, dtype=np.float32))
        buf_meta.append((sub_id, i))

        if len(buf_x) >= batch_size or i == n_windows - 1:
            batch = np.stack(buf_x, axis=0)                       # (B, C, T)
            # Per-channel z-score then clamp ±15 — NeuralBench reve.yaml
            # `scaler: StandardScaler` + `clamp: 15`. REVE was trained on
            # data in this distribution; without it we feed it OOD inputs.
            mu = batch.mean(axis=-1, keepdims=True)
            sigma = batch.std(axis=-1, keepdims=True) + 1e-8
            batch = (batch - mu) / sigma
            np.clip(batch, -15.0, 15.0, out=batch)
            feats = adapter.extract_features(
                loaded_model,
                {"eeg": batch, "electrode_names": buf_ch_names,
                 "ch_names": buf_ch_names},
            )

            # On first batch: learn shape, allocate output buffers.
            if acts_full is None:
                if feats.ndim == 3:
                    _, P, D = feats.shape
                else:
                    P, D = 1, feats.shape[-1]
                n_per_window = P
                d_model = D
                total_tokens = n_windows * P
                print(f"  [alloc] (N={total_tokens}, d_model={D}) → "
                      f"{total_tokens * D * 4 / 1e9:.1f} GB float32", flush=True)
                acts_full = np.empty((total_tokens, D), dtype=np.float32)
                meta_arrays = {
                    "subject_id": np.empty(total_tokens, dtype=object),
                    "window_idx": np.empty(total_tokens, dtype=np.int32),
                }
                for f in HBN_FIELDS:
                    if f == "sex":
                        meta_arrays[f] = np.empty(total_tokens, dtype=object)
                    else:
                        meta_arrays[f] = np.empty(total_tokens, dtype=np.float32)

            # Flatten this batch's features and write into the slice.
            if feats.ndim == 3:
                B, P, D = feats.shape
                feats_flat = feats.reshape(B * P, D)
            else:
                feats_flat = feats.reshape(feats.shape[0], -1)
                P = 1
            n = feats_flat.shape[0]
            acts_full[write_idx:write_idx + n] = feats_flat
            # Per-token metadata, also written into pre-allocated slices.
            md_cache: dict[str, dict] = {}
            for j, (sub_id, w_idx) in enumerate(buf_meta):
                if sub_id not in md_cache:
                    md_cache[sub_id] = subject_metadata(desc, sub_id)
                md = md_cache[sub_id]
                lo = write_idx + j * P
                hi = lo + P
                meta_arrays["subject_id"][lo:hi] = str(sub_id)
                meta_arrays["window_idx"][lo:hi] = w_idx
                for f in HBN_FIELDS:
                    meta_arrays[f][lo:hi] = md[f]
            write_idx += n
            buf_x, buf_meta = [], []

        if i % 50 == 0 and i > 0:
            dt = time.time() - t0
            print(f"  [{i}/{n_windows}] {dt:.1f}s "
                  f"({i / dt:.1f} win/s)", flush=True)

    # Trim trailing unused slice if the last batch was a partial.
    if write_idx < acts_full.shape[0]:
        acts_full = acts_full[:write_idx]
        for k in meta_arrays:
            meta_arrays[k] = meta_arrays[k][:write_idx]

    out = {
        "activations": acts_full,
        "layer": np.int32(adapter.layer),
        "d_model": np.int32(d_model),
        "n_per_window": np.int32(n_per_window),
    }
    for k, v in meta_arrays.items():
        out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="brain-bzh/reve-base",
                    help="HF model id ('brain-bzh/reve-base' or 'labram-base')")
    ap.add_argument("--layer", type=int, default=-1,
                    help="Transformer block to hook (negative = from end)")
    ap.add_argument("--release", type=int, default=1, choices=range(1, 12),
                    metavar="N", help="HBN release number 1..11 (default 1)")
    ap.add_argument("--full", action="store_true",
                    help="Use the full release (e.g. EEG2025R1, 136 subj) "
                         "instead of the MINI variant (20 subj)")
    ap.add_argument("--task", default="RestingState",
                    help="HBN task code: RestingState | DespicableMe | "
                         "DiaryOfAWimpyKid | FunwithFractals | ThePresent | "
                         "contrastChangeDetection | seqLearning6target | "
                         "seqLearning8target | surroundSupp | symbolSearch")
    ap.add_argument("--cache-dir", default="/data/derivatives/eegdash_cache")
    ap.add_argument("--out", required=True,
                    help="Output .npz path")
    ap.add_argument("--win-seconds", type=float, default=5.0)
    ap.add_argument("--target-sfreq", type=float, default=200.0,
                    help="Resample to this rate (REVE requires 200 Hz)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="Cap for smoke runs; default = all")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    concat_ds, desc = load_eegdash(
        release=args.release,
        mini=not args.full,
        cache_dir=args.cache_dir,
        task=args.task,
    )
    windows_ds = make_windows(
        concat_ds,
        win_seconds=args.win_seconds,
        target_sfreq=args.target_sfreq,
    )

    adapter, loaded = build_adapter(args.model, args.layer)
    result = extract_all(
        windows_ds, adapter, loaded,
        desc=desc, batch_size=args.batch_size, max_windows=args.max_windows,
    )

    np.savez_compressed(out_path, **result)
    print(f"[done] wrote {out_path}  "
          f"({result['activations'].shape[0]} tokens, "
          f"d_model={int(result['d_model'])})", flush=True)

    # Sidecar JSON with run config (so SAE training knows what produced this)
    sidecar = out_path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump({
            "model": args.model,
            "layer": int(args.layer),
            "release": args.release,
            "full": bool(args.full),
            "task": args.task,
            "win_seconds": args.win_seconds,
            "target_sfreq": args.target_sfreq,
            "n_tokens": int(result["activations"].shape[0]),
            "d_model": int(result["d_model"]),
            "n_per_window": int(result["n_per_window"]),
        }, f, indent=2)
    print(f"[done] sidecar at {sidecar}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
