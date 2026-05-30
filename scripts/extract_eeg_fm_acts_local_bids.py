#!/usr/bin/env python
"""Extract REVE/LaBraM activations from LOCAL BIDS data — bypass eegdash.

Drop-in alternative to ``extract_eeg_fm_acts.py`` for releases whose
eegdash registry returns 0 subjects (R7 = ds005511 and R11 = ds005516).
We downloaded their raw .set files via ``download_hbn_s3_direct.py``
into the eegdash cache dir, in BIDS layout. This script:

  1. Walks the local BIDS tree for the chosen ds00xxxx + task
  2. Reads each ``sub-X/eeg/sub-X_task-RestingState_eeg.set`` via mne_bids
  3. Wraps the resulting mne.Raws into a braindecode ``BaseConcatDataset``
  4. Hands that to the existing windowing + REVE-encode pipeline

Output schema matches ``extract_eeg_fm_acts.py`` so downstream SAE +
concept-probe scripts treat the result identically.
"""
from __future__ import annotations

# Same torchaudio + jaxoccoli stubs as the main extract script.
import sys as _sys
import types as _types
from importlib.machinery import ModuleSpec as _Spec

_TA_SUB = {'functional', 'transforms', 'io', 'models',
           'pipelines', 'datasets', '_extension'}

class _AnyClass:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        sub = type(name, (_AnyClass,), {})
        object.__setattr__(self, name, sub); return sub

class _StubModule(_types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        if self.__name__ == 'torchaudio' and name in _TA_SUB:
            full = f"torchaudio.{name}"
            if full in _sys.modules:
                setattr(self, name, _sys.modules[full])
                return _sys.modules[full]
        cls = type(name, (_AnyClass,), {})
        setattr(self, name, cls)
        return cls

if 'torchaudio' not in _sys.modules:
    for _n in ('torchaudio',) + tuple(f'torchaudio.{s}' for s in _TA_SUB):
        _m = _StubModule(_n)
        _m.__spec__ = _Spec(name=_n, loader=None)
        _m.__spec__.submodule_search_locations = []
        _m.__path__ = []; _m.__version__ = 'stub'
        _sys.modules[_n] = _m

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


HBN_RELEASE_TO_DS = {
    1: "ds005505", 2: "ds005506", 3: "ds005507", 4: "ds005508",
    5: "ds005509", 6: "ds005510", 7: "ds005511", 8: "ds005512",
    9: "ds005514", 10: "ds005515", 11: "ds005516",
}

HBN_FIELDS = ("age", "sex", "p_factor", "internalizing",
              "externalizing", "attention")


def _load_participants_metadata(bids_root: Path) -> dict:
    """Read participants.tsv (root of BIDS dataset) for subject demographics.

    HBN's participants.tsv columns we care about: participant_id, age, sex,
    plus bifactor columns (p_factor, internalizing, externalizing, attention)
    when /phenotype/HBN_Pheno_Scores.tsv exists.
    """
    out: dict[str, dict] = {}
    ptsv = bids_root / "participants.tsv"
    if not ptsv.exists():
        return out
    import pandas as pd
    df = pd.read_csv(ptsv, sep="\t", dtype=str)
    # Try to also merge bifactor scores from phenotype/
    pheno_dir = bids_root / "phenotype"
    if pheno_dir.exists():
        for f in pheno_dir.glob("*.tsv"):
            try:
                pdf = pd.read_csv(f, sep="\t", dtype=str)
                # Find an ID column to merge on
                id_col = next((c for c in ("participant_id", "Subject", "ID")
                              if c in pdf.columns), None)
                if id_col and "participant_id" in df.columns:
                    pdf = pdf.rename(columns={id_col: "participant_id"})
                    df = df.merge(pdf, on="participant_id", how="left",
                                   suffixes=("", f"_{f.stem}"))
            except Exception:
                pass

    for _, row in df.iterrows():
        pid = str(row.get("participant_id", "")).replace("sub-", "")
        if not pid:
            continue
        m = {f: np.nan for f in HBN_FIELDS}
        m["sex"] = "NA"
        for f in HBN_FIELDS:
            if f in row.index:
                v = row[f]
                if f == "sex":
                    m[f] = str(v) if v not in (None, "", "n/a") else "NA"
                else:
                    try:
                        m[f] = float(v)
                    except (TypeError, ValueError):
                        m[f] = np.nan
        out[pid] = m
    return out


def load_local_bids(release: int, cache_dir: str, task: str
                     ) -> tuple:
    """Build a braindecode BaseConcatDataset from local BIDS .set files.

    Bypasses eegdash entirely; relies on mne_bids to follow the BIDS
    convention. Returns (concat_ds, description_dict).
    """
    ds_id = HBN_RELEASE_TO_DS[release]
    bids_root = Path(cache_dir) / ds_id
    if not bids_root.exists():
        raise FileNotFoundError(f"BIDS root {bids_root} does not exist")

    # Find every <sub-X>/eeg/<sub-X>_task-{task}_[run-N_]eeg.set
    # The middle `*` matches both single-run files (e.g. RestingState:
    # sub-X_task-RestingState_eeg.set) and multi-run task files (e.g.
    # contrastChangeDetection: sub-X_task-contrastChangeDetection_run-1_eeg.set).
    set_files = sorted(bids_root.glob(
        f"sub-*/eeg/sub-*_task-{task}_*eeg.set"
    ))
    # Some tasks have no per-run files; the trailing `*eeg.set` will then
    # also try to match `_task-X_eeg.set` with `*` consuming empty. pathlib
    # glob handles that; we still need to dedupe in case both forms exist.
    set_files = sorted(set(set_files))
    if not set_files:
        raise FileNotFoundError(
            f"No matching .set files under {bids_root}/sub-*/eeg/ "
            f"for task={task}"
        )
    print(f"[local-bids] {ds_id} / task={task}: {len(set_files)} .set files",
          flush=True)

    desc = _load_participants_metadata(bids_root)

    import mne
    # braindecode 1.5.1 keeps these classes in .base but no longer
    # re-exports BaseDataset from the package root (BaseConcatDataset
    # is still exposed). Import from the submodule directly.
    from braindecode.datasets.base import BaseDataset, BaseConcatDataset

    ds_list = []
    skipped = 0
    for set_path in set_files:
        sub_id = set_path.stem.split("_")[0].replace("sub-", "")
        try:
            raw = mne.io.read_raw_eeglab(
                str(set_path), preload=False, verbose="ERROR",
            )
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"  [skip] {sub_id}: {type(e).__name__}: "
                      f"{str(e)[:100]}", flush=True)
            continue
        # braindecode BaseDataset wraps a raw + a description Series
        import pandas as pd
        desc_series = pd.Series({
            "subject": sub_id,
            "task": task,
            "dataset": ds_id,
        })
        ds_list.append(BaseDataset(raw=raw, description=desc_series))
    if skipped:
        print(f"  [skip] {skipped} unreadable .set files", flush=True)

    concat = BaseConcatDataset(ds_list)
    return concat, desc


def _drop_broken_subjects_local(concat_ds, min_duration_s: float = 6.0):
    """Lightweight validator — header-only checks, no preload.

    Three filters:
      1. Duration >= window+margin (need at least one window per subject).
      2. Channel count == modal count across the release. HBN subjects can
         have 128/129/varied channel counts depending on cap version;
         np.stack() in the extract loop fails on heterogeneous shapes.
      3. Anything that raises during header access (rare but possible).

    We use the unloaded header (info + n_times + ch_names) — no
    raw.load_data() — so peak memory stays at ~one Raw at a time.
    """
    from collections import Counter

    # First pass: tally channel counts to find the mode.
    nchan_counts: Counter = Counter()
    for ds_rec in concat_ds.datasets:
        try:
            nchan_counts[len(ds_rec.raw.ch_names)] += 1
        except Exception:
            pass
    target_nchan = nchan_counts.most_common(1)[0][0] if nchan_counts else None
    if target_nchan and len(nchan_counts) > 1:
        print(f"  [info] channel-count histogram: {dict(nchan_counts)}; "
              f"keeping {target_nchan}-channel recordings", flush=True)

    valid_idx, dropped = [], []
    for i, ds_rec in enumerate(concat_ds.datasets):
        try:
            raw = ds_rec.raw
            duration = raw.n_times / raw.info["sfreq"]
            nchan = len(raw.ch_names)
            if duration < min_duration_s:
                dropped.append((i, f"too short: {duration:.2f}s "
                                 f"< {min_duration_s}s window+margin"))
                continue
            if target_nchan and nchan != target_nchan:
                dropped.append((i, f"n_channels={nchan} != mode {target_nchan}"))
                continue
            valid_idx.append(i)
        except Exception as e:
            dropped.append((i, str(e).splitlines()[0][:120]))
    if dropped:
        print(f"  [drop] {len(dropped)} broken recording(s)", flush=True)
        for i, msg in dropped[:5]:
            print(f"    rec {i}: {msg}", flush=True)
        if len(dropped) > 5:
            print(f"    … and {len(dropped) - 5} more", flush=True)
        concat_ds = concat_ds.split(valid_idx)["0"]
    return concat_ds


def _estimate_n_windows(concat_ds, *, win_samples: int, target_sfreq: float
                         ) -> int:
    """Header-only estimate of total windows across all surviving subjects.

    Used to pre-allocate the activation buffer before streaming. Matches
    braindecode's ``create_fixed_length_windows(drop_last_window=True,
    stride=win_samples)`` math: ``n_w = n_resampled // win_samples``.
    Off-by-one from MNE's resample rounding is trimmed at the end.
    """
    total = 0
    for ds_rec in concat_ds.datasets:
        raw = ds_rec.raw  # header-only via preload=False
        n_resampled = int(raw.n_times * target_sfreq / raw.info["sfreq"])
        total += n_resampled // win_samples
    return total


def extract_streaming(concat_ds, adapter, loaded_model, *,
                      desc, batch_size: int, win_seconds: float,
                      target_sfreq: float, max_windows: int | None) -> dict:
    """Per-subject streaming extractor: bounded memory regardless of
    release size.

    OOM root cause: ``braindecode.preprocess(concat_ds, ...)`` on a 479-
    subject release simultaneously preloads + resamples + filters every
    Raw (>100 GB peak resident on R10). This iterates subjects one at a
    time — load → resample → filter → window-slice → encode → drop —
    so peak memory stays at ~1 Raw + the pre-allocated acts buffer.
    """
    import mne

    concat_ds = _drop_broken_subjects_local(
        concat_ds, min_duration_s=win_seconds * 1.2,
    )
    win_samples = int(win_seconds * target_sfreq)
    total_windows = _estimate_n_windows(
        concat_ds, win_samples=win_samples, target_sfreq=target_sfreq,
    )
    if max_windows is not None:
        total_windows = min(total_windows, max_windows)
    print(f"[stream] {len(concat_ds.datasets)} subjects, "
          f"~{total_windows} windows total, batch={batch_size}", flush=True)

    HBN_FIELDS = ("age", "sex", "p_factor", "internalizing",
                  "externalizing", "attention")

    acts_full = None
    meta_arrays = None
    write_idx = 0
    n_per_window = None
    d_model = None
    t0 = time.time()
    windows_seen = 0

    for sub_i, ds_rec in enumerate(concat_ds.datasets):
        if max_windows is not None and windows_seen >= max_windows:
            break
        sub_id = ds_rec.description.get("subject",
                                         f"sub-{sub_i:04d}")
        try:
            raw = ds_rec.raw.copy().load_data(verbose="ERROR")
            raw.resample(target_sfreq, verbose="ERROR")
            raw.filter(l_freq=0.5, h_freq=99.5, picks="all",
                       verbose="ERROR")
        except Exception as e:
            print(f"  [skip] {sub_id}: {type(e).__name__}: "
                  f"{str(e)[:120]}", flush=True)
            continue
        data = raw.get_data().astype(np.float32, copy=False)  # (C, T)
        ch_names = list(raw.ch_names)
        n_w = data.shape[1] // win_samples
        if n_w == 0:
            del raw, data
            continue

        # Slice into (n_w, C, win_samples) view without a copy.
        usable = data[:, :n_w * win_samples]
        windows = usable.reshape(usable.shape[0], n_w, win_samples
                                 ).transpose(1, 0, 2)  # (n_w, C, T)

        md = None  # lazy: only fetch metadata once per subject

        for batch_start in range(0, n_w, batch_size):
            if max_windows is not None and windows_seen >= max_windows:
                break
            batch_end = min(batch_start + batch_size, n_w)
            batch = windows[batch_start:batch_end].copy()           # (B, C, T)
            mu = batch.mean(axis=-1, keepdims=True)
            sigma = batch.std(axis=-1, keepdims=True) + 1e-8
            batch = (batch - mu) / sigma
            np.clip(batch, -15.0, 15.0, out=batch)
            feats = adapter.extract_features(
                loaded_model,
                {"eeg": batch, "electrode_names": ch_names,
                 "ch_names": ch_names},
            )

            if acts_full is None:
                if feats.ndim == 3:
                    _, P, D = feats.shape
                else:
                    P, D = 1, feats.shape[-1]
                n_per_window = P
                d_model = D
                buf_tokens = total_windows * P
                gb = buf_tokens * D * 4 / 1e9
                print(f"  [alloc] (N={buf_tokens}, d_model={D}) → "
                      f"{gb:.1f} GB float32", flush=True)
                acts_full = np.empty((buf_tokens, D), dtype=np.float32)
                meta_arrays = {
                    "subject_id": np.empty(buf_tokens, dtype=object),
                    "window_idx": np.empty(buf_tokens, dtype=np.int32),
                }
                for f in HBN_FIELDS:
                    if f == "sex":
                        meta_arrays[f] = np.empty(buf_tokens, dtype=object)
                    else:
                        meta_arrays[f] = np.empty(buf_tokens, dtype=np.float32)

            if feats.ndim == 3:
                B, P, D = feats.shape
                feats_flat = feats.reshape(B * P, D)
            else:
                feats_flat = feats.reshape(feats.shape[0], -1)
                P = 1
            n = feats_flat.shape[0]
            if write_idx + n > acts_full.shape[0]:
                # Resample-rounding overshoot: grow buffer by 10%.
                grow = max(n, int(acts_full.shape[0] * 0.1))
                acts_full = np.concatenate(
                    [acts_full, np.empty((grow, d_model), dtype=np.float32)],
                    axis=0,
                )
                for k, v in list(meta_arrays.items()):
                    pad_dtype = v.dtype
                    meta_arrays[k] = np.concatenate(
                        [v, np.empty(grow, dtype=pad_dtype)], axis=0,
                    )
            acts_full[write_idx:write_idx + n] = feats_flat
            if md is None:
                md = _subject_metadata(desc, sub_id)
            for j in range(batch_end - batch_start):
                w_idx = batch_start + j
                lo = write_idx + j * P
                hi = lo + P
                meta_arrays["subject_id"][lo:hi] = str(sub_id)
                meta_arrays["window_idx"][lo:hi] = w_idx
                for f in HBN_FIELDS:
                    meta_arrays[f][lo:hi] = md[f]
            write_idx += n
            windows_seen += (batch_end - batch_start)

        del raw, data, usable, windows
        if sub_i % 25 == 0 and sub_i > 0:
            dt = time.time() - t0
            rate = windows_seen / dt if dt > 0 else 0.0
            print(f"  [{sub_i}/{len(concat_ds.datasets)} subj | "
                  f"{windows_seen} win] {dt:.1f}s ({rate:.1f} win/s)",
                  flush=True)

    if acts_full is None:
        raise RuntimeError(
            "no surviving windows — every subject failed to load or window"
        )
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


def _subject_metadata(desc, sub_id) -> dict:
    """Pull HBN-style metadata for a subject from the per-pid dict built
    by ``_load_participants_metadata``. Falls back to NaNs."""
    HBN_FIELDS = ("age", "sex", "p_factor", "internalizing",
                  "externalizing", "attention")
    out = {f: np.nan for f in HBN_FIELDS}
    out["sex"] = "NA"
    if not desc:
        return out
    m = desc.get(str(sub_id)) or desc.get(f"sub-{sub_id}")
    if m is None:
        return out
    for f in HBN_FIELDS:
        if f in m:
            out[f] = m[f]
    return out


def main():
    # Reuse the bulk of extract_eeg_fm_acts.py's main() by importing it
    # AFTER our stubs are installed. We just replace the `load_eegdash`
    # step with `load_local_bids` and let the rest run.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import extract_eeg_fm_acts as _e

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="brain-bzh/reve-base")
    ap.add_argument("--layer", type=int, default=-1)
    ap.add_argument("--release", type=int, required=True,
                    choices=range(1, 12), metavar="N")
    ap.add_argument("--task", default="RestingState")
    ap.add_argument("--cache-dir", default="/data/derivatives/eegdash_cache")
    ap.add_argument("--out", required=True)
    ap.add_argument("--win-seconds", type=float, default=5.0)
    ap.add_argument("--target-sfreq", type=float, default=200.0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-windows", type=int, default=None)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    concat_ds, desc = load_local_bids(
        release=args.release, cache_dir=args.cache_dir, task=args.task,
    )

    adapter, loaded = _e.build_adapter(args.model, args.layer)
    result = extract_streaming(
        concat_ds, adapter, loaded,
        desc=desc, batch_size=args.batch_size,
        win_seconds=args.win_seconds, target_sfreq=args.target_sfreq,
        max_windows=args.max_windows,
    )

    np.savez_compressed(out_path, **result)
    print(f"[done] wrote {out_path}  "
          f"({result['activations'].shape[0]} tokens, "
          f"d_model={int(result['d_model'])})", flush=True)
    sidecar = out_path.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump({
            "model": args.model,
            "layer": int(args.layer),
            "release": args.release,
            "full": True,
            "task": args.task,
            "win_seconds": args.win_seconds,
            "target_sfreq": args.target_sfreq,
            "n_tokens": int(result["activations"].shape[0]),
            "d_model": int(result["d_model"]),
            "n_per_window": int(result["n_per_window"]),
            "loader": "local_bids",
        }, f, indent=2)
    print(f"[done] sidecar at {sidecar}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
