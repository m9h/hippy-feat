#!/usr/bin/env python
"""Walk R1..R11 of HBN-EEG and force-download every RestingState recording.

Pure data fetch — no preprocessing, no extraction. The goal is to populate
the eegdash cache so subsequent extraction jobs (which use the same cache)
read from local disk instead of hitting OpenNeuro S3.

Routes through ``EEGDashDataset`` with the OpenNeuro ``dataset`` filter
(the typed ``EEG2025R*`` classes have a NEMAR-routing bug — see
``reference_eegdash_eeg2025_workaround.md``).

For each release:
  1. instantiate the dataset
  2. ``_ensure_raw()`` on every recording (forces braindecode/mne to pull
     the .set file from S3 and stash it)
  3. log per-release counts + elapsed time

Idempotent — recordings already in the cache are skipped.
"""
from __future__ import annotations

# Same torchaudio stub as extract_eeg_fm_acts.py — see comment there.
import sys as _sys, types as _types
from importlib.machinery import ModuleSpec as _Spec
_TA_SUB = {'functional', 'transforms', 'io', 'models',
           'pipelines', 'datasets', '_extension'}
class _AnyClass:
    def __init__(self, *a, **kw): pass
    def __call__(self, *a, **kw): return self
    def __getattr__(self, name):
        if name.startswith('__'): raise AttributeError(name)
        sub = type(name, (_AnyClass,), {})
        object.__setattr__(self, name, sub)
        return sub
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
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/data/derivatives/eegdash_cache")
    ap.add_argument("--releases", type=int, nargs="+",
                    default=list(range(1, 12)),
                    help="HBN releases to fetch (default 1..11)")
    ap.add_argument("--task", default="RestingState")
    ap.add_argument("--per-subject-cap", type=int, default=None,
                    help="(debug) cap recordings per release")
    args = ap.parse_args()

    from eegdash import EEGDashDataset
    from eegdash.const import RELEASE_TO_OPENNEURO_DATASET_MAP

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    grand_total = 0
    grand_t0 = time.time()
    for rn in args.releases:
        release_key = f"R{rn}"
        if release_key not in RELEASE_TO_OPENNEURO_DATASET_MAP:
            print(f"[skip] unknown release {release_key}", flush=True)
            continue
        openneuro_id = RELEASE_TO_OPENNEURO_DATASET_MAP[release_key]

        print(f"\n=== {release_key} ({openneuro_id}) / task={args.task} ===",
              flush=True)
        t0 = time.time()
        try:
            ds = EEGDashDataset(
                cache_dir=str(cache_dir),
                query={"dataset": openneuro_id, "task": args.task},
            )
        except Exception as e:
            print(f"[error] could not instantiate {release_key}: {e}", flush=True)
            continue

        n_recordings = len(ds.datasets)
        print(f"  {n_recordings} recordings advertised", flush=True)
        if args.per_subject_cap is not None:
            n_recordings = min(n_recordings, args.per_subject_cap)

        ok, skipped, failed = 0, 0, 0
        for i in range(n_recordings):
            rec = ds.datasets[i]
            try:
                # Force the raw download. _ensure_raw() is the lowest-level
                # entry that triggers the actual S3 fetch + cache write.
                # CRITICAL: it ALSO loads the Raw into memory and caches it
                # on rec._raw. Across 322 R4 subjects that's ~48 GB of Raw
                # objects accumulating in RAM — instant OOM. We only want
                # the file on disk, so eagerly null _raw after downloading.
                if getattr(rec, "_raw", None) is not None:
                    skipped += 1
                else:
                    rec._ensure_raw()
                    rec._raw = None         # release the loaded Raw
                    ok += 1
            except Exception as e:
                # Don't let one bad recording sink the rest of the release.
                msg = str(e).splitlines()[0][:140]
                print(f"  [fail] rec {i}: {msg}", flush=True)
                failed += 1
            if (i + 1) % 50 == 0:
                dt = time.time() - t0
                print(f"  {i + 1}/{n_recordings} processed  "
                      f"({dt:.0f}s, {(i + 1)/dt:.1f} rec/s)", flush=True)

        dt = time.time() - t0
        print(f"[{release_key} done]  ok={ok} skipped={skipped} failed={failed}  "
              f"({dt:.0f}s)", flush=True)
        grand_total += ok + skipped

    grand_dt = time.time() - grand_t0
    print(f"\n=== ALL DONE  {grand_total} recordings present "
          f"({grand_dt / 60:.1f} min) ===", flush=True)


if __name__ == "__main__":
    sys.exit(main())
