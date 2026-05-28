#!/usr/bin/env python
"""Post-hoc patch HBN bifactor + demographic metadata into an extracted
activations .npz that was produced when participants.tsv was missing.

Job 1466 (R10) hit this: extract_eeg_fm_acts_local_bids.py ran ~1.5 min
before participants.tsv finished downloading, so `desc` was empty and
every metadata field is NaN/NA. We re-read participants.tsv now and
rewrite the .npz with corrected per-token metadata. Activations are
untouched.

Usage:
    python scripts/patch_npz_metadata.py \
        --npz /data/derivatives/eeg_sae/acts/brain-bzh_reve-base_L6_EEG2025R10_RestingState.npz \
        --participants /data/derivatives/eegdash_cache/ds005515/participants.tsv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


HBN_FIELDS = ("age", "sex", "p_factor", "internalizing",
              "externalizing", "attention")


def load_participants(path: Path) -> dict:
    import pandas as pd
    df = pd.read_csv(path, sep="\t", dtype=str)
    out = {}
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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--participants", type=Path, required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    print(f"[load] {args.npz}", flush=True)
    t0 = time.time()
    src = np.load(args.npz, allow_pickle=True)
    print(f"  keys: {list(src.keys())}", flush=True)
    sub_ids = src["subject_id"]
    n_tokens = len(sub_ids)
    print(f"  n_tokens: {n_tokens:,}", flush=True)

    desc = load_participants(args.participants)
    print(f"[load] {args.participants}: {len(desc)} subjects", flush=True)

    # Build per-token metadata by indexing through desc.
    new_meta = {f: None for f in HBN_FIELDS}
    new_meta["age"] = np.full(n_tokens, np.nan, dtype=np.float32)
    new_meta["sex"] = np.empty(n_tokens, dtype=object)
    new_meta["sex"][:] = "NA"
    for f in ("p_factor", "internalizing", "externalizing", "attention"):
        new_meta[f] = np.full(n_tokens, np.nan, dtype=np.float32)

    # Fill by unique-subject pass — many fewer lookups than per-token.
    unique = np.unique(sub_ids)
    print(f"  unique subjects in npz: {len(unique)}", flush=True)
    missing = 0
    for sub in unique:
        m = desc.get(str(sub))
        if m is None:
            missing += 1
            continue
        mask = sub_ids == sub
        for f in HBN_FIELDS:
            new_meta[f][mask] = m[f]
    print(f"  missing from participants.tsv: {missing}", flush=True)
    print(f"  age non-NaN after patch: {(~np.isnan(new_meta['age'])).mean():.3f}",
          flush=True)
    print(f"  sex F/M/NA: {(new_meta['sex']=='F').sum()}/"
          f"{(new_meta['sex']=='M').sum()}/{(new_meta['sex']=='NA').sum()}",
          flush=True)

    if args.dry_run:
        print("[dry-run] not writing", flush=True)
        return 0

    # Atomic rewrite — write to .tmp.npz then rename. ~47 GB, several min.
    tmp = args.npz.with_suffix(".tmp.npz")
    print(f"[write] {tmp}", flush=True)
    out_dict = {k: src[k] for k in src.keys() if k not in HBN_FIELDS}
    out_dict.update(new_meta)
    np.savez_compressed(tmp, **out_dict)
    tmp.rename(args.npz)
    dt = time.time() - t0
    print(f"[done] {args.npz}  ({dt:.1f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
