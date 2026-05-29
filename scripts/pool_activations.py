#!/usr/bin/env python
"""Pool a random subsample from each release's activations into one .npz.

Used to build the Phase C "all-HBN" training corpus for the SAE without
loading all 285 GB of release-level .npz files at once. Each release's
activations are mmap'd, indexed once with a deterministic random sample,
and the slices are written into a streaming output .npz.

Usage:
    python scripts/pool_activations.py \\
        --inputs /data/derivatives/eeg_sae/acts/brain-bzh_reve-base_L6_EEG2025R*_RestingState.npz \\
        --per-release 1000000 \\
        --out /data/derivatives/eeg_sae/acts/pool_L6_11release_11M.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


HBN_FIELDS = ("age", "sex", "p_factor", "internalizing",
              "externalizing", "attention")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Per-release activation .npz files")
    ap.add_argument("--per-release", type=int, default=1_000_000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    chunks_acts = []
    chunks_meta = {f: [] for f in HBN_FIELDS}
    chunks_meta["subject_id"] = []
    chunks_meta["window_idx"] = []
    chunks_meta["release"] = []

    d_model = None
    total = 0
    t0 = time.time()
    for f in sorted(args.inputs):
        p = Path(f)
        print(f"[load] {p.name}", flush=True)
        blob = np.load(p, allow_pickle=True)
        acts = blob["activations"]
        N = acts.shape[0]
        D = acts.shape[1]
        if d_model is None:
            d_model = D
        elif d_model != D:
            raise ValueError(f"{p}: d_model={D} != {d_model}")

        n_take = min(args.per_release, N)
        idx = rng.choice(N, size=n_take, replace=False)
        idx.sort()
        # Sort so the .npz mmap I/O is sequential.

        chunk = np.ascontiguousarray(acts[idx]).astype(np.float32, copy=False)
        chunks_acts.append(chunk)
        # Tag release for downstream filtering
        rel_tag = p.name.split("EEG2025")[1].split("_")[0]  # e.g. "R1", "R10"
        chunks_meta["release"].append(np.array([rel_tag] * n_take, dtype=object))

        # Carry per-token metadata
        for k in HBN_FIELDS:
            if k in blob.files:
                chunks_meta[k].append(blob[k][idx])
            else:
                chunks_meta[k].append(np.full(n_take, np.nan, dtype=np.float32))
        chunks_meta["subject_id"].append(blob["subject_id"][idx])
        chunks_meta["window_idx"].append(blob["window_idx"][idx])
        total += n_take
        del blob, acts
        print(f"  + {n_take:,} / {N:,} tokens from {rel_tag}", flush=True)

    print(f"[concat] {total:,} tokens, d_model={d_model}", flush=True)
    out = {
        "activations": np.concatenate(chunks_acts, axis=0),
        "d_model": np.int32(d_model),
    }
    for k, parts in chunks_meta.items():
        out[k] = np.concatenate(parts, axis=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] {args.out}  ({total * d_model * 4 / 1e9:.1f} GB float32)",
          flush=True)
    np.savez_compressed(args.out, **out)
    dt = time.time() - t0
    print(f"[done] {args.out}  ({dt:.1f}s)", flush=True)

    sidecar = args.out.with_suffix(".json")
    with open(sidecar, "w") as f:
        json.dump({
            "n_tokens": int(total),
            "d_model": int(d_model),
            "per_release": int(args.per_release),
            "releases": sorted([p.split("EEG2025")[1].split("_")[0]
                                for p in args.inputs if "EEG2025" in p]),
            "seed": args.seed,
            "source": "pool_activations.py",
        }, f, indent=2)
    print(f"[done] sidecar {sidecar}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
