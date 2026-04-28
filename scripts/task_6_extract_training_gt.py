#!/usr/bin/env python3
"""Extract OpenCLIP GT embeddings for all training stimuli (ses-01, ses-02).

Walks the ses-01 + ses-02 trial_id arrays, dedupes to unique image paths,
runs OpenCLIP ViT-bigG/14 with output_tokens=True to get (256, 1664) token
embeddings per image, and stores them in the same cache directory used by
mindeye_retrieval_eval.py so MVE-1 ridge retraining can read them off disk.

This is the slow path — ~600+ images × ~1.5s each on the GB10 = ~15 min.
Runs once; cache hits forever after.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindeye_retrieval_eval import compute_ground_truth_embeddings

CACHE_DIR = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas/gt_cache")
STIMULI_ROOT = Path("/data/derivatives/rtmindeye_paper/rt3t/data/all_stimuli")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["ses-01", "ses-02"])
    ap.add_argument("--betas-dir",
                    default="/data/derivatives/rtmindeye_paper/task_2_1_betas")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # Collect all unique image paths across requested sessions
    seen = set()
    paths = []
    skipped_blank = 0
    for ses in args.sessions:
        # G_fmriprep is the canonical training source for MVE-1
        ids_path = Path(args.betas_dir) / f"G_fmriprep_{ses}_trial_ids.npy"
        if not ids_path.exists():
            print(f"  WARN: missing {ids_path} — has factorial run for {ses} completed?")
            continue
        ids = np.load(ids_path, allow_pickle=True)
        print(f"  {ses}: {len(ids)} trials, {len(set(map(str, ids)))} unique images")
        for raw in ids:
            s = str(raw)
            # Skip placeholder blanks; they're not in the stimuli dir
            if "blank" in s.lower():
                skipped_blank += 1
                continue
            if s in seen:
                continue
            seen.add(s)
            # image_name is like "all_stimuli/.../foo.png" — resolve to disk path
            p = STIMULI_ROOT / Path(s).relative_to("all_stimuli")
            if not p.exists():
                # Older trials may use bare basenames — search subdirs
                cands = list(STIMULI_ROOT.rglob(Path(s).name))
                if not cands:
                    print(f"  MISS: cannot locate {s}")
                    continue
                p = cands[0]
            paths.append(p)

    print(f"\n[summary] {len(paths)} unique training stimuli to embed "
          f"(skipped {skipped_blank} blank trials)")
    if not paths:
        return

    # Use the same compute_ground_truth_embeddings — it caches to disk so a
    # rerun is fast. The function returns the stacked array but the cache is
    # the durable artifact we care about.
    t0 = time.time()
    emb = compute_ground_truth_embeddings(paths, device=device,
                                          cache_dir=CACHE_DIR)
    print(f"\n[done] embeddings shape: {emb.shape}  elapsed: {time.time()-t0:.1f}s")
    print(f"  cache: {CACHE_DIR}")


if __name__ == "__main__":
    main()
