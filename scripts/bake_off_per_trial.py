#!/usr/bin/env python3
"""Bake-off per-trial analysis: load all 6 conditions in one GPU pass, dump
per-trial top-k hits + cosine-sim matrices to disk for downstream stats.

Outputs (one file per condition under task_2_1_betas/per_trial/):
    {cond}_hits_top1.npy      bool (150,)   true if top-1 retrieval matched the trial's image
    {cond}_hits_top5.npy      bool (150,)   true if top-5 contains it
    {cond}_brain_hits.npy     bool (50,)    true if image's top-1 brain match was a trial of that image
    {cond}_sim_matrix.npy     float32 (150, 50)  full cosine sim of pred vs GT for diagnostics
    {cond}_trial_image_idx.npy int (150,)   ground-truth image index per trial
    bake_off_summary.json     summary table with bootstrap CIs

Reuses the GT cache populated by mindeye_retrieval_eval.py — no OpenCLIP load.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindeye_retrieval_eval import (
    load_condition_betas,
    filter_to_special515,
    load_mindeye,
    predict_clip,
    cosine_sim_tokens,
)

CONDITIONS = [
    "A_fmriprep_glover",
    "B_rtmotion_glmsingle",
    "RT_paper",
    "Offline_paper",
    "G_fmriprep",
    "G_rtmotion",
]
BETAS_ROOT = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas")
GT_CACHE = BETAS_ROOT / "gt_cache"
OUT_DIR = BETAS_ROOT / "per_trial"


def load_gt_from_cache(stimuli_dir: Path, unique_images: np.ndarray) -> np.ndarray:
    """Reload GT embeddings purely from the cache populated by retrieval runs."""
    import hashlib
    paths = [stimuli_dir / Path(n).name for n in unique_images]
    out = []
    missing = []
    for p in paths:
        key = GT_CACHE / f"{p.stem}_{hashlib.md5(str(p).encode()).hexdigest()[:8]}.npy"
        if not key.exists():
            missing.append(p.name)
            continue
        out.append(np.load(key))
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} GT cache entries missing — run retrieval at least once "
            f"on this image set first. First missing: {missing[0]}"
        )
    return np.stack(out, axis=0)


def bootstrap_ci(hits: np.ndarray, n_resamples: int = 1000,
                 alpha: float = 0.05, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(hits)
    means = np.empty(n_resamples, dtype=np.float32)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        means[i] = hits[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(hits.mean()), lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--checkpoint",
                    default="/data/derivatives/rtmindeye_paper/checkpoints/"
                            "data_scaling_exp/concat_glmsingle/checkpoints/"
                            "sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_"
                            "avgrepeats_finalmask_epochs_150.pth")
    ap.add_argument("--stimuli-dir",
                    default="/data/derivatives/rtmindeye_paper/rt3t/data/all_stimuli/special515")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # Use Cond A's trial set as the reference; all conditions share the same trial structure
    print("\n[1] Trial set + GT embeddings from cache")
    betas_a, ids_a = load_condition_betas("A_fmriprep_glover", args.session)
    _, ids_test, unique_images = filter_to_special515(betas_a, ids_a)
    img_to_idx = {str(u): i for i, u in enumerate(unique_images)}
    trial_idx = np.array([img_to_idx[t] for t in ids_test])
    np.save(OUT_DIR / "trial_image_idx.npy", trial_idx)

    gt_emb = load_gt_from_cache(Path(args.stimuli_dir), unique_images)
    print(f"  loaded {gt_emb.shape[0]} GT embeddings from cache")

    # Single model load — reused across all conditions
    print("\n[2] Loading MindEye checkpoint")
    model, ss, se = load_mindeye(Path(args.checkpoint),
                                 n_voxels=2792, device=device)

    summary = {}
    for cond in CONDITIONS:
        print(f"\n[cond] {cond}")
        t0 = time.time()
        betas, ids = load_condition_betas(cond, args.session)
        betas_z, _, _ = filter_to_special515(betas, ids)

        pred = predict_clip(model, betas_z, device=device,
                            clip_seq_dim=ss, clip_emb_dim=se)
        sim = cosine_sim_tokens(pred, gt_emb)             # (150, 50)
        topk = np.argsort(-sim, axis=1)
        top1 = topk[:, 0]
        top5 = topk[:, :5]

        hits1 = top1 == trial_idx
        hits5 = np.array([trial_idx[i] in top5[i] for i in range(len(sim))])

        # Brain retrieval: for each image, top-1 trial; check if that trial maps to this image
        brain_sim = sim.T                                  # (50, 150)
        brain_top = np.argsort(-brain_sim, axis=1)[:, 0]
        brain_hits = np.array([trial_idx[brain_top[i]] == i for i in range(50)])

        np.save(OUT_DIR / f"{cond}_hits_top1.npy", hits1)
        np.save(OUT_DIR / f"{cond}_hits_top5.npy", hits5)
        np.save(OUT_DIR / f"{cond}_brain_hits.npy", brain_hits)
        np.save(OUT_DIR / f"{cond}_sim_matrix.npy", sim.astype(np.float32))

        m, lo, hi = bootstrap_ci(hits1)
        m5, lo5, hi5 = bootstrap_ci(hits5)
        mb, lob, hib = bootstrap_ci(brain_hits)
        summary[cond] = {
            "top1_image": {"mean": m, "ci_lo": lo, "ci_hi": hi},
            "top5_image": {"mean": m5, "ci_lo": lo5, "ci_hi": hi5},
            "top1_brain": {"mean": mb, "ci_lo": lob, "ci_hi": hib},
            "elapsed_s": float(time.time() - t0),
        }
        print(f"  top1 image: {m:.3f}  [95% CI {lo:.3f}–{hi:.3f}]")
        print(f"  top1 brain: {mb:.3f}  [95% CI {lob:.3f}–{hib:.3f}]")

    with open(OUT_DIR / "bake_off_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Pretty-print final table
    print("\n=== Bake-off summary ===")
    print(f"{'Condition':<24} {'top1 img':<22} {'top1 brain':<22}")
    for cond in CONDITIONS:
        s = summary[cond]
        ti = s["top1_image"]
        tb = s["top1_brain"]
        print(f"{cond:<24} "
              f"{ti['mean']:.3f} [{ti['ci_lo']:.3f}-{ti['ci_hi']:.3f}]   "
              f"{tb['mean']:.3f} [{tb['ci_lo']:.3f}-{tb['ci_hi']:.3f}]")
    print(f"\nWrote: {OUT_DIR / 'bake_off_summary.json'}")


if __name__ == "__main__":
    main()
