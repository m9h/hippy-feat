#!/usr/bin/env python3
"""Task 6 pre-MVE: input-side variance transformations on G_rtmotion betas.

Tests whether posterior variance buys decoder-relevant information BEFORE
committing to retrain the ridge. Three input transformations of the same
(β_mean, β_var) pairs feed the frozen paper checkpoint:

    baseline       β_mean                                     (= current 71.3 %)
    var_weighted   β_mean / sqrt(β_var + ε)
    snr_gated      β_mean * (|β_mean|/sqrt(β_var) > threshold)

Each is z-scored voxelwise across the session before the ridge sees it
(matching paper §2.5). If any transformation beats baseline by >2 pp on
top-1 retrieval AND the paired bootstrap CI excludes zero, variance carries
exploitable information at the decoder boundary even without retraining.

Outputs JSON summary + per-trial hits to:
    /data/derivatives/rtmindeye_paper/task_2_1_betas/per_trial_t6/
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
    load_mindeye,
    predict_clip,
    cosine_sim_tokens,
)
from bake_off_per_trial import load_gt_from_cache, bootstrap_ci

BETAS_ROOT = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas")
OUT_DIR = BETAS_ROOT / "per_trial_t6"


def voxelwise_zscore(x: np.ndarray) -> np.ndarray:
    """Voxelwise z-score across the FIRST axis (trials).

    Match `mindeye_retrieval_eval.filter_to_special515`: z-score uses the FULL
    session's trial set (770 trials) before filtering down to the 150 special515
    test trials. Computing z-score on the 150 test trials in isolation gives a
    different (smaller-sample) reference distribution and inflates the baseline.
    """
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd


def filter_special515(betas, vars_, ids):
    mask = np.array([str(t).startswith("all_stimuli/special515/") for t in ids])
    return betas[mask], vars_[mask], np.asarray([str(t) for t in ids[mask]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="G_rtmotion")
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--snr-threshold", type=float, default=1.0,
                    help="|β|/√var threshold for snr_gated mask")
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

    # Load betas + vars + trial ids; filter to special515 test set
    betas_all, ids_all = load_condition_betas(args.condition, args.session)
    vars_all = np.load(BETAS_ROOT / f"{args.condition}_{args.session}_vars.npy")
    print(f"  loaded betas {betas_all.shape}, vars {vars_all.shape}")
    if vars_all.shape != betas_all.shape:
        raise RuntimeError(
            f"shape mismatch: betas {betas_all.shape} vs vars {vars_all.shape}"
        )

    eps = 1e-6
    # Compute three transforms on the FULL 770-trial array, then z-score
    # across all 770 trials, then filter to the 150 special515 test trials.
    # This matches the bake-off script's z-scoring scope (mindeye_retrieval_eval
    # .filter_to_special515) so the baseline absolute number agrees with the
    # bake-off table.
    snr_all = np.abs(betas_all) / np.sqrt(vars_all + eps)
    print(f"  SNR (770 trials): median={np.median(snr_all):.3f}, "
          f"mean={snr_all.mean():.3f}, 90th pct={np.quantile(snr_all, 0.9):.3f}")
    print(f"  fraction (trial,voxel) above {args.snr_threshold}: "
          f"{(snr_all > args.snr_threshold).mean():.3f}")

    raw_transforms = {
        "baseline": betas_all,
        "var_weighted": betas_all / np.sqrt(vars_all + eps),
        "snr_gated": betas_all * (snr_all > args.snr_threshold).astype(np.float32),
    }
    test_mask = np.array([str(t).startswith("all_stimuli/special515/")
                          for t in ids_all])
    transforms = {
        name: voxelwise_zscore(raw)[test_mask]
        for name, raw in raw_transforms.items()
    }
    ids_test = np.asarray([str(t) for t in ids_all[test_mask]])
    unique_images = np.array(sorted(set(ids_test)))
    img_to_idx = {str(u): i for i, u in enumerate(unique_images)}
    trial_idx = np.array([img_to_idx[t] for t in ids_test])

    # GT embeddings from cache
    image_paths = [Path(args.stimuli_dir) / Path(n).name for n in unique_images]
    gt_emb = load_gt_from_cache(Path(args.stimuli_dir), unique_images)

    # Load model once
    model, ss, se = load_mindeye(Path(args.checkpoint),
                                 n_voxels=2792, device=device)

    summary = {}
    for name, X in transforms.items():
        print(f"\n[{name}] z-scored input mean={X.mean():.3f} std={X.std():.3f}")
        t0 = time.time()
        pred = predict_clip(model, X, device=device,
                            clip_seq_dim=ss, clip_emb_dim=se)
        sim = cosine_sim_tokens(pred, gt_emb)
        topk = np.argsort(-sim, axis=1)
        hits1 = topk[:, 0] == trial_idx
        hits5 = np.array([trial_idx[i] in topk[i, :5] for i in range(len(sim))])
        brain_top = np.argsort(-sim.T, axis=1)[:, 0]
        brain_hits = np.array([trial_idx[brain_top[i]] == i for i in range(len(unique_images))])

        np.save(OUT_DIR / f"{args.condition}_{name}_hits_top1.npy", hits1)
        np.save(OUT_DIR / f"{args.condition}_{name}_brain_hits.npy", brain_hits)

        m, lo, hi = bootstrap_ci(hits1, n_resamples=2000)
        m5, lo5, hi5 = bootstrap_ci(hits5, n_resamples=2000)
        mb, lob, hib = bootstrap_ci(brain_hits, n_resamples=2000)
        summary[name] = {
            "top1_image": {"mean": m, "ci_lo": lo, "ci_hi": hi},
            "top5_image": {"mean": m5, "ci_lo": lo5, "ci_hi": hi5},
            "top1_brain": {"mean": mb, "ci_lo": lob, "ci_hi": hib},
            "elapsed_s": float(time.time() - t0),
        }
        print(f"  top1 image: {m:.3f} [{lo:.3f}-{hi:.3f}]")
        print(f"  top1 brain: {mb:.3f} [{lob:.3f}-{hib:.3f}]")

    # Paired diff vs baseline
    base = np.load(OUT_DIR / f"{args.condition}_baseline_hits_top1.npy").astype(int)
    print("\n=== Paired bootstrap vs baseline ===")
    for name in ("var_weighted", "snr_gated"):
        x = np.load(OUT_DIR / f"{args.condition}_{name}_hits_top1.npy").astype(int)
        diff = x - base
        n = len(diff)
        rng = np.random.default_rng(0)
        boot = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(2000)])
        p_lt0 = float((boot <= 0).mean())
        summary[name]["vs_baseline"] = {
            "mean_diff": float(diff.mean()),
            "ci_lo": float(np.quantile(boot, 0.025)),
            "ci_hi": float(np.quantile(boot, 0.975)),
            "p_diff_le_0": p_lt0,
        }
        print(f"  {name} - baseline: {diff.mean():+.3f} "
              f"CI=[{np.quantile(boot,0.025):+.3f},{np.quantile(boot,0.975):+.3f}] "
              f"P(<=0)={p_lt0:.3f}")

    out = OUT_DIR / f"{args.condition}_input_transforms_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
