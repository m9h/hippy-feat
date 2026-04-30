#!/usr/bin/env python3
"""(B) Merge/separate posterior tracker for neurofeedback.

For each pair of special515 stimuli, compute the posterior distribution of
the pairwise distance d(β_A, β_B) by MC-sampling β_A and β_B from their
posterior `(β_mean, β_var)` (Variant G output). Reports:

  - Per-pair posterior moments (mean, 95% CI of distance)
  - Calibration: how well does posterior confidence track empirical
    discriminability across pairs
  - Selective-feedback metrics: at confidence threshold τ, what fraction
    of pairs can the system reliably classify as "merged" (d small) or
    "separated" (d large) — vs the participant goal

Uses the existing `*_with_vars` Variant G cells we already saved on disk.
Cosine distance over voxel space; could also project through MindEye.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
PRE = LOCAL / "task_2_1_betas" / "prereg"
OUT_JSON = LOCAL / "task_2_1_betas" / "merge_separate_results.json"


def load_cell(name: str, session: str = "ses-03"):
    betas = np.load(PRE / f"{name}_{session}_betas.npy")              # (N, V)
    vars_ = np.load(PRE / f"{name}_{session}_vars.npy")               # (N, V)
    ids = np.load(PRE / f"{name}_{session}_trial_ids.npy", allow_pickle=True)
    ids = np.asarray([str(t) for t in ids])
    return betas, vars_, ids


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """1 - cos(a, b)"""
    a_n = a / (np.linalg.norm(a) + 1e-8)
    b_n = b / (np.linalg.norm(b) + 1e-8)
    return float(1.0 - np.dot(a_n, b_n))


def evaluate_pairs(name: str, n_mc: int = 200, threshold_merge: float = 0.5):
    print(f"\n=== {name} (n_mc={n_mc}) ===")
    betas, vars_, ids = load_cell(name)
    print(f"  loaded β {betas.shape}, var mean={vars_.mean():.4f}")

    # Filter to special515
    mask = np.array([t.startswith("all_stimuli/special515/") for t in ids])
    b_t = betas[mask]
    v_t = vars_[mask]
    ids_t = ids[mask]
    sd_t = np.sqrt(np.maximum(v_t, 1e-10))
    print(f"  test trials: {len(ids_t)}, unique images: {len(set(ids_t))}")

    # Pre-compute MC samples for each test trial
    rng = np.random.default_rng(0)
    n_t, V = b_t.shape
    eps = rng.standard_normal(size=(n_mc, n_t, V)).astype(np.float32)
    mc_samples = b_t[None, :, :] + sd_t[None, :, :] * eps             # (n_mc, n_t, V)

    # All pairs of trials (i, j) with i < j
    pairs = []
    n_t = b_t.shape[0]
    for i in range(n_t):
        for j in range(i + 1, n_t):
            pairs.append((i, j))

    # For each pair, compute posterior over distance via MC
    print(f"  computing posterior over {len(pairs)} pairs...")
    pair_data = []
    for (i, j) in pairs:
        d_samples = np.zeros(n_mc, dtype=np.float32)
        for k in range(n_mc):
            d_samples[k] = cosine_distance(mc_samples[k, i], mc_samples[k, j])
        same_image = (ids_t[i] == ids_t[j])
        pair_data.append({
            "i": i, "j": j,
            "img_a": ids_t[i], "img_b": ids_t[j],
            "same_image": bool(same_image),
            "d_mean": float(d_samples.mean()),
            "d_std": float(d_samples.std()),
            "d_p05": float(np.percentile(d_samples, 5)),
            "d_p95": float(np.percentile(d_samples, 95)),
            "p_merged": float((d_samples < threshold_merge).mean()),
        })

    # Aggregate metrics
    same_d = [p["d_mean"] for p in pair_data if p["same_image"]]
    diff_d = [p["d_mean"] for p in pair_data if not p["same_image"]]
    same_uncert = [p["d_std"] for p in pair_data if p["same_image"]]
    diff_uncert = [p["d_std"] for p in pair_data if not p["same_image"]]

    # Discriminability: difference between same-image and diff-image posterior means
    # (effect size in standard deviation units)
    pooled_sd = np.sqrt((np.var(same_d) + np.var(diff_d)) / 2)
    cohens_d = float((np.mean(diff_d) - np.mean(same_d)) / (pooled_sd + 1e-8))

    # Selective feedback: at each posterior-mean threshold, accuracy of
    # classifying pair as same/different image
    # AUC-style: just measure rank-based separation
    from itertools import product
    n_correct = 0
    n_compared = 0
    for s in same_d:
        for d in diff_d[:1000]:                                       # cap for speed
            n_correct += 1 if s < d else 0
            n_compared += 1
    auc_pair_class = n_correct / max(n_compared, 1)

    # Selective: drop pairs with high posterior std (uncertain), recompute AUC
    selective_results = {}
    for tau in [0.0, 0.05, 0.1, 0.15, 0.2]:
        same_keep = [p for p in pair_data if p["same_image"] and p["d_std"] <= tau or tau == 0]
        diff_keep = [p for p in pair_data if not p["same_image"] and p["d_std"] <= tau or tau == 0]
        if tau == 0:
            same_keep = [p for p in pair_data if p["same_image"]]
            diff_keep = [p for p in pair_data if not p["same_image"]]
        if len(same_keep) == 0 or len(diff_keep) == 0:
            selective_results[f"std_le_{tau}"] = {"coverage": 0.0, "auc": float("nan")}
            continue
        n_corr = 0
        n_cmp = 0
        for s in same_keep[:500]:
            for d in diff_keep[:500]:
                n_corr += 1 if s["d_mean"] < d["d_mean"] else 0
                n_cmp += 1
        coverage = (len(same_keep) + len(diff_keep)) / len(pair_data)
        selective_results[f"std_le_{tau}"] = {
            "coverage": float(coverage),
            "auc": float(n_corr / max(n_cmp, 1)),
            "n_same": len(same_keep), "n_diff": len(diff_keep),
        }

    summary = {
        "cell": name,
        "n_test_trials": int(b_t.shape[0]),
        "n_pairs": len(pair_data),
        "n_same_image_pairs": len(same_d),
        "n_diff_image_pairs": len(diff_d),
        "same_image_d_mean": float(np.mean(same_d)),
        "same_image_d_uncertainty_mean": float(np.mean(same_uncert)),
        "diff_image_d_mean": float(np.mean(diff_d)),
        "diff_image_d_uncertainty_mean": float(np.mean(diff_uncert)),
        "cohens_d": cohens_d,
        "pair_classification_AUC": auc_pair_class,
        "selective_feedback": selective_results,
    }
    print(f"  same-image  d̄={summary['same_image_d_mean']:.3f}  σ̂={summary['same_image_d_uncertainty_mean']:.3f}")
    print(f"  diff-image  d̄={summary['diff_image_d_mean']:.3f}  σ̂={summary['diff_image_d_uncertainty_mean']:.3f}")
    print(f"  Cohen's d: {cohens_d:.3f}  pair AUC: {auc_pair_class:.4f}")
    return summary


def main():
    cells = [
        "VariantG_glover_rtm_with_vars",
        "VariantG_glover_rtm_glmdenoise_fracridge_with_vars",
        "VariantG_glover_rtm_acompcor_with_vars",
        "VariantG_glover_rtm_streaming_pst8_with_vars",
    ]
    all_results = []
    for cell in cells:
        try:
            all_results.append(evaluate_pairs(cell, n_mc=100))
        except Exception as e:
            print(f"  FAILED {cell}: {e}")
            import traceback; traceback.print_exc()
    OUT_JSON.write_text(json.dumps(all_results, indent=2))
    print(f"\n=== summary ===")
    print(f"{'cell':50s} {'AUC':>8s} {'Cohen d':>10s} {'same_d̄':>8s} {'diff_d̄':>8s}")
    print("-" * 90)
    for r in all_results:
        print(f"{r['cell']:50s} {r['pair_classification_AUC']:>8.4f} "
              f"{r['cohens_d']:>10.3f} {r['same_image_d_mean']:>8.3f} {r['diff_image_d_mean']:>8.3f}")


if __name__ == "__main__":
    main()
