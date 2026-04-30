#!/usr/bin/env python3
"""Re-score all denoise/fracridge factorial cells on pairwise merge/separate
AUC (Cohen's d, AUC) — the actual neurofeedback target — instead of top-1
image retrieval.

Uses point-estimate βs (no MC, no var propagation). Same metric as
run_B_merge_separate_posterior.py but on point estimates.
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
OUT_JSON = LOCAL / "task_2_1_betas" / "AUC_factorial_results.json"


def load_betas(name: str, session: str = "ses-03"):
    betas = np.load(PRE / f"{name}_{session}_betas.npy")
    ids = np.load(PRE / f"{name}_{session}_trial_ids.npy", allow_pickle=True)
    ids = np.asarray([str(t) for t in ids])
    return betas, ids


def cosine_distance_matrix(B: np.ndarray) -> np.ndarray:
    """Pairwise (1 - cosine_sim) over the rows of B. (N, N)."""
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return 1.0 - Bn @ Bn.T


def evaluate_AUC(name: str):
    betas, ids = load_betas(name)
    mask = np.array([t.startswith("all_stimuli/special515/") for t in ids])
    B = betas[mask]
    I = ids[mask]
    D = cosine_distance_matrix(B)                                # (n, n)

    same = []
    diff = []
    n = B.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(D[i, j])
            if I[i] == I[j]:
                same.append(d)
            else:
                diff.append(d)

    same = np.asarray(same)
    diff = np.asarray(diff)
    pooled_sd = np.sqrt((same.var() + diff.var()) / 2)
    cohens_d = float((diff.mean() - same.mean()) / (pooled_sd + 1e-8))

    # Pair-classification AUC: fraction of (same, diff) pairs where same < diff
    # Vectorize via rank-based estimator
    from itertools import islice
    n_corr = 0
    n_cmp = 0
    diff_sample = diff if len(diff) <= 5000 else np.random.default_rng(0).choice(
        diff, 5000, replace=False)
    for s in same:
        n_corr += int((diff_sample > s).sum())
        n_cmp += len(diff_sample)
    auc = n_corr / max(n_cmp, 1)

    return {
        "cell": name,
        "n_test_trials": int(B.shape[0]),
        "n_same_pairs": int(len(same)),
        "n_diff_pairs": int(len(diff)),
        "same_image_d_mean": float(same.mean()),
        "diff_image_d_mean": float(diff.mean()),
        "cohens_d": cohens_d,
        "AUC": float(auc),
    }


CELLS = [
    # baseline plain OLS (no denoising, no fracridge)
    "OLS_glover_rtm",
    # K-sweep without fracridge (frac=1.0)
    "OLS_glover_rtm_denoiseK0",
    "OLS_glover_rtm_denoiseK5",
    "OLS_glover_rtm_denoiseK10",
    "OLS_glover_rtm_denoiseK15",
    # K × fracridge factorial (with proper SVD per-voxel)
    "OLS_glover_rtm_denoiseK0_fracR0p3",
    "OLS_glover_rtm_denoiseK0_fracR0p5",
    "OLS_glover_rtm_denoiseK0_fracR1p0",
    "OLS_glover_rtm_denoiseK5_fracR0p3",
    "OLS_glover_rtm_denoiseK5_fracR0p7",
    "OLS_glover_rtm_denoiseK10_fracR0p5",
    "OLS_glover_rtm_denoiseK15_fracR0p5",
    "OLS_glover_rtm_denoiseK5_fracR_CV",
    # AR(1) baseline + cells 7/8 originals for context
    "AR1freq_glover_rtm",
    "AR1freq_glover_rtm_glmdenoise_fracridge",
    "VariantG_glover_rtm",
    "VariantG_glover_rtm_glmdenoise_fracridge",
    "VariantG_glover_rtm_acompcor",
    # paper anchors
    "RT_paper_replica_full_streaming_pst8",
    "Offline_paper_replica_full",
]


if __name__ == "__main__":
    results = []
    for cell in CELLS:
        try:
            r = evaluate_AUC(cell)
            results.append(r)
            print(f"  {cell:55s} AUC={r['AUC']:.4f}  Cohen's d={r['cohens_d']:.3f}  "
                  f"n_trials={r['n_test_trials']}")
        except FileNotFoundError as e:
            print(f"  SKIP {cell}: {e}")
    OUT_JSON.write_text(json.dumps(results, indent=2))

    print()
    print("=== denoise × fracridge factorial — sorted by AUC ===")
    print(f"{'cell':55s} {'AUC':>7s} {'Cohen d':>9s} {'n':>5s}")
    print('-' * 85)
    for r in sorted(results, key=lambda x: -x["AUC"]):
        print(f"{r['cell']:55s} {r['AUC']:>7.4f} {r['cohens_d']:>9.3f} {r['n_test_trials']:>5d}")
