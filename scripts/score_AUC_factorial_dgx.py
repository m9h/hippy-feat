#!/usr/bin/env python3
"""DGX-side AUC factorial scoring — mirrors Mac's
`results/apple_silicon_2026-04-28/drivers/run_AUC_on_factorial.py`.

Re-scores every cell already on /data/derivatives/.../prereg/ on pairwise
merge/separate AUC + Cohen's d (the actual closed-loop neurofeedback
target — same-image vs different-image β-distance) instead of top-1
50-way image retrieval.

Top-1 retrieval undersells GLMdenoise: it contributes 0 pp to top-1 but
+0.26 AUC / +1.1 Cohen's d on the same data (Mac confirmed). For the
PCML deployment context this is the correct metric.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

PRE = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg")
OUT_JSON = PRE / "AUC_factorial_results.json"
SESSION = "ses-03"


def load_betas(name: str, session: str = SESSION):
    betas = np.load(PRE / f"{name}_{session}_betas.npy")
    ids = np.load(PRE / f"{name}_{session}_trial_ids.npy", allow_pickle=True)
    ids = np.asarray([str(t) for t in ids])
    return betas, ids


def cumulative_zscore(arr: np.ndarray) -> np.ndarray:
    """Causal per-trial cumulative z-score — match retrieval pass policy
    so the AUC and top-1 numbers come from the same β at the same point
    in the pipeline."""
    out = np.zeros_like(arr)
    n = arr.shape[0]
    for i in range(n):
        if i < 2:
            mu = arr[:max(i, 1)].mean(axis=0, keepdims=True) if i > 0 else 0.0
            sd = 1.0
        else:
            mu = arr[:i].mean(axis=0, keepdims=True)
            sd = arr[:i].std(axis=0, keepdims=True) + 1e-8
        out[i] = (arr[i] - mu) / sd
    return out


def cosine_distance_matrix(B: np.ndarray) -> np.ndarray:
    """Pairwise (1 - cosine_sim) over rows of B. (N, N)."""
    Bn = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return 1.0 - Bn @ Bn.T


def evaluate_AUC(name: str) -> dict:
    betas, ids = load_betas(name)

    # Match retrieval pass z-score policy: cells driven by
    # rt_paper_full_replica.py / Offline_paper_replica already cum-z'd
    # inside the cell driver; everything else needs causal cum-z applied
    # here for parity with retrieval scoring.
    if (name.startswith("RT_paper_replica") or
            name.startswith("Offline_paper_replica") or
            name.startswith("RT_streaming_pst")):
        betas_z = betas
    else:
        betas_z = cumulative_zscore(betas)

    mask = np.array([t.startswith("all_stimuli/special515/") for t in ids])
    B = betas_z[mask]
    I = ids[mask]
    if B.shape[0] < 2:
        return {"cell": name, "error": "no special515 trials"}
    D = cosine_distance_matrix(B)

    same: list[float] = []
    diff: list[float] = []
    n = B.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            d = float(D[i, j])
            if I[i] == I[j]:
                same.append(d)
            else:
                diff.append(d)

    same_a = np.asarray(same)
    diff_a = np.asarray(diff)
    if len(same_a) == 0 or len(diff_a) == 0:
        return {"cell": name, "error": "no same or diff pairs"}

    pooled_sd = float(np.sqrt((same_a.var() + diff_a.var()) / 2))
    cohens_d = float((diff_a.mean() - same_a.mean()) / (pooled_sd + 1e-8))

    diff_sample = (diff_a if len(diff_a) <= 5000 else
                    np.random.default_rng(0).choice(diff_a, 5000, replace=False))
    n_corr = 0
    n_cmp = 0
    for s in same_a:
        n_corr += int((diff_sample > s).sum())
        n_cmp += len(diff_sample)
    auc = n_corr / max(n_cmp, 1)

    return {
        "cell": name,
        "n_test_trials": int(B.shape[0]),
        "n_same_pairs": int(len(same_a)),
        "n_diff_pairs": int(len(diff_a)),
        "same_image_d_mean": float(same_a.mean()),
        "diff_image_d_mean": float(diff_a.mean()),
        "cohens_d": cohens_d,
        "AUC": float(auc),
    }


CELLS = [
    "OLS_glover_rtm",
    "AR1freq_glover_rtm",
    "VariantG_glover_rtm",
    "VariantG_glover_rtm_prior",
    "AR1freq_glmsingleS1_rtm",
    "AR1freq_glover_rtm_glmdenoise_fracridge",
    "VariantG_glover_rtm_glmdenoise_fracridge",
    "VariantG_glover_rtm_acompcor",
    "RT_paper_replica_partial",
    "RT_paper_replica_full",
    "Offline_paper_replica_full",
    "EKF_streaming_glover_rtm",
    "HOSVD_denoise_AR1freq_glover_rtm",
    "Riemannian_prewhiten_AR1freq_glover_rtm",
    "EKF_session_online_glover_rtm",
    "HybridOnline_AR1freq_glover_rtm",
    "LogSig_AR1freq_glover_rtm",
    "RT_paper_replica_streaming_pst4_partial",
    "RT_paper_replica_streaming_pst6_partial",
    "RT_paper_replica_streaming_pst8_partial",
    "RT_paper_replica_streaming_pst10_partial",
    "RT_paper_replica_streaming_pst8_full",
    "RT_streaming_pst8_HOSVD_K5_partial",
    "RT_streaming_pst8_HOSVD_K10_partial",
    "RT_streaming_pst8_HOSVD_K5_full",
    "HybridOnline_streaming_pst8_AR1freq_glover_rtm",
    "RT_streaming_pst8_ResidHOSVD_K5_partial",
    "RT_streaming_pst8_ResidHOSVD_K10_partial",
    "RT_streaming_pst8_ResidHOSVD_K5_full",
    # GLMsingle gap-fill cells (job 1044)
    "AR1freq_glmsingleFull_rtm",
    "VariantG_glmsingleFull_rtm",
    "VariantG_glmsingleS1_rtm",
    "AR1freq_glmsingleFull_fmriprep",
]


def main():
    results = []
    for cell in CELLS:
        try:
            r = evaluate_AUC(cell)
            results.append(r)
            if "error" in r:
                print(f"  {cell:55s} ERROR: {r['error']}")
            else:
                print(f"  {cell:55s} AUC={r['AUC']:.4f}  d={r['cohens_d']:>6.3f}  "
                      f"n={r['n_test_trials']}")
        except FileNotFoundError:
            print(f"  SKIP {cell}: betas not on disk")
    OUT_JSON.write_text(json.dumps(results, indent=2))

    print()
    print("=== sorted by AUC ===")
    print(f"{'cell':55s} {'AUC':>7s} {'Cohen d':>9s} {'n':>5s}")
    print('-' * 85)
    for r in sorted([x for x in results if "error" not in x],
                     key=lambda x: -x["AUC"]):
        print(f"{r['cell']:55s} {r['AUC']:>7.4f} {r['cohens_d']:>9.3f} "
              f"{r['n_test_trials']:>5d}")
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
