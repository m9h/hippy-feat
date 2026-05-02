#!/usr/bin/env python3
"""GPU retrieval pass over all pre-reg cells (DGX side).

Mirrors the analysis Apple-Silicon side ran in
`results/apple_silicon_2026-04-28/drivers/run_retrieval_local_v2.py`.
For each cell with betas at `task_2_1_betas/prereg/`, push the
already-z-scored βs through the **canonical** MindEye paper checkpoint
(ridge → BrainNetwork backbone → CLIP), score top-1/top-5 image
retrieval against the 50 special515 GT embeddings.

Cells 1-10 use the 150-trial denominator (50 special515 × 3 reps).
Cells 11-12 use 50 (post repeat-averaging done inside the cell driver).

Apple-Silicon side observed: cell 12 = 76.0% (matches paper Offline).
DGX expectation: cell 12 ≈ 76% AND cell 11 ≈ 66% (vs Mac's 74%) IF
the checkpoint mismatch is the only difference between platforms.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindeye_retrieval_eval import (
    cosine_sim_tokens,
    load_mindeye,
    predict_clip,
)
from bake_off_per_trial import load_gt_from_cache

PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
PREREG_ROOT = PAPER_ROOT / "task_2_1_betas" / "prereg"
GT_CACHE = PAPER_ROOT / "task_2_1_betas" / "gt_cache"
STIMULI = PAPER_ROOT / "rt3t" / "data" / "all_stimuli" / "special515"
CANONICAL_CKPT = (PAPER_ROOT / "checkpoints"
                   / "data_scaling_exp/concat_glmsingle/checkpoints"
                   / "sub-005_all_task-C_bs24_MST_rishab_repeats_3split_"
                     "sample=10_avgrepeats_finalmask_epochs_150.pth")
SESSION = "ses-03"
OUT_JSON = PREREG_ROOT / "prereg_retrieval_summary.json"

# Cells 11 and 12 collapse to 50 trials post repeat-avg; everything else is 150.
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
    # Cells 13-15: methods we coded up yesterday
    "EKF_streaming_glover_rtm",
    "HOSVD_denoise_AR1freq_glover_rtm",
    "Riemannian_prewhiten_AR1freq_glover_rtm",
    # Cell 16: TRULY online EKF — state accumulates across all 2112 TRs
    "EKF_session_online_glover_rtm",
    # Cell 17: noise model accumulates session-wide; β fit per-trial with
    # frozen-from-session ρ̂. The architecturally correct streaming hybrid.
    "HybridOnline_AR1freq_glover_rtm",
    # Cell 20: log-signature features of tCompCor PCs as nuisance regressors
    "LogSig_AR1freq_glover_rtm",
    # TASK_2_1_AMENDMENT_2026-04-28: locked Regime B (within-run streaming)
    # cells. Per-trial GLM fit on BOLD/events cropped to onset_TR + pst.
    # pst=8 with repeat-avg is the paper-RT canonical replica.
    "RT_paper_replica_streaming_pst4_partial",
    "RT_paper_replica_streaming_pst6_partial",
    "RT_paper_replica_streaming_pst8_partial",
    "RT_paper_replica_streaming_pst10_partial",
    "RT_paper_replica_streaming_pst8_full",
    # Regime C — cross-run HOSVD template confound (H3' deliverable)
    "RT_streaming_pst8_HOSVD_K5_partial",
    "RT_streaming_pst8_HOSVD_K10_partial",
    "RT_streaming_pst8_HOSVD_K5_full",
    # H3'-corrected variants
    "HybridOnline_streaming_pst8_AR1freq_glover_rtm",
    "RT_streaming_pst8_ResidHOSVD_K5_partial",
    "RT_streaming_pst8_ResidHOSVD_K10_partial",
    "RT_streaming_pst8_ResidHOSVD_K5_full",
    # GLMsingle gap-fill: full Stage 1+2+3 (rtmotion + fmriprep), Stage 1 + VG
    "AR1freq_glmsingleFull_rtm",
    "VariantG_glmsingleFull_rtm",
    "VariantG_glmsingleS1_rtm",
    "AR1freq_glmsingleFull_fmriprep",
    "AR1freq_glover_fmriprep_glmdenoise_fracridge",
    "VariantG_glover_fmriprep_glmdenoise_fracridge",
    # Paper's actual RT-pipeline betas at each Figure-3 delay
    "Paper_RT_actual_delay0",
    "Paper_RT_actual_delay1",
    "Paper_RT_actual_delay3",
    "Paper_RT_actual_delay5",
    "Paper_RT_actual_delay10",
    "Paper_RT_actual_delay15",
    "Paper_RT_actual_delay20",
    "Paper_RT_actual_delay63",
    # Princeton canonical GLMsingle (raw + repeat-avg)
    "Canonical_GLMsingle_ses-03",
    "Canonical_GLMsingle_ses-03_repeatavg",
    "Probe_canonical_cumz_firstrep",
    "Probe_canonical_cumz_repavg",
    "Probe_canonical_sessz_firstrep",
    "Probe_canonical_sessz_repavg",
    "Probe_canonical_noz_firstrep",
    "Probe_canonical_noz_repavg",
    # Streaming Stage 1+3 — RT-deployable canonical pipeline
    "Streaming_S1S3_pst8_AR1freq_rtm",
    "Streaming_S1S3_pst8_AR1freq_fmriprep",
    "FullRun_S1S3_AR1freq_rtm",
    "FullRun_S1S3_AR1freq_fmriprep",
    "FullRun_S1S3realFRAC_rtm",
    "FullRun_S1S3realFRAC_fmriprep",
    "Streaming_S1S3realFRAC_pst8_rtm",
    "Streaming_S1S3realFRAC_pst8_fmriprep",
]


def cumulative_zscore(arr: np.ndarray) -> np.ndarray:
    """Causal per-trial cumulative z-score: trial i uses stats from
    trials 0..i-1 only. Matches v2 of the Apple-Silicon retrieval eval
    (`run_retrieval_local_v2.py`) that landed cell 12 = 76% (paper exact).
    """
    out = np.zeros_like(arr)
    n = arr.shape[0]
    for i in range(n):
        if i < 2:
            # Too few trials to z-score; just center
            mu = arr[:max(i, 1)].mean(axis=0, keepdims=True) if i > 0 else 0.0
            sd = 1.0
        else:
            mu = arr[:i].mean(axis=0, keepdims=True)
            sd = arr[:i].std(axis=0, keepdims=True) + 1e-8
        out[i] = (arr[i] - mu) / sd
    return out


def filter_to_special515(betas: np.ndarray, ids: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask = np.array([str(t).startswith("all_stimuli/special515/") for t in ids])
    filtered = betas[mask]
    filtered_ids = np.asarray([str(t) for t in ids[mask]])
    unique_images = np.array(sorted(set(filtered_ids)))
    return filtered, filtered_ids, unique_images


def run_cell(cell: str, model, gt_emb: np.ndarray,
              unique_images: np.ndarray, img_to_idx: dict, device: str
              ) -> dict:
    betas_path = PREREG_ROOT / f"{cell}_{SESSION}_betas.npy"
    ids_path = PREREG_ROOT / f"{cell}_{SESSION}_trial_ids.npy"
    if not betas_path.exists() or not ids_path.exists():
        return {"cell": cell, "error": "betas or ids file missing"}

    betas_all = np.load(betas_path)
    ids_all = np.load(ids_path, allow_pickle=True)
    ids_all = np.asarray([str(t) for t in ids_all])

    # Cells 11 & 12 already have the cumulative z-score baked in inside
    # rt_paper_full_replica.py's `cumulative_zscore_with_optional_repeat_avg`.
    # All other cells need it applied here. Match Apple-Silicon v2 logic.
    # Cells 10-12 already have the cumulative z-score baked in inside
    # rt_paper_full_replica.py's `cumulative_zscore_with_optional_repeat_avg`
    # (post-2026-04-28 fix to causal cumulative). Skip re-z to avoid
    # double-application — Mac's v2 retrieval eval handles cell 10 the
    # same way.
    # All cells driven by rt_paper_full_replica.py already cum-z'd inside
    # `cumulative_zscore_with_optional_repeat_avg`; do not re-z-score here.
    if (cell.startswith("RT_paper_replica") or
            cell.startswith("Offline_paper_replica") or
            cell.startswith("RT_streaming_pst") or
            cell.startswith("Probe_canonical_")):
        betas_z = betas_all
    else:
        betas_z = cumulative_zscore(betas_all)

    # Filter to the 50 special515 images. For cells 11/12 the ids are the
    # post-collapse unique images (each appears ONCE).
    test_betas, test_ids, _ = filter_to_special515(betas_z, ids_all)
    if test_betas.shape[0] == 0:
        return {"cell": cell, "error": "no special515 trials"}

    trial_idx = np.array([img_to_idx[t] for t in test_ids])

    # Forward through ridge + backbone
    pred = predict_clip(model, test_betas, device=device)
    sim = cosine_sim_tokens(pred, gt_emb)                        # (n_test, 50)
    topk = np.argsort(-sim, axis=1)
    top1 = topk[:, 0]
    top5 = topk[:, :5]

    hits1 = top1 == trial_idx
    hits5 = np.array([trial_idx[i] in top5[i] for i in range(len(sim))])

    # First-rep-only mask (50 trials max) — matches Iyer et al. ICML 2026
    # Table 1 default eval ("single-trial betas from the first presentation
    # of the three repeats only"). For cells 11/12 / repeatavg etc. where
    # ids are already unique-per-image, every trial is a "first rep".
    seen = set()
    first_rep_mask = np.zeros(len(test_ids), dtype=bool)
    for i, name in enumerate(test_ids):
        if name not in seen:
            first_rep_mask[i] = True
            seen.add(name)

    return {
        "cell": cell,
        "n_test_trials": int(test_betas.shape[0]),
        "n_unique_test_images": int(len(np.unique(trial_idx))),
        "top1_image": float(hits1.mean()),
        "top5_image": float(hits5.mean()),
        "n_first_rep": int(first_rep_mask.sum()),
        "top1_image_first_rep": float(hits1[first_rep_mask].mean()),
        "top5_image_first_rep": float(hits5[first_rep_mask].mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(CANONICAL_CKPT))
    ap.add_argument("--cells", nargs="+", default=CELLS)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  torch {torch.__version__}", flush=True)
    print(f"[ckpt]  {Path(args.checkpoint).name}", flush=True)

    # Determine the canonical 50-image candidate set from cell 1's special515
    # trials (every cell uses the same 50 images; only the trial count varies)
    print("\n[1] Building 50-image candidate set + GT cache load", flush=True)
    seed_betas = np.load(PREREG_ROOT / f"OLS_glover_rtm_{SESSION}_betas.npy")
    seed_ids = np.load(PREREG_ROOT / f"OLS_glover_rtm_{SESSION}_trial_ids.npy",
                        allow_pickle=True)
    seed_ids = np.asarray([str(t) for t in seed_ids])
    _, _, unique_images = filter_to_special515(seed_betas, seed_ids)
    img_to_idx = {u: i for i, u in enumerate(unique_images)}
    gt_emb = load_gt_from_cache(STIMULI, unique_images)
    print(f"  {gt_emb.shape[0]} GT embeddings loaded from cache", flush=True)

    print("\n[2] Loading canonical MindEye checkpoint", flush=True)
    model, _, _ = load_mindeye(Path(args.checkpoint), n_voxels=2792, device=device)

    print("\n[3] Per-cell retrieval", flush=True)
    results = {}
    for cell in args.cells:
        t0 = time.time()
        r = run_cell(cell, model, gt_emb, unique_images, img_to_idx, device)
        r["elapsed_s"] = round(time.time() - t0, 2)
        results[cell] = r
        if "error" in r:
            print(f"  {cell:<46}  ERROR: {r['error']}", flush=True)
        else:
            fr = (f"top1_fr={r.get('top1_image_first_rep', 0.0):.3f} "
                  f"(n={r.get('n_first_rep', 0)})") if 'top1_image_first_rep' in r else ""
            print(f"  {cell:<46}  n={r['n_test_trials']:<4}  "
                  f"top1={r['top1_image']:.3f}  top5={r['top5_image']:.3f}  "
                  f"{fr}  ({r['elapsed_s']:.1f}s)", flush=True)

    OUT_JSON.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
