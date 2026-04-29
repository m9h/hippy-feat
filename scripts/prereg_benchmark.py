#!/usr/bin/env python3
"""Run the pre-registered hypotheses against the 12-cell variant sweep.

Reads cell betas from /data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/
and applies the locked H1–H5 hypotheses from TASK_2_1_PREREGISTRATION.md.

Decoder-free metric: β reliability (across-rep Pearson correlation).
Decoder-based metrics (top-1/top-5 retrieval) come from a separate GPU pass
through the MindEye ridge → backbone → CLIP and are run after this script.

Pre-registered hypotheses tested here at the β-reliability level:
    H1: AR(1) freq > OLS by ≥ 0.02 absolute (loose proxy for the 3 pp
        retrieval threshold via the harness gain we already saw)
    H2: VG (uninformative) ≈ AR(1) freq within 95% CI
    H3: VG (training prior) > VG (uninformative) by some positive Δ
    H4: AR(1) + GLMdenoise+fracridge > AR(1) freq alone
    H5: replicas reproduce paper's Offline > paper's RT — measured here
        as β reliability ordering since retrieval requires GPU pass
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_glm_variants import (
    beta_reliability,
    bootstrap_mean_ci,
    image_identifiability,
    paired_diff_ci,
    shuffle_null_reliability,
)

PREREG_ROOT = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg")
SESSION = "ses-03"
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
]


def load_cell(name: str):
    betas = np.load(PREREG_ROOT / f"{name}_{SESSION}_betas.npy")
    ids = np.load(PREREG_ROOT / f"{name}_{SESSION}_trial_ids.npy",
                   allow_pickle=True)
    ids = np.asarray([str(t) for t in ids])
    return betas, ids


def main():
    print(f"=== Pre-registered β reliability across {len(CELLS)} cells ===\n")
    per_cell = {}
    per_image_arrays = {}
    for cell in CELLS:
        try:
            betas, ids = load_cell(cell)
        except FileNotFoundError as e:
            print(f"  SKIP {cell}: {e}")
            continue
        rel_mean, rel_arr = beta_reliability(betas, ids)
        rel_ci = bootstrap_mean_ci(rel_arr)
        ident = image_identifiability(betas, ids)
        null = shuffle_null_reliability(betas, ids, n_perms=30)
        per_cell[cell] = {
            "n_trials": int(betas.shape[0]),
            "n_voxels": int(betas.shape[1]),
            "n_repeated_images": int(len(rel_arr)),
            "beta_reliability_mean": rel_ci[0],
            "beta_reliability_ci_lo": rel_ci[1],
            "beta_reliability_ci_hi": rel_ci[2],
            "shuffle_null_mean": float(null.mean()) if len(null) else float("nan"),
            "image_id_top1_hit": ident.get("top1_image_id_hit", float("nan")),
            "n_total_queries": ident.get("n_total_queries", 0),
        }
        per_image_arrays[cell] = rel_arr
        print(f"  {cell:<48}  rel={rel_ci[0]:+.4f} "
              f"[{rel_ci[1]:+.4f},{rel_ci[2]:+.4f}]  null≈{null.mean():+.4f}  "
              f"id_hit={ident['top1_image_id_hit']:.3f}  "
              f"n_trials={betas.shape[0]}")

    # ---- Pre-registered hypothesis tests ----
    print("\n=== H1–H5 pre-registered tests ===")
    H = {}

    def _paired(a_name, b_name, label):
        if a_name not in per_image_arrays or b_name not in per_image_arrays:
            return None
        a, b = per_image_arrays[a_name], per_image_arrays[b_name]
        # Match length (in case repeat-averaging changed N for replicas)
        n = min(len(a), len(b))
        d = paired_diff_ci(a[:n], b[:n])
        sig = "✓" if d["ci_lo"] > 0 else "—"
        print(f"  {label:<58}  Δ={d['mean_diff']:+.4f} "
              f"[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}]  P(Δ≤0)={d['p_diff_le_0']:.3f}  {sig}")
        return d

    H["H1_AR1freq_vs_OLS"] = _paired(
        "AR1freq_glover_rtm", "OLS_glover_rtm",
        "H1: AR(1) freq > OLS",
    )
    H["H2_VariantG_vs_AR1freq"] = _paired(
        "VariantG_glover_rtm", "AR1freq_glover_rtm",
        "H2: VG (uninformative) ≈ AR(1) freq",
    )
    H["H3_VariantG_prior_vs_uninformative"] = _paired(
        "VariantG_glover_rtm_prior", "VariantG_glover_rtm",
        "H3: VG (prior) > VG (uninformative)",
    )
    H["H4_AR1freq_glmdenoise_fracridge_vs_AR1freq"] = _paired(
        "AR1freq_glover_rtm_glmdenoise_fracridge", "AR1freq_glover_rtm",
        "H4: AR(1) + GLMdenoise+fracridge > AR(1) freq alone",
    )
    H["H4b_VG_glmdenoise_vs_VG"] = _paired(
        "VariantG_glover_rtm_glmdenoise_fracridge", "VariantG_glover_rtm",
        "H4b: VG + GLMdenoise+fracridge > VG alone",
    )
    H["H4c_VG_acompcor_vs_VG"] = _paired(
        "VariantG_glover_rtm_acompcor", "VariantG_glover_rtm",
        "H4c: VG + tCompCor > VG alone",
    )
    H["H5_Offline_vs_RT_replica"] = _paired(
        "Offline_paper_replica_full", "RT_paper_replica_full",
        "H5: Offline replica > RT replica (paper-style: 76 % > 66 %)",
    )

    out = PREREG_ROOT / "prereg_benchmark_summary.json"
    with open(out, "w") as f:
        json.dump({"per_cell": per_cell, "hypotheses": H}, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
