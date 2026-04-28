#!/usr/bin/env python3
"""Run the JAX-only prereg cells (1, 2, 4, 6, 7, 8, 9) locally.

Cell 5 (VariantG_glover_rtm_prior) is skipped because it needs a
ses-01 G_fmriprep prior we haven't generated yet.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import prereg_variant_sweep as P

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
P.PAPER_ROOT = LOCAL
P.RT3T = LOCAL / "rt3t" / "data"
P.FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye" / "data_sub-005"
                   / "bids" / "derivatives" / "fmriprep" / "sub-005")
P.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
P.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
P.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"
P.HRF_INDICES_PATH = str(LOCAL / "rt3t" / "data" / "avg_hrfs_s1_s2_full.npy")
P.HRF_LIB_PATH = str(LOCAL / "rt3t" / "data" / "getcanonicalhrflibrary.tsv")

sys.argv = [
    "run_jax_cells_local",
    "--cells",
    "OLS_glover_rtm",
    "AR1freq_glover_rtm",
    "VariantG_glover_rtm",
    "AR1freq_glmsingleS1_rtm",
    "AR1freq_glover_rtm_glmdenoise_fracridge",
    "VariantG_glover_rtm_glmdenoise_fracridge",
    "VariantG_glover_rtm_acompcor",
    "--session", "ses-03",
    "--runs", *[str(r) for r in range(1, 12)],
]

if __name__ == "__main__":
    P.main()
