#!/usr/bin/env python3
"""Generate ses-01 G_fmriprep betas as the empirical prior for cell 5.

Saves to PAPER_ROOT/task_2_1_betas/G_fmriprep_ses-01_betas.npy at the path
prereg_variant_sweep.py expects when --cells VariantG_glover_rtm_prior runs.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import task_2_1_factorial as T

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
T.PAPER_ROOT = LOCAL
T.RT3T_DATA = LOCAL / "rt3t" / "data"
T.FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye" / "data_sub-005"
                   / "bids" / "derivatives" / "fmriprep" / "sub-005")
T.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
T.BRAIN_MASK_PATH = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
T.RELMASK_PATH = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"

sys.argv = [
    "run_g_fmriprep_ses01",
    "--conditions", "G_fmriprep",
    "--session", "ses-01",
    "--runs", *[str(r) for r in range(1, 12)],
    "--out-dir", str(LOCAL / "task_2_1_betas"),
]

if __name__ == "__main__":
    T.main()
