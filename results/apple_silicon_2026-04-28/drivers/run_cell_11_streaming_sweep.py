#!/usr/bin/env python3
"""Sweep post_stim_TRs values for cell 11 streaming to bracket paper's 66%."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import rt_paper_full_replica as R

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
R.PAPER_ROOT = LOCAL
R.RT3T = LOCAL / "rt3t" / "data"
R.FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye" / "data_sub-005"
                   / "bids" / "derivatives" / "fmriprep" / "sub-005")
R.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
R.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
R.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
R.MC_DIR = LOCAL / "motion_corrected_resampled"
R.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"

def _local_load_mc(session, run):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    if par.exists():
        return np.loadtxt(par).astype(np.float32)
    return None
R.load_mc_params = _local_load_mc

SESSION = "ses-03"
RUNS = list(range(1, 12))

for post_stim in [6, 8, 10]:
    cell_name = f"RT_paper_replica_full_streaming_pst{post_stim}"
    print(f"\n=== {cell_name} ===")
    t0 = time.time()
    betas, trial_ids, config = R.run_cell(
        cell_name=cell_name,
        bold_loader=R.load_rtmotion_4d,
        session=SESSION, runs=RUNS,
        do_repeat_avg=True,
        streaming_post_stim_TRs=post_stim,
    )
    np.save(R.OUT_DIR / f"{cell_name}_{SESSION}_betas.npy", betas)
    np.save(R.OUT_DIR / f"{cell_name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{cell_name}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"saved {cell_name}: betas {betas.shape}  ({time.time()-t0:.1f}s)")
