#!/usr/bin/env python3
"""Run all 6 Regime C HOSVD cells locally.

Cells (all rtmotion BOLD, streaming pst=8, nilearn AR(1) + cum-z + repeat-avg):
  RT_streaming_pst8_HOSVD_K5_partial
  RT_streaming_pst8_HOSVD_K10_partial
  RT_streaming_pst8_HOSVD_K5_full
  RT_streaming_pst8_ResidHOSVD_K5_partial
  RT_streaming_pst8_ResidHOSVD_K10_partial
  RT_streaming_pst8_ResidHOSVD_K5_full

Existing wiring in scripts/rt_paper_full_replica.py from DGX commit 7a4690f
+ 3e63344 (residual variants). Just rebind paths and call CELLS dict.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import warnings
warnings.filterwarnings("ignore")

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

REGIME_C_CELLS = [
    "RT_streaming_pst8_HOSVD_K5_partial",
    "RT_streaming_pst8_HOSVD_K10_partial",
    "RT_streaming_pst8_HOSVD_K5_full",
    "RT_streaming_pst8_ResidHOSVD_K5_partial",
    "RT_streaming_pst8_ResidHOSVD_K10_partial",
    "RT_streaming_pst8_ResidHOSVD_K5_full",
]

for cell in REGIME_C_CELLS:
    cfg = R.CELLS[cell]
    print(f"\n=== {cell} ({cfg}) ===")
    t0 = time.time()
    try:
        kwargs = {k: v for k, v in cfg.items() if k != "loader"}
        betas, ids, conf = R.run_cell(cell, cfg["loader"], SESSION, RUNS, **kwargs)
        np.save(R.OUT_DIR / f"{cell}_{SESSION}_betas.npy", betas)
        np.save(R.OUT_DIR / f"{cell}_{SESSION}_trial_ids.npy", np.asarray(ids))
        with open(R.OUT_DIR / f"{cell}_{SESSION}_config.json", "w") as f:
            json.dump(conf, f, indent=2)
        print(f"  saved {cell}: betas {betas.shape}  ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"  FAILED {cell}: {e}")
        import traceback; traceback.print_exc()
