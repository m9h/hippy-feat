#!/usr/bin/env python3
"""Run cells 10 and 11 (RT_paper_replica_partial / _full) locally.

Both use rtmotion BOLD via load_rtmotion_4d, nilearn AR(1), MCFLIRT motion
params as confounds, cosine drift, HPF 0.01, cumulative z-score, then
optional repeat-averaging (full = repeat-avg ON).
"""
from __future__ import annotations

import sys
from pathlib import Path

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

# Local load_mc_params: read MCFLIRT .par files from MC_DIR.
def _local_load_mc(session: str, run: int):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    if par.exists():
        import numpy as np
        return np.loadtxt(par).astype("float32")
    return None
R.load_mc_params = _local_load_mc

sys.argv = [
    "run_cells_10_11_local",
    "--cells",
    "RT_paper_replica_partial",
    "RT_paper_replica_full",
    "--session", "ses-03",
    "--runs", *[str(r) for r in range(1, 12)],
]

if __name__ == "__main__":
    R.main()
