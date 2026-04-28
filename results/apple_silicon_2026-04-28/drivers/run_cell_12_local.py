#!/usr/bin/env python3
"""Run cell 12 (Offline_paper_replica_full) locally on this Mac.

Imports scripts/rt_paper_full_replica.py and rebinds the path constants to the
~/Workspace data layout — no edits to the tracked repo.
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

# load_mc_params has its own hardcoded /data/3t path; rebind via the function's globals.
import types
_orig_load_mc = R.load_mc_params
def _local_load_mc(session: str, run: int):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    if par.exists():
        import numpy as np
        return np.loadtxt(par).astype("float32")
    return None
R.load_mc_params = _local_load_mc

# Replace argv before calling main() so argparse picks up our cell selection.
sys.argv = [
    "run_cell_12_local",
    "--cells", "Offline_paper_replica_full",
    "--session", "ses-03",
    "--runs", *[str(r) for r in range(1, 12)],
]

if __name__ == "__main__":
    R.main()
