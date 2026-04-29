#!/usr/bin/env python3
"""Re-run cell 11 (RT_paper_replica_full) with the corrected streaming-decode crop.

Bug confirmed by the user: cell 11 was fitting nilearn FirstLevelModel on the
ENTIRE 192-TR run, giving offline-quality β regardless of BOLD source. Paper's
RT pipeline (Rishab's mindeye.ipynb cell 19) refits per-trial on BOLD cropped
to [..., :decode_TR+1] where decode_TR = onset_TR + post_stim_TRs.

This driver reuses the new fit_lss_nilearn(streaming_decode_TR=...) path
(commit e462681) by calling run_cell with streaming_post_stim_TRs=4 (paper's
HRF tail ~6s = 4 TRs at 1.5s).

Saves to a separate filename so we can compare side-by-side against the
buggy full-run cell 11.
"""
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

# Local load_mc_params (paths)
def _local_load_mc(session: str, run: int):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    if par.exists():
        return np.loadtxt(par).astype(np.float32)
    return None
R.load_mc_params = _local_load_mc

POST_STIM_TRS = 4  # ≈6 s after onset — covers HRF peak + early tail
SESSION = "ses-03"
RUNS = list(range(1, 12))

CELL_NAME = "RT_paper_replica_full_streaming"
print(f"\n=== {CELL_NAME} (streaming_post_stim_TRs={POST_STIM_TRS}) ===")
t0 = time.time()
betas, trial_ids, config = R.run_cell(
    cell_name=CELL_NAME,
    bold_loader=R.load_rtmotion_4d,
    session=SESSION, runs=RUNS,
    do_repeat_avg=True,
    streaming_post_stim_TRs=POST_STIM_TRS,
)
elapsed = time.time() - t0

R.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_betas.npy", betas)
np.save(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
with open(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"\nsaved {CELL_NAME}: betas {betas.shape}  ({elapsed:.1f}s)")
