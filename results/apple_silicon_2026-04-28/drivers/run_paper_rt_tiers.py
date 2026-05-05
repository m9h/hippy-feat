#!/usr/bin/env python3
"""Reproduce the paper's three RT tiers — Fast (pst=5 ≈7.85s), Slow (pst=20 ≈30s),
End-of-run (pst=None, full-run BOLD).

Same nilearn LSS + Glover + AR(1) + cumulative z; only the GLM-fit window changes.
Per paper §2.7, RT-tier evaluations use the FIRST presentation of each repeat —
so we save raw per-trial betas (no repeat-avg) and let the scorer filter.

End-of-run is RT_paper_replica_partial we already have on disk; we just rescore.
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


def _local_load_mc(session: str, run: int):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    if par.exists():
        return np.loadtxt(par).astype(np.float32)
    return None
R.load_mc_params = _local_load_mc

SESSION = "ses-03"
RUNS = list(range(1, 12))

# (cell_name, post_stim_TRs)
TIERS = [
    ("RT_paper_Fast_pst5",  5),    # ≈7.5s, paper says 7.85s
    ("RT_paper_Slow_pst20", 20),   # ≈30s,  paper says 29.45s
]

R.OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, pst in TIERS:
    print(f"\n=== {name}  (pst={pst} TRs ≈ {pst*1.5:.1f}s, single-rep) ===")
    t0 = time.time()
    betas, trial_ids, config = R.run_cell(
        cell_name=name,
        bold_loader=R.load_rtmotion_4d,
        session=SESSION, runs=RUNS,
        do_repeat_avg=False,                      # keep all 770 trials raw
        streaming_post_stim_TRs=pst,
    )
    np.save(R.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(R.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"saved {name}: betas {betas.shape}  ({time.time()-t0:.1f}s)")
