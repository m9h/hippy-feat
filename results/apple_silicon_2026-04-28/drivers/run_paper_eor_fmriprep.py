#!/usr/bin/env python3
"""End-of-run with fMRIPrep BOLD instead of rtmotion.

Tests if the rtmotion-vs-fMRIPrep BOLD source contributes the residual
−10pp EoR gap. Same pipeline (full-run, single-rep, inclusive cum-z,
Glover, AR(1)) — only the bold_loader changes from `load_rtmotion_4d`
to `load_fmriprep_4d`.

Note: paper's Offline anchor (76%) uses fMRIPrep BOLD + GLMsingle
(HRF library + GLMdenoise + fracridge over full run). This cell isolates
the BOLD source contribution by holding everything else constant — same
nilearn LSS, same Glover, same AR(1), just different BOLD timeseries.
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


def _local_load_mc(session, run):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    if par.exists():
        return np.loadtxt(par).astype(np.float32)
    return None
R.load_mc_params = _local_load_mc


def inclusive_cumz(beta_history, image_history, do_repeat_avg):
    arr = np.stack(beta_history, axis=0).astype(np.float32)
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    if not do_repeat_avg:
        return z, list(image_history)
    seen = {}
    out_b, out_i = [], []
    for i, img in enumerate(image_history):
        seen.setdefault(img, []).append(i)
        if len(seen[img]) == 1:
            out_b.append(z[i]); out_i.append(img)
        else:
            avg = z[seen[img]].mean(axis=0)
            first_pos = next(j for j, l in enumerate(out_i) if l == img)
            out_b[first_pos] = avg
    return np.stack(out_b, axis=0), out_i


R.cumulative_zscore_with_optional_repeat_avg = inclusive_cumz

SESSION = "ses-03"
RUNS = list(range(1, 12))

CELL_NAME = "RT_paper_EoR_fmriprep_inclz"
print(f"\n=== {CELL_NAME}  (full-run fMRIPrep BOLD, single-rep, inclusive cum-z) ===")
t0 = time.time()
betas, trial_ids, config = R.run_cell(
    cell_name=CELL_NAME,
    bold_loader=R.load_fmriprep_4d,
    session=SESSION, runs=RUNS,
    do_repeat_avg=False,
    streaming_post_stim_TRs=None,
)
config["cum_z_formula"] = "inclusive (arr[:i+1])"
config["bold_source"] = "fmriprep T1w preproc_bold"
np.save(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_betas.npy", betas)
np.save(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
with open(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"saved {CELL_NAME}: betas {betas.shape}  ({time.time()-t0:.1f}s)")
