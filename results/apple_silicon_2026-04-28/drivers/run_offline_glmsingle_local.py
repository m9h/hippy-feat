#!/usr/bin/env python3
"""Offline_paper_replica_full WITH GLMsingle Stages 1-3 — the paper's actual
Offline pipeline.

Cell 12 currently: fmriprep BOLD + Glover canonical HRF + nilearn AR(1) — hits
76% (paper-exact) without any GLMsingle. This driver runs the paper's stated
Offline pipeline: fmriprep + GLMsingle HRF library (Stage 1) + AR(1)-freq +
GLMdenoise + fracridge (Stages 2-3), then post-hoc cumulative z-score +
repeat-averaging.

If GLMsingle truly contributes the bulk of Offline's lift, this cell should
substantially exceed 76% — and the gap to RT (68%) should widen toward the
paper's 10pp. If it lands at ~76% too, GLMsingle Stages 1-3 are not
load-bearing on this checkpoint/data.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import prereg_variant_sweep as P
import rt_paper_full_replica as R

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

CELL = "Offline_paper_replica_full_glmsingle"
SESSION = "ses-03"
RUNS = list(range(1, 12))

print(f"\n=== {CELL} ===")
print("  bold=fmriprep, HRF=glmsingle library, GLM=ar1_freq,")
print("  denoise=glmdenoise+fracridge, post=cum-z + repeat-avg")
t0 = time.time()

# Stage 1: per-voxel HRF library
# Stages 2-3: GLMdenoise + fracridge
# Wrapped by run_glm_cell which handles loading + the JAX GLM call
raw_betas, trial_ids = P.run_glm_cell(
    CELL, mode="ar1_freq", bold_source="fmriprep",
    hrf_strategy="glmsingle_lib",
    session=SESSION, runs=RUNS,
    prior_mean=None, prior_var=None,
    denoise="glmdenoise_fracridge",
)
print(f"  raw beta shape: {raw_betas.shape}  (no z-score yet)")

# Apply paper-Offline post-processing: cum-z + repeat-avg
final_betas, final_ids = R.cumulative_zscore_with_optional_repeat_avg(
    [raw_betas[i] for i in range(raw_betas.shape[0])],
    list(trial_ids), do_repeat_avg=True,
)
print(f"  post cum-z + repeat-avg: {final_betas.shape}  (was {raw_betas.shape})")

P.OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(P.OUT_DIR / f"{CELL}_{SESSION}_betas.npy", final_betas)
np.save(P.OUT_DIR / f"{CELL}_{SESSION}_trial_ids.npy", np.asarray(final_ids))
config = {
    "cell": CELL, "session": SESSION, "runs": RUNS, "tr": 1.5,
    "bold_source": "fmriprep", "hrf_strategy": "glmsingle_lib",
    "mode": "ar1_freq", "denoise": "glmdenoise_fracridge",
    "post": "cumulative z-score + repeat-averaging",
    "n_trials_post": int(final_betas.shape[0]),
}
with open(P.OUT_DIR / f"{CELL}_{SESSION}_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  saved {CELL} ({time.time() - t0:.1f}s)")
