#!/usr/bin/env python3
"""Run actual GLMsingle on rtmotion BOLD over full ses-03 session.

Tests the synergistic-stack hypothesis: does the canonical GLMsingle
algorithm (HRF library + GLMdenoise + fracridge with joint CV across
stages) recover the 76% Offline anchor when applied to rtmotion BOLD
instead of fMRIPrep BOLD?

Setup:
- Full ses-03 session, 11 runs of rtmotion mc_boldres BOLD (X,Y,Z,T)
- Design: one condition column per unique image name, ones at trial
  onset TRs. Blanks excluded.
- stimdur = 3.0 (image duration; ITI varies but image is ~3s)
- TR = 1.5
- GLMsingle default params → TYPED_FITHRF_GLMDENOISE_RR.npz output
- Output saved to glmsingle/glmsingle_sub-005_ses-03_task-C_RTMOTION/

Comparison anchor: canonical Princeton GLMsingle on fMRIPrep BOLD gives
76% top-1 single-rep (`Canonical_GLMsingle_OfflineFull` cell). If
rtmotion + GLMsingle gives ~76%, BOLD source is confirmed irrelevant
and the joint stack is doing all the work. If it gives less, BOLD
source matters more when combined with GLMsingle than with nilearn LSS.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
RT3T = LOCAL / "rt3t" / "data"
MC_DIR = LOCAL / "motion_corrected_resampled"
EVENTS_DIR = RT3T / "events"
OUT_BASE = LOCAL / "glmsingle"
OUT_DIR = OUT_BASE / "glmsingle_sub-005_ses-03_task-C_RTMOTION"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5
STIMDUR = 3.0


# === Step 1: load rtmotion BOLD per run as 4D (X, Y, Z, T) ===
print("=== loading rtmotion BOLD per run ===")
data = []
events_per_run = []
for run in RUNS:
    pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(MC_DIR.glob(pattern))
    frames = [nib.load(v).get_fdata().astype(np.float32) for v in vols]
    bold = np.stack(frames, axis=-1)                   # (X, Y, Z, T)
    data.append(bold)
    events = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv",
                         sep="\t")
    events_per_run.append(events)
    print(f"  run-{run:02d}: BOLD {bold.shape}, {len(events)} events")


# === Step 2: collect unique non-blank images across the session ===
all_images = []
for ev in events_per_run:
    all_images.extend(ev[ev["image_name"] != "blank.jpg"]["image_name"].tolist())
unique_images = sorted(set(all_images))
img_to_col = {img: i for i, img in enumerate(unique_images)}
print(f"\n{len(unique_images)} unique non-blank images, {len(all_images)} total non-blank trials")


# === Step 3: build per-run design matrix (T, n_conditions) ===
print("\n=== building per-run design matrices ===")
design = []
n_conditions = len(unique_images)
for run_idx, ev in enumerate(events_per_run):
    n_trs = data[run_idx].shape[-1]
    dm = np.zeros((n_trs, n_conditions), dtype=np.int32)
    run_start = float(ev.iloc[0]["onset"])
    for _, row in ev.iterrows():
        if row["image_name"] == "blank.jpg":
            continue
        onset_TR = int(round((float(row["onset"]) - run_start) / TR))
        if onset_TR >= n_trs:
            continue
        col = img_to_col[row["image_name"]]
        dm[onset_TR, col] = 1
    design.append(dm)
    n_onsets = int(dm.sum())
    print(f"  run-{run_idx + 1:02d}: design {dm.shape}, {n_onsets} non-blank onsets")


# === Step 4: run GLMsingle ===
print("\n=== running GLMsingle (TYPED_FITHRF_GLMDENOISE_RR) ===")
print(f"   output dir: {OUT_DIR}")
print(f"   stimdur={STIMDUR}, tr={TR}, n_runs={len(RUNS)}, n_conditions={n_conditions}")

from glmsingle.glmsingle import GLM_single

opt = {
    "wantlibrary": 1,         # HRF library (Stage 1)
    "wantglmdenoise": 1,      # GLMdenoise (Stage 2)
    "wantfracridge": 1,       # fracridge (Stage 3)
    "wantfileoutputs": [1, 1, 1, 1],
    "wantmemoryoutputs": [0, 0, 0, 1],
}
glm = GLM_single(opt)

t0 = time.time()
results = glm.fit(
    design=design, data=data,
    stimdur=STIMDUR, tr=TR,
    outputdir=str(OUT_DIR),
)
print(f"\n=== GLMsingle done in {(time.time()-t0)/60:.1f} min ===")

# Save the chronological image label per beta (matches betasmd's last-axis order)
trial_ids = []
for ev in events_per_run:
    for _, row in ev.iterrows():
        if row["image_name"] == "blank.jpg":
            continue
        trial_ids.append(row["image_name"])
np.save(OUT_DIR / "trial_ids_chronological.npy", np.asarray(trial_ids))
print(f"saved {len(trial_ids)} chronological trial_ids alongside the GLMsingle output")
print(f"output files: {sorted(OUT_DIR.glob('*.npz'))}")
