#!/usr/bin/env python3
"""Run actual GLMsingle on fmriprep BOLD over full ses-03 session.

Verifies the canonical Offline pipeline (paper's GLMsingle on fmriprep BOLD)
reproduces. Compare to:
- canonical pre-baked output `glmsingle_sub-005_ses-03_task-C/TYPED_FITHRF_GLMDENOISE_RR.npz`
  (Princeton-produced; gives 76% complete-set, 62% single-rep).
- our local rtmotion-GLMsingle (`glmsingle_sub-005_ses-03_task-C_RTMOTION/`).
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, nibabel as nib, pandas as pd

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
RT3T = LOCAL / "rt3t/data"
FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005")
EVENTS_DIR = RT3T / "events"
OUT_DIR = LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C_FMRIPREP_LOCAL"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5
STIMDUR = 3.0


# === Step 1: load fmriprep BOLD per run ===
print("=== loading fmriprep BOLD per run ===", flush=True)
data = []
events_per_run = []
for run in RUNS:
    p = (FMRIPREP_ROOT / SESSION / "func"
         / f"sub-005_{SESSION}_task-C_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz")
    bold = nib.load(p).get_fdata().astype(np.float32)
    data.append(bold)
    events = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv",
                         sep="\t")
    events_per_run.append(events)
    print(f"  run-{run:02d}: BOLD {bold.shape}, {len(events)} events", flush=True)


# === Step 2: collect unique non-blank images ===
all_images = []
for ev in events_per_run:
    all_images.extend(ev[ev["image_name"] != "blank.jpg"]["image_name"].tolist())
unique_images = sorted(set(all_images))
img_to_col = {img: i for i, img in enumerate(unique_images)}
print(f"\n{len(unique_images)} unique non-blank images, {len(all_images)} total non-blank trials", flush=True)


# === Step 3: per-run design matrix ===
print("\n=== building per-run design matrices ===", flush=True)
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
    print(f"  run-{run_idx + 1:02d}: design {dm.shape}, {int(dm.sum())} onsets", flush=True)


# === Step 4: run GLMsingle ===
print("\n=== running GLMsingle (TYPED_FITHRF_GLMDENOISE_RR) ===", flush=True)
print(f"   output dir: {OUT_DIR}", flush=True)

from glmsingle.glmsingle import GLM_single
opt = {
    "wantlibrary": 1,
    "wantglmdenoise": 1,
    "wantfracridge": 1,
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
print(f"\n=== GLMsingle done in {(time.time()-t0)/60:.1f} min ===", flush=True)

# Save chronological trial_ids alongside βs
trial_ids = []
for ev in events_per_run:
    for _, row in ev.iterrows():
        if row["image_name"] != "blank.jpg":
            trial_ids.append(row["image_name"])
np.save(OUT_DIR / "trial_ids_chronological.npy", np.asarray(trial_ids))
print(f"saved {len(trial_ids)} chronological trial_ids", flush=True)
print(f"output files: {sorted(OUT_DIR.glob('*.npz'))}", flush=True)
