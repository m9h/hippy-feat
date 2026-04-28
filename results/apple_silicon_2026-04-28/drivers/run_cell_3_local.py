#!/usr/bin/env python3
"""Cell 3 (corrected): AR1freq_glover_rtm_nilearn — bare nilearn AR(1).

Differs from the prior version: drift_model=None and no high-pass filter,
to make this a true parity comparison against JAX cell 2 (`AR1freq_glover_rtm`)
which is `_variant_g_forward(pp_scalar=0)` with NO drift regressors.

Per prereg: |β_jax − β_nilearn| < 1e-3 per voxel.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import rt_paper_full_replica as R

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
R.RT3T = LOCAL / "rt3t" / "data"
R.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
R.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
R.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
R.MC_DIR = LOCAL / "motion_corrected_resampled"
OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"

CELL = "AR1freq_glover_rtm_nilearn"
SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5

warnings.filterwarnings("ignore", category=UserWarning, module="nilearn")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def fit_lss_bare_ar1(bold_4d, events, probe_trial, tr, mask_img):
    """LSS fit with FirstLevelModel(noise_model='ar1') only — NO drift, NO HPF."""
    from nilearn.glm.first_level import FirstLevelModel
    cropped = events.copy()
    cropped["onset"] = cropped["onset"].astype(float) - cropped["onset"].iloc[0]
    cropped["trial_type"] = np.where(
        np.arange(len(cropped)) == probe_trial, "probe", "reference"
    )
    cropped["duration"] = 1.0

    glm = FirstLevelModel(
        t_r=tr, slice_time_ref=0,
        hrf_model="glover",
        drift_model=None,           # NO drift (parity with JAX cell 2)
        high_pass=0.0,              # NO high-pass (parity with JAX cell 2)
        signal_scaling=False, smoothing_fwhm=None,
        noise_model="ar1",
        n_jobs=1, verbose=0,
        memory_level=0, minimize_memory=True,
        mask_img=mask_img if mask_img is not None else False,
    )
    glm.fit(run_imgs=bold_4d, events=cropped)
    eff = glm.compute_contrast("probe", output_type="effect_size")
    return eff.get_fdata()


flat_brain, rel = R.load_brain_paper_mask()
mask_img = nib.load(R.BRAIN_MASK)

all_betas: list[np.ndarray] = []
all_ids: list[str] = []
for run in RUNS:
    bold_4d = R.load_rtmotion_4d(SESSION, run)
    events = pd.read_csv(R.EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv", sep="\t")
    for trial_i in range(len(events)):
        t0 = time.time()
        beta_vol = fit_lss_bare_ar1(bold_4d, events, trial_i, TR, mask_img)
        beta_masked = beta_vol.flatten()[flat_brain][rel]
        all_betas.append(beta_masked.astype(np.float32))
        all_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))
        if trial_i == 0:
            print(f"  {CELL} run-{run:02d} trial 0 ({time.time()-t0:.2f}s)")

betas = np.stack(all_betas, axis=0)
OUT_DIR.mkdir(parents=True, exist_ok=True)
np.save(OUT_DIR / f"{CELL}_{SESSION}_betas.npy", betas)
np.save(OUT_DIR / f"{CELL}_{SESSION}_trial_ids.npy", np.asarray(all_ids))

config = {
    "cell": CELL, "session": SESSION, "runs": RUNS, "tr": TR,
    "nilearn_args": {"hrf_model": "glover", "drift_model": None,
                     "high_pass": 0.0, "noise_model": "ar1",
                     "signal_scaling": False},
    "post": "raw beta — no z-score, no repeat-avg, no drift, no HPF (bare AR1 parity vs JAX cell 2)",
}
with open(OUT_DIR / f"{CELL}_{SESSION}_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"saved {CELL}: betas {betas.shape}")
