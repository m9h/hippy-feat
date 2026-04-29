#!/usr/bin/env python3
"""Re-run cells 11 (RT, streaming pst=8) and 12 (Offline) with duration=3.0
to match Rishab's events.tsv stimulus duration. The tracked
fit_lss_nilearn hardcodes duration=1.0; this driver monkey-patches that
field after the events are cropped, before nilearn fits.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
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


# Patch fit_lss_nilearn to use duration=3.0 instead of 1.0
_orig_fit = R.fit_lss_nilearn


def fit_lss_nilearn_dur3(bold_4d, events, probe_trial, mc_params,
                         tr=1.5, mask_img=None, streaming_decode_TR=None):
    from nilearn.glm.first_level import FirstLevelModel
    base = events.copy()
    base["onset"] = base["onset"].astype(float) - base["onset"].iloc[0]
    if streaming_decode_TR is not None:
        decode_sec = streaming_decode_TR * tr
        base = base[base["onset"] <= decode_sec].reset_index(drop=True)
        if probe_trial >= len(base):
            return None
        bold_arr = bold_4d.get_fdata()[..., :streaming_decode_TR + 1]
        bold_used = nib.Nifti1Image(bold_arr, bold_4d.affine)
        mc_used = mc_params[:streaming_decode_TR + 1] if mc_params is not None else None
    else:
        bold_used = bold_4d
        mc_used = mc_params
    base["trial_type"] = np.where(
        np.arange(len(base)) == probe_trial, "probe", "reference"
    )
    # ---- THE FIX ----
    if "duration" in base.columns:
        # use Rishab's actual stimulus durations (~3s each)
        pass
    else:
        base["duration"] = 3.0
    # If duration column already came from events.tsv, leave its values intact
    # (they are 2.999-3.015s, not 1.0).

    confounds = (pd.DataFrame(mc_used,
                              columns=[f"mc_{i}" for i in range(mc_used.shape[1])])
                 if mc_used is not None else None)
    glm = FirstLevelModel(
        t_r=tr, slice_time_ref=0,
        hrf_model="glover",
        drift_model="cosine", drift_order=1, high_pass=0.01,
        signal_scaling=False, smoothing_fwhm=None,
        noise_model="ar1",
        n_jobs=1, verbose=0,
        memory_level=0, minimize_memory=True,
        mask_img=mask_img if mask_img is not None else False,
    )
    glm.fit(run_imgs=bold_used, events=base, confounds=confounds)
    eff = glm.compute_contrast("probe", output_type="effect_size")
    return eff.get_fdata()


R.fit_lss_nilearn = fit_lss_nilearn_dur3

SESSION = "ses-03"
RUNS = list(range(1, 12))


def run_one(cell_name, bold_loader, do_repeat_avg, streaming_post_stim_TRs=None):
    print(f"\n=== {cell_name} (duration=events.tsv ~3s, streaming_pst={streaming_post_stim_TRs}) ===")
    t0 = time.time()
    betas, ids, cfg = R.run_cell(cell_name, bold_loader, SESSION, RUNS,
                                  do_repeat_avg=do_repeat_avg,
                                  streaming_post_stim_TRs=streaming_post_stim_TRs)
    np.save(R.OUT_DIR / f"{cell_name}_{SESSION}_betas.npy", betas)
    np.save(R.OUT_DIR / f"{cell_name}_{SESSION}_trial_ids.npy", np.asarray(ids))
    cfg["fit_lss_duration"] = "events.tsv (~3s)"
    with open(R.OUT_DIR / f"{cell_name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {cell_name}: betas {betas.shape}  ({time.time()-t0:.1f}s)")


run_one("Offline_paper_replica_full_dur3",
        bold_loader=R.load_fmriprep_4d, do_repeat_avg=True)
run_one("RT_paper_replica_full_streaming_pst8_dur3",
        bold_loader=R.load_rtmotion_4d, do_repeat_avg=True,
        streaming_post_stim_TRs=8)
