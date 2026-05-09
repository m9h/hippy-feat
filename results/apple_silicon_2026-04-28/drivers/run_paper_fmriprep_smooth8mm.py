#!/usr/bin/env python3
"""fMRIPrep BOLD + 8mm FWHM Gaussian spatial smoothing at GLM time.

Tests the only Heunis-listed preprocessing axis we hadn't ablated: the
paper's pipeline (and ours) sets `smoothing_fwhm=None` everywhere
(`canonical_refs/mindeye_py_GLM_excerpt.py:28`); Heunis et al list 4-8mm
as the field-typical range. This cell flips the switch on at 8mm for
all three latencies (Fast pst=5, Slow pst=20, EoR full-run) on fmriprep
BOLD and saves prereg cells for direct comparison against the
unsmoothed `*_fmriprep_inclz` cells.

Output cells (suffix `_sm8mm` to disambiguate):
  RT_paper_Fast_pst5_fmriprep_sm8mm_inclz_ses-03
  RT_paper_Slow_pst20_fmriprep_sm8mm_inclz_ses-03
  RT_paper_EoR_fmriprep_sm8mm_inclz_ses-03
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))
import rt_paper_full_replica as R

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
R.PAPER_ROOT = LOCAL
R.RT3T = LOCAL / "rt3t/data"
R.FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005")
R.EVENTS_DIR = LOCAL / "rt3t/data/events"
R.BRAIN_MASK = LOCAL / "rt3t/data/sub-005_final_mask.nii.gz"
R.RELMASK = LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy"
R.MC_DIR = LOCAL / "motion_corrected_resampled"
R.OUT_DIR = LOCAL / "task_2_1_betas/prereg"

SMOOTH_FWHM = 8.0


def _local_load_mc(session, run):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    return np.loadtxt(par).astype(np.float32) if par.exists() else None
R.load_mc_params = _local_load_mc


def inclusive_cumz(beta_history, image_history, do_repeat_avg):
    arr = np.stack(beta_history, axis=0).astype(np.float32)
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z, list(image_history)
R.cumulative_zscore_with_optional_repeat_avg = inclusive_cumz


def fit_lss_nilearn_smoothed(bold_4d, events, probe_trial, mc_params,
                              tr=1.5, mask_img=None, streaming_decode_TR=None):
    """Identical to rt_paper_full_replica.fit_lss_nilearn but with
    `smoothing_fwhm=SMOOTH_FWHM` in the FirstLevelModel constructor."""
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
        if mc_params is not None:
            mc_used = mc_params[:streaming_decode_TR + 1]
        else:
            mc_used = None
    else:
        bold_used = bold_4d
        mc_used = mc_params

    base["trial_type"] = np.where(
        np.arange(len(base)) == probe_trial, "probe", "reference"
    )
    base["duration"] = 1.0

    confounds = (pd.DataFrame(mc_used,
                              columns=[f"mc_{i}" for i in range(mc_used.shape[1])])
                 if mc_used is not None else None)

    glm = FirstLevelModel(
        t_r=tr, slice_time_ref=0,
        hrf_model="glover",
        drift_model="cosine", drift_order=1, high_pass=0.01,
        signal_scaling=False,
        smoothing_fwhm=SMOOTH_FWHM,
        noise_model="ar1",
        n_jobs=1, verbose=0,
        memory_level=0, minimize_memory=True,
        mask_img=mask_img if mask_img is not None else False,
    )
    glm.fit(run_imgs=bold_used, events=base, confounds=confounds)
    eff = glm.compute_contrast("probe", output_type="effect_size")
    return eff.get_fdata()
R.fit_lss_nilearn = fit_lss_nilearn_smoothed


SESSION = "ses-03"
RUNS = list(range(1, 12))

CONFIGS = [
    ("RT_paper_Fast_pst5_fmriprep_sm8mm_inclz",  5),
    ("RT_paper_Slow_pst20_fmriprep_sm8mm_inclz", 20),
    ("RT_paper_EoR_fmriprep_sm8mm_inclz",        None),  # full-run
]

for cell, pst in CONFIGS:
    out = R.OUT_DIR / f"{cell}_{SESSION}_betas.npy"
    if out.exists():
        print(f"=== {cell} already exists — skip ===", flush=True)
        continue
    print(f"\n=== {cell} (fmriprep + 8mm + AR(1) LSS, pst={pst}) ===", flush=True)
    t0 = time.time()
    betas, trial_ids, config = R.run_cell(
        cell_name=cell, bold_loader=R.load_fmriprep_4d,
        session=SESSION, runs=RUNS, do_repeat_avg=False,
        streaming_post_stim_TRs=pst,
    )
    config.update({
        "cum_z_formula": "inclusive (arr[:i+1])",
        "bold_source": "fmriprep T1w preproc_bold",
        "streaming_post_stim_TRs": pst,
        "smoothing_fwhm": SMOOTH_FWHM,
    })
    np.save(out, betas)
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{cell}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell}: {betas.shape}  ({time.time()-t0:.1f}s)", flush=True)
