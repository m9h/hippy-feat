#!/usr/bin/env python3
"""HRF model sweep on top of the FAST K=7+HP+erode×1 champion config.

Holds everything else constant; only varies the HRF basis function in
the per-trial nilearn LSS GLM. This isolates the HRF-modeling axis.

Variants:
  glover                              — canonical Glover HRF (current champion)
  glover + derivative                 — canonical + temporal derivative (1 extra column)
  glover + derivative + dispersion    — full 3-basis canonical
  spm                                 — SPM canonical HRF (different from Glover)
  spm + derivative + dispersion       — full SPM 3-basis
  fir (n=8)                           — fully data-driven 8-basis FIR

The "probe" contrast picks the canonical (first basis) for retrieval β
in all multi-basis cases.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_erosion
from nilearn.signal import clean
from nilearn.image import resample_to_img

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import rt_paper_full_replica as R

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
R.PAPER_ROOT = LOCAL
R.RT3T = LOCAL / "rt3t" / "data"
R.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
R.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
R.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
R.MC_DIR = LOCAL / "motion_corrected_resampled"
R.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"

SESSION = "ses-03"
RUNS = list(range(1, 12))
K = 7
TR = 1.5
PVE_THRESH = 0.5


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


# Build FAST-based aCompCor noise pool (eroded ×1) — match champion config
print("=== building FAST CSF/WM noise pool (erode ×1) ===")
brain_img = nib.load(R.BRAIN_MASK)
brain_3d = brain_img.get_fdata() > 0
csf = nib.load(R.RT3T / "T1_brain_seg_pve_0.nii.gz")
wm = nib.load(R.RT3T / "T1_brain_seg_pve_2.nii.gz")
csf_b = resample_to_img(csf, brain_img, interpolation="linear",
                         force_resample=True, copy_header=True).get_fdata()
wm_b = resample_to_img(wm, brain_img, interpolation="linear",
                        force_resample=True, copy_header=True).get_fdata()
csfwm_3d = ((csf_b > PVE_THRESH) | (wm_b > PVE_THRESH)) & brain_3d
csfwm_e1 = binary_erosion(csfwm_3d, iterations=1)
print(f"  pool size after erode×1: {csfwm_e1.sum()}", flush=True)


def extract_ts(mask_3d):
    mask_flat = mask_3d.flatten()
    out = []
    for run in RUNS:
        pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
        vols = sorted(R.MC_DIR.glob(pattern))
        frames = [nib.load(v).get_fdata().flatten()[mask_flat].astype(np.float32) for v in vols]
        out.append(np.stack(frames, axis=1))
    return out


print("=== extracting noise-pool BOLD ===", flush=True)
ts_e1 = extract_ts(csfwm_e1)

noise_per_run = []
for ts in ts_e1:
    ts_c = clean(ts.T, t_r=TR, high_pass=0.01, detrend=False, standardize=False).T
    _, _, Vt = np.linalg.svd(ts_c, full_matrices=False)
    noise_per_run.append(Vt[:K].T.astype(np.float32))


def loader(session, run):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    mc = np.loadtxt(par).astype(np.float32) if par.exists() else None
    if run < 1 or run > len(noise_per_run): return mc
    comps = noise_per_run[run - 1]
    if mc is None: return comps
    n_match = min(mc.shape[0], comps.shape[0])
    return np.concatenate([mc[:n_match], comps[:n_match]], axis=1).astype(np.float32)


R.load_mc_params = loader


# ----- Override fit_lss_nilearn to accept hrf_model + fir_delays -----------
# Save original for reference
_orig_fit = R.fit_lss_nilearn


def make_fit_with_hrf(hrf_model: str, fir_delays=None):
    def fit_lss(bold_4d, events, probe_trial, mc_params,
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

        base["trial_type"] = np.where(np.arange(len(base)) == probe_trial, "probe", "reference")
        base["duration"] = 1.0

        confounds = (pd.DataFrame(mc_used,
                                    columns=[f"mc_{i}" for i in range(mc_used.shape[1])])
                     if mc_used is not None else None)

        kwargs = dict(t_r=tr, slice_time_ref=0,
                      hrf_model=hrf_model,
                      drift_model="cosine", drift_order=1, high_pass=0.01,
                      signal_scaling=False, smoothing_fwhm=None,
                      noise_model="ar1",
                      n_jobs=1, verbose=0,
                      memory_level=0, minimize_memory=True,
                      mask_img=mask_img if mask_img is not None else False)
        if fir_delays is not None:
            kwargs["fir_delays"] = fir_delays
        glm = FirstLevelModel(**kwargs)
        glm.fit(run_imgs=bold_used, events=base, confounds=confounds)
        eff = glm.compute_contrast("probe", output_type="effect_size")
        return eff.get_fdata()
    return fit_lss


# ----- Run each HRF variant -----------------------------------------------
HRF_CELLS = [
    # (cell_name_suffix, hrf_model, fir_delays)
    ("glov_td",      "glover + derivative",                    None),
    ("glov_tddisp",  "glover + derivative + dispersion",       None),
    ("spm",          "spm",                                     None),
    ("spm_tddisp",   "spm + derivative + dispersion",          None),
    ("fir8",         "fir",                                     list(range(0, 17, 2))),  # 0,2,4,...,16 TRs (24s)
]

import pandas as pd  # noqa: re-import for fit_lss_nilearn closure

for suffix, hrf_model, fir_delays in HRF_CELLS:
    cell = f"RT_paper_EoR_K7_CSFWM_HP_e1_{suffix}_inclz"
    if (R.OUT_DIR / f"{cell}_{SESSION}_betas.npy").exists():
        print(f"\n=== {cell} already exists — skip ===", flush=True)
        continue
    R.fit_lss_nilearn = make_fit_with_hrf(hrf_model, fir_delays)
    print(f"\n=== {cell}  hrf_model={hrf_model!r}  fir_delays={fir_delays} ===", flush=True)
    t0 = time.time()
    betas, trial_ids, config = R.run_cell(
        cell_name=cell, bold_loader=R.load_rtmotion_4d,
        session=SESSION, runs=RUNS, do_repeat_avg=False,
        streaming_post_stim_TRs=None,
    )
    config.update({
        "cum_z_formula": "inclusive (arr[:i+1])",
        "GLMdenoise_K": K, "GLMdenoise_pool": "FAST CSF ∪ WM PVE > 0.5, eroded ×1",
        "high_pass_filter_noise_pool": "0.01 Hz",
        "mask_erosion_iterations": 1,
        "hrf_model": hrf_model,
        "fir_delays": fir_delays,
    })
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_betas.npy", betas)
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{cell}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell}: {betas.shape}  ({time.time()-t0:.1f}s)", flush=True)
