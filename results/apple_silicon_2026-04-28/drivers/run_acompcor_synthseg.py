#!/usr/bin/env python3
"""aCompCor with SynthSeg-derived noise pool.

Replaces FAST PVE > 0.5 thresholded CSF/WM mask with SynthSeg discrete
labels, picking only the anatomically cleanest non-GM tissue:
  - Ventricles: 4 (LH lat-vent), 5 (LH inf-lat-vent), 14 (3rd vent),
                 15 (4th vent), 43 (RH lat-vent), 44 (RH inf-lat-vent)
  - Cerebral WM core: 2 (LH WM), 41 (RH WM)

Excluded vs FAST-based pool:
  - Label 24 (extracerebral CSF, includes GM-PV sulcal CSF)
  - Cerebellum WM (7, 46) — distant from cortical BOLD; physiology may differ

Other settings same as champion (K=7 + HP-filter + erode×1).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
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

VENTRICLES = [4, 5, 14, 15, 43, 44]
CEREBRAL_WM = [2, 41]
NOISE_LABELS = VENTRICLES + CEREBRAL_WM


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


print("=== building SynthSeg-derived noise pool ===")
brain_img = nib.load(R.BRAIN_MASK)
brain_3d = brain_img.get_fdata() > 0

synthseg = nib.load(R.RT3T / "T1_synthseg.nii.gz")
ss_data = synthseg.get_fdata().astype(int)

# Build label mask in T1 space (binary: True if voxel is in NOISE_LABELS)
ss_mask = np.isin(ss_data, NOISE_LABELS)
print(f"  SynthSeg noise mask in T1 space: {ss_mask.sum()} voxels")
print(f"  (ventricles {sum(np.isin(ss_data, VENTRICLES).flatten())} + cerebral WM {sum(np.isin(ss_data, CEREBRAL_WM).flatten())})")

ss_img = nib.Nifti1Image(ss_mask.astype(np.float32), synthseg.affine)
ss_resampled = resample_to_img(ss_img, brain_img,
                                interpolation="nearest", force_resample=True, copy_header=True)
ss_in_bold = (ss_resampled.get_fdata() > 0.5) & brain_3d
print(f"  SynthSeg noise mask resampled to BOLD ∩ flat_brain: {ss_in_bold.sum()} voxels")

# Erode by 1 voxel (matches our champion cell's preprocessing)
ss_eroded = binary_erosion(ss_in_bold, iterations=1)
print(f"  After erode ×1: {ss_eroded.sum()} voxels")


def extract_ts(mask_3d):
    mask_flat = mask_3d.flatten()
    out = []
    for run in RUNS:
        pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
        vols = sorted(R.MC_DIR.glob(pattern))
        frames = [nib.load(v).get_fdata().flatten()[mask_flat].astype(np.float32) for v in vols]
        out.append(np.stack(frames, axis=1))
    return out


print("\n=== extracting noise-pool BOLD ===")
ts_per_run = extract_ts(ss_eroded)
print(f"  ts shape per run: {ts_per_run[0].shape}")

# HP filter + SVD per run
noise_per_run = []
for ts in ts_per_run:
    ts_c = clean(ts.T, t_r=TR, high_pass=0.01, detrend=False, standardize=False).T
    _, _, Vt = np.linalg.svd(ts_c, full_matrices=False)
    noise_per_run.append(Vt[:K].T.astype(np.float32))


def loader(session, run):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    mc = np.loadtxt(par).astype(np.float32) if par.exists() else None
    if run < 1 or run > len(noise_per_run):
        return mc
    comps = noise_per_run[run - 1]
    if mc is None:
        return comps
    n_match = min(mc.shape[0], comps.shape[0])
    return np.concatenate([mc[:n_match], comps[:n_match]], axis=1).astype(np.float32)


R.load_mc_params = loader

CELL = "RT_paper_EoR_K7_synthseg_HP_e1_inclz"
print(f"\n=== {CELL}  K={K}, HP=on, erode×1, n_pool={ss_eroded.sum()} ===")
t0 = time.time()
betas, trial_ids, config = R.run_cell(
    cell_name=CELL, bold_loader=R.load_rtmotion_4d,
    session=SESSION, runs=RUNS, do_repeat_avg=False,
    streaming_post_stim_TRs=None,
)
config.update({
    "cum_z_formula": "inclusive (arr[:i+1])",
    "GLMdenoise_K": K,
    "GLMdenoise_pool": "SynthSeg ventricles {4,5,14,15,43,44} + cerebral WM {2,41}",
    "noise_pool_voxels": int(ss_eroded.sum()),
    "high_pass_filter_noise_pool": "0.01 Hz",
    "mask_erosion_iterations": 1,
    "segmentation": "FreeSurfer mri_synthseg --fast --cpu (T1_synthseg.nii.gz)",
})
np.save(R.OUT_DIR / f"{CELL}_{SESSION}_betas.npy", betas)
np.save(R.OUT_DIR / f"{CELL}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
with open(R.OUT_DIR / f"{CELL}_{SESSION}_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  saved {CELL}: {betas.shape}  ({time.time()-t0:.1f}s)")
