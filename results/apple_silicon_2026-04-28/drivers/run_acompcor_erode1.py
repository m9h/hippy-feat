#!/usr/bin/env python3
"""aCompCor with erode×1 — less aggressive than the prior erode×2 (which
emptied the pool). Two cells:
  - K=7 + erode×1
  - K=7 + HP filter + erode×1
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


brain_img = nib.load(R.BRAIN_MASK)
brain_3d = brain_img.get_fdata() > 0

from nilearn.image import resample_to_img
csf_pve = nib.load(R.RT3T / "T1_brain_seg_pve_0.nii.gz")
wm_pve = nib.load(R.RT3T / "T1_brain_seg_pve_2.nii.gz")
csf_b = resample_to_img(csf_pve, brain_img, interpolation="linear",
                         force_resample=True, copy_header=True).get_fdata()
wm_b = resample_to_img(wm_pve, brain_img, interpolation="linear",
                        force_resample=True, copy_header=True).get_fdata()
csfwm_3d = ((csf_b > PVE_THRESH) | (wm_b > PVE_THRESH)) & brain_3d
print(f"  unedoded CSF∪WM ∩ brain: {csfwm_3d.sum()} voxels")

csfwm_e1 = binary_erosion(csfwm_3d, iterations=1)
print(f"  eroded ×1 CSF∪WM ∩ brain: {csfwm_e1.sum()} voxels")


def extract_ts(mask_3d):
    mask_flat = mask_3d.flatten()
    out = []
    for run in RUNS:
        pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
        vols = sorted(R.MC_DIR.glob(pattern))
        frames = [nib.load(v).get_fdata().flatten()[mask_flat].astype(np.float32) for v in vols]
        out.append(np.stack(frames, axis=1))
    return out


print("\n=== pre-extracting eroded×1 timeseries ===")
ts_e1 = extract_ts(csfwm_e1)


def make_noise_per_run(ts_list, hp_filter):
    out = []
    for ts in ts_list:
        if hp_filter:
            ts_c = clean(ts.T, t_r=TR, high_pass=0.01, detrend=False, standardize=False).T
        else:
            ts_c = ts - ts.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(ts_c, full_matrices=False)
        out.append(Vt[:K].T.astype(np.float32))
    return out


def make_loader(noise_per_run):
    def loader(session, run):
        par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
        mc = np.loadtxt(par).astype(np.float32) if par.exists() else None
        if run < 1 or run > len(noise_per_run): return mc
        comps = noise_per_run[run - 1]
        if mc is None: return comps
        n_match = min(mc.shape[0], comps.shape[0])
        return np.concatenate([mc[:n_match], comps[:n_match]], axis=1).astype(np.float32)
    return loader


for cell, hp in [("RT_paper_EoR_K7_CSFWM_e1_inclz", False),
                 ("RT_paper_EoR_K7_CSFWM_HP_e1_inclz", True)]:
    if (R.OUT_DIR / f"{cell}_{SESSION}_betas.npy").exists():
        print(f"\n=== {cell} already exists — skip ===")
        continue
    noise = make_noise_per_run(ts_e1, hp)
    R.load_mc_params = make_loader(noise)
    print(f"\n=== {cell}  K={K}, HP={hp}, erode=1, n_pool={ts_e1[0].shape[0]} ===")
    t0 = time.time()
    betas, trial_ids, config = R.run_cell(
        cell_name=cell, bold_loader=R.load_rtmotion_4d,
        session=SESSION, runs=RUNS, do_repeat_avg=False,
        streaming_post_stim_TRs=None,
    )
    config.update({
        "cum_z_formula": "inclusive (arr[:i+1])",
        "GLMdenoise_K": K, "GLMdenoise_pool": "CSF ∪ WM via FAST PVEs > 0.5, eroded x1",
        "noise_pool_voxels": int(ts_e1[0].shape[0]),
        "high_pass_filter_noise_pool": "0.01 Hz" if hp else "off",
        "mask_erosion_iterations": 1,
    })
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_betas.npy", betas)
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{cell}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell}: {betas.shape}  ({time.time()-t0:.1f}s)")
