#!/usr/bin/env python3
"""aCompCor refinement variants on top of K=7 CSF/WM (the new RT-deployable champion).

Three variants tested + the K=7 baseline:

  v1: HP-filter noise pool at 0.01 Hz before SVD (matches nilearn's high_pass)
  v2: Erode CSF/WM masks by 2 voxels before extracting noise pool
  v3: Both HP-filter + mask erosion

Compare against the K=7 baseline (RT_paper_EoR_K7_CSFWM_inclz at 96.3% 2-AFC).
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


# Resample CSF + WM PVEs to BOLD grid + threshold (3D)
print("=== building CSF/WM masks (3D) ===")
brain_img = nib.load(R.BRAIN_MASK)
flat_brain = (brain_img.get_fdata() > 0).flatten()
brain_3d = brain_img.get_fdata() > 0

from nilearn.image import resample_to_img
csf_pve = nib.load(R.RT3T / "T1_brain_seg_pve_0.nii.gz")
wm_pve = nib.load(R.RT3T / "T1_brain_seg_pve_2.nii.gz")
csf_b = resample_to_img(csf_pve, brain_img, interpolation="linear",
                         force_resample=True, copy_header=True).get_fdata()
wm_b = resample_to_img(wm_pve, brain_img, interpolation="linear",
                        force_resample=True, copy_header=True).get_fdata()

csfwm_3d = (csf_b > PVE_THRESH) | (wm_b > PVE_THRESH)
csfwm_in_brain_3d = csfwm_3d & brain_3d
print(f"  unedoded CSF∪WM ∩ brain: {csfwm_in_brain_3d.sum()} voxels")

# Erode by 2 voxels
csfwm_eroded_3d = binary_erosion(csfwm_in_brain_3d, iterations=2)
print(f"  eroded (×2) CSF∪WM ∩ brain: {csfwm_eroded_3d.sum()} voxels")


def extract_ts(mask_3d):
    """Per-run (V_pool, T) timeseries, applying given 3D mask."""
    mask_flat = mask_3d.flatten()
    out = []
    for run in RUNS:
        pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
        vols = sorted(R.MC_DIR.glob(pattern))
        frames = [nib.load(v).get_fdata().flatten()[mask_flat].astype(np.float32) for v in vols]
        out.append(np.stack(frames, axis=1))
    return out


print("\n=== pre-extracting noise-pool timeseries ===")
ts_uneroded = extract_ts(csfwm_in_brain_3d)
ts_eroded   = extract_ts(csfwm_eroded_3d)
print(f"  uneroded ts shapes: {[t.shape for t in ts_uneroded[:3]]}")
print(f"  eroded   ts shapes: {[t.shape for t in ts_eroded[:3]]}")


def make_noise_per_run(ts_list, hp_filter: bool):
    out = []
    for ts in ts_list:                                    # (V_pool, T)
        T = ts.shape[1]
        if hp_filter:
            # nilearn.signal.clean expects (n_samples, n_features)
            ts_clean = clean(ts.T, t_r=TR, high_pass=0.01, detrend=False, standardize=False)
            ts_clean = ts_clean.T                         # back to (V, T)
        else:
            ts_clean = ts - ts.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(ts_clean, full_matrices=False)
        out.append(Vt[:K].T.astype(np.float32))           # (T, K)
    return out


def make_loader(noise_per_run):
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
    return loader


# Three variants + keep K=7 baseline as reference (already exists)
VARIANTS = [
    ("RT_paper_EoR_K7_CSFWM_HP_inclz",       ts_uneroded, True,  False),
    ("RT_paper_EoR_K7_CSFWM_erode_inclz",    ts_eroded,   False, True),
    ("RT_paper_EoR_K7_CSFWM_HP_erode_inclz", ts_eroded,   True,  True),
]

for cell, ts_list, hp, erode in VARIANTS:
    if (R.OUT_DIR / f"{cell}_{SESSION}_betas.npy").exists():
        print(f"\n=== {cell} already exists — skip ===")
        continue
    noise = make_noise_per_run(ts_list, hp_filter=hp)
    R.load_mc_params = make_loader(noise)
    n_pool = ts_list[0].shape[0]
    print(f"\n=== {cell}  K={K}, HP={hp}, eroded={erode}, n_pool={n_pool} ===")
    t0 = time.time()
    betas, trial_ids, config = R.run_cell(
        cell_name=cell, bold_loader=R.load_rtmotion_4d,
        session=SESSION, runs=RUNS, do_repeat_avg=False,
        streaming_post_stim_TRs=None,
    )
    config.update({
        "cum_z_formula": "inclusive (arr[:i+1])",
        "GLMdenoise_K": K, "GLMdenoise_pool": "CSF ∪ WM via FAST PVEs > 0.5",
        "noise_pool_voxels": int(n_pool),
        "high_pass_filter_noise_pool": "0.01 Hz" if hp else "off",
        "mask_erosion_iterations": 2 if erode else 0,
    })
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_betas.npy", betas)
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{cell}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell}: {betas.shape}  ({time.time()-t0:.1f}s)")
