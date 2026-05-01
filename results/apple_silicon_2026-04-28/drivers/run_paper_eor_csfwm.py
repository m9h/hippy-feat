#!/usr/bin/env python3
"""End-of-run + GLMdenoise K=10 with CSF/WM-derived noise pool.

Follow-up to run_paper_eor_k10.py which used a relmask-internal noise pool
(top-10%-variance voxels in the relmask). That hurt 6pp because relmask
voxels are by definition task-driven, so top-PCs absorbed image variance.

This version extracts noise components from voxels OUTSIDE the relmask:
intersection of (CSF mask) ∪ (WM mask) ∩ flat_brain — properly task-irrelevant
voxels. Uses FSL FAST PVEs at rt3t/data/T1_brain_seg_pve_{0,2}.nii.gz.

PVEs are in T1 space — but mc_boldres BOLD is already resampled to T1 via
rtmotion. So a simple resample-to-BOLD-grid via nilearn.image.resample_to_img
should align them.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

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
K = 10
PVE_THRESH = 0.5


def inclusive_cumulative_zscore(beta_history, image_history, do_repeat_avg):
    arr = np.stack(beta_history, axis=0).astype(np.float32)
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    if not do_repeat_avg:
        return z, list(image_history)
    seen = {}
    out_b, out_i = [], []
    for i, img in enumerate(image_history):
        seen.setdefault(img, []).append(i)
        if len(seen[img]) == 1:
            out_b.append(z[i]); out_i.append(img)
        else:
            avg = z[seen[img]].mean(axis=0)
            first_pos = next(j for j, l in enumerate(out_i) if l == img)
            out_b[first_pos] = avg
    return np.stack(out_b, axis=0), out_i


R.cumulative_zscore_with_optional_repeat_avg = inclusive_cumulative_zscore

# Resample CSF (pve_0) and WM (pve_2) to BOLD grid + threshold + flatten + brain ∩
print("=== building CSF/WM noise-pool mask ===")
brain_img = nib.load(R.BRAIN_MASK)
flat_brain = (brain_img.get_fdata() > 0).flatten()

from nilearn.image import resample_to_img
csf_pve = nib.load(R.RT3T / "T1_brain_seg_pve_0.nii.gz")
wm_pve  = nib.load(R.RT3T / "T1_brain_seg_pve_2.nii.gz")
csf_b = resample_to_img(csf_pve, brain_img, interpolation="linear", force_resample=True, copy_header=True).get_fdata()
wm_b  = resample_to_img(wm_pve,  brain_img, interpolation="linear", force_resample=True, copy_header=True).get_fdata()
csfwm_mask = ((csf_b > PVE_THRESH) | (wm_b > PVE_THRESH)).flatten()
csfwm_in_brain = csfwm_mask & flat_brain
print(f"  CSF voxels (pve>0.5): {(csf_b > PVE_THRESH).sum()}")
print(f"  WM voxels (pve>0.5): {(wm_b > PVE_THRESH).sum()}")
print(f"  CSF ∪ WM ∩ flat_brain: {csfwm_in_brain.sum()}")

# Pre-extract K=10 noise components from CSF/WM timeseries per run
print(f"\n=== precomputing K={K} noise components per run from CSF/WM pool ===")
noise_per_run = []
for run in RUNS:
    pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(R.MC_DIR.glob(pattern))
    frames = []
    for v in vols:
        f = nib.load(v).get_fdata().flatten()[csfwm_in_brain]
        frames.append(f.astype(np.float32))
    ts = np.stack(frames, axis=1)                          # (V_csfwm, T)
    ts_c = ts - ts.mean(axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(ts_c, full_matrices=False)
    comps = Vt[:K].T.astype(np.float32)                    # (T, K)
    noise_per_run.append(comps)
    print(f"  run-{run:02d}: ts {ts.shape} → comps {comps.shape}")


def load_mc_with_noise(session: str, run: int):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    mc = np.loadtxt(par).astype(np.float32) if par.exists() else None
    if run < 1 or run > len(noise_per_run):
        return mc
    comps = noise_per_run[run - 1]
    if mc is None:
        return comps.astype(np.float32)
    n_match = min(mc.shape[0], comps.shape[0])
    return np.concatenate([mc[:n_match], comps[:n_match]], axis=1).astype(np.float32)


R.load_mc_params = load_mc_with_noise

CELL_NAME = "RT_paper_EoR_K10_CSFWM_inclz"
print(f"\n=== {CELL_NAME}  (pst=None full-run, K={K} CSF/WM, single-rep, inclusive cum-z) ===")
t0 = time.time()
betas, trial_ids, config = R.run_cell(
    cell_name=CELL_NAME,
    bold_loader=R.load_rtmotion_4d,
    session=SESSION, runs=RUNS,
    do_repeat_avg=False,
    streaming_post_stim_TRs=None,
)
config["cum_z_formula"] = "inclusive (arr[:i+1])"
config["GLMdenoise_K"] = K
config["GLMdenoise_pool"] = "CSF ∪ WM via FAST PVEs > 0.5, intersected with flat_brain"
config["n_csfwm_voxels"] = int(csfwm_in_brain.sum())
np.save(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_betas.npy", betas)
np.save(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
with open(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"\nsaved {CELL_NAME}: betas {betas.shape}  ({time.time()-t0:.1f}s)")
