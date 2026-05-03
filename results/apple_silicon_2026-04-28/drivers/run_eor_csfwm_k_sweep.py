#!/usr/bin/env python3
"""K-sweep for CSF/WM-derived GLMdenoise on EoR (full-run rtmotion BOLD).

Prior K-sweep was on partial windowing (pst=8) with relmask noise pool,
which had task leakage. This sweep:
  - Full-run BOLD (EoR config, paper-correct)
  - Noise pool from CSF (FAST PVE 0) ∪ WM (PVE 2), thresholded at 0.5,
    intersected with the brain mask (= ~64K task-irrelevant voxels)
  - Sweep K ∈ {0, 3, 5, 7, 10, 15, 20}
  - Pre-regress noise components from BOLD before LSS fit

Compared against `RT_paper_EoR_K10_CSFWM_inclz` baseline (K=10) which gave
2-AFC 95.7%, AUC 0.944, top-1 52% on fold-10.
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
K_VALUES = [0, 3, 5, 7, 15, 20]
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


# Build CSF/WM mask once (resample PVEs to BOLD grid)
print("=== building CSF/WM noise-pool mask ===")
brain_img = nib.load(R.BRAIN_MASK)
flat_brain = (brain_img.get_fdata() > 0).flatten()

from nilearn.image import resample_to_img
csf_pve = nib.load(R.RT3T / "T1_brain_seg_pve_0.nii.gz")
wm_pve = nib.load(R.RT3T / "T1_brain_seg_pve_2.nii.gz")
csf_b = resample_to_img(csf_pve, brain_img, interpolation="linear",
                         force_resample=True, copy_header=True).get_fdata()
wm_b = resample_to_img(wm_pve, brain_img, interpolation="linear",
                        force_resample=True, copy_header=True).get_fdata()
csfwm_mask = ((csf_b > PVE_THRESH) | (wm_b > PVE_THRESH)).flatten()
csfwm_in_brain = csfwm_mask & flat_brain
print(f"  CSF ∪ WM ∩ flat_brain: {csfwm_in_brain.sum()} voxels")


# Pre-extract per-run timeseries from CSF/WM voxels
print("\n=== pre-extracting CSF/WM timeseries per run ===")
ts_per_run = []
for run in RUNS:
    pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(R.MC_DIR.glob(pattern))
    frames = [nib.load(v).get_fdata().flatten()[csfwm_in_brain].astype(np.float32) for v in vols]
    ts = np.stack(frames, axis=1)
    ts_per_run.append(ts)
    print(f"  run-{run:02d}: ts {ts.shape}")


def make_load_mc(K_max):
    """Return a load_mc_params(session, run) closure that appends K_max
    PCA components from CSF/WM to the motion regressors. K_max=0 → motion only."""
    if K_max == 0:
        def loader(session, run):
            par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
            return np.loadtxt(par).astype(np.float32) if par.exists() else None
        return loader

    # Pre-compute per-run components for K=K_max
    noise_per_run = []
    for ts in ts_per_run:
        ts_c = ts - ts.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(ts_c, full_matrices=False)
        noise_per_run.append(Vt[:K_max].T.astype(np.float32))   # (T, K)

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


# Run the sweep
for K in K_VALUES:
    cell = f"RT_paper_EoR_K{K}_CSFWM_inclz"
    if (R.OUT_DIR / f"{cell}_{SESSION}_betas.npy").exists():
        print(f"\n=== {cell} already exists — skip ===")
        continue
    print(f"\n=== {cell}  (K={K} CSF/WM noise components) ===")
    R.load_mc_params = make_load_mc(K)
    t0 = time.time()
    betas, trial_ids, config = R.run_cell(
        cell_name=cell, bold_loader=R.load_rtmotion_4d,
        session=SESSION, runs=RUNS, do_repeat_avg=False,
        streaming_post_stim_TRs=None,
    )
    config["cum_z_formula"] = "inclusive (arr[:i+1])"
    config["GLMdenoise_K"] = K
    config["GLMdenoise_pool"] = "CSF ∪ WM via FAST PVEs > 0.5, intersected with flat_brain"
    config["n_csfwm_voxels"] = int(csfwm_in_brain.sum())
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_betas.npy", betas)
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{cell}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell}: {betas.shape}  ({time.time()-t0:.1f}s)")
