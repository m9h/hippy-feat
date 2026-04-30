#!/usr/bin/env python3
"""Variant A+N (CSF/WM nuisance regression) — finally run.

Resamples T1-space FSL FAST PVE files (CSF=pve_0, WM=pve_2) to BOLD space,
thresholds at 0.5 to make boolean masks at the 2792-voxel finalmask layout,
then runs the existing VariantA_NuisanceRegression class on rtmotion BOLD.

Tests both alone and stacked with GLMdenoise K=10 (the AUC winner).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn.image import resample_to_img

warnings.filterwarnings("ignore")
LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import prereg_variant_sweep as P
import rt_glm_variants as V
from prereg_variant_sweep import load_mask, load_rtmotion, load_events
from rt_glm_variants import build_design_matrix, make_glover_hrf

P.RT3T = LOCAL / "rt3t" / "data"
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
P.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
P.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
P.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5

T1_DATA = LOCAL / "rt3t" / "data"
PVE_CSF = T1_DATA / "T1_brain_seg_pve_0.nii.gz"     # CSF
PVE_GM = T1_DATA / "T1_brain_seg_pve_1.nii.gz"      # GM
PVE_WM = T1_DATA / "T1_brain_seg_pve_2.nii.gz"      # WM


def build_csf_wm_masks_at_bold_res() -> tuple[np.ndarray, np.ndarray]:
    """Resample T1-space PVEs to BOLD space, threshold to bool, apply
    brain∩rel mask to get 2792-voxel boolean arrays."""
    brain_img = nib.load(P.BRAIN_MASK)                              # BOLD-space template
    print(f"  brain_mask: {brain_img.shape}")

    csf_t1 = nib.load(PVE_CSF)
    wm_t1 = nib.load(PVE_WM)
    print(f"  PVE_CSF (T1): {csf_t1.shape}, PVE_WM (T1): {wm_t1.shape}")

    # Resample to BOLD space
    csf_bold = resample_to_img(csf_t1, brain_img, interpolation="linear",
                                copy_header=True, force_resample=True)
    wm_bold = resample_to_img(wm_t1, brain_img, interpolation="linear",
                               copy_header=True, force_resample=True)
    print(f"  resampled CSF: {csf_bold.shape}, WM: {wm_bold.shape}")

    # Apply brain mask + relmask
    flat_brain = (brain_img.get_fdata() > 0).flatten()              # (V_total,)
    rel = np.load(P.RELMASK)                                         # (19174,) bool
    csf_v = csf_bold.get_fdata().flatten()[flat_brain][rel]          # (2792,)
    wm_v = wm_bold.get_fdata().flatten()[flat_brain][rel]            # (2792,)

    # Threshold at 0.5 partial volume
    csf_mask = csf_v > 0.5
    wm_mask = wm_v > 0.5
    print(f"  CSF mask: {csf_mask.sum()} voxels, WM mask: {wm_mask.sum()} voxels")
    return csf_mask, wm_mask


def fit_one_cell(name: str, csf_mask: np.ndarray, wm_mask: np.ndarray,
                 glmdenoise_K: int = 0):
    print(f"\n=== {name} (K={glmdenoise_K}, A+N CSF/WM nuisance) ===")
    t0 = time.time()
    flat_brain, rel = load_mask()

    all_ts_per_run = [load_rtmotion(SESSION, r, flat_brain, rel) for r in RUNS]
    noise_per_run = None
    if glmdenoise_K > 0:
        from prereg_variant_sweep import _extract_noise_components_per_run
        noise_per_run = _extract_noise_components_per_run(
            all_ts_per_run, max_K=glmdenoise_K, pool_frac=0.10,
        )
        print(f"  GLMdenoise K={glmdenoise_K} components per run")

    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))
    all_betas = []
    trial_ids = []

    for run_idx, run in enumerate(RUNS):
        ts = all_ts_per_run[run_idx]
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]
            ts = (ts - (ts @ comps) @ comps.T).astype(np.float32)
        events = load_events(SESSION, run)
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]

        # Compute CSF/WM mean timeseries for this run
        wm_ts = ts[wm_mask].mean(axis=0) if wm_mask.any() else np.zeros(n_trs_run)
        csf_ts = ts[csf_mask].mean(axis=0) if csf_mask.any() else np.zeros(n_trs_run)

        for trial_i in range(len(onsets)):
            # Build standard design + add CSF/WM nuisance regressors
            dm, probe_col = build_design_matrix(onsets, TR, n_trs_run, hrf, trial_i)
            # Stack nuisance columns at the end
            dm_aug = np.concatenate([dm, wm_ts[:, None], csf_ts[:, None]], axis=1)
            XtX_inv = np.linalg.inv(dm_aug.T @ dm_aug + 1e-6 * np.eye(dm_aug.shape[1]))
            beta_full = (XtX_inv @ dm_aug.T @ ts.T).T                # (V, p+2)
            all_betas.append(beta_full[:, probe_col].astype(np.float32))
            trial_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))

    betas = np.stack(all_betas, axis=0)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": "OLS_LSS_with_CSF_WM_nuisance",
           "glmdenoise_K": glmdenoise_K, "bold_source": "rtmotion",
           "n_csf_voxels": int(csf_mask.sum()),
           "n_wm_voxels": int(wm_mask.sum()),
           "pve_threshold": 0.5}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    print("=== resampling T1 PVE to BOLD space ===")
    csf_mask, wm_mask = build_csf_wm_masks_at_bold_res()
    fit_one_cell("OLS_AplusN_glover_rtm",            csf_mask, wm_mask, glmdenoise_K=0)
    fit_one_cell("OLS_AplusN_K10_glover_rtm",        csf_mask, wm_mask, glmdenoise_K=10)
