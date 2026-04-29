#!/usr/bin/env python3
"""
Rerun the top candidates for Realtime-MindEye performance improvements.
Populates the task_2_1 suite with:
  1. Streaming Kalman Filter (AR1)
  2. Batch Variant G (with AR1)
  3. Riemannian Tangent Space (on parcels)
  4. Baseline Glover OLS
"""

import sys
import time
from pathlib import Path
import numpy as np
import nibabel as nib
import pandas as pd
import jax
import jax.numpy as jnp

# Ensure jaxoccoli is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jaxoccoli.realtime import RTPipeline, RTPipelineConfig, make_glover_hrf, build_lss_design_matrix
from jaxoccoli.streaming_kalman import init_streaming_kalman_ar1, streaming_kalman_ar1_update
from jaxoccoli.matrix import tangent_project_spd

# Load utilities from original factorial script
sys.path.insert(0, str(Path(__file__).resolve().parent))
import task_2_1_factorial

# --- Local Path Fixes (Monkeypatch task_2_1_factorial for local run) ---
LOCAL_ROOT = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
task_2_1_factorial.PAPER_ROOT = LOCAL_ROOT
task_2_1_factorial.RT3T_DATA = LOCAL_ROOT / "rt3t" / "data"
task_2_1_factorial.BRAIN_MASK_PATH = task_2_1_factorial.RT3T_DATA / "sub-005_final_mask.nii.gz"
task_2_1_factorial.RELMASK_PATH = task_2_1_factorial.RT3T_DATA / "sub-005_ses-01_task-C_relmask.npy"
task_2_1_factorial.EVENTS_DIR = task_2_1_factorial.RT3T_DATA / "events"

from task_2_1_factorial import load_paper_finalmask

# Paths
ROOT = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
BOLD_PATH = ROOT / "fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005/ses-03/func/sub-005_ses-03_task-C_run-01_space-T1w_desc-preproc_bold.nii.gz"
EVENTS_PATH = ROOT / "rt3t/data/events/sub-005_ses-03_task-C_run-01_events.tsv"
OUT_DIR = ROOT / "task_2_1_betas/top_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def run_top_candidates(session="ses-03", run=1):
    print(f"Loading data for {session} run {run}...")
    flat_brain, rel = load_paper_finalmask()
    img = nib.load(BOLD_PATH)
    bold_4d = img.get_fdata()
    T = bold_4d.shape[-1]
    # (2792, T)
    ts = bold_4d.reshape(-1, T)[flat_brain][rel].astype(np.float32)
    n_voxels = ts.shape[0]
    
    events = pd.read_csv(EVENTS_PATH, sep="\t")
    onsets = events["onset"].astype(float).values - events["onset"].iloc[0]
    tr = 1.5 # ses-03 is 1.5s TR
    
    hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
    
    results = {}
    
    # --- 1. Baseline Glover OLS ---
    print("Running Baseline Glover OLS...")
    betas_ols = []
    for i in range(len(onsets)):
        dm, probe_idx = build_lss_design_matrix(onsets, tr, T, hrf, i)
        # Surgical OLS
        XtX = dm.T @ dm + 1e-6 * jnp.eye(dm.shape[1])
        b = jnp.linalg.solve(XtX, dm.T @ ts.T).T
        betas_ols.append(np.array(b[:, probe_idx]))
    results["Baseline_Glover"] = np.stack(betas_ols)

    # --- 2. Batch Variant G (AR1) ---
    print("Running Batch Variant G (AR1)...")
    config = RTPipelineConfig(
        tr=tr,
        mask=np.ones(n_voxels, dtype=bool), # ts is already masked
        onsets_sec=onsets,
        max_trs=T + 10
    )
    pipeline = RTPipeline(config)
    pipeline.precompute()
    
    betas_vg = np.zeros((len(onsets), n_voxels))
    for t_idx in range(T):
        res = pipeline.on_volume(ts[:, t_idx], t_idx)
        if res is not None:
            betas_vg[res["probe_trial"]] = res["beta_mean"]
    results["Variant_G"] = betas_vg

    # --- 3. Streaming Kalman Filter (AR1) ---
    print("Running Streaming Kalman (AR1)...")
    # LSS design matrix for all trials
    dm_full = []
    for i in range(len(onsets)):
        dm_i, _ = build_lss_design_matrix(onsets, tr, T, hrf, i)
        dm_full.append(dm_i[:, 0]) # Keep only the probe regressor
    dm_full = np.stack(dm_full, axis=1)
    dm_full = np.column_stack([dm_full, np.ones(T)]) # Add intercept
    
    state = init_streaming_kalman_ar1(dm_full.shape[1], n_voxels)
    for t_idx in range(T):
        state = streaming_kalman_ar1_update(state, jnp.asarray(dm_full[t_idx]), jnp.asarray(ts[:, t_idx]))
    
    betas_kalman = np.array([state.beta_mean[:, i] for i in range(len(onsets))])
    results["Kalman_AR1"] = betas_kalman

    # --- Save Results ---
    for name, betas in results.items():
        np.save(OUT_DIR / f"{name}_betas.npy", betas)
        print(f"Saved {name}: {betas.shape}")
    
    # Save trial IDs (image names)
    img_names = events["image_name"].values
    np.save(OUT_DIR / "trial_ids.npy", img_names)
    
    print("\nSummary (Voxelwise Mean Abs Beta):")
    for name, betas in results.items():
        print(f"{name:<20}: {np.mean(np.abs(betas)):.4f}")

if __name__ == "__main__":
    run_top_candidates()
