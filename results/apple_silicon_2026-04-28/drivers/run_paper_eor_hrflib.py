#!/usr/bin/env python3
"""End-of-run + per-voxel HRF library (GLMsingle Stage 1).

Replaces the single Glover HRF in the EoR cell with per-voxel HRF lookup
from the GLMsingle 20-HRF library, indexed by `avg_hrfs_s1_s2_full.npy`.

This is a self-contained OLS LSS implementation (nilearn's FirstLevelModel
doesn't support per-voxel HRF). Full-run BOLD, single-rep, inclusive cum-z,
NO AR(1) — to keep the swap surgical we change ONLY the HRF.

Comparison baseline: needs a matched OLS+Glover EoR cell to attribute the
contribution cleanly. We emit BOTH cells from this driver so the AR(1) =
nilearn confound is held constant across the comparison.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

from prereg_variant_sweep import load_mask, load_rtmotion, load_events
from rt_glm_variants import (
    build_design_matrix, make_glover_hrf,
    load_glmsingle_hrf_library, resample_hrf,
)

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
RT3T = LOCAL / "rt3t" / "data"
OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"

import prereg_variant_sweep as P
P.RT3T = RT3T
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.BRAIN_MASK = RT3T / "sub-005_final_mask.nii.gz"
P.RELMASK = RT3T / "sub-005_ses-01_task-C_relmask.npy"
P.EVENTS_DIR = RT3T / "events"

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5
HRF_LIB_PATH = str(RT3T / "getcanonicalhrflibrary.tsv")
HRF_INDICES_PATH = str(RT3T / "avg_hrfs_s1_s2_full.npy")


def fit_run_lss_per_voxel_hrf(ts: np.ndarray, onsets: np.ndarray,
                                hrf_indices_2792: np.ndarray,
                                hrf_library: np.ndarray, base_time: np.ndarray,
                                n_trs: int) -> np.ndarray:
    """OLS LSS over a full run with per-voxel HRF lookup.

    Args:
        ts: (V=2792, T) timeseries
        onsets: (n_trials,) seconds-relative-to-run-start
        hrf_indices_2792: (V=2792,) int 0..19
        hrf_library: (n_native, 20)
        base_time: (n_native,) time axis for hrf_library

    Returns: (n_trials, V) per-trial betas via "current vs all others" LSS.
    """
    n_hrf_trs = int(np.ceil(32.0 / TR))
    V = ts.shape[0]
    n_trials = len(onsets)
    out = np.zeros((n_trials, V), dtype=np.float32)
    unique_hrfs = np.unique(hrf_indices_2792)

    # Precompute resampled HRFs per index
    resampled = {int(h): resample_hrf(hrf_library[:, int(h)], base_time, TR, n_hrf_trs)
                 for h in unique_hrfs}

    for trial_i in range(n_trials):
        for h in unique_hrfs:
            voxel_ids = np.where(hrf_indices_2792 == int(h))[0]
            if len(voxel_ids) == 0:
                continue
            dm, probe_col = build_design_matrix(onsets, TR, n_trs, resampled[int(h)], trial_i)
            XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
            beta = (XtX_inv @ dm.T @ ts[voxel_ids].T).T            # (V_h, p)
            out[trial_i, voxel_ids] = beta[:, probe_col].astype(np.float32)
    return out


def fit_run_lss_glover(ts: np.ndarray, onsets: np.ndarray, n_trs: int) -> np.ndarray:
    n_hrf_trs = int(np.ceil(32.0 / TR))
    hrf = make_glover_hrf(TR, n_hrf_trs)
    n_trials = len(onsets)
    V = ts.shape[0]
    out = np.zeros((n_trials, V), dtype=np.float32)
    for trial_i in range(n_trials):
        dm, probe_col = build_design_matrix(onsets, TR, n_trs, hrf, trial_i)
        XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
        beta = (XtX_inv @ dm.T @ ts.T).T
        out[trial_i] = beta[:, probe_col].astype(np.float32)
    return out


def inclusive_cumz(arr: np.ndarray) -> np.ndarray:
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


# Load HRF library + per-voxel indices, project to relmask voxels
print("=== loading HRF library + per-voxel indices ===")
flat_brain, rel = load_mask()
print(f"  flat_brain={flat_brain.sum()}  rel={rel.sum()}")
hrf_vol = np.load(HRF_INDICES_PATH)[:, :, :, 0]                  # (76,90,74)
hrf_brain = hrf_vol.flatten()[flat_brain].astype(int)             # (19174,)
hrf_2792 = hrf_brain[rel]                                          # (2792,)
print(f"  per-voxel HRF idx (2792): min={hrf_2792.min()} max={hrf_2792.max()} unique={len(np.unique(hrf_2792))}")
base_time, hrf_library = load_glmsingle_hrf_library(HRF_LIB_PATH)
print(f"  HRF library: {hrf_library.shape}, base_time {base_time.shape}")

CELLS = [
    ("RT_paper_EoR_OLS_glover_inclz", "glover"),
    ("RT_paper_EoR_OLS_hrflib_inclz", "hrflib"),
]

for cell_name, hrf_mode in CELLS:
    print(f"\n=== {cell_name}  (full-run, OLS, {hrf_mode}, single-rep, inclusive cum-z) ===")
    t0 = time.time()
    all_betas = []
    image_history = []

    for run in RUNS:
        ts = load_rtmotion(SESSION, run, flat_brain, rel)
        events = load_events(SESSION, run)
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs = ts.shape[1]
        if hrf_mode == "glover":
            run_betas = fit_run_lss_glover(ts, onsets, n_trs)
        else:
            run_betas = fit_run_lss_per_voxel_hrf(
                ts, onsets, hrf_2792, hrf_library, base_time, n_trs,
            )
        all_betas.append(run_betas)
        image_history.extend([
            str(events.iloc[i].get("image_name", str(i))) for i in range(len(onsets))
        ])
        print(f"  run-{run:02d}: {run_betas.shape} ({time.time()-t0:.1f}s elapsed)")

    raw = np.concatenate(all_betas, axis=0)                       # (n_trials, V)
    z = inclusive_cumz(raw)
    np.save(OUT_DIR / f"{cell_name}_{SESSION}_betas.npy", z)
    np.save(OUT_DIR / f"{cell_name}_{SESSION}_trial_ids.npy", np.asarray(image_history))
    cfg = {
        "cell": cell_name, "session": SESSION, "runs": RUNS, "tr": TR,
        "engine": "OLS LSS (no AR1)", "hrf_mode": hrf_mode,
        "bold_source": "rtmotion", "windowing": "full-run (EoR equivalent)",
        "cum_z_formula": "inclusive (arr[:i+1])",
    }
    with open(OUT_DIR / f"{cell_name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {cell_name}: betas {z.shape}  ({time.time()-t0:.1f}s)")
