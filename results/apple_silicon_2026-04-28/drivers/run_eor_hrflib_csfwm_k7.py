#!/usr/bin/env python3
"""EoR + per-voxel HRF library + CSF/WM K=7 + AR(1) — joint RT-streamable.

Combines two RT-streamable improvements that haven't been tested together:
  - GLMdenoise K=7 from CSF/WM noise pool (best from K-sweep)
  - Per-voxel HRF library (GLMsingle Stage 1 — RT-streamable as just a lookup)
  - AR(1) prewhitening (canonical paper RT engine)

Full-run rtmotion BOLD, inclusive causal cum-z, single-rep filter at scoring.

Hypothesis: HRF library captures per-voxel timing variability that Glover
misses; combined with K=7 CSF/WM noise regression, this should push past
the K=7 ceiling on top-1 / brain retrieval. AUC may also improve.
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

from prereg_variant_sweep import (
    load_mask, load_rtmotion, load_events, _glm_glmsingle_per_voxel_hrf,
)
from rt_glm_variants import load_glmsingle_hrf_library

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
K = 7
PVE_THRESH = 0.5
HRF_LIB_PATH = str(RT3T / "getcanonicalhrflibrary.tsv")
HRF_INDICES_PATH = str(RT3T / "avg_hrfs_s1_s2_full.npy")


def inclusive_cumz(arr: np.ndarray) -> np.ndarray:
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


# Build CSF/WM noise pool
print("=== building CSF/WM noise pool ===")
brain_img = nib.load(P.BRAIN_MASK)
flat_brain = (brain_img.get_fdata() > 0).flatten()
from nilearn.image import resample_to_img
csf = nib.load(RT3T / "T1_brain_seg_pve_0.nii.gz")
wm = nib.load(RT3T / "T1_brain_seg_pve_2.nii.gz")
csf_b = resample_to_img(csf, brain_img, interpolation="linear",
                         force_resample=True, copy_header=True).get_fdata()
wm_b = resample_to_img(wm, brain_img, interpolation="linear",
                        force_resample=True, copy_header=True).get_fdata()
csfwm_brain = (((csf_b > PVE_THRESH) | (wm_b > PVE_THRESH)).flatten()) & flat_brain
print(f"  csfwm voxels: {csfwm_brain.sum()}")

# HRF library + per-voxel indices (project to relmask)
print("\n=== loading HRF library + per-voxel indices ===")
flat_brain2, rel = load_mask()
assert (flat_brain == flat_brain2).all()
hrf_vol = np.load(HRF_INDICES_PATH)[:, :, :, 0]
hrf_2792 = hrf_vol.flatten()[flat_brain][rel].astype(int)
print(f"  HRF idx (2792): unique={len(np.unique(hrf_2792))}, range=[{hrf_2792.min()},{hrf_2792.max()}]")
base_time, hrf_library = load_glmsingle_hrf_library(HRF_LIB_PATH)
print(f"  HRF library: {hrf_library.shape}")


# Per-run BOLD pre-processing: extract relmask timeseries, pre-regress K CSF/WM PCs
print(f"\n=== per-run BOLD pre-processing (CSF/WM K={K} pre-regression) ===")
ts_per_run = []        # cleaned (V=2792, T) per run
events_per_run = []
for run in RUNS:
    ev = load_events(SESSION, run)
    events_per_run.append(ev)
    # Load full-brain BOLD for noise-pool extraction + relmask BOLD for fitting
    pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(P.MC_DIR.glob(pattern))
    csfwm_ts_frames = []
    rel_ts_frames = []
    for v in vols:
        f = nib.load(v).get_fdata().flatten()
        csfwm_ts_frames.append(f[csfwm_brain].astype(np.float32))
        rel_ts_frames.append(f[flat_brain][rel].astype(np.float32))
    csfwm_ts = np.stack(csfwm_ts_frames, axis=1)        # (V_csfwm, T)
    rel_ts = np.stack(rel_ts_frames, axis=1)              # (V=2792, T)

    # Compute K=7 noise components from CSF/WM; pre-regress out of relmask BOLD
    cw_centered = csfwm_ts - csfwm_ts.mean(axis=1, keepdims=True)
    _, _, Vt = np.linalg.svd(cw_centered, full_matrices=False)
    comps = Vt[:K].T.astype(np.float32)                  # (T, K)
    rel_centered = rel_ts - rel_ts.mean(axis=1, keepdims=True)
    beta_n = rel_centered @ comps                        # (V, K)
    rel_clean = rel_ts - beta_n @ comps.T
    ts_per_run.append(rel_clean.astype(np.float32))
    print(f"  run-{run:02d}: BOLD {rel_ts.shape}, cleaned-by-K={K}-CSF/WM-PCs")


# Per-trial fit: HRF library + AR(1) prewhitened OLS
print("\n=== fitting per-trial GLM (HRF library + AR(1)) ===")
all_betas = []
all_image_names = []
t0 = time.time()
for run_idx, run in enumerate(RUNS):
    ts = ts_per_run[run_idx]
    ev = events_per_run[run_idx]
    onsets = ev["onset_rel"].values.astype(np.float32)
    n_trs = ts.shape[1]
    for trial_i in range(len(onsets)):
        beta = _glm_glmsingle_per_voxel_hrf(
            ts, onsets, trial_i, TR, n_trs,
            hrf_indices=hrf_2792, hrf_library=hrf_library, base_time=base_time,
            mode="ar1_freq",
        )
        all_betas.append(beta.astype(np.float32))
        all_image_names.append(str(ev.iloc[trial_i].get("image_name", str(trial_i))))
    print(f"  run-{run:02d} done ({time.time()-t0:.1f}s)")

raw = np.stack(all_betas, axis=0)
print(f"\n  raw βs: {raw.shape}")

# Inclusive causal cum-z
z = inclusive_cumz(raw)

cell_name = f"RT_paper_EoR_hrflib_K{K}_CSFWM_AR1_inclz"
np.save(OUT_DIR / f"{cell_name}_{SESSION}_betas.npy", z)
np.save(OUT_DIR / f"{cell_name}_{SESSION}_trial_ids.npy", np.asarray(all_image_names))
cfg = {
    "cell": cell_name, "session": SESSION, "runs": RUNS, "tr": TR,
    "engine": "JAX OLS+AR1freq + per-voxel HRF library",
    "GLMdenoise_K": K, "GLMdenoise_pool": "CSF ∪ WM PVE>0.5 ∩ flat_brain",
    "n_csfwm_voxels": int(csfwm_brain.sum()),
    "hrf_library": "GLMsingle 20-HRF library, per-voxel index from avg_hrfs_s1_s2_full.npy",
    "noise_model": "ar1_freq (per-voxel AR(1) prewhitening)",
    "cum_z_formula": "inclusive (arr[:i+1])",
    "bold_source": "rtmotion",
    "windowing": "full-run (EoR equivalent)",
}
with open(OUT_DIR / f"{cell_name}_{SESSION}_config.json", "w") as f:
    json.dump(cfg, f, indent=2)
print(f"  saved {cell_name}: {z.shape}  ({time.time()-t0:.1f}s)")
