#!/usr/bin/env python3
"""Stein-shrinkage variants on top of the FAST K=7+HP+e1 champion config.

Rebuilds the champion pipeline saving RAW LSS βs (pre-cum-z), then applies
per-trial shrinkage toward the running mean before applying causal cum-z:

    β_shrunk[i] = α * β[i] + (1 - α) * mean(β[:i])

α=1.0  → no shrinkage (= champion baseline)
α=0.95 → mild shrinkage toward past trials' running mean
α=0.85 → stronger
α=0.70 → aggressive

For α<1, recent trials are pulled toward the running session mean,
which acts as a stabilizer for early trials (small i, more weight on
running mean) and asymptotes to identity for late trials.

Note: applying Stein on POST-cum-z betas is a no-op for retrieval (the
running mean is already zero by construction). Must apply pre-cum-z.
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
ALPHAS = [1.0, 0.95, 0.85, 0.7]


def causal_cumz(arr: np.ndarray) -> np.ndarray:
    n, V = arr.shape
    z = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


def stein_shrink(raw: np.ndarray, alpha: float) -> np.ndarray:
    """β_shrunk[i] = α * β[i] + (1-α) * mean(β[:i])."""
    n, V = raw.shape
    out = np.zeros_like(raw, dtype=np.float32)
    out[0] = raw[0]  # i=0: no past mean, leave as-is
    for i in range(1, n):
        running_mean = raw[:i].mean(axis=0)
        out[i] = alpha * raw[i] + (1 - alpha) * running_mean
    return out


# Build aCompCor pool (FAST K=7 + HP + erode×1) — same as champion
print("=== building FAST CSF/WM aCompCor pool (erode ×1) ===")
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
print(f"  pool: {csfwm_e1.sum()} voxels", flush=True)

ts_per_run = []
for run in RUNS:
    pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(R.MC_DIR.glob(pattern))
    frames = [nib.load(v).get_fdata().flatten()[csfwm_e1.flatten()].astype(np.float32) for v in vols]
    ts_per_run.append(np.stack(frames, axis=1))

noise_per_run = []
for ts in ts_per_run:
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


# Patch cumulative_zscore to be IDENTITY (so run_cell saves RAW betas).
def identity(beta_history, image_history, do_repeat_avg):
    return np.stack(beta_history, axis=0).astype(np.float32), list(image_history)


R.cumulative_zscore_with_optional_repeat_avg = identity

# Run once to extract raw betas
RAW_CELL = "RT_paper_EoR_K7_CSFWM_HP_e1_RAW"
raw_path = R.OUT_DIR / f"{RAW_CELL}_{SESSION}_betas.npy"
if raw_path.exists():
    print(f"\n=== raw betas already on disk: {raw_path.name} ===")
    raw = np.load(raw_path)
    trial_ids = np.load(R.OUT_DIR / f"{RAW_CELL}_{SESSION}_trial_ids.npy", allow_pickle=True)
else:
    print("\n=== fitting RAW LSS βs (K=7+HP+e1, no cum-z) ===", flush=True)
    t0 = time.time()
    raw, trial_ids, _ = R.run_cell(
        cell_name=RAW_CELL, bold_loader=R.load_rtmotion_4d,
        session=SESSION, runs=RUNS, do_repeat_avg=False,
        streaming_post_stim_TRs=None,
    )
    np.save(raw_path, raw)
    np.save(R.OUT_DIR / f"{RAW_CELL}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    print(f"  saved {RAW_CELL}: {raw.shape}  ({time.time()-t0:.1f}s)", flush=True)

# Apply Stein shrinkage at each α + causal cum-z
print(f"\n=== applying Stein shrinkage + cum-z for α ∈ {ALPHAS} ===", flush=True)
for alpha in ALPHAS:
    cell = f"RT_paper_EoR_K7_CSFWM_HP_e1_stein{int(alpha*100):03d}_inclz"
    if (R.OUT_DIR / f"{cell}_{SESSION}_betas.npy").exists():
        print(f"  {cell} exists — skip")
        continue
    shrunk = stein_shrink(raw, alpha)
    z = causal_cumz(shrunk)
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_betas.npy", z)
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    cfg = {
        "cell": cell, "session": SESSION, "runs": RUNS, "tr": TR,
        "GLMdenoise_K": K, "GLMdenoise_pool": "FAST CSF ∪ WM PVE > 0.5, eroded ×1",
        "high_pass_filter_noise_pool": "0.01 Hz",
        "stein_alpha": alpha,
        "stein_formula": "β_shrunk[i] = α·β[i] + (1-α)·mean(β[:i])  (i≥1; β_shrunk[0]=β[0])",
        "z_score": "causal cum-z (inclusive arr[:i+1]) on shrunk βs",
        "engine": "nilearn FirstLevelModel + Glover + AR(1) + cosine drift",
    }
    with open(R.OUT_DIR / f"{cell}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {cell}: {z.shape}", flush=True)
print("done")
