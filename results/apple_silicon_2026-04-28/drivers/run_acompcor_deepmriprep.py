#!/usr/bin/env python3
"""aCompCor with DeepMriPrep-derived noise pool (T1Prep's segmentation engine).

Same recipe as the FAST-based champion (K=7 + HP-filter + erode×1) but
swap FAST PVEs for DeepMriPrep p2 (WM) and p3 (CSF) probability maps.

DeepMriPrep is a deep-learning T1 segmentation tool (CAT12-equivalent)
that runs natively on Apple Silicon via PyTorch. T1Prep wraps it.
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
PVE_THRESH = 0.5

DMP_DIR = R.RT3T / "deepmriprep_out"
P2 = DMP_DIR / "p2sub-005_desc-preproc_T1w.nii.gz"   # WM
P3 = DMP_DIR / "p3sub-005_desc-preproc_T1w.nii.gz"   # CSF


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


print("=== building DeepMriPrep noise pool ===")
brain_img = nib.load(R.BRAIN_MASK)
brain_3d = brain_img.get_fdata() > 0

p2 = nib.load(P2)
p3 = nib.load(P3)
p2_b = resample_to_img(p2, brain_img, interpolation="linear",
                        force_resample=True, copy_header=True).get_fdata()
p3_b = resample_to_img(p3, brain_img, interpolation="linear",
                        force_resample=True, copy_header=True).get_fdata()

csfwm_3d = ((p2_b > PVE_THRESH) | (p3_b > PVE_THRESH)) & brain_3d
print(f"  uneroded CSF∪WM ∩ brain (DeepMriPrep): {csfwm_3d.sum()} voxels")

csfwm_e1 = binary_erosion(csfwm_3d, iterations=1)
print(f"  eroded ×1: {csfwm_e1.sum()} voxels")


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
ts_per_run = extract_ts(csfwm_e1)
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
    if run < 1 or run > len(noise_per_run): return mc
    comps = noise_per_run[run - 1]
    if mc is None: return comps
    n_match = min(mc.shape[0], comps.shape[0])
    return np.concatenate([mc[:n_match], comps[:n_match]], axis=1).astype(np.float32)


R.load_mc_params = loader

CELL = "RT_paper_EoR_K7_deepmriprep_HP_e1_inclz"
print(f"\n=== {CELL}  K={K}, HP=on, erode×1, n_pool={csfwm_e1.sum()} ===")
t0 = time.time()
betas, trial_ids, config = R.run_cell(
    cell_name=CELL, bold_loader=R.load_rtmotion_4d,
    session=SESSION, runs=RUNS, do_repeat_avg=False,
    streaming_post_stim_TRs=None,
)
config.update({
    "cum_z_formula": "inclusive (arr[:i+1])",
    "GLMdenoise_K": K,
    "GLMdenoise_pool": "DeepMriPrep p2 (WM) ∪ p3 (CSF) > 0.5, eroded ×1",
    "noise_pool_voxels": int(csfwm_e1.sum()),
    "high_pass_filter_noise_pool": "0.01 Hz",
    "mask_erosion_iterations": 1,
    "segmentation": "deepmriprep-cli (uv tool install --python 3.12 deepmriprep)",
})
np.save(R.OUT_DIR / f"{CELL}_{SESSION}_betas.npy", betas)
np.save(R.OUT_DIR / f"{CELL}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
with open(R.OUT_DIR / f"{CELL}_{SESSION}_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"  saved {CELL}: {betas.shape}  ({time.time()-t0:.1f}s)")
