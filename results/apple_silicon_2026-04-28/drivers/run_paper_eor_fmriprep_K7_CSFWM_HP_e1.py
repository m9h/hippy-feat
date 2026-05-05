#!/usr/bin/env python3
"""fMRIPrep BOLD + K=7 CSF/WM aCompCor + HP filter + erode×1 (deployment champion stack on fmriprep).

Mirrors `RT_paper_EoR_K7_CSFWM_HP_e1_inclz` exactly except BOLD source is
fmriprep T1w preproc_bold instead of rtmotion mc_boldres.

Tests whether the ~4pp single-rep gap (Offline 62 vs best EoR 58) closes
when we combine the deployment champion noise stack with fmriprep BOLD.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, nibabel as nib
from scipy.ndimage import binary_erosion
from nilearn.signal import clean
from nilearn.image import resample_to_img

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))
import rt_paper_full_replica as R

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
R.PAPER_ROOT = LOCAL
R.RT3T = LOCAL / "rt3t" / "data"
R.FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005")
R.EVENTS_DIR = LOCAL / "rt3t/data/events"
R.BRAIN_MASK = LOCAL / "rt3t/data/sub-005_final_mask.nii.gz"
R.RELMASK = LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy"
R.MC_DIR = LOCAL / "motion_corrected_resampled"
R.OUT_DIR = LOCAL / "task_2_1_betas/prereg"

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

# Build noise-pool mask (CSF + WM PVE > 0.5, eroded ×1, in T1w/brain space)
brain_img = nib.load(R.BRAIN_MASK)
brain_3d = brain_img.get_fdata() > 0
csf_pve = nib.load(R.RT3T / "T1_brain_seg_pve_0.nii.gz")
wm_pve = nib.load(R.RT3T / "T1_brain_seg_pve_2.nii.gz")
csf_b = resample_to_img(csf_pve, brain_img, interpolation="linear",
                         force_resample=True, copy_header=True).get_fdata()
wm_b = resample_to_img(wm_pve, brain_img, interpolation="linear",
                        force_resample=True, copy_header=True).get_fdata()
csfwm_3d = ((csf_b > PVE_THRESH) | (wm_b > PVE_THRESH)) & brain_3d
csfwm_e1 = binary_erosion(csfwm_3d, iterations=1)
print(f"  noise-pool voxels (CSF∪WM eroded×1 ∩ brain): {csfwm_e1.sum()}", flush=True)

# Extract timeseries from fmriprep BOLD per run
print("\n=== extracting noise-pool timeseries from fmriprep BOLD ===", flush=True)
mask_flat = csfwm_e1.flatten()
ts_e1 = []
for run in RUNS:
    p = (R.FMRIPREP_ROOT / SESSION / "func"
         / f"sub-005_{SESSION}_task-C_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz")
    bold4d = nib.load(p).get_fdata()    # (X, Y, Z, T)
    flat = bold4d.reshape(-1, bold4d.shape[-1]).astype(np.float32)
    ts_e1.append(flat[mask_flat])
    print(f"  run-{run:02d}: BOLD {bold4d.shape}, noise-pool ts {ts_e1[-1].shape}", flush=True)

# HP-filter + SVD per run → top-K components
def make_noise_per_run(ts_list):
    out = []
    for ts in ts_list:
        ts_c = clean(ts.T, t_r=TR, high_pass=0.01, detrend=False, standardize=False).T
        _, _, Vt = np.linalg.svd(ts_c, full_matrices=False)
        out.append(Vt[:K].T.astype(np.float32))
    return out

print("\n=== fitting K=7 PCs per run ===", flush=True)
noise = make_noise_per_run(ts_e1)
print(f"  noise shape per run: {noise[0].shape}", flush=True)

# Loader: combine MCFLIRT .par + noise PCs as confounds
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

R.load_mc_params = make_loader(noise)

CELL = "RT_paper_EoR_fmriprep_K7_CSFWM_HP_e1_inclz"
print(f"\n=== {CELL} ===", flush=True)
t0 = time.time()
betas, trial_ids, config = R.run_cell(
    cell_name=CELL, bold_loader=R.load_fmriprep_4d,
    session=SESSION, runs=RUNS, do_repeat_avg=False,
    streaming_post_stim_TRs=None,
)
config.update({
    "cum_z_formula": "inclusive (arr[:i+1])",
    "bold_source": "fmriprep T1w preproc_bold",
    "GLMdenoise_K": K,
    "GLMdenoise_pool": "CSF ∪ WM via FAST PVEs > 0.5, eroded x1",
    "noise_pool_voxels": int(csfwm_e1.sum()),
    "high_pass_filter_noise_pool": "0.01 Hz",
    "mask_erosion_iterations": 1,
})
np.save(R.OUT_DIR / f"{CELL}_{SESSION}_betas.npy", betas)
np.save(R.OUT_DIR / f"{CELL}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
with open(R.OUT_DIR / f"{CELL}_{SESSION}_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"\n  saved {CELL}: {betas.shape}  ({time.time()-t0:.1f}s)")
