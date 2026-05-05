#!/usr/bin/env python3
"""Extract Fast (pst=5) per-trial AR(1) LSS βs for ses-01 and ses-02 from fmriprep BOLD.

Mirrors the existing RT_paper_Fast_pst5_inclz cell on ses-03 but using fmriprep BOLD
(since rtmotion BOLD isn't local for ses-01/02). For training a Fast-tier refiner,
extra training data from these sessions improves the per-voxel refiner's generalization.

Output cells:
  RT_paper_Fast_pst5_fmriprep_inclz_ses-01
  RT_paper_Fast_pst5_fmriprep_inclz_ses-02
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))
import rt_paper_full_replica as R

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
R.PAPER_ROOT = LOCAL
R.RT3T = LOCAL / "rt3t/data"
R.FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005")
R.EVENTS_DIR = LOCAL / "rt3t/data/events"
R.BRAIN_MASK = LOCAL / "rt3t/data/sub-005_final_mask.nii.gz"
R.RELMASK = LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy"
R.MC_DIR = LOCAL / "motion_corrected_resampled"
R.OUT_DIR = LOCAL / "task_2_1_betas/prereg"


def _local_load_mc(session, run):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    if par.exists():
        return np.loadtxt(par).astype(np.float32)
    return None
R.load_mc_params = _local_load_mc


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

RUNS = list(range(1, 12))
PST = 5

for SESSION in ["ses-01", "ses-02"]:
    cell = f"RT_paper_Fast_pst{PST}_fmriprep_inclz"
    out_betas = R.OUT_DIR / f"{cell}_{SESSION}_betas.npy"
    if out_betas.exists():
        print(f"=== {cell} {SESSION} already exists — skip ===", flush=True)
        continue
    print(f"\n=== {cell} {SESSION} (fmriprep + AR(1) LSS, pst={PST}) ===", flush=True)
    t0 = time.time()
    try:
        betas, trial_ids, config = R.run_cell(
            cell_name=cell, bold_loader=R.load_fmriprep_4d,
            session=SESSION, runs=RUNS, do_repeat_avg=False,
            streaming_post_stim_TRs=PST,
        )
    except Exception as e:
        print(f"  FAILED: {e}", flush=True)
        continue
    config.update({
        "cum_z_formula": "inclusive (arr[:i+1])",
        "bold_source": "fmriprep T1w preproc_bold",
        "streaming_post_stim_TRs": PST,
    })
    np.save(out_betas, betas)
    np.save(R.OUT_DIR / f"{cell}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{cell}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {cell} {SESSION}: {betas.shape}  ({time.time()-t0:.1f}s)", flush=True)
