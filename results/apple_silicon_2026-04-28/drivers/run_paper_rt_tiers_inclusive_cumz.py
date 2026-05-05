#!/usr/bin/env python3
"""Retry RT tiers with INCLUSIVE causal cum-z (arr[:i+1]) — match canonical
mindeye.py:770-784 which uses np.mean(all_betas) over all-up-to-and-including-current.

Our prior version used arr[:i] (exclusive). Difference matters most when n is
small; for end-of-run with n~770 by end of session, the effect on late-session
stats should be small but on early-session test trials may be larger.

Three cells re-run: Fast pst=5, Slow pst=20, End-of-run pst=None.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import rt_paper_full_replica as R

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
R.PAPER_ROOT = LOCAL
R.RT3T = LOCAL / "rt3t" / "data"
R.FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye" / "data_sub-005"
                   / "bids" / "derivatives" / "fmriprep" / "sub-005")
R.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
R.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
R.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
R.MC_DIR = LOCAL / "motion_corrected_resampled"
R.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"


def _local_load_mc(session: str, run: int):
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    if par.exists():
        return np.loadtxt(par).astype(np.float32)
    return None
R.load_mc_params = _local_load_mc


# ------------------------------------------------------------ INCLUSIVE cum-z
def inclusive_cumulative_zscore_with_optional_repeat_avg(
        beta_history, image_history, do_repeat_avg):
    """At trial i, stats use arr[:i+1] (INCLUSIVE — current trial counted).
    Matches canonical mindeye.py:770 np.mean(all_betas) where all_betas already
    contains the current trial. Diverges from rt_paper_full_replica's
    exclusive version arr[:i]."""
    arr = np.stack(beta_history, axis=0).astype(np.float32)
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        # arr[:i+1] guarantees at least one trial; std over a single trial is 0
        # so add the same +1e-6 epsilon as the canonical
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd

    if not do_repeat_avg:
        return z, list(image_history)

    seen = {}
    out_betas, out_ids = [], []
    for i, img in enumerate(image_history):
        seen.setdefault(img, []).append(i)
        if len(seen[img]) == 1:
            out_betas.append(z[i])
            out_ids.append(img)
        else:
            avg = z[seen[img]].mean(axis=0)
            first_pos = next(j for j, l in enumerate(out_ids) if l == img)
            out_betas[first_pos] = avg
    return np.stack(out_betas, axis=0), out_ids


# Monkey-patch
R.cumulative_zscore_with_optional_repeat_avg = inclusive_cumulative_zscore_with_optional_repeat_avg


SESSION = "ses-03"
RUNS = list(range(1, 12))

TIERS = [
    ("RT_paper_Fast_pst5_inclz",      5),       # Fast
    ("RT_paper_Slow_pst20_inclz",     20),      # Slow
    ("RT_paper_EndOfRun_pst_None_inclz", None), # End-of-run = full-run BOLD
]

R.OUT_DIR.mkdir(parents=True, exist_ok=True)

for name, pst in TIERS:
    pst_label = f"pst={pst} TRs ≈ {pst*1.5:.1f}s" if pst is not None else "pst=None (full-run)"
    print(f"\n=== {name}  ({pst_label}, single-rep, INCLUSIVE cum-z) ===")
    t0 = time.time()
    betas, trial_ids, config = R.run_cell(
        cell_name=name,
        bold_loader=R.load_rtmotion_4d,
        session=SESSION, runs=RUNS,
        do_repeat_avg=False,
        streaming_post_stim_TRs=pst,
    )
    config["cum_z_formula"] = "inclusive (arr[:i+1])"
    np.save(R.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(R.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    with open(R.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"saved {name}: betas {betas.shape}  ({time.time()-t0:.1f}s)")
