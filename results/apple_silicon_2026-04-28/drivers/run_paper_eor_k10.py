#!/usr/bin/env python3
"""End-of-run + GLMdenoise K=10 hypothesis test.

Question: does adding GLMdenoise K=10 noise-pool PCA components to the
End-of-run cell close the 10pp gap (56% reproduced vs 66% paper)?

Setup: identical to RT_paper_EndOfRun_pst_None_inclz, but adds the K=10
noise components (extracted via `_extract_noise_components_per_run` from
the relmask timeseries, top-10% variance pool) as nuisance regressors in
the per-trial nilearn FirstLevelModel fit. This is equivalent to canonical
GLMdenoise (PCA noise-pool components added to design matrix).

Mechanism implementation: monkey-patch R.load_mc_params to return
[motion_par | noise_components] stacked horizontally per run. The existing
fit_lss_nilearn pipeline picks them up as confounds via the standard
`confounds` argument to FirstLevelModel.
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
from prereg_variant_sweep import (
    load_mask, load_rtmotion, _extract_noise_components_per_run,
)

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

import prereg_variant_sweep as P
P.RT3T = R.RT3T
P.MC_DIR = R.MC_DIR
P.BRAIN_MASK = R.BRAIN_MASK
P.RELMASK = R.RELMASK
P.EVENTS_DIR = R.EVENTS_DIR

SESSION = "ses-03"
RUNS = list(range(1, 12))
K = 10
POOL_FRAC = 0.10


def inclusive_cumulative_zscore(beta_history, image_history, do_repeat_avg):
    arr = np.stack(beta_history, axis=0).astype(np.float32)
    n, V = arr.shape
    z = np.zeros_like(arr)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    if not do_repeat_avg:
        return z, list(image_history)
    seen = {}
    out_b, out_i = [], []
    for i, img in enumerate(image_history):
        seen.setdefault(img, []).append(i)
        if len(seen[img]) == 1:
            out_b.append(z[i]); out_i.append(img)
        else:
            avg = z[seen[img]].mean(axis=0)
            first_pos = next(j for j, l in enumerate(out_i) if l == img)
            out_b[first_pos] = avg
    return np.stack(out_b, axis=0), out_i


R.cumulative_zscore_with_optional_repeat_avg = inclusive_cumulative_zscore


print(f"=== precomputing K={K} noise components per run (pool_frac={POOL_FRAC}) ===")
flat_brain, rel = load_mask()
timeseries_per_run = []
for run in RUNS:
    ts = load_rtmotion(SESSION, run, flat_brain, rel)
    timeseries_per_run.append(ts)
    print(f"  run-{run:02d}: ts shape {ts.shape}")

noise_per_run = _extract_noise_components_per_run(
    timeseries_per_run, max_K=K, pool_frac=POOL_FRAC,
)
print(f"  extracted noise components: {[c.shape for c in noise_per_run]}")


def load_mc_with_noise(session: str, run: int):
    """Return [motion(T,6) | noise_components(T,K)] horizontally stacked."""
    par = R.MC_DIR / f"{session}_run-{run:02d}_motion.par"
    mc = np.loadtxt(par).astype(np.float32) if par.exists() else None
    if run < 1 or run > len(noise_per_run):
        return mc
    comps = noise_per_run[run - 1]                              # (T_r, K)
    if mc is None:
        return comps.astype(np.float32)
    n_match = min(mc.shape[0], comps.shape[0])
    return np.concatenate([mc[:n_match], comps[:n_match]], axis=1).astype(np.float32)


R.load_mc_params = load_mc_with_noise

CELL_NAME = "RT_paper_EoR_K10_inclz"
print(f"\n=== {CELL_NAME}  (pst=None full-run, K={K}, single-rep, inclusive cum-z) ===")
t0 = time.time()
betas, trial_ids, config = R.run_cell(
    cell_name=CELL_NAME,
    bold_loader=R.load_rtmotion_4d,
    session=SESSION, runs=RUNS,
    do_repeat_avg=False,
    streaming_post_stim_TRs=None,
)
config["cum_z_formula"] = "inclusive (arr[:i+1])"
config["GLMdenoise_K"] = K
config["GLMdenoise_pool_frac"] = POOL_FRAC
config["GLMdenoise_application"] = "noise components passed as nilearn confounds (alongside MCFLIRT motion)"
np.save(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_betas.npy", betas)
np.save(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
with open(R.OUT_DIR / f"{CELL_NAME}_{SESSION}_config.json", "w") as f:
    json.dump(config, f, indent=2)
print(f"saved {CELL_NAME}: betas {betas.shape}  ({time.time()-t0:.1f}s)")
