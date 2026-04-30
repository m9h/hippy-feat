#!/usr/bin/env python3
"""K-sweep for GLMdenoise alone (no fracridge contamination).

K = 0 (no denoising), 5, 10, 15. All cells use plain OLS + Glover + frac=1.0.
Tests how much GLMdenoise PCA-on-noise-pool helps in isolation.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import prereg_variant_sweep as P
from prereg_variant_sweep import (
    load_mask, load_rtmotion, load_events,
    _extract_noise_components_per_run,
)
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


def fit_one(name: str, K: int, pool_frac: float = 0.10):
    print(f"\n=== {name} (K={K}, pool_frac={pool_frac}, frac=1.0) ===")
    t0 = time.time()
    flat_brain, rel = load_mask()

    timeseries_per_run = []
    events_per_run = []
    for run in RUNS:
        ts = load_rtmotion(SESSION, run, flat_brain, rel)
        ev = load_events(SESSION, run)
        timeseries_per_run.append(ts)
        events_per_run.append(ev)

    noise_per_run = None
    if K > 0:
        noise_per_run = _extract_noise_components_per_run(
            timeseries_per_run, max_K=K, pool_frac=pool_frac,
        )
        print(f"  GLMdenoise extracted K={K} components per run")

    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))
    all_betas, trial_ids = [], []
    for run_idx, run in enumerate(RUNS):
        ts = timeseries_per_run[run_idx]
        events = events_per_run[run_idx]
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]
            beta_noise = ts @ comps
            ts = (ts - beta_noise @ comps.T).astype(np.float32)
        for trial_i in range(len(onsets)):
            dm, probe_col = build_design_matrix(onsets, TR, n_trs_run, hrf, trial_i)
            XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
            beta_full = (XtX_inv @ dm.T @ ts.T).T               # (V, p)
            all_betas.append(beta_full[:, probe_col].astype(np.float32))
            trial_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))

    betas = np.stack(all_betas, axis=0)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": "OLS_full_beta_probe_extract", "bold_source": "rtmotion",
           "GLMdenoise_K": K, "pool_frac": pool_frac, "fracridge": "1.0 (off)"}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")


CELLS = [
    ("OLS_glover_rtm_denoiseK0",  0),
    ("OLS_glover_rtm_denoiseK5",  5),
    ("OLS_glover_rtm_denoiseK10", 10),
    ("OLS_glover_rtm_denoiseK15", 15),
]

if __name__ == "__main__":
    for name, K in CELLS:
        fit_one(name, K=K)
