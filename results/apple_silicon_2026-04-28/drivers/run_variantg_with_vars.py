#!/usr/bin/env python3
"""Re-run Variant G cells saving both β_mean and β_var per trial.

The standard run_glm_cell discards the per-voxel variance returned by
_glm_jax. For Bayesian classification (the actual point of Variant G),
we need both. Save as `{cell}_ses-03_vars.npy` alongside `_betas.npy`.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import jax.numpy as jnp

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import prereg_variant_sweep as P
from prereg_variant_sweep import (
    load_mask, load_rtmotion, load_events,
    _glm_jax, _extract_noise_components_per_run,
)

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
P.PAPER_ROOT = LOCAL
P.RT3T = LOCAL / "rt3t" / "data"
P.FMRIPREP_ROOT = (LOCAL / "fmriprep_mindeye" / "data_sub-005"
                   / "bids" / "derivatives" / "fmriprep" / "sub-005")
P.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
P.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
P.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5

# Cells with Bayesian posterior output worth re-evaluating
CELLS = [
    ("VariantG_glover_rtm_with_vars", dict(mode="variant_g", denoise=None)),
    ("VariantG_glover_rtm_glmdenoise_fracridge_with_vars",
        dict(mode="variant_g", denoise="glmdenoise_fracridge")),
    ("VariantG_glover_rtm_acompcor_with_vars",
        dict(mode="variant_g", denoise="tcompcor")),
    # Plus streaming pst=8 version of bare Variant G — for cross-window comparison
    ("VariantG_glover_rtm_streaming_pst8_with_vars",
        dict(mode="variant_g", denoise=None, streaming_pst=8)),
]


def run_one(cell_name: str, mode: str, denoise: str | None,
            streaming_pst: int | None = None) -> None:
    flat_brain, rel = load_mask()
    all_betas, all_vars, trial_ids = [], [], []

    # Load all runs first so we can do per-run noise components if needed
    timeseries_per_run = []
    events_per_run = []
    for run in RUNS:
        ts = load_rtmotion(SESSION, run, flat_brain, rel)
        ev = load_events(SESSION, run)
        timeseries_per_run.append(ts)
        events_per_run.append(ev)

    noise_per_run = None
    if denoise in ("glmdenoise_fracridge", "tcompcor"):
        K = 5
        pool_frac = 0.05 if denoise == "tcompcor" else 0.10
        noise_per_run = _extract_noise_components_per_run(
            timeseries_per_run, max_K=K, pool_frac=pool_frac,
        )
        print(f"  [denoise={denoise}] extracted K={K} per run")

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
            if streaming_pst is not None:
                onset_TR = int(round(onsets[trial_i] / TR))
                decode_TR = min(onset_TR + streaming_pst, n_trs_run - 1)
                if decode_TR < 8:
                    continue
                n_eff = decode_TR + 1
                ts_used = ts[:, :n_eff]
            else:
                n_eff = n_trs_run
                ts_used = ts
            beta, var = _glm_jax(
                ts_used, onsets, trial_i, TR, n_eff, mode=mode, max_trs=200,
            )
            all_betas.append(np.asarray(beta, dtype=np.float32))
            all_vars.append(np.asarray(var, dtype=np.float32))
            trial_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))

    betas = np.stack(all_betas, axis=0)
    vars_ = np.stack(all_vars, axis=0)
    P.OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(P.OUT_DIR / f"{cell_name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{cell_name}_{SESSION}_vars.npy", vars_)
    np.save(P.OUT_DIR / f"{cell_name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    print(f"  saved {cell_name}: β {betas.shape}, var mean={vars_.mean():.4f}")


for name, kw in CELLS:
    print(f"\n=== {name} ({kw}) ===")
    t0 = time.time()
    run_one(name, **kw)
    print(f"  elapsed: {time.time()-t0:.1f}s")
