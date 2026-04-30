#!/usr/bin/env python3
"""Delay-model sensitivity sweep — does boxcar duration in HRF convolution matter?

All cells: OLS + GLMdenoise K=10 + Glover canonical HRF + LSS, only the
boxcar duration before HRF-convolution varies. Same JAX backend, same
mask, same noise-pool extraction.

Cells:
  duration=0 (pure delta at onset_TR)             — what JAX cells used
  duration=1 (1s boxcar)                          — what nilearn cells used
  duration=2 (2s boxcar)
  duration=3 (3s, the events.tsv true duration)   — what we tested earlier on nilearn cells
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
from prereg_variant_sweep import load_mask, load_rtmotion, load_events, _extract_noise_components_per_run
from rt_glm_variants import make_glover_hrf

P.RT3T = LOCAL / "rt3t" / "data"
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
P.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
P.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
P.OUT_DIR = LOCAL / "task_2_1_betas" / "prereg"

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5


def build_design_matrix_with_duration(events_onsets: np.ndarray, tr: float, n_trs: int,
                                       hrf: np.ndarray, probe_trial: int,
                                       duration: float) -> tuple[np.ndarray, int]:
    """LSS design matrix with explicit boxcar duration before HRF convolution.

    duration=0 → delta at onset_TR (impulse)
    duration>0 → boxcar of `duration` seconds starting at onset
    """
    duration_trs = max(1, int(round(duration / tr))) if duration > 0 else 1
    use_delta = duration <= 0

    probe_onset = events_onsets[probe_trial]
    probe_boxcar = np.zeros(n_trs, dtype=np.float32)
    probe_tr = int(round(probe_onset / tr))
    if 0 <= probe_tr < n_trs:
        if use_delta:
            probe_boxcar[probe_tr] = 1.0
        else:
            for k in range(duration_trs):
                if probe_tr + k < n_trs:
                    probe_boxcar[probe_tr + k] = 1.0 / duration_trs

    ref_boxcar = np.zeros(n_trs, dtype=np.float32)
    for i, onset in enumerate(events_onsets):
        if i == probe_trial:
            continue
        ref_tr = int(round(onset / tr))
        if 0 <= ref_tr < n_trs:
            if use_delta:
                ref_boxcar[ref_tr] = 1.0
            else:
                for k in range(duration_trs):
                    if ref_tr + k < n_trs:
                        ref_boxcar[ref_tr + k] = 1.0 / duration_trs

    probe_reg = np.convolve(probe_boxcar, hrf)[:n_trs]
    ref_reg = np.convolve(ref_boxcar, hrf)[:n_trs]

    intercept = np.ones(n_trs, dtype=np.float32)
    t = np.arange(n_trs, dtype=np.float32) / max(n_trs - 1, 1)
    drift = np.cos(2 * np.pi * t)

    dm = np.column_stack([probe_reg, ref_reg, intercept, drift]).astype(np.float32)
    probe_idx = 0
    return dm, probe_idx


def fit_one_cell(name: str, duration: float, glmdenoise_K: int = 10):
    print(f"\n=== {name} (K={glmdenoise_K}, duration={duration}s) ===")
    t0 = time.time()
    flat_brain, rel = load_mask()

    all_ts_per_run = [load_rtmotion(SESSION, r, flat_brain, rel) for r in RUNS]
    noise_per_run = _extract_noise_components_per_run(
        all_ts_per_run, max_K=glmdenoise_K, pool_frac=0.10,
    ) if glmdenoise_K > 0 else None

    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))
    all_betas, trial_ids = [], []

    for run_idx, run in enumerate(RUNS):
        ts = all_ts_per_run[run_idx]
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]
            ts = (ts - (ts @ comps) @ comps.T).astype(np.float32)
        events = load_events(SESSION, run)
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]

        for trial_i in range(len(onsets)):
            dm, probe_col = build_design_matrix_with_duration(
                onsets, TR, n_trs_run, hrf, trial_i, duration=duration,
            )
            XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
            beta_full = (XtX_inv @ dm.T @ ts.T).T
            all_betas.append(beta_full[:, probe_col].astype(np.float32))
            trial_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))

    betas = np.stack(all_betas, axis=0)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "hrf": "glover_canonical", "boxcar_duration_s": duration,
           "glmdenoise_K": glmdenoise_K, "bold_source": "rtmotion"}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")


CELLS = [
    ("OLS_K10_dur0_glover_rtm", 0.0),    # delta at onset
    ("OLS_K10_dur1_glover_rtm", 1.0),    # paper nilearn default
    ("OLS_K10_dur2_glover_rtm", 2.0),
    ("OLS_K10_dur3_glover_rtm", 3.0),    # events.tsv true stim duration
]


if __name__ == "__main__":
    for name, dur in CELLS:
        fit_one_cell(name, duration=dur)
