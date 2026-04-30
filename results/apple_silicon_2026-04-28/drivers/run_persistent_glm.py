#!/usr/bin/env python3
"""Persistent GLM (Ernest Lo's terminology) — LSA single-fit instead of LSS per-trial.

Two flavors:
  1. Per-run persistent: single GLM per run with all 70 trial regressors fit jointly
  2. Cross-run persistent: single GLM across all 770 trials of ses-03 ('truly persistent')

LSA: β = (X'X)⁻¹ X'y where X has 70 (or 770) trial regressors + drift/intercept.
Each trial's β = one column of the joint fit. Different from LSS (per-trial refit
with one probe + reference + drift).

Composed with GLMdenoise K=10 (the AUC winner).
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


def build_lsa_design(onsets: np.ndarray, tr: float, n_trs: int,
                     hrf: np.ndarray) -> np.ndarray:
    """LSA design: one regressor per trial + intercept + cosine drift.
    Each trial regressor = delta at onset_TR convolved with HRF.
    Returns (T, n_trials + 2)."""
    n_trials = len(onsets)
    X = np.zeros((n_trs, n_trials + 2), dtype=np.float32)
    for ti, onset in enumerate(onsets):
        boxcar = np.zeros(n_trs, dtype=np.float32)
        onset_tr = int(round(onset / tr))
        if 0 <= onset_tr < n_trs:
            boxcar[onset_tr] = 1.0
        X[:, ti] = np.convolve(boxcar, hrf)[:n_trs]
    X[:, -2] = 1.0
    t = np.arange(n_trs, dtype=np.float32) / max(n_trs - 1, 1)
    X[:, -1] = np.cos(2 * np.pi * t)
    return X


# ============================================================================
# Per-run persistent GLM (LSA): single fit per run with all 70 trial regressors
# ============================================================================

def fit_per_run_lsa(name: str, glmdenoise_K: int = 10):
    print(f"\n=== {name} (per-run LSA, K={glmdenoise_K}) ===")
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

        X = build_lsa_design(onsets, TR, n_trs_run, hrf)              # (T, n_trials+2)
        XtX = X.T @ X + 1e-6 * np.eye(X.shape[1])
        XtX_inv = np.linalg.inv(XtX)
        betas_lsa = (XtX_inv @ X.T @ ts.T).T                          # (V, n_trials+2)

        # Each trial's β = corresponding column
        for trial_i in range(len(onsets)):
            all_betas.append(betas_lsa[:, trial_i].astype(np.float32))
            trial_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))

    betas = np.stack(all_betas, axis=0)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": "LSA_per_run", "glmdenoise_K": glmdenoise_K,
           "bold_source": "rtmotion"}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")


# ============================================================================
# Cross-run persistent GLM: single fit across all 770 trials of ses-03
# ============================================================================

def fit_cross_run_lsa(name: str, glmdenoise_K: int = 10):
    print(f"\n=== {name} (cross-run persistent LSA, K={glmdenoise_K}) ===")
    t0 = time.time()
    flat_brain, rel = load_mask()

    all_ts_per_run = [load_rtmotion(SESSION, r, flat_brain, rel) for r in RUNS]
    noise_per_run = _extract_noise_components_per_run(
        all_ts_per_run, max_K=glmdenoise_K, pool_frac=0.10,
    ) if glmdenoise_K > 0 else None

    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))

    # Build cross-run LSA: concat BOLD across runs; design has all 770 trial cols
    # plus per-run intercept + per-run drift (block-diagonal)
    all_y_concat = []
    all_trial_X = []
    run_drift_blocks = []
    run_intercept_blocks = []
    trial_ids_all = []

    for run_idx, run in enumerate(RUNS):
        ts = all_ts_per_run[run_idx]
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]
            ts = (ts - (ts @ comps) @ comps.T).astype(np.float32)
        events = load_events(SESSION, run)
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]

        # Trial regressors for this run
        run_trial_regs = np.zeros((n_trs_run, len(onsets)), dtype=np.float32)
        for ti, onset in enumerate(onsets):
            boxcar = np.zeros(n_trs_run, dtype=np.float32)
            onset_tr = int(round(onset / TR))
            if 0 <= onset_tr < n_trs_run:
                boxcar[onset_tr] = 1.0
            run_trial_regs[:, ti] = np.convolve(boxcar, hrf)[:n_trs_run]
        all_trial_X.append(run_trial_regs)

        # Per-run intercept + drift
        intercept = np.ones(n_trs_run, dtype=np.float32)
        t = np.arange(n_trs_run, dtype=np.float32) / max(n_trs_run - 1, 1)
        drift = np.cos(2 * np.pi * t)
        run_intercept_blocks.append(intercept)
        run_drift_blocks.append(drift)

        all_y_concat.append(ts.T)                                     # (T_run, V)
        for trial_i in range(len(onsets)):
            trial_ids_all.append(
                str(events.iloc[trial_i].get("image_name", str(trial_i)))
            )

    y_full = np.concatenate(all_y_concat, axis=0).astype(np.float32)  # (T_total, V)

    # Block-diagonal trial design: trials of run k go in cols [start_k : end_k]
    # Plus per-run intercept and drift as block-diagonal columns
    n_trs_per_run = [a.shape[0] for a in all_trial_X]
    n_total_T = sum(n_trs_per_run)
    n_total_trials = sum(a.shape[1] for a in all_trial_X)
    n_runs = len(RUNS)
    n_design_cols = n_total_trials + 2 * n_runs                        # trial cols + per-run intercept + per-run drift
    X_full = np.zeros((n_total_T, n_design_cols), dtype=np.float32)

    t_offset = 0
    trial_offset = 0
    for run_idx in range(n_runs):
        T_r = n_trs_per_run[run_idx]
        n_trials_r = all_trial_X[run_idx].shape[1]
        # Trial cols
        X_full[t_offset:t_offset+T_r,
               trial_offset:trial_offset+n_trials_r] = all_trial_X[run_idx]
        # Per-run intercept (column = total_trials + 2*run_idx)
        X_full[t_offset:t_offset+T_r, n_total_trials + 2*run_idx] = run_intercept_blocks[run_idx]
        # Per-run drift
        X_full[t_offset:t_offset+T_r, n_total_trials + 2*run_idx + 1] = run_drift_blocks[run_idx]
        t_offset += T_r
        trial_offset += n_trials_r

    print(f"  cross-run design: {X_full.shape}, y: {y_full.shape}")
    XtX = X_full.T @ X_full + 1e-3 * np.eye(X_full.shape[1])           # bigger ridge for stability
    XtX_inv = np.linalg.inv(XtX)
    betas_lsa = (XtX_inv @ X_full.T @ y_full).T                        # (V, n_design_cols)

    all_betas = []
    for trial_i in range(n_total_trials):
        all_betas.append(betas_lsa[:, trial_i].astype(np.float32))

    betas = np.stack(all_betas, axis=0)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids_all))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": "LSA_cross_run_persistent",
           "glmdenoise_K": glmdenoise_K, "bold_source": "rtmotion",
           "n_design_cols": int(n_design_cols),
           "n_total_trials": int(n_total_trials),
           "ridge_lambda": 1e-3}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    fit_per_run_lsa("OLS_persistentLSA_K10_glover_rtm",            glmdenoise_K=10)
    fit_per_run_lsa("OLS_persistentLSA_K0_glover_rtm",             glmdenoise_K=0)
    fit_cross_run_lsa("OLS_persistentLSA_crossrun_K10_glover_rtm", glmdenoise_K=10)
    fit_cross_run_lsa("OLS_persistentLSA_crossrun_K0_glover_rtm",  glmdenoise_K=0)
