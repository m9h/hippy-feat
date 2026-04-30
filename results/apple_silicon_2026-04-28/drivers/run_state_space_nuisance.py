#!/usr/bin/env python3
"""State-space nuisance model — the architecturally-correct LDS variant.

Per voxel, per run:
  1. Fit LSA OLS: get rough estimate of β_per_trial across all trials
  2. Subtract predicted-signal from BOLD → residual r_v(t)
  3. Fit AR(1) state-space on r_v: state = AR-correlated nuisance
  4. Kalman smoother → x̂_v(t) = smoothed nuisance estimate
  5. Cleaned BOLD: y'_v(t) = y_v(t) - x̂_v(t)
  6. Refit per-trial β via LSS on cleaned BOLD

Crucial architectural difference from the previous LDS cell:
  - Old: smooth β across trials (homogenized neighbors → killed AUC)
  - New: smooth NUISANCE across TRs, then refit β per trial on cleaned BOLD
         (per-trial discriminability preserved)

Cells:
  OLS_SSnuisance_glover_rtm                 plain OLS LSS + state-space nuisance
  OLS_denoiseK10_SSnuisance_glover_rtm      GLMdenoise K=10 + SS nuisance
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
from prereg_variant_sweep import load_mask, load_rtmotion, load_events
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


def build_lsa_design(onsets: np.ndarray, tr: float, n_trs: int,
                     hrf: np.ndarray) -> np.ndarray:
    """LSA: one regressor per trial, plus intercept + cosine drift.
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
    t_axis = np.arange(n_trs, dtype=np.float32) / max(n_trs - 1, 1)
    X[:, -1] = np.cos(2 * np.pi * t_axis)
    return X


def kalman_smoother_vectorized(y: np.ndarray, a: np.ndarray,
                                q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """RTS smoother on a univariate AR(1) state-space.

    State:    x_t = a · x_{t-1} + ε_w,  ε_w ~ N(0, q)
    Observed: y_t = x_t + ε_v,           ε_v ~ N(0, r)

    All quantities vectorized across voxels.
    Args:
      y: (V, T)  observations
      a: (V,)    per-voxel AR(1) coefficient
      q: (V,)    per-voxel state noise variance
      r: (V,)    per-voxel observation noise variance
    Returns:
      smoothed: (V, T) RTS-smoothed posterior mean of x_t
    """
    V, T = y.shape
    x_pred = np.zeros((T + 1, V), dtype=np.float64)
    P_pred = np.full((T + 1, V), 1e6, dtype=np.float64)
    x_filt = np.zeros((T, V), dtype=np.float64)
    P_filt = np.zeros((T, V), dtype=np.float64)

    for t in range(T):
        x_pred[t] = a * (x_filt[t - 1] if t > 0 else 0.0)
        P_pred[t] = (a ** 2) * (P_filt[t - 1] if t > 0 else 1e6) + q
        K = P_pred[t] / (P_pred[t] + r)
        x_filt[t] = x_pred[t] + K * (y[:, t] - x_pred[t])
        P_filt[t] = (1 - K) * P_pred[t]

    x_smooth = np.zeros_like(x_filt)
    x_smooth[-1] = x_filt[-1]
    for t in range(T - 2, -1, -1):
        G = P_filt[t] * a / np.maximum(P_pred[t + 1], 1e-12)
        x_smooth[t] = x_filt[t] + G * (x_smooth[t + 1] - x_pred[t + 1])

    return x_smooth.T.astype(np.float32)                          # (V, T)


def fit_one_cell(name: str, glmdenoise_K: int = 0):
    print(f"\n=== {name} (K={glmdenoise_K}, state-space nuisance) ===")
    t0 = time.time()
    flat_brain, rel = load_mask()

    all_ts_per_run = [load_rtmotion(SESSION, r, flat_brain, rel) for r in RUNS]
    noise_per_run = None
    if glmdenoise_K > 0:
        from prereg_variant_sweep import _extract_noise_components_per_run
        noise_per_run = _extract_noise_components_per_run(
            all_ts_per_run, max_K=glmdenoise_K, pool_frac=0.10,
        )
        print(f"  GLMdenoise K={glmdenoise_K} components per run")

    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))
    all_betas, trial_ids_all = [], []

    for run_idx, run in enumerate(RUNS):
        ts = all_ts_per_run[run_idx]
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]
            ts = (ts - (ts @ comps) @ comps.T).astype(np.float32)
        events = load_events(SESSION, run)
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]

        # Step 1: LSA OLS for initial β estimate
        X_lsa = build_lsa_design(onsets, TR, n_trs_run, hrf)
        XtX_inv = np.linalg.inv(X_lsa.T @ X_lsa + 1e-6 * np.eye(X_lsa.shape[1]))
        beta_lsa = (XtX_inv @ X_lsa.T @ ts.T).T                  # (V, n_trials+2)
        pred = beta_lsa @ X_lsa.T                                # (V, T)
        resid = ts - pred                                         # (V, T)

        # Step 2: per-voxel AR(1) parameters from residual
        r1 = (resid[:, 1:] * resid[:, :-1]).sum(axis=1)
        r0 = (resid ** 2).sum(axis=1)
        a = np.clip(r1 / (r0 + 1e-12), -0.99, 0.99).astype(np.float64)
        # State noise variance (residual variance minus AR1-explained portion)
        resid_pred_ar = a[:, None] * resid[:, :-1]
        innov = resid[:, 1:] - resid_pred_ar
        q = innov.var(axis=1).astype(np.float64) + 1e-6
        # Observation noise: small relative to state — use a fraction of total residual var
        r = q.copy() * 0.5                                        # tunable

        # Step 3: smoother → smoothed nuisance estimate
        x_smooth = kalman_smoother_vectorized(resid.astype(np.float64), a, q, r)

        # Step 4: cleaned BOLD = original - smoothed nuisance
        ts_clean = (ts - x_smooth).astype(np.float32)

        # Step 5: refit per-trial β via LSS on cleaned BOLD
        for trial_i in range(len(onsets)):
            dm, probe_col = build_design_matrix(onsets, TR, n_trs_run, hrf, trial_i)
            XtX_inv2 = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
            beta_full = (XtX_inv2 @ dm.T @ ts_clean.T).T          # (V, p)
            all_betas.append(beta_full[:, probe_col].astype(np.float32))
            trial_ids_all.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))

        print(f"  run-{run:02d}: a̅={a.mean():+.3f}, q̅={q.mean():.3f}, "
              f"resid_var̅={(resid**2).mean(axis=1).mean():.3f}")

    betas = np.stack(all_betas, axis=0)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids_all))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": "OLS_LSS_on_state_space_nuisance_cleaned_BOLD",
           "glmdenoise_K": glmdenoise_K, "bold_source": "rtmotion",
           "smoother_kind": "AR(1) RTS smoother on LSA residual (per-voxel)"}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    fit_one_cell("OLS_SSnuisance_glover_rtm",            glmdenoise_K=0)
    fit_one_cell("OLS_denoiseK10_SSnuisance_glover_rtm", glmdenoise_K=10)
