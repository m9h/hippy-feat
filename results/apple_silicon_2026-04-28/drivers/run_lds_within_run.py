#!/usr/bin/env python3
"""LDS smoother on per-trial β within each run.

For each voxel v independently:
  State:    x_t = β_v at trial t (V independent univariate LDS)
  Dynamics: x_t = a · x_{t-1} + ε_w     (AR(1) on β across trials)
  Observed: y_t = x_t + ε_v              (LSS per-trial β with its variance)
  Smoother: forward Kalman filter + RTS backward pass → smoothed β_t

Estimates `a` per voxel from the LSS β sequence's empirical lag-1 autocorr.
Per-trial observation noise comes from the LSS fit's per-voxel β variance.

Each run is fit independently; trial order within a run is the temporal order.
This is the within-run version of the state-space approach — testing whether
cross-trial smoothing captures evidence per-trial LSS misses.

Two cells:
  OLS_LDS_glover_rtm           — pure OLS LSS β + LDS smoother
  AR1freq_LDS_glover_rtm       — AR(1) prewhitened LSS β + LDS smoother (stacked)
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
from prereg_variant_sweep import load_mask, load_rtmotion, load_events, _glm_jax
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


def kalman_smoother_per_voxel(beta_seq: np.ndarray, var_seq: np.ndarray,
                               a: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Vectorized univariate Kalman smoother across voxels.

    Each voxel is an independent AR(1)-state LDS. Returns RTS-smoothed
    posterior mean of β_t for each (trial, voxel).

    Args:
      beta_seq: (T, V) per-trial β sequence (observations)
      var_seq:  (T, V) per-trial observation variance (from LSS fit)
      a:        (V,) per-voxel AR(1) coefficient on β
      q:        (V,) per-voxel state-noise variance

    Returns:
      smoothed: (T, V) RTS-smoothed posterior mean
    """
    T, V = beta_seq.shape
    # Forward filter
    x_pred = np.zeros((T + 1, V), dtype=np.float64)
    P_pred = np.full((T + 1, V), 1e6, dtype=np.float64)        # uninformative initial
    x_filt = np.zeros((T, V), dtype=np.float64)
    P_filt = np.zeros((T, V), dtype=np.float64)

    for t in range(T):
        # Predict
        x_pred[t] = a * (x_filt[t - 1] if t > 0 else 0.0)
        P_pred[t] = (a ** 2) * (P_filt[t - 1] if t > 0 else 1e6) + q
        # Update with observation y_t
        K = P_pred[t] / (P_pred[t] + var_seq[t])               # Kalman gain
        x_filt[t] = x_pred[t] + K * (beta_seq[t] - x_pred[t])
        P_filt[t] = (1 - K) * P_pred[t]

    # Backward RTS smoother
    x_smooth = np.zeros_like(x_filt)
    P_smooth = np.zeros_like(P_filt)
    x_smooth[-1] = x_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for t in range(T - 2, -1, -1):
        # Smoother gain G = P_filt[t] * a / P_pred[t+1]
        G = P_filt[t] * a / np.maximum(P_pred[t + 1], 1e-12)
        x_smooth[t] = x_filt[t] + G * (x_smooth[t + 1] - x_pred[t + 1])
        P_smooth[t] = P_filt[t] + (G ** 2) * (P_smooth[t + 1] - P_pred[t + 1])

    return x_smooth.astype(np.float32)


def estimate_per_voxel_AR1(beta_seq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-voxel AR(1) coefficient + state-noise variance estimate.

    a_v = corr(β_v[1:], β_v[:-1])
    q_v = var(β_v[1:] - a_v * β_v[:-1])
    """
    T, V = beta_seq.shape
    if T < 2:
        return np.zeros(V, dtype=np.float64), np.ones(V, dtype=np.float64)
    x_curr = beta_seq[1:]                                         # (T-1, V)
    x_prev = beta_seq[:-1]                                        # (T-1, V)
    # Pearson per voxel
    num = ((x_curr - x_curr.mean(axis=0)) * (x_prev - x_prev.mean(axis=0))).sum(axis=0)
    den = (np.sqrt(((x_curr - x_curr.mean(axis=0)) ** 2).sum(axis=0)) *
           np.sqrt(((x_prev - x_prev.mean(axis=0)) ** 2).sum(axis=0)) + 1e-12)
    a = (num / den).clip(-0.99, 0.99).astype(np.float64)
    resid = x_curr - a * x_prev
    q = resid.var(axis=0).astype(np.float64)
    q = np.maximum(q, 1e-6)
    return a, q


def fit_one_cell(name: str, mode: str = "ols", glmdenoise_K: int = 0):
    print(f"\n=== {name} (mode={mode}, K={glmdenoise_K}, LDS smoother) ===")
    t0 = time.time()
    flat_brain, rel = load_mask()

    # Pre-load all runs so we can apply GLMdenoise K
    all_ts_per_run = [load_rtmotion(SESSION, r, flat_brain, rel) for r in RUNS]
    noise_per_run = None
    if glmdenoise_K > 0:
        from prereg_variant_sweep import _extract_noise_components_per_run
        noise_per_run = _extract_noise_components_per_run(
            all_ts_per_run, max_K=glmdenoise_K, pool_frac=0.10,
        )
        print(f"  GLMdenoise K={glmdenoise_K} components per run")

    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))
    all_smoothed_betas = []
    trial_ids_all = []

    for run_idx, run in enumerate(RUNS):
        ts = all_ts_per_run[run_idx]
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]
            ts = (ts - (ts @ comps) @ comps.T).astype(np.float32)
        events = load_events(SESSION, run)
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]

        # Step 1: per-trial LSS β (mode=ols or ar1_freq)
        beta_seq = np.zeros((len(onsets), ts.shape[0]), dtype=np.float32)
        var_seq = np.zeros((len(onsets), ts.shape[0]), dtype=np.float32)
        for trial_i in range(len(onsets)):
            if mode == "ols":
                dm, probe_col = build_design_matrix(onsets, TR, n_trs_run, hrf, trial_i)
                XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
                beta_full = (XtX_inv @ dm.T @ ts.T).T
                beta_seq[trial_i] = beta_full[:, probe_col]
                pred = beta_full @ dm.T
                rss = ((ts - pred) ** 2).sum(axis=1)
                sigma2 = rss / max(n_trs_run - dm.shape[1], 1)
                var_seq[trial_i] = sigma2 * XtX_inv[probe_col, probe_col]
            else:
                beta, var = _glm_jax(ts, onsets, trial_i, TR, n_trs_run,
                                      mode=mode, max_trs=200)
                beta_seq[trial_i] = np.asarray(beta, dtype=np.float32)
                var_seq[trial_i] = np.maximum(np.asarray(var, dtype=np.float32), 1e-6)

        # Step 2: estimate per-voxel AR(1) on the trial-β sequence
        a, q = estimate_per_voxel_AR1(beta_seq)

        # Step 3: Kalman smoother
        smoothed = kalman_smoother_per_voxel(
            beta_seq.astype(np.float64),
            var_seq.astype(np.float64),
            a, q,
        )
        all_smoothed_betas.append(smoothed)
        trial_ids_all.extend(
            str(events.iloc[ti].get("image_name", str(ti))) for ti in range(len(onsets))
        )
        print(f"  run-{run:02d}: {len(onsets)} trials, "
              f"a̅={a.mean():+.3f}, q̅={q.mean():.3f}")

    betas_out = np.concatenate(all_smoothed_betas, axis=0).astype(np.float32)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas_out)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids_all))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": f"{mode}+within-run-LDS-smoother",
           "bold_source": "rtmotion", "n_voxels": int(betas_out.shape[1]),
           "smoother_kind": "RTS univariate per voxel"}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas_out.shape}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    fit_one_cell("OLS_LDS_glover_rtm",                       mode="ols",      glmdenoise_K=0)
    fit_one_cell("AR1freq_LDS_glover_rtm",                   mode="ar1_freq", glmdenoise_K=0)
    fit_one_cell("OLS_denoiseK10_LDS_glover_rtm",            mode="ols",      glmdenoise_K=10)
    fit_one_cell("AR1freq_denoiseK10_LDS_glover_rtm",        mode="ar1_freq", glmdenoise_K=10)
