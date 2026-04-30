#!/usr/bin/env python3
"""3 cells from the broader rt-fMRI NF field that our pipeline didn't include:
  1. Temporal smoothing (Gaussian σ=1.5 TR) before GLM
  2. Band-pass filter (HPF 0.01, LPF 0.15 Hz) before GLM
  3. Frame censoring (drop TRs where FD > 0.5 mm based on MCFLIRT .par)

All composed with GLMdenoise K=10 (the AUC winner from the factorial).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from scipy import signal as sps

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


def temporal_gaussian_smooth(ts: np.ndarray, sigma_tr: float = 1.5) -> np.ndarray:
    """Per-voxel Gaussian smoothing along the time axis."""
    width = int(np.ceil(4 * sigma_tr))                # 4σ kernel
    t = np.arange(-width, width + 1)
    kernel = np.exp(-0.5 * (t / sigma_tr) ** 2)
    kernel /= kernel.sum()
    # Convolve along axis=1 (time)
    out = np.empty_like(ts)
    for v in range(ts.shape[0]):
        out[v] = np.convolve(ts[v], kernel, mode="same")
    return out.astype(np.float32)


def bandpass_filter(ts: np.ndarray, fs: float, hpf: float, lpf: float) -> np.ndarray:
    """Per-voxel zero-phase Butterworth bandpass (4th order)."""
    sos = sps.butter(4, [hpf, lpf], btype="band", fs=fs, output="sos")
    return sps.sosfiltfilt(sos, ts, axis=1).astype(np.float32)


def compute_FD(par: np.ndarray) -> np.ndarray:
    """Framewise displacement from MCFLIRT .par (6 cols: rx, ry, rz, tx, ty, tz).
    Returns (T,) array of FD per TR. Power et al. 2012 formula:
      FD = |Δtx| + |Δty| + |Δtz| + 50·(|Δrx| + |Δry| + |Δrz|)
    """
    diff = np.abs(np.diff(par, axis=0, prepend=par[:1]))
    fd = diff[:, 3:6].sum(axis=1) + 50 * diff[:, 0:3].sum(axis=1)
    return fd.astype(np.float32)


def fit_with_preproc(name: str, glmdenoise_K: int, preproc: str):
    print(f"\n=== {name} (K={glmdenoise_K}, preproc={preproc}) ===")
    t0 = time.time()
    flat_brain, rel = load_mask()

    all_ts_per_run = [load_rtmotion(SESSION, r, flat_brain, rel) for r in RUNS]
    noise_per_run = None
    if glmdenoise_K > 0:
        from prereg_variant_sweep import _extract_noise_components_per_run
        noise_per_run = _extract_noise_components_per_run(
            all_ts_per_run, max_K=glmdenoise_K, pool_frac=0.10,
        )
        print(f"  GLMdenoise K={glmdenoise_K}")

    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))
    all_betas, trial_ids_all = [], []
    n_censored_total = 0
    n_total_TRs = 0

    for run_idx, run in enumerate(RUNS):
        ts = all_ts_per_run[run_idx]
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]
            ts = (ts - (ts @ comps) @ comps.T).astype(np.float32)

        events = load_events(SESSION, run)
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]

        # ---- preproc step -------
        keep_mask = np.ones(n_trs_run, dtype=bool)
        if preproc == "tempsmooth":
            ts = temporal_gaussian_smooth(ts, sigma_tr=1.5)
        elif preproc == "bandpass":
            ts = bandpass_filter(ts, fs=1.0 / TR, hpf=0.01, lpf=0.15)
        elif preproc == "framecensor":
            par_path = P.MC_DIR / f"{SESSION}_run-{run:02d}_motion.par"
            if par_path.exists():
                par = np.loadtxt(par_path)
                fd = compute_FD(par)
                # Censor TRs with FD > 0.5 mm
                keep_mask = fd <= 0.5
                n_censored = int((~keep_mask).sum())
                n_censored_total += n_censored
                n_total_TRs += n_trs_run
                if n_censored > 0:
                    print(f"  run-{run:02d}: censored {n_censored}/{n_trs_run} TRs (FD>0.5mm)")
                # Apply mask to ts (drop columns)
                # ALSO need to recompute n_trs_run effective; but design matrix shape needs to match
                # → simpler: keep ts shape, but in the GLM, weight censored TRs to zero via design matrix mask
        # ----------------------

        for trial_i in range(len(onsets)):
            dm, probe_col = build_design_matrix(onsets, TR, n_trs_run, hrf, trial_i)
            if preproc == "framecensor" and not keep_mask.all():
                # Drop censored rows from dm + ts
                dm_eff = dm[keep_mask]
                ts_eff = ts[:, keep_mask]
            else:
                dm_eff = dm
                ts_eff = ts
            XtX_inv = np.linalg.inv(dm_eff.T @ dm_eff + 1e-6 * np.eye(dm_eff.shape[1]))
            beta_full = (XtX_inv @ dm_eff.T @ ts_eff.T).T
            all_betas.append(beta_full[:, probe_col].astype(np.float32))
            trial_ids_all.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))

    if preproc == "framecensor":
        print(f"  total censored: {n_censored_total}/{n_total_TRs} "
              f"({n_censored_total/max(n_total_TRs,1)*100:.1f}%)")

    betas = np.stack(all_betas, axis=0)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids_all))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "glmdenoise_K": glmdenoise_K, "preproc": preproc,
           "bold_source": "rtmotion", "n_voxels": int(betas.shape[1])}
    if preproc == "tempsmooth":
        cfg["sigma_tr"] = 1.5
    elif preproc == "bandpass":
        cfg["hpf_hz"] = 0.01; cfg["lpf_hz"] = 0.15
    elif preproc == "framecensor":
        cfg["fd_threshold_mm"] = 0.5
        cfg["n_censored_total"] = n_censored_total
        cfg["n_total_TRs"] = n_total_TRs
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")


CELLS = [
    ("OLS_K10_TempSmooth_glover_rtm",   10, "tempsmooth"),
    ("OLS_K10_BandPass_glover_rtm",     10, "bandpass"),
    ("OLS_K10_FrameCensor_glover_rtm",  10, "framecensor"),
]


if __name__ == "__main__":
    for name, K, preproc in CELLS:
        fit_with_preproc(name, glmdenoise_K=K, preproc=preproc)
