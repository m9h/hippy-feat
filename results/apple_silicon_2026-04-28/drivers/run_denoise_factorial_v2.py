#!/usr/bin/env python3
"""8-cell GLMdenoise × fracridge factorial — VERSION 2 (proper fracridge).

Uses plain OLS (no AR(1)) for the GLM step so we can keep the full 4-component
β vector per (trial, voxel). Fracridge then applies per-voxel SVD-based
shrinkage on those full vectors before extracting the probe column.

Cells:
  OLS_glover_rtm_denoiseK0_fracR0p3       K=0,  frac=0.3 (uniform)
  OLS_glover_rtm_denoiseK0_fracR0p5       K=0,  frac=0.5
  OLS_glover_rtm_denoiseK0_fracR1p0       K=0,  frac=1.0 (= plain OLS, no shrinkage)
  OLS_glover_rtm_denoiseK5_fracR0p3       K=5,  frac=0.3
  OLS_glover_rtm_denoiseK5_fracR0p7       K=5,  frac=0.7
  OLS_glover_rtm_denoiseK10_fracR0p5      K=10, frac=0.5
  OLS_glover_rtm_denoiseK15_fracR0p5      K=15, frac=0.5
  OLS_glover_rtm_denoiseK5_fracR_CV       K=5,  per-voxel CV-tuned frac
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


def ols_full_beta(timeseries: np.ndarray, dm: np.ndarray) -> np.ndarray:
    """Plain OLS, returns full β: (V, p)."""
    XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
    pinv = XtX_inv @ dm.T                                       # (p, T)
    return (pinv @ timeseries.T).T                              # (V, p)


def fracridge_per_voxel(beta_full: np.ndarray, X: np.ndarray,
                        frac: float) -> np.ndarray:
    """SVD-based per-voxel fracridge (Rokem & Kay 2020).

    For each voxel v: solve λ_v such that ||β_ridge(λ_v)|| / ||β_OLS_v|| = frac
    via the SVD of X. Returns shrunk β_ridge of shape (V, p).
    """
    if frac >= 1.0:
        return beta_full.copy()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    V_mat = Vt.T                                                 # (p, p)
    # z = V' β_OLS  (per-voxel; shape V x p)
    z_ols = beta_full @ V_mat                                    # (V_voxels, p)
    ols_norm_sq = (z_ols ** 2).sum(axis=1) + 1e-12               # (V_voxels,)

    # f²(λ) = sum_p (s_p²/(s_p²+λ))² z_ols_p² / ||z_ols||²
    s_sq = s ** 2
    lam_grid = np.logspace(-4, 6, 200)
    weights_sq = (s_sq[None, :] / (s_sq[None, :] + lam_grid[:, None])) ** 2  # (G, p)
    ratio = (z_ols ** 2 @ weights_sq.T) / ols_norm_sq[:, None]   # (V_voxels, G)
    target_sq = frac ** 2
    cross_idx = np.argmax(ratio <= target_sq, axis=1)
    cross_idx = np.clip(cross_idx, 0, len(lam_grid) - 1)
    lam_per_voxel = lam_grid[cross_idx]                          # (V_voxels,)

    # β_ridge = V @ diag(s²/(s²+λ_v)) @ V' β_OLS
    scale = s_sq[None, :] / (s_sq[None, :] + lam_per_voxel[:, None])  # (V_voxels, p)
    z_ridge = z_ols * scale
    return (z_ridge @ V_mat.T).astype(np.float32)                # (V_voxels, p)


def fit_one_cell(name: str, K: int, frac, do_cv_frac: bool = False,
                 representative_X: np.ndarray | None = None):
    print(f"\n=== {name} (K={K}, frac={frac}, CV={do_cv_frac}) ===")
    t0 = time.time()
    flat_brain, rel = load_mask()

    # Pre-load all runs
    timeseries_per_run = []
    events_per_run = []
    for run in RUNS:
        ts = load_rtmotion(SESSION, run, flat_brain, rel)
        ev = load_events(SESSION, run)
        timeseries_per_run.append(ts)
        events_per_run.append(ev)

    # GLMdenoise (Stage 2)
    noise_per_run = None
    if K > 0:
        noise_per_run = _extract_noise_components_per_run(
            timeseries_per_run, max_K=K, pool_frac=0.10,
        )
        print(f"  GLMdenoise extracted K={K} components per run")

    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))

    # Pass 1: collect per-trial OLS full β (probe + ref + intercept + drift)
    all_full_betas = []
    trial_ids = []
    repr_X = None

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
            beta_full = ols_full_beta(ts, dm)                    # (V, p)
            all_full_betas.append(beta_full)
            trial_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))
            if repr_X is None:
                repr_X = dm.astype(np.float32)

    full_betas = np.stack(all_full_betas, axis=0)                # (N, V, p)
    print(f"  full β: {full_betas.shape}")
    probe_col = 0  # build_design_matrix puts probe first

    # Apply fracridge
    if do_cv_frac:
        # Per-voxel CV: pick frac from grid that maximizes same-image SNR
        frac_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        # Group special515 trials by image
        img_groups: dict[str, list[int]] = {}
        for i, t in enumerate(trial_ids):
            if t.startswith("all_stimuli/special515/"):
                img_groups.setdefault(t, []).append(i)
        img_groups = {k: v for k, v in img_groups.items() if len(v) >= 2}

        N, V, p = full_betas.shape
        best_frac_per_voxel = np.zeros(V, dtype=np.float32)
        # Score each frac on a per-voxel signal-to-cross-rep-noise ratio
        scores = np.zeros((len(frac_grid), V), dtype=np.float32)
        for fi, f in enumerate(frac_grid):
            if f >= 1.0:
                betas_probe_f = full_betas[..., probe_col]
            else:
                # Apply fracridge per-trial (the rep-X within special515 trials varies, but minor)
                betas_probe_f_list = []
                for i in range(N):
                    bf = fracridge_per_voxel(full_betas[i], repr_X, frac=float(f))
                    betas_probe_f_list.append(bf[:, probe_col])
                betas_probe_f = np.stack(betas_probe_f_list, axis=0)
            # SNR proxy: mean |β| / std β across same-image trials
            snr_v = np.zeros(V, dtype=np.float32)
            n_imgs = 0
            for img, idxs in img_groups.items():
                stack = betas_probe_f[idxs]
                mu = stack.mean(axis=0)
                sd = stack.std(axis=0) + 1e-8
                snr_v += np.abs(mu) / sd
                n_imgs += 1
            scores[fi] = snr_v / max(n_imgs, 1)
        best_idx = scores.argmax(axis=0)
        best_frac_per_voxel = frac_grid[best_idx].astype(np.float32)
        print(f"  CV frac per voxel: mean={best_frac_per_voxel.mean():.3f}, "
              f"min={best_frac_per_voxel.min():.2f}, max={best_frac_per_voxel.max():.2f}")
        np.save(P.OUT_DIR / f"{name}_{SESSION}_frac_per_voxel.npy", best_frac_per_voxel)

        # Apply best frac per voxel to all trials (group voxels by frac value)
        N, V, p = full_betas.shape
        betas_probe_final = np.zeros((N, V), dtype=np.float32)
        for f in frac_grid:
            mask = best_frac_per_voxel == f
            if not mask.any(): continue
            for i in range(N):
                if f >= 1.0:
                    betas_probe_final[i, mask] = full_betas[i, mask, probe_col]
                else:
                    bf = fracridge_per_voxel(full_betas[i, mask], repr_X, frac=float(f))
                    betas_probe_final[i, mask] = bf[:, probe_col]
    else:
        # Fixed frac across all voxels
        N, V, p = full_betas.shape
        betas_probe_final = np.zeros((N, V), dtype=np.float32)
        for i in range(N):
            if frac >= 1.0:
                betas_probe_final[i] = full_betas[i, :, probe_col]
            else:
                bf = fracridge_per_voxel(full_betas[i], repr_X, frac=float(frac))
                betas_probe_final[i] = bf[:, probe_col]

    # Save
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas_probe_final)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": "OLS", "bold_source": "rtmotion",
           "GLMdenoise_K": K,
           "fracridge": ("per-voxel CV" if do_cv_frac else float(frac))}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas_probe_final.shape}  ({time.time()-t0:.1f}s)")


CELLS = [
    ("OLS_glover_rtm_denoiseK0_fracR0p3",  0,   0.3,  False),
    ("OLS_glover_rtm_denoiseK0_fracR0p5",  0,   0.5,  False),
    ("OLS_glover_rtm_denoiseK0_fracR1p0",  0,   1.0,  False),
    ("OLS_glover_rtm_denoiseK5_fracR0p3",  5,   0.3,  False),
    ("OLS_glover_rtm_denoiseK5_fracR0p7",  5,   0.7,  False),
    ("OLS_glover_rtm_denoiseK10_fracR0p5", 10,  0.5,  False),
    ("OLS_glover_rtm_denoiseK15_fracR0p5", 15,  0.5,  False),
    ("OLS_glover_rtm_denoiseK5_fracR_CV",  5,   None, True),
]

if __name__ == "__main__":
    for name, K, frac, do_cv in CELLS:
        try:
            fit_one_cell(name, K=K, frac=frac, do_cv_frac=do_cv)
        except Exception as e:
            print(f"  FAILED {name}: {e}")
            import traceback; traceback.print_exc()
