#!/usr/bin/env python3
"""(1) Fix CV-per-voxel fracridge to use F-ratio (AUC-aligned objective).
(2) Run soft scalar fracridge in isolation (no GLMdenoise).

Both on plain OLS + Glover, full β kept for proper SVD-fracridge math.
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


def fracridge_per_voxel_full(beta_full: np.ndarray, X: np.ndarray,
                             frac: float) -> np.ndarray:
    """SVD per-voxel fracridge on full β (V, p)."""
    if frac >= 1.0:
        return beta_full.copy()
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    V = Vt.T
    z_ols = beta_full @ V                                        # (Vox, p)
    ols_norm_sq = (z_ols ** 2).sum(axis=1) + 1e-12
    s_sq = s ** 2
    lam_grid = np.logspace(-4, 6, 200)
    weights_sq = (s_sq[None, :] / (s_sq[None, :] + lam_grid[:, None])) ** 2
    ratio = (z_ols ** 2 @ weights_sq.T) / ols_norm_sq[:, None]
    target_sq = frac ** 2
    cross_idx = np.argmax(ratio <= target_sq, axis=1)
    cross_idx = np.clip(cross_idx, 0, len(lam_grid) - 1)
    lam_per_voxel = lam_grid[cross_idx]
    scale = s_sq[None, :] / (s_sq[None, :] + lam_per_voxel[:, None])
    z_ridge = z_ols * scale
    return (z_ridge @ V.T).astype(np.float32)


def collect_full_betas(K: int = 0):
    """Load all runs, optionally apply GLMdenoise K, return per-trial full β."""
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
        from prereg_variant_sweep import _extract_noise_components_per_run
        noise_per_run = _extract_noise_components_per_run(
            timeseries_per_run, max_K=K, pool_frac=0.10,
        )
    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))
    full_betas, trial_ids = [], []
    repr_X = None
    for run_idx, run in enumerate(RUNS):
        ts = timeseries_per_run[run_idx]
        events = events_per_run[run_idx]
        onsets = events["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]
        if noise_per_run is not None:
            comps = noise_per_run[run_idx]
            ts = (ts - (ts @ comps) @ comps.T).astype(np.float32)
        for trial_i in range(len(onsets)):
            dm, _ = build_design_matrix(onsets, TR, n_trs_run, hrf, trial_i)
            XtX_inv = np.linalg.inv(dm.T @ dm + 1e-6 * np.eye(dm.shape[1]))
            beta_full = (XtX_inv @ dm.T @ ts.T).T               # (V, p)
            full_betas.append(beta_full.astype(np.float32))
            trial_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))
            if repr_X is None:
                repr_X = dm.astype(np.float32)
    return np.stack(full_betas, axis=0), trial_ids, repr_X    # (N, V, p), list, (T, p)


def cv_per_voxel_F_ratio(full_betas: np.ndarray, trial_ids: list,
                          repr_X: np.ndarray) -> np.ndarray:
    """Per-voxel CV-tuned frac via F-ratio: between-image-variance / within-image-variance.

    Higher F = more discriminative for AUC. Per-voxel sweep over frac grid.
    """
    N, V, p = full_betas.shape
    probe_col = 0  # build_design_matrix puts probe first

    img_groups: dict[str, list[int]] = {}
    for i, t in enumerate(trial_ids):
        if t.startswith("all_stimuli/special515/"):
            img_groups.setdefault(t, []).append(i)
    img_groups = {k: v for k, v in img_groups.items() if len(v) >= 2}
    print(f"  CV groups: {len(img_groups)} images with ≥2 reps")

    frac_grid = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                         dtype=np.float32)
    f_ratio = np.zeros((len(frac_grid), V), dtype=np.float32)

    for fi, f in enumerate(frac_grid):
        # Apply per-trial fracridge to all special515 trials
        spec_idxs = sorted(set(idx for grp in img_groups.values() for idx in grp))
        spec_betas = np.zeros((len(spec_idxs), V), dtype=np.float32)
        for k, i in enumerate(spec_idxs):
            if f >= 1.0:
                spec_betas[k] = full_betas[i, :, probe_col]
            else:
                bf = fracridge_per_voxel_full(full_betas[i], repr_X, frac=float(f))
                spec_betas[k] = bf[:, probe_col]
        idx_to_pos = {i: k for k, i in enumerate(spec_idxs)}

        # Per-voxel F = between / within
        per_image_means = []
        per_image_vars = []
        for img, idxs in img_groups.items():
            pos = [idx_to_pos[i] for i in idxs]
            stack = spec_betas[pos]
            per_image_means.append(stack.mean(axis=0))
            per_image_vars.append(stack.var(axis=0))
        means_stack = np.stack(per_image_means)               # (n_imgs, V)
        vars_stack = np.stack(per_image_vars)                  # (n_imgs, V)
        between = means_stack.var(axis=0) + 1e-12             # variance of means
        within = vars_stack.mean(axis=0) + 1e-12              # mean of within-image variances
        f_ratio[fi] = between / within

    best_idx = f_ratio.argmax(axis=0)
    best_frac = frac_grid[best_idx].astype(np.float32)
    print(f"  CV per-voxel f*: mean={best_frac.mean():.3f}, "
          f"min={best_frac.min():.2f}, max={best_frac.max():.2f}")
    return best_frac


# =============================================================================
# Cell 1: CV-fixed fracridge with F-ratio objective
# =============================================================================

def run_cv_fixed_with_K(K: int, name: str):
    print(f"\n=== {name} (K={K}, F-ratio CV fracridge) ===")
    t0 = time.time()
    full_betas, trial_ids, repr_X = collect_full_betas(K=K)
    print(f"  full β: {full_betas.shape}")

    best_frac = cv_per_voxel_F_ratio(full_betas, trial_ids, repr_X)

    # Apply per-voxel f* to all trials
    N, V, p = full_betas.shape
    probe_col = 0
    out = np.zeros((N, V), dtype=np.float32)
    frac_grid_unique = np.unique(best_frac)
    for f in frac_grid_unique:
        mask = best_frac == f
        if not mask.any(): continue
        for i in range(N):
            if f >= 1.0:
                out[i, mask] = full_betas[i, mask, probe_col]
            else:
                bf = fracridge_per_voxel_full(full_betas[i, mask], repr_X, frac=float(f))
                out[i, mask] = bf[:, probe_col]

    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", out)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    np.save(P.OUT_DIR / f"{name}_{SESSION}_frac_per_voxel.npy", best_frac)
    cfg = {"cell": name, "GLMdenoise_K": K, "fracridge": "per-voxel CV (F-ratio)",
           "frac_mean": float(best_frac.mean()),
           "frac_min": float(best_frac.min()),
           "frac_max": float(best_frac.max())}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {out.shape}  ({time.time()-t0:.1f}s)")


# =============================================================================
# Cell 2: Soft scalar fracridge by itself (no denoise, no per-voxel SVD)
# =============================================================================

def run_soft_scalar_only():
    name = "OLS_glover_rtm_softFrac_only"
    print(f"\n=== {name} (no denoise, soft scalar fracridge from cells 7/8) ===")
    t0 = time.time()
    full_betas, trial_ids, _ = collect_full_betas(K=0)
    probe_col = 0
    betas_probe = full_betas[..., probe_col]                   # (N, V)
    # Apply the soft formula trial-by-trial (matching prereg_variant_sweep:343-346)
    out = np.zeros_like(betas_probe)
    for i in range(betas_probe.shape[0]):
        beta = betas_probe[i]
        ols_norm = float(np.linalg.norm(beta) + 1e-12)
        out[i] = beta * 0.5 * (1.0 + ols_norm / (ols_norm + 1e-3))
    print(f"  scalar applied: mean(|out|/|in|) = "
          f"{(np.abs(out).mean() / (np.abs(betas_probe).mean() + 1e-12)):.4f} "
          f"(=0.5×(1+1)=1.0 expected for typical |β|>>1e-3, ie no-op)")

    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", out)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    cfg = {"cell": name, "GLMdenoise_K": 0,
           "fracridge": "soft scalar 0.5*(1 + ||β||/(||β||+1e-3))"}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {out.shape}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    run_cv_fixed_with_K(0, "OLS_glover_rtm_denoiseK0_fracR_CV_Fratio")
    run_cv_fixed_with_K(5, "OLS_glover_rtm_denoiseK5_fracR_CV_Fratio")
    run_cv_fixed_with_K(10, "OLS_glover_rtm_denoiseK10_fracR_CV_Fratio")
    run_soft_scalar_only()
