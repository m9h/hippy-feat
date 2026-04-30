#!/usr/bin/env python3
"""Three follow-up cells in one driver:

1. HybridOnline_AR1freq_glover_rtm — cell 17 with session-wide ρ̂ (already
   wired by DGX in 3e63344, mode="ar1_session_rho"). Run via prereg sweep.

2. SameImagePrior_VariantG_glover_rtm — for each trial, use past trials of
   the same image as a per-trial empirical-Bayes prior. Causally accumulating
   evidence per image.

3. VariantB_FLOBS_fitted_glover_rtm — fit per-voxel FLOBS basis weights from
   ses-01 fmriprep BOLD as a training set, then use those weights instead of
   the default 1/3 each.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib
import jax.numpy as jnp

warnings.filterwarnings("ignore")

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import prereg_variant_sweep as P
import rt_glm_variants as V

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
P.HRF_INDICES_PATH = str(LOCAL / "rt3t" / "data" / "avg_hrfs_s1_s2_full.npy")
P.HRF_LIB_PATH = str(LOCAL / "rt3t" / "data" / "getcanonicalhrflibrary.tsv")

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5

flat_brain = (nib.load(P.BRAIN_MASK).get_fdata() > 0).flatten()
rel = np.load(P.RELMASK)
N_VOXELS = int(rel.sum())  # 2792


# ============================================================================
# 1. HybridOnline (cell 17) — already wired, just call existing run_glm_cell
# ============================================================================

def run_hybrid_online():
    name = "HybridOnline_AR1freq_glover_rtm"
    print(f"\n=== {name} (mode=ar1_session_rho) ===")
    t0 = time.time()
    betas, ids = P.run_glm_cell(
        name, mode="ar1_session_rho", bold_source="rtmotion",
        hrf_strategy="glover",
        session=SESSION, runs=RUNS,
        prior_mean=None, prior_var=None, denoise=None,
    )
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(ids))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": "ar1_session_rho", "bold_source": "rtmotion",
           "n_voxels": int(betas.shape[1])}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")


# ============================================================================
# 2. Same-image-prior — causal Bayesian update per image
# ============================================================================

def run_same_image_prior():
    name = "SameImagePrior_VariantG_glover_rtm"
    print(f"\n=== {name} ===")
    t0 = time.time()

    flat_b, rel_ = P.load_mask()
    timeseries_per_run = []
    events_per_run = []
    for run in RUNS:
        ts = P.load_rtmotion(SESSION, run, flat_b, rel_)
        ev = P.load_events(SESSION, run)
        timeseries_per_run.append(ts)
        events_per_run.append(ev)

    # Walk trials in temporal order, maintain {image: list[β]}
    image_history: dict[str, list[np.ndarray]] = {}
    all_betas = []
    trial_ids = []

    for run_idx, run in enumerate(RUNS):
        ts = timeseries_per_run[run_idx]
        ev = events_per_run[run_idx]
        onsets = ev["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]

        for trial_i in range(len(onsets)):
            img = str(ev.iloc[trial_i].get("image_name", str(trial_i)))
            past = image_history.get(img, [])
            if len(past) >= 2:
                # Have ≥2 past observations of this image → empirical prior
                stack = np.stack(past, axis=0)
                prior_mean = stack.mean(axis=0).astype(np.float32)
                prior_var = (stack.var(axis=0).astype(np.float32) + 1.0)  # add 1.0 regularization
                mode = "variant_g_prior"
            elif len(past) == 1:
                # 1 past observation — use it as mean with high variance
                prior_mean = past[0].astype(np.float32)
                prior_var = (np.ones(N_VOXELS, dtype=np.float32) * 100.0)
                mode = "variant_g_prior"
            else:
                # No past — uninformative; fall back to plain Variant G
                prior_mean = None
                prior_var = None
                mode = "variant_g"

            beta, _v = P._glm_jax(
                ts, onsets, trial_i, TR, n_trs_run, mode=mode,
                prior_mean=prior_mean, prior_var=prior_var,
            )
            beta_np = np.asarray(beta, dtype=np.float32)
            all_betas.append(beta_np)
            trial_ids.append(img)
            image_history.setdefault(img, []).append(beta_np)

    betas = np.stack(all_betas, axis=0)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": "variant_g_prior (per-image causal Bayes)",
           "bold_source": "rtmotion", "n_voxels": int(betas.shape[1]),
           "n_unique_images": len(image_history)}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")
    print(f"  {len(image_history)} unique images; "
          f"max repeats: {max(len(v) for v in image_history.values())}")


# ============================================================================
# 3. Variant B with per-voxel FLOBS weights fit from ses-01 fmriprep BOLD
# ============================================================================

def fit_flobs_weights_ses01() -> np.ndarray:
    """Fit per-voxel FLOBS basis weights using ses-01 fmriprep BOLD.

    For each ses-01 run: build FLOBS LSA (all trials, 3 bases each) design,
    fit OLS, get per-voxel βs for each (trial, basis). Average across trials
    per voxel → (n_voxels, 3) preferred basis weighting. Normalize.
    """
    print("  fitting FLOBS weights from ses-01 fmriprep BOLD...")
    t0 = time.time()
    flobs_path = "/Users/mhough/fsl/data/feat5/default_flobs.flobs/hrfbasisfns.txt"
    flobs_basis = V.load_flobs_basis(flobs_path)            # (559, 3)
    print(f"  FLOBS basis: {flobs_basis.shape}")

    flat_b, rel_ = P.load_mask()
    accumulated_weights = np.zeros((N_VOXELS, 3), dtype=np.float64)
    n_trials_total = 0

    for run in RUNS:
        ts = P.load_fmriprep("ses-01", run, flat_b, rel_)   # (V, T)
        ev = P.load_events("ses-01", run)
        onsets = ev["onset_rel"].values.astype(np.float32)
        n_trs_run = ts.shape[1]

        # Build FLOBS LSA: each trial gets 3 basis regressors → 3*n_trials cols
        # Plus intercept and cosine drift
        all_regressors = []
        for trial_i in range(len(onsets)):
            dm_trial, _ = V.build_design_matrix_flobs(
                onsets, TR, n_trs_run, flobs_basis, trial_i,
            )
            # dm_trial has 3 probe basis cols + reference + drift; we want only the 3 probes
            all_regressors.append(dm_trial[:, :3])           # (T, 3)
        # Stack: (T, 3 * n_trials)
        dm_lsa = np.concatenate(all_regressors, axis=1).astype(np.float32)

        # Add intercept + cosine drift
        intercept = np.ones((n_trs_run, 1), dtype=np.float32)
        t_axis = np.arange(n_trs_run, dtype=np.float32) / max(n_trs_run - 1, 1)
        drift = np.cos(2 * np.pi * t_axis).reshape(-1, 1)
        dm_lsa = np.concatenate([dm_lsa, intercept, drift], axis=1)

        # Fit OLS: β = (X'X)^-1 X' Y where Y = ts.T (T, V)
        XtX = dm_lsa.T @ dm_lsa + 1e-3 * np.eye(dm_lsa.shape[1])
        XtX_inv = np.linalg.inv(XtX)
        Xty = dm_lsa.T @ ts.T                                # (3*nt+2, V)
        betas_full = XtX_inv @ Xty                           # (3*nt+2, V)
        # Drop intercept + drift, reshape (3, n_trials, V)
        betas_probes = betas_full[:3 * len(onsets)].reshape(len(onsets), 3, N_VOXELS)
        # Average across trials → (3, V)
        avg_per_voxel = betas_probes.mean(axis=0)            # (3, V)
        accumulated_weights += avg_per_voxel.T                # (V, 3)
        n_trials_total += len(onsets)

    # Normalize each voxel's weights so they sum to 1 (sign-preserving)
    raw_weights = accumulated_weights / len(RUNS)            # (V, 3) avg over runs
    # Take the dominant direction at each voxel
    norms = np.abs(raw_weights).sum(axis=1, keepdims=True) + 1e-8
    voxel_weights = (raw_weights / norms).astype(np.float32)
    print(f"  fitted weights mean: {voxel_weights.mean(axis=0)}  "
          f"({time.time()-t0:.1f}s, {n_trials_total} training trials)")
    return voxel_weights


def run_variant_b_fitted():
    name = "VariantB_FLOBS_fitted_glover_rtm"
    print(f"\n=== {name} ===")
    t0 = time.time()

    voxel_weights = fit_flobs_weights_ses01()

    # Run Variant B on ses-03 with fitted weights
    cfg_v = V.VariantConfig(
        tr=TR, n_voxels=N_VOXELS, vol_shape=(76, 90, 74), max_trs=200,
        flobs_path="/Users/mhough/fsl/data/feat5/default_flobs.flobs/hrfbasisfns.txt",
        brain_mask_path=str(P.BRAIN_MASK),
        events_dir=str(P.EVENTS_DIR),
        mc_volumes_dir=str(P.MC_DIR),
        output_base=str(P.OUT_DIR),
    )
    variant = V.VariantB_FLOBS(cfg_v)
    variant.precompute()
    variant.voxel_weights = voxel_weights                   # override the 1/3 default

    flat_b, rel_ = P.load_mask()
    all_betas, trial_ids = [], []
    for run in RUNS:
        ts = P.load_rtmotion(SESSION, run, flat_b, rel_)
        ev = P.load_events(SESSION, run)
        onsets = ev["onset_rel"].values.astype(np.float32)
        for trial_i in range(len(onsets)):
            beta = variant.process_tr(ts, ts.shape[1] - 1, onsets, trial_i)
            all_betas.append(np.asarray(beta, dtype=np.float32))
            trial_ids.append(str(ev.iloc[trial_i].get("image_name", str(trial_i))))

    betas = np.stack(all_betas, axis=0)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_betas.npy", betas)
    np.save(P.OUT_DIR / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    np.save(P.OUT_DIR / f"{name}_{SESSION}_voxel_weights.npy", voxel_weights)
    cfg = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
           "mode": "FLOBS 3-basis with fitted per-voxel weights",
           "training_session": "ses-01 fmriprep", "n_voxels": int(betas.shape[1])}
    with open(P.OUT_DIR / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  saved {name}: {betas.shape}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    run_hybrid_online()
    run_same_image_prior()
    run_variant_b_fitted()
