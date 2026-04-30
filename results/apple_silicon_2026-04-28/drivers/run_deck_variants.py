#!/usr/bin/env python3
"""Run the original team-deck variants that were never wired into the prereg
sweep: A+N (CSF/WM nuisance), B (FLOBS 3-basis HRF), E (Spatial Laplacian),
C+D (per-voxel HRF + Bayesian).

Uses the existing Variant classes from scripts/rt_glm_variants.py (created
back when the deck was assembled, validated by tests but never wired into
the prereg sweep).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import nibabel as nib

warnings.filterwarnings("ignore")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

import rt_glm_variants as V
import prereg_variant_sweep as P

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
P.RT3T = LOCAL / "rt3t" / "data"
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
P.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
P.EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"
P.HRF_INDICES_PATH = str(LOCAL / "rt3t" / "data" / "avg_hrfs_s1_s2_full.npy")
P.HRF_LIB_PATH = str(LOCAL / "rt3t" / "data" / "getcanonicalhrflibrary.tsv")
OUT = LOCAL / "task_2_1_betas" / "prereg"

SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5

# Build VariantConfig for sub-005 finalmask layout (2792 voxels)
flat_brain = (nib.load(P.BRAIN_MASK).get_fdata() > 0).flatten()
rel = np.load(P.RELMASK)
N_VOXELS = int(rel.sum())  # 2792

cfg = V.VariantConfig(
    tr=TR, n_voxels=N_VOXELS, vol_shape=(76, 90, 74), max_trs=200,
    hrf_indices_path=P.HRF_INDICES_PATH,
    hrf_library_path=P.HRF_LIB_PATH,
    flobs_path="/Users/mhough/fsl/data/feat5/default_flobs.flobs/hrfbasisfns.txt",
    union_mask_path=str(LOCAL / "rt3t" / "data" / "union_mask_from_ses-01-02.npy"),
    brain_mask_path=str(P.BRAIN_MASK),
    events_dir=str(P.EVENTS_DIR),
    mc_volumes_dir=str(P.MC_DIR),
    output_base=str(OUT),
)


def run_variant(name: str, variant, precompute_kwargs: dict | None = None):
    print(f"\n=== {name} ===")
    t0 = time.time()
    if precompute_kwargs is None:
        variant.precompute()
    else:
        variant.precompute(**precompute_kwargs)
    print(f"  precomputed ({time.time()-t0:.1f}s)")

    all_betas, trial_ids = [], []
    for run in RUNS:
        ts = P.load_rtmotion(SESSION, run, flat_brain, rel)        # (V, T)
        events = P.load_events(SESSION, run)
        onsets = events["onset_rel"].values.astype(np.float32)
        for trial_i in range(len(onsets)):
            beta = variant.process_tr(ts, ts.shape[1] - 1, onsets, trial_i)
            all_betas.append(np.asarray(beta, dtype=np.float32))
            trial_ids.append(str(events.iloc[trial_i].get("image_name", str(trial_i))))

    betas = np.stack(all_betas, axis=0)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}_{SESSION}_betas.npy", betas)
    np.save(OUT / f"{name}_{SESSION}_trial_ids.npy", np.asarray(trial_ids))
    config = {"cell": name, "session": SESSION, "runs": RUNS, "tr": TR,
              "variant_class": type(variant).__name__,
              "bold_source": "rtmotion", "window": "full-run",
              "n_voxels": N_VOXELS}
    with open(OUT / f"{name}_{SESSION}_config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"  saved {name}: betas {betas.shape}  ({time.time()-t0:.1f}s)")


# ------------ Variant B: FLOBS 3-basis HRF -----------------
run_variant("VariantB_FLOBS_glover_rtm",
            V.VariantB_FLOBS(cfg))

# ------------ Variant E: Spatial Laplacian -----------------
e_variant = V.VariantE_Spatial(cfg, lam=0.1)
run_variant("VariantE_Spatial_glover_rtm",
            e_variant,
            precompute_kwargs=dict(brain_mask_flat=flat_brain, union_mask=rel))

# ------------ Variant C+D: per-voxel HRF + Bayesian shrinkage -----------------
# Use ses-01 G_fmriprep betas as training prior for D (we already produced these)
ses01_priorpath = LOCAL / "task_2_1_betas" / "G_fmriprep_ses-01_betas.npy"
if ses01_priorpath.exists():
    training_betas = np.load(ses01_priorpath)
    # Match shape to (n_trials, n_voxels=2792)
    print(f"\n  using ses-01 G_fmriprep training betas: {training_betas.shape}")
else:
    training_betas = None
    print(f"\n  WARN: no ses-01 prior at {ses01_priorpath}; CD using uninformative prior")

cd_variant = V.VariantCD_Combined(cfg)
run_variant("VariantCD_Combined_glover_rtm", cd_variant,
            precompute_kwargs=dict(brain_mask_flat=flat_brain, union_mask=rel,
                                    training_betas=training_betas))
