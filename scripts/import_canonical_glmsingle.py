#!/usr/bin/env python3
"""Import the canonical Princeton GLMsingle output (TYPED_FITHRF_GLMDENOISE_RR.npz)
as scorable cells in our prereg betas directory.

For sub-005 ses-03 (the Offline anchor session) the canonical pipeline
output sits at:
    /data/derivatives/rtmindeye_paper/glmsingle/glmsingle_sub-005_ses-03_task-C/
        TYPED_FITHRF_GLMDENOISE_RR.npz   (183408 brain voxels × 693 trials)
        sub-005_ses-03_task-C_brain.nii.gz   (canonical brain mask)
        sub-005_ses-03_task-C_nsdgeneral.nii.gz

This script projects the canonical betas onto the MindEye-decoder mask
(2792 voxels = finalmask & relmask), reconstructs trial IDs from events,
and saves them as cells the existing AUC scorer + retrieval pass can
consume without modification.

Two cells produced per session:
  Canonical_GLMsingle_{ses}            — raw canonical βs (693 trials),
                                          score with causal cum-z policy
  Canonical_GLMsingle_{ses}_repeatavg  — post repeat-averaged βs (50 unique
                                          special515 images), for top-1 anchor
"""
from __future__ import annotations
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

GS_ROOT = Path("/data/derivatives/rtmindeye_paper/glmsingle")
RT3T = Path("/data/derivatives/rtmindeye_paper/rt3t/data")
EVENTS = RT3T / "events"
PREREG = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg")

SESSIONS = [
    ("ses-01", "glmsingle_sub-005_ses-01_task-C"),
    ("ses-02", "glmsingle_sub-005_ses-02_task-C"),
    ("ses-03", "glmsingle_sub-005_ses-03_task-C"),
    ("ses-06", "glmsingle_sub-005_ses-06_task-C"),
]


def load_finalmask_to_canonical_indices() -> np.ndarray:
    """Return indices into a length-183408 canonical-brain-flat array
    that map to the 2792 MindEye-decoder voxels."""
    final_mask = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
    canon_brain = nib.load(
        GS_ROOT / "glmsingle_sub-005_ses-03_task-C/sub-005_ses-03_task-C_brain.nii.gz"
    ).get_fdata() > 0
    assert canon_brain.shape == final_mask.shape

    # Position (in flat 3D) of every MindEye finalmask voxel
    me_positions = np.where(final_mask.flatten())[0]                 # (19174,)
    me_positions = me_positions[relmask]                              # (2792,)

    # For each canonical brain voxel, what's its rank in the canonical-brain-flat order?
    canon_flat_to_brain_idx = -np.ones(canon_brain.size, dtype=np.int64)
    canon_flat_to_brain_idx[canon_brain.flatten()] = np.arange(canon_brain.sum())
    me_in_canon_idx = canon_flat_to_brain_idx[me_positions]
    assert (me_in_canon_idx >= 0).all(), "MindEye voxel not in canonical brain mask"
    return me_in_canon_idx


def reconstruct_trial_ids(session: str) -> np.ndarray:
    """693 non-blank trials in order across runs 1..11 — matches both the
    canonical betas and the paper's saved RT betas."""
    out: list[str] = []
    for run in range(1, 12):
        e = pd.read_csv(EVENTS / f"sub-005_{session}_task-C_run-{run:02d}_events.tsv",
                         sep="\t")
        e = e[e["image_name"] != "blank.jpg"]
        out += e["image_name"].astype(str).tolist()
    return np.asarray(out)


def causal_cumulative_zscore(arr: np.ndarray) -> np.ndarray:
    """Per-trial causal cumulative z-score — match retrieval pass policy."""
    out = np.zeros_like(arr, dtype=np.float32)
    n = arr.shape[0]
    for i in range(n):
        if i < 2:
            mu = arr[:max(i, 1)].mean(axis=0, keepdims=True) if i > 0 else 0.0
            sd = 1.0
        else:
            mu = arr[:i].mean(axis=0, keepdims=True)
            sd = arr[:i].std(axis=0, keepdims=True) + 1e-8
        out[i] = (arr[i] - mu) / sd
    return out.astype(np.float32)


def repeat_avg(betas: np.ndarray, ids: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray]:
    """For each unique image, average its rep βs — only on special515."""
    mask = np.array([str(t).startswith("all_stimuli/special515/") for t in ids])
    sb = betas[mask]
    si = ids[mask]
    out_betas, out_ids = [], []
    seen: dict[str, list[int]] = {}
    for i, img in enumerate(si):
        seen.setdefault(img, []).append(i)
    for img, idxs in seen.items():
        out_betas.append(sb[idxs].mean(axis=0))
        out_ids.append(img)
    return np.stack(out_betas, axis=0).astype(np.float32), np.asarray(out_ids)


def main():
    PREREG.mkdir(parents=True, exist_ok=True)
    me_idx = load_finalmask_to_canonical_indices()
    print(f"MindEye-mask projection: {len(me_idx)} voxels into canonical-brain-flat space")

    for ses_label, dir_name in SESSIONS:
        gs_dir = GS_ROOT / dir_name
        npz_path = gs_dir / "TYPED_FITHRF_GLMDENOISE_RR.npz"
        if not npz_path.exists():
            print(f"  SKIP {ses_label}: {npz_path} missing")
            continue

        z = np.load(npz_path, allow_pickle=True)
        betas_full = z["betasmd"].squeeze().astype(np.float32)        # (183408, n_trials)
        pcnum = int(z["pcnum"])
        frac_mean = float(z["FRACvalue"].mean())
        print(f"\n=== {ses_label} === pcnum={pcnum}  frac_mean={frac_mean:.3f}  "
              f"trials={betas_full.shape[1]}")

        # Project: canonical brain → MindEye 2792
        # betas_full is (V_canon=183408, T)
        betas_me = betas_full[me_idx, :]                              # (2792, T)
        betas_per_trial = betas_me.T                                  # (T, 2792)
        del betas_full

        trial_ids = reconstruct_trial_ids(ses_label)
        if betas_per_trial.shape[0] != len(trial_ids):
            print(f"  TRIAL MISMATCH: betas {betas_per_trial.shape[0]} vs ids "
                  f"{len(trial_ids)} — saving anyway with first N matched")
            n = min(betas_per_trial.shape[0], len(trial_ids))
            betas_per_trial = betas_per_trial[:n]
            trial_ids = trial_ids[:n]

        # Cell A — raw canonical βs (693 trials, all conditions)
        cell = f"Canonical_GLMsingle_{ses_label}"
        np.save(PREREG / f"{cell}_ses-03_betas.npy", betas_per_trial)
        np.save(PREREG / f"{cell}_ses-03_trial_ids.npy", trial_ids)
        n_special = int(sum(1 for t in trial_ids
                             if str(t).startswith("all_stimuli/special515/")))
        print(f"  saved {cell}: {betas_per_trial.shape}  n_special515={n_special}")

        # Cell B — repeat-averaged on special515 only (50 unique trials)
        # Apply paper post-processing: causal cum-z first, then repeat-avg
        betas_z = causal_cumulative_zscore(betas_per_trial)
        ra_betas, ra_ids = repeat_avg(betas_z, trial_ids)
        cell_ra = f"Canonical_GLMsingle_{ses_label}_repeatavg"
        np.save(PREREG / f"{cell_ra}_ses-03_betas.npy", ra_betas)
        np.save(PREREG / f"{cell_ra}_ses-03_trial_ids.npy", ra_ids)
        print(f"  saved {cell_ra}: {ra_betas.shape}  n_unique={len(ra_ids)}")


if __name__ == "__main__":
    main()
