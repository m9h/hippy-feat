#!/usr/bin/env python3
"""
Task 2.1: isolate fMRIPrep vs GLMsingle contributions to the RT decoding gap.

The paper (Iyer et al. ICML 2026) reports a 10 pp image-retrieval gap between
'Offline 3T single-trial' (76 %, fMRIPrep + GLMsingle) and 'End-of-run RT'
(66 %, FSL MCFLIRT + nilearn canonical Glover). This script runs the 2×2
factorial that decomposes that gap.

                   RT motion                  fMRIPrep motion
  Glover (RT GLM)  [End-of-run RT, 66 %]     condition A  ←- ISOLATES fMRIPrep
  GLMsingle HRF    condition B  ← ISOLATES   [Offline 3T, 76 %]
                                   GLMsingle

Outputs per-trial betas per condition; downstream inference (model.ridge →
backbone → CLIP embedding) is run in the MindEye PyTorch container to avoid
JAX/Torch conflicts in this venv (see Dockerfile.mindeye-variants).

Data prerequisites (all downloaded by companion sbatch scripts):
  /data/derivatives/rtmindeye_paper/
    fmriprep_mindeye/ data_sub-005/bids/derivatives/fmriprep/sub-005/ses-{01,03,06}/
    rt3t/             data/model/, data/events/, data/sub-005_final_mask.nii.gz, ...
    checkpoints/      data_scaling_exp/concat_glmsingle/checkpoints/...sample=10...pth
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Literal

import numpy as np
import nibabel as nib
import pandas as pd
import jax.numpy as jnp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_glm_variants import (
    VariantConfig,
    VariantA_Baseline,
    VariantC_PerVoxelHRF,
    make_glover_hrf,
    build_design_matrix,
    load_glmsingle_hrf_library,
    load_hrf_indices,
    load_brain_mask,
    _ols_fit,
    _variant_g_forward,
)

# -----------------------------------------------------------------------------
# Paths (as laid out by download_*.sbatch)
# -----------------------------------------------------------------------------

PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
FMRIPREP_ROOT = PAPER_ROOT / "fmriprep_mindeye" / "data_sub-005" / "bids" / "derivatives" / "fmriprep" / "sub-005"
RT3T_DATA = PAPER_ROOT / "rt3t" / "data"
CHECKPOINT_ROOT = PAPER_ROOT / "checkpoints"

# NOTE on mask naming (important):
# `sub-005_final_mask.nii.gz` is misleadingly named — it is actually the
# 19174-voxel BRAIN mask, not the paper's finalmask.
# The paper's canonical "finalmask" = brain mask (19174) filtered by
# `sub-005_ses-01_task-C_relmask.npy` (2792 True voxels after the r > 0.2
# reliability threshold described in Section 2.6.2 of the preprint).
# 2792 also matches the training betas at /data/3t/data/real_time_betas/.
BRAIN_MASK_PATH = RT3T_DATA / "sub-005_final_mask.nii.gz"         # 19174 voxels
RELMASK_PATH = RT3T_DATA / "sub-005_ses-01_task-C_relmask.npy"    # (19174,) bool, 2792 True
EVENTS_DIR = RT3T_DATA / "events"


# -----------------------------------------------------------------------------
# Mask loading — paper's finalmask = brain mask ∩ reliability mask = 2792 voxels
# -----------------------------------------------------------------------------

def load_paper_finalmask() -> tuple[np.ndarray, np.ndarray]:
    """Load the paper-canonical finalmask as (flat_volume_mask, relmask_bool).

    Returns:
        flat_brain_mask: (X*Y*Z,) bool — True for the 19174 brain voxels
        relmask: (19174,) bool — True for the 2792 reliable voxels (over
                 brain-mask-indexed voxels)

    To get a timeseries at the paper's 2792-voxel resolution, apply
    both sequentially:
        masked = vol_3d.flatten()[flat_brain_mask][relmask]
    """
    brain_img = nib.load(BRAIN_MASK_PATH)
    flat_brain = (brain_img.get_fdata() > 0).flatten()
    if int(flat_brain.sum()) != 19174:
        raise RuntimeError(
            f"Brain mask has {int(flat_brain.sum())} voxels, expected 19174"
        )
    rel = np.load(RELMASK_PATH)
    if rel.shape != (19174,) or rel.dtype != bool:
        raise RuntimeError(
            f"Relmask shape/dtype unexpected: {rel.shape} {rel.dtype}"
        )
    return flat_brain, rel


def apply_paper_mask(vol_3d: np.ndarray) -> np.ndarray:
    """Apply brain mask then reliability mask → (2792,) float32."""
    flat_brain, rel = load_paper_finalmask()
    return vol_3d.flatten()[flat_brain][rel].astype(np.float32)


# -----------------------------------------------------------------------------
# Volume-source adapters
# -----------------------------------------------------------------------------

def load_fmriprep_bold(session: str, run: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (masked_timeseries, affine) from the fMRIPrep'd run at 2792 voxels."""
    p = FMRIPREP_ROOT / session / "func" / (
        f"sub-005_{session}_task-C_run-{run:02d}_space-T1w_desc-preproc_bold.nii.gz"
    )
    img = nib.load(p)
    vol_4d = img.get_fdata()  # (X, Y, Z, T)
    flat_brain, rel = load_paper_finalmask()
    T = vol_4d.shape[-1]
    # Apply brain mask then reliability mask
    return vol_4d.reshape(-1, T)[flat_brain][rel].astype(np.float32), img.affine


def load_rt_motion_bold(session: str, run: int) -> np.ndarray:
    """
    Return masked timeseries from RT-motion-corrected volumes at 2792 voxels.
    Uses the mc_boldres files at /data/3t/derivatives/motion_corrected_resampled/
    (only ses-06 run-01 fully available locally at time of writing — this
    function will raise a clear error for sessions that need the jaxoccoli
    MC pipeline run first).
    """
    mc_dir = Path("/data/3t/derivatives/motion_corrected_resampled")
    pattern = f"{session.replace('ses-', 'ses-')}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(mc_dir.glob(pattern))
    if not vols:
        raise FileNotFoundError(
            f"No RT-motion-corrected volumes for {session} run-{run:02d}. "
            f"Run scripts/motion_correct_resampled.py first, or use fMRIPrep path."
        )

    flat_brain, rel = load_paper_finalmask()
    frames = []
    for v in vols:
        vol_3d = nib.load(v).get_fdata()
        frames.append(vol_3d.flatten()[flat_brain][rel].astype(np.float32))
    return np.stack(frames, axis=1)  # (2792, T)


# -----------------------------------------------------------------------------
# GLM variants for the 2×2 cells
# -----------------------------------------------------------------------------

def glm_rt_glover(timeseries: np.ndarray, onsets: np.ndarray, probe_trial: int,
                  tr: float, n_trs: int) -> np.ndarray:
    """Canonical Glover HRF + cosine drift, LSS design. Matches paper's RT GLM."""
    hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
    dm, probe_idx = build_design_matrix(onsets, tr, n_trs, hrf, probe_trial)
    betas = _ols_fit(jnp.asarray(dm), jnp.asarray(timeseries))
    return np.asarray(betas[:, probe_idx], dtype=np.float32)


def glm_glmsingle_style(timeseries: np.ndarray, onsets: np.ndarray, probe_trial: int,
                        tr: float, n_trs: int, hrf_indices: np.ndarray,
                        hrf_library: np.ndarray, base_time: np.ndarray) -> np.ndarray:
    """Per-voxel HRF from GLMsingle 20-HRF library (Variant C logic)."""
    n_voxels = timeseries.shape[0]
    result = np.zeros(n_voxels, dtype=np.float32)
    unique_hrfs = np.unique(hrf_indices)
    n_hrf_trs = int(np.ceil(32.0 / tr))
    for h in unique_hrfs:
        voxel_ids = np.where(hrf_indices == int(h))[0]
        if len(voxel_ids) == 0:
            continue
        from rt_glm_variants import resample_hrf
        hrf = resample_hrf(hrf_library[:, int(h)], base_time, tr, n_hrf_trs)
        dm, probe_idx = build_design_matrix(onsets, tr, n_trs, hrf, probe_trial)
        betas = _ols_fit(jnp.asarray(dm), jnp.asarray(timeseries[voxel_ids]))
        result[voxel_ids] = np.asarray(betas[:, probe_idx], dtype=np.float32)
    return result


def glm_variant_g(timeseries: np.ndarray, onsets: np.ndarray, probe_trial: int,
                  tr: float, n_trs: int, max_trs: int = 200
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Bayesian conjugate AR(1) GLM. Returns (β_mean, β_var) for the probe trial.

    Posterior variance is the novel signal that's typically discarded — Task 6
    MVE retrains the ridge to consume it. Caller should save both arrays.
    """
    hrf = make_glover_hrf(tr, int(np.ceil(32.0 / tr)))
    dm, probe_idx = build_design_matrix(onsets, tr, n_trs, hrf, probe_trial)
    dm_pad = np.zeros((max_trs, dm.shape[1]), dtype=np.float32)
    dm_pad[:n_trs] = dm
    ts_pad = np.zeros((timeseries.shape[0], max_trs), dtype=np.float32)
    ts_pad[:, :n_trs] = timeseries.astype(np.float32)
    betas, vars_ = _variant_g_forward(jnp.asarray(dm_pad), jnp.asarray(ts_pad),
                                      jnp.asarray(n_trs, dtype=jnp.int32))
    return (np.asarray(betas[:, probe_idx], dtype=np.float32),
            np.asarray(np.maximum(vars_[:, probe_idx], 1e-10), dtype=np.float32))


# -----------------------------------------------------------------------------
# Factorial driver
# -----------------------------------------------------------------------------

Condition = Literal["A_fmriprep_glover", "B_rtmotion_glmsingle",
                    "RT_paper", "Offline_paper",
                    "G_fmriprep", "G_rtmotion"]


def run_condition(condition: Condition, session: str, runs: list[int],
                  tr: float = 1.5) -> tuple[np.ndarray, list[int]]:
    """Compute per-trial betas for all probe trials in `runs`.

    Returns:
        betas: (N_trials, N_voxels_finalmask) float32
        trial_image_ids: list of image IDs for each trial (to align with labels)
    """
    flat_brain, rel = load_paper_finalmask()
    n_voxels = int(rel.sum())  # 2792

    if condition in ("A_fmriprep_glover", "Offline_paper", "G_fmriprep"):
        loader = load_fmriprep_bold
    elif condition in ("B_rtmotion_glmsingle", "RT_paper", "G_rtmotion"):
        loader = lambda s, r: (load_rt_motion_bold(s, r), None)
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if condition in ("B_rtmotion_glmsingle", "Offline_paper"):
        base_time, hrf_library = load_glmsingle_hrf_library(
            "/data/3t/data/getcanonicalhrflibrary.tsv"
        )
        # GLMsingle HRF indices are a full-volume (76,90,74,1) npy. Index
        # into it with the same brain→reliability pipeline we use on BOLD.
        hrf_indices_vol = np.load("/data/3t/data/avg_hrfs_s1_s2_full.npy")[:, :, :, 0].astype(int)
        hrf_indices = hrf_indices_vol.flatten()[flat_brain][rel]
    else:
        hrf_library = base_time = hrf_indices = None

    all_betas, all_trial_ids, all_vars = [], [], []
    for run in runs:
        ts, _ = loader(session, run) if condition in ("A_fmriprep_glover", "Offline_paper", "G_fmriprep") \
                else (loader(session, run)[0], None)
        events = pd.read_csv(EVENTS_DIR / f"sub-005_{session}_task-C_run-{run:02d}_events.tsv",
                             sep="\t")
        onsets = events["onset"].astype(float).values - events["onset"].iloc[0]
        n_trs = ts.shape[1]

        for probe_idx_trial in range(len(onsets)):
            if condition in ("A_fmriprep_glover", "RT_paper"):
                beta = glm_rt_glover(ts, onsets, probe_idx_trial, tr, n_trs)
            elif condition in ("G_fmriprep", "G_rtmotion"):
                beta, var = glm_variant_g(ts, onsets, probe_idx_trial, tr, n_trs)
                all_vars.append(var)
            else:
                beta = glm_glmsingle_style(ts, onsets, probe_idx_trial, tr, n_trs,
                                           hrf_indices, hrf_library, base_time)
            all_betas.append(beta)
            # Events TSV uses `image_name` column (e.g.,
            # 'all_stimuli/unchosen_nsd_1000_images/unchosen_7211_cocoid_59250.png').
            # Keep the full string so downstream inference can pair with
            # stimulus CLIP embeddings via image filename.
            img_name = events.iloc[probe_idx_trial].get(
                "image_name",
                events.iloc[probe_idx_trial].get("image_id", str(probe_idx_trial))
            )
            all_trial_ids.append(str(img_name))

    if all_vars:
        return (np.stack(all_betas, axis=0), all_trial_ids,
                np.stack(all_vars, axis=0))
    return np.stack(all_betas, axis=0), all_trial_ids


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Task 2.1 factorial: fMRIPrep vs GLMsingle contributions")
    ap.add_argument("--conditions", nargs="+",
                    default=["A_fmriprep_glover", "B_rtmotion_glmsingle"],
                    choices=["A_fmriprep_glover", "B_rtmotion_glmsingle",
                             "RT_paper", "Offline_paper",
                             "G_fmriprep", "G_rtmotion"])
    ap.add_argument("--session", default="ses-03", help="Test session (paper uses ses-03)")
    ap.add_argument("--runs", nargs="+", type=int, default=list(range(1, 12)))
    ap.add_argument("--out-dir", default="/data/derivatives/rtmindeye_paper/task_2_1_betas")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in args.conditions:
        t0 = time.time()
        print(f"\n=== condition: {cond} ===")
        try:
            result = run_condition(cond, args.session, args.runs)
            if len(result) == 3:
                betas, trial_ids, vars_ = result
                np.save(out_dir / f"{cond}_{args.session}_vars.npy", vars_)
                print(f"  vars shape: {vars_.shape} (saved alongside betas)")
            else:
                betas, trial_ids = result
            np.save(out_dir / f"{cond}_{args.session}_betas.npy", betas)
            np.save(out_dir / f"{cond}_{args.session}_trial_ids.npy",
                    np.asarray(trial_ids))
            print(f"  betas shape: {betas.shape}  elapsed: {time.time()-t0:.1f}s")
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    print("\nBetas saved to:", out_dir)
    print("Next step: run MindEye inference in the PyTorch container:")
    print(f"  docker run --gpus all nvcr.io/nvidia/pytorch:26.02-py3 \\")
    print(f"    python scripts/run_mindeye_inference.py --betas {out_dir}/...")


if __name__ == "__main__":
    main()
