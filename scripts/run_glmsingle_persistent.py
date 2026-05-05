#!/usr/bin/env python3
"""Test (b): run REAL GLMsingle (cvnlab/GLMsingle Python package) on
sub-005 ses-03 BOLD as a persistent / global LSR fit across all 11 runs.

Two cells produced:
  GLMsingle_persistent_rtmotion  — same pipeline as canonical paper output
                                   but on rtmotion (RT-deployable) BOLD
  GLMsingle_persistent_fmriprep  — should match canonical .npz (matches
                                   paper Offline anchor)

Goal: directly resolve the residual ~10 pp gap between End-of-run RT
(per-run LSS, rtmotion BOLD) and canonical Offline (persistent global
fit, fmriprep BOLD + GLMsingle Stages 1+3). Two factors entangled —
this script untangles them by running the same GLMsingle pipeline on
both BOLD sources.

Pipeline mirror of canonical TYPED_FITHRF_GLMDENOISE_RR:
  - Stage 1: per-voxel HRF library (default; 20 candidate HRFs)
  - Stage 2: GLMdenoise (CV-selected K; for sub-005 ses-03 will likely
             pick pcnum=0 like canonical does)
  - Stage 3: per-voxel SVD fracridge (CV-selected fraction per voxel)

Stim duration = 3 s, ITI = 1 s, TR = 1.5 s per Methods §3T fMRI Acquisition.

Output: TYPED_FITHRF_GLMDENOISE_RR.npz in OUTDIR/{rtmotion,fmriprep}/
plus a probe cell saved to PREREG_DIR.
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

PAPER_ROOT = Path("/data/derivatives/rtmindeye_paper")
RT3T = PAPER_ROOT / "rt3t" / "data"
EVENTS_DIR = RT3T / "events"
MC_DIR = Path("/data/3t/derivatives/motion_corrected_resampled")
FMRIPREP_ROOT = (PAPER_ROOT / "fmriprep_mindeye" / "data_sub-005"
                  / "bids" / "derivatives" / "fmriprep" / "sub-005")
PREREG_DIR = PAPER_ROOT / "task_2_1_betas" / "prereg"
GLMSINGLE_OUT_ROOT = PAPER_ROOT / "task_2_1_betas" / "glmsingle_persistent"
SESSION = "ses-03"
RUNS = list(range(1, 12))
TR = 1.5
STIMDUR = 3.0


def load_rtmotion_4d(run: int) -> np.ndarray:
    """Stack per-TR mc_boldres files into one (X, Y, Z, T) array."""
    pattern = f"{SESSION}_run-{run:02d}_*_mc_boldres.nii.gz"
    vols = sorted(MC_DIR.glob(pattern))
    if not vols:
        raise FileNotFoundError(f"no mc_boldres for {SESSION} run-{run:02d}")
    frames = [nib.load(v).get_fdata().astype(np.float32) for v in vols]
    return np.stack(frames, axis=-1)                                # (X, Y, Z, T)


def load_fmriprep_4d(run: int) -> np.ndarray:
    p = (FMRIPREP_ROOT / SESSION / "func"
         / f"sub-005_{SESSION}_task-C_run-{run:02d}"
           f"_space-T1w_desc-preproc_bold.nii.gz")
    return nib.load(p).get_fdata().astype(np.float32)


def build_global_image_index() -> tuple[dict[str, int], list[str]]:
    """One column per unique image across the full session — required for
    GLMsingle Stages 2 + 3 cross-validation across reps."""
    seen: list[str] = []
    idx: dict[str, int] = {}
    for run in range(1, 12):
        events = pd.read_csv(
            EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv",
            sep="\t",
        )
        events = events[events["image_name"] != "blank.jpg"]
        for img in events["image_name"].astype(str).tolist():
            if img not in idx:
                idx[img] = len(seen)
                seen.append(img)
    return idx, seen


def build_design_matrix_condition_mode(run: int, n_trs: int,
                                         image_index: dict[str, int]
                                         ) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Condition-mode design: one column per unique image (across the
    whole session), multiple ones per column for repeated images. Returns
    (design (T, n_unique_images), per-trial (col_index, image_name) list).
    """
    events = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv",
                          sep="\t")
    events = events[events["image_name"] != "blank.jpg"].reset_index(drop=True)
    onsets = np.asarray(events["onset"].astype(float).values, dtype=np.float64).copy()
    onsets -= onsets[0]
    images = events["image_name"].astype(str).tolist()
    n_conditions = len(image_index)
    design = np.zeros((n_trs, n_conditions), dtype=np.float32)
    trial_log: list[tuple[int, str]] = []
    for onset, img in zip(onsets, images):
        col = image_index[img]
        onset_tr = int(round(onset / TR))
        if 0 <= onset_tr < n_trs:
            design[onset_tr, col] = 1.0
            trial_log.append((col, img))
    return design, trial_log


def run_glmsingle_persistent(bold_source: str) -> tuple[np.ndarray, np.ndarray]:
    """Run GLMsingle's persistent global fit and return (betas, trial_ids)."""
    from glmsingle.glmsingle import GLM_single

    print(f"\n=== GLMsingle persistent ({bold_source}) ===", flush=True)
    print("[1/4] loading per-run BOLD", flush=True)
    loader = load_rtmotion_4d if bold_source == "rtmotion" else load_fmriprep_4d
    data, n_trials_per_run, total_trials = [], [], 0
    for r in RUNS:
        bold = loader(r)
        data.append(bold)
        n = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{r:02d}_events.tsv",
                         sep="\t")
        n = n[n["image_name"] != "blank.jpg"]
        n_trials_per_run.append(len(n))
        total_trials += len(n)
        print(f"  run-{r:02d}  bold {bold.shape}  trials {n_trials_per_run[-1]}", flush=True)
    print(f"  total trials across {len(RUNS)} runs: {total_trials}", flush=True)

    print("[2/4] building per-run condition-mode design matrices", flush=True)
    image_index, _ = build_global_image_index()
    n_conditions = len(image_index)
    design, all_trial_log = [], []                                   # per-run (T, n_unique_images)
    for r, bold in zip(RUNS, data):
        d, trial_log = build_design_matrix_condition_mode(r, bold.shape[-1], image_index)
        design.append(d)
        all_trial_log += trial_log                                    # session-ordered (col, image)
    print(f"  design: {len(design)} runs × (T_run × {n_conditions} unique images)",
          f"total_trials={len(all_trial_log)}", flush=True)

    outdir = GLMSINGLE_OUT_ROOT / bold_source
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[3/4] running GLM_single.fit (output → {outdir})", flush=True)

    # Default options — same pipeline as canonical .npz: HRF library +
    # GLMdenoise (CV pcnum) + per-voxel SVD fracridge.
    opt = dict(
        wantlibrary=1,         # Stage 1: HRF library on
        wantglmdenoise=1,      # Stage 2: GLMdenoise on (CV picks pcnum)
        wantfracridge=1,       # Stage 3: fracridge on
        wantfileoutputs=[1, 1, 1, 1],   # save all 4 outputs (A, B, C, D)
        wantmemoryoutputs=[0, 0, 0, 1], # we only need TYPED in memory
    )
    glm = GLM_single(opt)
    t0 = time.time()
    results = glm.fit(design, data, STIMDUR, TR, outputdir=str(outdir))
    print(f"  GLMsingle done in {time.time() - t0:.0f}s", flush=True)

    # Extract trial-level βs — local GLMsingle saves dict-form .npy
    # files, not the .npz format the canonical Princeton dataset uses.
    npy_paths = list(outdir.glob("TYPED_FITHRF_GLMDENOISE_RR.np?"))
    if not npy_paths:
        raise FileNotFoundError(
            f"no TYPED output in {outdir}; got {sorted(p.name for p in outdir.iterdir())}"
        )
    p = npy_paths[0]
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=True)
        betas_3d = z["betasmd"]
        pcnum = int(z["pcnum"])
        fracmean = float(z["FRACvalue"].mean())
    else:
        z = np.load(p, allow_pickle=True).item()
        betas_3d = z["betasmd"]                                       # (X, Y, Z, T)
        pcnum = int(z.get("pcnum", -1))
        fracmean = (float(z["FRACvalue"].mean())
                     if "FRACvalue" in z else float("nan"))
    # Always squeeze to 2D (V, T) — handles both (X,Y,Z,T) and (V_brain, T)
    if betas_3d.ndim == 4:
        # 3D vol + trial — flatten spatial axes
        betas_full = betas_3d.reshape(-1, betas_3d.shape[-1]).astype(np.float32)
    else:
        betas_full = betas_3d.squeeze().astype(np.float32)
    print(f"  pcnum={pcnum}  FRACvalue mean={fracmean:.3f}  betas {betas_full.shape}",
          flush=True)
    print(f"[4/4] projecting to MindEye 2792-voxel relmask + saving cell", flush=True)
    # Local GLMsingle output is in full-volume 3D form. betas_full is
    # already flattened to (V_full = X*Y*Z, T). Apply finalmask + relmask.
    final_mask = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
    if betas_full.shape[0] != final_mask.size:
        raise RuntimeError(
            f"betas_full shape[0]={betas_full.shape[0]} doesn't match "
            f"finalmask size {final_mask.size}"
        )
    betas_me = betas_full[final_mask.flatten()][relmask]              # (2792, T)
    # Map condition → trial output: GLMsingle returns one β per trial
    # occurrence in the order trials appear across runs (matches our
    # all_trial_log). Use the trial_log's image labels as IDs.
    all_trial_ids = [img for _, img in all_trial_log]
    return betas_me.T, np.asarray(all_trial_ids)                      # (T, 2792)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bold-source", choices=["rtmotion", "fmriprep"],
                     required=True)
    args = ap.parse_args()
    PREREG_DIR.mkdir(parents=True, exist_ok=True)
    betas, ids = run_glmsingle_persistent(args.bold_source)
    cell = f"GLMsingle_persistent_{args.bold_source}"
    np.save(PREREG_DIR / f"{cell}_{SESSION}_betas.npy", betas)
    np.save(PREREG_DIR / f"{cell}_{SESSION}_trial_ids.npy", ids)
    print(f"\nsaved {cell}: betas {betas.shape}  ids {len(ids)}", flush=True)


if __name__ == "__main__":
    main()
