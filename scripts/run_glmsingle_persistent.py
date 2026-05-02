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


def build_design_matrix(run: int, n_trs: int, n_total_trials: int,
                         trial_offset: int) -> tuple[np.ndarray, list[str]]:
    """Per-run design matrix: each non-blank trial gets its own column,
    one-hot at the onset TR. GLMsingle then produces one β per column."""
    events = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{run:02d}_events.tsv",
                          sep="\t")
    events = events[events["image_name"] != "blank.jpg"].reset_index(drop=True)
    onsets = events["onset"].astype(float).values
    onsets -= onsets[0]                                              # 0-relative
    images = events["image_name"].astype(str).tolist()
    design = np.zeros((n_trs, n_total_trials), dtype=np.float32)
    trial_ids: list[str] = []
    for trial_local, (onset, img) in enumerate(zip(onsets, images)):
        col = trial_offset + trial_local
        onset_tr = int(round(onset / TR))
        if 0 <= onset_tr < n_trs:
            design[onset_tr, col] = 1.0
            trial_ids.append(img)
    return design, trial_ids


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

    print("[2/4] building per-run design matrices", flush=True)
    design, all_trial_ids = [], []
    cursor = 0
    for r, bold in zip(RUNS, data):
        d, ids = build_design_matrix(r, bold.shape[-1], total_trials, cursor)
        design.append(d)
        all_trial_ids += ids
        cursor += n_trials_per_run[RUNS.index(r)]
    print(f"  design: {len(design)} runs × ({design[0].shape[0]} TRs × {total_trials} trials)",
          flush=True)

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

    # Extract trial-level βs from the TYPED (Type D) output
    npz = outdir / "TYPED_FITHRF_GLMDENOISE_RR.npz"
    z = np.load(npz, allow_pickle=True)
    betas_full = z["betasmd"].squeeze().astype(np.float32)            # (V_brain, T)
    pcnum = int(z["pcnum"])
    fracmean = float(z["FRACvalue"].mean())
    print(f"  pcnum={pcnum}  FRACvalue mean={fracmean:.3f}  betas {betas_full.shape}",
          flush=True)
    print(f"[4/4] projecting to MindEye 2792-voxel relmask + saving cell", flush=True)
    # Reuse projection logic from import_canonical_glmsingle
    if bold_source == "rtmotion":
        canon_brain = nib.load(GLMSINGLE_OUT_ROOT / "rtmotion" /
                                "sub-005_ses-03_task-C_brain.nii.gz"
                                ).get_fdata() > 0 \
            if (GLMSINGLE_OUT_ROOT / "rtmotion" /
                "sub-005_ses-03_task-C_brain.nii.gz").exists() \
            else (data[0][..., 0] != 0)                              # rough fallback
    else:
        canon_brain = (data[0][..., 0] != 0)
    final_mask = nib.load(RT3T / "sub-005_final_mask.nii.gz").get_fdata() > 0
    relmask = np.load(RT3T / "sub-005_ses-01_task-C_relmask.npy")
    me_positions = np.where(final_mask.flatten())[0][relmask]
    canon_brain_idx = -np.ones(canon_brain.size, dtype=np.int64)
    canon_brain_idx[canon_brain.flatten()] = np.arange(canon_brain.sum())
    me_in_canon = canon_brain_idx[me_positions]
    if (me_in_canon < 0).any():
        # Fallback: GLMsingle output may use the BOLD's own data mask,
        # not our finalmask. Use the betas_full's first dim alignment
        # by intersecting with the first run's nonzero voxels.
        print(f"  WARN: {(me_in_canon < 0).sum()} relmask voxels not in "
              f"canon_brain; using vector-based fallback projection", flush=True)
        # GLMsingle trims to brain voxels itself; betas_full first dim should
        # match canon_brain.sum(). Need to align our finalmask via volume reshape.
        # Easier: reshape betas_full back to 3D using canon_brain shape, then
        # apply finalmask + relmask in 3D space.
        if betas_full.ndim == 2 and betas_full.shape[0] == int(canon_brain.sum()):
            vol = np.zeros((canon_brain.size, betas_full.shape[1]), dtype=np.float32)
            vol[canon_brain.flatten()] = betas_full
            betas_me = vol[final_mask.flatten()][relmask]
        else:
            raise RuntimeError(f"unexpected betas shape {betas_full.shape}")
    else:
        betas_me = betas_full[me_in_canon]
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
