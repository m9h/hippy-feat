#!/usr/bin/env python3
"""Build aCompCor K=7 PCs from CSF∪WM eroded×1, HP-filtered at 0.01 Hz.

Replicates the Mac champion recipe (`results/apple_silicon_2026-04-28/`):
- threshold CSF and WM PVE at 0.99
- erode each by 1 voxel
- combine CSF ∪ WM noise mask
- per BOLD run: extract noise-voxel timeseries, HP-filter (cosine 0.01 Hz cutoff),
  SVD → top 7 PCs
- save (T_run, 7) per run × 11 runs

Output: /data/derivatives/rtmindeye_paper/task_2_1_betas/acompcor/
        sub-005_ses-03_run-NN_acompcor_K7_csfwm_hp_e1.npy   (n_TR, 7)
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion
from nilearn.image import resample_to_img

PAPER = Path("/data/derivatives/rtmindeye_paper")
T1_DATA = Path("/data/3t/data")
FMRIPREP = (PAPER / "fmriprep_mindeye/data_sub-005/bids/derivatives/fmriprep/sub-005")
OUT = PAPER / "task_2_1_betas" / "acompcor"
SESSION = "ses-03"
TR = 1.5
HP_HZ = 0.01
K = 7


def hp_cosine_drift(n_tr: int, tr: float, hp_hz: float) -> np.ndarray:
    """Cosine basis spanning frequencies < hp_hz to be regressed out."""
    period = 1.0 / hp_hz
    nyquist = 0.5 / tr
    n_drift = int(np.floor(2 * n_tr * tr * hp_hz)) + 1
    if n_drift < 2:
        n_drift = 2
    t = np.arange(n_tr) * tr
    duration = n_tr * tr
    cols = [np.ones(n_tr)]
    for k in range(1, n_drift):
        cols.append(np.cos(np.pi * k * (2 * t + 1) / (2 * n_tr)))
    return np.stack(cols, axis=1).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("[1] loading CSF/WM PVE (T1w space)")
    csf = nib.load(T1_DATA / "T1_brain_seg_pve_0.nii.gz")
    wm = nib.load(T1_DATA / "T1_brain_seg_pve_2.nii.gz")
    csf_t1 = (csf.get_fdata() > 0.99).astype(np.uint8)
    wm_t1 = (wm.get_fdata() > 0.99).astype(np.uint8)
    print(f"  CSF voxels in T1w: {csf_t1.sum()}  WM voxels: {wm_t1.sum()}")

    # Use first run as resampling target (all runs share BOLD grid)
    sample_bold = nib.load(
        FMRIPREP / SESSION / "func"
        / f"sub-005_{SESSION}_task-C_run-01_space-T1w_desc-preproc_bold.nii.gz"
    )
    bold_3d = nib.Nifti1Image(sample_bold.get_fdata()[..., 0], sample_bold.affine)

    print("[2] resampling CSF/WM masks to BOLD grid (nearest)")
    csf_bold = resample_to_img(
        nib.Nifti1Image(csf_t1, csf.affine), bold_3d,
        interpolation="nearest", force_resample=True, copy_header=True,
    ).get_fdata().astype(bool)
    wm_bold = resample_to_img(
        nib.Nifti1Image(wm_t1, wm.affine), bold_3d,
        interpolation="nearest", force_resample=True, copy_header=True,
    ).get_fdata().astype(bool)
    print(f"  CSF voxels in BOLD: {csf_bold.sum()}  WM: {wm_bold.sum()}")

    print("[3] erode×1 each then union")
    csf_e = binary_erosion(csf_bold, iterations=1)
    wm_e = binary_erosion(wm_bold, iterations=1)
    noise = csf_e | wm_e
    print(f"  after erode×1: CSF={csf_e.sum()} WM={wm_e.sum()} union={noise.sum()}")
    if noise.sum() < 500:
        raise RuntimeError(
            f"noise voxel count too low ({noise.sum()}); check thresholds/erosion"
        )

    flat_idx = np.where(noise.flatten())[0]
    np.save(OUT / "noise_mask_idx.npy", flat_idx)
    print(f"  saved noise mask indices ({len(flat_idx)} voxels)")

    print("[4] per-run aCompCor extraction")
    for run in range(1, 12):
        bold_path = (FMRIPREP / SESSION / "func"
                     / f"sub-005_{SESSION}_task-C_run-{run:02d}"
                       f"_space-T1w_desc-preproc_bold.nii.gz")
        bold = nib.load(bold_path).get_fdata().astype(np.float32)        # (X, Y, Z, T)
        T = bold.shape[-1]
        Y_noise = bold.reshape(-1, T)[flat_idx]                          # (V_noise, T)
        # Mean-center per voxel
        Y_noise -= Y_noise.mean(1, keepdims=True)
        # HP-filter: regress out cosine drift basis spanning < 0.01 Hz
        drift = hp_cosine_drift(T, TR, HP_HZ)                            # (T, n_drift)
        proj = drift @ np.linalg.pinv(drift)                             # (T, T)
        Y_hp = Y_noise - Y_noise @ proj.T                                # (V, T)
        # SVD on the time × voxel matrix; top K right singular vectors are
        # the aCompCor PCs (T-dim time courses).
        Yt = Y_hp.T.astype(np.float32)                                   # (T, V)
        U, S, Vt = np.linalg.svd(Yt, full_matrices=False)
        pcs = U[:, :K] * S[:K]                                           # (T, K)
        # Z-score per PC
        pcs = (pcs - pcs.mean(0, keepdims=True)) / (pcs.std(0, keepdims=True) + 1e-8)
        out_path = OUT / f"sub-005_{SESSION}_run-{run:02d}_acompcor_K{K}_csfwm_hp_e1.npy"
        np.save(out_path, pcs.astype(np.float32))
        print(f"  run-{run:02d}: T={T}  noise V={Y_noise.shape[0]}  → {pcs.shape}  "
              f"saved {out_path.name}")

    print(f"\nWrote {OUT}/sub-005_ses-03_run-*.npy")


if __name__ == "__main__":
    main()
