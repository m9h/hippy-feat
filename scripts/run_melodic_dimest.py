#!/usr/bin/env python3
"""
Run FSL MELODIC dimensionality estimation on MindEye betas.

Converts beta matrices to pseudo-4D NIfTI volumes and runs MELODIC
with each dimensionality estimator (lap, bic, mdl, aic, mean).
Also runs full MELODIC ICA with automatic dimensionality as oracle.

Outputs added to dimensionality_summary.csv and new figures.
"""

import os
import sys
import subprocess
import tempfile
import csv
from pathlib import Path

import numpy as np
import nibabel as nib

# FSL setup
FSLDIR = "/home/mhough/fsl"
MELODIC = f"{FSLDIR}/bin/melodic"
os.environ["FSLDIR"] = FSLDIR
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

# Data
SESSIONS = ["ses-01", "ses-02", "ses-03", "ses-06"]
RUNS = list(range(1, 12))
PER_RUN_TEMPLATE = "/data/3t/derivatives/sub-005_{ses}_task-C_run-{run:02d}_recons/betas_run-{run:02d}.npy"
N_VOXELS = 8627
BRAIN_MASK_PATH = "/data/3t/data/sub-005_final_mask.nii.gz"
UNION_MASK_PATH = "/data/3t/data/union_mask_from_ses-01-02.npy"
OUT_DIR = Path("/data/derivatives/mindeye_variants/comparison")
DIMEST_METHODS = ["lap", "bic", "mdl", "aic", "mean"]


def load_session_betas(session: str) -> np.ndarray:
    """Load and concatenate per-run betas for a session."""
    run_betas = []
    for run in RUNS:
        p = Path(PER_RUN_TEMPLATE.format(ses=session, run=run))
        if p.exists():
            arr = np.load(str(p)).squeeze()
            if arr.ndim == 1:
                arr = arr[np.newaxis, :]
            if arr.shape[-1] == N_VOXELS:
                run_betas.append(arr)
    if not run_betas:
        return None
    return np.concatenate(run_betas, axis=0).astype(np.float32)


def betas_to_nifti(betas: np.ndarray, brain_mask_path: str,
                   union_mask_path: str, output_path: str) -> str:
    """
    Convert (n_trials, 8627) betas back to 4D NIfTI for MELODIC.
    Maps masked voxels back into 3D volume space.
    """
    brain_img = nib.load(brain_mask_path)
    brain_mask = brain_img.get_fdata().flatten() > 0
    union_mask = np.load(union_mask_path)

    vol_shape = brain_img.shape  # (76, 90, 74)
    n_trials = betas.shape[0]

    # Reconstruct 4D volume
    vol_4d = np.zeros((*vol_shape, n_trials), dtype=np.float32)
    brain_indices = np.where(brain_mask)[0]
    union_indices = np.where(union_mask)[0]

    for t in range(n_trials):
        flat = np.zeros(np.prod(vol_shape), dtype=np.float32)
        # Place betas back into brain mask positions at union mask locations
        flat_brain = np.zeros(brain_mask.sum(), dtype=np.float32)
        flat_brain[union_indices] = betas[t]
        flat[brain_indices] = flat_brain
        vol_4d[:, :, :, t] = flat.reshape(vol_shape)

    img = nib.Nifti1Image(vol_4d, brain_img.affine, brain_img.header)
    nib.save(img, output_path)
    return output_path


def run_melodic_dimest(nifti_path: str, method: str, outdir: str,
                       mask_path: str = None) -> int:
    """Run MELODIC with a specific dimensionality estimator, return estimated dim."""
    cmd = [
        MELODIC,
        "-i", nifti_path,
        "-o", outdir,
        "--dimest", method,
        "--nobet",
        "--Opca",
        "--no_mm",
        "--vn",  # turn off variance normalization to match raw betas
    ]
    if mask_path:
        cmd.extend(["-m", mask_path])

    print(f"    Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

    if result.returncode != 0:
        print(f"    MELODIC failed: {result.stderr[:500]}", flush=True)
        return -1

    # Parse estimated dimensionality from MELODIC output
    dim_file = Path(outdir) / "melodic_ICstats"
    if dim_file.exists():
        # Number of rows = number of components
        stats = np.loadtxt(str(dim_file))
        if stats.ndim == 1:
            return 1
        return stats.shape[0]

    # Alternative: check melodic_IC shape
    ic_file = Path(outdir) / "melodic_IC.nii.gz"
    if ic_file.exists():
        ic_img = nib.load(str(ic_file))
        if ic_img.ndim == 4:
            return ic_img.shape[3]
        return 1

    # Parse from log
    log_file = Path(outdir) / "log.txt"
    if log_file.exists():
        with open(log_file) as f:
            for line in f:
                if "estimated" in line.lower() and "dim" in line.lower():
                    # Try to extract number
                    parts = line.split()
                    for p in parts:
                        try:
                            return int(p)
                        except ValueError:
                            continue

    print(f"    Could not determine dimensionality from {outdir}", flush=True)
    return -1


def main():
    print("=" * 60, flush=True)
    print("MELODIC Dimensionality Estimation", flush=True)
    print("=" * 60, flush=True)

    # Create brain mask NIfTI for MELODIC
    brain_img = nib.load(BRAIN_MASK_PATH)

    results = {}

    for ses in SESSIONS:
        print(f"\n--- {ses} ---", flush=True)
        betas = load_session_betas(ses)
        if betas is None:
            print(f"  No data for {ses}", flush=True)
            continue
        print(f"  Loaded betas: {betas.shape}", flush=True)

        # Subsample if too many trials (MELODIC is slow on large datasets)
        max_trials = 500
        if betas.shape[0] > max_trials:
            rng = np.random.RandomState(42)
            idx = rng.choice(betas.shape[0], max_trials, replace=False)
            idx.sort()
            betas_sub = betas[idx]
            print(f"  Subsampled to {betas_sub.shape[0]} trials for MELODIC speed", flush=True)
        else:
            betas_sub = betas

        # Convert to 4D NIfTI
        with tempfile.TemporaryDirectory() as tmpdir:
            nifti_path = os.path.join(tmpdir, f"{ses}_betas.nii.gz")
            print(f"  Converting to NIfTI...", flush=True)
            betas_to_nifti(betas_sub, BRAIN_MASK_PATH, UNION_MASK_PATH, nifti_path)

            ses_results = {}
            for method in DIMEST_METHODS:
                mel_outdir = os.path.join(tmpdir, f"melodic_{method}")
                os.makedirs(mel_outdir, exist_ok=True)
                print(f"  Estimating dim ({method})...", flush=True)
                dim = run_melodic_dimest(
                    nifti_path, method, mel_outdir,
                    mask_path=BRAIN_MASK_PATH
                )
                ses_results[method] = dim
                print(f"    {method}: {dim} components", flush=True)

            results[ses] = ses_results

    # Print summary table
    print("\n" + "=" * 70, flush=True)
    print("MELODIC DIMENSIONALITY ESTIMATION SUMMARY", flush=True)
    print("=" * 70, flush=True)
    header = f"{'Session':<12}" + "".join(f"{m:>8}" for m in DIMEST_METHODS)
    print(header, flush=True)
    print("-" * 52, flush=True)
    for ses in SESSIONS:
        if ses not in results:
            continue
        row = f"{ses:<12}"
        for method in DIMEST_METHODS:
            val = results[ses].get(method, -1)
            row += f"{val:>8}"
        print(row, flush=True)
    print("=" * 70, flush=True)

    # Save to CSV
    csv_path = OUT_DIR / "melodic_dimest.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["session"] + DIMEST_METHODS)
        for ses in SESSIONS:
            if ses not in results:
                continue
            row = [ses] + [results[ses].get(m, -1) for m in DIMEST_METHODS]
            writer.writerow(row)
    print(f"\nSaved: {csv_path}", flush=True)

    # Merge with existing dimensionality_summary.csv
    existing_csv = OUT_DIR / "dimensionality_summary.csv"
    if existing_csv.exists():
        import pandas as pd
        df_existing = pd.read_csv(existing_csv)
        # Add MELODIC columns
        for method in DIMEST_METHODS:
            col = f"melodic_{method}"
            df_existing[col] = -1
            for ses, res in results.items():
                mask = df_existing["session"] == ses
                if mask.any():
                    df_existing.loc[mask, col] = res.get(method, -1)
        df_existing.to_csv(existing_csv, index=False)
        print(f"Updated: {existing_csv}", flush=True)
        print(df_existing.to_string(), flush=True)

    return results


if __name__ == "__main__":
    main()
