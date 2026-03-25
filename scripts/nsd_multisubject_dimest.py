#!/usr/bin/env python3
"""
NSD Multi-Subject Dimensionality Estimation

Runs eigenspectrum analysis + MELODIC on nsdgeneral-masked betas
for all 8 NSD subjects (sessions 01-03).
Generates presentation-ready figures comparing dimensionality across subjects.
"""

import os
import sys
import subprocess
import csv
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Config
NSD_DIR = Path("/data/3t/nsd_multisubject")
OUT_DIR = Path("/data/derivatives/mindeye_variants/comparison")
FSLDIR = "/home/mhough/fsl"
MELODIC = f"{FSLDIR}/bin/melodic"
os.environ["FSLDIR"] = FSLDIR
os.environ["FSLOUTPUTTYPE"] = "NIFTI_GZ"

SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
SESSIONS = ["01", "02", "03"]
DPI = 150
COLORS = plt.cm.Set2(np.linspace(0, 1, 8))

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


def load_and_mask_betas(subj: str) -> Optional[np.ndarray]:
    """Load sessions 01-03 betas, apply nsdgeneral mask, return (n_trials, n_voxels)."""
    mask_path = NSD_DIR / f"{subj}_nsdgeneral.nii.gz"
    if not mask_path.exists():
        print(f"  {subj}: no nsdgeneral mask", flush=True)
        return None

    mask_img = nib.load(str(mask_path))
    mask = mask_img.get_fdata().flatten() > 0
    n_voxels = int(mask.sum())
    print(f"  {subj}: nsdgeneral={n_voxels} voxels", flush=True)

    all_betas = []
    for ses in SESSIONS:
        beta_path = NSD_DIR / subj / f"betas_session{ses}.nii.gz"
        if not beta_path.exists():
            print(f"    session {ses}: not found", flush=True)
            continue
        img = nib.load(str(beta_path))
        data = img.get_fdata()  # (X, Y, Z, n_trials)
        n_trials = data.shape[3]
        # Flatten spatial dims and apply mask
        flat = data.reshape(-1, n_trials)  # (n_voxels_total, n_trials)
        masked = flat[mask, :]  # (n_nsdgeneral, n_trials)
        all_betas.append(masked.T)  # (n_trials, n_nsdgeneral)
        print(f"    session {ses}: {n_trials} trials", flush=True)

    if not all_betas:
        return None
    combined = np.concatenate(all_betas, axis=0).astype(np.float32)
    print(f"  {subj}: total {combined.shape}", flush=True)
    return combined


def compute_eigenspectrum(X: np.ndarray, max_k: int = 200) -> Tuple[np.ndarray, np.ndarray]:
    """Truncated SVD eigenspectrum."""
    from scipy.sparse.linalg import svds
    n, p = X.shape
    X_c = X - X.mean(axis=0, keepdims=True)
    k = min(max_k, min(n, p) - 1)
    _, s, _ = svds(X_c, k=k, which='LM')
    s = s[::-1]
    eig = (s ** 2) / (n - 1)
    total_var = max(np.sum(X_c ** 2) / (n - 1), eig.sum())
    return eig, eig / total_var


def broken_stick(p: int) -> np.ndarray:
    expected = np.zeros(p)
    for k in range(p):
        expected[k] = np.sum(1.0 / np.arange(k + 1, p + 1))
    expected /= p
    return expected


def estimate_dim_broken_stick(var_explained: np.ndarray) -> int:
    bs = broken_stick(len(var_explained))
    dim = 0
    for s, b in zip(var_explained, bs):
        if s > b:
            dim += 1
        else:
            break
    return max(dim, 1)


def run_melodic_on_betas(betas: np.ndarray, subj: str, mask_path: str) -> Dict[str, int]:
    """Convert betas to NIfTI and run MELODIC with all estimators."""
    import tempfile

    mask_img = nib.load(mask_path)
    mask = mask_img.get_fdata().flatten() > 0
    vol_shape = mask_img.shape

    # Subsample for speed
    max_trials = 500
    if betas.shape[0] > max_trials:
        idx = np.random.RandomState(42).choice(betas.shape[0], max_trials, replace=False)
        idx.sort()
        betas_sub = betas[idx]
    else:
        betas_sub = betas

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build 4D NIfTI
        n_trials = betas_sub.shape[0]
        vol_4d = np.zeros((*vol_shape, n_trials), dtype=np.float32)
        brain_indices = np.where(mask)[0]
        for t in range(n_trials):
            flat = np.zeros(np.prod(vol_shape), dtype=np.float32)
            flat[brain_indices] = betas_sub[t]
            vol_4d[:, :, :, t] = flat.reshape(vol_shape)

        nifti_path = os.path.join(tmpdir, f"{subj}_betas.nii.gz")
        nib.save(nib.Nifti1Image(vol_4d, mask_img.affine, mask_img.header), nifti_path)

        # Create binary mask NIfTI for MELODIC
        mask_nifti = os.path.join(tmpdir, f"{subj}_mask.nii.gz")
        mask_vol = mask.reshape(vol_shape).astype(np.float32)
        nib.save(nib.Nifti1Image(mask_vol, mask_img.affine), mask_nifti)

        results = {}
        for method in ["lap", "bic", "mdl", "aic", "mean"]:
            mel_out = os.path.join(tmpdir, f"mel_{method}")
            os.makedirs(mel_out, exist_ok=True)
            cmd = [MELODIC, "-i", nifti_path, "-o", mel_out,
                   f"--dimest={method}", "--nobet", "--Opca", "--no_mm",
                   "-m", mask_nifti]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            dim = -1
            log = os.path.join(mel_out, "log.txt")
            if os.path.exists(log):
                with open(log) as f:
                    for line in f:
                        if "whitening using" in line.lower():
                            parts = line.split()
                            for i, p in enumerate(parts):
                                if p == "using":
                                    try:
                                        dim = int(parts[i + 1])
                                    except (ValueError, IndexError):
                                        pass
            results[method] = dim
        return results


def main():
    print("=" * 60, flush=True)
    print("NSD Multi-Subject Dimensionality Estimation", flush=True)
    print("=" * 60, flush=True)

    all_results = {}
    all_eigenspectra = {}

    for subj in SUBJECTS:
        print(f"\n--- {subj} ---", flush=True)
        betas = load_and_mask_betas(subj)
        if betas is None:
            continue

        # Eigenspectrum
        print(f"  Computing eigenspectrum...", flush=True)
        eig, var_exp = compute_eigenspectrum(betas)
        all_eigenspectra[subj] = (eig, var_exp)

        cum_var = np.cumsum(var_exp)
        dim_bs = estimate_dim_broken_stick(var_exp)

        pcs_80 = int(np.searchsorted(cum_var, 0.80)) + 1
        pcs_90 = int(np.searchsorted(cum_var, 0.90)) + 1
        pcs_95 = int(np.searchsorted(cum_var, 0.95)) + 1

        print(f"  Broken stick: {dim_bs}", flush=True)
        print(f"  80%={pcs_80}, 90%={pcs_90}, 95%={pcs_95}", flush=True)

        # MELODIC
        mask_path = str(NSD_DIR / f"{subj}_nsdgeneral.nii.gz")
        print(f"  Running MELODIC...", flush=True)
        mel_results = run_melodic_on_betas(betas, subj, mask_path)
        print(f"  MELODIC: {mel_results}", flush=True)

        all_results[subj] = {
            "n_trials": betas.shape[0],
            "n_voxels": betas.shape[1],
            "broken_stick": dim_bs,
            "pcs_80": pcs_80,
            "pcs_90": pcs_90,
            "pcs_95": pcs_95,
            **{f"melodic_{k}": v for k, v in mel_results.items()},
        }

    if not all_results:
        print("No data processed!", flush=True)
        sys.exit(1)

    # === Summary Table ===
    print("\n" + "=" * 90, flush=True)
    print(f"{'Subject':<10} {'N':>6} {'Vox':>6} {'BrkSt':>6} {'LAP':>6} {'BIC':>6} {'MDL':>6} {'AIC':>6} {'Mean':>6} {'90%':>6}", flush=True)
    print("-" * 90, flush=True)
    for subj in SUBJECTS:
        if subj not in all_results:
            continue
        r = all_results[subj]
        print(f"{subj:<10} {r['n_trials']:>6} {r['n_voxels']:>6} {r['broken_stick']:>6} "
              f"{r['melodic_lap']:>6} {r['melodic_bic']:>6} {r['melodic_mdl']:>6} "
              f"{r['melodic_aic']:>6} {r['melodic_mean']:>6} {r['pcs_90']:>6}", flush=True)
    print("=" * 90, flush=True)

    # === Save CSV ===
    csv_path = OUT_DIR / "nsd_multisubject_dimest.csv"
    fieldnames = ["subject", "n_trials", "n_voxels", "broken_stick",
                  "melodic_lap", "melodic_bic", "melodic_mdl", "melodic_aic", "melodic_mean",
                  "pcs_80", "pcs_90", "pcs_95"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for subj in SUBJECTS:
            if subj not in all_results:
                continue
            row = {"subject": subj, **all_results[subj]}
            w.writerow({k: row.get(k, -1) for k in fieldnames})
    print(f"\nSaved: {csv_path}", flush=True)

    # === Figures ===

    # 1. Eigenspectrum overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for i, (subj, (eig, var_exp)) in enumerate(all_eigenspectra.items()):
        ax1.semilogy(range(1, min(151, len(eig) + 1)), eig[:150],
                     color=COLORS[i], label=subj, linewidth=1.5, alpha=0.8)
        cum = np.cumsum(var_exp)
        ax2.plot(range(1, min(201, len(cum) + 1)), cum[:200] * 100,
                 color=COLORS[i], label=subj, linewidth=1.5, alpha=0.8)

    ax1.set_xlabel("Component")
    ax1.set_ylabel("Eigenvalue (log scale)")
    ax1.set_title("Eigenspectrum by Subject (NSD)")
    ax1.legend(fontsize=8, ncol=2)

    ax2.set_xlabel("Component")
    ax2.set_ylabel("Cumulative Variance (%)")
    ax2.set_title("Cumulative Variance Explained")
    for thresh in [80, 90, 95]:
        ax2.axhline(y=thresh, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax2.legend(fontsize=8, ncol=2)
    ax2.set_ylim(0, 101)

    plt.tight_layout()
    fig_path = OUT_DIR / "nsd_eigenspectrum_by_subject.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {fig_path}", flush=True)

    # 2. Dimensionality comparison bar chart
    subjs_with_data = [s for s in SUBJECTS if s in all_results]
    n_subj = len(subjs_with_data)
    methods = ["broken_stick", "melodic_bic", "melodic_mdl", "melodic_aic", "melodic_lap"]
    method_labels = ["Broken Stick", "MELODIC BIC", "MELODIC MDL", "MELODIC AIC", "MELODIC LAP"]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(n_subj)
    width = 0.15
    method_colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']

    for j, (method, label, color) in enumerate(zip(methods, method_labels, method_colors)):
        vals = [all_results[s].get(method, 0) for s in subjs_with_data]
        ax.bar(x + j * width, vals, width, label=label, color=color, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(subjs_with_data)
    ax.set_ylabel("Estimated Dimensionality")
    ax.set_title("Dimensionality Estimation Across NSD Subjects")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig_path = OUT_DIR / "nsd_dimest_by_subject.png"
    plt.savefig(fig_path, dpi=DPI, bbox_inches='tight')
    print(f"Saved: {fig_path}", flush=True)

    print("\nDone!", flush=True)


if __name__ == "__main__":
    main()
