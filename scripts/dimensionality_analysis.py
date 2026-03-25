#!/usr/bin/env python3
"""
MindEye fMRI Dimensionality Estimation

Estimates the intrinsic dimensionality of the MindEye NSD fMRI betas using:
  1. Eigenspectrum analysis (PCA via SVD)
  2. Cumulative variance explained curves
  3. Broken stick model
  4. Parallel analysis (comparison to shuffled data)
  5. MELODIC-style consensus subsampling

Outputs (saved to /data/derivatives/mindeye_variants/comparison/):
  - eigenspectrum_by_session.png
  - cumvar_by_session.png
  - dimensionality_summary.csv
  - consensus_dimensionality.png

Usage:
    python scripts/dimensionality_analysis.py
"""

import os
import sys
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import stats as sp_stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUT_DIR = Path("/data/derivatives/mindeye_variants/comparison")

# Per-run beta paths — sessions and runs to scan
SESSIONS = ["ses-01", "ses-02", "ses-03", "ses-06"]
RUNS = list(range(1, 12))  # run-01 … run-11

PER_RUN_TEMPLATE = (
    "/data/3t/derivatives/sub-005_{ses}_task-C_run-{run:02d}_recons/betas_run-{run:02d}.npy"
)

# Pre-concatenated session betas (fallback)
PRECAT_TEMPLATE = "/data/3t/data/real_time_betas/all_betas_{ses}_all_runs_delay0.npy"

UNION_MASK_PATH = "/data/3t/data/union_mask_from_ses-01-02.npy"

# Parallel analysis / consensus parameters
N_PARALLEL_SHUFFLES = 5
N_CONSENSUS_SUBSAMPLES = 10
CONSENSUS_FRAC = 0.80  # fraction of samples per subsample

DPI = 150
SEED = 42
N_VOXELS = 8627

# ---------------------------------------------------------------------------
# Publication-quality matplotlib defaults
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

SESSION_COLORS = {
    "ses-01": "#1f77b4",
    "ses-02": "#ff7f0e",
    "ses-03": "#2ca02c",
    "ses-06": "#d62728",
    "all":    "#7f7f7f",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session_betas(session: str) -> Optional[np.ndarray]:
    """Load betas for a session by concatenating per-run files or falling back
    to the pre-concatenated file. Returns (n_trials, n_voxels) or None."""

    # Try per-run loading first
    run_betas = []
    for run in RUNS:
        p = Path(PER_RUN_TEMPLATE.format(ses=session, run=run))
        if p.exists():
            arr = np.load(str(p))
            run_betas.append(arr)

    if run_betas:
        # Filter to consistent voxel count (8627) — some runs may use a different mask
        target_nvox = N_VOXELS
        consistent = []
        for rb in run_betas:
            rb = rb.squeeze()  # handle extra dims
            if rb.ndim == 1:
                rb = rb[np.newaxis, :]
            if rb.shape[-1] == target_nvox:
                consistent.append(rb)
        if not consistent:
            print(f"  {session}: no runs with {target_nvox} voxels, skipping.")
            return None
        combined = np.concatenate(consistent, axis=0)
        print(f"  {session}: loaded {len(consistent)}/{len(run_betas)} runs -> shape {combined.shape}")
        return combined.astype(np.float64)

    # Fallback to pre-concatenated
    p = Path(PRECAT_TEMPLATE.format(ses=session))
    if p.exists():
        arr = np.load(str(p))
        print(f"  {session}: loaded pre-concatenated -> shape {arr.shape}")
        return arr.astype(np.float64)

    print(f"  {session}: no data found, skipping.")
    return None


def load_all_data() -> Dict[str, np.ndarray]:
    """Load betas for every available session. Returns {session_name: array}."""
    print("Loading data...")
    data = {}
    for ses in SESSIONS:
        arr = load_session_betas(ses)
        if arr is not None:
            data[ses] = arr
    if not data:
        print("ERROR: no session data found at all.")
        sys.exit(1)
    return data


# ---------------------------------------------------------------------------
# Core PCA / eigenspectrum via SVD
# ---------------------------------------------------------------------------

def compute_eigenspectrum(X: np.ndarray, max_k: int = 300) -> Tuple[np.ndarray, np.ndarray]:
    """Center data, compute truncated SVD, return (eigenvalues, variance_explained_ratio).
    eigenvalues = squared singular values / (n-1), i.e. PCA eigenvalues of the
    covariance matrix. Uses truncated SVD for speed on large matrices."""
    from scipy.sparse.linalg import svds
    n, p = X.shape
    X_centered = X - X.mean(axis=0, keepdims=True)
    k = min(max_k, min(n, p) - 1)
    # scipy svds returns smallest k; we want largest, so use which='LM'
    U, s, Vt = svds(X_centered, k=k, which='LM')
    # svds returns in ascending order — reverse
    s = s[::-1]
    eigenvalues = (s ** 2) / (n - 1)
    # Estimate total variance from the data directly
    total_var = np.sum(np.var(X_centered, axis=0, ddof=1)) * (n - 1) / (n - 1)
    total_var = max(np.sum(X_centered ** 2) / (n - 1), eigenvalues.sum())
    var_explained = eigenvalues / total_var
    return eigenvalues, var_explained


# ---------------------------------------------------------------------------
# Dimensionality estimation methods
# ---------------------------------------------------------------------------

def broken_stick(n_components: int) -> np.ndarray:
    """Expected eigenvalue proportions under the broken stick (null) model.
    p_k = (1/p) * sum_{i=k}^{p} 1/i  for k = 1..p."""
    p = n_components
    expected = np.zeros(p)
    for k in range(p):
        expected[k] = np.sum(1.0 / np.arange(k + 1, p + 1))
    expected /= p
    return expected


def estimate_dim_broken_stick(var_explained: np.ndarray) -> int:
    """Number of components whose variance proportion exceeds the broken stick
    expectation."""
    bs = broken_stick(len(var_explained))
    significant = var_explained > bs
    # Count leading run of True
    dim = 0
    for s in significant:
        if s:
            dim += 1
        else:
            break
    return max(dim, 1)


def parallel_analysis(X: np.ndarray, n_shuffles: int = N_PARALLEL_SHUFFLES,
                      rng: Optional[np.random.Generator] = None,
                      max_k: int = 300
                      ) -> Tuple[int, np.ndarray]:
    """Parallel analysis: shuffle each column independently, compute truncated SVD,
    average the eigenvalues. Return (estimated_dim, shuffled_eigenvalues).
    Dimensionality = number of real eigenvalues exceeding the 95th-percentile
    of the shuffled distribution."""
    from scipy.sparse.linalg import svds
    if rng is None:
        rng = np.random.default_rng(SEED)
    n, p = X.shape
    k = min(max_k, min(n, p) - 1)
    X_centered = X - X.mean(axis=0, keepdims=True)

    # Real eigenvalues (truncated)
    _, s_real, _ = svds(X_centered, k=k, which='LM')
    s_real = s_real[::-1]
    eig_real = (s_real ** 2) / (n - 1)

    # Shuffled eigenvalues
    eig_shuffled_all = np.zeros((n_shuffles, k))
    for i in range(n_shuffles):
        X_shuf = X_centered.copy()
        for col in range(p):
            rng.shuffle(X_shuf[:, col])
        _, s_shuf, _ = svds(X_shuf, k=k, which='LM')
        s_shuf = s_shuf[::-1]
        eig_shuffled_all[i] = (s_shuf ** 2) / (n - 1)
        print(f"    parallel analysis shuffle {i + 1}/{n_shuffles}", flush=True)

    eig_shuffled_95 = np.percentile(eig_shuffled_all, 95, axis=0)

    # Dimensionality: leading run where real > shuffled 95th pct
    dim = 0
    for r, s in zip(eig_real, eig_shuffled_95):
        if r > s:
            dim += 1
        else:
            break
    return max(dim, 1), eig_shuffled_95


def estimate_dim_elbow(eigenvalues: np.ndarray, max_k: int = 200) -> int:
    """Elbow detection via maximum second derivative of eigenvalue curve.
    Uses the log-eigenvalue spectrum for better sensitivity."""
    k = min(len(eigenvalues), max_k)
    log_eig = np.log(eigenvalues[:k] + 1e-30)
    # Second derivative (discrete)
    d2 = np.diff(log_eig, n=2)
    # The elbow is where the curvature is maximal (most positive second derivative
    # after the initial steep drop). Skip the first few components.
    start = 2
    if len(d2) <= start:
        return 1
    idx = start + np.argmax(d2[start:])
    return int(idx + 1)  # +1 because diff reduces length


# ---------------------------------------------------------------------------
# MELODIC-style consensus subsampling
# ---------------------------------------------------------------------------

def consensus_dimensionality(X: np.ndarray,
                             n_subsamples: int = N_CONSENSUS_SUBSAMPLES,
                             frac: float = CONSENSUS_FRAC,
                             rng: Optional[np.random.Generator] = None
                             ) -> Tuple[np.ndarray, float, float]:
    """Run PCA on random subsamples, estimate dim via broken stick each time.
    Return (array of dims, median, IQR)."""
    if rng is None:
        rng = np.random.default_rng(SEED + 1)
    n = X.shape[0]
    k = max(int(n * frac), 10)
    dims = []
    for i in range(n_subsamples):
        idx = rng.choice(n, size=k, replace=False)
        Xsub = X[idx]
        _, ve = compute_eigenspectrum(Xsub)
        d = estimate_dim_broken_stick(ve)
        dims.append(d)
        if (i + 1) % 10 == 0:
            print(f"    consensus subsample {i + 1}/{n_subsamples}")
    dims = np.array(dims)
    return dims, float(np.median(dims)), float(sp_stats.iqr(dims))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_eigenspectrum(results: Dict, outpath: Path):
    """Overlaid eigenvalue decay curves per session."""
    fig, ax = plt.subplots(figsize=(10, 6))
    max_k = 150
    for label, res in results.items():
        eig = res["eigenvalues"]
        k = min(len(eig), max_k)
        color = SESSION_COLORS.get(label, "#333333")
        ax.semilogy(range(1, k + 1), eig[:k], label=label, color=color)

    ax.set_xlabel("Component")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Eigenspectrum by Session")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(str(outpath))
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_cumvar(results: Dict, outpath: Path):
    """Cumulative variance explained per session."""
    fig, ax = plt.subplots(figsize=(10, 6))
    max_k = 300
    for label, res in results.items():
        ve = res["var_explained"]
        k = min(len(ve), max_k)
        cumvar = np.cumsum(ve[:k])
        color = SESSION_COLORS.get(label, "#333333")
        ax.plot(range(1, k + 1), cumvar * 100, label=label, color=color)

    # Reference lines
    for pct in [80, 90, 95]:
        ax.axhline(pct, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.text(max_k - 5, pct + 0.5, f"{pct}%", fontsize=9, color="gray",
                ha="right")

    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Variance Explained (%)")
    ax.set_title("Cumulative Variance Explained by Session")
    ax.set_ylim(0, 101)
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(str(outpath))
    plt.close(fig)
    print(f"  Saved {outpath}")


def plot_consensus(results: Dict, outpath: Path):
    """Stability plot of consensus dimensionality across subsamples."""
    sessions = [k for k in results if "consensus_dims" in results[k]]
    if not sessions:
        print("  No consensus data to plot.")
        return

    fig, axes = plt.subplots(1, len(sessions), figsize=(5 * len(sessions), 5),
                              squeeze=False)
    for i, label in enumerate(sessions):
        ax = axes[0, i]
        dims = results[label]["consensus_dims"]
        ax.hist(dims, bins=range(int(dims.min()) - 1, int(dims.max()) + 3),
                color=SESSION_COLORS.get(label, "#333333"), edgecolor="white",
                alpha=0.8)
        med = results[label]["consensus_median"]
        ax.axvline(med, color="red", linestyle="--", linewidth=2,
                   label=f"median = {med:.0f}")
        ax.set_xlabel("Estimated Dimensionality")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}")
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Consensus Dimensionality (MELODIC-style Subsampling)", y=1.02)
    fig.tight_layout()
    fig.savefig(str(outpath))
    plt.close(fig)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def save_summary_csv(results: Dict, outpath: Path):
    """Write dimensionality_summary.csv."""
    rows = []
    for label, res in results.items():
        row = {
            "session": label,
            "n_trials": res["n_trials"],
            "n_voxels": res["n_voxels"],
            "dim_elbow": res.get("dim_elbow", ""),
            "dim_broken_stick": res.get("dim_broken_stick", ""),
            "dim_parallel_analysis": res.get("dim_parallel_analysis", ""),
            "consensus_median": res.get("consensus_median", ""),
            "consensus_iqr": res.get("consensus_iqr", ""),
            "var_at_50_components": f"{res.get('var_at_50', 0):.1f}%",
            "var_at_100_components": f"{res.get('var_at_100', 0):.1f}%",
            "var_at_200_components": f"{res.get('var_at_200', 0):.1f}%",
            "components_for_80pct": res.get("k80", ""),
            "components_for_90pct": res.get("k90", ""),
            "components_for_95pct": res.get("k95", ""),
        }
        rows.append(row)

    fieldnames = list(rows[0].keys())
    with open(str(outpath), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved {outpath}")


# ---------------------------------------------------------------------------
# Main analysis pipeline
# ---------------------------------------------------------------------------

def analyze_session(label: str, X: np.ndarray, rng: np.random.Generator
                    ) -> Dict:
    """Full dimensionality analysis for one dataset."""
    n, p = X.shape
    print(f"\n--- Analyzing {label} ({n} trials x {p} voxels) ---")

    # Eigenspectrum
    print("  Computing eigenspectrum (SVD)...")
    eigenvalues, var_explained = compute_eigenspectrum(X)

    # Cumulative variance milestones
    cumvar = np.cumsum(var_explained)
    var_at_50 = cumvar[min(49, len(cumvar) - 1)] * 100
    var_at_100 = cumvar[min(99, len(cumvar) - 1)] * 100
    var_at_200 = cumvar[min(199, len(cumvar) - 1)] * 100
    k80 = int(np.searchsorted(cumvar, 0.80) + 1)
    k90 = int(np.searchsorted(cumvar, 0.90) + 1)
    k95 = int(np.searchsorted(cumvar, 0.95) + 1)
    print(f"  Variance: 50 PCs={var_at_50:.1f}%, 100 PCs={var_at_100:.1f}%, "
          f"200 PCs={var_at_200:.1f}%")
    print(f"  Components for 80%={k80}, 90%={k90}, 95%={k95}")

    # Elbow
    dim_elbow = estimate_dim_elbow(eigenvalues)
    print(f"  Elbow (log-eigenvalue curvature): {dim_elbow}")

    # Broken stick
    dim_bs = estimate_dim_broken_stick(var_explained)
    print(f"  Broken stick: {dim_bs}")

    # Parallel analysis
    print("  Running parallel analysis...")
    dim_pa, shuffled_eig = parallel_analysis(X, rng=rng)
    print(f"  Parallel analysis: {dim_pa}")

    # Consensus
    print("  Running consensus subsampling...")
    cons_dims, cons_med, cons_iqr = consensus_dimensionality(X, rng=rng)
    print(f"  Consensus: median={cons_med:.0f}, IQR={cons_iqr:.1f}")

    return {
        "n_trials": n,
        "n_voxels": p,
        "eigenvalues": eigenvalues,
        "var_explained": var_explained,
        "dim_elbow": dim_elbow,
        "dim_broken_stick": dim_bs,
        "dim_parallel_analysis": dim_pa,
        "shuffled_eig_95": shuffled_eig,
        "consensus_dims": cons_dims,
        "consensus_median": cons_med,
        "consensus_iqr": cons_iqr,
        "var_at_50": var_at_50,
        "var_at_100": var_at_100,
        "var_at_200": var_at_200,
        "k80": k80,
        "k90": k90,
        "k95": k95,
    }


def main():
    rng = np.random.default_rng(SEED)

    # Load data
    session_data = load_all_data()

    # Create output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Analyze each session
    results = {}
    for ses in SESSIONS:
        if ses in session_data:
            results[ses] = analyze_session(ses, session_data[ses], rng)

    # Combined across all sessions
    if len(session_data) > 1:
        print("\nCombining all sessions...")
        all_betas = np.concatenate(list(session_data.values()), axis=0)
        results["all"] = analyze_session("all_sessions", all_betas, rng)

    # Generate plots
    print("\n--- Generating figures ---")
    plot_eigenspectrum(results, OUT_DIR / "eigenspectrum_by_session.png")
    plot_cumvar(results, OUT_DIR / "cumvar_by_session.png")
    plot_consensus(results, OUT_DIR / "consensus_dimensionality.png")

    # Summary CSV
    save_summary_csv(results, OUT_DIR / "dimensionality_summary.csv")

    # Print final summary
    print("\n" + "=" * 70)
    print("DIMENSIONALITY ESTIMATION SUMMARY")
    print("=" * 70)
    header = f"{'Session':<12} {'N':>5} {'Elbow':>6} {'BrkStk':>7} {'ParAnl':>7} {'Consns':>7} {'80%':>5} {'90%':>5} {'95%':>5}"
    print(header)
    print("-" * len(header))
    for label, res in results.items():
        print(f"{label:<12} {res['n_trials']:>5} "
              f"{res['dim_elbow']:>6} {res['dim_broken_stick']:>7} "
              f"{res['dim_parallel_analysis']:>7} "
              f"{res['consensus_median']:>7.0f} "
              f"{res['k80']:>5} {res['k90']:>5} {res['k95']:>5}")
    print("=" * 70)
    print(f"\nAll outputs saved to {OUT_DIR}/")
    print("Figures: eigenspectrum_by_session.png, cumvar_by_session.png, "
          "consensus_dimensionality.png")
    print("Table:   dimensionality_summary.csv")


if __name__ == "__main__":
    main()
