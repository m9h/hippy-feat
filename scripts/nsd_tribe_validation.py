#!/usr/bin/env python3
"""NSD × TRIBEv2 Validation: predicted vs actual fMRI representations.

Compares TRIBEv2-predicted BOLD responses to real NSD betas across
8 subjects using RSA, noise ceiling estimation, and FC comparison.

Pipeline:
    1. Load real NSD betas (nsdgeneral-masked, sessions 01-03)
    2. Generate TRIBEv2 predictions (or synthetic fallback)
    3. Compare predicted vs actual:
       a. RDM correlation (Spearman on upper triangle)
       b. Noise ceiling (split-half reliability)
       c. Category selectivity (face/place dissociation)
       d. FC matrix similarity (Wasserstein distance)
    4. Generate publication figures

Targets DGX Spark:
    /data/3t/nsd_multisubject/{subj01..subj08}/betas_session{01..03}.nii.gz
    /data/3t/nsd_multisubject/{subj01..subj08}_nsdgeneral.nii.gz

Falls back to synthetic data when NSD is not available (e.g. local dev).

Usage:
    python scripts/nsd_tribe_validation.py [--subjects subj01 subj02] [--synthetic]
    python scripts/nsd_tribe_validation.py --all-subjects

Output:
    /data/derivatives/tribe_validation/  (DGX)
    figures/tribe_validation/            (local fallback)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

from jaxoccoli.nsd import (
    rdm_from_betas,
    compare_rdms,
    noise_ceiling_r,
    category_selectivity,
    load_nsd_betas,
)
from jaxoccoli.covariance import corr
from jaxoccoli.transport import wasserstein_fc_distance

# Try matplotlib (not available in all containers)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NSD_DIR = Path("/data/3t/nsd_multisubject")
OUT_DIR_DGX = Path("/data/derivatives/tribe_validation")
OUT_DIR_LOCAL = Path("figures/tribe_validation")

SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
SESSIONS = ["01", "02", "03"]

# NSD COCO categories (simplified — real NSD has 80 COCO categories;
# we group into supercategories for the validation)
COCO_SUPERCATEGORIES = ["person", "animal", "vehicle", "outdoor", "food",
                        "furniture", "electronic", "kitchen", "indoor"]


# ---------------------------------------------------------------------------
# Synthetic fallback
# ---------------------------------------------------------------------------

def generate_synthetic_betas(
    n_trials: int = 300,
    n_voxels: int = 5000,
    n_categories: int = 5,
    snr: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, list[str]]:
    """Generate synthetic betas with embedded category structure.

    Returns (betas, categories) simulating what TRIBEv2 would predict
    for an NSD session with category-specific activation patterns.
    """
    rng = np.random.RandomState(seed)

    categories = [COCO_SUPERCATEGORIES[i % n_categories] for i in range(n_trials)]
    rng.shuffle(categories)

    # Category templates
    templates = {}
    block = n_voxels // n_categories
    for i, cat in enumerate(COCO_SUPERCATEGORIES[:n_categories]):
        t = rng.randn(n_voxels).astype(np.float32) * 0.1
        t[i * block:(i + 1) * block] += snr
        templates[cat] = t

    # Per-trial betas = template + per-image variation + noise
    betas = np.zeros((n_trials, n_voxels), dtype=np.float32)
    for i, cat in enumerate(categories):
        image_offset = rng.randn(n_voxels).astype(np.float32) * 0.2
        noise = rng.randn(n_voxels).astype(np.float32) * 0.5
        betas[i] = templates[cat] + image_offset + noise

    return betas, categories


def generate_synthetic_predicted_betas(
    actual_betas: np.ndarray,
    categories: list[str],
    prediction_noise: float = 0.8,
    seed: int = 42,
) -> np.ndarray:
    """Simulate TRIBEv2 predicted betas as a noisy version of actuals.

    In a real validation, this would be replaced by:
        model = TribeModel.from_pretrained('facebook/tribev2')
        for img in nsd_images:
            events = model.get_events_dataframe(video_path=img)
            preds, _ = model.predict(events=events)
            predicted_betas.append(preds.mean(axis=0))  # avg over TRs
    """
    rng = np.random.RandomState(seed)
    noise = rng.randn(*actual_betas.shape).astype(np.float32) * prediction_noise
    # Predicted = scaled actual + noise (simulates imperfect encoding model)
    predicted = actual_betas * 0.6 + noise
    return predicted


# ---------------------------------------------------------------------------
# Validation analysis
# ---------------------------------------------------------------------------

def validate_subject(
    actual_betas: np.ndarray,
    predicted_betas: np.ndarray,
    categories: list[str],
    subject: str = "unknown",
) -> dict:
    """Run full validation suite on one subject.

    Returns dict of metrics.
    """
    actual_jax = jnp.array(actual_betas)
    predicted_jax = jnp.array(predicted_betas)

    results = {"subject": subject}

    # --- RDM comparison ---
    rdm_actual = rdm_from_betas(actual_jax)
    rdm_predicted = rdm_from_betas(predicted_jax)
    results["rdm_spearman"] = float(compare_rdms(rdm_predicted, rdm_actual))

    # --- Noise ceiling ---
    nc_lower, nc_upper = noise_ceiling_r(actual_jax, n_splits=20, seed=0)
    results["noise_ceiling_lower"] = float(nc_lower)
    results["noise_ceiling_upper"] = float(nc_upper)

    # Predicted-to-actual as fraction of noise ceiling
    if float(nc_upper) > 0:
        results["fraction_of_ceiling"] = results["rdm_spearman"] / float(nc_upper)
    else:
        results["fraction_of_ceiling"] = 0.0

    # --- Category selectivity ---
    sel_actual = category_selectivity(actual_jax, categories)
    sel_predicted = category_selectivity(predicted_jax, categories)

    # Correlation of selectivity maps per category
    sel_corrs = {}
    for cat in sel_actual:
        r = float(jnp.corrcoef(
            jnp.stack([sel_actual[cat], sel_predicted[cat]])
        )[0, 1])
        sel_corrs[cat] = r
    results["selectivity_corr"] = sel_corrs
    results["mean_selectivity_corr"] = np.mean(list(sel_corrs.values()))

    # --- FC comparison ---
    # Beta-series FC (parcel-free: use top PCs as pseudo-parcels)
    n_pc = min(50, actual_betas.shape[1])
    # Quick PCA via SVD for FC
    U_a, _, _ = jnp.linalg.svd(actual_jax - actual_jax.mean(0), full_matrices=False)
    U_p, _, _ = jnp.linalg.svd(predicted_jax - predicted_jax.mean(0), full_matrices=False)
    fc_actual = corr(U_a[:, :n_pc].T)
    fc_predicted = corr(U_p[:, :n_pc].T)

    try:
        w_dist = float(wasserstein_fc_distance(fc_actual, fc_predicted))
        results["fc_wasserstein"] = w_dist
    except Exception:
        results["fc_wasserstein"] = float("nan")

    # Direct FC correlation
    from jaxoccoli.nsd import upper_triangle
    fc_r = float(jnp.corrcoef(
        jnp.stack([upper_triangle(fc_actual), upper_triangle(fc_predicted)])
    )[0, 1])
    results["fc_correlation"] = fc_r

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_validation_results(
    all_results: list[dict],
    out_dir: Path,
):
    """Generate publication figures from validation results."""
    if not HAS_MPL:
        print("matplotlib not available — skipping figures")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = [r["subject"] for r in all_results]
    rdm_rhos = [r["rdm_spearman"] for r in all_results]
    nc_lowers = [r["noise_ceiling_lower"] for r in all_results]
    nc_uppers = [r["noise_ceiling_upper"] for r in all_results]
    frac_ceilings = [r["fraction_of_ceiling"] for r in all_results]

    plt.rcParams.update({
        "font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
        "figure.facecolor": "white",
    })

    # --- Fig 1: RDM correlation vs noise ceiling ---
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(subjects))
    width = 0.25

    ax.bar(x - width, nc_uppers, width, label="Noise ceiling (upper)", color="#2196F3", alpha=0.7)
    ax.bar(x, rdm_rhos, width, label="TRIBEv2 prediction", color="#4CAF50", alpha=0.9)
    ax.bar(x + width, nc_lowers, width, label="Noise ceiling (lower)", color="#2196F3", alpha=0.4)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Spearman ρ (RDM)")
    ax.set_title("TRIBEv2 Predicted vs Actual NSD Representations")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects, rotation=45)
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.1)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_rdm_vs_noise_ceiling.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_rdm_vs_noise_ceiling.png")

    # --- Fig 2: Category selectivity correlation ---
    all_cats = sorted({cat for r in all_results for cat in r["selectivity_corr"]})
    fig, ax = plt.subplots(figsize=(8, 5))

    for i, r in enumerate(all_results):
        cats = sorted(r["selectivity_corr"].keys())
        vals = [r["selectivity_corr"][c] for c in cats]
        ax.scatter([i] * len(vals), vals, alpha=0.6, s=30)
        ax.scatter([i], [r["mean_selectivity_corr"]], marker="D", s=80,
                   color="red", zorder=5)

    ax.set_xlabel("Subject")
    ax.set_ylabel("Selectivity map correlation (predicted vs actual)")
    ax.set_title("Category Selectivity Preservation")
    ax.set_xticks(range(len(subjects)))
    ax.set_xticklabels(subjects, rotation=45)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_selectivity_by_category.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_selectivity_by_category.png")

    # --- Fig 3: FC correlation ---
    fc_corrs = [r["fc_correlation"] for r in all_results]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(subjects, fc_corrs, color="#FF9800", alpha=0.8)
    ax.set_ylabel("Pearson r (FC upper triangle)")
    ax.set_title("Beta-Series FC Similarity")
    ax.set_xticklabels(subjects, rotation=45)

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_fc_correlation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved fig3_fc_correlation.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NSD × TRIBEv2 Validation")
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Subject IDs (default: all 8)")
    parser.add_argument("--all-subjects", action="store_true")
    parser.add_argument("--synthetic", action="store_true",
                        help="Force synthetic data (skip NSD loading)")
    parser.add_argument("--n-trials", type=int, default=300,
                        help="Trials per subject for synthetic mode")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--encoder", choices=["synthetic", "tribev2", "raramuri"],
                        default="synthetic",
                        help=("Prediction backend. 'tribev2' and 'raramuri' require "
                              "video stimuli; NSD ships static COCO images, so both "
                              "currently fall back to synthetic and emit a warning. "
                              "Left wired so a future NSD-movies extension or a "
                              "single-frame-video wrapper can swap in without "
                              "touching this script."))
    parser.add_argument("--raramuri-server",
                        default="http://localhost:8765",
                        help="Raramuri hot server URL (only used when --encoder raramuri)")
    args = parser.parse_args()

    if args.encoder in ("tribev2", "raramuri"):
        print(f"WARNING: --encoder {args.encoder} requested, but NSD trials are "
              f"static images not video. Falling back to synthetic predictions. "
              f"Use scripts/raramuri_benchmark.py for real Raramuri/TRIBEv2 "
              f"speedup measurements on video clips.")

    subjects = args.subjects or (SUBJECTS if args.all_subjects else SUBJECTS[:3])

    # Output directory
    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif OUT_DIR_DGX.parent.exists():
        out_dir = OUT_DIR_DGX
    else:
        out_dir = OUT_DIR_LOCAL
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 64)
    print("NSD × TRIBEv2 Validation")
    print("=" * 64)
    print(f"Subjects:   {subjects}")
    print(f"Output:     {out_dir}")
    print(f"Synthetic:  {args.synthetic or not NSD_DIR.exists()}")

    use_synthetic = args.synthetic or not NSD_DIR.exists()
    all_results = []

    for subj in subjects:
        print(f"\n--- {subj} ---")
        t0 = time.time()

        # Load or generate actual betas
        if use_synthetic:
            seed = hash(subj) % 2**31
            actual_betas, categories = generate_synthetic_betas(
                n_trials=args.n_trials, n_voxels=5000,
                n_categories=5, snr=1.0, seed=seed,
            )
            print(f"  Synthetic: {actual_betas.shape}")
        else:
            actual_betas = load_nsd_betas(NSD_DIR, subj, SESSIONS)
            if actual_betas is None:
                print(f"  Skipping {subj}: data not found")
                continue
            print(f"  Loaded: {actual_betas.shape}")
            # For real NSD, we'd load COCO categories from the stim info
            # Fallback: assign random categories for now
            n = actual_betas.shape[0]
            rng = np.random.RandomState(hash(subj) % 2**31)
            categories = [COCO_SUPERCATEGORIES[i % 5] for i in range(n)]
            rng.shuffle(categories)

        # Generate predicted betas (synthetic TRIBEv2 fallback)
        # With real TRIBEv2:
        #   model = TribeModel.from_pretrained('facebook/tribev2')
        #   predicted = predict_nsd_images(model, nsd_image_paths)
        predicted_betas = generate_synthetic_predicted_betas(
            actual_betas, categories,
            prediction_noise=0.8, seed=hash(subj) % 2**31 + 1,
        )

        # Validate
        results = validate_subject(actual_betas, predicted_betas, categories, subj)
        elapsed = time.time() - t0

        print(f"  RDM Spearman ρ:       {results['rdm_spearman']:.4f}")
        print(f"  Noise ceiling:        [{results['noise_ceiling_lower']:.4f}, "
              f"{results['noise_ceiling_upper']:.4f}]")
        print(f"  Fraction of ceiling:  {results['fraction_of_ceiling']:.4f}")
        print(f"  Mean selectivity r:   {results['mean_selectivity_corr']:.4f}")
        print(f"  FC correlation:       {results['fc_correlation']:.4f}")
        print(f"  Time:                 {elapsed:.1f}s")

        all_results.append(results)

    # Summary
    if all_results:
        print(f"\n{'='*64}")
        print("Summary across subjects:")
        rdm_rhos = [r["rdm_spearman"] for r in all_results]
        fracs = [r["fraction_of_ceiling"] for r in all_results]
        sel_rs = [r["mean_selectivity_corr"] for r in all_results]
        fc_rs = [r["fc_correlation"] for r in all_results]

        print(f"  RDM ρ:              {np.mean(rdm_rhos):.4f} ± {np.std(rdm_rhos):.4f}")
        print(f"  % noise ceiling:    {np.mean(fracs)*100:.1f}% ± {np.std(fracs)*100:.1f}%")
        print(f"  Selectivity r:      {np.mean(sel_rs):.4f} ± {np.std(sel_rs):.4f}")
        print(f"  FC correlation:     {np.mean(fc_rs):.4f} ± {np.std(fc_rs):.4f}")

        # Figures
        print(f"\nGenerating figures...")
        plot_validation_results(all_results, out_dir)

        # Save CSV
        import csv
        csv_path = out_dir / "validation_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "subject", "rdm_spearman", "noise_ceiling_lower",
                "noise_ceiling_upper", "fraction_of_ceiling",
                "mean_selectivity_corr", "fc_correlation", "fc_wasserstein",
            ])
            writer.writeheader()
            for r in all_results:
                row = {k: r[k] for k in writer.fieldnames}
                writer.writerow(row)
        print(f"  Saved {csv_path}")

    print(f"\n{'='*64}")
    print("To run with real TRIBEv2 predictions on DGX Spark:")
    print("  pip install tribev2")
    print("  # Replace generate_synthetic_predicted_betas with model.predict()")
    print("  python scripts/nsd_tribe_validation.py --all-subjects")
    print("=" * 64)


if __name__ == "__main__":
    main()
