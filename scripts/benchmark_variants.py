#!/usr/bin/env python3
"""
MindEye Preprocessing Variant Benchmark

Replays ses-06 data through each variant, logs to trackio,
and saves results to /data/derivatives/mindeye_variants/.
"""

import os
import sys
import time
import json
import csv
import argparse
from pathlib import Path

# Ensure GPU — refuse to run on CPU
import jax
if jax.default_backend() != "gpu":
    print(f"ERROR: JAX backend is '{jax.default_backend()}', not 'gpu'.")
    print("Install jax[cuda12] or run inside an NGC container.")
    sys.exit(1)
from typing import List, Dict, Optional

import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_glm_variants import (
    VariantConfig,
    RTPreprocessingVariant,
    VARIANT_REGISTRY,
    create_variant,
    load_brain_mask,
    load_union_mask,
    apply_masks,
    load_hrf_indices,
    resample_hrf,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_events(events_dir: str, session: str, run: int) -> pd.DataFrame:
    """Load events TSV for a given session and run."""
    path = Path(events_dir) / f"sub-005_{session}_task-C_run-{run:02d}_events.tsv"
    return pd.read_csv(path, sep="\t")


def load_tr_labels(events_dir: str, session: str, run: int) -> pd.DataFrame:
    """Load TR labels CSV for a given session and run."""
    path = Path(events_dir) / f"sub-005_{session}_task-C_run-{run:02d}_tr_labels.csv"
    return pd.read_csv(path)


def get_mc_volumes(mc_dir: str, session: str, run: int) -> List[str]:
    """Get sorted list of motion-corrected volume paths for a run."""
    prefix = f"{session.replace('ses-', 'ses-')}_run-{run:02d}_"
    volumes = sorted([
        str(p) for p in Path(mc_dir).glob(f"{prefix}*_mc_boldres.nii.gz")
    ])
    return volumes


def load_volume_masked(nifti_path: str, brain_mask_flat: np.ndarray,
                       union_mask: np.ndarray) -> np.ndarray:
    """Load a NIfTI volume and apply two-stage masking → (8627,)."""
    img = nib.load(nifti_path)
    vol_3d = img.get_fdata()
    return apply_masks(vol_3d, brain_mask_flat, union_mask)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_variant_benchmark(
    variant: RTPreprocessingVariant,
    session: str,
    run: int,
    config: VariantConfig,
    brain_mask_flat: np.ndarray,
    union_mask: np.ndarray,
    use_trackio: bool = True,
) -> Dict:
    """
    Run a single variant on one run of data.

    Returns dict with timing and beta info.
    """
    # Load events
    events_df = load_events(config.events_dir, session, run)
    events_df["onset"] = events_df["onset"].astype(float)
    run_start_time = events_df["onset"].iloc[0]
    events_df["onset"] -= run_start_time
    onsets = events_df["onset"].values

    # Load TR labels
    tr_labels = load_tr_labels(config.events_dir, session, run)

    # Get volume paths
    vol_paths = get_mc_volumes(config.mc_volumes_dir, session, run)
    n_trs = len(vol_paths)
    print(f"  {variant.name}: {n_trs} TRs, {len(onsets)} events")

    # Initialize trackio
    trackio_run = None
    if use_trackio:
        try:
            import trackio
            trackio_run = trackio.init(
                project="mindeye-variants",
                name=f"variant_{variant.name}_{session}_run-{run:02d}",
                group=session,
                config={
                    "variant": variant.name,
                    "n_voxels": config.n_voxels,
                    "tr": config.tr,
                    "session": session,
                    "run": run,
                    "n_trs": n_trs,
                },
            )
        except Exception as e:
            print(f"  Warning: trackio init failed: {e}")
            use_trackio = False

    # Accumulate timeseries
    timeseries = np.zeros((config.n_voxels, 0), dtype=np.float32)
    timing_records = []
    stimulus_trial_counter = 0
    all_betas = []

    for tr_idx in range(n_trs):
        # Load and mask volume
        vol_masked = load_volume_masked(vol_paths[tr_idx], brain_mask_flat, union_mask)
        timeseries = np.column_stack([timeseries, vol_masked])

        # Check if this TR has a stimulus (not blank)
        if tr_idx < len(tr_labels):
            label = str(tr_labels.iloc[tr_idx].get("tr_label_hrf", "blank"))
        else:
            label = "blank"

        if label != "blank":
            # Process this TR
            t0 = time.time()
            try:
                raw_beta = variant.process_tr(
                    timeseries, tr_idx, onsets, probe_trial=stimulus_trial_counter
                )
                elapsed = time.time() - t0

                # Z-score
                z_beta = variant.z_score_beta(raw_beta)
                all_betas.append(z_beta)
                variant._betas.append(z_beta)
                variant._timing.append(elapsed)

                timing_records.append({
                    "tr_index": tr_idx,
                    "trial": stimulus_trial_counter,
                    "wall_time_s": elapsed,
                    "label": label,
                })

                # Log to trackio
                if use_trackio and trackio_run:
                    try:
                        import trackio
                        trackio.log({
                            "tr": tr_idx,
                            "trial": stimulus_trial_counter,
                            "wall_time_s": elapsed,
                            "beta_mean": float(z_beta.mean()),
                            "beta_std": float(z_beta.std()),
                            "beta_norm": float(np.linalg.norm(z_beta)),
                        }, step=tr_idx)
                    except Exception:
                        pass

                if (stimulus_trial_counter + 1) % 10 == 0:
                    print(f"    TR {tr_idx}: trial {stimulus_trial_counter}, "
                          f"time={elapsed:.3f}s, beta_norm={np.linalg.norm(z_beta):.2f}")

            except Exception as e:
                elapsed = time.time() - t0
                print(f"    TR {tr_idx}: ERROR - {e}")
                timing_records.append({
                    "tr_index": tr_idx,
                    "trial": stimulus_trial_counter,
                    "wall_time_s": elapsed,
                    "label": label,
                    "error": str(e),
                })

            stimulus_trial_counter += 1

    if use_trackio and trackio_run:
        try:
            import trackio
            trackio.finish()
        except Exception:
            pass

    return {
        "variant": variant.name,
        "session": session,
        "run": run,
        "n_trs": n_trs,
        "n_trials": stimulus_trial_counter,
        "timing": timing_records,
        "mean_time": np.mean([r["wall_time_s"] for r in timing_records]) if timing_records else 0,
        "max_time": np.max([r["wall_time_s"] for r in timing_records]) if timing_records else 0,
    }


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_variant_results(variant: RTPreprocessingVariant, results: Dict,
                         output_base: str):
    """Save variant results to disk."""
    out_dir = Path(output_base) / f"variant_{variant.name}"
    variant.save_results(str(out_dir))

    # Save metrics
    metrics = {
        "variant": results["variant"],
        "session": results["session"],
        "run": results["run"],
        "n_trials": results["n_trials"],
        "mean_time_s": results["mean_time"],
        "max_time_s": results["max_time"],
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved results to {out_dir}")


def save_comparison(all_results: List[Dict], output_base: str):
    """Generate comparison summary across all variants."""
    comp_dir = Path(output_base) / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in all_results:
        rows.append({
            "variant": r["variant"],
            "session": r["session"],
            "run": r["run"],
            "n_trials": r["n_trials"],
            "mean_time_s": f"{r['mean_time']:.4f}",
            "max_time_s": f"{r['max_time']:.4f}",
        })

    df = pd.DataFrame(rows)
    df.to_csv(comp_dir / "metrics_summary.csv", index=False)
    print(f"\nComparison saved to {comp_dir / 'metrics_summary.csv'}")
    print(df.to_string())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MindEye variant benchmark")
    parser.add_argument("--variants", nargs="+",
                        default=["a_baseline", "c_pervoxel_hrf", "d_bayesian"],
                        choices=list(VARIANT_REGISTRY.keys()),
                        help="Variants to benchmark")
    parser.add_argument("--session", default="ses-06", help="Session to process")
    parser.add_argument("--runs", nargs="+", type=int, default=[1],
                        help="Run numbers to process")
    parser.add_argument("--no-trackio", action="store_true",
                        help="Disable trackio logging")
    parser.add_argument("--output", default="/data/derivatives/mindeye_variants",
                        help="Output directory")
    args = parser.parse_args()

    config = VariantConfig(output_base=args.output)

    print("Loading masks...")
    brain_mask_flat = load_brain_mask(config.brain_mask_path)
    union_mask = load_union_mask(config.union_mask_path)
    print(f"  Brain mask: {brain_mask_flat.sum()} voxels")
    print(f"  Union mask: {union_mask.sum()} voxels")

    all_results = []

    for variant_name in args.variants:
        print(f"\n{'='*60}")
        print(f"Variant: {variant_name}")
        print(f"{'='*60}")

        variant = create_variant(variant_name, config=config)

        # Precompute (variant-specific setup)
        print("  Precomputing...")
        t0 = time.time()
        if variant_name in ("c_pervoxel_hrf",):
            variant.precompute(brain_mask_flat=brain_mask_flat, union_mask=union_mask)
        elif variant_name in ("e_spatial",):
            variant.precompute(brain_mask_flat=brain_mask_flat, union_mask=union_mask)
        elif variant_name in ("cd_combined",):
            variant.precompute(brain_mask_flat=brain_mask_flat, union_mask=union_mask)
        elif variant_name in ("d_bayesian", "f_logsig"):
            # Try to load training betas for prior
            try:
                train_betas = np.load(config.training_betas_path)
                if train_betas.shape[1] != config.n_voxels:
                    print(f"  Training betas shape mismatch ({train_betas.shape[1]} vs {config.n_voxels}), using defaults")
                    variant.precompute()
                else:
                    variant.precompute(training_betas=train_betas)
            except Exception as e:
                print(f"  Could not load training betas: {e}, using defaults")
                variant.precompute()
        else:
            variant.precompute()
        print(f"  Precompute done in {time.time() - t0:.2f}s")

        # Warmup
        print("  JIT warmup...")
        variant.warmup()

        # Run benchmark for each run
        for run_num in args.runs:
            print(f"\n  Processing {args.session} run-{run_num:02d}...")
            try:
                result = run_variant_benchmark(
                    variant=variant,
                    session=args.session,
                    run=run_num,
                    config=config,
                    brain_mask_flat=brain_mask_flat,
                    union_mask=union_mask,
                    use_trackio=not args.no_trackio,
                )
                save_variant_results(variant, result, args.output)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR running {variant_name} on run {run_num}: {e}")
                import traceback
                traceback.print_exc()

    if all_results:
        save_comparison(all_results, args.output)


if __name__ == "__main__":
    main()
