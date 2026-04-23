#!/usr/bin/env python3
"""
Generate 8627-voxel training priors for Variant D/G shrinkage.

Existing real_time_betas/*.npy files are at a 2792-voxel mask that
pre-dates the current 8627-voxel union mask. Runs VariantA_Baseline over
the ses-01 raw per-TR volumes (/data/3t/derivatives/vols/sub-005/ses-XX/)
and aggregates probe betas across runs, producing a
(n_trials, 8627) npy that can be loaded via VariantConfig.training_betas_path.

Note: ses-01/02 volumes here are NOT motion-corrected. The resulting priors
will include motion-driven variance, which is a mildly conservative prior —
acceptable for shrinkage demonstrations and decoder benchmarking.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_glm_variants import (
    VariantConfig,
    VariantA_Baseline,
    load_brain_mask,
    load_union_mask,
    apply_masks,
)


def load_events(events_dir: str, session: str, run: int) -> pd.DataFrame:
    path = Path(events_dir) / f"sub-005_{session}_task-C_run-{run:02d}_events.tsv"
    return pd.read_csv(path, sep="\t")


def load_tr_labels(events_dir: str, session: str, run: int) -> pd.DataFrame:
    path = Path(events_dir) / f"sub-005_{session}_task-C_run-{run:02d}_tr_labels.csv"
    return pd.read_csv(path)


def get_raw_volumes(vols_dir: str, session: str, run: int) -> list:
    pattern = f"sub-005_{session}_task-C_run-{run:02d}_bold_*.nii.gz"
    return sorted(str(p) for p in Path(vols_dir).glob(pattern))


def run_one(session: str, run: int, variant, config, brain_mask_flat, union_mask,
            vols_dir: str) -> np.ndarray:
    events = load_events(config.events_dir, session, run)
    events["onset"] = events["onset"].astype(float)
    events["onset"] -= events["onset"].iloc[0]
    onsets = events["onset"].values

    tr_labels = load_tr_labels(config.events_dir, session, run)
    vol_paths = get_raw_volumes(f"{vols_dir}/{session}", session, run)
    if not vol_paths:
        print(f"  WARN: no volumes found for {session} run-{run:02d}, skipping")
        return np.zeros((0, config.n_voxels), dtype=np.float32)

    n_trs = len(vol_paths)
    timeseries = np.zeros((config.n_voxels, 0), dtype=np.float32)
    betas, stim_trial = [], 0

    for tr_idx in range(n_trs):
        vol_3d = nib.load(vol_paths[tr_idx]).get_fdata()
        vol_masked = apply_masks(vol_3d, brain_mask_flat, union_mask)
        timeseries = np.column_stack([timeseries, vol_masked])

        label = "blank"
        if tr_idx < len(tr_labels):
            label = str(tr_labels.iloc[tr_idx].get("tr_label_hrf", "blank"))

        if label != "blank":
            raw_beta = variant.process_tr(timeseries, tr_idx, onsets,
                                          probe_trial=stim_trial)
            betas.append(raw_beta)
            stim_trial += 1

    print(f"  {session} run-{run:02d}: {stim_trial} probe betas from {n_trs} TRs")
    return np.array(betas, dtype=np.float32) if betas else np.zeros((0, config.n_voxels), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", default=["ses-01"])
    ap.add_argument("--runs", nargs="+", type=int,
                    default=list(range(1, 12)))
    ap.add_argument("--vols-dir", default="/data/3t/derivatives/vols/sub-005")
    ap.add_argument("--output",
                    default="/data/3t/data/real_time_betas/priors_ses-01_a_baseline_8627v.npy")
    args = ap.parse_args()

    config = VariantConfig()
    print("Loading masks…")
    brain_mask_flat = load_brain_mask(config.brain_mask_path)
    union_mask = load_union_mask(config.union_mask_path)
    print(f"  brain={brain_mask_flat.sum()}  union={union_mask.sum()}")

    variant = VariantA_Baseline(config)
    variant.precompute()
    variant.warmup()

    all_betas = []
    for session in args.sessions:
        for run in args.runs:
            t0 = time.time()
            betas = run_one(session, run, variant, config,
                            brain_mask_flat, union_mask, args.vols_dir)
            if betas.size:
                all_betas.append(betas)
            print(f"    elapsed {time.time() - t0:.1f}s")

    if not all_betas:
        print("ERROR: no betas collected")
        sys.exit(1)

    stacked = np.concatenate(all_betas, axis=0).astype(np.float32)
    print(f"\nAggregated shape: {stacked.shape}")
    print(f"Mean: {stacked.mean():.3f}  Std: {stacked.std():.3f}")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, stacked)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
