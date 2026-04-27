#!/usr/bin/env python3
"""GLM-variant benchmarking harness — v1.

Operates on the per-trial β arrays already produced by `task_2_1_factorial.py`
for ses-03. Computes pre-registered, decoder-free metrics with proper null
floors and bootstrap CIs:

  1. β reliability — for each repeated stimulus image, correlate the per-trial
     β patterns across its repetitions. Higher = more consistent signal across
     reps = better single-trial signal-to-noise. This is the NSD-style noise
     ceiling computation, applied here as a within-method metric.
  2. β reliability of label-shuffled null — shuffle trial→image_id assignment,
     recompute, gives the empirical chance distribution.
  3. Effective top-K coverage — fraction of images whose 3-rep mean β
     pattern is uniquely identifiable among other images' 3-rep means
     (independent of any decoder).

Bootstrap-resampled across images for CIs. Paired-bootstrap for method
comparisons. NO MindEye decoder involved — this is a pure GLM-quality
benchmark that doesn't depend on the trained ridge or the candidate set.

Use:
  python scripts/benchmark_glm_variants.py
  python scripts/benchmark_glm_variants.py --methods A_fmriprep_glover G_rtmotion
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


BETAS_ROOT = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas")
OUT_DIR = BETAS_ROOT / "benchmark_v1"
DEFAULT_METHODS = (
    "A_fmriprep_glover",
    "B_rtmotion_glmsingle",
    "RT_paper",
    "Offline_paper",
    "G_fmriprep",
    "G_rtmotion",
)


def load_betas(method: str, session: str = "ses-03"
               ) -> tuple[np.ndarray, np.ndarray]:
    betas = np.load(BETAS_ROOT / f"{method}_{session}_betas.npy")
    ids = np.load(BETAS_ROOT / f"{method}_{session}_trial_ids.npy",
                  allow_pickle=True)
    ids = np.asarray([str(t) for t in ids])
    return betas, ids


def voxelwise_zscore_full(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd


def beta_reliability(betas: np.ndarray, image_ids: np.ndarray,
                     min_reps: int = 2,
                     z_score: bool = True) -> tuple[float, np.ndarray]:
    """Mean across-rep Pearson correlation of β for each repeated image.

    Returns (mean reliability, per-image array of pairwise-rep mean r).

    For each image with >= min_reps trials, compute the average of the
    pairwise Pearson r across its repetition β vectors. Then average
    across all eligible images.

    z_score: if True, voxelwise z-score across trials before computing
    correlations (matches the retrieval pipeline's input to the decoder).
    """
    X = voxelwise_zscore_full(betas) if z_score else betas
    img_to_idxs = {}
    for i, iid in enumerate(image_ids):
        img_to_idxs.setdefault(iid, []).append(i)

    per_image_r = []
    for iid, idxs in img_to_idxs.items():
        if len(idxs) < min_reps:
            continue
        sub = X[idxs]                        # (n_reps, V)
        # Pairwise Pearson r across reps; np.corrcoef gives full (n,n) matrix
        if sub.shape[0] == 1:
            continue
        c = np.corrcoef(sub)                 # (n_reps, n_reps)
        # Off-diagonal mean
        n = c.shape[0]
        off = c[np.triu_indices(n, k=1)]
        per_image_r.append(float(np.nanmean(off)))
    arr = np.asarray(per_image_r, dtype=np.float32)
    return float(np.nanmean(arr)), arr


def shuffle_null_reliability(betas: np.ndarray, image_ids: np.ndarray,
                             n_perms: int = 50, seed: int = 0,
                             z_score: bool = True) -> np.ndarray:
    """Empirical null: shuffle image_ids, recompute reliability, repeat."""
    rng = np.random.default_rng(seed)
    null_means = []
    for _ in range(n_perms):
        shuffled = image_ids[rng.permutation(len(image_ids))]
        m, _ = beta_reliability(betas, shuffled, z_score=z_score)
        if not np.isnan(m):
            null_means.append(m)
    return np.asarray(null_means, dtype=np.float32)


def image_identifiability(betas: np.ndarray, image_ids: np.ndarray,
                          z_score: bool = True) -> dict:
    """Mean-β-of-each-image identifiability via 1-NN top-1.

    Compute per-image mean β across reps, then for each image, ask whether
    its mean β vector is most-similar (cosine) to the OTHER reps of itself
    versus reps of other images. Reports top-1 hit rate (each rep's 1-NN
    among other-rep mean vectors is the SAME image).
    """
    X = voxelwise_zscore_full(betas) if z_score else betas
    img_to_idxs = {}
    for i, iid in enumerate(image_ids):
        img_to_idxs.setdefault(iid, []).append(i)
    repeats = {iid: idxs for iid, idxs in img_to_idxs.items() if len(idxs) >= 2}
    if not repeats:
        return {"top1_image_id_hit": float("nan"), "n_eligible_images": 0}

    images = list(repeats.keys())
    # Leave-one-rep-out: for each rep of each image, build the "library"
    # from mean(other reps of all images) and ask if 1-NN matches.
    hits = 0
    n_total = 0
    for held_image in images:
        held_idxs = repeats[held_image]
        for held_i in held_idxs:
            # Library: mean β of OTHER reps for each image
            lib = []
            lib_ids = []
            for img in images:
                idxs = [i for i in repeats[img] if i != held_i]
                if not idxs:
                    continue
                lib.append(X[idxs].mean(axis=0))
                lib_ids.append(img)
            if not lib:
                continue
            lib = np.stack(lib, axis=0)
            q = X[held_i]
            qn = q / (np.linalg.norm(q) + 1e-8)
            ln = lib / (np.linalg.norm(lib, axis=1, keepdims=True) + 1e-8)
            sims = ln @ qn
            top1 = lib_ids[int(np.argmax(sims))]
            if top1 == held_image:
                hits += 1
            n_total += 1
    return {
        "top1_image_id_hit": float(hits) / max(n_total, 1),
        "n_total_queries": int(n_total),
        "n_eligible_images": int(len(images)),
    }


def bootstrap_mean_ci(arr: np.ndarray, n_resamples: int = 2000,
                      alpha: float = 0.05, seed: int = 0
                      ) -> tuple[float, float, float]:
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.array([arr[rng.integers(0, len(arr), len(arr))].mean()
                     for _ in range(n_resamples)])
    return (float(arr.mean()),
            float(np.quantile(boot, alpha / 2)),
            float(np.quantile(boot, 1 - alpha / 2)))


def paired_diff_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 2000,
                   seed: int = 0) -> dict:
    """Paired bootstrap of (a - b). Both arrays must be same length / aligned."""
    mask = ~(np.isnan(a) | np.isnan(b))
    a, b = a[mask], b[mask]
    diff = a - b
    rng = np.random.default_rng(seed)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean()
                     for _ in range(n_resamples)])
    return {
        "mean_diff": float(diff.mean()),
        "ci_lo": float(np.quantile(boot, 0.025)),
        "ci_hi": float(np.quantile(boot, 0.975)),
        "p_diff_le_0": float((boot <= 0).mean()),
        "n_paired_images": int(len(diff)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--n-perms", type=int, default=50,
                    help="Number of label-shuffle permutations for null floor")
    ap.add_argument("--reference-method", default="A_fmriprep_glover",
                    help="Method to compare other methods against pairwise")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Per-method metrics ----
    print("[1] Per-method β reliability + identifiability + null floor")
    method_summary = {}
    per_image_r = {}
    for m in args.methods:
        try:
            betas, ids = load_betas(m, args.session)
        except FileNotFoundError as e:
            print(f"  SKIP {m}: {e}")
            continue
        rel_mean, rel_arr = beta_reliability(betas, ids)
        rel_ci = bootstrap_mean_ci(rel_arr)
        ident = image_identifiability(betas, ids)
        null = shuffle_null_reliability(betas, ids, n_perms=args.n_perms)
        method_summary[m] = {
            "n_trials": int(betas.shape[0]),
            "n_voxels": int(betas.shape[1]),
            "n_repeated_images": int(len(rel_arr)),
            "beta_reliability_mean": rel_ci[0],
            "beta_reliability_ci_lo": rel_ci[1],
            "beta_reliability_ci_hi": rel_ci[2],
            "shuffle_null_mean": float(null.mean()) if len(null) else float("nan"),
            "shuffle_null_p95": float(np.quantile(null, 0.95)) if len(null) else float("nan"),
            "image_id_top1_hit": ident["top1_image_id_hit"],
            "n_total_queries": ident["n_total_queries"],
        }
        per_image_r[m] = rel_arr
        print(f"  {m:<26}  rel={rel_ci[0]:+.4f} [{rel_ci[1]:+.4f},{rel_ci[2]:+.4f}]  "
              f"null≈{null.mean():+.4f}  id_hit={ident['top1_image_id_hit']:.3f}")

    # ---- Paired comparisons against reference ----
    print(f"\n[2] Paired bootstrap vs {args.reference_method}")
    if args.reference_method not in per_image_r:
        print(f"  SKIP: reference method {args.reference_method} not loaded")
        paired = {}
    else:
        ref_arr = per_image_r[args.reference_method]
        paired = {}
        for m in args.methods:
            if m == args.reference_method or m not in per_image_r:
                continue
            # Align: both arrays are per-image in the same image order
            # (set determined by image_ids order in load_betas). For safety
            # re-compute aligned by image_id.
            betas_m, ids_m = load_betas(m, args.session)
            betas_r, ids_r = load_betas(args.reference_method, args.session)
            if not np.array_equal(ids_m, ids_r):
                print(f"  SKIP {m}: trial_id alignment mismatch with {args.reference_method}")
                continue
            _, arr_m = beta_reliability(betas_m, ids_m)
            _, arr_r = beta_reliability(betas_r, ids_r)
            d = paired_diff_ci(arr_m, arr_r)
            paired[m] = d
            sig = "✓" if d["ci_lo"] > 0 else "—"
            print(f"  {m:<26} - {args.reference_method:<22}: "
                  f"Δ={d['mean_diff']:+.4f} [{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}] "
                  f"P(Δ≤0)={d['p_diff_le_0']:.3f}  {sig}")

    # ---- Save ----
    summary = {
        "session": args.session,
        "n_perms_null": args.n_perms,
        "reference_method": args.reference_method,
        "per_method": method_summary,
        "paired_vs_reference": paired,
    }
    with open(OUT_DIR / "benchmark_v1_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    # Per-image arrays for downstream analysis
    for m, arr in per_image_r.items():
        np.save(OUT_DIR / f"per_image_reliability_{m}.npy", arr)
    print(f"\nWrote {OUT_DIR / 'benchmark_v1_summary.json'}")


if __name__ == "__main__":
    main()
