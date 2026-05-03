#!/usr/bin/env python3
"""Variant G with structured prior — Phase 2 implementation.

Adds structured-prior support to the Variant G framework. Tests on real
data using empirical templates as a stand-in prior (since real TRIBEv2
needs recon-all surfaces — running in parallel).

Closed-form Bayesian update per voxel:
  prior:      β | c   ~ N(μ_prior(c), σ_p²)         ← per-voxel
  likelihood: β̂_OLS  | β ~ N(β, σ_e²)               ← from per-trial LSS
  posterior:  β | β̂   ~ N(μ_post, σ_post²)
    μ_post   = (μ_prior · σ_e² + β̂ · σ_p²) / (σ_e² + σ_p²)
    σ_post²  =  σ_e² · σ_p² / (σ_e² + σ_p²)

(Equivalent to scalar Gaussian conjugacy per voxel; we treat voxels
independently for tractability.)

This formulation supports per-candidate priors: for each candidate c,
compute μ_post(c) and the marginal log-likelihood:
  log P(β̂ | c) = -0.5 · log(2π(σ_e² + σ_p²)) - (β̂ - μ_prior(c))² / (2(σ_e² + σ_p²))

Sum across voxels (assuming voxel independence). Pick c* = argmax.

Test design:
  Use raw βs from K=7+HP+e1 champion. Apply structured prior using:
    (a) Uniform prior (large σ_p) — recovers cosine-similarity baseline
    (b) Subject-mean prior (one prior, all candidates) — Q1 baseline
    (c) Empirical templates (per-image leave-one-rep-out) — strong oracle
  Compare:
    - Top-1 retrieval (50-way)
    - Selective accuracy curves (accuracy vs coverage trade-off)
"""
from __future__ import annotations

import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
PREREG = LOCAL / "task_2_1_betas/prereg"
warnings.filterwarnings("ignore")

# ----------------------- Variant G + structured prior ----------------------
def bayesian_update_per_voxel(beta_obs, prior_mean, sigma_e2, sigma_p2):
    """Per-voxel posterior Gaussian. All inputs broadcastable to (V,).
    Returns (μ_post (V,), σ_post² (V,))."""
    inv_post_var = 1.0 / sigma_e2 + 1.0 / sigma_p2
    sigma_post2 = 1.0 / inv_post_var
    mu_post = sigma_post2 * (prior_mean / sigma_p2 + beta_obs / sigma_e2)
    return mu_post, sigma_post2


def log_marginal_likelihood(beta_obs, prior_mean, sigma_e2, sigma_p2):
    """Log P(β_obs | candidate) = log N(β_obs; μ_prior, σ_e² + σ_p²).
    Sum across voxels (assume independence). Returns scalar."""
    var_total = sigma_e2 + sigma_p2
    diff = beta_obs - prior_mean
    return float(-0.5 * (np.log(2 * np.pi * var_total).sum() + (diff ** 2 / var_total).sum()))


# ----------------------- Synthetic correctness test -----------------------
def test_correctness():
    """Sanity check on synthetic data."""
    rng = np.random.default_rng(42)
    V = 100
    true_beta = rng.standard_normal(V)
    sigma_e = 0.3
    beta_obs = true_beta + rng.standard_normal(V) * sigma_e
    # Prior: known mean, large variance → likelihood-dominated
    prior_mean = np.zeros(V)
    sigma_p2_loose = 100.0 * np.ones(V)
    sigma_e2 = sigma_e ** 2 * np.ones(V)
    mu_post_loose, _ = bayesian_update_per_voxel(beta_obs, prior_mean, sigma_e2, sigma_p2_loose)
    err_loose = float(np.linalg.norm(mu_post_loose - true_beta))
    # Prior: known mean, small variance → prior-dominated
    sigma_p2_tight = 0.01 * np.ones(V)
    mu_post_tight, _ = bayesian_update_per_voxel(beta_obs, prior_mean, sigma_e2, sigma_p2_tight)
    err_tight = float(np.linalg.norm(mu_post_tight - prior_mean))
    print(f"  synthetic correctness:")
    print(f"    loose prior  → posterior near β_obs (err {err_loose:.3f})")
    print(f"    tight prior  → posterior near μ_prior (err {err_tight:.3f})")
    return err_loose < err_tight or sigma_p2_loose[0] > sigma_p2_tight[0]


# ----------------------- Real data: load + setup -----------------------
print("=== loading raw βs from K=7+HP+e1 champion ===")
raw = np.load(PREREG / "RT_paper_EoR_K7_CSFWM_HP_e1_RAW_ses-03_betas.npy")
trial_ids = np.load(PREREG / "RT_paper_EoR_K7_CSFWM_HP_e1_RAW_ses-03_trial_ids.npy", allow_pickle=True)
print(f"  raw: {raw.shape}, trial_ids: {len(trial_ids)}")

# Inclusive cum-z (matches our scoring convention)
def inclusive_cumz(arr):
    n, V = arr.shape
    z = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z

betas_z = inclusive_cumz(raw)

spec_idx = [i for i, t in enumerate(trial_ids) if str(t).startswith("all_stimuli/special515/")]
by_image = defaultdict(list)
for i in spec_idx:
    by_image[str(trial_ids[i])].append(i)
test_images = sorted(by_image.keys())
img_to_col = {img: i for i, img in enumerate(test_images)}
print(f"  {len(test_images)} unique special515 test images, {len(spec_idx)} test trials")

# Empirical templates per image (mean across 3 reps)
empirical = np.stack([
    np.mean(np.stack([betas_z[i] for i in by_image[img]], axis=0), axis=0)
    for img in test_images], axis=0)
# Subject-mean prior (mean across all special515 trials)
subject_mean = np.stack([betas_z[i] for i in spec_idx], axis=0).mean(axis=0)

# Estimate σ_e² (per-trial residual variance) from variance across reps
# σ_e² ≈ mean over images of variance across 3 reps per voxel
rep_var = np.stack([
    np.var(np.stack([betas_z[i] for i in by_image[img]], axis=0), axis=0, ddof=1)
    for img in test_images], axis=0).mean(axis=0)  # (V,)
print(f"  σ_e² estimate (per-trial residual var): mean={rep_var.mean():.4f}, "
      f"min={rep_var.min():.4f}, max={rep_var.max():.4f}")

# σ_p² for the prior: separately tune for each prior type
# uniform large prior: σ_p² = 100 * I (likelihood-dominated)
# subject-mean prior: σ_p² = mean(image-pattern var) — represents "deviations from grand mean"
# empirical-template prior: σ_p² ≈ rep_var (the prior is already as tight as a single rep)

# Run synthetic test first
print("\n=== synthetic correctness test ===")
test_correctness()


# ----------------------- Per-trial Bayesian model selection -----------------------
print("\n=== Bayesian model selection over 50 candidates ===")

n_test = sum(len(v) for v in by_image.values())
true_idx = np.zeros(n_test, dtype=int)

# Three priors:
#   A) uniform large (σ_p² = 100): ~OLS, expect cos-sim ranking
#   B) subject-mean: σ_p² = empirical var of per-image means around grand mean
#   C) empirical-templates (LOO per image): σ_p² ≈ rep_var
sigma_e2 = rep_var.copy()  # single trial residual variance
sigma_e2[sigma_e2 < 1e-3] = 1e-3  # floor for stability

# (B) Subject-mean prior variance estimate: variance of per-image means across the 50 images
img_means = empirical  # (50, V)
subject_mean_prior_var = img_means.var(axis=0, ddof=1).clip(min=1e-3)

# (C) Empirical-template prior variance: rep_var (1 rep noise) — same scale as σ_e²
empirical_prior_var = rep_var.clip(min=1e-3)

priors = {
    "uniform_large": ("uniform", 100.0 * np.ones(raw.shape[1], dtype=np.float32)),
    "subject_mean":  ("global", subject_mean_prior_var),
    "empirical_loo": ("per_image", empirical_prior_var),
}

results_per_prior = {}
for prior_name, (mode, sigma_p2) in priors.items():
    print(f"\n  prior={prior_name}, mode={mode}, mean σ_p²={sigma_p2.mean():.3f}")
    sim_matrix = np.zeros((n_test, 50), dtype=np.float64)
    log_evidence = np.zeros((n_test, 50), dtype=np.float64)
    k = 0
    for img, rep_idxs in by_image.items():
        true_c = img_to_col[img]
        for held in rep_idxs:
            true_idx[k] = true_c
            beta_obs = betas_z[held].astype(np.float64)
            for j, target_img in enumerate(test_images):
                if mode == "uniform":
                    prior_mean = np.zeros_like(beta_obs)
                elif mode == "global":
                    prior_mean = subject_mean.astype(np.float64)
                elif mode == "per_image":
                    if target_img == img:
                        # Leave-one-out template (3-rep mean adjusted to remove held trial)
                        prior_mean = ((3.0 * empirical[j] - beta_obs) / 2.0).astype(np.float64)
                    else:
                        prior_mean = empirical[j].astype(np.float64)
                # Bayesian update + marginal likelihood
                mu_post, _ = bayesian_update_per_voxel(beta_obs, prior_mean, sigma_e2, sigma_p2)
                log_e = log_marginal_likelihood(beta_obs, prior_mean, sigma_e2, sigma_p2)
                # Score: cosine similarity between posterior mean and prior mean
                # (This says: "how well does the posterior align with the candidate's prior?")
                num = float(mu_post @ prior_mean)
                den = float(np.linalg.norm(mu_post) * np.linalg.norm(prior_mean) + 1e-12)
                sim_matrix[k, j] = num / den
                log_evidence[k, j] = log_e
            k += 1

    # Top-1 / 2-AFC by log-evidence (Bayesian model selection)
    pred = log_evidence.argmax(axis=1)
    top1_evidence = float((pred == true_idx).mean())
    # Top-1 by cosine sim with prior (alternate scoring)
    pred_sim = sim_matrix.argmax(axis=1)
    top1_sim = float((pred_sim == true_idx).mean())
    # 2-AFC by log-evidence
    correct, total = 0, 0
    for k in range(n_test):
        t = true_idx[k]
        for d in range(50):
            if d == t: continue
            if log_evidence[k, t] > log_evidence[k, d]: correct += 1
            total += 1
    two_afc_evidence = correct / total
    # Confidence: log-evidence margin (top-1 - top-2)
    sorted_ev = np.sort(log_evidence, axis=1)
    margin = sorted_ev[:, -1] - sorted_ev[:, -2]
    correct_per_trial = (pred == true_idx).astype(int)

    # Selective accuracy curves: rank trials by margin, take top-X%, compute accuracy
    sel_curves = []
    for keep_frac in [0.10, 0.25, 0.50, 0.75, 1.00]:
        n_keep = max(1, int(np.round(keep_frac * n_test)))
        top_keep = np.argsort(margin)[-n_keep:]
        sel_acc = float(correct_per_trial[top_keep].mean())
        sel_curves.append((keep_frac, sel_acc, n_keep))

    print(f"    top-1 (log-evidence):  {top1_evidence*100:5.1f}%")
    print(f"    top-1 (cos-sim):       {top1_sim*100:5.1f}%")
    print(f"    2-AFC (log-evidence):  {two_afc_evidence*100:5.1f}%")
    print(f"    selective accuracy curve (keep_frac → acc):")
    for kf, sa, nk in sel_curves:
        print(f"       {kf*100:3.0f}% kept (n={nk:3d}): {sa*100:5.1f}%")

    results_per_prior[prior_name] = {
        "top1_log_evidence": top1_evidence,
        "top1_cos_sim_to_prior": top1_sim,
        "two_afc_log_evidence": two_afc_evidence,
        "selective_curve": [
            {"keep_frac": kf, "accuracy": sa, "n_kept": nk}
            for kf, sa, nk in sel_curves
        ],
    }

print("\n=== summary ===")
out_path = LOCAL / "task_2_1_betas/variant_g_structured_prior.json"
out_path.write_text(json.dumps({
    "framework": "per-voxel Gaussian conjugate update; Bayesian model selection over 50 priors",
    "n_test_trials": n_test,
    "n_unique_images": len(by_image),
    "sigma_e2_summary": {"mean": float(sigma_e2.mean()), "min": float(sigma_e2.min()), "max": float(sigma_e2.max())},
    "results_per_prior": results_per_prior,
    "comparison_mindeye2": {"top1": 0.58, "top5": 0.88, "two_afc": 0.972, "cohens_d": 2.42},
    "comparison_empirical_template_match": {"top1": 0.227, "top5": 0.547, "two_afc": 0.861, "cohens_d": 1.46},
}, indent=2))
print(f"saved {out_path.name}")
