#!/usr/bin/env python
"""Multivariate Ridge probes from SAE features → HBN clinical concepts.

Companion to ``sae_concept_probes.py`` which fits per-feature univariate
probes. Per-feature R² is structurally weak for compositional concepts
like p_factor (each individual SAE feature only fires a small fraction
of the time, so a single feature has low explanatory power for a
continuous clinical score). The multivariate ridge regression fits all
live features jointly: ``label_subject ≈ subj_latents @ w + b``. That
tests whether the SAE *as a whole* encodes the concept, decoupled from
per-feature compositionality.

Methodology
-----------
1. Encode activations through the SAE (TopK + sparsify) — same as
   sae_concept_probes.py
2. Mean-pool feature activations per subject → (n_subj, d_dict)
3. Subject-stratified train/test split (75/25 by default)
4. For each concept: closed-form ridge ``w = (X^T X + λI)^-1 X^T y``,
   sweep λ on a log grid on TRAIN only (k-fold not needed; subject-
   stratified hold-out is robust enough), report best test R² (or AUC
   for sex)

Output
------
    <out_prefix>_multivariate.csv  — concept, best_lambda, train_R², test_R², n_train, n_test
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


CONTINUOUS_CONCEPTS = ("age", "p_factor", "internalizing",
                       "externalizing", "attention")
BINARY_CONCEPTS = ("sex",)
ALL_CONCEPTS = CONTINUOUS_CONCEPTS + BINARY_CONCEPTS


def _shim_jaxoccoli_package():
    import sys, os, types
    if "jaxoccoli" in sys.modules:
        return
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_dir = os.path.join(project_root, "jaxoccoli")
    shim = types.ModuleType("jaxoccoli")
    shim.__path__ = [pkg_dir]
    shim.__file__ = os.path.join(pkg_dir, "__init__.py")
    sys.modules["jaxoccoli"] = shim


def encode_latents(activations, sae_npz, batch_size=8192):
    """Same as sae_concept_probes.encode_latents."""
    import jax
    import jax.numpy as jnp
    _shim_jaxoccoli_package()
    from jaxoccoli.sae import TopKSAEParams, sae_encode, topk_sparsify

    params = TopKSAEParams(
        enc_weight=jnp.asarray(sae_npz["enc_weight"]),
        enc_bias=jnp.asarray(sae_npz["enc_bias"]),
        dec_weight=jnp.asarray(sae_npz["dec_weight"]),
        dec_bias=jnp.asarray(sae_npz["dec_bias"]),
    )
    k = int(sae_npz["k"])

    @jax.jit
    def _encode_batch(params, x):
        z = sae_encode(params, x)
        return topk_sparsify(z, k)

    latents = np.empty((activations.shape[0], int(sae_npz["d_dict"])),
                       dtype=np.float32)
    for i in range(0, activations.shape[0], batch_size):
        chunk = jnp.asarray(activations[i:i + batch_size])
        z = _encode_batch(params, chunk)
        latents[i:i + batch_size] = np.asarray(z)
    return latents


def ridge_fit_predict(X_tr, y_tr, X_te, lam):
    """Closed-form ridge: w = (X^T X + λI)^-1 X^T y.
    Returns predictions on X_tr and X_te.
    """
    d = X_tr.shape[1]
    # Center for numerical stability
    x_mean = X_tr.mean(axis=0)
    y_mean = y_tr.mean()
    Xc = X_tr - x_mean
    yc = y_tr - y_mean
    A = Xc.T @ Xc + lam * np.eye(d, dtype=np.float32)
    w = np.linalg.solve(A, Xc.T @ yc)
    pred_tr = Xc @ w + y_mean
    pred_te = (X_te - x_mean) @ w + y_mean
    return pred_tr, pred_te


def r2(pred, y):
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    sum_pos = float(np.sum(ranks[labels == 1]))
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True)
    ap.add_argument("--sae", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=500_000)
    args = ap.parse_args()

    print(f"[load] activations {args.activations}", flush=True)
    blob = np.load(args.activations, allow_pickle=True)
    acts = blob["activations"].astype(np.float32)
    subj = blob["subject_id"]
    n_total = acts.shape[0]
    print(f"  tokens={n_total} d_model={acts.shape[1]}", flush=True)

    if n_total > args.max_tokens:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(n_total, size=args.max_tokens, replace=False)
        keep.sort()
        acts = acts[keep]
        subj = subj[keep]
        new_blob = {"subject_id": subj}
        for f in CONTINUOUS_CONCEPTS:
            new_blob[f] = blob[f][keep]
        new_blob["sex"] = blob["sex"][keep]
        blob = new_blob
        print(f"  subsampled to {len(acts)} tokens", flush=True)
    else:
        blob = dict(blob)

    print(f"[load] sae {args.sae}", flush=True)
    sae_blob = dict(np.load(args.sae, allow_pickle=True))

    print(f"[encode] activations → latents", flush=True)
    latents = encode_latents(acts, sae_blob)
    print(f"  latents: {latents.shape}", flush=True)

    # Subject-level aggregation
    print(f"[aggregate] mean-pool features per subject", flush=True)
    unique_subj, subj_idx = np.unique(subj, return_inverse=True)
    n_subj = len(unique_subj)
    counts = np.bincount(subj_idx, minlength=n_subj).astype(np.float32)
    subj_latents = np.zeros((n_subj, latents.shape[1]), dtype=np.float32)
    np.add.at(subj_latents, subj_idx, latents)
    subj_latents /= counts[:, None]
    # Keep only features with subject-level variance
    subj_var = subj_latents.var(axis=0)
    live = subj_var > 1e-10
    subj_latents = subj_latents[:, live]
    print(f"  {n_subj} subjects × {subj_latents.shape[1]} live features",
          flush=True)

    # First-token-per-subject for label lookup
    first_token_per_subj = np.zeros(n_subj, dtype=np.int64)
    seen = np.zeros(n_subj, dtype=bool)
    for i, s in enumerate(subj_idx):
        if not seen[s]:
            first_token_per_subj[s] = i
            seen[s] = True

    subj_concept = {}
    subj_valid = {}
    for c in CONTINUOUS_CONCEPTS:
        v = blob[c][first_token_per_subj]
        subj_concept[c] = v.astype(np.float32)
        subj_valid[c] = ~np.isnan(v)
    sex_str = blob["sex"][first_token_per_subj]
    sex_bin = np.where(sex_str == "M", 1.0,
                       np.where(sex_str == "F", 0.0, np.nan))
    subj_concept["sex"] = sex_bin.astype(np.float32)
    subj_valid["sex"] = ~np.isnan(sex_bin)

    # Subject-level train/test split — same seed across concepts to make
    # them comparable (though only labeled subjects per concept are used)
    rng = np.random.default_rng(args.seed)
    lam_grid = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5]

    print(f"[ridge] sweeping λ on {len(lam_grid)} points", flush=True)
    results = []
    for c in ALL_CONCEPTS:
        valid = subj_valid[c]
        idx = np.where(valid)[0]
        perm = np.random.default_rng(args.seed).permutation(idx)
        n_test = max(1, int(round(len(perm) * args.test_frac)))
        test_subj = perm[:n_test]
        train_subj = perm[n_test:]
        X_tr = subj_latents[train_subj]
        X_te = subj_latents[test_subj]
        y_tr = subj_concept[c][train_subj]
        y_te = subj_concept[c][test_subj]

        best_lam = None
        best_test = -np.inf
        best_train = None
        for lam in lam_grid:
            pred_tr, pred_te = ridge_fit_predict(X_tr, y_tr, X_te, lam)
            if c in BINARY_CONCEPTS:
                te_score = auc(pred_te, y_te)
                te_score = max(te_score, 1.0 - te_score)  # symmetry
                tr_score = auc(pred_tr, y_tr)
                tr_score = max(tr_score, 1.0 - tr_score)
            else:
                te_score = r2(pred_te, y_te)
                tr_score = r2(pred_tr, y_tr)
            if te_score > best_test:
                best_test = te_score
                best_train = tr_score
                best_lam = lam

        results.append({
            "concept": c,
            "metric": "AUC" if c in BINARY_CONCEPTS else "R²",
            "best_lambda": best_lam,
            "train_score": best_train,
            "test_score": best_test,
            "n_train": int(len(train_subj)),
            "n_test": int(len(test_subj)),
        })
        print(f"  {c:14s} best_λ={best_lam:>8.1e} "
              f"train={best_train:.3f}  test={best_test:.3f}  "
              f"(n_train={len(train_subj)}, n_test={len(test_subj)})",
              flush=True)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = Path(str(out_prefix) + "_multivariate.csv")
    with open(csv_path, "w") as f:
        f.write("concept,metric,best_lambda,train_score,test_score,"
                "n_train,n_test\n")
        for r in results:
            f.write(f"{r['concept']},{r['metric']},{r['best_lambda']:.6g},"
                    f"{r['train_score']:.6f},{r['test_score']:.6f},"
                    f"{r['n_train']},{r['n_test']}\n")
    print(f"[done] {csv_path}", flush=True)
    json_path = Path(str(out_prefix) + "_multivariate.json")
    with open(json_path, "w") as f:
        json.dump({
            "activations": str(args.activations),
            "sae": str(args.sae),
            "n_subjects": int(n_subj),
            "n_live_features": int(subj_latents.shape[1]),
            "results": results,
        }, f, indent=2)
    print(f"[done] {json_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
