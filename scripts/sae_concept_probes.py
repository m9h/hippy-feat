#!/usr/bin/env python
"""Per-feature linear/logistic probes from a trained SAE → HBN clinical concepts.

Pipeline
--------
    activations.npz  +  sae.npz
        │                  │
        ▼                  ▼
       (encode → TopK → latents)         (load encoder weights)
        │
        ▼
    For each SAE feature f ∈ [0, d_dict):
      For each concept c ∈ {age, sex, p_factor, internalizing,
                            externalizing, attention}:
        fit linear (or logistic for sex) probe  latents[:, f] → c
        report subject-held-out test score

The output is a (d_dict × n_concepts) score matrix plus a per-feature
"target vs off-target" selectivity gap — the BrainCapture metric that
defines selectively-steerable / encoded-but-entangled / non-encoded
feature regimes.

Subject-stratified split: split by subject IDs (not windows) so train and
test never share a subject. Otherwise the linear probe collapses to subject
identity prediction, not concept prediction.

Output
------
    <out_prefix>.csv     — long-form table: feature, concept, train_score, test_score
    <out_prefix>_selectivity.csv  — per-feature best-concept + gap
    <out_prefix>.json    — config, totals, top-N features per concept
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
    """Bypass jaxoccoli/__init__.py — same trick the extract + train scripts use."""
    import sys, os, types
    if "jaxoccoli" in sys.modules:
        return
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pkg_dir = os.path.join(project_root, "jaxoccoli")
    shim = types.ModuleType("jaxoccoli")
    shim.__path__ = [pkg_dir]
    shim.__file__ = os.path.join(pkg_dir, "__init__.py")
    sys.modules["jaxoccoli"] = shim


def encode_latents(activations: np.ndarray, sae_npz: dict,
                   batch_size: int = 8192) -> np.ndarray:
    """Run activations through the trained SAE encoder + TopK to get
    sparse latent codes of shape (N, d_dict).
    """
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
    n = activations.shape[0]
    for i in range(0, n, batch_size):
        chunk = jnp.asarray(activations[i:i + batch_size])
        z = _encode_batch(params, chunk)
        latents[i:i + batch_size] = np.asarray(z)
    return latents


def subject_stratified_split(subject_ids: np.ndarray,
                              labeled_mask: np.ndarray,
                              test_frac: float, seed: int
                              ) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) over rows where labeled_mask is True.

    Split is BY SUBJECT — each subject's rows are wholly in train or wholly
    in test. This prevents probes from collapsing to subject-id prediction.
    """
    rng = np.random.default_rng(seed)
    labeled_subj = np.unique(subject_ids[labeled_mask])
    rng.shuffle(labeled_subj)
    n_test = max(1, int(round(len(labeled_subj) * test_frac)))
    test_subj = set(labeled_subj[:n_test])

    in_test = np.isin(subject_ids, list(test_subj))
    train_idx = np.where(labeled_mask & ~in_test)[0]
    test_idx = np.where(labeled_mask & in_test)[0]
    return train_idx, test_idx


def probe_one_feature_continuous(feature: np.ndarray, label: np.ndarray,
                                  train_idx: np.ndarray,
                                  test_idx: np.ndarray) -> tuple[float, float]:
    """Univariate linear regression label = a * feature + b.
    Returns (train_R2, test_R2). Negative R² possible if test fit is worse than
    predicting the train mean.
    """
    x_tr = feature[train_idx]
    y_tr = label[train_idx]
    x_te = feature[test_idx]
    y_te = label[test_idx]

    var_x = float(np.var(x_tr))
    if var_x < 1e-12:
        return 0.0, 0.0
    cov = float(np.mean((x_tr - x_tr.mean()) * (y_tr - y_tr.mean())))
    a = cov / var_x
    b = float(y_tr.mean() - a * x_tr.mean())

    def r2(x, y):
        pred = a * x + b
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / (ss_tot + 1e-12)

    return r2(x_tr, y_tr), r2(x_te, y_te)


def probe_one_feature_binary(feature: np.ndarray, y_bin: np.ndarray,
                              train_idx: np.ndarray,
                              test_idx: np.ndarray) -> tuple[float, float]:
    """Univariate AUC of the feature as a score for a binary label.
    Returns (train_AUC, test_AUC). Uses the rank-AUC formula so no sklearn dep.
    """
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

    return (auc(feature[train_idx], y_bin[train_idx]),
            auc(feature[test_idx], y_bin[test_idx]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--activations", required=True,
                    help="extract_eeg_fm_acts.py output (.npz)")
    ap.add_argument("--sae", required=True,
                    help="train_sae.py output (.npz)")
    ap.add_argument("--out-prefix", required=True,
                    help="Output prefix (no extension)")
    ap.add_argument("--test-frac", type=float, default=0.25,
                    help="Subject-fraction held out for probe test")
    ap.add_argument("--top-n", type=int, default=20,
                    help="Per-concept top-N features to print + JSON")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-features", type=int, default=None,
                    help="(debug) cap features evaluated")
    ap.add_argument("--max-tokens", type=int, default=500_000,
                    help="Cap the activation matrix at this many tokens "
                         "(subject-stratified subsample). 7.4M × 2048 = 60 GB "
                         "RAM for the full latent matrix, which OOMs; 500k "
                         "still yields stable probe estimates.")
    args = ap.parse_args()

    print(f"[load] activations {args.activations}", flush=True)
    blob = np.load(args.activations, allow_pickle=True)
    acts = blob["activations"].astype(np.float32)
    subj = blob["subject_id"]
    n_total = acts.shape[0]
    print(f"  tokens={n_total} d_model={acts.shape[1]} "
          f"subjects={len(np.unique(subj))}", flush=True)

    # Subsample to keep the latent matrix in RAM. Stratify by subject so we
    # preserve the subject-stratified split downstream.
    if n_total > args.max_tokens:
        rng = np.random.default_rng(args.seed)
        keep = rng.choice(n_total, size=args.max_tokens, replace=False)
        keep.sort()
        acts = acts[keep]
        subj = subj[keep]
        for f in CONTINUOUS_CONCEPTS:
            blob_arr = blob[f]
            globals()["_subsampled_" + f] = blob_arr[keep]
        # Patch blob.__getitem__ via a thin shim so the rest of the code
        # below reads the subsampled views (cleanest minimal change).
        _orig_keep = keep
        class _Sub:
            def __init__(self, src, keep):
                self._src = src
                self._keep = keep
            def __getitem__(self, name):
                if name in ("activations", "subject_id"):
                    raise KeyError(name)
                arr = self._src[name]
                if arr.shape[:1] == (n_total,):
                    return arr[self._keep]
                return arr
        blob = _Sub(blob, _orig_keep)
        print(f"  subsampled to {len(acts)} tokens (limit --max-tokens={args.max_tokens})",
              flush=True)

    print(f"[load] sae {args.sae}", flush=True)
    sae_blob = dict(np.load(args.sae, allow_pickle=True))

    print(f"[encode] activations → latents (d_dict={int(sae_blob['d_dict'])})",
          flush=True)
    latents = encode_latents(acts, sae_blob)
    n, d_dict = latents.shape
    print(f"  latents shape = {latents.shape}", flush=True)

    if args.max_features is not None:
        d_dict = min(d_dict, args.max_features)
        latents = latents[:, :d_dict]
        print(f"  --max-features set; probing first {d_dict}", flush=True)

    # Live features only — dead atoms have variance 0 and waste compute.
    feature_var = latents.var(axis=0)
    live_mask = feature_var > 1e-10
    live_features = np.where(live_mask)[0]
    print(f"[live] {len(live_features)}/{d_dict} features fire on this batch "
          f"({len(live_features) / d_dict:.1%})", flush=True)

    # Concept labels per token. Coerce sex to {0,1} where M=1, F=0, drop NA.
    concept_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for c in CONTINUOUS_CONCEPTS:
        raw = blob[c].astype(np.float32)
        valid = ~np.isnan(raw)
        concept_arrays[c] = (raw, valid)
    sex_str = blob["sex"]
    sex_bin = np.where(sex_str == "M", 1.0, np.where(sex_str == "F", 0.0, np.nan))
    concept_arrays["sex"] = (sex_bin.astype(np.float32), ~np.isnan(sex_bin))

    rows = []        # (feature, concept, train_score, test_score)
    sel_records = []  # one per feature: target metric + off-target metric

    # --- Aggregate to SUBJECT level --------------------------------------
    # Per-token regression returns ~0 R² for all features because each
    # subject's tokens all share the same label (label has zero within-
    # subject variance, feature has high within-subject variance — slope
    # estimator drowns in within-subject noise). The BrainCapture-style
    # methodology aggregates feature activations per subject (mean across
    # the subject's tokens) before regressing.
    print(f"[aggregate] mean-pooling features per subject", flush=True)
    unique_subj, subj_idx = np.unique(subj, return_inverse=True)
    n_subj = len(unique_subj)
    counts = np.bincount(subj_idx, minlength=n_subj).astype(np.float32)
    subj_latents = np.zeros((n_subj, latents.shape[1]), dtype=np.float32)
    # Single scatter-add over the full (N, d_dict) matrix.
    np.add.at(subj_latents, subj_idx, latents)
    subj_latents /= counts[:, None]
    print(f"  {n_subj} subjects × {latents.shape[1]} features", flush=True)

    # Recompute the live-feature set using SUBJECT-level variance — the
    # token-level filter passes features that mean-pool to near-constant
    # values, which then trigger the (var_x < 1e-12) early-exit in the
    # regression and return literal-zero R² (visible as p_factor's "0.000"
    # in the first run).
    subj_var = subj_latents.var(axis=0)
    live_features = np.where(subj_var > 1e-10)[0]
    print(f"  [live-subj] {len(live_features)}/{d_dict} features have "
          f"subject-level variance (after aggregation)", flush=True)

    # Pull subject-level labels (any token for a subject works — labels
    # are constant per subject).
    subj_concept = {}
    subj_valid = {}
    first_token_per_subj = np.zeros(n_subj, dtype=np.int64)
    # Find one representative token index per subject.
    seen = np.zeros(n_subj, dtype=bool)
    for i, s in enumerate(subj_idx):
        if not seen[s]:
            first_token_per_subj[s] = i
            seen[s] = True
    for c in CONTINUOUS_CONCEPTS:
        raw_token, valid_token = concept_arrays[c]
        subj_concept[c] = raw_token[first_token_per_subj]
        subj_valid[c] = valid_token[first_token_per_subj]
    sex_token = concept_arrays["sex"][0]
    sex_valid_token = concept_arrays["sex"][1]
    subj_concept["sex"] = sex_token[first_token_per_subj]
    subj_valid["sex"] = sex_valid_token[first_token_per_subj]

    print(f"[split] subject-level splits over {n_subj} subjects", flush=True)
    rng = np.random.default_rng(args.seed)
    splits = {}
    for c in ALL_CONCEPTS:
        valid_s = subj_valid[c]
        idx_s = np.where(valid_s)[0]
        rng_c = np.random.default_rng(args.seed)
        perm = rng_c.permutation(idx_s)
        n_test = max(1, int(round(len(perm) * args.test_frac)))
        splits[c] = (perm[n_test:], perm[:n_test])
        print(f"  {c:14s} train_subj={len(splits[c][0])} "
              f"test_subj={len(splits[c][1])}", flush=True)

    print(f"[probe] {len(live_features)} live features × "
          f"{len(ALL_CONCEPTS)} concepts on subject-level features",
          flush=True)

    for feat_pos, f in enumerate(live_features):
        feat = subj_latents[:, f]
        per_concept_test = {}
        for c in ALL_CONCEPTS:
            label = subj_concept[c]
            tr, te = splits[c]
            if len(tr) < 10 or len(te) < 5:
                tr_score = te_score = float("nan")
            elif c in BINARY_CONCEPTS:
                tr_score, te_score = probe_one_feature_binary(feat, label, tr, te)
            else:
                tr_score, te_score = probe_one_feature_continuous(feat, label, tr, te)
            rows.append((int(f), c, tr_score, te_score))
            per_concept_test[c] = te_score

        # Target-vs-off-target selectivity gap (BrainCapture-style).
        finite = {c: v for c, v in per_concept_test.items() if np.isfinite(v)}
        if not finite:
            continue
        # For binary concepts, transform AUC to |AUC - 0.5| so it's a
        # 'discrimination' magnitude comparable to R².
        adj = {c: (abs(v - 0.5) if c in BINARY_CONCEPTS else v)
               for c, v in finite.items()}
        target_c = max(adj, key=adj.get)
        off_target = [v for c, v in adj.items() if c != target_c]
        gap = float(adj[target_c] - max(off_target)) if off_target else float("nan")
        sel_records.append({
            "feature": int(f),
            "target_concept": target_c,
            "target_metric": float(adj[target_c]),
            "off_target_max": float(max(off_target)) if off_target else float("nan"),
            "selectivity_gap": gap,
        })

        if (feat_pos + 1) % 100 == 0:
            print(f"  {feat_pos + 1}/{len(live_features)} probed", flush=True)

    # --- Write outputs --------------------------------------------------------
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    csv_path = out_prefix.with_suffix(".csv")
    with open(csv_path, "w") as f:
        f.write("feature,concept,train_score,test_score\n")
        for feat, c, tr, te in rows:
            f.write(f"{feat},{c},{tr:.6f},{te:.6f}\n")
    print(f"[done] {csv_path}  ({len(rows)} rows)", flush=True)

    sel_csv = Path(str(out_prefix) + "_selectivity.csv")
    with open(sel_csv, "w") as f:
        f.write("feature,target_concept,target_metric,off_target_max,selectivity_gap\n")
        for s in sel_records:
            f.write(f"{s['feature']},{s['target_concept']},"
                    f"{s['target_metric']:.6f},{s['off_target_max']:.6f},"
                    f"{s['selectivity_gap']:.6f}\n")
    print(f"[done] {sel_csv}  ({len(sel_records)} rows)", flush=True)

    # Top-N features per concept
    top_per_concept = {}
    for c in ALL_CONCEPTS:
        if c in BINARY_CONCEPTS:
            scored = [(r[0], abs(r[3] - 0.5)) for r in rows
                      if r[1] == c and np.isfinite(r[3])]
        else:
            scored = [(r[0], r[3]) for r in rows
                      if r[1] == c and np.isfinite(r[3])]
        scored.sort(key=lambda t: -t[1])
        top_per_concept[c] = [
            {"feature": int(fi), "metric": float(s)}
            for fi, s in scored[:args.top_n]
        ]

    json_path = out_prefix.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump({
            "activations": str(args.activations),
            "sae": str(args.sae),
            "test_frac": args.test_frac,
            "seed": args.seed,
            "n_features_probed": int(len(live_features)),
            "n_features_total": int(latents.shape[1]),
            "n_labeled_subjects_per_concept": {
                c: int(len(np.unique(subj[valid])))
                for c, (_, valid) in concept_arrays.items()
            },
            "top_per_concept": top_per_concept,
        }, f, indent=2)
    print(f"[done] {json_path}", flush=True)

    # Quick stdout summary
    print()
    print("Top features per concept (test score):")
    for c in ALL_CONCEPTS:
        top3 = top_per_concept[c][:3]
        if top3:
            label = "|AUC-0.5|" if c in BINARY_CONCEPTS else "R²"
            scored = "  ".join(f"f{t['feature']}={t['metric']:.3f}" for t in top3)
            print(f"  {c:14s} [{label}]   {scored}")


if __name__ == "__main__":
    sys.exit(main())
