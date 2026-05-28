#!/usr/bin/env python
"""WeightWatcher (HT-SR) analysis of REVE's transformer blocks.

What this does: load the brain-bzh/reve-base checkpoint, hand it to
WeightWatcher, and dump the per-layer alpha + alpha_hat quality metrics
along with a summary that maps block index → mean alpha across its
constituent Q/K/V/proj/FF matrices.

Why: WeightWatcher's HT-SR theory (Martin & Mahoney 2018-2024) says the
empirical spectral density of a well-trained layer follows a heavy-tailed
power law with exponent ``alpha`` in [2, 6]. alpha < 2 means under-trained
or rank-collapsed; alpha > 6 means over-parameterised relative to data.

Cross-check with our SAE results:
  * Layer -1 SAE: 81-95% dead features, EV=1.0 on tiny recon_mse →
    classic rank collapse. Predict alpha_{-1} < 2.
  * Layer 6 SAE: 47% dead, EV=0.974, 1090 live features → rich
    representation. Predict alpha_6 in the healthy 2-6 range.

If the alpha-spectrum agrees, we have an independent confirmation and a
ranking of which blocks would benefit most from LoRA fine-tuning
(targeting alpha < 2 layers per Martin's recipe).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="brain-bzh/reve-base")
    ap.add_argument("--cache-dir", default="/data/derivatives/eeg_sae/hf_cache")
    ap.add_argument("--out-prefix", required=True,
                    help="Output prefix (no extension)")
    ap.add_argument("--min-evals", type=int, default=50,
                    help="WeightWatcher filter: skip layers with fewer "
                         "than N eigenvalues (too small to fit a power law).")
    args = ap.parse_args()

    print(f"[load] {args.model_id}", flush=True)
    import torch
    from transformers import AutoModel
    import weightwatcher as ww

    model = AutoModel.from_pretrained(
        args.model_id, trust_remote_code=True, cache_dir=args.cache_dir,
    ).eval()
    print(f"  loaded; param count = {sum(p.numel() for p in model.parameters()):,}",
          flush=True)

    print(f"[wW] analysing weight matrices …", flush=True)
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(min_evals=args.min_evals)
    summary = watcher.get_summary(details)
    print(f"  wW done; {len(details)} layers analysed", flush=True)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(out_prefix.with_suffix(".csv"), index=False)
    print(f"[done] per-layer details → {out_prefix.with_suffix('.csv')}",
          flush=True)

    # Per-block aggregation. REVE stores its 12 blocks under
    # ``transformer.layers[k]``; each block has an Attention (Q,K,V,proj)
    # and a FeedForward (2 linears) — typically 6 analysable matrices.
    block_summary: list[dict] = []
    if "longname" in details.columns:
        for k in range(12):
            mask = details["longname"].astype(str).str.contains(
                f"transformer.layers.{k}.", regex=False,
            )
            sub = details[mask]
            if len(sub) == 0:
                continue
            block_summary.append({
                "block": k,
                "n_matrices": int(len(sub)),
                "alpha_mean": float(sub["alpha"].mean()),
                "alpha_median": float(sub["alpha"].median()),
                "alpha_min": float(sub["alpha"].min()),
                "alpha_max": float(sub["alpha"].max()),
                "fraction_undertrained_alpha_lt_2": float((sub["alpha"] < 2).mean()),
                "fraction_healthy_2_6": float(
                    ((sub["alpha"] >= 2) & (sub["alpha"] <= 6)).mean()
                ),
                "alpha_weighted_mean": (
                    float((sub["alpha"] * sub["log_norm"]).sum()
                          / sub["log_norm"].sum())
                    if "log_norm" in sub.columns and sub["log_norm"].sum() != 0
                    else float("nan")
                ),
            })
    by_block = pd.DataFrame(block_summary)
    block_path = Path(str(out_prefix) + "_by_block.csv")
    by_block.to_csv(block_path, index=False)
    print(f"[done] per-block summary → {block_path}", flush=True)

    # Compact JSON summary readable on the head node without pandas
    js = {
        "model_id": args.model_id,
        "n_layers_analysed": int(len(details)),
        "global_alpha_mean": float(details["alpha"].mean()),
        "global_alpha_median": float(details["alpha"].median()),
        "fraction_undertrained_alpha_lt_2": float((details["alpha"] < 2).mean()),
        "fraction_healthy_2_6": float(
            ((details["alpha"] >= 2) & (details["alpha"] <= 6)).mean()
        ),
        "by_block": block_summary,
        "weightwatcher_summary": {
            k: float(v) if isinstance(v, (int, float)) else str(v)
            for k, v in summary.items()
        },
    }
    json_path = out_prefix.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(js, f, indent=2)
    print(f"[done] json → {json_path}", flush=True)

    # Stdout: the headline result
    print()
    print("=" * 60)
    print(f"Global mean alpha     : {js['global_alpha_mean']:.3f}")
    print(f"Global median alpha   : {js['global_alpha_median']:.3f}")
    print(f"Fraction alpha < 2    : {js['fraction_undertrained_alpha_lt_2']:.1%}")
    print(f"Fraction alpha in 2-6 : {js['fraction_healthy_2_6']:.1%}")
    print()
    print("Per-block alpha (REVE.transformer.layers.k):")
    print(f"  {'block':>5}  {'α-mean':>8}  {'α-med':>8}  "
          f"{'%α<2':>6}  {'%α 2-6':>7}  {'n':>3}")
    for row in block_summary:
        print(f"  {row['block']:>5}  {row['alpha_mean']:>8.3f}  "
              f"{row['alpha_median']:>8.3f}  "
              f"{row['fraction_undertrained_alpha_lt_2']:>6.1%}  "
              f"{row['fraction_healthy_2_6']:>7.1%}  "
              f"{row['n_matrices']:>3}")


if __name__ == "__main__":
    sys.exit(main())
