#!/usr/bin/env python
"""WeightWatcher (HT-SR) per-block analysis of ZUNA's transformer blocks.

ZUNA (Zyphra, mhough/zuna-base) is a 380M masked-diffusion EEG autoencoder
with a 16-block encoder and a 16-block decoder. The 6-model comparison in
``analyze_eegfm_weightwatcher.py`` pools all weight matrices into one global
alpha; that mixes encoder + decoder and isn't comparable to REVE's
per-block result (``analyze_reve_weightwatcher.py``).

This script loads ZUNA via the transformers custom-code path (needs the
``zuna`` package + deps on PYTHONPATH; see the sbatch), runs WeightWatcher
once over the whole model, then splits the per-layer details by
encoder/decoder and aggregates alpha per block. The encoder-only global
alpha is the figure to line up against REVE block-6 (alpha=3.6, our SAE
target) and the 6-model rankings.

Module naming (from named_modules):
    model.encoder.layers.{0..15}.attention.{wq,wk,wv,wo}   (512x1024 / 1024x512)
    model.encoder.layers.{0..15}.feed_forward.{w1,w2,w3}   (2816x1024 / 1024x2816)
    model.decoder.layers.{0..15}. ... (+ cross_attention)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def _names(details: pd.DataFrame) -> pd.Series:
    """The descriptive name column. We analyse a synthetic module built from
    the state_dict, so the original dotted key is preserved in ``origname``;
    fall back to WW's ``longname`` for a live-module analysis."""
    col = "origname" if "origname" in details.columns else "longname"
    return details[col].astype(str)


def _block_rows(details: pd.DataFrame, section: str, n_blocks: int) -> list[dict]:
    """Aggregate per-block alpha for ``.{section}.layers.{k}.`` matrices."""
    rows: list[dict] = []
    ln = _names(details)
    for k in range(n_blocks):
        mask = ln.str.contains(f"{section}.layers.{k}.", regex=False)
        sub = details[mask]
        if len(sub) == 0:
            continue
        a = sub["alpha"]
        rows.append({
            "section": section,
            "block": k,
            "n_matrices": int(len(sub)),
            "alpha_mean": float(a.mean()),
            "alpha_median": float(a.median()),
            "alpha_min": float(a.min()),
            "alpha_max": float(a.max()),
            "fraction_undertrained_alpha_lt_2": float((a < 2).mean()),
            "fraction_healthy_2_6": float(((a >= 2) & (a <= 6)).mean()),
            "alpha_weighted_mean": (
                float((a * sub["log_norm"]).sum() / sub["log_norm"].sum())
                if "log_norm" in sub.columns and sub["log_norm"].sum() != 0
                else float("nan")
            ),
        })
    return rows


def _section_stats(details: pd.DataFrame, section: str) -> dict:
    ln = _names(details)
    sub = details[ln.str.contains(f"{section}.layers.", regex=False)]
    if len(sub) == 0:
        return {}
    a = sub["alpha"]
    return {
        "n_matrices": int(len(sub)),
        "alpha_mean": float(a.mean()),
        "alpha_median": float(a.median()),
        "fraction_undertrained_alpha_lt_2": float((a < 2).mean()),
        "fraction_healthy_2_6": float(((a >= 2) & (a <= 6)).mean()),
        "fraction_overparam_alpha_gt_6": float((a > 6).mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-id", default="mhough/zuna-base")
    ap.add_argument("--cache-dir", default="/data/derivatives/eeg_sae/hf_cache")
    ap.add_argument("--out-prefix", required=True,
                    help="Output prefix (no extension)")
    ap.add_argument("--n-blocks", type=int, default=16)
    ap.add_argument("--min-evals", type=int, default=50)
    args = ap.parse_args()

    print(f"[load] {args.model_id} (safetensors state_dict)", flush=True)
    import torch
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    import weightwatcher as ww

    # WeightWatcher's generic torch walker chokes on ZUNA's custom lingua
    # modules (some register a sub-Module under a ``.weight`` attribute, so
    # ``layer.weight.data`` raises). We don't need the live module: the
    # state_dict keys already carry full typing
    # (``model.encoder.layers.{k}.attention.wq.weight`` …), so wrap each 2D
    # tensor as a plain bias-less nn.Linear and keep a safe→original name map
    # for the per-block split.
    weights_path = hf_hub_download(
        repo_id=args.model_id, filename="model.safetensors",
        cache_dir=args.cache_dir,
    )
    state = load_file(weights_path)

    namemap: dict[str, str] = {}

    class _W(nn.Module):
        def __init__(self, sd):
            super().__init__()
            kept = 0
            for name, t in sd.items():
                if t is None or not torch.is_tensor(t) or t.ndim < 2:
                    continue
                t_flat = t if t.ndim == 2 else t.reshape(t.shape[0], -1)
                layer = nn.Linear(t_flat.shape[1], t_flat.shape[0], bias=False)
                with torch.no_grad():
                    layer.weight.copy_(t_flat.float())
                safe = name.replace(".", "__")
                setattr(self, safe, layer)
                namemap[safe] = name
                kept += 1
            self._kept = kept

        def forward(self, x):
            return x

    model = _W(state).eval()
    n_kept = model._kept
    print(f"  wrapped {n_kept} 2D+ weight tensors "
          f"(param count = {sum(p.numel() for p in model.parameters()):,})",
          flush=True)

    print("[wW] analysing weight matrices …", flush=True)
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze(min_evals=args.min_evals)
    print(f"  wW done; {len(details)} layers analysed", flush=True)

    # Recover the original dotted name from WW's longname (which is the safe
    # attr name, possibly with extra qualifiers).
    def _to_orig(longname: str) -> str:
        s = str(longname)
        if s in namemap:
            return namemap[s]
        for safe, orig in namemap.items():
            if safe in s:
                return orig
        return s

    if "longname" in details.columns:
        details["origname"] = details["longname"].map(_to_orig)

    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    details.to_csv(out_prefix.with_suffix(".csv"), index=False)
    print(f"[done] per-layer details → {out_prefix.with_suffix('.csv')}",
          flush=True)

    enc_blocks = _block_rows(details, "encoder", args.n_blocks)
    dec_blocks = _block_rows(details, "decoder", args.n_blocks)
    by_block = pd.DataFrame(enc_blocks + dec_blocks)
    block_path = Path(str(out_prefix) + "_by_block.csv")
    by_block.to_csv(block_path, index=False)
    print(f"[done] per-block summary → {block_path}", flush=True)

    enc = _section_stats(details, "encoder")
    dec = _section_stats(details, "decoder")
    js = {
        "model_id": args.model_id,
        "n_layers_analysed": int(len(details)),
        "global_alpha_mean": float(details["alpha"].mean()),
        "global_alpha_median": float(details["alpha"].median()),
        "encoder": enc,
        "decoder": dec,
        "by_block": enc_blocks + dec_blocks,
    }
    json_path = out_prefix.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(js, f, indent=2)
    print(f"[done] json → {json_path}", flush=True)

    print()
    print("=" * 64)
    print(f"Global mean alpha (enc+dec): {js['global_alpha_mean']:.3f}  "
          f"median {js['global_alpha_median']:.3f}")
    if enc:
        print(f"ENCODER  α-mean={enc['alpha_mean']:.3f}  "
              f"%α<2={enc['fraction_undertrained_alpha_lt_2']:.1%}  "
              f"%[2,6]={enc['fraction_healthy_2_6']:.1%}  "
              f"(n={enc['n_matrices']})")
    if dec:
        print(f"DECODER  α-mean={dec['alpha_mean']:.3f}  "
              f"%α<2={dec['fraction_undertrained_alpha_lt_2']:.1%}  "
              f"%[2,6]={dec['fraction_healthy_2_6']:.1%}  "
              f"(n={dec['n_matrices']})")
    print()
    print("Per-block alpha:")
    print(f"  {'sec':>7}  {'blk':>3}  {'α-mean':>8}  {'α-med':>8}  "
          f"{'%α<2':>6}  {'%2-6':>6}  {'n':>3}")
    for row in enc_blocks + dec_blocks:
        print(f"  {row['section']:>7}  {row['block']:>3}  "
              f"{row['alpha_mean']:>8.3f}  {row['alpha_median']:>8.3f}  "
              f"{row['fraction_undertrained_alpha_lt_2']:>6.1%}  "
              f"{row['fraction_healthy_2_6']:>6.1%}  {row['n_matrices']:>3}")


if __name__ == "__main__":
    sys.exit(main())
