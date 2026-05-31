#!/usr/bin/env python
"""WeightWatcher (HT-SR) analysis across NeuralBench-EEG's foundation models.

Pulls each of the 6 FMs and runs the same alpha-spectrum analysis we already
have for REVE (`analyze_reve_weightwatcher.py`). One CSV + JSON per model;
plus a combined summary table for cross-model comparison.

Models (per NeuralBench-EEG v1.0):
    reve     — brain-bzh/reve-base               (gated; needs HF_TOKEN)
    labram   — braindecode/labram-pretrained      (braindecode)
    bendr    — braindecode/braindecode-bendr      (braindecode)
    biot     — braindecode/biot-pretrained-six-datasets-18chs (braindecode)
    cbramod  — braindecode/cbramod-pretrained     (braindecode)
    luna     — PulpBio/LUNA                       (PulpBio; via neuraltrain.NtLuna)

What's compared per model:
    * global mean alpha + median alpha
    * % under-trained (alpha < 2) — directly tells us LoRA fine-tuning yield
    * % healthy (2 <= alpha <= 6)
    * largest outlier (max alpha — over-parameterised matrices)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# Each entry: (model_id, loader_fn) where loader_fn returns a torch.nn.Module.
# Loaders are local-imports so the script doesn't fail at parse-time if some
# class is missing.

def _load_reve(model_id: str, cache_dir: str):
    from transformers import AutoModel
    return AutoModel.from_pretrained(
        model_id, trust_remote_code=True, cache_dir=cache_dir,
    ).eval()


def _strip_parametrizations(model):
    """BENDR (and any model using ``nn.utils.weight_norm``) attaches a
    ``ParametrizationList`` to weights that WW can't read as a regular tensor.
    Materialise the post-parametrization weight + detach so WW sees a plain
    ``nn.Parameter``.
    """
    import torch
    for mod in model.modules():
        if hasattr(mod, "parametrizations"):
            for name in list(getattr(mod, "parametrizations", {}).keys()):
                try:
                    torch.nn.utils.parametrize.remove_parametrizations(
                        mod, name, leave_parametrized=True,
                    )
                except Exception:
                    pass
    return model


def _load_braindecode_model(cls_name: str, model_id: str, cache_dir: str,
                              **extra_kwargs):
    """Generic braindecode loader for models exposing .from_pretrained().

    Some classes (CBraMod) require ``n_outputs`` even for analysis-only
    loads — passed via ``extra_kwargs``.
    """
    from braindecode import models as bm
    cls = getattr(bm, cls_name)
    try:
        model = cls.from_pretrained(model_id, cache_dir=cache_dir,
                                     **extra_kwargs)
    except TypeError:
        import os
        if cache_dir:
            os.environ.setdefault("HF_HOME", cache_dir)
        model = cls.from_pretrained(model_id, **extra_kwargs)
    model = _strip_parametrizations(model)
    return model.eval()


def _state_dict_wrapper(state_dict):
    """Wrap a state_dict into a synthetic ``nn.Module`` of one ``nn.Linear``
    per 2D+ weight tensor, so WW can walk it without needing the original
    model architecture. Convs are flattened to (out, prod(in_dims)).

    Used when the upstream architecture default doesn't match the released
    checkpoint dimensions (CBraMod, LUNA), or when constructing the right
    config would require chasing more upstream metadata than is worth it.
    """
    import torch
    import torch.nn as nn

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
                    layer.weight.copy_(t_flat)
                # Sanitize name for setattr
                safe = name.replace(".", "__")
                setattr(self, safe, layer)
                kept += 1
            self._kept = kept
        def forward(self, x):
            return x

    return _W(state_dict)


def _load_from_hf_safetensors(model_id: str, cache_dir: str,
                                filename: str = "model.safetensors"):
    """Direct safetensors download → state_dict wrapper.

    Architecture-agnostic — just analyses every 2D+ weight in the file.
    """
    from huggingface_hub import hf_hub_download
    import torch

    weights_path = hf_hub_download(
        repo_id=model_id, filename=filename, cache_dir=cache_dir,
    )
    if weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state = load_file(weights_path)
    else:
        state = torch.load(weights_path, map_location="cpu")
    return _state_dict_wrapper(state).eval()


def _load_zuna(model_id: str, cache_dir: str):
    """ZUNA (Zyphra 380M masked diffusion AE), transformers custom-code wrap.

    Prefer ``AutoModel(trust_remote_code)`` so WW sees real per-layer typing
    (encoder/decoder attention vs MLP). Fall back to raw-safetensors ESD if
    the custom ``modeling_zuna.py`` deps aren't importable in this container.
    """
    try:
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            model_id, trust_remote_code=True, cache_dir=cache_dir,
        ).eval()
    except Exception as e:
        print(f"  [zuna AutoModel failed: {type(e).__name__}: {e}; "
              f"falling back to safetensors ESD]", flush=True)
        return _load_from_hf_safetensors(model_id, cache_dir,
                                          filename="model.safetensors")


def _load_cbramod(model_id: str, cache_dir: str):
    return _load_from_hf_safetensors(model_id, cache_dir,
                                       filename="model.safetensors")


def _load_luna(model_id: str, cache_dir: str):
    return _load_from_hf_safetensors(model_id, cache_dir,
                                       filename="LUNA_large.safetensors")


MODELS = {
    "reve":    ("brain-bzh/reve-base",                              _load_reve),
    "labram":  ("braindecode/labram-pretrained",                    lambda i, c: _load_braindecode_model("Labram",  i, c)),
    "bendr":   ("braindecode/braindecode-bendr",                    lambda i, c: _load_braindecode_model("BENDR",   i, c)),
    "biot":    ("braindecode/biot-pretrained-six-datasets-18chs",   lambda i, c: _load_braindecode_model("BIOT",    i, c)),
    "cbramod": ("braindecode/cbramod-pretrained",                   _load_cbramod),
    "luna":    ("PulpBio/LUNA",                                     _load_luna),
    "zuna":    ("mhough/zuna-base",                                 _load_zuna),
}


def analyse_model(name: str, model_id: str, loader, cache_dir: str,
                  out_dir: Path, min_evals: int = 50) -> dict:
    print(f"\n=== {name} ({model_id}) ===", flush=True)
    import weightwatcher as ww

    try:
        model = loader(model_id, cache_dir)
    except Exception as e:
        print(f"  [load failed] {type(e).__name__}: {e}", flush=True)
        return {"model": name, "status": "load_failed", "error": str(e)[:240]}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded; params = {n_params:,}", flush=True)

    try:
        watcher = ww.WeightWatcher(model=model)
        details = watcher.analyze(min_evals=min_evals)
    except Exception as e:
        print(f"  [WW failed] {type(e).__name__}: {e}", flush=True)
        return {"model": name, "status": "ww_failed", "error": str(e)[:240]}

    out_csv = out_dir / f"{name}.csv"
    details.to_csv(out_csv, index=False)

    alpha = details["alpha"] if "alpha" in details.columns else pd.Series([])
    summary = {
        "model": name,
        "model_id": model_id,
        "status": "ok",
        "n_params": n_params,
        "n_layers": int(len(details)),
        "alpha_mean": float(alpha.mean()) if len(alpha) else None,
        "alpha_median": float(alpha.median()) if len(alpha) else None,
        "alpha_min": float(alpha.min()) if len(alpha) else None,
        "alpha_max": float(alpha.max()) if len(alpha) else None,
        "fraction_undertrained_alpha_lt_2": float((alpha < 2).mean()) if len(alpha) else None,
        "fraction_healthy_2_6": float(((alpha >= 2) & (alpha <= 6)).mean()) if len(alpha) else None,
        "fraction_overparam_alpha_gt_6": float((alpha > 6).mean()) if len(alpha) else None,
    }
    print(f"  α-mean={summary['alpha_mean']:.3f}  "
          f"%α<2={summary['fraction_undertrained_alpha_lt_2']:.1%}  "
          f"%[2,6]={summary['fraction_healthy_2_6']:.1%}  "
          f"%α>6={summary['fraction_overparam_alpha_gt_6']:.1%}",
          flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="/data/derivatives/eeg_sae/hf_cache")
    ap.add_argument("--out-dir", default="/data/derivatives/eeg_sae/weightwatcher")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                    help=f"Subset of: {list(MODELS.keys())}")
    ap.add_argument("--min-evals", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for name in args.models:
        if name not in MODELS:
            print(f"[skip] unknown model {name!r}", flush=True)
            continue
        model_id, loader = MODELS[name]
        summaries.append(
            analyse_model(name, model_id, loader, args.cache_dir, out_dir,
                          min_evals=args.min_evals)
        )

    # Cross-model summary table — merge with any existing summary by model
    # name so models can be added incrementally without clobbering prior rows.
    summary_path = out_dir / "all_models_summary.csv"
    summary_json = out_dir / "all_models_summary.json"
    merged: dict[str, dict] = {}
    if summary_json.exists():
        try:
            for row in json.loads(summary_json.read_text()):
                merged[row["model"]] = row
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    for row in summaries:
        merged[row["model"]] = row
    ordered = [merged[n] for n in MODELS if n in merged]
    ordered += [r for n, r in merged.items() if n not in MODELS]
    sdf = pd.DataFrame(ordered)
    sdf.to_csv(summary_path, index=False)
    summary_json.write_text(json.dumps(ordered, indent=2))

    print()
    print("=" * 75)
    print("Cross-model summary")
    print("=" * 75)
    cols = ["model", "n_params", "n_layers", "alpha_mean", "alpha_median",
            "fraction_undertrained_alpha_lt_2", "fraction_healthy_2_6"]
    have = [c for c in cols if c in sdf.columns]
    print(sdf[have].to_string(index=False, float_format=lambda v:
            f"{v:.3f}" if isinstance(v, float) else str(v)))
    print(f"\n[done] {summary_path}\n[done] {summary_json}")


if __name__ == "__main__":
    sys.exit(main())
