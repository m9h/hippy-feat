#!/usr/bin/env python3
"""Retrieval eval with FIXED cumulative-z-score (paper §2.5) + skip for already-z'd cells.

Differences vs run_retrieval_local.py:
  • Replaces session-level z-score (data leak: test trials see their own stats)
    with causal cumulative z-score: trial i uses stats from trials 0..i-1 only.
  • For cells with `_already_cumz=True` (cells 11, 12: paper RT/Offline replicas
    that already cum-z'd inside their own driver), skips re-z-scoring entirely
    and just filters to special515.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
RT_MINDEYE = Path("/Users/mhough/Workspace/rt_mindEye2/src")

# stubs for diffusers.models.vae.Decoder + generative_models.sgm.* ----------
import diffusers, diffusers.models  # noqa
vae_mod = types.ModuleType("diffusers.models.vae")
class _DecStub(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): return None
vae_mod.Decoder = _DecStub
sys.modules["diffusers.models.vae"] = vae_mod
diffusers.models.vae = vae_mod

gm = types.ModuleType("generative_models")
sgm = types.ModuleType("generative_models.sgm")
sgm_util = types.ModuleType("generative_models.sgm.util")
sgm_modules = types.ModuleType("generative_models.sgm.modules")
sgm_enc = types.ModuleType("generative_models.sgm.modules.encoders")
sgm_enc_mods = types.ModuleType("generative_models.sgm.modules.encoders.modules")
sgm_util.append_dims = lambda x, n: x
class _Stub(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): return None
sgm_enc_mods.FrozenOpenCLIPImageEmbedder = _Stub
sgm_enc_mods.FrozenOpenCLIPEmbedder2 = _Stub
for mod in [gm, sgm, sgm_util, sgm_modules, sgm_enc, sgm_enc_mods]:
    sys.modules[mod.__name__] = mod
sys.modules["sgm"] = sgm
sys.path.insert(0, str(RT_MINDEYE))
sys.path.insert(0, str(REPO / "scripts"))

import mindeye_retrieval_eval as M  # noqa
M.RTCLOUD_MINDEYE = RT_MINDEYE


# ---- causal cumulative z-score ---------------------------------------------

ALREADY_CUMZ = {"RT_paper_replica_partial", "RT_paper_replica_full",
                "Offline_paper_replica_full"}


def _cumulative_zscore(betas: np.ndarray) -> np.ndarray:
    """Per-trial causal z-score: z[i,v] = (β[i,v] - μ_{0..i-1,v}) / σ_{0..i-1,v}.

    Trial 0 has no baseline; we keep it as 0 (paper §2.5 effectively does this
    by skipping the first trial in retrieval scoring or by using a tiny prior).
    Trial 1 uses just trial 0's value as μ; σ undefined → use 1.
    """
    n, V = betas.shape
    out = np.zeros_like(betas, dtype=np.float32)
    cumsum = np.zeros(V, dtype=np.float64)
    cumsum_sq = np.zeros(V, dtype=np.float64)
    for i in range(n):
        if i == 0:
            out[0] = 0.0  # no baseline yet
        else:
            mu = cumsum / i
            var = cumsum_sq / i - mu ** 2
            sd = np.sqrt(np.maximum(var, 1e-12))
            out[i] = ((betas[i] - mu) / (sd + 1e-8)).astype(np.float32)
        cumsum += betas[i]
        cumsum_sq += betas[i].astype(np.float64) ** 2
    return out


def filter_special515_v2(betas, ids, condition: str):
    """Z-score correctly per condition, then filter to special515 trials."""
    if condition in ALREADY_CUMZ:
        # Already cum-z'd inside the cell driver — don't re-z, just filter.
        z = betas
        print(f"  [{condition}] already cum-z'd inside driver — skip re-z")
    else:
        z = _cumulative_zscore(betas)
        print(f"  [{condition}] applied causal cumulative z-score "
              f"(trial-i uses 0..i-1 stats only)")
    mask = np.array([str(t).startswith("all_stimuli/special515/") for t in ids])
    z_test = z[mask]
    ids_test = np.asarray([str(t) for t in ids[mask]])
    unique_images = np.array(sorted(set(ids_test)))
    print(f"  test trials: {z_test.shape[0]}  unique images: {len(unique_images)}")
    return z_test, ids_test, unique_images


# ---- MPS autocast wrappers (same as v1) ------------------------------------

def _autocast_for(device: str):
    if device.startswith("mps"):
        return torch.amp.autocast("mps", dtype=torch.float16)
    if device == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    return torch.amp.autocast("cpu", dtype=torch.bfloat16)


def predict_clip_mps(model, betas, device, clip_seq_dim=256, clip_emb_dim=1664):
    out = []
    b = torch.from_numpy(betas.astype(np.float32)).to(device).unsqueeze(1)
    with torch.no_grad(), _autocast_for(device):
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            backbone_out = model.backbone(voxel_ridge)
            clip_voxels = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
            out.append(clip_voxels.float().cpu().numpy())
    preds = np.concatenate(out, axis=0)
    return preds.reshape(-1, clip_seq_dim, clip_emb_dim)


# ---- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--betas-root", default=str(LOCAL / "task_2_1_betas" / "prereg"))
    ap.add_argument("--checkpoint", default=str(
        LOCAL / "rt3t" / "data" / "model" /
        "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
    ))
    ap.add_argument("--out-json", default=str(LOCAL / "task_2_1_betas" / "retrieval_results_v2.json"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")
    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    betas, ids = M.load_condition_betas(args.condition, args.session,
                                         betas_root=Path(args.betas_root))
    test_betas, test_ids, unique_images = filter_special515_v2(betas, ids, args.condition)

    image_paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(n).name
                   for n in unique_images]
    img_to_idx = {str(u): i for i, u in enumerate(unique_images)}
    trial_idx = np.array([img_to_idx[t] for t in test_ids])

    cache_dir = LOCAL / "task_2_1_betas" / "gt_cache"
    # reuse cache produced by v1 driver
    from run_retrieval_local import compute_gt_mps
    gt_emb = compute_gt_mps(image_paths, device=device, cache_dir=cache_dir)

    model, ss, se = M.load_mindeye(Path(args.checkpoint),
                                    n_voxels=test_betas.shape[1], device=device)
    pred_emb = predict_clip_mps(model, test_betas, device=device,
                                clip_seq_dim=ss, clip_emb_dim=se)

    sim = M.cosine_sim_tokens(pred_emb, gt_emb)
    top1_image = M.top_k_retrieval(sim, trial_idx, k=1)
    top5_image = M.top_k_retrieval(sim, trial_idx, k=5)
    print(f"  top-1: {top1_image:.4f}   top-5: {top5_image:.4f}")

    out = {
        "condition": args.condition, "session": args.session,
        "n_test_trials": int(test_betas.shape[0]),
        "n_unique_images": int(len(unique_images)),
        "top1_image_retrieval": float(top1_image),
        "top5_image_retrieval": float(top5_image),
        "z_score_method": "skipped (already cum-z'd)" if args.condition in ALREADY_CUMZ
                          else "causal cumulative",
    }
    out_path = Path(args.out_json)
    existing = json.loads(out_path.read_text()) if out_path.exists() else []
    if isinstance(existing, dict): existing = [existing]
    existing.append(out)
    out_path.write_text(json.dumps(existing, indent=2))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main()
