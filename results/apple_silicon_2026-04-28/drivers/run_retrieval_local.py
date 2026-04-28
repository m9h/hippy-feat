#!/usr/bin/env python3
"""Run mindeye_retrieval_eval against a chosen condition on Apple MPS.

Usage:
    python run_retrieval_local.py --condition VariantG_glover_rtm
    python run_retrieval_local.py --condition Offline_paper_replica_full

Stubs out generative_models.sgm and diffusers.models.vae.Decoder (only used by
the unCLIP / blurry-recon paths; we do retrieval-only). Re-binds RTCLOUD_MINDEYE
to ~/Workspace/rt_mindEye2/src and routes autocast through the MPS device.
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

# ---- module stubs (must run before models import) -------------------------

# diffusers.models.vae.Decoder is only used for blurry_recon=True — stub it.
import diffusers, diffusers.models  # noqa
vae_mod = types.ModuleType("diffusers.models.vae")
class _DecStub(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): return None
vae_mod.Decoder = _DecStub
sys.modules["diffusers.models.vae"] = vae_mod
diffusers.models.vae = vae_mod

# generative_models.sgm.* stubs (utils_mindeye imports FrozenOpenCLIP*).
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

# Patch RTCLOUD_MINDEYE before importing the eval module.
import mindeye_retrieval_eval as M  # noqa
M.RTCLOUD_MINDEYE = RT_MINDEYE


# ---- MPS-friendly autocast --------------------------------------------------

def _autocast_for(device: str):
    if device.startswith("mps"):
        # MPS supports torch.amp.autocast as of torch 2.x with fp16/bfloat16.
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


def compute_gt_mps(image_paths, device, cache_dir):
    """OpenCLIP ViT-bigG/14 token embeddings on MPS-friendly autocast."""
    import hashlib
    from PIL import Image
    import open_clip

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = list(image_paths)
    cache_keys = [
        cache_dir / f"{p.stem}_{hashlib.md5(str(p).encode()).hexdigest()[:8]}.npy"
        for p in paths
    ]
    if all(k.exists() for k in cache_keys):
        print(f"  loaded all {len(paths)} GT embeddings from cache ({cache_dir})")
        return np.stack([np.load(k) for k in cache_keys])

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device
    )
    model.eval()
    model.visual.output_tokens = True

    embeddings = []
    with torch.no_grad(), _autocast_for(device):
        for p, key in zip(paths, cache_keys):
            img = Image.open(p).convert("RGB")
            inp = preprocess(img).unsqueeze(0).to(device)
            out = model.encode_image(inp)
            tokens = out[1] if isinstance(out, tuple) else out
            arr = tokens.float().cpu().numpy()[0]
            np.save(key, arr)
            embeddings.append(arr)
    return np.stack(embeddings, axis=0)


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True)
    ap.add_argument("--session", default="ses-03")
    ap.add_argument("--betas-root", default=str(LOCAL / "task_2_1_betas" / "prereg"))
    ap.add_argument("--checkpoint", default=str(
        LOCAL / "rt3t" / "data" / "model" /
        "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
    ))
    ap.add_argument("--stimuli-dir", default=str(LOCAL / "rt3t" / "data" / "all_stimuli" / "special515"))
    ap.add_argument("--out-json", default=str(LOCAL / "task_2_1_betas" / "retrieval_results.json"))
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    warnings.filterwarnings("ignore")

    device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"
    print(f"[device] {device}  torch {torch.__version__}")

    print(f"\n[1/5] Loading betas for {args.condition}")
    betas, ids = M.load_condition_betas(args.condition, args.session, betas_root=Path(args.betas_root))
    print(f"  betas: {betas.shape}  ids: {ids.shape}")
    test_betas, test_ids, unique_images = M.filter_to_special515(betas, ids)

    image_paths = [Path(args.stimuli_dir) / Path(n).name for n in unique_images]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(f"  WARN: {len(missing)} stimuli missing — first: {missing[0]}")
    img_to_idx = {str(u): i for i, u in enumerate(unique_images)}
    trial_idx = np.array([img_to_idx[t] for t in test_ids])

    print(f"\n[2/5] OpenCLIP GT for {len(image_paths)} images")
    cache_dir = LOCAL / "task_2_1_betas" / "gt_cache"
    t0 = time.time()
    gt_emb = compute_gt_mps(image_paths, device=device, cache_dir=cache_dir)
    print(f"  gt shape: {gt_emb.shape}  ({time.time()-t0:.1f}s)")

    print(f"\n[3/5] Loading MindEye checkpoint")
    model, ss, se = M.load_mindeye(Path(args.checkpoint),
                                    n_voxels=test_betas.shape[1], device=device)

    print(f"\n[4/5] Forward pass for {test_betas.shape[0]} test trials")
    t0 = time.time()
    pred_emb = predict_clip_mps(model, test_betas, device=device,
                                clip_seq_dim=ss, clip_emb_dim=se)
    print(f"  pred shape: {pred_emb.shape}  ({time.time()-t0:.1f}s)")

    print(f"\n[5/5] Retrieval")
    sim = M.cosine_sim_tokens(pred_emb, gt_emb)
    top1_image = M.top_k_retrieval(sim, trial_idx, k=1)
    top5_image = M.top_k_retrieval(sim, trial_idx, k=5)
    top1_brain = M.top_k_retrieval(sim.T, np.arange(sim.shape[1])[:, None].repeat(
        max(1, len(trial_idx) // sim.shape[1]), axis=1
    ).flatten()[:sim.T.shape[0]] if False else trial_idx, k=1)
    print(f"  top-1 image: {top1_image:.4f}")
    print(f"  top-5 image: {top5_image:.4f}")

    out = {
        "condition": args.condition, "session": args.session,
        "device": device, "n_test_trials": int(test_betas.shape[0]),
        "n_unique_images": int(len(unique_images)),
        "top1_image_retrieval": float(top1_image),
        "top5_image_retrieval": float(top5_image),
        "checkpoint": str(args.checkpoint),
    }
    out_path = Path(args.out_json)
    if out_path.exists():
        existing = json.loads(out_path.read_text())
        if isinstance(existing, dict): existing = [existing]
    else:
        existing = []
    existing.append(out)
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"\nResults appended to {out_path}")


if __name__ == "__main__":
    main()
