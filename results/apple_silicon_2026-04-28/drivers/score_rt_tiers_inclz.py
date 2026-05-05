#!/usr/bin/env python3
"""Score the INCLUSIVE-cumz RT tier replicas with single-rep filter.

Cells expected to match paper anchors:
  RT_paper_Fast_pst5_inclz           → 36% top-1 (paper Fast)
  RT_paper_Slow_pst20_inclz          → 58% top-1 (paper Slow)
  RT_paper_EndOfRun_pst_None_inclz   → 66% top-1 (paper End-of-run)
"""
from __future__ import annotations

import json
import sys
import types
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
RT_MINDEYE = Path("/Users/mhough/Workspace/rt_mindEye2/src")

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
sys.path.insert(0, str(Path(__file__).parent))

import mindeye_retrieval_eval as M
M.RTCLOUD_MINDEYE = RT_MINDEYE
from run_retrieval_local import compute_gt_mps

warnings.filterwarnings("ignore")

PREREG = LOCAL / "task_2_1_betas" / "prereg"
CKPT = LOCAL / "rt3t" / "data" / "model" / "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
CACHE = LOCAL / "task_2_1_betas" / "gt_cache"
RESULTS_PATH = LOCAL / "task_2_1_betas" / "retrieval_results_v2.json"

device = "mps" if torch.backends.mps.is_available() else "cpu"

CELLS = [
    ("RT_paper_Fast_pst5_inclz",         "ses-03", 0.36),
    ("RT_paper_Slow_pst20_inclz",        "ses-03", 0.58),
    ("RT_paper_EndOfRun_pst_None_inclz", "ses-03", 0.66),
]


def filter_first_rep(betas, trial_ids):
    keep_idx, seen = [], set()
    for i, t in enumerate(trial_ids):
        if not str(t).startswith("all_stimuli/special515/"):
            continue
        if t in seen:
            continue
        seen.add(t)
        keep_idx.append(i)
    return betas[keep_idx], [str(trial_ids[i]) for i in keep_idx]


def score_cell(name, ses, paper_anchor):
    print(f"\n=== scoring {name} ===")
    raw_betas = np.load(PREREG / f"{name}_{ses}_betas.npy")
    raw_ids = np.load(PREREG / f"{name}_{ses}_trial_ids.npy")
    print(f"  raw betas: {raw_betas.shape}  mean={raw_betas.mean():.3f} std={raw_betas.std():.3f} max={np.abs(raw_betas).max():.2f}")

    test_betas, test_ids = filter_first_rep(raw_betas, raw_ids)
    print(f"  filtered first-rep: {test_betas.shape}, {len(set(test_ids))} unique")

    image_paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(n).name
                   for n in test_ids]
    gt = compute_gt_mps(image_paths, device=device, cache_dir=CACHE)

    model, ss, se = M.load_mindeye(Path(CKPT), n_voxels=2792, device=device)
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(test_betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            backbone_out = model.backbone(voxel_ridge)
            clip_voxels = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
            out.append(clip_voxels.float().cpu().numpy())
    pred = np.concatenate(out, axis=0).reshape(-1, ss, se)

    img_to_idx = {im: i for i, im in enumerate(test_ids)}
    trial_idx = np.array([img_to_idx[t] for t in test_ids])
    sim = M.cosine_sim_tokens(pred, gt)
    top1 = M.top_k_retrieval(sim, trial_idx, k=1)
    top5 = M.top_k_retrieval(sim, trial_idx, k=5)
    print(f"  TOP-1: {top1*100:.2f}%  TOP-5: {top5*100:.2f}%  (paper: {paper_anchor*100:.0f}%)")

    return {
        "condition": f"{name}_singleRep",
        "session": ses,
        "n_test_trials": int(test_betas.shape[0]),
        "n_unique_images": int(len(set(test_ids))),
        "top1_image_retrieval": float(top1),
        "top5_image_retrieval": float(top5),
        "paper_anchor_top1": float(paper_anchor),
        "delta_vs_paper_pp": float(top1 - paper_anchor) * 100,
        "notes": "INCLUSIVE causal cum-z (arr[:i+1]); paper §2.7 first-rep filter",
    }


def main():
    results = json.loads(RESULTS_PATH.read_text())
    for name, ses, anchor in CELLS:
        if not (PREREG / f"{name}_{ses}_betas.npy").exists():
            print(f"  SKIP {name} — betas not on disk")
            continue
        res = score_cell(name, ses, anchor)
        results = [r for r in results if r.get("condition") != res["condition"]]
        results.append(res)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))

    print("\n=== INCLUSIVE cum-z ladder reproduction ===")
    print(f"{'cell':45s}  {'top1':>7s}  {'paper':>7s}  {'delta':>7s}")
    for r in results:
        cond = r.get("condition", "")
        if cond.endswith("_inclz_singleRep"):
            print(f"{cond:45s}  "
                  f"{r['top1_image_retrieval']*100:6.2f}%  "
                  f"{r['paper_anchor_top1']*100:6.0f}%  "
                  f"{r['delta_vs_paper_pp']:+6.1f}pp")


if __name__ == "__main__":
    main()
