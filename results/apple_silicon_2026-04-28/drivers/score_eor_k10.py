#!/usr/bin/env python3
"""Score the RT_paper_EoR_K10_inclz cell with single-rep filter.

Compares to:
  - 56% baseline (RT_paper_EndOfRun_pst_None_inclz, no GLMdenoise)
  - 66% paper End-of-run anchor
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

CELL = "RT_paper_EoR_K10_inclz"
SES = "ses-03"
PAPER_ANCHOR = 0.66


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


print(f"=== scoring {CELL} ===")
raw_betas = np.load(PREREG / f"{CELL}_{SES}_betas.npy")
raw_ids = np.load(PREREG / f"{CELL}_{SES}_trial_ids.npy")
print(f"  raw betas: {raw_betas.shape}  mean={raw_betas.mean():.3f} "
      f"std={raw_betas.std():.3f} max={np.abs(raw_betas).max():.2f}")

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

print()
print(f"  TOP-1: {top1*100:.2f}%   TOP-5: {top5*100:.2f}%")
print(f"  paper End-of-run anchor: {PAPER_ANCHOR*100:.0f}%")
print(f"  delta vs paper:    {(top1 - PAPER_ANCHOR)*100:+.1f}pp")
print(f"  delta vs no-K=10 EoR baseline (56%):  {(top1 - 0.56)*100:+.1f}pp")

result = {
    "condition": f"{CELL}_singleRep",
    "session": SES,
    "n_test_trials": int(test_betas.shape[0]),
    "n_unique_images": int(len(set(test_ids))),
    "top1_image_retrieval": float(top1),
    "top5_image_retrieval": float(top5),
    "paper_anchor_top1": float(PAPER_ANCHOR),
    "delta_vs_paper_pp": float(top1 - PAPER_ANCHOR) * 100,
    "delta_vs_baseline_pp": float(top1 - 0.56) * 100,
    "notes": "EoR + GLMdenoise K=10 (PCA noise components passed as nilearn confounds); "
             "INCLUSIVE causal cum-z; single-rep filter",
}

results = json.loads(RESULTS_PATH.read_text())
results = [r for r in results if r.get("condition") != result["condition"]]
results.append(result)
RESULTS_PATH.write_text(json.dumps(results, indent=2))
print(f"\nappended to {RESULTS_PATH.name}")
