#!/usr/bin/env python3
"""Bootstrap N=50 retrieval CIs for the 4 paper tiers + variants.

For each tier, we have 50 unique test images. Resample with replacement 5000x,
compute top-1 retrieval per resample, take percentile CIs. Report whether the
paper anchor falls inside our 95% CI — if so, our point estimate is statistically
indistinguishable from theirs.
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
device = "mps" if torch.backends.mps.is_available() else "cpu"

CELLS = [
    ("RT_paper_Fast_pst5_inclz",            "ses-03", 0.36, "Fast"),
    ("RT_paper_Slow_pst20_inclz",           "ses-03", 0.58, "Slow"),
    ("RT_paper_EndOfRun_pst_None_inclz",    "ses-03", 0.66, "End-of-run"),
    ("Canonical_GLMsingle_OfflineFull",     "ses-03", 0.76, "Offline-3T"),
]
N_BOOT = 5000


def filter_first_rep(betas, trial_ids):
    keep_idx, seen = [], set()
    for i, t in enumerate(trial_ids):
        ts = str(t)
        if not (ts.startswith("all_stimuli/special515/") or ts.startswith("special515/")):
            continue
        if ts in seen:
            continue
        seen.add(ts)
        keep_idx.append(i)
    return betas[keep_idx], [str(trial_ids[i]) for i in keep_idx]


# Load model once
model, ss, se = M.load_mindeye(Path(CKPT), n_voxels=2792, device=device)


def get_pred_and_gt(name, ses):
    raw_betas = np.load(PREREG / f"{name}_{ses}_betas.npy")
    raw_ids = np.load(PREREG / f"{name}_{ses}_trial_ids.npy")

    if "Canonical_GLMsingle" in name:
        # Already filtered to 50 unique images by score_canonical_glmsingle.py
        test_betas, test_ids = raw_betas, [str(t) for t in raw_ids]
    else:
        test_betas, test_ids = filter_first_rep(raw_betas, raw_ids)

    image_paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(n).name
                   for n in test_ids]
    gt = compute_gt_mps(image_paths, device=device, cache_dir=CACHE)

    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(test_betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(b.shape[0]):
            voxel_ridge = model.ridge(b[i:i+1], 0)
            backbone_out = model.backbone(voxel_ridge)
            clip_voxels = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
            out.append(clip_voxels.float().cpu().numpy())
    pred = np.concatenate(out, axis=0).reshape(-1, ss, se)
    return pred, gt


def bootstrap_top1(pred, gt, n_boot=N_BOOT, seed=42):
    """Resample the QUERIES with replacement; gallery stays fixed (50 unique
    images). Each query's top-1 correctness is well-defined against the full
    50-image gallery, so resampling just averages this binary correctness over
    a synthetic 50-query test set."""
    rng = np.random.default_rng(seed)
    n = pred.shape[0]
    sim = M.cosine_sim_tokens(pred, gt)            # (n, n) — full gallery = full pred set
    correct = np.array([np.argmax(sim[i]) == i for i in range(n)], dtype=np.float32)
    base = correct.mean()

    boots = np.empty(n_boot, dtype=np.float32)
    for b_i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b_i] = correct[idx].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return base, lo, hi, boots


print(f"\n=== Bootstrap N=50 retrieval CIs ({N_BOOT} resamples) ===\n")
print(f"{'Tier':12s} {'Cell':40s} {'Point':>7s} {'95% CI':>16s} {'Paper':>7s} {'In CI?':>7s}")
results_summary = []
for cell, ses, anchor, tier in CELLS:
    if not (PREREG / f"{cell}_{ses}_betas.npy").exists():
        print(f"  SKIP {cell}")
        continue
    pred, gt = get_pred_and_gt(cell, ses)
    base, lo, hi, boots = bootstrap_top1(pred, gt)
    in_ci = lo <= anchor <= hi
    print(f"{tier:12s} {cell:40s} {base*100:6.1f}% [{lo*100:5.1f}-{hi*100:5.1f}] {anchor*100:6.0f}% {'YES' if in_ci else 'no':>7s}")
    results_summary.append({
        "tier": tier, "cell": cell, "point_top1": float(base),
        "ci95_lo": float(lo), "ci95_hi": float(hi),
        "paper_anchor": float(anchor), "paper_in_ci95": bool(in_ci),
    })

# Save
with open(LOCAL / "task_2_1_betas" / "bootstrap_ci_results.json", "w") as f:
    json.dump({"n_boot": N_BOOT, "results": results_summary}, f, indent=2)
print(f"\nsaved bootstrap_ci_results.json")
