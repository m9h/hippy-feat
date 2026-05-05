#!/usr/bin/env python3
"""Score paper RT-tier replicas with FIRST-presentation-only filter (paper §2.7).

For each cell: load (770, 2792) betas + 770 trial_ids, filter to special515
trials, keep only FIRST occurrence of each unique image (50 trials of 50 unique
images), score against MindEye2 ckpt → top-1 / top-5 image retrieval.

Cells expected to match paper anchors:
  RT_paper_Fast_pst5      → 36% top-1 (paper Fast)
  RT_paper_Slow_pst20     → 58% top-1 (paper Slow)
  RT_paper_replica_partial→ 66% top-1 (paper End-of-run)

Appends to retrieval_results_v2.json under conditions
{cell}_singleRep so they don't collide with prior all-trials scoring rows.
"""
from __future__ import annotations

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

# stub diffusers + sgm before importing mindeye_retrieval_eval
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
    ("RT_paper_Fast_pst5",       "ses-03", 0.36),    # paper Fast anchor
    ("RT_paper_Slow_pst20",      "ses-03", 0.58),    # paper Slow anchor
    ("RT_paper_replica_partial", "ses-03", 0.66),    # paper End-of-run anchor
]


def filter_first_rep(betas: np.ndarray, trial_ids: np.ndarray
                      ) -> tuple[np.ndarray, list[str]]:
    """Filter to special515 trials, keep ONLY the first occurrence per unique
    image — matches paper §2.7 'first presentation of three repeats'."""
    keep_idx, seen = [], set()
    for i, t in enumerate(trial_ids):
        if not str(t).startswith("all_stimuli/special515/"):
            continue
        if t in seen:
            continue
        seen.add(t)
        keep_idx.append(i)
    return betas[keep_idx], [str(trial_ids[i]) for i in keep_idx]


def causal_cumulative_zscore(betas: np.ndarray) -> np.ndarray:
    """Causal cum-z over the input ordering — paper formula (mindeye.py:770-784).
    Trial i uses past-only stats from trials 0..i-1."""
    n, V = betas.shape
    z = np.zeros_like(betas)
    for i in range(n):
        if i == 0:
            z[i] = betas[i]
        elif i == 1:
            z[i] = betas[i] - betas[0]
        else:
            mu = betas[:i].mean(axis=0)
            sd = betas[:i].std(axis=0) + 1e-6
            z[i] = (betas[i] - mu) / sd
    return z


def score_cell(name: str, ses: str, paper_anchor: float):
    print(f"\n=== scoring {name} ===")
    raw_betas = np.load(PREREG / f"{name}_{ses}_betas.npy")
    raw_ids = np.load(PREREG / f"{name}_{ses}_trial_ids.npy")
    print(f"  raw betas: {raw_betas.shape}, ids: {raw_ids.shape}")

    # All RT_paper_* cells from rt_paper_full_replica.run_cell already have
    # cum-z applied internally — do NOT double z-score.
    sess_betas = raw_betas

    # Filter to first-rep-only of special515
    test_betas, test_ids = filter_first_rep(sess_betas, raw_ids)
    print(f"  filtered first-rep: {test_betas.shape}, "
          f"{len(set(test_ids))} unique images")

    # Compute ground-truth CLIP embeddings
    image_paths = [LOCAL / "rt3t" / "data" / "all_stimuli" / "special515" / Path(n).name
                   for n in test_ids]
    gt = compute_gt_mps(image_paths, device=device, cache_dir=CACHE)
    print(f"  gt: {gt.shape}")

    # Forward through MindEye2
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
    print(f"  TOP-1: {top1*100:.2f}%  TOP-5: {top5*100:.2f}%  "
          f"(paper anchor: {paper_anchor*100:.0f}%)")

    return {
        "condition": f"{name}_singleRep",
        "session": ses,
        "n_test_trials": int(test_betas.shape[0]),
        "n_unique_images": int(len(set(test_ids))),
        "top1_image_retrieval": float(top1),
        "top5_image_retrieval": float(top5),
        "paper_anchor_top1": float(paper_anchor),
        "delta_vs_paper_pp": float(top1 - paper_anchor) * 100,
        "notes": "paper §2.7 first-rep filter; cum-z already in saved betas",
    }


def main():
    results = json.loads(RESULTS_PATH.read_text())
    for name, ses, anchor in CELLS:
        if not (PREREG / f"{name}_{ses}_betas.npy").exists():
            print(f"\n  SKIP {name} — betas not yet on disk")
            continue
        res = score_cell(name, ses, anchor)
        # remove any prior row with same condition string
        results = [r for r in results if r.get("condition") != res["condition"]]
        results.append(res)
        RESULTS_PATH.write_text(json.dumps(results, indent=2))
        print(f"  appended to {RESULTS_PATH}")

    # summary table
    print("\n=== ladder reproduction ===")
    print(f"{'cell':40s}  {'top1':>7s}  {'paper':>7s}  {'delta':>7s}")
    for r in results:
        cond = r.get("condition", "")
        if cond.startswith("RT_paper_") and cond.endswith("_singleRep"):
            print(f"{cond:40s}  "
                  f"{r['top1_image_retrieval']*100:6.2f}%  "
                  f"{r['paper_anchor_top1']*100:6.0f}%  "
                  f"{r['delta_vs_paper_pp']:+6.1f}pp")


if __name__ == "__main__":
    main()
