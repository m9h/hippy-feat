#!/usr/bin/env python3
"""Score Fast/Slow/EoR/Offline RT tiers on fold-0 ckpt with BOTH first-rep and
avg-of-3-reps modes — to disambiguate which paper Table 1 row each matches.

Paper Table 1 retrieval anchors (3T sub-005 ses-03):
  Offline 3T (avg. 3 reps.):  Image 90%, Brain 88%
  Offline 3T:                 Image 76%, Brain 64%
  End-of-run real-time:       Image 66%, Brain 62%
  Slow real-time (36s):       Image 58%, Brain 58%
  Fast real-time (14.5s):     Image 36%, Brain 40%

Eval modes:
  first_rep: keep only first occurrence of each special515 (50 βs, no avg)
  avg3:      mean across all 3 occurrences per image (50 averaged βs)

Output: comparison table for all (tier, mode) pairs.
"""
from __future__ import annotations
import json, sys, types, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np, torch, torch.nn as nn

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
RT_MINDEYE = Path("/Users/mhough/Workspace/rt_mindEye2/src")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
LOCAL_DRIVERS = Path("/Users/mhough/Workspace/local_drivers")

import diffusers, diffusers.models  # noqa
vae_mod = types.ModuleType("diffusers.models.vae")
class _Stub(nn.Module):
    def __init__(self, *a, **k): super().__init__()
    def forward(self, *a, **k): return None
vae_mod.Decoder = _Stub
sys.modules["diffusers.models.vae"] = vae_mod
diffusers.models.vae = vae_mod
gm = types.ModuleType("generative_models")
sgm = types.ModuleType("generative_models.sgm")
sgm_util = types.ModuleType("generative_models.sgm.util")
sgm_modules = types.ModuleType("generative_models.sgm.modules")
sgm_enc = types.ModuleType("generative_models.sgm.modules.encoders")
sgm_enc_mods = types.ModuleType("generative_models.sgm.modules.encoders.modules")
sgm_util.append_dims = lambda x, n: x
sgm_enc_mods.FrozenOpenCLIPImageEmbedder = _Stub
sgm_enc_mods.FrozenOpenCLIPEmbedder2 = _Stub
for mod in [gm, sgm, sgm_util, sgm_modules, sgm_enc, sgm_enc_mods]:
    sys.modules[mod.__name__] = mod
sys.modules["sgm"] = sgm
sys.path.insert(0, str(RT_MINDEYE))
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(LOCAL_DRIVERS))
import mindeye_retrieval_eval as M
M.RTCLOUD_MINDEYE = RT_MINDEYE
from run_retrieval_local import compute_gt_mps
warnings.filterwarnings("ignore")

PREREG = LOCAL / "task_2_1_betas" / "prereg"
CKPT = LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
CACHE = LOCAL / "task_2_1_betas/gt_cache"
device = "mps" if torch.backends.mps.is_available() else "cpu"

TIERS = [
    ("Fast",  "RT_paper_Fast_pst5",        {"image": 0.36, "brain": 0.40}),
    ("Slow",  "RT_paper_Slow_pst20",       {"image": 0.58, "brain": 0.58}),
    ("EoR",   "RT_paper_replica_partial",  {"image": 0.66, "brain": 0.62}),
]


def filter_first_rep(betas, ids):
    keep, seen = [], set()
    for i, t in enumerate(ids):
        ts = str(t)
        if not ("special515" in ts):
            continue
        if ts in seen:
            continue
        seen.add(ts)
        keep.append(i)
    return betas[keep], [str(ids[i]) for i in keep]


def avg_3_reps(betas, ids):
    """Average all repeats of each special515 image."""
    by = defaultdict(list)
    for i, t in enumerate(ids):
        ts = str(t)
        if "special515" not in ts:
            continue
        by[ts].append(i)
    out_b, out_ids = [], []
    for k, idxs in by.items():
        if len(idxs) < 1: continue
        out_b.append(betas[idxs].mean(axis=0))
        out_ids.append(k)
    return np.stack(out_b), out_ids


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1)
    gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


print(f"=== loading fold-0 ckpt: {CKPT.name} ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval().requires_grad_(False)


def fwd(test_betas):
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = torch.from_numpy(test_betas.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(0, b.shape[0], 8):
            vr = model.ridge(b[i:i+8], 0)
            o = model.backbone(vr)
            cv = o[1] if isinstance(o, tuple) else o
            out.append(cv.float().cpu().numpy())
    return np.concatenate(out, 0).reshape(-1, ss, se)


results = []
for label, prefix, anchor in TIERS:
    betas = np.load(PREREG / f"{prefix}_ses-03_betas.npy")
    ids = np.load(PREREG / f"{prefix}_ses-03_trial_ids.npy")
    print(f"\n--- {label} ({prefix}): raw βs {betas.shape} ---", flush=True)

    for mode_name, fn in [("first_rep", filter_first_rep), ("avg_3reps", avg_3_reps)]:
        tb, tids = fn(betas, ids)
        if len(tids) != 50:
            print(f"  {mode_name}: WARN got {len(tids)} test items, expected 50")
            continue
        img_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in tids]
        gt = compute_gt_mps(img_paths, device=device, cache_dir=CACHE)
        pred = fwd(tb)
        t1_fwd = topk(pred, gt, k=1)
        t1_bwd = topk(gt, pred, k=1)
        results.append({
            "tier": label, "mode": mode_name,
            "image_top1": t1_fwd, "brain_top1": t1_bwd,
            "anchor_image": anchor["image"], "anchor_brain": anchor["brain"],
        })
        print(f"  {mode_name:10s}: Image={t1_fwd*100:5.2f}%  Brain={t1_bwd*100:5.2f}%  "
              f"(paper {label}: I={anchor['image']*100:.0f}/B={anchor['brain']*100:.0f})", flush=True)

print("\n========== SUMMARY (fold-0) ==========")
print(f"{'Tier':6s} {'Mode':10s} {'Image':>8s} {'Brain':>8s}  {'paper I/B':>12s}")
for r in results:
    print(f"{r['tier']:6s} {r['mode']:10s} {r['image_top1']*100:7.2f}% {r['brain_top1']*100:7.2f}%  "
          f"{int(r['anchor_image']*100):>3d}/{int(r['anchor_brain']*100):>3d}")

(LOCAL / "task_2_1_betas/rt_tiers_both_modes_fold0.json").write_text(json.dumps(results, indent=2))
print(f"\n  saved rt_tiers_both_modes_fold0.json")
