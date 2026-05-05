#!/usr/bin/env python3
"""Re-score every prereg cell in both first-rep and avg-of-3 modes against
fold-0, producing a single unified table with proper paper anchors.

This corrects the doc-layer mislabel where avg-of-3 results were reported
as if they were first-rep. Each row of the output now has BOTH numbers,
clearly labeled, and can be compared to the right paper anchor for its tier.

Paper anchors (Image retrieval):
  Offline 3T avg-of-3:  76% (filter_and_average_repeats path)
  Offline 3T first-rep: ~60% (no separate paper number; paper Brain=64% matches)
  EoR running-avg:      66% (paper)
  Slow first-rep:       58% (paper, with non-causal cum-z)
  Fast first-rep:       36% (paper)
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

PREREG = LOCAL / "task_2_1_betas/prereg"
CKPT = LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
CACHE = LOCAL / "task_2_1_betas/gt_cache"
device = "mps" if torch.backends.mps.is_available() else "cpu"


def filter_first_rep(betas, ids):
    keep, seen = [], set()
    for i, t in enumerate(ids):
        ts = str(t)
        if "special515" not in ts:
            continue
        if ts in seen:
            continue
        seen.add(ts)
        keep.append(i)
    return betas[keep], [str(ids[i]) for i in keep]


def avg_3_reps(betas, ids):
    by = defaultdict(list)
    for i, t in enumerate(ids):
        ts = str(t)
        if "special515" not in ts:
            continue
        by[ts].append(i)
    out_b, out_ids = [], []
    for k, idxs in by.items():
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


# Cache GT once per unique image set
_gt_cache = {}
def get_gt(ids):
    key = tuple(sorted(ids))
    if key not in _gt_cache:
        paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in ids]
        _gt_cache[key] = compute_gt_mps(paths, device=device, cache_dir=CACHE)
    return _gt_cache[key]


# Find all prereg cells
beta_files = sorted(PREREG.glob("*_betas.npy"))
print(f"\nfound {len(beta_files)} prereg cells\n", flush=True)

results = []
for bf in beta_files:
    cell = bf.name.replace("_ses-03_betas.npy", "").replace("_betas.npy", "")
    ids_path = bf.parent / bf.name.replace("_betas.npy", "_trial_ids.npy")
    if not ids_path.exists():
        continue
    try:
        betas = np.load(bf)
        ids = np.load(ids_path)
    except Exception as e:
        print(f"  SKIP {cell}: load error {e}", flush=True)
        continue

    if betas.ndim != 2 or betas.shape[1] != 2792:
        # not in 2792-vox space; skip
        continue

    row = {"cell": cell, "n_trials": int(betas.shape[0])}
    for mode_name, fn in [("first_rep", filter_first_rep), ("avg_3reps", avg_3_reps)]:
        try:
            tb, tids = fn(betas, ids)
        except Exception as e:
            row[f"{mode_name}_error"] = str(e)[:80]
            continue
        n = len(tids)
        if n == 0:
            row[f"{mode_name}_n"] = 0
            continue
        if n != 50:
            row[f"{mode_name}_n"] = n
            row[f"{mode_name}_skip"] = "n!=50"
            continue
        try:
            gt = get_gt(tids)
            pred = fwd(tb)
            row[f"{mode_name}_image"] = topk(pred, gt, k=1)
            row[f"{mode_name}_brain"] = topk(gt, pred, k=1)
            row[f"{mode_name}_image_top5"] = topk(pred, gt, k=5)
        except Exception as e:
            row[f"{mode_name}_error"] = str(e)[:80]

    results.append(row)
    img1 = row.get("first_rep_image")
    img3 = row.get("avg_3reps_image")
    s1 = f"{img1*100:5.1f}" if img1 is not None else "  ---"
    s3 = f"{img3*100:5.1f}" if img3 is not None else "  ---"
    print(f"  {cell:60s}  n={row['n_trials']:4d}  fwd1r={s1}  fwd3a={s3}", flush=True)

out_path = LOCAL / "task_2_1_betas/factorial_both_modes_fold0.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nsaved {out_path}\n  {len(results)} cells", flush=True)
