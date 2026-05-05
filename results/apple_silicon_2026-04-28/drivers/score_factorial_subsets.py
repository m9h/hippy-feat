#!/usr/bin/env python3
"""Re-score every prereg cell with subset0 / subset1 / subset2 semantics
on fold-0, the right framework for comparing RT and Offline tiers.

  subset0 = single-rep prediction (β of 1st occurrence)
  subset1 = avg-of-2 prediction   (mean of 1st+2nd β)
  subset2 = avg-of-3 prediction   (mean of all 3 βs)

Paper's RT-tier rows (Slow, EoR) = subset1; Offline rows = subset2 via
`filter_and_average_repeats`. Image-column anchors:
  Fast    subset0 = 36%
  Slow    subset1 = 58%
  EoR     subset1 = 66%
  Offline subset2 = 76%

The Offline ↔ EoR gap is therefore 76 − 66 = 10pp. This script computes the
subset1 score for every cell so we can see which preprocessing changes get
EoR subset1 closest to the Offline subset2 = 76% target.
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


def get_subsets(betas, ids):
    by = defaultdict(list)
    for i, t in enumerate(ids):
        ts = str(t)
        if "special515" not in ts:
            continue
        by[ts].append(i)
    out = {0: [], 1: [], 2: []}
    nm  = {0: [], 1: [], 2: []}
    for img, idxs in by.items():
        idxs = idxs[:3]
        if len(idxs) >= 1:
            out[0].append(betas[idxs[0]]); nm[0].append(img)
        if len(idxs) >= 2:
            out[1].append(betas[[idxs[0], idxs[1]]].mean(axis=0)); nm[1].append(img)
        if len(idxs) >= 3:
            out[2].append(betas[idxs].mean(axis=0)); nm[2].append(img)
    res = {}
    for n, s in out.items():
        if len(s) == 50:
            res[n] = (np.stack(s), nm[n])
    return res


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


print(f"=== loading fold-0 ckpt ===", flush=True)
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


_gt = {}
def get_gt(ids):
    key = tuple(sorted(ids))
    if key not in _gt:
        paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in ids]
        _gt[key] = compute_gt_mps(paths, device=device, cache_dir=CACHE)
    return _gt[key]


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
        continue
    if betas.ndim != 2 or betas.shape[1] != 2792:
        continue
    if betas.shape[0] < 100:    # cells already pre-filtered to 50 trials etc.
        continue

    row = {"cell": cell, "n_trials": int(betas.shape[0])}
    subs = get_subsets(betas, ids)
    for sub_n, (b, n) in subs.items():
        try:
            gt = get_gt(n)
            pred = fwd(b)
            row[f"subset{sub_n}_image"] = topk(pred, gt, k=1)
            row[f"subset{sub_n}_brain"] = topk(gt, pred, k=1)
        except Exception as e:
            row[f"subset{sub_n}_error"] = str(e)[:80]

    results.append(row)
    s0 = row.get("subset0_image"); s1 = row.get("subset1_image"); s2 = row.get("subset2_image")
    f0 = f"{s0*100:5.1f}" if s0 is not None else "  ---"
    f1 = f"{s1*100:5.1f}" if s1 is not None else "  ---"
    f2 = f"{s2*100:5.1f}" if s2 is not None else "  ---"
    print(f"  {cell:55s}  n={row['n_trials']:4d}  s0={f0}  s1={f1}  s2={f2}", flush=True)

out_path = LOCAL / "task_2_1_betas/factorial_subsets_fold0.json"
out_path.write_text(json.dumps(results, indent=2))
print(f"\nsaved {out_path}\n  {len(results)} cells", flush=True)
