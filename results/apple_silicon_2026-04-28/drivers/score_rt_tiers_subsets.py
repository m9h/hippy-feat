#!/usr/bin/env python3
"""Score RT tiers with the three subsets that mindeye.py:947-955 distinguishes:

  subset0 = first-rep prediction   (single-rep β)
  subset1 = second-rep prediction  (avg-of-1+2 by running average)
  subset2 = third-rep prediction   (avg-of-1+2+3 by running average)

Per `mindeye.py:773-782`, when `is_repeat==True`, the stored prediction at
trial k_i is `mean(z-scored βs at indices [k_1, k_2, ..., k_i])`. The eval
extracts `duplicated[:, 0]` (first-rep predictions) and `duplicated[:, 1]`
(second-rep predictions).

Paper's RT tiers report 50-way retrieval. The likely correspondence:
  Fast/Slow report subset0 (single-rep at fast/slow pst)
  EoR reports subset1 (avg-of-2, since running average happens by 2nd rep)

If correct: paper EoR 66% ≈ our avg-of-2 calculation, paper Slow 58% = our
single-rep, paper Fast 36% = our single-rep at shorter pst.
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

TIERS = [
    ("Fast",  "RT_paper_Fast_pst5",        {"image": 0.36, "brain": 0.40}),
    ("Slow",  "RT_paper_Slow_pst20",       {"image": 0.58, "brain": 0.58}),
    ("Slow25","RT_paper_Slow_pst25_inclz", {"image": 0.58, "brain": 0.58}),
    ("EoR",   "RT_paper_replica_partial",  {"image": 0.66, "brain": 0.62}),
    ("EoR_HP","RT_paper_EoR_K7_CSFWM_HP_e1_inclz", {"image": 0.66, "brain": 0.62}),
]


def get_subsets(betas, ids):
    """Return three subsets per `mindeye.py:947-955` semantics:
       subset_n = the prediction at the n-th repeat = mean(βs of reps 0..n).
    """
    by = defaultdict(list)
    for i, t in enumerate(ids):
        ts = str(t)
        if "special515" not in ts:
            continue
        by[ts].append(i)

    subsets = {0: [], 1: [], 2: []}   # by repeat number
    names = {0: [], 1: [], 2: []}
    for img, idxs in by.items():
        idxs = idxs[:3]   # safety: only consider first 3 reps
        if len(idxs) < 1:
            continue
        # subset0 = single-rep (β of rep 0)
        subsets[0].append(betas[idxs[0]])
        names[0].append(img)
        # subset1 = avg-of-1+2 (mean of rep 0 and rep 1)
        if len(idxs) >= 2:
            subsets[1].append(betas[[idxs[0], idxs[1]]].mean(axis=0))
            names[1].append(img)
        # subset2 = avg-of-1+2+3 (mean of all 3 reps)
        if len(idxs) >= 3:
            subsets[2].append(betas[idxs].mean(axis=0))
            names[2].append(img)
    return {n: (np.stack(s), names[n]) for n, s in subsets.items() if len(s) == 50}


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


results = []
for label, prefix, anchor in TIERS:
    bf = PREREG / f"{prefix}_ses-03_betas.npy"
    idf = PREREG / f"{prefix}_ses-03_trial_ids.npy"
    if not bf.exists():
        print(f"\n--- {label}: SKIP (no βs at {bf.name})")
        continue
    betas = np.load(bf)
    ids = np.load(idf)
    print(f"\n--- {label} ({prefix}): raw βs {betas.shape} ---", flush=True)
    subs = get_subsets(betas, ids)
    for sub_n, (b, n) in subs.items():
        gt = get_gt(n)
        pred = fwd(b)
        i_acc = topk(pred, gt, k=1)
        b_acc = topk(gt, pred, k=1)
        sub_label = {0: "subset0 (single-rep)",
                     1: "subset1 (avg-of-2)",
                     2: "subset2 (avg-of-3)"}[sub_n]
        results.append({
            "tier": label, "prefix": prefix, "subset": sub_n,
            "image": i_acc, "brain": b_acc,
            "anchor_image": anchor["image"], "anchor_brain": anchor["brain"],
        })
        marker = ""
        if abs(i_acc - anchor["image"]) <= 0.04:
            marker = "  ← Image match!"
        elif abs(b_acc - anchor["brain"]) <= 0.04:
            marker = "  ← Brain match!"
        print(f"  {sub_label:25s}: Image={i_acc*100:5.2f}%  Brain={b_acc*100:5.2f}%  "
              f"(paper {label}: I={anchor['image']*100:.0f}/B={anchor['brain']*100:.0f}){marker}",
              flush=True)

(LOCAL / "task_2_1_betas/rt_tiers_subsets_fold0.json").write_text(json.dumps(results, indent=2))
print(f"\nsaved rt_tiers_subsets_fold0.json")
