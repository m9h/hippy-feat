#!/usr/bin/env python3
"""3-way comparison: per-trial AR(1) LSS vs streaming OLS GLM vs streaming AR(1) GLM.

Resolves whether the +12pp Slow gain in streaming RLS GLM came from:
(a) the joint growing-design alone, or
(b) missing AR(1) prewhitening (which the LSS baseline has)
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

CKPT = LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
device = "mps" if torch.backends.mps.is_available() else "cpu"
PREREG = LOCAL / "task_2_1_betas/prereg"

def get_subsets(b, ids):
    by = defaultdict(list)
    for i, t in enumerate(ids):
        if "special515" in str(t):
            by[str(t)].append(i)
    out = {0: [], 1: [], 2: []}; nm = {0: [], 1: [], 2: []}
    for img, idxs in by.items():
        idxs = idxs[:3]
        if len(idxs) >= 1: out[0].append(b[idxs[0]]); nm[0].append(img)
        if len(idxs) >= 2: out[1].append(b[[idxs[0], idxs[1]]].mean(axis=0)); nm[1].append(img)
        if len(idxs) >= 3: out[2].append(b[idxs].mean(axis=0)); nm[2].append(img)
    return {n: (np.stack(s), nm[n]) for n, s in out.items() if len(s) == 50}


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


print("=== loading fold-0 ckpt ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval().requires_grad_(False)


def fwd(b):
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        bb = torch.from_numpy(b.astype(np.float32)).to(device).unsqueeze(1)
        for i in range(0, bb.shape[0], 8):
            vr = model.ridge(bb[i:i+8], 0)
            o = model.backbone(vr)
            cv = o[1] if isinstance(o, tuple) else o
            out.append(cv.float().cpu().numpy())
    return np.concatenate(out, 0).reshape(-1, ss, se)


def score_cell(prefix, ses="ses-03"):
    bf = PREREG / f"{prefix}_{ses}_betas.npy"
    if not bf.exists():
        return None
    betas = np.load(bf)
    ids = np.load(PREREG / f"{prefix}_{ses}_trial_ids.npy")
    subs = get_subsets(betas, ids)
    res = {}
    for n in (0, 1, 2):
        if n not in subs: continue
        b, names = subs[n]
        paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(nm).name for nm in names]
        gt = compute_gt_mps(paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
        pred = fwd(b)
        res[f"subset{n}"] = {"image": topk(pred, gt, 1), "brain": topk(gt, pred, 1)}
    return res


# 3-way comparison configs
TIERS = [
    ("Fast", "RT_paper_Fast_pst5_inclz",
             "RT_paper_RLS_Fast_pst5_K7CSFWM_HP_e1_inclz",
             "RT_paper_RLS_AR1_pst5_K7CSFWM_HP_e1_inclz",
             {"image": 0.36, "brain": 0.40}),
    ("Slow", "RT_paper_Slow_pst20_inclz",
             "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz",
             "RT_paper_RLS_AR1_pst20_K7CSFWM_HP_e1_inclz",
             {"image": 0.58, "brain": 0.58}),
    ("EoR", "RT_paper_EoR_K7_CSFWM_HP_e1_inclz",
            "RT_paper_RLS_EoR_K7CSFWM_HP_e1_inclz",
            "RT_paper_RLS_AR1_EoR_K7CSFWM_HP_e1_inclz",
            {"image": 0.66, "brain": 0.62}),
]

results = {}
for label, lss, sols, sar1, anchor in TIERS:
    print(f"\n========== {label} ==========", flush=True)
    r_lss = score_cell(lss); print(f"  LSS AR(1):       {r_lss}", flush=True)
    r_sols = score_cell(sols); print(f"  streaming OLS:   {r_sols}", flush=True)
    r_sar1 = score_cell(sar1); print(f"  streaming AR(1): {r_sar1}", flush=True)
    results[label] = {
        "lss_ar1": r_lss, "streaming_ols": r_sols, "streaming_ar1": r_sar1,
        "anchor": anchor,
    }

(LOCAL / "task_2_1_betas/streaming_ar1_vs_ols_fold0.json").write_text(json.dumps(results, indent=2))

# Print summary table
print("\n\n========== 3-WAY SUMMARY (Image %, fold-0, n=50) ==========")
print(f"{'Tier':6s} {'Subset':12s} {'LSS AR(1)':>10s} {'Stream OLS':>10s} {'Stream AR(1)':>13s} {'Paper':>6s}")
for label, _, _, _, anchor in TIERS:
    r = results[label]
    pa = int(anchor['image']*100)
    for sub in (0, 1, 2):
        key = f"subset{sub}"
        l_v = r["lss_ar1"][key]["image"]*100 if r["lss_ar1"] and key in r["lss_ar1"] else None
        o_v = r["streaming_ols"][key]["image"]*100 if r["streaming_ols"] and key in r["streaming_ols"] else None
        a_v = r["streaming_ar1"][key]["image"]*100 if r["streaming_ar1"] and key in r["streaming_ar1"] else None
        ls = f"{l_v:8.1f}%" if l_v is not None else "    ---"
        os_ = f"{o_v:8.1f}%" if o_v is not None else "    ---"
        as_ = f"{a_v:11.1f}%" if a_v is not None else "       ---"
        print(f"{label:6s} subset{sub}      {ls}   {os_}   {as_}      {pa:3d}%")
