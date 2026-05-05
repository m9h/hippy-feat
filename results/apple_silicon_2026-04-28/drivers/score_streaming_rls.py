#!/usr/bin/env python3
"""Score streaming RLS GLM cells (Fast/Slow/EoR) at subset0/1/2 vs the
deployment-champion baseline (per-trial LSS K7+CSFWM+HP+e1)."""
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
    out = {0: [], 1: [], 2: []}; nm  = {0: [], 1: [], 2: []}
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
    sim = pn @ gn.T; labels = np.arange(p.shape[0])
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


CELLS = [
    ("Fast", "RT_paper_RLS_Fast_pst5_K7CSFWM_HP_e1_inclz", {"image": 0.36, "brain": 0.40}),
    ("Slow", "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz", {"image": 0.58, "brain": 0.58}),
    ("EoR",  "RT_paper_RLS_EoR_K7CSFWM_HP_e1_inclz",        {"image": 0.66, "brain": 0.62}),
]
BASELINES = {
    "Fast": ("RT_paper_Fast_pst5_inclz", "deployment champion (per-trial LSS)"),
    "Slow": ("RT_paper_Slow_pst20_inclz", "deployment champion (per-trial LSS)"),
    "EoR":  ("RT_paper_EoR_K7_CSFWM_HP_e1_inclz", "deployment champion (per-trial LSS K7+CSFWM+HP+e1)"),
}

results = {}
for label, prefix, anchor in CELLS:
    bf = PREREG / f"{prefix}_ses-03_betas.npy"
    if not bf.exists():
        print(f"\n--- {label}: SKIP (missing {bf.name})", flush=True)
        continue
    betas = np.load(bf)
    ids = np.load(PREREG / f"{prefix}_ses-03_trial_ids.npy")
    print(f"\n--- {label} ({prefix}): βs {betas.shape} ---", flush=True)
    subs = get_subsets(betas, ids)
    rls = {}
    for n in (0, 1, 2):
        if n not in subs: continue
        b, names = subs[n]
        paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(nm).name for nm in names]
        gt = compute_gt_mps(paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
        pred = fwd(b)
        i_acc = topk(pred, gt, k=1); b_acc = topk(gt, pred, k=1)
        rls[f"subset{n}"] = {"image": i_acc, "brain": b_acc}
        slabel = ["single-rep", "avg-of-2", "complete-set"][n]
        print(f"  RLS streaming {slabel}: Image={i_acc*100:5.1f}%  Brain={b_acc*100:5.1f}%", flush=True)

    # baseline for comparison
    base_prefix, base_desc = BASELINES[label]
    base_betas = np.load(PREREG / f"{base_prefix}_ses-03_betas.npy")
    base_ids = np.load(PREREG / f"{base_prefix}_ses-03_trial_ids.npy")
    base_subs = get_subsets(base_betas, base_ids)
    base_results = {}
    for n in (0, 1, 2):
        if n not in base_subs: continue
        b, names = base_subs[n]
        paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(nm).name for nm in names]
        gt = compute_gt_mps(paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
        pred = fwd(b)
        i_acc = topk(pred, gt, k=1); b_acc = topk(gt, pred, k=1)
        base_results[f"subset{n}"] = {"image": i_acc, "brain": b_acc}
        slabel = ["single-rep", "avg-of-2", "complete-set"][n]
        print(f"  baseline LSS {slabel}: Image={i_acc*100:5.1f}%  Brain={b_acc*100:5.1f}%", flush=True)

    delta_s0 = (rls["subset0"]["image"] - base_results["subset0"]["image"]) * 100 if "subset0" in rls and "subset0" in base_results else None
    delta_s1 = (rls["subset1"]["image"] - base_results["subset1"]["image"]) * 100 if "subset1" in rls and "subset1" in base_results else None
    delta_s2 = (rls["subset2"]["image"] - base_results["subset2"]["image"]) * 100 if "subset2" in rls and "subset2" in base_results else None

    results[label] = {
        "rls": rls, "baseline_lss": base_results,
        "anchor": anchor,
        "delta_s0_image": delta_s0, "delta_s1_image": delta_s1, "delta_s2_image": delta_s2,
    }
    print(f"\n  Δ vs LSS baseline (Image): subset0={delta_s0:+.1f}pp  subset1={delta_s1:+.1f}pp  subset2={delta_s2:+.1f}pp")
    print(f"  paper anchor: I={anchor['image']*100:.0f}/B={anchor['brain']*100:.0f}")

(LOCAL / "task_2_1_betas/streaming_rls_subsets_fold0.json").write_text(json.dumps(results, indent=2))
print(f"\nsaved streaming_rls_subsets_fold0.json", flush=True)

print("\n========== SUMMARY: STREAMING RLS GLM ==========")
print(f"{'Tier':6s} {'Subset':12s} {'RLS':>8s} {'LSS base':>9s} {'Δ':>6s} {'Paper':>6s}")
for tier in ("Fast", "Slow", "EoR"):
    if tier not in results: continue
    r = results[tier]
    paper_anchor_image = r["anchor"]["image"] * 100
    for sub in (0, 1, 2):
        key = f"subset{sub}"
        if key not in r["rls"] or key not in r["baseline_lss"]: continue
        rls_v = r["rls"][key]["image"] * 100
        bas_v = r["baseline_lss"][key]["image"] * 100
        delta = rls_v - bas_v
        marker = " *" if delta >= 4.0 else ""
        print(f"{tier:6s} subset{sub}     {rls_v:7.1f}% {bas_v:8.1f}% {delta:+6.1f} {paper_anchor_image:5.0f}%{marker}")
