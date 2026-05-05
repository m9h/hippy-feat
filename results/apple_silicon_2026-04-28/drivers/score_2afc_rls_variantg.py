#!/usr/bin/env python3
"""Score 2-AFC pairwise AUC on streaming RLS GLM + Variant G prereg cells.

For each cell, single-rep filter (50 first-occurrence special515), forward
through fold-0, compute pairwise discrimination accuracy:
  for each (i, j), j != i: count if sim(pred_i, gt_i) > sim(pred_i, gt_j)
  2-AFC AUC = correct / total

This is the deployment-relevant binary-discrimination metric for closed-loop
neurofeedback (chance = 50%). Compare to:
  Deployment champion (per-trial AR(1) LSS K7+CSFWM+HP+e1): 97.2% 2-AFC
"""
from __future__ import annotations
import json, sys, types, warnings
from collections import Counter, defaultdict
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


def first_rep_filter(b, ids):
    seen, keep = set(), []
    for i, t in enumerate(ids):
        ts = str(t)
        if "special515" not in ts: continue
        if ts in seen: continue
        seen.add(ts); keep.append(i)
    return b[keep], [str(ids[i]) for i in keep]


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


def two_afc(p, g):
    """Pairwise: sim(pred_i, gt_i) > sim(pred_i, gt_j) for j != i."""
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T          # (50, 50)
    n = sim.shape[0]
    diag = np.diag(sim).reshape(-1, 1)              # (n, 1) sim(pred_i, gt_i)
    # For each i, count j != i where diag[i] > sim[i, j]
    correct = (diag > sim).astype(np.float32)
    np.fill_diagonal(correct, 0)                     # exclude i==j
    return float(correct.sum() / (n * (n - 1)))


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


def score_cell(prefix, label):
    bf = PREREG / f"{prefix}_ses-03_betas.npy"
    if not bf.exists():
        return None
    b = np.load(bf)
    ids = np.load(PREREG / f"{prefix}_ses-03_trial_ids.npy", allow_pickle=True)
    ids = np.asarray([str(t) for t in ids])
    sb, names = first_rep_filter(b, ids)
    if len(names) != 50:
        return None
    paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in names]
    gt = compute_gt_mps(paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
    pred = fwd(sb)
    return {
        "cell": prefix,
        "image_top1": topk(pred, gt, 1),
        "brain_top1": topk(gt, pred, 1),
        "image_top5": topk(pred, gt, 5),
        "two_afc_image": two_afc(pred, gt),
        "two_afc_brain": two_afc(gt, pred),
    }


CELLS = {
    # Streaming RLS GLM (Mac fold-0, OLS variant)
    "RT_paper_RLS_Fast_pst5_K7CSFWM_HP_e1_inclz": "Streaming RLS Fast (OLS)",
    "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz": "Streaming RLS Slow (OLS)",
    "RT_paper_RLS_EoR_K7CSFWM_HP_e1_inclz": "Streaming RLS EoR (OLS)",
    # Streaming RLS GLM with AR(1) prewhitening
    "RT_paper_RLS_AR1_pst5_K7CSFWM_HP_e1_inclz": "Streaming RLS Fast AR(1)",
    "RT_paper_RLS_AR1_pst20_K7CSFWM_HP_e1_inclz": "Streaming RLS Slow AR(1)",
    "RT_paper_RLS_AR1_EoR_K7CSFWM_HP_e1_inclz": "Streaming RLS EoR AR(1)",
    # Variant G family
    "VariantG_glover_rtm": "Variant G base",
    "VariantG_glover_rtm_acompcor": "Variant G + aCompCor",
    "VariantG_glover_rtm_acompcor_with_vars": "Variant G + aCompCor + with_vars",
    "VariantG_glover_rtm_glmdenoise_fracridge": "Variant G + GLMdenoise + fracridge",
    "VariantG_glover_rtm_glmdenoise_fracridge_with_vars": "Variant G + GLMdenoise + fracridge + vars",
    "VariantG_glover_rtm_prior": "Variant G + prior",
    "VariantG_glover_rtm_streaming_pst8": "Variant G streaming pst=8",
    "VariantG_glover_rtm_with_vars": "Variant G + with_vars",
    "SameImagePrior_VariantG_glover_rtm": "SameImagePrior Variant G",
    # Reference: deployment champion (per-trial AR(1) LSS)
    "RT_paper_EoR_K7_CSFWM_HP_e1_inclz": "Deployment champion (per-trial AR(1) LSS)",
    "RT_paper_Fast_pst5_inclz": "Per-trial AR(1) LSS Fast",
    "RT_paper_Slow_pst20_inclz": "Per-trial AR(1) LSS Slow",
}

results = {}
for prefix, label in CELLS.items():
    print(f"\n{label} ({prefix}):", flush=True)
    r = score_cell(prefix, label)
    if r is None:
        print(f"  SKIP (missing or wrong shape)"); continue
    print(f"  Image top-1: {r['image_top1']*100:5.1f}%  Brain top-1: {r['brain_top1']*100:5.1f}%  Top-5: {r['image_top5']*100:5.1f}%", flush=True)
    print(f"  2-AFC Image: {r['two_afc_image']*100:5.2f}%  2-AFC Brain: {r['two_afc_brain']*100:5.2f}%", flush=True)
    results[prefix] = {"label": label, **r}

(LOCAL / "task_2_1_betas/two_afc_rls_variantg.json").write_text(json.dumps(results, indent=2))

print("\n\n========== 2-AFC SUMMARY (sorted by 2-AFC Image) ==========")
ranked = sorted(results.items(), key=lambda kv: -kv[1]["two_afc_image"])
print(f"{'Cell':50s} {'2-AFC I':>8s} {'2-AFC B':>8s} {'Top-1 I':>8s} {'Top-1 B':>8s}")
for prefix, r in ranked:
    print(f"{r['label'][:50]:50s} {r['two_afc_image']*100:7.2f}% {r['two_afc_brain']*100:7.2f}% "
          f"{r['image_top1']*100:7.1f}% {r['brain_top1']*100:7.1f}%")
