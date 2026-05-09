#!/usr/bin/env python3
"""Score 4mm-smoothed fmriprep cells (Fast/Slow/EoR) at subset0/1/2 vs the
unsmoothed fmriprep baselines."""
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
    ("Fast", "RT_paper_Fast_pst5_fmriprep_sm4mm_inclz",  "RT_paper_Fast_pst5_fmriprep_inclz"),
    ("Slow", "RT_paper_Slow_pst20_fmriprep_sm4mm_inclz", "RT_paper_Slow_pst20_fmriprep_inclz"),
    ("EoR",  "RT_paper_EoR_fmriprep_sm4mm_inclz",         "RT_paper_EoR_fmriprep_inclz"),
]


def score_cell(prefix):
    bf = PREREG / f"{prefix}_ses-03_betas.npy"
    if not bf.exists():
        return None
    betas = np.load(bf)
    ids = np.load(PREREG / f"{prefix}_ses-03_trial_ids.npy")
    subs = get_subsets(betas, ids)
    out = {}
    for n in (0, 1, 2):
        if n not in subs: continue
        b, names = subs[n]
        paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(nm).name for nm in names]
        gt = compute_gt_mps(paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
        pred = fwd(b)
        out[f"subset{n}"] = {"image": topk(pred, gt, 1), "brain": topk(gt, pred, 1)}
    return out


results = {}
for label, smooth_prefix, base_prefix in CELLS:
    print(f"\n--- {label}: {smooth_prefix} vs {base_prefix} ---", flush=True)
    sm = score_cell(smooth_prefix)
    bs = score_cell(base_prefix)
    if sm is None:
        print(f"  SKIP — smoothed cell missing", flush=True); continue
    if bs is None:
        print(f"  SKIP — baseline cell missing", flush=True); continue
    row = {"smoothed": sm, "unsmoothed": bs, "delta_image": {}, "delta_brain": {}}
    for n in (0, 1, 2):
        k = f"subset{n}"
        if k in sm and k in bs:
            di = (sm[k]["image"] - bs[k]["image"]) * 100
            db = (sm[k]["brain"] - bs[k]["brain"]) * 100
            row["delta_image"][k] = di; row["delta_brain"][k] = db
            slabel = ["single-rep", "avg-of-2", "complete-set"][n]
            print(f"  {slabel:13s}: 4mm I={sm[k]['image']*100:5.1f}%  vs unsm I={bs[k]['image']*100:5.1f}%  ΔI={di:+5.1f}pp  |  4mm B={sm[k]['brain']*100:5.1f}%  vs unsm B={bs[k]['brain']*100:5.1f}%  ΔB={db:+5.1f}pp", flush=True)
    results[label] = row

(LOCAL / "task_2_1_betas/smoothing_4mm_subsets_fold0.json").write_text(json.dumps(results, indent=2))
print(f"\nsaved smoothing_4mm_subsets_fold0.json", flush=True)

print("\n========== SUMMARY: 4mm fmriprep smoothing vs unsmoothed ==========")
print(f"{'Tier':6s} {'Subset':12s} {'4mm':>8s} {'unsm':>8s} {'ΔI':>6s}    {'4mmB':>6s} {'unsmB':>7s} {'ΔB':>6s}")
for tier in ("Fast", "Slow", "EoR"):
    if tier not in results: continue
    r = results[tier]
    for sub in (0, 1, 2):
        k = f"subset{sub}"
        if k not in r["smoothed"] or k not in r["unsmoothed"]: continue
        s = r["smoothed"][k]["image"] * 100; u = r["unsmoothed"][k]["image"] * 100
        sb = r["smoothed"][k]["brain"] * 100; ub = r["unsmoothed"][k]["brain"] * 100
        di = s - u; db = sb - ub
        marker = " *" if abs(di) >= 4.0 else ""
        print(f"{tier:6s} subset{sub}     {s:7.1f}% {u:7.1f}% {di:+6.1f}    {sb:5.1f}% {ub:6.1f}% {db:+6.1f}{marker}")
