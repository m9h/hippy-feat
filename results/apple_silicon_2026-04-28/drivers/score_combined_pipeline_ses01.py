#!/usr/bin/env python3
"""Deploy combined pipeline (Fast refiner + streaming Slow GLM) on ses-01.

ses-01 special515 are 50 different images from ses-03's special515 — true
held-out test. The refiner was trained on ses-03 rtmotion BOLD; ses-01 βs are
fmriprep BOLD. Tests cross-session AND cross-BOLD-source generalization.

Comparisons:
  baseline_fast       = fold-0 on ses-01 Fast pst=5 fmriprep
  refined_fast        = fold-0 on refiner(ses-01 Fast pst=5 fmriprep)
  streaming_slow      = fold-0 on ses-01 streaming Slow pst=20 fmriprep
  refined_streaming   = fold-0 on refiner(ses-01 streaming Slow)
                        (sanity — refiner shouldn't help when input is already Slow)

Subset modes: subset0 (single-rep) and subset1 (avg-of-2) for each.
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
SES = "ses-01"


class PerVoxelScalar(nn.Module):
    def __init__(self, n_vox=2792):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_vox))
        self.bias = nn.Parameter(torch.zeros(n_vox))
    def forward(self, x):
        return x * self.gain + self.bias


def get_subsets(b, ids):
    """Restrict to the 50 special515 images with exactly 3 reps (the held-out test set)."""
    counts = Counter([str(t) for t in ids])
    test_imgs = {n for n, c in counts.items() if c == 3 and "special515" in n}
    by = defaultdict(list)
    for i, t in enumerate(ids):
        if str(t) in test_imgs:
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


def fwd_eval(model, ss, se, x, batch=8):
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = x.unsqueeze(1)
        for i in range(0, b.shape[0], batch):
            vr = model.ridge(b[i:i+batch], 0)
            o = model.backbone(vr)
            cv = o[1] if isinstance(o, tuple) else o
            out.append(cv.float())
    return torch.cat(out, 0).reshape(-1, ss, se)


print("=== loading fold-0 ckpt ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval().requires_grad_(False)


# Load refiner (saved by train_fast_distill_from_slow.py)
refiner_state = torch.load(LOCAL / "task_2_1_betas/fast_refiner_state.pth", map_location=device, weights_only=True)
refiner = PerVoxelScalar(n_vox=2792).to(device)
refiner.load_state_dict(refiner_state)
refiner.eval()
print(f"  refiner gain mean={refiner.gain.mean().item():.3f}, bias mean={refiner.bias.mean().item():.3f}", flush=True)


# Load ses-01 βs
fast_b = np.load(PREREG / f"RT_paper_Fast_pst5_fmriprep_inclz_{SES}_betas.npy")
fast_ids = np.load(PREREG / f"RT_paper_Fast_pst5_fmriprep_inclz_{SES}_trial_ids.npy")
fast_ids = np.asarray([str(t) for t in fast_ids])
slow_b = np.load(PREREG / f"RT_paper_RLS_AR1_pst20_K7CSFWM_HP_e1_inclz_distill_fmriprep_{SES}_betas.npy")
slow_ids = np.load(PREREG / f"RT_paper_RLS_AR1_pst20_K7CSFWM_HP_e1_inclz_distill_fmriprep_{SES}_trial_ids.npy")
slow_ids = np.asarray([str(t) for t in slow_ids])
assert (fast_ids == slow_ids).all()


fast_subs = get_subsets(fast_b, fast_ids)
slow_subs = get_subsets(slow_b, slow_ids)


# Score one config
def score(betas, names, label):
    paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in names]
    gt = compute_gt_mps(paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
    x = torch.from_numpy(betas.astype(np.float32)).to(device)
    p = fwd_eval(model, ss, se, x, batch=8).cpu().numpy()
    img = topk(p, gt, 1); bra = topk(gt, p, 1)
    print(f"  {label}: Image={img*100:5.1f}%  Brain={bra*100:5.1f}%", flush=True)
    return img, bra


def score_refined(betas, names, label):
    paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in names]
    gt = compute_gt_mps(paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
    x = torch.from_numpy(betas.astype(np.float32)).to(device)
    with torch.no_grad():
        x_ref = refiner(x)
    p = fwd_eval(model, ss, se, x_ref, batch=8).cpu().numpy()
    img = topk(p, gt, 1); bra = topk(gt, p, 1)
    print(f"  {label}: Image={img*100:5.1f}%  Brain={bra*100:5.1f}%", flush=True)
    return img, bra


print(f"\n=== ses-01 deployment test ===", flush=True)
for sub_n in (0, 1, 2):
    if sub_n not in fast_subs or sub_n not in slow_subs:
        continue
    fb, fnames = fast_subs[sub_n]
    sb, snames = slow_subs[sub_n]
    print(f"\n--- subset{sub_n} ({['single-rep','avg-of-2','complete-set'][sub_n]}) ---", flush=True)
    score(fb, fnames, "Fast (no refiner)")
    score_refined(fb, fnames, "Fast (refined, ses-03-trained refiner applied)")
    score(sb, snames, "streaming Slow GLM (no refiner)")
    score_refined(sb, snames, "streaming Slow GLM (refined — sanity, should not help)")


print("\n=== compared to ses-03 (the training session) ===")
print("  ses-03 Fast baseline: 36% Image, 34% Brain")
print("  ses-03 Fast refined: 40% Image, 48% Brain (v1)")
print("  ses-03 streaming Slow s0: ~52% Image (subset0), ~70% Image (subset1)")
