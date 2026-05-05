#!/usr/bin/env python3
"""Score the locally-produced GLMsingle-on-fmriprep output (#1 of the
fmriprep ablation triplet) at subset0/1/2.

Compares to:
- Canonical Princeton GLMsingle on fmriprep (subset0=62, subset1=76, subset2=76)
- Our local GLMsingle on rtmotion (RTmotion_GLMsingle cells)
"""
from __future__ import annotations
import json, sys, types, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np, nibabel as nib, pandas as pd, torch, torch.nn as nn

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

# === Load local-fmriprep GLMsingle output ===
print("=== loading local GLMsingle-on-fmriprep ===", flush=True)
gs_arr = np.load(LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C_FMRIPREP_LOCAL/TYPED_FITHRF_GLMDENOISE_RR.npy",
                 allow_pickle=True)
gs = gs_arr.item()
betasmd = gs["betasmd"].astype(np.float32)               # (76, 90, 74, 693)
print(f"  betasmd: {betasmd.shape}, pcnum={gs['pcnum']}", flush=True)

# Reshape (X, Y, Z, T) → (V, T) → apply finalmask + relmask
V = np.prod(betasmd.shape[:3])
betas_flat = betasmd.reshape(V, betasmd.shape[-1])

# Try direct masking first (assumes same XYZ as final_mask)
final_mask = nib.load(LOCAL / "rt3t/data/sub-005_final_mask.nii.gz")
fm_3d = final_mask.get_fdata() > 0
print(f"  finalmask volume shape: {fm_3d.shape}, sum: {fm_3d.sum()}", flush=True)
print(f"  GLMsingle output volume shape: {betasmd.shape[:3]}", flush=True)
if fm_3d.shape != betasmd.shape[:3]:
    raise ValueError(f"shape mismatch — need to resample")

fm_flat = fm_3d.flatten()
relmask = np.load(LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy")
betas_finalmask = betas_flat[fm_flat]                    # (19174, 693)
betas_2792 = betas_finalmask[relmask]                    # (2792, 693)
vox = betas_2792.T                                        # (693, 2792)
print(f"  vox: {vox.shape}", flush=True)

# Build vox_image_names
names = []
for run in range(1, 12):
    ev = pd.read_csv(LOCAL / f"rt3t/data/events/sub-005_ses-03_task-C_run-{run:02d}_events.tsv", sep="\t")
    ev = ev[ev["image_name"].astype(str).ne("blank.jpg")
            & ev["image_name"].astype(str).ne("nan")
            & ev["image_name"].notna()]
    names.extend(ev["image_name"].astype(str).tolist())
names = np.asarray(names)
assert len(names) == vox.shape[0]

# Identify test images
counts = Counter(names)
test_imgs = sorted({n for n in names if counts[n] == 3 and "special515" in n})
assert len(test_imgs) == 50

# Group test trial indices by image
test_idx_per_img = defaultdict(list)
for i, n in enumerate(names):
    if n in set(test_imgs):
        test_idx_per_img[n].append(i)

# Train indices (everything not in 50×3 test trials)
test_idx_set = {i for idxs in test_idx_per_img.values() for i in idxs}
train_idx = np.array([i for i in range(len(names)) if i not in test_idx_set])

# Train mean/std (filter_and_average_repeats over train side)
from copy import deepcopy
def fil(v, n):
    rep = {}
    for i, im in enumerate(n):
        rep.setdefault(im, []).append(i)
    keep = np.ones(len(v), dtype=bool); out = deepcopy(v).astype(np.float32)
    for idxs in rep.values():
        if len(idxs) > 1:
            out[idxs[0]] = v[idxs].mean(axis=0); keep[idxs[1:]] = False
    return out[keep]

train_avg = fil(vox[train_idx], names[train_idx])
print(f"  train after avg-of-reps: {train_avg.shape}", flush=True)
train_mean = train_avg.mean(axis=0)
train_std = train_avg.std(axis=0) + 1e-6

# Build test βs at subset0/1/2
def build(level):
    rows = []
    for img in test_imgs:
        idxs = test_idx_per_img[img][:level + 1]
        rows.append(vox[idxs[0]] if level == 0 else vox[idxs].mean(axis=0))
    return np.stack(rows)

subs = {n: (build(n) - train_mean) / train_std for n in (0, 1, 2)}

# Forward fold-0
print("\n=== forward fold-0 ===", flush=True)
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

def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T; labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))

img_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_imgs]
gt = compute_gt_mps(img_paths, device=device, cache_dir=LOCAL / "task_2_1_betas/gt_cache")

results = {}
for n in (0, 1, 2):
    pred = fwd(subs[n])
    img_acc = topk(pred, gt, k=1)
    bra_acc = topk(gt, pred, k=1)
    results[f"subset{n}"] = {"image": img_acc, "brain": bra_acc}
    label = ["single-rep", "avg-of-2", "complete-set"][n]
    print(f"  subset{n} ({label}): Image={img_acc*100:5.2f}%  Brain={bra_acc*100:5.2f}%")

(LOCAL / "task_2_1_betas/fmriprep_glmsingle_local_subsets.json").write_text(json.dumps(results, indent=2))
print(f"\n  saved fmriprep_glmsingle_local_subsets.json")
