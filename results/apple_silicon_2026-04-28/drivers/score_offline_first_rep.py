#!/usr/bin/env python3
"""Match paper Table 1 row "Offline 3T" (76%, no avg-3-reps suffix) using
FIRST-REP test βs only. This is to disambiguate whether our 76% reproduction
came from the avg-of-3-reps path or the first-rep path on fold-0.

Same pipeline as score_avg_repeats_offline.py except the test set is the
50 first-occurrence betas (no averaging).
"""
from __future__ import annotations
import json, sys, types, warnings
from collections import Counter
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

# Load ses-03 GLMsingle
gs = np.load(LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C/TYPED_FITHRF_GLMDENOISE_RR.npz", allow_pickle=True)
betasmd = gs["betasmd"].squeeze().astype(np.float32)
brain = nib.load(LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C/sub-005_ses-03_task-C_brain.nii.gz").get_fdata().flatten() > 0
betas_full = np.zeros((506160, betasmd.shape[1]), dtype=np.float32)
betas_full[brain] = np.nan_to_num(betasmd, nan=0.0)
final_mask = nib.load(LOCAL / "rt3t/data/sub-005_final_mask.nii.gz").get_fdata().flatten() > 0
relmask = np.load(LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy")
vox = betas_full[final_mask][relmask].T  # (693, 2792)

# Build vox_image_names
names = []
for run in range(1, 12):
    ev = pd.read_csv(LOCAL / f"rt3t/data/events/sub-005_ses-03_task-C_run-{run:02d}_events.tsv", sep="\t")
    ev = ev[ev["image_name"].astype(str).ne("blank.jpg") & ev["image_name"].astype(str).ne("nan") & ev["image_name"].notna()]
    names.extend(ev["image_name"].astype(str).tolist())
names = np.asarray(names)

# === FIRST-REP test set: keep only first occurrence of each special515 with 3 reps ===
counts = Counter(names)
seen = set()
test_idx = []
for i, n in enumerate(names):
    if counts[n] == 3 and "special515" in n and n not in seen:
        seen.add(n)
        test_idx.append(i)
test_idx = np.array(test_idx)
print(f"first-rep test indices: {len(test_idx)}")
assert len(test_idx) == 50

# Train mask: everything that's NOT a special515-3rep test trial (using all 3 reps for those that aren't tests)
train_idx = np.array([i for i in range(len(names)) if not (counts[names[i]] == 3 and "special515" in names[i])])
print(f"train trials: {len(train_idx)}")

# Train-only z-score (mean/std from train trials)
train_mean = vox[train_idx].mean(axis=0)
train_std = vox[train_idx].std(axis=0) + 1e-6
vox_z = (vox - train_mean) / train_std

test_betas = vox_z[test_idx]               # (50, 2792)
test_names = names[test_idx]                # (50,)

# Forward
print(f"=== fold-0 first-rep eval ===")
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval().requires_grad_(False)
preds = []
with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
    b = torch.from_numpy(test_betas.astype(np.float32)).to(device).unsqueeze(1)
    for i in range(0, b.shape[0], 8):
        vr = model.ridge(b[i:i+8], 0)
        out = model.backbone(vr)
        cv = out[1] if isinstance(out, tuple) else out
        preds.append(cv.float().cpu().numpy())
pred = np.concatenate(preds, 0).reshape(-1, ss, se)

# GT embeddings
img_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_names]
gt = compute_gt_mps(img_paths, device=device, cache_dir=LOCAL / "task_2_1_betas/gt_cache")

# Cosine top-k
def topk_acc(p, g, k=1):
    p_flat = p.reshape(p.shape[0], -1); g_flat = g.reshape(g.shape[0], -1)
    pn = p_flat / (np.linalg.norm(p_flat, axis=1, keepdims=True) + 1e-8)
    gn = g_flat / (np.linalg.norm(g_flat, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))

t1_fwd = topk_acc(pred, gt, k=1)
t1_bwd = topk_acc(gt, pred, k=1)
print(f"\n  TOP-1 fwd (image retr): {t1_fwd*100:.2f}%   [paper Offline 3T target: 76%]")
print(f"  TOP-1 bwd (brain retr): {t1_bwd*100:.2f}%   [paper Offline 3T target: 64%]")

(LOCAL / "task_2_1_betas/first_rep_offline_score.json").write_text(json.dumps({
    "method": "fold-0 first-rep, train-only z (test=50 first-occurrence special515 3-reps)",
    "n_test": 50, "top1_fwd": t1_fwd, "top1_bwd": t1_bwd,
    "paper_offline_3t_first_rep_target": {"image": 0.76, "brain": 0.64},
    "paper_offline_3t_avg3reps_target": {"image": 0.90, "brain": 0.88},
}, indent=2))
