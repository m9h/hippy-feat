#!/usr/bin/env python3
"""Test the post-model-averaging hypothesis for paper Table 1's
"Offline 3T (avg. 3 reps.)" row at 90% Image / 88% Brain.

Our current reproduction with pre-model β-averaging
(`utils.filter_and_average_repeats`) gives 76% Image / 88% Brain.

If paper does post-model averaging instead — run model 3× per image (one rep
at a time, no β averaging), then average the 3 clip_voxel outputs — the
result could differ because of the nonlinear backbone.

Pipeline:
  1. Load ses-03 GLMsingle (per-trial βs, 693 trials)
  2. Project to finalmask + relmask (2792 vox)
  3. For the 50 special515 with 3 repeats: keep ALL 3 trial βs per image (150 βs)
  4. Train-only z-score using mean/std from non-test trials
  5. Forward through fold-0 ckpt for all 150 reps, get clip_voxels (150, 256, 1664)
  6. Average the 3 outputs per image → (50, 256, 1664)
  7. Cosine top-1 vs CLIP-image GT (50, 256, 1664)

Compare to:
  - Pre-model β-avg (76% Image / 88% Brain) — from score_avg_repeats_offline.py
  - First-rep      (60% Image / 64% Brain) — from score_offline_first_rep.py
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

# Load ses-03 GLMsingle
gs = np.load(LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C/TYPED_FITHRF_GLMDENOISE_RR.npz",
             allow_pickle=True)
betasmd = gs["betasmd"].squeeze().astype(np.float32)
brain = nib.load(LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C/sub-005_ses-03_task-C_brain.nii.gz").get_fdata().flatten() > 0
betas_full = np.zeros((506160, betasmd.shape[1]), dtype=np.float32)
betas_full[brain] = np.nan_to_num(betasmd, nan=0.0)
final_mask = nib.load(LOCAL / "rt3t/data/sub-005_final_mask.nii.gz").get_fdata().flatten() > 0
relmask = np.load(LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy")
vox = betas_full[final_mask][relmask].T   # (693, 2792)

# Build vox_image_names
names = []
for run in range(1, 12):
    ev = pd.read_csv(LOCAL / f"rt3t/data/events/sub-005_ses-03_task-C_run-{run:02d}_events.tsv", sep="\t")
    ev = ev[ev["image_name"].astype(str).ne("blank.jpg")
            & ev["image_name"].astype(str).ne("nan")
            & ev["image_name"].notna()]
    names.extend(ev["image_name"].astype(str).tolist())
names = np.asarray(names)

# Identify the 50 special515 with 3 repeats
counts = Counter(names)
test_imgs_3rep = sorted({n for n in names if counts[n] == 3 and "special515" in n})
assert len(test_imgs_3rep) == 50

# Group all 150 trial indices by image (3 indices per image)
by = defaultdict(list)
for i, n in enumerate(names):
    if n in test_imgs_3rep:
        by[n].append(i)
for n in test_imgs_3rep:
    assert len(by[n]) == 3, f"{n} has {len(by[n])} reps, expected 3"

# Train mask: trials NOT in the test set
test_trial_idx = sorted([i for n in test_imgs_3rep for i in by[n]])
train_trial_idx = sorted(set(range(len(names))) - set(test_trial_idx))
print(f"  test trials: {len(test_trial_idx)} (50 imgs × 3 reps), train: {len(train_trial_idx)}")

# Train-only z-score (NOTE: paper uses post-avg train mean/std, but we have
# raw trials so we use raw-train-trial mean/std. The pre-model-avg path uses
# avg-train means; the difference matters slightly).
train_mean = vox[train_trial_idx].mean(axis=0)
train_std = vox[train_trial_idx].std(axis=0) + 1e-6
vox_z = (vox - train_mean) / train_std

# === Forward through fold-0 ===
print(f"\n=== fold-0 post-model averaging ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval().requires_grad_(False)

# Build (150, 2792) test β tensor in image-grouped order (3 reps consecutive per image)
ordered_idx = [i for n in test_imgs_3rep for i in by[n]]
test_betas_150 = vox_z[ordered_idx]                # (150, 2792)

preds_150 = []
with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
    b = torch.from_numpy(test_betas_150.astype(np.float32)).to(device).unsqueeze(1)
    for i in range(0, b.shape[0], 8):
        vr = model.ridge(b[i:i+8], 0)
        out = model.backbone(vr)
        cv = out[1] if isinstance(out, tuple) else out
        preds_150.append(cv.float().cpu().numpy())
pred_150 = np.concatenate(preds_150, 0).reshape(-1, ss, se)    # (150, 256, 1664)

# Average 3 outputs per image → (50, 256, 1664)
pred_50_postavg = pred_150.reshape(50, 3, ss, se).mean(axis=1)

# GT embeddings
img_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_imgs_3rep]
gt = compute_gt_mps(img_paths, device=device, cache_dir=LOCAL / "task_2_1_betas/gt_cache")

# Cosine top-k
def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))

t1_fwd_post = topk(pred_50_postavg, gt, k=1)
t1_bwd_post = topk(gt, pred_50_postavg, k=1)
t5_fwd_post = topk(pred_50_postavg, gt, k=5)

# Bonus: per-rep individual scoring (no averaging) for context
preds_per_rep = pred_150.reshape(50, 3, ss, se)
for rep_idx in range(3):
    p = preds_per_rep[:, rep_idx]
    print(f"    rep-{rep_idx} only:  Image={topk(p, gt, k=1)*100:5.2f}%   Brain={topk(gt, p, k=1)*100:5.2f}%")

print(f"\n  POST-MODEL AVG: Image={t1_fwd_post*100:5.2f}%   Brain={t1_bwd_post*100:5.2f}%   top-5={t5_fwd_post*100:5.2f}%")
print(f"  paper (avg 3 reps): Image=90.00%  Brain=88.00%")
print(f"  paper (Offline 3T): Image=76.00%  Brain=64.00%")
print(f"  pre-model avg:      Image=76.00%  Brain=88.00%   (from score_avg_repeats_offline.py)")
print(f"  first-rep:          Image=60.00%  Brain=64.00%   (from score_offline_first_rep.py)")

result = {
    "method": "post-model averaging on fold-0",
    "n_test": 50,
    "post_model_avg": {"image_top1": t1_fwd_post, "brain_top1": t1_bwd_post, "image_top5": t5_fwd_post},
    "per_rep_individual": [{"rep": r, "image": topk(preds_per_rep[:, r], gt, k=1),
                            "brain": topk(gt, preds_per_rep[:, r], k=1)} for r in range(3)],
    "paper_avg3reps_target": {"image": 0.90, "brain": 0.88},
    "paper_offline3t_target": {"image": 0.76, "brain": 0.64},
    "pre_model_avg_baseline":  {"image": 0.76, "brain": 0.88},
    "first_rep_baseline":      {"image": 0.60, "brain": 0.64},
}
out = LOCAL / "task_2_1_betas/post_model_avg_score.json"
out.write_text(json.dumps(result, indent=2))
print(f"\n  saved {out.name}")
