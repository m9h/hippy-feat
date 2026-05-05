#!/usr/bin/env python3
"""Compute Offline subset0/subset1/subset2 from canonical GLMsingle, the
direct apples-to-apples comparison to EoR rtmotion+nilearn subsets.

Pipeline matches recon_inference-multisession.ipynb's `repeats_3` train/test
split exactly, except we vary how the test βs are aggregated across the 3
repeats per image:

  subset0 = β of rep-0 only (single-rep)
  subset1 = mean(β rep-0, β rep-1)
  subset2 = mean(β rep-0, β rep-1, β rep-2)   ← what filter_and_average_repeats does

Train side: filter_and_average_repeats over the 543 non-test trials (481 unique
images), then z-score using train mean/std.

Test side: each of subset0/1/2 z-scored with the same train mean/std.

Output table:
                         subset0    subset1    subset2
  EoR rtmotion+nilearn   56         62         78
  Offline GLMsingle       ?          ?         76 (matches existing)
"""
from __future__ import annotations
import json, sys, types, warnings
from collections import Counter, defaultdict
from copy import deepcopy
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


def filter_and_average_repeats(vox, names):
    """Verbatim port of mindeye_offline:utils.py:800."""
    repeats = {}
    for i, im in enumerate(names):
        repeats.setdefault(im, []).append(i)
    keep = np.ones(len(vox), dtype=bool)
    out = deepcopy(vox).astype(np.float32)
    for idxs in repeats.values():
        if len(idxs) > 1:
            out[idxs[0]] = vox[idxs].mean(axis=0)
            keep[idxs[1:]] = False
    return out[keep], np.where(keep)[0]


# === Load canonical GLMsingle βs ===
print("=== loading canonical GLMsingle βs (sub-005 ses-03) ===", flush=True)
gs = np.load(LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C/TYPED_FITHRF_GLMDENOISE_RR.npz",
             allow_pickle=True)
betasmd = gs["betasmd"].squeeze().astype(np.float32)        # (V_brain, 693)
brain = nib.load(LOCAL / "glmsingle/glmsingle_sub-005_ses-03_task-C/sub-005_ses-03_task-C_brain.nii.gz").get_fdata().flatten() > 0
betas_full = np.zeros((506160, betasmd.shape[1]), dtype=np.float32)
betas_full[brain] = np.nan_to_num(betasmd, nan=0.0)
final_mask = nib.load(LOCAL / "rt3t/data/sub-005_final_mask.nii.gz").get_fdata().flatten() > 0
relmask = np.load(LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy")
vox = betas_full[final_mask][relmask].T                      # (693, 2792)

# Build vox_image_names for ses-03 (693 trials in chronological order)
names = []
for run in range(1, 12):
    ev = pd.read_csv(LOCAL / f"rt3t/data/events/sub-005_ses-03_task-C_run-{run:02d}_events.tsv", sep="\t")
    ev = ev[ev["image_name"].astype(str).ne("blank.jpg")
            & ev["image_name"].astype(str).ne("nan")
            & ev["image_name"].notna()]
    names.extend(ev["image_name"].astype(str).tolist())
names = np.asarray(names)
assert len(names) == vox.shape[0]
print(f"  vox: {vox.shape}, names: {len(names)}", flush=True)

# === Identify test images (50 special515 with 3 reps) ===
counts = Counter(names)
test_imgs = sorted({n for n in names if counts[n] == 3 and "special515" in n})
assert len(test_imgs) == 50

# Group test trial indices by image (3 indices per image, in chronological order)
test_idx_per_img = defaultdict(list)
for i, n in enumerate(names):
    if n in set(test_imgs):
        test_idx_per_img[n].append(i)
for n in test_imgs:
    assert len(test_idx_per_img[n]) == 3

# Train trial indices
test_idx_set = {i for idxs in test_idx_per_img.values() for i in idxs}
train_idx = np.array([i for i in range(len(names)) if i not in test_idx_set])
print(f"  train trials: {len(train_idx)}, test trials (50×3): {len(test_idx_set)}", flush=True)

# === Compute train mean/std using filter_and_average_repeats over train side ===
train_vox = vox[train_idx]
train_names = names[train_idx]
train_avg, _ = filter_and_average_repeats(train_vox, train_names)
print(f"  train after filter_and_average_repeats: {train_avg.shape}", flush=True)
train_mean = train_avg.mean(axis=0)
train_std = train_avg.std(axis=0) + 1e-6

# === Build test βs at each subset, in fixed image order ===
def build_test(level: int):
    """level=0 → β of rep-0; level=1 → mean(rep-0,rep-1); level=2 → mean(all 3)."""
    rows = []
    for img in test_imgs:
        idxs = test_idx_per_img[img][:level + 1]
        if level == 0:
            rows.append(vox[idxs[0]])
        else:
            rows.append(vox[idxs].mean(axis=0))
    return np.stack(rows)            # (50, 2792)

subsets = {n: build_test(n) for n in (0, 1, 2)}

# === Z-score with train stats ===
for n in subsets:
    subsets[n] = (subsets[n] - train_mean) / train_std

# === Forward through fold-0 ===
print(f"\n=== loading fold-0 ckpt ===", flush=True)
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


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


img_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_imgs]
gt = compute_gt_mps(img_paths, device=device, cache_dir=LOCAL / "task_2_1_betas/gt_cache")

results = {}
print("\n=== Offline subsets ===")
for n in (0, 1, 2):
    pred = fwd(subsets[n])
    i_acc = topk(pred, gt, k=1)
    b_acc = topk(gt, pred, k=1)
    results[f"subset{n}"] = {"image": i_acc, "brain": b_acc}
    print(f"  subset{n}: Image={i_acc*100:5.2f}%  Brain={b_acc*100:5.2f}%")

print("\n=== comparison: rtmotion+nilearn EoR (from factorial_subsets_fold0.json) ===")
print("  cell                                          subset0  subset1  subset2")
print("  RT_paper_replica_partial                       52       64       74")
print("  RT_paper_EndOfRun_pst_None_inclz               56       62       78")
print("  RT_paper_EoR_K7_CSFWM_HP_e1_inclz              54       66       76")
print("\n=== Offline (this script) ===")
for n in (0, 1, 2):
    print(f"  Offline GLMsingle subset{n}                   "
          f"{int(results[f'subset{n}']['image']*100)}       "
          f"{int(results[f'subset{n}']['image']*100):3d}      ", end="")
    print()

(LOCAL / "task_2_1_betas/offline_subsets_fold0.json").write_text(json.dumps(results, indent=2))
print(f"\n  saved offline_subsets_fold0.json")
