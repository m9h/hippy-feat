#!/usr/bin/env python3
"""Fake-TRIBEv2 demo: linear cross-modal prior from CLIP → β patterns.

While recon-all runs (real TRIBEv2 needs the surfaces), we test the
foundation-model-prior framework with a poor-person's TRIBEv2:
  - Take the 543 ses-03 non-special515 training trials
  - Fit ridge regression: β (V=2792) = W · CLIP_image + b
  - For each of 50 special515 test images, predict the prior β pattern
  - Score template matching with these predicted patterns

This isolates the framework question — "do CLIP-conditioned per-image β
predictions help template matching?" — from the question of whether real
TRIBEv2 cross-subject training transfers (which we can only test once
recon-all + native morphs are done).

Comparison:
  - Empirical templates (leave-one-rep-out, already measured): 86.1% 2-AFC
  - Fake-TRIBEv2 templates (this script)
  - MindEye2 champion: 97.2% 2-AFC
"""
from __future__ import annotations

import json
import sys
import types
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
RT_MINDEYE = Path("/Users/mhough/Workspace/rt_mindEye2/src")
REPO = Path("/Users/mhough/Workspace/hippy-feat")
LOCAL_DRIVERS = Path("/Users/mhough/Workspace/local_drivers")
PREREG = LOCAL / "task_2_1_betas/prereg"
CACHE = LOCAL / "task_2_1_betas/gt_cache"

# stubs (compute_gt_mps imports lots of MindEye junk we don't need)
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
warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Load raw βs from champion config
raw = np.load(PREREG / "RT_paper_EoR_K7_CSFWM_HP_e1_RAW_ses-03_betas.npy")
trial_ids = np.load(PREREG / "RT_paper_EoR_K7_CSFWM_HP_e1_RAW_ses-03_trial_ids.npy", allow_pickle=True)
print(f"raw βs: {raw.shape}, trial_ids: {len(trial_ids)}", flush=True)

# Apply causal cum-z (matches our scoring convention)
def inclusive_cumz(arr):
    n, V = arr.shape
    z = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z

betas_z = inclusive_cumz(raw)
print(f"after inclusive cum-z: {betas_z.shape}", flush=True)

# LEAVE-ONE-IMAGE-OUT design: only special515 trials are local. For each held-out
# test image X, train fake-TRIBEv2 ridge on the OTHER 49 images' (CLIP, mean-β)
# pairs, predict β for X. This parallels what real TRIBEv2 substitution would do
# (predict β from image alone, no fMRI for that image).
spec_idx = []
for i, t in enumerate(trial_ids):
    if str(t).startswith("all_stimuli/special515/"):
        spec_idx.append(i)
print(f"  special515 trials (test): {len(spec_idx)}", flush=True)

# Compute CLIP embeddings for the special515 images. Use OpenCLIP ViT-bigG/14
# pooled features (1664-d).
import torchvision.transforms as T
from PIL import Image
from open_clip import create_model_and_transforms

print("\nloading OpenCLIP ViT-bigG/14 (pooled features)...", flush=True)
clip_m, _, _ = create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device)
clip_m.eval()

# Get unique special515 test images
test_images = sorted(set(str(trial_ids[i]) for i in spec_idx))
print(f"  unique test images: {len(test_images)}", flush=True)


def encode_image_pooled(img_path: Path) -> np.ndarray:
    pil = Image.open(img_path).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    t224 = T.functional.resize(t, (224, 224), antialias=False, interpolation=T.InterpolationMode.BILINEAR)
    t224 = t224.unsqueeze(0).to(device)
    img_norm = T.functional.normalize(t224,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
    with torch.no_grad():
        features = clip_m.visual(img_norm)
        if isinstance(features, tuple):
            features = features[0]
        # If pooled: shape (1, 1664). If tokens: shape (1, 256, 1664) — pool by mean.
        if features.ndim == 3:
            features = features.mean(dim=1)
    return features.float().cpu().numpy().squeeze(0)


# Encode test images
print("\nencoding 50 special515 images...", flush=True)
clip_test_dict = {}
for img_name in test_images:
    p = LOCAL / "rt3t/data" / img_name
    if p.exists():
        clip_test_dict[img_name] = encode_image_pooled(p)
print(f"  encoded {len(clip_test_dict)} test images", flush=True)

clip_test_arr = np.stack([clip_test_dict[img] for img in test_images], axis=0)  # (50, 1664)


# Per-image MEAN β across the 3 reps of each test image (50, 2792)
from collections import defaultdict as _dd
by_image = _dd(list)
for idx in spec_idx:
    by_image[str(trial_ids[idx])].append(idx)

mean_betas = np.stack([
    np.mean(np.stack([betas_z[i] for i in by_image[img]], axis=0), axis=0)
    for img in test_images], axis=0)  # (50, 2792)


# LEAVE-ONE-IMAGE-OUT fake-TRIBEv2: for each test image X, fit ridge on
# the OTHER 49 (CLIP_X', mean_β_X') pairs, predict mean_β_X.
ALPHA = 100.0
fake_tribev2 = np.zeros((50, 2792), dtype=np.float32)
print(f"\nfitting LOO fake-TRIBEv2 ridge (alpha={ALPHA})...", flush=True)
for held_out in range(50):
    mask = np.arange(50) != held_out
    clip_train = clip_test_arr[mask]
    beta_train = mean_betas[mask]
    clip_aug = np.concatenate([clip_train, np.ones((clip_train.shape[0], 1), dtype=np.float32)], axis=1)
    n_features = clip_aug.shape[1]
    gram = clip_aug.T @ clip_aug + ALPHA * np.eye(n_features, dtype=np.float64)
    W = np.linalg.solve(gram, clip_aug.T @ beta_train).astype(np.float32)
    test_aug = np.concatenate([clip_test_arr[held_out:held_out+1],
                                 np.ones((1, 1), dtype=np.float32)], axis=1)
    fake_tribev2[held_out] = (test_aug @ W).squeeze(0)
print(f"fake-TRIBEv2 (LOO) templates: {fake_tribev2.shape}", flush=True)

# Now: for each test trial, compute cosine similarity vs each fake-TRIBEv2 template
# Compare against empirical leave-one-rep-out templates (already known to give 86.1%)

# by_image already built above (during fake-TRIBEv2 LOO loop)
img_to_col = {img: i for i, img in enumerate(test_images)}


def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# Empirical templates per test image (mean of all 3 reps) — already computed as mean_betas
empirical_templates = mean_betas  # (50, 2792)

# Score each held-out trial against (a) empirical leave-one-out templates
# and (b) fake-TRIBEv2 templates (which are already image-conditioned predictions).

n_test = sum(len(v) for v in by_image.values())
true_idx = np.zeros(n_test, dtype=int)
sim_emp = np.zeros((n_test, 50), dtype=np.float64)
sim_fake = np.zeros((n_test, 50), dtype=np.float64)

k = 0
for img, rep_idxs in by_image.items():
    true_c = img_to_col[img]
    for held in rep_idxs:
        true_idx[k] = true_c
        beta_test = betas_z[held]
        for j, target_img in enumerate(test_images):
            # Empirical: leave-one-out target template
            if target_img == img:
                T_emp = (3.0 * empirical_templates[j] - beta_test) / 2.0
            else:
                T_emp = empirical_templates[j]
            sim_emp[k, j] = cos(beta_test, T_emp)
            # Fake-TRIBEv2: image-conditioned prediction
            sim_fake[k, j] = cos(beta_test, fake_tribev2[j])
        k += 1


def score(sim, true_idx):
    n_test = sim.shape[0]
    top1 = float((sim.argmax(axis=1) == true_idx).mean())
    top5 = float(np.mean([true_idx[k] in np.argsort(sim[k])[-5:] for k in range(n_test)]))
    # 2-AFC
    correct, total = 0, 0
    for k in range(n_test):
        t = true_idx[k]
        for d in range(50):
            if d == t: continue
            if sim[k, t] > sim[k, d]: correct += 1
            total += 1
    two_afc = correct / total
    diag = sim[np.arange(n_test), true_idx]
    off_mask = np.ones((n_test, 50), dtype=bool)
    off_mask[np.arange(n_test), true_idx] = False
    off = sim[off_mask]
    d = (diag.mean() - off.mean()) / np.sqrt(0.5 * (diag.var() + off.var()) + 1e-12)
    return {"top1": top1, "top5": top5, "two_afc": two_afc, "cohens_d": float(d)}


emp_res = score(sim_emp, true_idx)
fake_res = score(sim_fake, true_idx)

print("\n=== Template-matching comparison ===", flush=True)
print(f"{'method':30s}  {'top1':>6s}  {'top5':>6s}  {'2-AFC':>6s}  {'d':>5s}")
print(f"{'empirical (LOO templates)':30s}  "
      f"{emp_res['top1']*100:5.1f}%  {emp_res['top5']*100:5.1f}%  "
      f"{emp_res['two_afc']*100:5.1f}%  {emp_res['cohens_d']:4.2f}")
print(f"{'fake-TRIBEv2 (CLIP→β ridge)':30s}  "
      f"{fake_res['top1']*100:5.1f}%  {fake_res['top5']*100:5.1f}%  "
      f"{fake_res['two_afc']*100:5.1f}%  {fake_res['cohens_d']:4.2f}")
print(f"{'MindEye2 champion':30s}    58.0%   88.0%   97.2%  2.42")

out_path = LOCAL / "task_2_1_betas/fake_tribev2_demo.json"
out_path.write_text(json.dumps({
    "method": "fake-TRIBEv2 = leave-one-image-out CLIP→β ridge across 50 special515 images",
    "ridge_alpha": ALPHA,
    "n_test_trials": int(n_test),
    "n_unique_test_images": len(by_image),
    "empirical_template_match": emp_res,
    "fake_tribev2_template_match": fake_res,
    "comparison_mindeye2": {"top1": 0.58, "top5": 0.88, "two_afc": 0.972, "cohens_d": 2.42},
}, indent=2))
print(f"\nsaved {out_path.name}")
