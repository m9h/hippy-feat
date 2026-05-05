#!/usr/bin/env python3
"""Fast-tier teacher-student refiner.

Architecture:
  input (B, 2792) Fast pst=5 cum-z'd β
    → per-voxel refiner (gain · β + bias), 5584 trainable params, init gain=1, bias=0
    → frozen fold-0 ckpt
    → clip_voxels (B, 256, 1664)
  Loss: 1 - cosine_sim(pred_clip_voxels, CLIP-image-tokens)

Training data: ses-03 non-test trials + ses-01 + ses-02 (training session) trials.
Test: ses-03 first-rep special515 (the 50 we always test).

Compares against:
  - fold-0 frozen baseline: 36% Image (paper anchor)
  - any improvement is a real Fast-tier gain
"""
from __future__ import annotations
import json, sys, types, warnings, time
from collections import Counter
from pathlib import Path

import numpy as np, torch, torch.nn as nn
import torch.nn.functional as F

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


# === Per-voxel refiner ===
class PerVoxelRefiner(nn.Module):
    def __init__(self, n_vox=2792):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_vox))
        self.bias = nn.Parameter(torch.zeros(n_vox))

    def forward(self, x):
        # x: (B, V)
        return x * self.gain + self.bias


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


def cosine_loss(pred, gt):
    p = F.normalize(pred.reshape(pred.shape[0], -1), dim=-1)
    g = F.normalize(gt.reshape(gt.shape[0], -1), dim=-1)
    return 1 - (p * g).sum(dim=-1).mean()


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


def fwd_train(model, x, batch=8):
    out = []
    b = x.unsqueeze(1)
    for i in range(0, b.shape[0], batch):
        vr = model.ridge(b[i:i+batch], 0)
        o = model.backbone(vr)
        cv = o[1] if isinstance(o, tuple) else o
        out.append(cv)
    return torch.cat(out, 0)


# === Load fold-0 frozen ===
print("=== loading fold-0 ckpt (frozen) ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval()
for p in model.parameters():
    p.requires_grad = False


# === Resolve image path for any trial name ===
def resolve_image(name):
    if not isinstance(name, str): return None
    base = Path(name).name
    for sub in ("special515", "MST_pairs", "unchosen_nsd_1000_images", "shared1000_notspecial"):
        p = LOCAL / "rt3t/data/all_stimuli" / sub / base
        if p.exists(): return p
    p = LOCAL / "rt3t/data/all_stimuli" / name
    return p if p.exists() else None


# === Build training set: ses-03 non-test trials + ses-01 + ses-02 ===
print("\n=== building training set ===", flush=True)
train_betas_list = []
train_imgs_list = []

# ses-03 (rtmotion Fast pst=5)
b03 = np.load(PREREG / "RT_paper_Fast_pst5_inclz_ses-03_betas.npy")
ids03 = np.load(PREREG / "RT_paper_Fast_pst5_inclz_ses-03_trial_ids.npy")
ids03 = np.asarray([str(t) for t in ids03])
counts = Counter(ids03)
test_imgs_set = {n for n in ids03 if counts[n] == 3 and "special515" in n}
test_idx_set = {i for i, n in enumerate(ids03) if n in test_imgs_set}
# Keep ONLY first occurrence of each test image as TEST; everything else is TRAIN
test_first_idx = []
seen = set()
for i, n in enumerate(ids03):
    if n in test_imgs_set and n not in seen:
        seen.add(n); test_first_idx.append(i)
test_first_idx = np.array(sorted(test_first_idx))
# Train = all trials NOT in test_first_idx (so other 2 reps of test images are EXCLUDED for cleanliness)
train_idx_03 = np.array([i for i in range(len(ids03)) if i not in test_idx_set])
print(f"  ses-03: {len(train_idx_03)} train trials (excluding all 3 reps of test imgs), {len(test_first_idx)} test trials", flush=True)

for i in train_idx_03:
    p = resolve_image(ids03[i])
    if p is not None:
        train_betas_list.append(b03[i])
        train_imgs_list.append(p)

# ses-01, ses-02 (fmriprep Fast pst=5) — if available
for ses in ["ses-01", "ses-02"]:
    bp = PREREG / f"RT_paper_Fast_pst5_fmriprep_inclz_{ses}_betas.npy"
    if not bp.exists():
        print(f"  {ses}: SKIP (Fast βs not extracted yet)", flush=True)
        continue
    b = np.load(bp)
    ids = np.load(PREREG / f"RT_paper_Fast_pst5_fmriprep_inclz_{ses}_trial_ids.npy")
    ids = np.asarray([str(t) for t in ids])
    cnt = 0
    for i in range(len(ids)):
        p = resolve_image(ids[i])
        if p is not None and "special515" not in ids[i]:
            # Don't include any special515 (test set generalization)
            train_betas_list.append(b[i])
            train_imgs_list.append(p)
            cnt += 1
    print(f"  {ses}: +{cnt} train trials with images on disk", flush=True)

train_betas = np.stack(train_betas_list)
print(f"  total training pairs: {train_betas.shape}", flush=True)

# Build CLIP-image embeddings for training
print("\n=== building CLIP-image embeddings (train) ===", flush=True)
gt_train = compute_gt_mps(train_imgs_list, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
gt_train = np.asarray(gt_train)
print(f"  gt_train: {gt_train.shape}", flush=True)

# Test set
test_betas = b03[test_first_idx]                                 # (50, 2792)
test_names = [str(ids03[i]) for i in test_first_idx]
test_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_names]
gt_test = compute_gt_mps(test_paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
gt_test = np.asarray(gt_test)
print(f"  gt_test: {gt_test.shape}", flush=True)

# Train/val split
n_train = train_betas.shape[0]
rng = np.random.RandomState(42)
perm = rng.permutation(n_train)
n_val = max(int(n_train * 0.1), 30)
val_idx = perm[:n_val]
tr_idx = perm[n_val:]
print(f"  train split: {len(tr_idx)} train / {n_val} val", flush=True)

X_tr = torch.from_numpy(train_betas[tr_idx].astype(np.float32)).to(device)
gt_tr = torch.from_numpy(gt_train[tr_idx].astype(np.float32)).to(device)
X_val = torch.from_numpy(train_betas[val_idx].astype(np.float32)).to(device)
gt_val = gt_train[val_idx]
X_test = torch.from_numpy(test_betas.astype(np.float32)).to(device)


# === Baseline (frozen fold-0, no refiner) on test ===
p_base = fwd_eval(model, ss, se, X_test, batch=8).cpu().numpy()
base_img = topk(p_base, gt_test, 1); base_bra = topk(gt_test, p_base, 1)
print(f"\n  baseline (no refiner) test: Image={base_img*100:.1f}%  Brain={base_bra*100:.1f}%", flush=True)

# === Train refiner ===
print("\n=== training per-voxel refiner ===", flush=True)
refiner = PerVoxelRefiner(n_vox=2792).to(device)
opt = torch.optim.AdamW(refiner.parameters(), lr=5e-3, weight_decay=1e-4)
bs = 32
n_epochs = 100
patience = 20
best_val = -1.0
best_state = None
no_improve = 0
history = []

for epoch in range(n_epochs):
    refiner.train()
    perm_tr = torch.randperm(X_tr.shape[0])
    loss_sum = 0.0
    for i in range(0, X_tr.shape[0], bs):
        idx_b = perm_tr[i:i+bs]
        x_b = X_tr[idx_b]
        gt_b = gt_tr[idx_b]
        x_refined = refiner(x_b)
        cv = fwd_train(model, x_refined, batch=bs)
        loss = cosine_loss(cv, gt_b)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += float(loss) * x_b.shape[0]
    loss_sum /= X_tr.shape[0]

    refiner.eval()
    with torch.no_grad():
        p_val = fwd_eval(model, ss, se, refiner(X_val), batch=8).cpu().numpy()
        p_test = fwd_eval(model, ss, se, refiner(X_test), batch=8).cpu().numpy()
    val_img = topk(p_val, gt_val, 1)
    test_img = topk(p_test, gt_test, 1)
    test_bra = topk(gt_test, p_test, 1)
    history.append({"epoch": epoch, "loss": loss_sum, "val_image": val_img,
                    "test_image": test_img, "test_brain": test_bra})
    if val_img > best_val:
        best_val = val_img
        best_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
    print(f"    epoch {epoch:3d}: loss={loss_sum:.4f}  val Image={val_img*100:5.1f}%  test Image={test_img*100:5.1f}%  Brain={test_bra*100:5.1f}%", flush=True)
    if no_improve >= patience:
        print(f"  early stop at epoch {epoch} (no val improvement for {patience} epochs)", flush=True)
        break

# Load best, eval on test
if best_state is not None:
    refiner.load_state_dict(best_state)
refiner.eval()
with torch.no_grad():
    x_refined_test = refiner(X_test)
    p_test = fwd_eval(model, ss, se, x_refined_test, batch=8).cpu().numpy()
final_img = topk(p_test, gt_test, 1)
final_bra = topk(gt_test, p_test, 1)

print(f"\n========== FAST REFINER RESULT ==========")
print(f"  baseline (fold-0 frozen, no refiner): Image={base_img*100:.1f}%  Brain={base_bra*100:.1f}%")
print(f"  refiner (best-val checkpoint):        Image={final_img*100:.1f}%  Brain={final_bra*100:.1f}%")
print(f"  Δ vs baseline: Image={(final_img-base_img)*100:+.1f}pp  Brain={(final_bra-base_bra)*100:+.1f}pp")
print(f"  paper Fast anchor: Image=36%  Brain=40%")

# Save
torch.save(refiner.state_dict(), LOCAL / "task_2_1_betas/fast_refiner_state.pth")
result = {
    "method": "Fast tier per-voxel refiner trained with cosine-to-CLIP loss",
    "n_train": int(X_tr.shape[0]),
    "n_val": int(X_val.shape[0]),
    "n_test": int(X_test.shape[0]),
    "baseline_image": base_img, "baseline_brain": base_bra,
    "refiner_image": final_img, "refiner_brain": final_bra,
    "best_val_image": best_val,
    "history": history,
}
(LOCAL / "task_2_1_betas/fast_refiner_results.json").write_text(json.dumps(result, indent=2))
print(f"\n  saved fast_refiner_results.json + fast_refiner_state.pth", flush=True)
