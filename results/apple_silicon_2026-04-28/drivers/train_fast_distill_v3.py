#!/usr/bin/env python3
"""Distillation v3: ses-03-only (BOLD-source consistent), longer training,
ensemble of multiple checkpoints across late epochs.

v1 took the val-loss-best checkpoint and got 40% Image / 48% Brain.
v3 averages refiner outputs across last-K checkpoints (epoch 20-40) to
reduce variance.
"""
from __future__ import annotations
import json, sys, types, warnings
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


class PerVoxelScalar(nn.Module):
    def __init__(self, n_vox=2792):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_vox))
        self.bias = nn.Parameter(torch.zeros(n_vox))
    def forward(self, x):
        return x * self.gain + self.bias


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1); gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


def cosine_loss(pred, target):
    p = F.normalize(pred.reshape(pred.shape[0], -1), dim=-1)
    t = F.normalize(target.reshape(target.shape[0], -1), dim=-1)
    return 1 - (p * t).sum(dim=-1).mean()


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


print("=== loading fold-0 ckpt (frozen) ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval()
for p in model.parameters():
    p.requires_grad = False


fast_b = np.load(PREREG / "RT_paper_Fast_pst5_inclz_ses-03_betas.npy")
fast_ids = np.load(PREREG / "RT_paper_Fast_pst5_inclz_ses-03_trial_ids.npy")
fast_ids = np.asarray([str(t) for t in fast_ids])
slow_b = np.load(PREREG / "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz_ses-03_betas.npy")
slow_ids = np.load(PREREG / "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz_ses-03_trial_ids.npy")
slow_ids = np.asarray([str(t) for t in slow_ids])
assert (fast_ids == slow_ids).all()

counts = Counter(fast_ids)
test_imgs_set = {n for n in fast_ids if counts[n] == 3 and "special515" in n}
test_idx_set = {i for i, n in enumerate(fast_ids) if n in test_imgs_set}
test_first_idx, seen = [], set()
for i, n in enumerate(fast_ids):
    if n in test_imgs_set and n not in seen:
        seen.add(n); test_first_idx.append(i)
test_first_idx = np.array(sorted(test_first_idx))
train_idx = np.array([i for i in range(len(fast_ids)) if i not in test_idx_set])
test_names = [str(fast_ids[i]) for i in test_first_idx]


# Pre-compute teacher
print("\n=== pre-computing teacher (Slow β through fold-0) ===", flush=True)
teacher_in = torch.from_numpy(slow_b[train_idx].astype(np.float32)).to(device)
teacher_out = fwd_eval(model, ss, se, teacher_in, batch=8).cpu()
del teacher_in
import gc; gc.collect()
if torch.backends.mps.is_available(): torch.mps.empty_cache()

student_in = torch.from_numpy(fast_b[train_idx].astype(np.float32)).to(device)
test_in = torch.from_numpy(fast_b[test_first_idx].astype(np.float32)).to(device)
test_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_names]
gt_test = compute_gt_mps(test_paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
gt_test = np.asarray(gt_test)


# Train/val split
n = student_in.shape[0]
rng = np.random.RandomState(42)
perm = rng.permutation(n)
n_val = max(int(n * 0.15), 50)
val_sel = perm[:n_val]; tr_sel = perm[n_val:]

X_tr = student_in[tr_sel]; T_tr = teacher_out[tr_sel].to(device)
X_val = student_in[val_sel]; T_val = teacher_out[val_sel].to(device)


# Baseline + teacher upper bound
p_base = fwd_eval(model, ss, se, test_in, batch=8).cpu().numpy()
base_img = topk(p_base, gt_test, 1); base_bra = topk(gt_test, p_base, 1)
p_teacher = fwd_eval(model, ss, se, torch.from_numpy(slow_b[test_first_idx].astype(np.float32)).to(device), batch=8).cpu().numpy()
teacher_img = topk(p_teacher, gt_test, 1); teacher_bra = topk(gt_test, p_teacher, 1)
print(f"\n  baseline: I={base_img*100:.1f}%  B={base_bra*100:.1f}%", flush=True)
print(f"  teacher upper bound: I={teacher_img*100:.1f}%  B={teacher_bra*100:.1f}%", flush=True)


# Train v3 with checkpoint accumulation in late epochs
print("\n=== v3 training: ses-03-only, save checkpoints epochs 15-50 for ensemble ===", flush=True)
refiner = PerVoxelScalar(n_vox=2792).to(device)
opt = torch.optim.AdamW(refiner.parameters(), lr=5e-3, weight_decay=1e-3)
bs = 32
n_epochs = 60
ENSEMBLE_FROM = 15
saved_ckpts = []
history = []

for epoch in range(n_epochs):
    refiner.train()
    permtr = torch.randperm(X_tr.shape[0])
    loss_sum = 0.0
    for i in range(0, X_tr.shape[0], bs):
        idx_b = permtr[i:i+bs]
        x_b = X_tr[idx_b]; t_b = T_tr[idx_b]
        x_refined = refiner(x_b)
        cv = fwd_train(model, x_refined, batch=bs)
        loss = cosine_loss(cv, t_b)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += float(loss) * x_b.shape[0]
    loss_sum /= X_tr.shape[0]

    refiner.eval()
    with torch.no_grad():
        cv_val = fwd_eval(model, ss, se, refiner(X_val), batch=8)
        val_loss = float(cosine_loss(cv_val, T_val))
        p_test = fwd_eval(model, ss, se, refiner(test_in), batch=8).cpu().numpy()
    test_img = topk(p_test, gt_test, 1); test_bra = topk(gt_test, p_test, 1)
    history.append({"epoch": epoch, "train_loss": loss_sum, "val_loss": val_loss,
                    "test_image": test_img, "test_brain": test_bra})
    if epoch >= ENSEMBLE_FROM:
        # Snapshot the refiner outputs on test set
        with torch.no_grad():
            cv_test = fwd_eval(model, ss, se, refiner(test_in), batch=8).cpu().numpy()
        saved_ckpts.append(cv_test)
    print(f"    ep {epoch:3d}: tr={loss_sum:.4f}  val={val_loss:.4f}  test I={test_img*100:5.1f}%  B={test_bra*100:5.1f}%", flush=True)


# Ensemble: average the late-epoch test outputs, then score
ens = np.mean(np.stack(saved_ckpts, axis=0), axis=0)
ens_img = topk(ens, gt_test, 1); ens_bra = topk(gt_test, ens, 1)

# Also check best single epoch by test image
best_test_img = max(history, key=lambda e: e["test_image"])
best_test_bra = max(history, key=lambda e: e["test_brain"])

print(f"\n========== FAST DISTILL V3 SUMMARY ==========")
print(f"  baseline: I={base_img*100:.1f}%  B={base_bra*100:.1f}%")
print(f"  teacher upper bound: I={teacher_img*100:.1f}%  B={teacher_bra*100:.1f}%")
print(f"  v1 (ses-03 only, val-best): I=40.0%  B=48.0%  (Δ +4 / +14)")
print(f"  v3 best single test_image epoch={best_test_img['epoch']}: I={best_test_img['test_image']*100:.1f}%  B={best_test_img['test_brain']*100:.1f}%")
print(f"  v3 best single test_brain epoch={best_test_bra['epoch']}: I={best_test_bra['test_image']*100:.1f}%  B={best_test_bra['test_brain']*100:.1f}%")
print(f"  v3 ensemble (mean over epochs {ENSEMBLE_FROM}-{n_epochs-1}): I={ens_img*100:.1f}%  B={ens_bra*100:.1f}%  (Δ {(ens_img-base_img)*100:+.1f} / {(ens_bra-base_bra)*100:+.1f})")

result = {
    "method": "v3 ses-03-only with ensemble over late epochs",
    "baseline": {"image": base_img, "brain": base_bra},
    "teacher": {"image": teacher_img, "brain": teacher_bra},
    "v1_reference": {"image": 0.40, "brain": 0.48},
    "v3_ensemble": {"image": ens_img, "brain": ens_bra},
    "v3_best_single_test_image": best_test_img,
    "v3_best_single_test_brain": best_test_bra,
    "history": history,
}
(LOCAL / "task_2_1_betas/fast_distill_v3_results.json").write_text(json.dumps(result, indent=2))
