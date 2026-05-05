#!/usr/bin/env python3
"""Cross-latency distillation: Fast student ← streaming-GLM-Slow teacher.

Teacher signal:
  For each ses-03 non-test trial i, the teacher output is:
    fold-0(streaming_GLM_Slow_β_i) → clip_voxels_teacher_i
  This captures what fold-0 produces from a Slow-latency β. The
  streaming-GLM Slow β had access to 20 TRs of post-stim BOLD plus
  joint design across all preceding trials, so its β is much cleaner
  than the Fast pst=5 β.

Student input: Fast pst=5 β (single trial, 7.5s post-stim window)
Student architecture: per-voxel scalar (gain·β + bias), 5584 params
Student forward: refiner → frozen fold-0 → clip_voxels_student
Loss: 1 - cos_sim(clip_voxels_student, clip_voxels_teacher)

Compare to CLIP-cosine refiner (previous null result at 32%).
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


class PerVoxelRefiner(nn.Module):
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


# --- Load fold-0 frozen ---
print("=== loading fold-0 ckpt (frozen) ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval()
for p in model.parameters():
    p.requires_grad = False


# --- Load Fast and Slow βs for ses-03 ---
fast_b = np.load(PREREG / "RT_paper_Fast_pst5_inclz_ses-03_betas.npy")
fast_ids = np.load(PREREG / "RT_paper_Fast_pst5_inclz_ses-03_trial_ids.npy")
fast_ids = np.asarray([str(t) for t in fast_ids])

slow_b = np.load(PREREG / "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz_ses-03_betas.npy")
slow_ids = np.load(PREREG / "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz_ses-03_trial_ids.npy")
slow_ids = np.asarray([str(t) for t in slow_ids])

assert (fast_ids == slow_ids).all(), "Fast and Slow trial_ids must align"

# Test set indices (first occurrence of each 3-rep special515)
counts = Counter(fast_ids)
test_imgs_set = {n for n in fast_ids if counts[n] == 3 and "special515" in n}
test_idx_set = {i for i, n in enumerate(fast_ids) if n in test_imgs_set}
test_first_idx, seen = [], set()
for i, n in enumerate(fast_ids):
    if n in test_imgs_set and n not in seen:
        seen.add(n); test_first_idx.append(i)
test_first_idx = np.array(sorted(test_first_idx))
# Train: everything not in any of the test reps (excludes all 3 reps of test imgs)
train_idx = np.array([i for i in range(len(fast_ids)) if i not in test_idx_set])
print(f"  train: {len(train_idx)} trials, test: {len(test_first_idx)} trials", flush=True)


# --- Pre-compute teacher clip_voxels for ALL training trials (one forward pass each) ---
print("\n=== pre-computing teacher clip_voxels (frozen fold-0 on Slow βs) ===", flush=True)
teacher_input = torch.from_numpy(slow_b[train_idx].astype(np.float32)).to(device)
teacher_out = fwd_eval(model, ss, se, teacher_input, batch=8).cpu()        # (N_train, 256, 1664)
print(f"  teacher_clip_voxels: {teacher_out.shape}", flush=True)

# Student inputs (Fast βs for same trials)
student_in = torch.from_numpy(fast_b[train_idx].astype(np.float32)).to(device)
test_in = torch.from_numpy(fast_b[test_first_idx].astype(np.float32)).to(device)


# --- GT for test scoring (CLIP-image embeddings) ---
test_names = [str(fast_ids[i]) for i in test_first_idx]
test_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_names]
gt_test = compute_gt_mps(test_paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
gt_test = np.asarray(gt_test)
print(f"  gt_test: {gt_test.shape}", flush=True)


# --- Train/val split of training trials ---
n_train_total = len(train_idx)
rng = np.random.RandomState(42)
perm = rng.permutation(n_train_total)
n_val = max(int(n_train_total * 0.15), 50)
val_sel = perm[:n_val]
tr_sel = perm[n_val:]
print(f"  train split: {len(tr_sel)} train / {n_val} val", flush=True)

X_tr = student_in[tr_sel]
T_tr = teacher_out[tr_sel].to(device)
X_val = student_in[val_sel]
T_val = teacher_out[val_sel].to(device)


# --- Baseline: fold-0 frozen on Fast input (no refiner) ---
p_base = fwd_eval(model, ss, se, test_in, batch=8).cpu().numpy()
base_img = topk(p_base, gt_test, 1); base_bra = topk(gt_test, p_base, 1)
print(f"\n  baseline (no refiner) test: Image={base_img*100:.1f}%  Brain={base_bra*100:.1f}%", flush=True)

# Also score teacher on test (sanity: should be Slow s0 = 54%)
teacher_test = torch.from_numpy(slow_b[test_first_idx].astype(np.float32)).to(device)
p_teacher_test = fwd_eval(model, ss, se, teacher_test, batch=8).cpu().numpy()
teacher_img = topk(p_teacher_test, gt_test, 1); teacher_bra = topk(gt_test, p_teacher_test, 1)
print(f"  teacher (Slow β through fold-0): Image={teacher_img*100:.1f}%  Brain={teacher_bra*100:.1f}%  ← upper bound", flush=True)


# --- Train refiner with teacher target ---
print("\n=== training refiner with cross-latency teacher signal ===", flush=True)
refiner = PerVoxelRefiner(n_vox=2792).to(device)
opt = torch.optim.AdamW(refiner.parameters(), lr=5e-3, weight_decay=1e-3)
bs = 32
n_epochs = 80
patience = 15
best_val_loss = float("inf")
best_test_img = base_img
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
        t_b = T_tr[idx_b]
        x_refined = refiner(x_b)
        cv = fwd_train(model, x_refined, batch=bs)
        loss = cosine_loss(cv, t_b)
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += float(loss) * x_b.shape[0]
    loss_sum /= X_tr.shape[0]

    # Validation: cosine to teacher AND test retrieval
    refiner.eval()
    with torch.no_grad():
        x_val_refined = refiner(X_val)
        cv_val = fwd_eval(model, ss, se, x_val_refined, batch=8)
        val_loss = float(cosine_loss(cv_val, T_val))
        x_test_refined = refiner(test_in)
        p_test = fwd_eval(model, ss, se, x_test_refined, batch=8).cpu().numpy()
    test_img = topk(p_test, gt_test, 1); test_bra = topk(gt_test, p_test, 1)

    history.append({"epoch": epoch, "train_loss": loss_sum, "val_loss": val_loss,
                    "test_image": test_img, "test_brain": test_bra})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_test_img = test_img
        best_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
    print(f"    epoch {epoch:3d}: train_loss={loss_sum:.4f}  val_loss={val_loss:.4f}  test Image={test_img*100:5.1f}%  Brain={test_bra*100:5.1f}%", flush=True)
    if no_improve >= patience:
        print(f"  early stop at epoch {epoch}", flush=True)
        break


# Best-val checkpoint final eval
if best_state is not None:
    refiner.load_state_dict(best_state)
refiner.eval()
with torch.no_grad():
    x_test_refined = refiner(test_in)
    p_test = fwd_eval(model, ss, se, x_test_refined, batch=8).cpu().numpy()
final_img = topk(p_test, gt_test, 1)
final_bra = topk(gt_test, p_test, 1)


print(f"\n========== FAST DISTILLATION RESULT ==========")
print(f"  baseline (fold-0 on Fast β):              Image={base_img*100:5.1f}%  Brain={base_bra*100:5.1f}%")
print(f"  teacher (fold-0 on Slow β, upper bound):  Image={teacher_img*100:5.1f}%  Brain={teacher_bra*100:5.1f}%")
print(f"  student (Fast β + refiner, best-val):     Image={final_img*100:5.1f}%  Brain={final_bra*100:5.1f}%")
print(f"  Δ vs baseline: Image={(final_img-base_img)*100:+.1f}pp  Brain={(final_bra-base_bra)*100:+.1f}pp")

(LOCAL / "task_2_1_betas/fast_distill_results.json").write_text(json.dumps({
    "method": "Cross-latency distillation: Fast student ← Slow teacher (streaming GLM Slow + fold-0)",
    "n_train": int(X_tr.shape[0]), "n_val": int(X_val.shape[0]), "n_test": int(test_in.shape[0]),
    "baseline_image": base_img, "baseline_brain": base_bra,
    "teacher_image": teacher_img, "teacher_brain": teacher_bra,
    "student_image": final_img, "student_brain": final_bra,
    "best_val_loss": best_val_loss,
    "history": history,
}, indent=2))
print(f"\n  saved fast_distill_results.json", flush=True)
