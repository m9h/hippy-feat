#!/usr/bin/env python3
"""Distillation v2: scale training data + try low-rank student.

Same teacher signal as v1 (streaming GLM Slow → fold-0 → clip_voxels), but:
  - Training data scaled from 527 (ses-03 only) to ~1500 (ses-01+02+03)
  - Two student variants:
      v2a: per-voxel scalar (5584 params), same as v1
      v2b: low-rank refiner 2792→64→2792 (~360k params)
  - Same per-voxel cosine-to-teacher loss, same patience-based early stopping

Mixed BOLD sources during training: ses-03 uses rtmotion, ses-01/02 use fmriprep.
Test set: ses-03 first-rep (rtmotion) — matches the training distribution for that session.
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


# Two architectures
class PerVoxelScalar(nn.Module):
    def __init__(self, n_vox=2792):
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_vox))
        self.bias = nn.Parameter(torch.zeros(n_vox))
    def forward(self, x):
        return x * self.gain + self.bias


class LowRankRefiner(nn.Module):
    def __init__(self, n_vox=2792, rank=64):
        super().__init__()
        self.down = nn.Linear(n_vox, rank, bias=True)
        self.up = nn.Linear(rank, n_vox, bias=True)
        self.scale = nn.Parameter(torch.tensor(0.0))   # init to identity
        nn.init.xavier_normal_(self.down.weight, gain=0.01)
        nn.init.xavier_normal_(self.up.weight, gain=0.01)
        nn.init.zeros_(self.down.bias); nn.init.zeros_(self.up.bias)
    def forward(self, x):
        delta = self.up(F.relu(self.down(x)))
        return x + self.scale * delta


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


# --- Build (Fast, Slow) pairs across all 3 sessions ---
print("\n=== assembling training set ===", flush=True)
def load_pair(fast_cell, slow_cell):
    f = np.load(PREREG / f"{fast_cell}_betas.npy")
    s = np.load(PREREG / f"{slow_cell}_betas.npy")
    fids = np.load(PREREG / f"{fast_cell}_trial_ids.npy")
    sids = np.load(PREREG / f"{slow_cell}_trial_ids.npy")
    fids = np.asarray([str(t) for t in fids])
    sids = np.asarray([str(t) for t in sids])
    assert (fids == sids).all(), f"trial id mismatch {fast_cell} vs {slow_cell}"
    return f, s, fids


sources = [
    ("ses-03",
     "RT_paper_Fast_pst5_inclz_ses-03",
     "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz_ses-03"),
    ("ses-01",
     "RT_paper_Fast_pst5_fmriprep_inclz_ses-01",
     "RT_paper_RLS_AR1_pst20_K7CSFWM_HP_e1_inclz_distill_fmriprep_ses-01"),
    ("ses-02",
     "RT_paper_Fast_pst5_fmriprep_inclz_ses-02",
     "RT_paper_RLS_AR1_pst20_K7CSFWM_HP_e1_inclz_distill_fmriprep_ses-02"),
]

train_fast = []; train_slow = []; train_ses = []
for ses, fast, slow in sources:
    fbp = PREREG / f"{fast}_betas.npy"
    sbp = PREREG / f"{slow}_betas.npy"
    if not fbp.exists() or not sbp.exists():
        print(f"  {ses}: SKIP (missing {fbp.name if not fbp.exists() else sbp.name})", flush=True)
        continue
    f, s, fids = load_pair(fast, slow)
    if ses == "ses-03":
        # Exclude all 3 reps of test images; identify test_first_idx
        counts = Counter(fids)
        test_imgs_set = {n for n in fids if counts[n] == 3 and "special515" in n}
        test_idx_set = {i for i, n in enumerate(fids) if n in test_imgs_set}
        test_first_idx, seen = [], set()
        for i, n in enumerate(fids):
            if n in test_imgs_set and n not in seen:
                seen.add(n); test_first_idx.append(i)
        test_first_idx = np.array(sorted(test_first_idx))
        # Save these for testing later
        if ses == "ses-03":
            test_fast_03 = f[test_first_idx]
            test_slow_03 = s[test_first_idx]
            test_names_03 = [str(fids[i]) for i in test_first_idx]
        train_idx = np.array([i for i in range(len(fids)) if i not in test_idx_set])
    else:
        # All trials are training (no test set on these sessions)
        # But exclude any special515 to avoid leakage
        train_idx = np.array([i for i in range(len(fids)) if "special515" not in fids[i]])
    train_fast.append(f[train_idx])
    train_slow.append(s[train_idx])
    train_ses.extend([ses] * len(train_idx))
    print(f"  {ses}: +{len(train_idx)} train pairs (fast {f.shape}, slow {s.shape})", flush=True)


train_fast = np.concatenate(train_fast, axis=0)
train_slow = np.concatenate(train_slow, axis=0)
print(f"\n  total train: {train_fast.shape}", flush=True)


# Pre-compute teacher outputs (frozen fold-0 on Slow βs)
print("\n=== pre-computing teacher clip_voxels ===", flush=True)
teacher_in = torch.from_numpy(train_slow.astype(np.float32)).to(device)
teacher_out = fwd_eval(model, ss, se, teacher_in, batch=8).cpu()
print(f"  teacher_out: {teacher_out.shape}", flush=True)
del teacher_in
import gc; gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()


# Test set
test_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_names_03]
gt_test = compute_gt_mps(test_paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
gt_test = np.asarray(gt_test)
test_fast_t = torch.from_numpy(test_fast_03.astype(np.float32)).to(device)
test_slow_t = torch.from_numpy(test_slow_03.astype(np.float32)).to(device)


# Train/val split (random 15%)
n_total = train_fast.shape[0]
rng = np.random.RandomState(42)
perm = rng.permutation(n_total)
n_val = max(int(n_total * 0.1), 80)
val_sel = perm[:n_val]; tr_sel = perm[n_val:]
print(f"  split: {len(tr_sel)} train / {n_val} val", flush=True)

X_tr = torch.from_numpy(train_fast[tr_sel].astype(np.float32)).to(device)
T_tr = teacher_out[tr_sel].to(device)
X_val = torch.from_numpy(train_fast[val_sel].astype(np.float32)).to(device)
T_val = teacher_out[val_sel].to(device)


# Baseline + teacher upper bound
p_base = fwd_eval(model, ss, se, test_fast_t, batch=8).cpu().numpy()
base_img = topk(p_base, gt_test, 1); base_bra = topk(gt_test, p_base, 1)
p_teacher = fwd_eval(model, ss, se, test_slow_t, batch=8).cpu().numpy()
teacher_img = topk(p_teacher, gt_test, 1); teacher_bra = topk(gt_test, p_teacher, 1)
print(f"\n  baseline (no refiner): Image={base_img*100:.1f}%  Brain={base_bra*100:.1f}%", flush=True)
print(f"  teacher (Slow β through fold-0): Image={teacher_img*100:.1f}%  Brain={teacher_bra*100:.1f}%", flush=True)


def train_variant(name, model_cls):
    print(f"\n========== {name} ==========", flush=True)
    refiner = model_cls(n_vox=2792).to(device)
    n_params = sum(p.numel() for p in refiner.parameters() if p.requires_grad)
    print(f"  trainable params: {n_params}", flush=True)
    opt = torch.optim.AdamW(refiner.parameters(), lr=5e-3, weight_decay=1e-3)
    bs = 32
    n_epochs = 80
    patience = 15
    best_val_loss = float("inf"); best_state = None; no_improve = 0
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
            p_test = fwd_eval(model, ss, se, refiner(test_fast_t), batch=8).cpu().numpy()
        test_img = topk(p_test, gt_test, 1); test_bra = topk(gt_test, p_test, 1)
        history.append({"epoch": epoch, "train_loss": loss_sum, "val_loss": val_loss,
                        "test_image": test_img, "test_brain": test_bra})
        if val_loss < best_val_loss:
            best_val_loss = val_loss; no_improve = 0
            best_state = {k: v.detach().cpu().clone() for k, v in refiner.state_dict().items()}
        else:
            no_improve += 1
        print(f"    ep {epoch:3d}: tr={loss_sum:.4f}  val={val_loss:.4f}  test I={test_img*100:5.1f}%  B={test_bra*100:5.1f}%", flush=True)
        if no_improve >= patience:
            break
    if best_state is not None:
        refiner.load_state_dict(best_state)
    refiner.eval()
    with torch.no_grad():
        p_test = fwd_eval(model, ss, se, refiner(test_fast_t), batch=8).cpu().numpy()
    fimg = topk(p_test, gt_test, 1); fbra = topk(gt_test, p_test, 1)
    print(f"  {name}: Image={fimg*100:.1f}% (Δ {(fimg-base_img)*100:+.1f})  Brain={fbra*100:.1f}% (Δ {(fbra-base_bra)*100:+.1f})", flush=True)
    return {"name": name, "n_params": n_params,
            "test_image": fimg, "test_brain": fbra,
            "best_val_loss": best_val_loss, "history": history}


results = {}
results["v2a_per_voxel_scalar"] = train_variant("v2a per-voxel scalar (5584 params)", PerVoxelScalar)
results["v2b_low_rank_64"] = train_variant("v2b low-rank refiner rank=64", LowRankRefiner)

results["baseline"] = {"image": base_img, "brain": base_bra}
results["teacher"] = {"image": teacher_img, "brain": teacher_bra}
(LOCAL / "task_2_1_betas/fast_distill_v2_results.json").write_text(json.dumps(results, indent=2))

print("\n========== FAST DISTILL V2 SUMMARY ==========")
print(f"  baseline: I={base_img*100:.1f}%  B={base_bra*100:.1f}%")
print(f"  teacher (upper bound): I={teacher_img*100:.1f}%  B={teacher_bra*100:.1f}%")
for name, r in results.items():
    if name in ("baseline", "teacher"): continue
    img = r["test_image"]*100; bra = r["test_brain"]*100
    di = (r["test_image"]-base_img)*100; db = (r["test_brain"]-base_bra)*100
    print(f"  {name}: Image={img:.1f}% (Δ {di:+.1f})  Brain={bra:.1f}% (Δ {db:+.1f})")
