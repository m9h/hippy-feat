#!/usr/bin/env python3
"""Cross-latency distillation: Fast student ← streaming-GLM-Slow teacher.

DGX port of `results/apple_silicon_2026-04-28/drivers/train_fast_distill_from_slow.py`.

Differences from the Mac source:
  - Paths rebound to /data/derivatives/rtmindeye_paper/...
  - Device → cuda; autocast device → "cuda"
  - Fast student β = `Paper_RT_actual_delay0` (Rishab's pre-saved delay=0 LSS)
    + session-level z-score (matches mindeye.py:770-784 at-session-end running
    mean/std semantics, what apple-silicon labels `_inclz`).
  - Slow teacher β = existing `RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz`
    cell (job 1090 on 2026-05-04).
  - GT CLIP embeddings: load from existing 515-entry cache at
    /data/derivatives/rtmindeye_paper/task_2_1_betas/gt_cache/.

Teacher signal:
  For each ses-03 non-test trial i, the teacher output is
    fold-0(streaming-Slow β_i) → clip_voxels_teacher_i
  The streaming-Slow β has 20 TRs of post-stim BOLD plus joint design across
  preceding trials, so it's a much cleaner β than the Fast pst=5 single-trial.

Student input:  Fast pst=5 β (single trial, 7.5 s post-stim window).
Student arch :  per-voxel scalar (gain·β + bias), 5584 trainable params,
                frozen fold-0 downstream.
Loss:           1 − cos_sim(clip_voxels_student, clip_voxels_teacher)

Apple-silicon ses-03 result: 36 → 40% Image (+4 pp), 34 → 48% Brain (+14 pp).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
LOCAL = Path("/data/derivatives/rtmindeye_paper")
PREREG = LOCAL / "task_2_1_betas" / "prereg"
GT_CACHE = LOCAL / "task_2_1_betas" / "gt_cache"
STIMULI = LOCAL / "rt3t" / "data" / "all_stimuli" / "special515"

# rt_mindEye2 source location (mirrors what mindeye_retrieval_eval expects)
RT_MINDEYE = LOCAL / "repos" / "rtcloud-projects" / "mindeye" / "rt_mindEye2" / "src"
sys.path.insert(0, str(RT_MINDEYE))
sys.path.insert(0, str(Path(__file__).resolve().parent))     # for mindeye_retrieval_eval

import mindeye_retrieval_eval as M                            # noqa: E402

warnings.filterwarnings("ignore")

CKPT = Path(
    "/data/3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth"
)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"=== device={device}  torch={torch.__version__} ===", flush=True)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
class PerVoxelRefiner(nn.Module):
    def __init__(self, n_vox: int = 2792) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.ones(n_vox))
        self.bias = nn.Parameter(torch.zeros(n_vox))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gain + self.bias


def topk(p: np.ndarray, g: np.ndarray, k: int = 1) -> float:
    pf = p.reshape(p.shape[0], -1)
    gf = g.reshape(g.shape[0], -1)
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = F.normalize(pred.reshape(pred.shape[0], -1), dim=-1)
    t = F.normalize(target.reshape(target.shape[0], -1), dim=-1)
    return 1 - (p * t).sum(dim=-1).mean()


def fwd_eval(model, ss: int, se: int, x: torch.Tensor, batch: int = 8) -> torch.Tensor:
    out = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        b = x.unsqueeze(1)
        for i in range(0, b.shape[0], batch):
            vr = model.ridge(b[i:i + batch], 0)
            o = model.backbone(vr)
            cv = o[1] if isinstance(o, tuple) else o
            out.append(cv.float())
    return torch.cat(out, 0).reshape(-1, ss, se)


def fwd_train(model, x: torch.Tensor, batch: int = 8) -> torch.Tensor:
    out = []
    b = x.unsqueeze(1)
    for i in range(0, b.shape[0], batch):
        vr = model.ridge(b[i:i + batch], 0)
        o = model.backbone(vr)
        cv = o[1] if isinstance(o, tuple) else o
        out.append(cv)
    return torch.cat(out, 0)


def session_zscore(arr: np.ndarray) -> np.ndarray:
    """Session-level z-score — matches mindeye.py:770-784 at-session-end
    running mean/std semantics (what apple-silicon labels `_inclz`)."""
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True) + 1e-8
    return ((arr - mu) / sd).astype(np.float32)


def load_gt_from_cache(image_paths: list[Path]) -> np.ndarray:
    """Load CLIP-image-token embeddings for `image_paths` from the existing
    gt_cache. Falls back to error if any are missing — apple-silicon already
    populated all 515 special515 entries on this rig."""
    keys = [GT_CACHE / f"{p.stem}_{hashlib.md5(str(p).encode()).hexdigest()[:8]}.npy"
             for p in image_paths]
    missing = [k for k in keys if not k.exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} GT cache entries missing — first: {missing[0]}.\n"
            f"Run any retrieval driver first to populate the cache."
        )
    return np.stack([np.load(k) for k in keys])


# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
print("\n=== loading fold-0 ckpt (frozen) ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval()
for p in model.parameters():
    p.requires_grad = False
print(f"  ss={ss}  se={se}", flush=True)


# ---------------------------------------------------------------------------
# Load βs
# ---------------------------------------------------------------------------
# Fast student input — Rishab's pre-saved Paper_RT_actual_delay0 (LSS at
# pst=5), then session-z to match _inclz semantics.
fast_raw = np.load(PREREG / "Paper_RT_actual_delay0_ses-03_betas.npy")
fast_ids = np.load(PREREG / "Paper_RT_actual_delay0_ses-03_trial_ids.npy",
                    allow_pickle=True)
fast_ids = np.asarray([str(t) for t in fast_ids])
fast_b = session_zscore(fast_raw)

# Slow teacher input — load `_raw` and apply session_zscore inline.
# The `_inclz` files from job 1090 are corrupted (job 1100 diagnostic:
# fold-0 retrieval drops to 4%/2% on _inclz vs 50%/52% on _raw+session_z).
slow_raw = np.load(PREREG / "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw_ses-03_betas.npy")
slow_ids = np.load(PREREG / "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw_ses-03_trial_ids.npy",
                    allow_pickle=True)
slow_ids = np.asarray([str(t) for t in slow_ids])
slow_b = session_zscore(slow_raw)

assert fast_b.shape == slow_b.shape, f"shape mismatch: {fast_b.shape} vs {slow_b.shape}"
assert (fast_ids == slow_ids).all(), "Fast and Slow trial_ids must align"
print(f"  fast_b: {fast_b.shape}  slow_b: {slow_b.shape}  trials: {len(fast_ids)}",
      flush=True)


# ---------------------------------------------------------------------------
# Train / val / test split (matches Mac driver lines 132-143)
# ---------------------------------------------------------------------------
counts = Counter(fast_ids)
test_imgs_set = {n for n in fast_ids if counts[n] == 3 and "special515" in n}
test_idx_set = {i for i, n in enumerate(fast_ids) if n in test_imgs_set}
test_first_idx, seen = [], set()
for i, n in enumerate(fast_ids):
    if n in test_imgs_set and n not in seen:
        seen.add(n)
        test_first_idx.append(i)
test_first_idx = np.array(sorted(test_first_idx))
train_idx = np.array([i for i in range(len(fast_ids)) if i not in test_idx_set])
print(f"  train: {len(train_idx)} trials  test: {len(test_first_idx)} trials",
      flush=True)


# ---------------------------------------------------------------------------
# Pre-compute teacher clip_voxels (one frozen forward pass per trial)
# ---------------------------------------------------------------------------
print("\n=== pre-computing teacher clip_voxels (frozen fold-0 on Slow βs) ===",
      flush=True)
teacher_input = torch.from_numpy(slow_b[train_idx].astype(np.float32)).to(device)
teacher_out = fwd_eval(model, ss, se, teacher_input, batch=8).cpu()
print(f"  teacher_clip_voxels: {teacher_out.shape}", flush=True)

student_in = torch.from_numpy(fast_b[train_idx].astype(np.float32)).to(device)
test_in = torch.from_numpy(fast_b[test_first_idx].astype(np.float32)).to(device)


# ---------------------------------------------------------------------------
# Test-set GT
# ---------------------------------------------------------------------------
test_names = [str(fast_ids[i]) for i in test_first_idx]
test_paths = [STIMULI / Path(n).name for n in test_names]
gt_test = load_gt_from_cache(test_paths)
print(f"  gt_test: {gt_test.shape}", flush=True)


# ---------------------------------------------------------------------------
# Train/val split
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Baseline + teacher upper bound on test set
# ---------------------------------------------------------------------------
p_base = fwd_eval(model, ss, se, test_in, batch=8).cpu().numpy()
base_img = topk(p_base, gt_test, 1)
base_bra = topk(gt_test, p_base, 1)
print(f"\n  baseline (no refiner) test: Image={base_img * 100:.1f}%  "
      f"Brain={base_bra * 100:.1f}%", flush=True)

teacher_test = torch.from_numpy(slow_b[test_first_idx].astype(np.float32)).to(device)
p_teacher_test = fwd_eval(model, ss, se, teacher_test, batch=8).cpu().numpy()
teacher_img = topk(p_teacher_test, gt_test, 1)
teacher_bra = topk(gt_test, p_teacher_test, 1)
print(f"  teacher (Slow β through fold-0): Image={teacher_img * 100:.1f}%  "
      f"Brain={teacher_bra * 100:.1f}%  ← upper bound", flush=True)


# ---------------------------------------------------------------------------
# Train refiner with cross-latency teacher signal
# ---------------------------------------------------------------------------
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

t0 = time.time()
for epoch in range(n_epochs):
    refiner.train()
    perm_tr = torch.randperm(X_tr.shape[0])
    loss_sum = 0.0
    for i in range(0, X_tr.shape[0], bs):
        idx_b = perm_tr[i:i + bs]
        x_b = X_tr[idx_b]
        t_b = T_tr[idx_b]
        x_refined = refiner(x_b)
        cv = fwd_train(model, x_refined, batch=bs)
        loss = cosine_loss(cv, t_b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_sum += float(loss) * x_b.shape[0]
    loss_sum /= X_tr.shape[0]

    refiner.eval()
    with torch.no_grad():
        x_val_refined = refiner(X_val)
        cv_val = fwd_eval(model, ss, se, x_val_refined, batch=8)
        val_loss = float(cosine_loss(cv_val, T_val))
        x_test_refined = refiner(test_in)
        p_test = fwd_eval(model, ss, se, x_test_refined, batch=8).cpu().numpy()
    test_img = topk(p_test, gt_test, 1)
    test_bra = topk(gt_test, p_test, 1)

    history.append({"epoch": epoch, "train_loss": loss_sum, "val_loss": val_loss,
                    "test_image": test_img, "test_brain": test_bra})
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_test_img = test_img
        best_state = {k: v.detach().cpu().clone()
                       for k, v in refiner.state_dict().items()}
        no_improve = 0
    else:
        no_improve += 1
    print(f"    epoch {epoch:3d}: train_loss={loss_sum:.4f}  "
          f"val_loss={val_loss:.4f}  test Image={test_img * 100:5.1f}%  "
          f"Brain={test_bra * 100:5.1f}%", flush=True)
    if no_improve >= patience:
        print(f"  early stop at epoch {epoch}", flush=True)
        break
print(f"  total training time: {(time.time() - t0):.1f}s", flush=True)


# ---------------------------------------------------------------------------
# Best-val final eval
# ---------------------------------------------------------------------------
if best_state is not None:
    refiner.load_state_dict(best_state)
refiner.eval()
with torch.no_grad():
    x_test_refined = refiner(test_in)
    p_test = fwd_eval(model, ss, se, x_test_refined, batch=8).cpu().numpy()
final_img = topk(p_test, gt_test, 1)
final_bra = topk(gt_test, p_test, 1)


# ---------------------------------------------------------------------------
# Report + save
# ---------------------------------------------------------------------------
print(f"\n========== FAST DISTILLATION RESULT (DGX) ==========")
print(f"  baseline (fold-0 on Fast β):              "
      f"Image={base_img * 100:5.1f}%  Brain={base_bra * 100:5.1f}%")
print(f"  teacher (fold-0 on Slow β, upper bound):  "
      f"Image={teacher_img * 100:5.1f}%  Brain={teacher_bra * 100:5.1f}%")
print(f"  student (Fast β + refiner, best-val):     "
      f"Image={final_img * 100:5.1f}%  Brain={final_bra * 100:5.1f}%")
print(f"  Δ vs baseline: Image={(final_img - base_img) * 100:+.1f}pp  "
      f"Brain={(final_bra - base_bra) * 100:+.1f}pp")

out_path = LOCAL / "task_2_1_betas" / "fast_distill_results_dgx.json"
out_path.write_text(json.dumps({
    "method": "Cross-latency distillation (DGX port): Fast student ← Slow teacher (streaming GLM Slow + fold-0)",
    "device": device,
    "ckpt": str(CKPT),
    "fast_source": "Paper_RT_actual_delay0 (Rishab pre-saved) + session_zscore",
    "slow_source": "RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw (job 1090) + session_zscore (job 1100 diagnostic confirmed _inclz files broken)",
    "n_train": int(X_tr.shape[0]), "n_val": int(X_val.shape[0]),
    "n_test": int(test_in.shape[0]),
    "baseline_image": base_img, "baseline_brain": base_bra,
    "teacher_image": teacher_img, "teacher_brain": teacher_bra,
    "student_image": final_img, "student_brain": final_bra,
    "best_val_loss": best_val_loss,
    "history": history,
}, indent=2))
print(f"\n  saved {out_path}", flush=True)
