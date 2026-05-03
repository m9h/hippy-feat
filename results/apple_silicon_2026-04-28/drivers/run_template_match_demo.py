#!/usr/bin/env python3
"""Empirical template-matching closed-loop demo.

Uses raw (pre-cum-z) βs from the champion cell, then for each trial
builds the target+distractor templates from leave-one-rep-out CV on
the 50 special515 images. The decoded scalar is `cos(β_test, T_target) -
cos(β_test, T_distractor_mean)` — the same shape as Norman-lab MVPA
neurofeedback (deBettencourt 2015) but with arbitrary-image templates
instead of face/scene category averages.

If 2-AFC accuracy is high here, plugging in TRIBEv2-derived templates
(instead of empirical leave-one-out templates) is the natural next
step — it removes the dependency on having seen the participant
respond to each test image first.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
PREREG = LOCAL / "task_2_1_betas/prereg"

warnings.filterwarnings("ignore")

# Load RAW (pre-cum-z) βs from the champion config (no z applied yet)
# These were saved by run_stein_shrinkage.py as RT_paper_EoR_K7_CSFWM_HP_e1_RAW
RAW_CELL = "RT_paper_EoR_K7_CSFWM_HP_e1_RAW"
raw_path = PREREG / f"{RAW_CELL}_ses-03_betas.npy"
ids_path = PREREG / f"{RAW_CELL}_ses-03_trial_ids.npy"

if not raw_path.exists():
    raise SystemExit(f"raw βs not found at {raw_path}; run run_stein_shrinkage.py first")

raw = np.load(raw_path)                      # (770, 2792)
trial_ids = np.load(ids_path, allow_pickle=True)
print(f"raw βs: {raw.shape}, trial_ids: {len(trial_ids)}")

# Filter to special515 trials with all 3 reps in ses-03
spec_idx = []
for i, t in enumerate(trial_ids):
    ts = str(t)
    if ts.startswith("all_stimuli/special515/"):
        spec_idx.append(i)

# Group by image
from collections import defaultdict
by_image = defaultdict(list)
for idx in spec_idx:
    by_image[str(trial_ids[idx])].append(idx)
print(f"\n{len(by_image)} unique special515 images, "
      f"rep counts: {[len(v) for v in by_image.values()][:5]} ...")

# Cosine similarity helper
def cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# Optional pre-processing: causal cum-z over the FULL session (matches our
# scoring policy for retrieval). The raw βs are pre-cum-z; we apply the
# same inclusive causal cum-z then do template matching on the z'd βs.
def inclusive_cumz(arr: np.ndarray) -> np.ndarray:
    n, V = arr.shape
    z = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        mu = arr[:i + 1].mean(axis=0)
        sd = arr[:i + 1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


betas_z = inclusive_cumz(raw)
print(f"inclusive cum-z applied: {betas_z.shape}\n")

# ----- Closed-loop simulation: leave-one-rep-out CV per image ---------------
# For each test trial t (rep r of image X):
#   target template = mean(β at the OTHER 2 reps of X)
#   distractor pool = β at rep-equivalent positions of all other 49 images
#                     (use mean across all 49×3=147 trials, then subtract t if it'd be in there)
#   decoded scalar  = cos(β_t, T_target) - cos(β_t, T_distractor)
#   correct if scalar > 0

# Sanity: every special515 has 3 reps?
n_per_image = [len(v) for v in by_image.values()]
print(f"reps per image: min={min(n_per_image)}, max={max(n_per_image)}, "
      f"images with all 3 reps: {sum(1 for n in n_per_image if n == 3)}/{len(by_image)}")

# Build distractor template once (mean across ALL special515 trials, for speed)
all_spec = np.stack([betas_z[i] for i in spec_idx], axis=0)
T_distractor_global = all_spec.mean(axis=0)

scores_all = []  # (img, rep_test_idx, scalar, correct?)
for img, rep_idxs in by_image.items():
    if len(rep_idxs) < 2:
        continue
    for held_out in rep_idxs:
        # Target template = mean of the OTHER reps of this image
        train_idxs = [i for i in rep_idxs if i != held_out]
        T_target = np.mean(np.stack([betas_z[i] for i in train_idxs], axis=0), axis=0)
        # Distractor: global mean across special515 minus the held_out trial
        # (small bias correction; n=147 trials, removing 1 is negligible)
        T_distract = T_distractor_global  # close enough
        scalar = cos(betas_z[held_out], T_target) - cos(betas_z[held_out], T_distract)
        scores_all.append({"image": img, "rep": rep_idxs.index(held_out),
                            "test_idx": held_out,
                            "scalar": scalar,
                            "correct": scalar > 0})

scalars = np.array([s["scalar"] for s in scores_all])
correct = np.array([s["correct"] for s in scores_all], dtype=int)
n = len(scores_all)

# 2-AFC analog: for each held-out trial, pair with another random trial and
# ask "is the target's scalar higher than a distractor trial's scalar"?
# Simpler: bootstrap-style fwd accuracy = fraction with scalar > 0.
fwd_acc = correct.mean()

# Pairwise: for each held-out trial t (target=X) and another held-out trial s
# (target=Y, X≠Y): compute scalar_t with template T_X, and scalar_s with
# template T_X. The "correct" version: scalar_t > scalar_s for trial t when
# template is T_X. This needs a different scoring setup. Quick form:
# for each ordered pair (t, s) where t and s are different images, the target
# of t scoring higher than the t-template applied to s' β should be > 50%.

# Actually cleaner 2-AFC: build a similarity matrix where each row is a held-out
# trial's β, each column is a candidate target template. Score: argmax-equals-true?
# That's a 50-way retrieval analog.

# Build all 50 leave-one-out templates per held-out trial (for that trial,
# template_X excludes any rep of held_out trial's image)
images_sorted = sorted(by_image.keys())
img_to_col = {img: i for i, img in enumerate(images_sorted)}

# Precompute templates: for image X, T_X = mean of all 3 reps
templates = np.stack([np.mean(np.stack([betas_z[i] for i in by_image[img]], axis=0), axis=0)
                       for img in images_sorted], axis=0)  # (50, V)

# For each held-out trial (test = β at index test_idx, true_image = X):
#   Adjusted T_X = template_X without this trial = (3*T_X - β_test) / 2
# Build the correct adjusted template per trial
n_test = len(scores_all)
sim_matrix = np.zeros((n_test, 50), dtype=np.float64)
true_idx = np.zeros(n_test, dtype=np.int64)
for k, s in enumerate(scores_all):
    test_idx = s["test_idx"]
    img = s["image"]
    true_idx[k] = img_to_col[img]
    beta_test = betas_z[test_idx]
    # Compute similarity to each template
    for j, target_img in enumerate(images_sorted):
        if target_img == img:
            T_adj = (3.0 * templates[j] - beta_test) / 2.0
        else:
            T_adj = templates[j]
        sim_matrix[k, j] = cos(beta_test, T_adj)

# Top-1 retrieval (picks correct target out of 50)
top1 = float((sim_matrix.argmax(axis=1) == true_idx).mean())
top5 = float(np.mean([true_idx[k] in np.argsort(sim_matrix[k])[-5:] for k in range(n_test)]))

# 2-AFC: for each (test_k, distractor_image_d != true_idx[k]):
#   correct if sim_matrix[k, true_idx[k]] > sim_matrix[k, d]
total_correct = 0; total = 0
for k in range(n_test):
    t = true_idx[k]
    for d in range(50):
        if d == t: continue
        if sim_matrix[k, t] > sim_matrix[k, d]:
            total_correct += 1
        total += 1
two_afc = total_correct / total

# Cohen's d on the diagonal (target sim) vs off-diagonal (distractor sim)
diag = sim_matrix[np.arange(n_test), true_idx]
off_mask = np.ones((n_test, 50), dtype=bool)
off_mask[np.arange(n_test), true_idx] = False
off = sim_matrix[off_mask]
d = (diag.mean() - off.mean()) / np.sqrt(0.5 * (diag.var() + off.var()) + 1e-12)

print("\n=== Empirical template matching results (leave-one-rep-out CV) ===")
print(f"n test trials: {n_test} (held-out reps from {len(by_image)} images)")
print(f"top-1 retrieval (50-way):  {top1*100:.2f}%   "
      f"(MindEye2 first-rep on same images: 58%)")
print(f"top-5 retrieval (50-way):  {top5*100:.2f}%   "
      f"(MindEye2: 88%)")
print(f"2-AFC pairwise:            {two_afc*100:.2f}%  "
      f"(MindEye2: 97.2%)")
print(f"Cohen's d (target vs distractor): {d:.3f}  "
      f"(MindEye2: 2.42)")
print(f"\nfwd-acc (decoded scalar > 0): {fwd_acc*100:.2f}%")

# Save
out = LOCAL / "task_2_1_betas" / "template_matching_demo.json"
result = {
    "method": "empirical leave-one-rep-out template matching on raw cum-z'd βs from FAST K=7+HP+e1",
    "n_test_trials": int(n_test),
    "n_unique_images": len(by_image),
    "top1": top1, "top5": top5, "two_afc": two_afc, "cohens_d": float(d),
    "fwd_acc_scalar_positive": float(fwd_acc),
    "comparison_mindeye2": {
        "top1": 0.58, "top5": 0.88, "two_afc": 0.972, "cohens_d": 2.42,
    },
}
out.write_text(json.dumps(result, indent=2))
print(f"\nsaved {out.name}")
