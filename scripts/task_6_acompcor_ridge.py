#!/usr/bin/env python3
"""Task 6 MVE-1: retrain only the sub-005 ridge layer of MindEye 2 with
posterior variance from Variant G as part of the input.

Tests four flavors of the 2792-voxel → 1024-latent adapter, all with the
backbone (BrainNetwork) FROZEN at paper checkpoint weights:

    baseline       Linear(2792,  1024)        input = β_mean
    var_weighted   Linear(2792,  1024)        input = β_mean / sqrt(β_var + ε)
    concat         Linear(5584,  1024)        input = [β_mean, log β_var]
    mlp_concat     MLP(5584, 1024, 1024)      input = [β_mean, log β_var]

Train on Variant G output from sub-005 ses-01 + ses-02.
Targets: OpenCLIP ViT-bigG/14 token embeddings of each trial's stimulus
image (cached via task_6_extract_training_gt.py).

Eval on ses-03 special515 retrieval — same 150 trials, same GT cache,
same metrics as the bake-off.

Output:
    /data/derivatives/rtmindeye_paper/task_6_ridges/{flavor}.pth
    /data/derivatives/rtmindeye_paper/task_6_ridges/{flavor}_summary.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mindeye_retrieval_eval import (
    MindEyeModule,
    RidgeRegression,
    load_mindeye,
    cosine_sim_tokens,
)
from bake_off_per_trial import bootstrap_ci

BETAS_ROOT = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg")
# Old location had some files in parent
OLD_BETAS_ROOT = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas")
GT_CACHE = OLD_BETAS_ROOT / "gt_cache"
STIMULI_ROOT = Path("/data/derivatives/rtmindeye_paper/rt3t/data/all_stimuli")
OUT_DIR = Path("/data/derivatives/rtmindeye_paper/task_6_ridges")
HIDDEN_DIM = 1024
N_VOXELS = 2792
EPS = 1e-6


def voxelwise_zscore_full(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True) + 1e-8
    return (x - mu) / sd


def load_session(session: str, condition: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Check both potential roots
    p_betas = BETAS_ROOT / f"{condition}_{session}_betas.npy"
    if not p_betas.exists():
        p_betas = OLD_BETAS_ROOT / f"{condition}_{session}_betas.npy"
        
    p_vars = BETAS_ROOT / f"{condition}_{session}_vars.npy"
    if not p_vars.exists():
        p_vars = OLD_BETAS_ROOT / f"{condition}_{session}_vars.npy"
        
    p_ids = BETAS_ROOT / f"{condition}_{session}_trial_ids.npy"
    if not p_ids.exists():
        p_ids = OLD_BETAS_ROOT / f"{condition}_{session}_trial_ids.npy"
        
    betas = np.load(p_betas)
    # vars might not exist for non-G variants; fallback to ones
    if p_vars.exists():
        vars_ = np.load(p_vars)
    else:
        vars_ = np.ones_like(betas)
        
    ids = np.load(p_ids, allow_pickle=True)
    return betas, vars_, np.asarray([str(i) for i in ids])


def make_input(flavor: str, betas: np.ndarray, vars_: np.ndarray) -> np.ndarray:
    """Apply input transformation, then voxelwise z-score across all trials.

    For `concat`, z-score each half independently before stacking.
    """
    if flavor == "baseline":
        return voxelwise_zscore_full(betas)
    if flavor == "var_weighted":
        return voxelwise_zscore_full(betas / np.sqrt(vars_ + EPS))
    if flavor in ("concat", "mlp_concat"):
        a = voxelwise_zscore_full(betas)
        b = voxelwise_zscore_full(np.log(vars_ + EPS))
        return np.concatenate([a, b], axis=1)
    raise ValueError(flavor)


def lookup_gt_embedding(image_id: str) -> np.ndarray | None:
    """Return cached (256, 1664) embedding for a trial's stimulus, or None."""
    if "blank" in image_id.lower():
        return None
    p = STIMULI_ROOT / Path(image_id).relative_to("all_stimuli")
    if not p.exists():
        cands = list(STIMULI_ROOT.rglob(Path(image_id).name))
        if not cands:
            return None
        p = cands[0]
    key = GT_CACHE / f"{p.stem}_{hashlib.md5(str(p).encode()).hexdigest()[:8]}.npy"
    if not key.exists():
        return None
    return np.load(key)


def assemble_targets(trial_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_valid, 256, 1664) target tensor + index of valid trials."""
    targets = []
    valid_idx = []
    for i, tid in enumerate(trial_ids):
        emb = lookup_gt_embedding(tid)
        if emb is None:
            continue
        targets.append(emb)
        valid_idx.append(i)
    return np.stack(targets), np.asarray(valid_idx)


class MLPRidge(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def init_ridge(flavor: str, paper_ridge: nn.Linear) -> nn.Module:
    """Initialize a fresh ridge layer/MLP; copy paper weights when shapes align."""
    in_dim = 2 * N_VOXELS if flavor in ("concat", "mlp_concat") else N_VOXELS
    if flavor == "mlp_concat":
        layer = MLPRidge(in_dim, HIDDEN_DIM, HIDDEN_DIM)
        with torch.no_grad():
            # Initialize first half from paper, second half (log-var part) at zero
            layer.net[0].weight[:, :N_VOXELS].copy_(paper_ridge.weight)
            layer.net[0].weight[:, N_VOXELS:].zero_()
            layer.net[0].bias.copy_(paper_ridge.bias)
        return layer

    layer = nn.Linear(in_dim, HIDDEN_DIM)
    if flavor in ("baseline", "var_weighted"):
        with torch.no_grad():
            layer.weight.copy_(paper_ridge.weight)
            layer.bias.copy_(paper_ridge.bias)
    elif flavor == "concat":
        with torch.no_grad():
            # Initialize first half from paper, second half (log-var part) at zero
            layer.weight[:, :N_VOXELS].copy_(paper_ridge.weight)
            layer.weight[:, N_VOXELS:].zero_()
            layer.bias.copy_(paper_ridge.bias)
    return layer


def cosine_distance_loss(pred: torch.Tensor, target: torch.Tensor
                          ) -> torch.Tensor:
    p = pred.reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    p = p / (p.norm(dim=1, keepdim=True) + 1e-8)
    t = t / (t.norm(dim=1, keepdim=True) + 1e-8)
    return 1.0 - (p * t).sum(dim=1).mean()


def train_one_flavor(flavor: str, X_train: np.ndarray, Y_train: np.ndarray,
                     model: nn.Module, device: str,
                     n_epochs: int, batch_size: int, lr: float,
                     loss_kind: str = "cosine"):
    paper_ridge = model.ridge.linears[0]
    new_ridge = init_ridge(flavor, paper_ridge).to(device)
    optim = torch.optim.AdamW(new_ridge.parameters(), lr=lr, weight_decay=1e-4)

    # Freeze everything except new_ridge
    for p in model.parameters():
        p.requires_grad = False

    n = X_train.shape[0]
    Xt = torch.from_numpy(X_train.astype(np.float32))
    Yt = torch.from_numpy(Y_train.astype(np.float32))

    print(f"\n[{flavor}] training ({loss_kind} loss): "
          f"X {X_train.shape} -> Y {Y_train.shape}, "
          f"{n_epochs} epochs, batch={batch_size}, lr={lr}")

    rng = np.random.default_rng(0)
    for epoch in range(n_epochs):
        perm = rng.permutation(n)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = Xt[idx].to(device)
            yb = Yt[idx].to(device)

            latent = new_ridge(xb)                    # (B, 1024)
            latent = latent.unsqueeze(1)              # (B, 1, 1024) seq_len=1
            with torch.amp.autocast("cuda" if device == "cuda" else "cpu"):
                bk_out = model.backbone(latent)
                clip_pred = (bk_out[1] if isinstance(bk_out, tuple)
                             else bk_out).float()

            if loss_kind == "cosine":
                loss = cosine_distance_loss(clip_pred, yb)
            else:
                loss = ((clip_pred - yb) ** 2).mean()

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            ep_loss += float(loss.item())
            n_batches += 1

        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch+1:>3}/{n_epochs}  "
                  f"{loss_kind}_loss={ep_loss/n_batches:.4f}")
    return new_ridge


def evaluate(flavor: str, ridge: nn.Module, X_test: np.ndarray,
             trial_idx: np.ndarray, gt_test: np.ndarray, model: nn.Module,
             device: str) -> dict:
    Xt = torch.from_numpy(X_test.astype(np.float32)).to(device)
    with torch.no_grad():
        latent = ridge(Xt).unsqueeze(1)
        bk_out = model.backbone(latent)
        clip_pred = (bk_out[1] if isinstance(bk_out, tuple) else bk_out).float().cpu().numpy()

    sim = cosine_sim_tokens(clip_pred, gt_test)        # (n_test, n_images)
    topk = np.argsort(-sim, axis=1)
    hits1 = topk[:, 0] == trial_idx
    hits5 = np.array([trial_idx[i] in topk[i, :5] for i in range(len(sim))])
    
    # First-rep metric
    first_rep_idx = []
    seen = set()
    for i, tid in enumerate(trial_idx):
        if tid not in seen:
            first_rep_idx.append(i)
            seen.add(tid)
    first_rep_idx = np.array(first_rep_idx)
    hits1_first = hits1[first_rep_idx]
    
    brain_top = np.argsort(-sim.T, axis=1)[:, 0]
    brain_hits = np.array([trial_idx[brain_top[i]] == i
                           for i in range(gt_test.shape[0])])
    m, lo, hi = bootstrap_ci(hits1, n_resamples=2000)
    m1f, lo1f, hi1f = bootstrap_ci(hits1_first, n_resamples=2000)

    np.save(OUT_DIR / f"{flavor}_hits_top1.npy", hits1)
    return {
        "top1_image_all": {"mean": m, "ci_lo": lo, "ci_hi": hi},
        "top1_image_first_rep": {"mean": m1f, "ci_lo": lo1f, "ci_hi": hi1f},
        "top1_brain": {"mean": float(brain_hits.mean())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="VariantG_glover_fmriprep_acompcor")
    ap.add_argument("--test-condition", default="VariantG_glover_rtm_acompcor")
    ap.add_argument("--flavors", nargs="+",
                    default=["baseline", "concat"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--loss", choices=("cosine", "mse"), default="cosine")
    ap.add_argument("--checkpoint",
                    default="/data/derivatives/rtmindeye_paper/checkpoints/"
                            "data_scaling_exp/concat_glmsingle/checkpoints/"
                            "sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_"
                            "avgrepeats_finalmask_epochs_150.pth")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    # ---- Load training data ----
    print(f"\n[1] Loading training betas + GT targets ({args.condition})")
    train_xs_per_flavor = {f: [] for f in args.flavors}
    train_ys = []
    for ses in ("ses-01", "ses-02"):
        b, v, ids = load_session(ses, condition=args.condition)
        targets, valid_idx = assemble_targets(ids)
        print(f"  {ses}: {len(valid_idx)}/{len(ids)} trials with cached GT")
        b_v = b[valid_idx]; v_v = v[valid_idx]
        for f in args.flavors:
            train_xs_per_flavor[f].append(make_input(f, b_v, v_v))
        train_ys.append(targets)
    Y_train = np.concatenate(train_ys, axis=0)

    # ---- Load test data ----
    print(f"\n[2] Loading ses-03 test set ({args.test_condition}, special515)")
    b03, v03, ids03 = load_session("ses-03", condition=args.test_condition)
    test_mask = np.array([t.startswith("all_stimuli/special515/") for t in ids03])
    b03_t = b03[test_mask]; v03_t = v03[test_mask]
    ids_test = ids03[test_mask]
    unique_images = np.array(sorted(set(ids_test)))
    img_to_idx = {u: i for i, u in enumerate(unique_images)}
    trial_idx = np.array([img_to_idx[t] for t in ids_test])

    gt_test = np.stack([lookup_gt_embedding(u) for u in unique_images])
    print(f"  test trials: {len(ids_test)}  unique images: {len(unique_images)}")

    test_xs_per_flavor = {f: make_input(f, b03_t, v03_t) for f in args.flavors}

    # ---- Load backbone (frozen) ----
    print("\n[3] Loading Fold-10 paper checkpoint")
    model, ss, se = load_mindeye(Path(args.checkpoint),
                                 n_voxels=N_VOXELS, device=device)
    model.eval()

    # ---- Train + eval each flavor ----
    summary = {}
    for flavor in args.flavors:
        X_train = np.concatenate(train_xs_per_flavor[flavor], axis=0)
        print(f"\n[{flavor}] X_train {X_train.shape} Y_train {Y_train.shape}")
        ridge = train_one_flavor(flavor, X_train, Y_train, model, device,
                                 n_epochs=args.epochs, batch_size=args.batch_size,
                                 lr=args.lr, loss_kind=args.loss)
        torch.save({"state_dict": ridge.state_dict(),
                    "flavor": flavor, "input_dim": X_train.shape[1],
                    "epochs": args.epochs, "lr": args.lr},
                   OUT_DIR / f"{flavor}_acompcor.pth")
        m = evaluate(flavor, ridge, test_xs_per_flavor[flavor], trial_idx,
                     gt_test, model, device)
        summary[flavor] = m
        ti = m["top1_image_all"]; tif = m["top1_image_first_rep"]
        print(f"\n[{flavor}] top1 image (all): {ti['mean']:.3f} [{ti['ci_lo']:.3f}, {ti['ci_hi']:.3f}]")
        print(f"[{flavor}] top1 image (1st): {tif['mean']:.3f} [{tif['ci_lo']:.3f}, {tif['ci_hi']:.3f}]")

    out = OUT_DIR / "mve_acompcor_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
