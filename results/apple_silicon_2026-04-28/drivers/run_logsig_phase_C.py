#!/usr/bin/env python3
"""Phase C — richer per-voxel feature input + longer training.

Same architecture as Phase B (per-voxel MLP through frozen fold-0 with cosine
retrieval loss) but with 9 features per voxel instead of 3:

  channel 0: β (existing AR(1) LSS β, cum-z'd)
  channel 1: increment over post-stim window
  channel 2: Lévy area
  channel 3: window mean
  channel 4: window std
  channel 5: window max
  channel 6: window min
  channel 7: window range (max - min)
  channel 8: linear slope (best-fit linear coefficient over window)

Train for 80 epochs with validation split (last 15% of train trials).
Save model with best validation Image accuracy. Test on ses-03 first-rep.

If Phase B's 3 features were insufficient and the projector needed richer
context to beat baseline, Phase C catches that. If still no improvement,
the technique fundamentally doesn't add value at the scalar-per-voxel
interface.
"""
from __future__ import annotations
import json, sys, types, warnings, time
from collections import Counter
from pathlib import Path

import numpy as np, nibabel as nib, pandas as pd, torch, torch.nn as nn
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
TR = 1.5
EVENTS_DIR = LOCAL / "rt3t/data/events"
MC_DIR = LOCAL / "motion_corrected_resampled"

final_mask = nib.load(LOCAL / "rt3t/data/sub-005_final_mask.nii.gz").get_fdata().flatten() > 0
relmask = np.load(LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy")


def compute_features(bold_run, onset_TR, pst):
    end = min(onset_TR + pst + 1, bold_run.shape[0])
    window = bold_run[onset_TR:end]                 # (W, V)
    V = bold_run.shape[1]
    if window.shape[0] < 2:
        return tuple(np.zeros(V, dtype=np.float32) for _ in range(9))
    increment = (window[-1] - window[0]).astype(np.float32)
    diffs = window[1:] - window[:-1]
    sums = window[:-1] + window[1:]
    t_mid = (np.arange(window.shape[0] - 1, dtype=np.float32) + 0.5)[:, None]
    levy = (0.5 * (t_mid * diffs - 0.5 * sums).sum(axis=0)).astype(np.float32)
    mean = window.mean(axis=0).astype(np.float32)
    std = window.std(axis=0).astype(np.float32)
    mx = window.max(axis=0).astype(np.float32)
    mn = window.min(axis=0).astype(np.float32)
    rng = (mx - mn).astype(np.float32)
    # linear slope (best-fit a in y = a·t + b)
    t = np.arange(window.shape[0], dtype=np.float32)
    t_c = t - t.mean()
    denom = (t_c**2).sum() + 1e-8
    slope = ((t_c[:, None] * (window - window.mean(axis=0, keepdims=True))).sum(axis=0) / denom).astype(np.float32)
    return increment, levy, mean, std, mx, mn, rng, slope


def features_for_session(session, runs, pst):
    """All trials including blanks, to match β-cell trial indexing."""
    feats = [[] for _ in range(8)]
    names = []
    for run in runs:
        pattern = f"{session}_run-{run:02d}_*_mc_boldres.nii.gz"
        vols = sorted(MC_DIR.glob(pattern))
        if not vols:
            continue
        frames = [nib.load(v).get_fdata().flatten()[final_mask][relmask].astype(np.float32) for v in vols]
        bold = np.stack(frames, axis=0)
        ev = pd.read_csv(EVENTS_DIR / f"sub-005_{session}_task-C_run-{run:02d}_events.tsv", sep="\t")
        run_start = float(ev.iloc[0]["onset"])
        for _, row in ev.iterrows():
            name = "nan" if pd.isna(row["image_name"]) else row["image_name"]
            try:
                onset_TR = int(round((float(row["onset"]) - run_start) / TR))
                f = compute_features(bold, onset_TR, pst)
            except Exception:
                V = bold.shape[1]
                f = tuple(np.zeros(V, dtype=np.float32) for _ in range(8))
            for k in range(8):
                feats[k].append(f[k])
            names.append(str(name))
    return tuple(np.stack(fk) for fk in feats), np.asarray(names)


def inclusive_cumz(arr):
    n = arr.shape[0]
    z = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        mu = arr[:i+1].mean(axis=0)
        sd = arr[:i+1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


class PerVoxelProjector(nn.Module):
    def __init__(self, in_features=9, hidden=16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        self.beta_residual_weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        # x: (B, V, F)
        B, V, F = x.shape
        delta = self.mlp(x.reshape(-1, F)).reshape(B, V)
        beta = x[..., 0]
        return self.beta_residual_weight * beta + delta


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


print("=== loading fold-0 ckpt (frozen) ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval()
for p in model.parameters():
    p.requires_grad = False


configs = [("Fast", 5,  "RT_paper_Fast_pst5_inclz",  0.36),
           ("Slow", 20, "RT_paper_Slow_pst20_inclz", 0.58)]

results = {}

for tier, pst, beta_cell, paper_anchor in configs:
    print(f"\n========== Phase C — {tier} (pst={pst}) ==========", flush=True)
    t0 = time.time()
    SESSION = "ses-03"
    RUNS = list(range(1, 12))

    derived, names = features_for_session(SESSION, RUNS, pst)  # 8 derived features
    derived_z = [inclusive_cumz(f) for f in derived]            # (n, V) each

    beta_z = np.load(PREREG / f"{beta_cell}_ses-03_betas.npy")
    beta_names = np.load(PREREG / f"{beta_cell}_ses-03_trial_ids.npy")
    beta_names = np.asarray([str(t) for t in beta_names])
    assert len(beta_names) == len(names)

    # Build (n, V, 9): β + 8 derived
    X_all = np.stack([beta_z, *derived_z], axis=-1)            # (n, V, 9)

    counts = Counter(names)
    test_imgs_set = {n for n in names if counts[n] == 3 and "special515" in n}
    test_idx, seen = [], set()
    for i, n in enumerate(names):
        if n in test_imgs_set and n not in seen:
            seen.add(n); test_idx.append(i)
    test_idx = np.array(sorted(test_idx))
    test_idx_set = {i for i, n in enumerate(names) if n in test_imgs_set}
    train_idx = np.array([i for i in range(len(names)) if i not in test_idx_set])
    print(f"  train: {len(train_idx)}, test: {len(test_idx)}", flush=True)

    # Resolve image paths for training set
    train_paths = []
    for n in names[train_idx]:
        n = str(n)
        for sub in ("special515", "MST_pairs", "unchosen_nsd_1000_images", "shared1000_notspecial"):
            p = LOCAL / "rt3t/data/all_stimuli" / sub / Path(n).name
            if p.exists():
                train_paths.append(p); break
        else:
            p = LOCAL / "rt3t/data/all_stimuli" / n
            train_paths.append(p if p.exists() else None)

    valid = [(i, p) for i, p in enumerate(train_paths) if p is not None]
    train_idx_v = train_idx[[i for i, _ in valid]]
    train_paths_v = [p for _, p in valid]
    print(f"  train images on disk: {len(train_idx_v)}", flush=True)
    test_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(str(names[i])).name for i in test_idx]
    gt_train = compute_gt_mps(train_paths_v, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")
    gt_test = compute_gt_mps(test_paths, device=device, cache_dir=LOCAL/"task_2_1_betas/gt_cache")

    # Train/val split (last 15% of train)
    n_train = len(train_idx_v)
    n_val = max(int(n_train * 0.15), 10)
    rng = np.random.RandomState(42)
    perm = rng.permutation(n_train)
    val_sel = perm[:n_val]
    tr_sel = perm[n_val:]

    X_train = X_all[train_idx_v[tr_sel]]
    X_val   = X_all[train_idx_v[val_sel]]
    X_test  = X_all[test_idx]
    gt_tr   = np.asarray(gt_train)[tr_sel]
    gt_val  = np.asarray(gt_train)[val_sel]

    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    X_val_t   = torch.from_numpy(X_val.astype(np.float32)).to(device)
    X_test_t  = torch.from_numpy(X_test.astype(np.float32)).to(device)
    gt_tr_t   = torch.from_numpy(gt_tr.astype(np.float32)).to(device)

    # Baseline (β alone)
    p_base = fwd_eval(model, ss, se, X_test_t[..., 0], batch=8).cpu().numpy()
    base_img = topk(p_base, gt_test, 1); base_bra = topk(gt_test, p_base, 1)
    print(f"  baseline (β alone): Image={base_img*100:5.1f}%  Brain={base_bra*100:5.1f}%", flush=True)

    proj = PerVoxelProjector(in_features=9, hidden=16).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    n_tr = X_train_t.shape[0]
    bs = 16
    history = []
    best_val_img = -1.0
    best_state = None
    n_epochs = 80
    patience = 15
    no_improve = 0
    for epoch in range(n_epochs):
        proj.train()
        perm_tr = torch.randperm(n_tr)
        loss_sum = 0.0
        for i in range(0, n_tr, bs):
            idx_b = perm_tr[i:i+bs]
            x_b = X_train_t[idx_b]
            gt_b = gt_tr_t[idx_b]
            inp = proj(x_b)
            cv = fwd_train(model, inp, batch=bs)
            loss = cosine_loss(cv, gt_b)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += float(loss) * x_b.shape[0]
        loss_sum /= n_tr

        proj.eval()
        with torch.no_grad():
            inp_val = proj(X_val_t)
            p_val = fwd_eval(model, ss, se, inp_val, batch=8).cpu().numpy()
            inp_test = proj(X_test_t)
            p_test = fwd_eval(model, ss, se, inp_test, batch=8).cpu().numpy()
        val_img = topk(p_val, gt_val, 1); val_bra = topk(gt_val, p_val, 1)
        test_img = topk(p_test, gt_test, 1); test_bra = topk(gt_test, p_test, 1)
        history.append({"epoch": epoch, "train_loss": loss_sum,
                        "val_image": val_img, "val_brain": val_bra,
                        "test_image": test_img, "test_brain": test_bra})
        if val_img > best_val_img:
            best_val_img = val_img
            best_state = {k: v.detach().cpu().clone() for k, v in proj.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        print(f"    epoch {epoch:2d}: loss={loss_sum:.4f}  val Image={val_img*100:5.1f}%  test Image={test_img*100:5.1f}%", flush=True)
        if no_improve >= patience:
            print(f"  early stop at epoch {epoch} (no val improvement for {patience} epochs)", flush=True)
            break

    # Load best, eval on test
    if best_state is not None:
        proj.load_state_dict(best_state)
    proj.eval()
    with torch.no_grad():
        inp_test = proj(X_test_t)
        p_test = fwd_eval(model, ss, se, inp_test, batch=8).cpu().numpy()
    final_img = topk(p_test, gt_test, 1); final_bra = topk(gt_test, p_test, 1)

    results[tier] = {
        "baseline_image": base_img, "baseline_brain": base_bra,
        "final_image": final_img, "final_brain": final_bra,
        "best_val_image": best_val_img,
        "n_features": 9,
        "n_train": int(n_tr), "n_val": int(X_val_t.shape[0]), "n_test": int(X_test_t.shape[0]),
        "elapsed_min": (time.time() - t0) / 60,
        "paper_anchor": paper_anchor,
        "history": history,
    }
    delta = (final_img - base_img) * 100
    print(f"\n  {tier}: baseline {base_img*100:.1f}% → final {final_img*100:.1f}% (Δ {delta:+.1f}pp)  paper {paper_anchor*100:.0f}%  elapsed {(time.time()-t0)/60:.1f}m", flush=True)

(LOCAL / "task_2_1_betas/logsig_phase_C.json").write_text(json.dumps(results, indent=2))
print("\nsaved logsig_phase_C.json", flush=True)

print("\n========== Phase C SUMMARY ==========")
for tier, r in results.items():
    delta = (r["final_image"] - r["baseline_image"]) * 100
    print(f"  {tier}: β={r['baseline_image']*100:.1f}% → projector={r['final_image']*100:.1f}% (Δ {delta:+.1f}pp)  paper {r['paper_anchor']*100:.0f}%")
