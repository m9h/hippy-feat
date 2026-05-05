#!/usr/bin/env python3
"""Phase B — train a per-voxel projector that combines (β, increment, Lévy area)
into a single scalar input to fold-0, end-to-end with cosine retrieval loss
through the frozen MindEye2 model.

Architecture:
  Per-voxel input (B, 2792, 3): [β, increment, Lévy_area] (each cum-z'd independently)
  Per-voxel projector: (3,) → (1,) — small MLP with one hidden layer (3→8→1) shared across voxels
  Initialized so that initial output ≈ β (i.e., projector starts as identity on dim 0)
  → (B, 2792) input to frozen fold-0
  Loss: 1 - cosine(model.clip_voxels, CLIP-image-tokens) over training trials

Training data: ses-01 + ses-02 trials, per-trial βs from
  RT_paper_Slow_pst20_inclz on respective sessions.
  Wait — we have ses-03 only for the prereg cells. Need to extract for ses-01-02.

Workaround: train on ses-03's NON-test trials (the 481 trials that aren't the
  50 special515 first-rep test set). After fold-0 training on ses-01,
  ses-03 train trials should be unseen by the model — fine for projector
  training as long as we test on the held-out 50.

This is a lighter-weight test than full ses-01-02 retraining but still
informative — if a per-voxel projector can't lift Fast/Slow Image above
the β baseline using ses-03 train trials, it likely can't with more data either.

Test: ses-03 first-rep 50 special515.
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

# Load brain mask + relmask
final_mask = nib.load(LOCAL / "rt3t/data/sub-005_final_mask.nii.gz").get_fdata().flatten() > 0
relmask = np.load(LOCAL / "rt3t/data/sub-005_ses-01_task-C_relmask.npy")


# --- compute log-sig features for ses-03 + a given pst ---
def compute_levy_increment(bold_run, onset_TR, pst):
    end = min(onset_TR + pst + 1, bold_run.shape[0])
    window = bold_run[onset_TR:end]
    if window.shape[0] < 2:
        V = bold_run.shape[1]
        return np.zeros(V, dtype=np.float32), np.zeros(V, dtype=np.float32)
    increment = window[-1] - window[0]
    diffs = window[1:] - window[:-1]
    sums = window[:-1] + window[1:]
    t_mid = (np.arange(window.shape[0] - 1, dtype=np.float32) + 0.5)[:, None]
    levy = 0.5 * (t_mid * diffs - 0.5 * sums).sum(axis=0)
    return increment.astype(np.float32), levy.astype(np.float32)


def features_for_session(session, runs, pst):
    """All trials including blanks, to match β-cell trial indexing."""
    incs, levs, names = [], [], []
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
                inc, lev = compute_levy_increment(bold, onset_TR, pst)
            except Exception:
                V = bold.shape[1]
                inc = np.zeros(V, dtype=np.float32); lev = np.zeros(V, dtype=np.float32)
            incs.append(inc); levs.append(lev); names.append(str(name))
    return np.stack(incs), np.stack(levs), np.asarray(names)


def inclusive_cumz(arr):
    n = arr.shape[0]
    z = np.zeros_like(arr, dtype=np.float32)
    for i in range(n):
        mu = arr[:i+1].mean(axis=0)
        sd = arr[:i+1].std(axis=0) + 1e-6
        z[i] = (arr[i] - mu) / sd
    return z


# Per-voxel projector: shared MLP applied independently across voxels
class PerVoxelProjector(nn.Module):
    def __init__(self, in_features=3, hidden=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Initialize so output ≈ β (input dim 0): set last layer weights small,
        # then add a residual connection to the β channel.
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        self.beta_residual_weight = nn.Parameter(torch.tensor(1.0))   # learnable

    def forward(self, x):
        # x: (B, V, 3) where channel 0 = β, 1 = increment, 2 = Lévy
        B, V, F = x.shape
        delta = self.mlp(x.reshape(-1, F)).reshape(B, V)              # (B, V)
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


def cosine_loss(pred_voxels, gt_clip):
    p = F.normalize(pred_voxels.reshape(pred_voxels.shape[0], -1), dim=-1)
    g = F.normalize(gt_clip.reshape(gt_clip.shape[0], -1), dim=-1)
    sim = (p * g).sum(dim=-1)
    return 1 - sim.mean()


def fwd_model(model, ss, se, x, batch_size=8):
    """Run frozen MindEye2 fold-0 on (N, 2792) input → (N, ss, se)."""
    out = []
    with torch.no_grad(), torch.amp.autocast("mps", dtype=torch.float16):
        b = x.unsqueeze(1)
        for i in range(0, b.shape[0], batch_size):
            vr = model.ridge(b[i:i+batch_size], 0)
            o = model.backbone(vr)
            cv = o[1] if isinstance(o, tuple) else o
            out.append(cv.float())
    return torch.cat(out, 0).reshape(-1, ss, se)


def fwd_model_grad(model, x, batch_size=8):
    """Same forward but gradient-enabled (for training the projector)."""
    out = []
    b = x.unsqueeze(1)
    for i in range(0, b.shape[0], batch_size):
        vr = model.ridge(b[i:i+batch_size], 0)
        o = model.backbone(vr)
        cv = o[1] if isinstance(o, tuple) else o
        out.append(cv)
    return torch.cat(out, 0)


# === Load fold-0 model frozen ===
print("=== loading fold-0 ckpt (frozen) ===", flush=True)
model, ss, se = M.load_mindeye(CKPT, n_voxels=2792, device=device)
model.eval()
for p in model.parameters():
    p.requires_grad = False


configs = [("Fast", 5,  "RT_paper_Fast_pst5_inclz",  0.36),
           ("Slow", 20, "RT_paper_Slow_pst20_inclz", 0.58)]

results_per_tier = {}

for tier, pst, beta_cell, paper_anchor in configs:
    print(f"\n========== Phase B — {tier} (pst={pst}) ==========", flush=True)
    t0 = time.time()
    SESSION = "ses-03"
    RUNS = list(range(1, 12))

    # --- features
    incs, levs, names = features_for_session(SESSION, RUNS, pst)
    incs_z = inclusive_cumz(incs)
    levs_z = inclusive_cumz(levs)

    # --- existing β
    beta_z = np.load(PREREG / f"{beta_cell}_ses-03_betas.npy")
    beta_names = np.load(PREREG / f"{beta_cell}_ses-03_trial_ids.npy")
    beta_names = np.asarray([str(t) for t in beta_names])
    assert len(beta_names) == len(names)

    # --- train/test split
    counts = Counter(names)
    test_imgs_set = {n for n in names if counts[n] == 3 and "special515" in n}

    test_idx, seen = [], set()
    for i, n in enumerate(names):
        if n in test_imgs_set and n not in seen:
            seen.add(n); test_idx.append(i)
    test_idx = np.array(sorted(test_idx))

    # train indices = all non-special515-3rep trials (≈ 543)
    test_idx_set = {i for i, n in enumerate(names) if n in test_imgs_set}
    train_idx = np.array([i for i in range(len(names)) if i not in test_idx_set])
    print(f"  train: {len(train_idx)}, test: {len(test_idx)}", flush=True)

    # --- precompute CLIP-image embeddings for train + test ---
    train_imgs = [str(n) for n in names[train_idx]]
    test_imgs = [str(names[i]) for i in test_idx]

    # CLIP embeddings: use existing GT cache if exists; build train embeddings too
    # train images come from various directories; we need to look them up
    train_paths = []
    for n in train_imgs:
        p = LOCAL / "rt3t/data/all_stimuli" / n
        if p.exists():
            train_paths.append(p)
        else:
            for sub in ("special515", "MST_pairs", "unchosen_nsd_1000_images", "shared1000_notspecial"):
                p2 = LOCAL / "rt3t/data/all_stimuli" / sub / Path(n).name
                if p2.exists():
                    train_paths.append(p2); break
            else:
                train_paths.append(None)

    valid_train = [(i, p) for i, p in enumerate(train_paths) if p is not None]
    train_idx_valid = train_idx[[i for i, _ in valid_train]]
    train_paths_valid = [p for _, p in valid_train]
    print(f"  train images with stimuli on disk: {len(train_idx_valid)}", flush=True)

    test_paths = [LOCAL / "rt3t/data/all_stimuli/special515" / Path(n).name for n in test_imgs]
    gt_train = compute_gt_mps(train_paths_valid, device=device,
                               cache_dir=LOCAL / "task_2_1_betas/gt_cache")
    gt_test = compute_gt_mps(test_paths, device=device,
                              cache_dir=LOCAL / "task_2_1_betas/gt_cache")
    print(f"  gt_train: {gt_train.shape}, gt_test: {gt_test.shape}", flush=True)

    # --- build feature tensors (B, V, 3) ---
    X_train = np.stack([beta_z[train_idx_valid], incs_z[train_idx_valid], levs_z[train_idx_valid]], axis=-1)
    X_test  = np.stack([beta_z[test_idx],         incs_z[test_idx],         levs_z[test_idx]],         axis=-1)
    print(f"  X_train: {X_train.shape}, X_test: {X_test.shape}", flush=True)

    X_train_t = torch.from_numpy(X_train.astype(np.float32)).to(device)
    X_test_t  = torch.from_numpy(X_test.astype(np.float32)).to(device)
    gt_train_t = torch.from_numpy(np.asarray(gt_train).astype(np.float32)).to(device)

    # --- baseline (β alone) score ---
    p_test_baseline = fwd_model(model, ss, se, X_test_t[..., 0], batch_size=8)
    base_img = topk(p_test_baseline.cpu().numpy(), gt_test, 1)
    base_bra = topk(gt_test, p_test_baseline.cpu().numpy(), 1)
    print(f"  baseline (β alone) test:    Image={base_img*100:5.1f}%  Brain={base_bra*100:5.1f}%", flush=True)

    # --- train projector ---
    proj = PerVoxelProjector(in_features=3, hidden=8).to(device)
    opt = torch.optim.AdamW(proj.parameters(), lr=1e-3, weight_decay=1e-4)
    proj.train()
    n_train = X_train_t.shape[0]
    bs = 16
    best_val = -1.0
    history = []
    n_epochs = 30
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        for i in range(0, n_train, bs):
            idx_batch = perm[i:i+bs]
            x_b = X_train_t[idx_batch]                        # (b, V, 3)
            gt_b = gt_train_t[idx_batch]                      # (b, 256, 1664)
            inp = proj(x_b)                                    # (b, V)
            cv = fwd_model_grad(model, inp, batch_size=bs)    # (b, ss, se)
            loss = cosine_loss(cv, gt_b)
            opt.zero_grad(); loss.backward(); opt.step()
            epoch_loss += float(loss) * x_b.shape[0]
        epoch_loss /= n_train

        # val on test
        proj.eval()
        with torch.no_grad():
            inp_test = proj(X_test_t)
            p_test = fwd_model(model, ss, se, inp_test, batch_size=8)
        val_img = topk(p_test.cpu().numpy(), gt_test, 1)
        val_bra = topk(gt_test, p_test.cpu().numpy(), 1)
        proj.train()
        history.append({"epoch": epoch, "train_loss": epoch_loss,
                        "test_image": val_img, "test_brain": val_bra})
        if val_img > best_val:
            best_val = val_img
        print(f"    epoch {epoch:2d}: train_loss={epoch_loss:.4f}  test Image={val_img*100:5.1f}%  Brain={val_bra*100:5.1f}%", flush=True)

    # final
    proj.eval()
    with torch.no_grad():
        inp_test = proj(X_test_t)
        p_test = fwd_model(model, ss, se, inp_test, batch_size=8)
    final_img = topk(p_test.cpu().numpy(), gt_test, 1)
    final_bra = topk(gt_test, p_test.cpu().numpy(), 1)

    results_per_tier[tier] = {
        "baseline_image": base_img, "baseline_brain": base_bra,
        "final_image": final_img, "final_brain": final_bra,
        "best_val_image": best_val, "history": history,
        "n_train": int(n_train), "n_test": int(X_test_t.shape[0]),
        "elapsed_min": (time.time() - t0) / 60,
        "paper_anchor": paper_anchor,
    }
    print(f"\n  {tier}: baseline {base_img*100:.1f}% → final {final_img*100:.1f}% "
          f"(best-val {best_val*100:.1f}%)  Δ vs base = {(final_img-base_img)*100:+.1f}pp  "
          f"elapsed {(time.time()-t0)/60:.1f}m", flush=True)

(LOCAL / "task_2_1_betas/logsig_phase_B.json").write_text(json.dumps(results_per_tier, indent=2))
print("\nsaved logsig_phase_B.json", flush=True)

print("\n========== Phase B SUMMARY ==========")
for tier, r in results_per_tier.items():
    delta = (r["final_image"] - r["baseline_image"]) * 100
    print(f"  {tier}: β={r['baseline_image']*100:.1f}% → projector={r['final_image']*100:.1f}% (Δ {delta:+.1f}pp)  paper {r['paper_anchor']*100:.0f}%")
