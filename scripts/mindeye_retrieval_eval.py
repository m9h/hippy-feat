#!/usr/bin/env python3
"""
MindEye retrieval-only evaluation for Task 2.1.

Inputs:
    - Per-trial betas (N_trials, 2792 voxels) from scripts/task_2_1_factorial.py
    - Matching trial_ids array (image_name strings)
    - MindEye2 condensed-architecture checkpoint (sample=10 / finalmask paper-canonical)
    - 50 special515 test images (for ground-truth OpenCLIP embeddings)

Output:
    Top-1 image retrieval accuracy and top-1 brain retrieval accuracy
    over the 150 special515 test trials (50 images × 3 repeats).

This script SKIPS the diffusion prior + SDXL unCLIP reconstruction branch —
retrieval is computed directly from backbone CLIP embeddings, which is what
we need for Task 2.1's fMRIPrep-vs-GLMsingle gap decomposition. Full
reconstruction path would add the SDXL stack and is not required for the
retrieval metrics reported in the paper's Table 1.

Run via Slurm + apptainer on the NGC PyTorch container:
    sbatch scripts/mindeye_retrieval_eval.sbatch
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

# MindEye2 condensed architecture:
#   BrainNetwork         — imported from mindeye/models/models.py
#   RidgeRegression      — defined inline below (copied from mindeye/scripts/mindeye.py,
#                          fixed to use self.seq_len instead of a module-level global)
#   MindEyeModule        — defined inline below (trivial wrapper)
# We intentionally DO NOT import mindeye.py since importing it triggers
# SDXL / generative_models dependencies we're not carrying.
RTCLOUD_MINDEYE = Path("/data/derivatives/rtmindeye_paper/repos/rtcloud-projects/mindeye")
sys.path.insert(0, str(RTCLOUD_MINDEYE / "models"))
sys.path.insert(0, str(RTCLOUD_MINDEYE / "scripts"))  # for utils_mindeye if models.py needs it


# Stub-patch generative_models.sgm.* — utils_mindeye imports
# `FrozenOpenCLIPImageEmbedder` and `append_dims` from it at module scope,
# but we never actually call them for retrieval (they're used by the
# diffusion prior / unCLIP reconstruction path we skip). Inserting stubs
# into sys.modules lets the import succeed cleanly.
import types as _types

def _install_sgm_stubs():
    if "generative_models" in sys.modules:
        return
    gm = _types.ModuleType("generative_models")
    sgm = _types.ModuleType("generative_models.sgm")
    sgm_util = _types.ModuleType("generative_models.sgm.util")
    sgm_modules = _types.ModuleType("generative_models.sgm.modules")
    sgm_enc = _types.ModuleType("generative_models.sgm.modules.encoders")
    sgm_enc_mods = _types.ModuleType("generative_models.sgm.modules.encoders.modules")

    # Minimal symbol surface that utils_mindeye imports
    sgm_util.append_dims = lambda x, n: x
    class _Stub(torch.nn.Module):
        def __init__(self, *a, **k): super().__init__()
        def forward(self, *a, **k): return None
    sgm_enc_mods.FrozenOpenCLIPImageEmbedder = _Stub
    sgm_enc_mods.FrozenOpenCLIPEmbedder2 = _Stub

    sys.modules["generative_models"] = gm
    sys.modules["generative_models.sgm"] = sgm
    sys.modules["generative_models.sgm.util"] = sgm_util
    sys.modules["generative_models.sgm.modules"] = sgm_modules
    sys.modules["generative_models.sgm.modules.encoders"] = sgm_enc
    sys.modules["generative_models.sgm.modules.encoders.modules"] = sgm_enc_mods
    # Also provide a bare `sgm` alias since mindeye.py imports `import sgm` directly
    sys.modules["sgm"] = sgm


_install_sgm_stubs()


class MindEyeModule(nn.Module):
    """Trivial wrapper — the checkpoint assigns .ridge and .backbone as attrs."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class RidgeRegression(nn.Module):
    """Per-subject linear adapter. Copied from rtcloud-projects mindeye.py:126
    with the seq_len loop bound to self.seq_len instead of a module global."""
    def __init__(self, input_sizes, out_features, seq_len):
        super().__init__()
        self.out_features = out_features
        self.seq_len = seq_len
        self.linears = nn.ModuleList([
            nn.Linear(input_size, out_features) for input_size in input_sizes
        ])

    def forward(self, x, subj_idx):
        out = torch.cat([
            self.linears[subj_idx](x[:, seq]).unsqueeze(1)
            for seq in range(self.seq_len)
        ], dim=1)
        return out


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_condition_betas(condition: str, session: str = "ses-03",
                         betas_root: Path = Path("/data/derivatives/rtmindeye_paper/task_2_1_betas")
                        ) -> tuple[np.ndarray, np.ndarray]:
    betas = np.load(betas_root / f"{condition}_{session}_betas.npy")
    trial_ids = np.load(betas_root / f"{condition}_{session}_trial_ids.npy",
                        allow_pickle=True)
    return betas, trial_ids


def filter_to_special515(betas: np.ndarray, trial_ids: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep only special515 trials (paper's test set: 50 images × 3 repeats)."""
    mask = np.array([str(t).startswith("all_stimuli/special515/") for t in trial_ids])
    filtered_betas = betas[mask]
    filtered_ids = np.asarray([str(t) for t in trial_ids[mask]])
    unique_images = np.array(sorted(set(filtered_ids)))
    print(f"  test trials: {filtered_betas.shape[0]}  unique images: {len(unique_images)}")
    return filtered_betas, filtered_ids, unique_images


# -----------------------------------------------------------------------------
# Ground-truth CLIP embeddings via open_clip
# -----------------------------------------------------------------------------

def compute_ground_truth_embeddings(image_paths: list[Path], device: str = "cuda"
                                   ) -> np.ndarray:
    """Return OpenCLIP ViT-bigG/14 token embeddings, shape (N, 256, 1664).

    Uses Stability sgm's FrozenOpenCLIPImageEmbedder with the EXACT
    config the paper uses (recon_inference.ipynb):
        arch="ViT-bigG-14", version="laion2b_s39b_b160k",
        output_tokens=True, only_tokens=True.

    This runs the full open_clip pipeline (transformer → ln_post → proj)
    and returns patch tokens with CLS dropped — matching what the
    BrainNetwork's clip_proj output is trained against. Earlier
    hand-rolled extraction stopped after the transformer (no ln_post,
    no proj), giving misaligned embeddings → near-chance retrieval.
    """
    from sgm.modules.encoders.modules import FrozenOpenCLIPImageEmbedder
    from PIL import Image
    import torchvision.transforms as T

    embedder = FrozenOpenCLIPImageEmbedder(
        arch="ViT-bigG-14",
        version="laion2b_s39b_b160k",
        output_tokens=True,
        only_tokens=True,
    ).to(device)
    embedder.eval()

    # Standard CLIP-ViT preprocessing (matches what the paper does in
    # recon_inference.ipynb — embedder accepts a tensor batch).
    preprocess = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711)),
    ])

    embeddings = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            inp = preprocess(img).unsqueeze(0).to(device)
            feats = embedder(inp)
            # FrozenOpenCLIPImageEmbedder returns either tokens directly
            # (only_tokens=True) or a tuple (pooled, tokens) (only_tokens=False).
            if isinstance(feats, tuple):
                feats = feats[1]  # tokens
            embeddings.append(feats.float().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


# -----------------------------------------------------------------------------
# MindEye2 forward: ridge → backbone → CLIP prediction
# -----------------------------------------------------------------------------

def load_mindeye(checkpoint_path: Path, n_voxels: int = 2792,
                 device: str = "cuda"):
    """Load MindEye2 condensed model and checkpoint weights.

    Architecture is defined in rtcloud-projects/mindeye/scripts/models.py.
    Condensed config per the paper's Section 2.5:
        - hidden_dim (shared-subject latent): 1024 (not 4096)
        - No low-level submodule
        - OpenCLIP ViT-bigG/14 penultimate: (256, 1664)
    """
    # Only import BrainNetwork from models.py; RidgeRegression + MindEyeModule
    # are defined above to avoid triggering SDXL / generative_models imports.
    from models import BrainNetwork  # noqa: F401

    # Model config must match the checkpoint — values per paper Section 2.5.
    clip_seq_dim, clip_emb_dim = 256, 1664
    hidden_dim = 1024

    model = MindEyeModule()
    model.ridge = RidgeRegression(
        input_sizes=[n_voxels], out_features=hidden_dim, seq_len=1
    )
    model.backbone = BrainNetwork(
        h=hidden_dim, in_dim=hidden_dim,
        seq_len=1, n_blocks=4,
        clip_size=clip_emb_dim, out_dim=clip_emb_dim * clip_seq_dim,
        blurry_recon=False,  # condensed: no low-level submodule
        clip_scale=1.0,
    )
    model.to(device)

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  missing keys: {len(missing)}  (first 3: {missing[:3]})")
    if unexpected:
        print(f"  unexpected keys: {len(unexpected)}  (first 3: {unexpected[:3]})")
    model.eval()
    return model, clip_seq_dim, clip_emb_dim


def predict_clip(model, betas: np.ndarray, device: str = "cuda",
                 clip_seq_dim: int = 256, clip_emb_dim: int = 1664
                ) -> np.ndarray:
    """Forward betas through ridge → backbone → predicted CLIP embedding.

    Returns (N_trials, clip_seq_dim, clip_emb_dim) on CPU.
    """
    out = []
    b = torch.from_numpy(betas.astype(np.float32)).to(device)
    # Shape expected by ridge: (batch, seq_len=1, n_voxels)
    b = b.unsqueeze(1)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        for i in range(b.shape[0]):
            bi = b[i:i+1]
            voxel_ridge = model.ridge(bi, 0)  # subject idx 0 (sub-005)
            backbone_out = model.backbone(voxel_ridge)
            # backbone returns (clip_pred, clip_voxels, ...) per models.py
            clip_voxels = backbone_out[1] if isinstance(backbone_out, tuple) else backbone_out
            out.append(clip_voxels.float().cpu().numpy())
    preds = np.concatenate(out, axis=0)
    return preds.reshape(-1, clip_seq_dim, clip_emb_dim)


# -----------------------------------------------------------------------------
# Retrieval metrics
# -----------------------------------------------------------------------------

def cosine_sim_tokens(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Mean cosine sim over 256 tokens between each (pred, gt) pair.

    pred: (N_trials, 256, 1664)
    gt:   (N_images, 256, 1664)
    returns: (N_trials, N_images)
    """
    # Flatten the token dimension into features, then cosine sim.
    p = pred.reshape(pred.shape[0], -1)
    g = gt.reshape(gt.shape[0], -1)
    p /= (np.linalg.norm(p, axis=1, keepdims=True) + 1e-8)
    g /= (np.linalg.norm(g, axis=1, keepdims=True) + 1e-8)
    return p @ g.T


def top_k_retrieval(sim: np.ndarray, trial_to_image_idx: np.ndarray,
                    k: int = 1) -> float:
    """Top-k retrieval accuracy.

    sim: (N_trials, N_images)  similarity matrix
    trial_to_image_idx: (N_trials,) ground-truth image index per trial
    """
    topk = np.argsort(-sim, axis=1)[:, :k]
    hits = np.array([trial_to_image_idx[i] in topk[i] for i in range(len(sim))])
    return hits.mean()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="A_fmriprep_glover")
    ap.add_argument("--session", default="ses-03")
    ap.add_argument(
        "--checkpoint",
        default="/data/derivatives/rtmindeye_paper/checkpoints/"
                "data_scaling_exp/concat_glmsingle/checkpoints/"
                "sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_"
                "avgrepeats_finalmask_epochs_150.pth",
    )
    ap.add_argument(
        "--stimuli-dir",
        default="/data/derivatives/rtmindeye_paper/rt3t/data/all_stimuli/special515",
    )
    ap.add_argument(
        "--out-json",
        default="/data/derivatives/rtmindeye_paper/task_2_1_betas/retrieval_results.json",
    )
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}  torch {torch.__version__}")

    # 1. Load betas + filter to test set
    print(f"\n[1/5] Loading betas for {args.condition}")
    betas_all, trial_ids_all = load_condition_betas(args.condition, args.session)
    betas_test, ids_test, unique_images = filter_to_special515(betas_all, trial_ids_all)

    # 2. Build image index for retrieval
    image_paths = [Path(args.stimuli_dir) / Path(n).name for n in unique_images]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(f"  WARN: {len(missing)} stimuli missing — first: {missing[0]}")
    img_to_idx = {str(u): i for i, u in enumerate(unique_images)}
    trial_idx = np.array([img_to_idx[t] for t in ids_test])

    # 3. Ground-truth CLIP embeddings
    print(f"\n[2/5] OpenCLIP GT for {len(image_paths)} images")
    t0 = time.time()
    gt_emb = compute_ground_truth_embeddings(image_paths, device=device)
    print(f"  gt shape: {gt_emb.shape}  ({time.time()-t0:.1f}s)")

    # 4. MindEye predictions
    print(f"\n[3/5] Loading MindEye checkpoint")
    model, ss, se = load_mindeye(Path(args.checkpoint), n_voxels=betas_test.shape[1],
                                  device=device)

    print(f"\n[4/5] Forward pass for {betas_test.shape[0]} test trials")
    t0 = time.time()
    pred_emb = predict_clip(model, betas_test, device=device,
                            clip_seq_dim=ss, clip_emb_dim=se)
    print(f"  pred shape: {pred_emb.shape}  ({time.time()-t0:.1f}s)")

    # 5. Retrieval scores
    print(f"\n[5/5] Retrieval")
    sim = cosine_sim_tokens(pred_emb, gt_emb)  # (N_trials, N_images)
    top1_img = top_k_retrieval(sim, trial_idx, k=1)
    top5_img = top_k_retrieval(sim, trial_idx, k=5)
    # Brain retrieval: for each image, find trial whose pred is most similar
    brain_sim = sim.T  # (N_images, N_trials)
    # Ground-truth: for each image, any trial whose trial_idx equals that image
    brain_hits = []
    for img_idx in range(len(unique_images)):
        ranked_trials = np.argsort(-brain_sim[img_idx])
        top1_trial = ranked_trials[0]
        brain_hits.append(trial_idx[top1_trial] == img_idx)
    top1_brain = float(np.mean(brain_hits))

    import json
    result = {
        "condition": args.condition,
        "session": args.session,
        "n_test_trials": int(betas_test.shape[0]),
        "n_test_images": int(len(unique_images)),
        "top1_image_retrieval": float(top1_img),
        "top5_image_retrieval": float(top5_img),
        "top1_brain_retrieval": top1_brain,
    }
    print(json.dumps(result, indent=2))
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {args.out_json}")


if __name__ == "__main__":
    main()
