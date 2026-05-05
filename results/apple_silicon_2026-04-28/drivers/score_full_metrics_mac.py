#!/usr/bin/env python3
"""Mac-MPS port of scripts/score_full_metrics.py.

Differences from the DGX version:
  - Path setup rebound to ~/Workspace local layout
  - DiffusionEngine instantiated from sgm config + ckpt (no pickle dependency)
  - clip_img_embedder instantiated from open_clip (no pickle dependency)
  - xformers monkey-patch falls back to torch SDPA (xformers has no MPS backend)
  - Device defaults to mps; per-batch sizes lowered for memory
  - Optional --quick flag for n-cells=1, n-seeds=1 validation runs

Output: JSON with all 10 metrics per cell × per seed.
"""
from __future__ import annotations
import argparse, json, sys, time, os
import pickle
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torchvision import transforms

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
HF_HOME = Path("/Users/mhough/Workspace/hippy-feat")
PRE = LOCAL / "task_2_1_betas/prereg"
RT3T = LOCAL / "rt3t/data"
GT_CACHE = LOCAL / "task_2_1_betas/gt_cache"
STIMULI = RT3T / "all_stimuli/special515"
RT_ALL_DATA = LOCAL / "rt_all_data"
RTCLOUD_MINDEYE = Path("/Users/mhough/Workspace/rtcloud-projects-mindeye")
RT_MINDEYE_SRC = Path("/Users/mhough/Workspace/rt_mindEye2/src")
GENMODELS_PATH = RT_MINDEYE_SRC / "generative_models"

# --- xformers monkey-patch with torch SDPA (MPS has no xformers) ---
import torch.nn.functional as _F
import types as _types
import importlib.machinery as _im

def _make_fake_module(name):
    m = _types.ModuleType(name)
    m.__spec__ = _im.ModuleSpec(name, None)
    m.__spec__.has_location = False
    m.__version__ = "0.0.30"
    return m

_xformers_module = _make_fake_module("xformers")
_xformers_ops = _make_fake_module("xformers.ops")
def _sdpa_attention(q, k, v, attn_bias=None, p=0.0, scale=None, op=None):
    """Translate xformers (B, M, H, K) → SDPA (B, H, M, K) and back."""
    if q.dim() == 4:
        q4 = q.transpose(1, 2); k4 = k.transpose(1, 2); v4 = v.transpose(1, 2)
        out = _F.scaled_dot_product_attention(q4, k4, v4, attn_mask=attn_bias,
                                               dropout_p=p, scale=scale)
        return out.transpose(1, 2).contiguous()
    return _F.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias,
                                            dropout_p=p, scale=scale)
_xformers_ops.memory_efficient_attention = _sdpa_attention
_xformers_ops.MemoryEfficientAttentionFlashAttentionOp = None
_xformers_module.ops = _xformers_ops
sys.modules["xformers"] = _xformers_module
sys.modules["xformers.ops"] = _xformers_ops

# --- Path setup so sgm + utils_mindeye + models import correctly ---
sys.path.insert(0, str(RTCLOUD_MINDEYE))
sys.path.insert(0, str(RT_MINDEYE_SRC))
sys.path.insert(0, str(GENMODELS_PATH))
sys.path.insert(0, str(HF_HOME / "scripts"))

# --- diffusers Decoder moved in 0.37 from .models.vae to .models.autoencoders.vae ---
import diffusers.models.autoencoders.vae as _vae_new
import diffusers.models as _diffusers_models
_vae_legacy = _types.ModuleType("diffusers.models.vae")
_vae_legacy.Decoder = _vae_new.Decoder
_vae_legacy.__spec__ = _im.ModuleSpec("diffusers.models.vae", None)
sys.modules["diffusers.models.vae"] = _vae_legacy
_diffusers_models.vae = _vae_legacy

# --- sgm.models.autoencoder.AutoencoderKLInferenceWrapper not in local sgm; shim it ---
import sgm.models.autoencoder as _ae
if not hasattr(_ae, "AutoencoderKLInferenceWrapper"):
    class AutoencoderKLInferenceWrapper(_ae.AutoencoderKL):
        def encode(self, x):
            posterior = super().encode(x)
            return posterior.sample()
    _ae.AutoencoderKLInferenceWrapper = AutoencoderKLInferenceWrapper

import utils_mindeye  # noqa: E402

# utils_mindeye sets `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')` at import.
# Override to MPS when available so unclip_recon's hardcoded .to(device) calls land on the right device.
if torch.backends.mps.is_available():
    utils_mindeye.device = torch.device("mps")

warnings.filterwarnings("ignore")

HIDDEN_DIM = 1024
N_BLOCKS = 4
SEQ_LEN = 1
CLIP_SEQ_DIM = 256
CLIP_EMB_DIM = 1664
IMSIZE = 224


class MindEyeModule(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class RidgeRegression(nn.Module):
    def __init__(self, input_sizes, out_features, seq_len):
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(d, out_features) for d in input_sizes])
    def forward(self, x, subj_idx):
        return torch.cat(
            [self.linears[subj_idx](x[:, seq]).unsqueeze(1) for seq in range(SEQ_LEN)], dim=1)


def build_mindeye(num_voxels, device):
    from models import BrainNetwork, PriorNetwork, BrainDiffusionPrior
    model = MindEyeModule()
    model.ridge = RidgeRegression([num_voxels], out_features=HIDDEN_DIM, seq_len=SEQ_LEN)
    model.backbone = BrainNetwork(
        h=HIDDEN_DIM, in_dim=HIDDEN_DIM, seq_len=SEQ_LEN,
        clip_size=CLIP_EMB_DIM, out_dim=CLIP_EMB_DIM * CLIP_SEQ_DIM,
        n_blocks=N_BLOCKS, blurry_recon=False)
    out_dim = CLIP_EMB_DIM
    depth, dim_head = 6, 52
    heads = CLIP_EMB_DIM // dim_head
    timesteps = 100
    prior_network = PriorNetwork(
        dim=out_dim, depth=depth, dim_head=dim_head, heads=heads,
        causal=False, num_tokens=CLIP_SEQ_DIM, learned_query_mode="pos_emb")
    model.diffusion_prior = BrainDiffusionPrior(
        net=prior_network, image_embed_dim=out_dim,
        condition_on_text_encodings=False, timesteps=timesteps,
        cond_drop_prob=0.2, image_embed_scale=None)
    model.to(device)
    return model


def load_diffusion_engine(device):
    """Instantiate from config + ckpt (no pickle dependency)."""
    from omegaconf import OmegaConf
    from sgm.util import instantiate_from_config

    cfg = OmegaConf.load(GENMODELS_PATH / "configs/unclip6.yaml")
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["model"]["params"]["sampler_config"]["params"]["num_steps"] = 38
    cfg["model"]["params"]["sampler_config"]["params"]["device"] = device   # MPS instead of default "cuda"
    # Local sgm DiffusionEngine doesn't accept ckpt_config; drop it
    cfg["model"]["params"].pop("ckpt_config", None)
    cfg = OmegaConf.create(cfg)
    engine = instantiate_from_config(cfg.model)
    engine.eval().requires_grad_(False).to(device)

    ckpt_path = RT_ALL_DATA / "cache/unclip6_epoch0_step110000.ckpt"
    print(f"  loading ckpt {ckpt_path.name} ({ckpt_path.stat().st_size / 1e9:.1f}GB)...", flush=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    engine.load_state_dict(ckpt["state_dict"], strict=False)

    batch = {
        "jpg": torch.randn(1, 3, 1, 1).to(device),
        "original_size_as_tuple": torch.ones(1, 2).to(device) * 768,
        "crop_coords_top_left": torch.zeros(1, 2).to(device),
    }
    with torch.no_grad():
        out = engine.conditioner(batch)
    vector_suffix = out["vector"].to(device)
    return engine, vector_suffix


def load_clip_img_embedder(device):
    """OpenCLIP ViT-bigG/14 wrapper matching paper's used embedder."""
    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device)
    model.eval().requires_grad_(False)

    class Wrapper:
        def __init__(self, m): self.m = m
        def __call__(self, x):
            with torch.no_grad():
                self.m.visual.output_tokens = True
                _, tokens = self.m.visual(x)
                return tokens
        def to(self, dev): self.m = self.m.to(dev); return self
    return Wrapper(model)


def load_cell(cell, ses="ses-03"):
    betas = np.load(PRE / f"{cell}_{ses}_betas.npy")
    ids = np.load(PRE / f"{cell}_{ses}_trial_ids.npy", allow_pickle=True)
    ids = np.asarray([str(t) for t in ids])
    return betas, ids


def first_rep_filter(betas, ids):
    """Keep only first occurrence per unique image (special515 only)."""
    seen, keep = set(), []
    for i, t in enumerate(ids):
        if "special515" not in t: continue
        if t in seen: continue
        seen.add(t); keep.append(i)
    return betas[keep], [ids[i] for i in keep]


def fwd_clip_voxels(model, x, device, batch=4):
    """Run model.ridge → backbone → clip_voxels for the input βs."""
    out = []
    with torch.no_grad():
        b = x.unsqueeze(1)
        for i in range(0, b.shape[0], batch):
            vr = model.ridge(b[i:i+batch], 0)
            ob = model.backbone(vr)
            cv = ob[1] if isinstance(ob, tuple) else ob
            out.append(cv.float())
    return torch.cat(out, 0)


def diffusion_prior_sample(model, clip_voxels, device, batch=2):
    """Run diffusion prior to get prior_out for unCLIP."""
    out = []
    with torch.no_grad():
        for i in range(0, clip_voxels.shape[0], batch):
            cv = clip_voxels[i:i+batch]
            prior = model.diffusion_prior.p_sample_loop(
                cv.shape, text_cond=dict(text_embed=cv),
                cond_scale=1., timesteps=20)
            out.append(prior)
    return torch.cat(out, 0)


def reconstruct_one_seed(model, diffusion_engine, vector_suffix, clip_voxels, device, seed):
    """Generate IMSIZE×IMSIZE reconstructions for all 50 trials at one seed."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    prior_out = diffusion_prior_sample(model, clip_voxels, device, batch=2)
    recons = []
    for i in range(prior_out.shape[0]):
        with torch.no_grad():
            samp = utils_mindeye.unclip_recon(
                prior_out[[i]], diffusion_engine, vector_suffix,
                num_samples=1)
        recons.append(samp)
    recons = torch.cat(recons, 0)            # (50, 3, 256, 256)
    return recons


def compute_recon_metrics(recons_all_seeds, gt, device):
    """recons_all_seeds: (S, 50, 3, 256, 256). gt: (50, 3, 256, 256). Returns 8 metrics with std across seeds."""
    from utils_mindeye import (
        calculate_pixcorr, calculate_ssim, calculate_alexnet,
        calculate_inception_v3, calculate_clip, calculate_efficientnet_b1,
        calculate_swav)
    metrics = {k: [] for k in ("pixcorr","ssim","alex2","alex5","inception","clip","effnet","swav")}
    for s in range(recons_all_seeds.shape[0]):
        rec = recons_all_seeds[s].to(device)
        metrics["pixcorr"].append(float(calculate_pixcorr(rec, gt)))
        metrics["ssim"].append(float(calculate_ssim(rec, gt)))
        a2, a5 = calculate_alexnet(rec, gt)
        metrics["alex2"].append(float(a2))
        metrics["alex5"].append(float(a5))
        metrics["inception"].append(float(calculate_inception_v3(rec, gt)))
        metrics["clip"].append(float(calculate_clip(rec, gt)))
        metrics["effnet"].append(float(calculate_efficientnet_b1(rec, gt)))
        metrics["swav"].append(float(calculate_swav(rec, gt)))
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v} for k, v in metrics.items()}


def topk(p, g, k=1):
    pf = p.reshape(p.shape[0], -1).cpu().numpy()
    gf = g.reshape(g.shape[0], -1).cpu().numpy()
    pn = pf / (np.linalg.norm(pf, axis=1, keepdims=True) + 1e-8)
    gn = gf / (np.linalg.norm(gf, axis=1, keepdims=True) + 1e-8)
    sim = pn @ gn.T
    labels = np.arange(p.shape[0])
    idx = np.argsort(-sim, axis=1)[:, :k]
    return float(np.mean([lbl in idx[i] for i, lbl in enumerate(labels)]))


def score_cell(cell, model, clip_img_embedder, diffusion_engine, vector_suffix, device, n_seeds, first_rep):
    print(f"\n========== {cell} ==========", flush=True)
    t0 = time.time()
    betas, ids = load_cell(cell)
    if first_rep:
        betas, names = first_rep_filter(betas, ids)
    else:
        names = list(ids)
    print(f"  test trials: {betas.shape[0]} ({len(names)} unique)", flush=True)
    if betas.shape[0] != 50:
        print(f"  skipping — expected 50 test trials, got {betas.shape[0]}", flush=True)
        return None

    # Load GT images at 256x256
    img_paths = [STIMULI / Path(n).name for n in names]
    gt_imgs = []
    for p in img_paths:
        im = Image.open(p).convert("RGB").resize((256, 256), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        gt_imgs.append(torch.from_numpy(arr).permute(2, 0, 1))
    gt = torch.stack(gt_imgs).to(device)         # (50, 3, 256, 256)
    print(f"  GT images: {gt.shape}", flush=True)

    # Predicted clip_voxels
    x = torch.from_numpy(betas.astype(np.float32)).to(device)
    clip_voxels = fwd_clip_voxels(model, x, device).to(device)        # (50, 256, 1664)
    print(f"  clip_voxels: {clip_voxels.shape}", flush=True)

    # Retrieval
    with torch.no_grad():
        gt_resized = nn.functional.interpolate(gt, size=(IMSIZE, IMSIZE), mode='bilinear', align_corners=False)
        gt_norm = transforms.functional.normalize(
            gt_resized,
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711])
        gt_clip = clip_img_embedder(gt_norm).float()  # (50, 256, 1664)
    img_acc = topk(clip_voxels, gt_clip, k=1)
    bra_acc = topk(gt_clip, clip_voxels, k=1)
    print(f"  retrieval: Image={img_acc*100:.1f}% Brain={bra_acc*100:.1f}%", flush=True)

    # Reconstructions across seeds
    print(f"  reconstructing {n_seeds} seed(s) × 50 imgs (~{n_seeds*8:.0f} min on MPS)...", flush=True)
    recons_all = []
    for s in range(n_seeds):
        t_seed = time.time()
        rec = reconstruct_one_seed(model, diffusion_engine, vector_suffix,
                                    clip_voxels, device, seed=s)
        recons_all.append(rec.cpu())
        print(f"    seed {s}: {(time.time()-t_seed):.1f}s, recons shape {rec.shape}", flush=True)
    recons_all = torch.stack(recons_all, 0)        # (S, 50, 3, 256, 256)

    # Recon metrics
    print(f"  computing 8 perceptual metrics...", flush=True)
    metrics = compute_recon_metrics(recons_all, gt, device)

    result = {
        "cell": cell, "n_test": len(names),
        "image_top1": img_acc, "brain_top1": bra_acc,
        "n_seeds": n_seeds,
        **{k: m for k, m in metrics.items()},
        "elapsed_s": time.time() - t0,
    }
    print(f"  {cell}: PixCorr={metrics['pixcorr']['mean']:.3f}  SSIM={metrics['ssim']['mean']:.3f}  "
          f"Alex2={metrics['alex2']['mean']*100:.1f}%  Alex5={metrics['alex5']['mean']*100:.1f}%  "
          f"Incep={metrics['inception']['mean']*100:.1f}%  CLIP={metrics['clip']['mean']*100:.1f}%  "
          f"Eff={metrics['effnet']['mean']:.3f}  SwAV={metrics['swav']['mean']:.3f}  "
          f"Image={img_acc*100:.1f}%  Brain={bra_acc*100:.1f}%  ({(time.time()-t0)/60:.1f}m)", flush=True)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cells", type=str, default="Probe_canonical_sessztrain_firstrep")
    p.add_argument("--checkpoint", type=Path,
                   default=LOCAL / "rt3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth")
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--first-rep", action="store_true", default=True)
    p.add_argument("--num-voxels", type=int, default=2792)
    p.add_argument("--quick", action="store_true", help="n-seeds=1 + first cell only")
    p.add_argument("--out", type=Path,
                   default=LOCAL / "task_2_1_betas/full_metrics_mac.json")
    args = p.parse_args()

    if args.quick:
        args.n_seeds = 1
        args.cells = args.cells.split(",")[0]

    cells = args.cells.split(",")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"=== Mac MPS full Table 1 metrics ===", flush=True)
    print(f"  device={device}, ckpt={args.checkpoint.name}", flush=True)
    print(f"  cells: {cells}, n_seeds={args.n_seeds}, first_rep={args.first_rep}", flush=True)

    # Build + load MindEye
    print(f"\n  building MindEye...", flush=True)
    model = build_mindeye(args.num_voxels, device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"  loaded ckpt: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    model.eval().requires_grad_(False)

    # Load diffusion engine + clip embedder
    print(f"\n  loading diffusion engine (SDXL Turbo unCLIP)...", flush=True)
    diffusion_engine, vector_suffix = load_diffusion_engine(device)
    print(f"\n  loading CLIP image embedder...", flush=True)
    clip_img_embedder = load_clip_img_embedder(device)

    # Score cells
    results = []
    for cell in cells:
        try:
            r = score_cell(cell, model, clip_img_embedder, diffusion_engine,
                           vector_suffix, device, args.n_seeds, args.first_rep)
            if r is not None:
                results.append(r)
        except Exception as e:
            print(f"  ERROR on {cell}: {e}", flush=True)
            import traceback; traceback.print_exc()
            continue

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n  saved {args.out.name}: {len(results)} cells", flush=True)


if __name__ == "__main__":
    main()
