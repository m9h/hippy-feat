"""Score retrieval directly on the canonical predicted CLIP voxels (paper output)
to verify our test set + retrieval logic, isolating only the model forward pass."""
from pathlib import Path
import sys, types, warnings
import numpy as np
import torch
import torch.nn as nn

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

ROOT = LOCAL / "glmsingle/fold10_canonical_seed0/seedwise_runs_dump/offline/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/0"
prefix = "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150"

clipvoxels = torch.load(ROOT / f"{prefix}_all_clipvoxels.pt", map_location="cpu", weights_only=False)
images = torch.load(ROOT / f"{prefix}_all_images.pt", map_location="cpu", weights_only=False)
mst_id = np.load(ROOT / f"{prefix}_MST_ID.npy", allow_pickle=True)

print(f"clipvoxels: {type(clipvoxels).__name__} shape={tuple(clipvoxels.shape) if hasattr(clipvoxels,'shape') else '?'}  dtype={clipvoxels.dtype if hasattr(clipvoxels,'dtype') else '?'}")
print(f"images: {type(images).__name__} shape={tuple(images.shape) if hasattr(images,'shape') else '?'}  dtype={images.dtype if hasattr(images,'dtype') else '?'}")
print(f"MST_ID: shape={mst_id.shape}  unique={len(np.unique(mst_id))}")

# Normalize and score retrieval directly on these canonical clipvoxels.
# Use Mac's CLIP encoder to encode the same images for GT.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\ncomputing GT embeddings via OpenCLIP ViT-bigG/14 on {device}...")

# `images` is (n, 3, H, W) — same images used to fit the ridge.
import torchvision.transforms as T
from open_clip import create_model_and_transforms
clip_m, _, preprocess = create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device)
clip_m.eval()
img_input = images.to(device).float()
# Open-CLIP normalize matches the paper's preprocessing: mean/std applied at the model level.
clip_m_visual = clip_m.visual
clip_m_visual.output_tokens = True
with torch.no_grad():
    # Resize to 224x224 (visual model expects)
    img_resized = T.functional.resize(img_input, (224, 224), antialias=True)
    img_norm = T.functional.normalize(img_resized,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
    _, gt_tokens = clip_m_visual(img_norm)  # (n, 256, 1664)
print(f"  gt_tokens shape: {gt_tokens.shape}")

# Score canonical clipvoxels (n, 256, 1664) against gt_tokens
sim = M.cosine_sim_tokens(clipvoxels.to(device).float().cpu().numpy(),
                           gt_tokens.float().cpu().numpy())
print(f"  sim shape: {sim.shape}")

# Top-1 retrieval: argmax over similarity
correct_top1 = np.array([np.argmax(sim[i]) == i for i in range(sim.shape[0])])
correct_top5 = np.array([i in np.argsort(sim[i])[-5:] for i in range(sim.shape[0])])
print(f"\n  TOP-1 on canonical clipvoxels: {correct_top1.mean()*100:.2f}%")
print(f"  TOP-5 on canonical clipvoxels: {correct_top5.mean()*100:.2f}%")
print(f"  paper Table 1 Offline 3T avg-3-rep (paper anchor): 90% / 88%")
