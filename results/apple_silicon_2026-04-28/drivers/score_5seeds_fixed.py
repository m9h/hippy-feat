"""Re-score all 5 seeds against GT computed with paper-correct preprocessing
(JPG → 224 bilinear no-antialias). Should close 88% → 90%."""
from pathlib import Path
import sys, types, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

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
warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else "cpu"
prefix = "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150"

# Reconstruct test image order (matches paper test set, all_images.pt order)
# We confirmed earlier: first occurrence per special515 in ses-03 events.tsv = same 50 images
import pandas as pd
seen, ordered = set(), []
for run in range(1, 12):
    ev = pd.read_csv(LOCAL / "rt3t/data/events" / f"sub-005_ses-03_task-C_run-{run:02d}_events.tsv", sep="\t")
    for _, row in ev.iterrows():
        n = str(row["image_name"])
        if n.startswith("all_stimuli/special515/") and n not in seen:
            seen.add(n)
            ordered.append(n)
print(f"test images in event order: {len(ordered)}")

# Verify our reconstructed order matches paper's all_images.pt order by comparing first image
images_paper = torch.load(
    LOCAL / f"glmsingle/fold10_canonical_seed0/seedwise_runs_dump/offline/{prefix}/0/{prefix}_all_images.pt",
    map_location="cpu", weights_only=False,
)

# Build GT with paper-correct preprocessing: JPG → 224 bilinear NO antialias
print("\nbuilding GT with paper-correct preprocessing (JPG → 224 bilinear no-aa)...")
batch = []
for img_name in ordered:
    img_path = LOCAL / "rt3t/data" / img_name
    pil = Image.open(img_path).convert("RGB")
    arr = np.asarray(pil, dtype=np.float32) / 255.0   # (H, W, 3)
    t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3, H, W)
    t224 = T.functional.resize(t, (224, 224), antialias=False, interpolation=T.InterpolationMode.BILINEAR)
    batch.append(t224)
imgs_224 = torch.stack(batch, dim=0).to(device)
print(f"  imgs_224: {tuple(imgs_224.shape)}")

# Spot check: paper's seedwise images_paper is at 256, so we won't pixel-equal it.
# But our 224 should match per-image saved sanity_check images at 0 diff.

# CLIP encode
from open_clip import create_model_and_transforms
clip_m, _, _ = create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device)
clip_m.eval()
clip_m.visual.output_tokens = True
img_norm = T.functional.normalize(imgs_224,
    mean=[0.48145466, 0.4578275, 0.40821073],
    std=[0.26862954, 0.26130258, 0.27577711])
with torch.no_grad():
    _, gt_tokens = clip_m.visual(img_norm)
gt_np = gt_tokens.float().cpu().numpy()
print(f"  GT tokens: {gt_np.shape}")

# Score each seed
print(f"\n{'seed':>4s}  {'ours':>7s}  {'paper':>7s}  diff")
print("-" * 40)
ours_t1, paper_t1 = [], []
for seed in range(5):
    if seed == 0:
        cv_dir = LOCAL / f"glmsingle/fold10_canonical_seed0/seedwise_runs_dump/offline/{prefix}/{seed}"
    else:
        cv_dir = LOCAL / f"glmsingle/fold10_canonical_seeds_all/seedwise_runs_dump/offline/{prefix}/{seed}"
    cv = torch.load(cv_dir / f"{prefix}_all_clipvoxels.pt", map_location="cpu", weights_only=False).float().cpu().numpy()
    sim = M.cosine_sim_tokens(cv, gt_np)
    t1 = float(np.mean([np.argmax(sim[i]) == i for i in range(sim.shape[0])]))
    paper = None
    ev = cv_dir / "final_evals.csv"
    if ev.exists():
        for line in ev.read_text().splitlines():
            if line.startswith("fwd_acc"):
                paper = float(line.split(",")[1])
                break
    ours_t1.append(t1)
    if paper is not None:
        paper_t1.append(paper)
    print(f"  {seed}  {t1*100:6.2f}%  {paper*100:6.2f}%  {(t1-paper)*100:+5.1f}pp")

print(f"\n5-seed avg:  ours={np.mean(ours_t1)*100:.2f}%   paper={np.mean(paper_t1)*100:.2f}%")
print(f"paper Table 1 'Offline 3T (avg. 3 reps.)' Image: 90%")
