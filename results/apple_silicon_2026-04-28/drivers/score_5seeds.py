"""Score canonical clipvoxels across all 5 seeds + average; compare to paper's
reported 90% top-1 (which is the 5-seed average per Table 1 caption)."""
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
warnings.filterwarnings("ignore")

device = "mps" if torch.backends.mps.is_available() else "cpu"
prefix = "sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150"

# Load images once (same for all seeds — test set is fixed)
images = torch.load(
    LOCAL / f"glmsingle/fold10_canonical_seed0/seedwise_runs_dump/offline/{prefix}/0/{prefix}_all_images.pt",
    map_location="cpu", weights_only=False,
)

# Compute GT once
import torchvision.transforms as T
from open_clip import create_model_and_transforms
clip_m, _, _ = create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k", device=device)
clip_m.eval()
clip_m.visual.output_tokens = True
img_input = images.to(device).float()
with torch.no_grad():
    img_resized = T.functional.resize(img_input, (224, 224), antialias=True)
    img_norm = T.functional.normalize(img_resized,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711])
    _, gt_tokens = clip_m.visual(img_norm)
gt_np = gt_tokens.float().cpu().numpy()
print(f"GT computed: {gt_np.shape}")

# Score each seed
results = []
for seed in range(5):
    if seed == 0:
        cv_path = LOCAL / f"glmsingle/fold10_canonical_seed0/seedwise_runs_dump/offline/{prefix}/{seed}/{prefix}_all_clipvoxels.pt"
        ev_path = LOCAL / f"glmsingle/fold10_canonical_seed0/seedwise_runs_dump/offline/{prefix}/{seed}/final_evals.csv"
    else:
        cv_path = LOCAL / f"glmsingle/fold10_canonical_seeds_all/seedwise_runs_dump/offline/{prefix}/{seed}/{prefix}_all_clipvoxels.pt"
        ev_path = LOCAL / f"glmsingle/fold10_canonical_seeds_all/seedwise_runs_dump/offline/{prefix}/{seed}/final_evals.csv"

    cv = torch.load(cv_path, map_location="cpu", weights_only=False).float().cpu().numpy()
    sim = M.cosine_sim_tokens(cv, gt_np)
    t1 = float(np.mean([np.argmax(sim[i]) == i for i in range(sim.shape[0])]))
    t5 = float(np.mean([i in np.argsort(sim[i])[-5:] for i in range(sim.shape[0])]))

    # Read paper's own reported numbers from CSV
    paper_fwd = paper_bwd = None
    if ev_path.exists():
        with open(ev_path) as f:
            for line in f:
                if line.startswith("fwd_acc"):
                    paper_fwd = float(line.split(",")[1])
                elif line.startswith("bwd_acc"):
                    paper_bwd = float(line.split(",")[1])
    results.append({"seed": seed, "our_top1": t1, "our_top5": t5,
                     "paper_fwd": paper_fwd, "paper_bwd": paper_bwd})
    print(f"seed {seed}:  ours top-1={t1*100:5.2f}%  top-5={t5*100:5.2f}%  | "
          f"paper fwd={paper_fwd*100:5.2f}%  bwd={paper_bwd*100:5.2f}%" if paper_fwd else "")

ours_t1 = np.mean([r["our_top1"] for r in results])
ours_t5 = np.mean([r["our_top5"] for r in results])
paper_t1 = np.mean([r["paper_fwd"] for r in results if r["paper_fwd"] is not None])
print(f"\n5-SEED AVERAGE:")
print(f"  ours:  top-1={ours_t1*100:.2f}%  top-5={ours_t5*100:.2f}%")
print(f"  paper: fwd_acc={paper_t1*100:.2f}%")
print(f"  Paper Table 1 'Offline 3T (avg. 3 reps.)' anchor: 90.0% / 88.0%")
