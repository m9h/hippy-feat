"""Compare paper's saved 224x224 preprocessed image with our 256→224 resize.
Paper's `all_ground_truth.pt` for special_67295 is shape (3, 3, 224, 224) —
3 reps × CHW image. That's the input fed to OpenCLIP."""
from pathlib import Path
import sys, types, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
device = "mps" if torch.backends.mps.is_available() else "cpu"

# Paper's 224×224 preprocessed image
paper_img_path = LOCAL / "glmsingle/canonical_per_image_gt/data_scaling_exp/concat_glmsingle/sub-005_all_task-C_bs24_MST_rishab_repeats_3split_sample=10_avgrepeats_finalmask_epochs_150_delay=0/0/sanity_check_individual_reps/special_67295/all_ground_truth.pt"
paper_img = torch.load(paper_img_path, map_location="cpu", weights_only=False)
print(f"paper input image (3 reps): shape={tuple(paper_img.shape)}  range=[{paper_img.min():.4f}, {paper_img.max():.4f}]")

# All 3 reps should be identical since it's the same image
diffs = (paper_img[0] - paper_img[1]).abs().max(), (paper_img[0] - paper_img[2]).abs().max()
print(f"  max diff between reps: {diffs}")

# Load paper's 256×256 image from seedwise dump for the SAME image
# images.pt is (50, 3, 256, 256). We need index of special_67295.
# The images.pt order should match the test set order.
images_256 = torch.load(
    LOCAL / "glmsingle/fold10_canonical_seed0/seedwise_runs_dump/offline/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150/0/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_10_avgrepeats_finalmask_epochs_150_all_images.pt",
    map_location="cpu", weights_only=False
)
print(f"paper test images (50 × 256×256): shape={tuple(images_256.shape)}  range=[{images_256.min():.4f}, {images_256.max():.4f}]")

# Try our 256→224 resize on first image (assumed special_67295 is index 0)
test_idx = 0  # ses-03 first occurrence is special_67295
src_256 = images_256[test_idx:test_idx+1]
print(f"\nsrc image at idx {test_idx}: {tuple(src_256.shape)}")

# Try multiple resize methods
methods = {
    "bilinear+antialias": dict(size=(224,224), antialias=True, interpolation=T.InterpolationMode.BILINEAR),
    "bicubic+antialias":  dict(size=(224,224), antialias=True, interpolation=T.InterpolationMode.BICUBIC),
    "bilinear (no aa)":   dict(size=(224,224), antialias=False, interpolation=T.InterpolationMode.BILINEAR),
}

for name, kw in methods.items():
    our_224 = T.functional.resize(src_256, **kw)
    paper_224_rep0 = paper_img[0:1]  # (1, 3, 224, 224)
    # Pixel-level diff
    diff = (our_224 - paper_224_rep0).abs()
    print(f"  resize='{name}': max-pixel-diff={diff.max():.4f} mean-pixel-diff={diff.mean():.4f}")

# What if paper's 224 came from a CENTER CROP of 256 instead of resize?
crop = T.functional.center_crop(src_256, (224, 224))
diff = (crop - paper_img[0:1]).abs()
print(f"  center_crop:        max-pixel-diff={diff.max():.4f} mean-pixel-diff={diff.mean():.4f}")

# Or from the original JPG via PIL
print("\n  trying PIL JPEG load + paper's likely preprocessing chain:")
img = Image.open(LOCAL / "rt3t/data/all_stimuli/special515/special_67295.jpg")
print(f"    PIL image size: {img.size}, mode={img.mode}")
# The JPEG might be at higher res; check
arr = np.asarray(img)
print(f"    np array: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min()}, {arr.max()}]")

# Convert to tensor
t = torch.from_numpy(arr).permute(2,0,1).float() / 255.0
t = t.unsqueeze(0)
print(f"    tensor: shape={tuple(t.shape)}")
for name, kw in methods.items():
    direct_224 = T.functional.resize(t, **kw)
    diff = (direct_224 - paper_img[0:1]).abs()
    print(f"    direct JPG → resize '{name}': max-diff={diff.max():.4f} mean-diff={diff.mean():.4f}")
