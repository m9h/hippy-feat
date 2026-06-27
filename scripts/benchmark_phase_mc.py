#!/usr/bin/env python3
import time
import numpy as np
import nibabel as nib
import jax.numpy as jnp
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jaxoccoli.motion_phase import estimate_translation, apply_translation

# Use a sample run from ses-03
RAW_BOLD = "/data/derivatives/rtmindeye_paper/rt3t/data/raw_bids/sub-005/ses-03/func/sub-005_ses-03_task-C_run-01_bold.nii.gz"

def benchmark_phase_mc():
    print(f"Loading sample run: {RAW_BOLD}")
    img = nib.load(RAW_BOLD)
    data = img.get_fdata()
    T = data.shape[-1]
    template = data[..., 0] # Use first volume as template
    
    print(f"Volume shape: {template.shape}, TRs: {T}")
    
    translations = []
    t0 = time.time()
    
    # Process first 20 volumes for benchmark
    n_benchmark = 20
    for t in range(1, n_benchmark):
        moving = data[..., t]
        shift = estimate_translation(jnp.asarray(template), jnp.asarray(moving))
        translations.append(shift)
        if t % 5 == 0:
            print(f"  TR {t:02d}: shift {shift}")
            
    elapsed = time.time() - t0
    avg_time = (elapsed / (n_benchmark - 1)) * 1000
    print(f"\nAverage time per volume: {avg_time:.2f} ms")
    
    translations = np.array(translations)
    print(f"Max translation (voxels): {np.abs(translations).max(axis=0)}")
    print(f"Mean translation (voxels): {translations.mean(axis=0)}")
    
    # Check RMS reduction
    sample_moving = data[..., 1]
    shift_1 = translations[0]
    registered_1 = np.asarray(apply_translation(jnp.asarray(sample_moving), shift_1))
    
    rms_raw = np.sqrt(np.mean((template - sample_moving)**2))
    rms_reg = np.sqrt(np.mean((template - registered_1)**2))
    
    print(f"\nRMS diff (TR 0 vs 1):")
    print(f"  Raw:        {rms_raw:.2f}")
    print(f"  Registered: {rms_reg:.2f}")
    print(f"  Reduction:  {(rms_raw - rms_reg)/rms_raw*100:.1f}%")

if __name__ == "__main__":
    benchmark_phase_mc()
