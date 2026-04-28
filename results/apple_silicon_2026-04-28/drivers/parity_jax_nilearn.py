#!/usr/bin/env python3
"""Single-run parity check between JAX `_variant_g_forward(pp=0, ρ prior var=1e8)`
and nilearn `FirstLevelModel(noise_model='ar1')`.

Method: build the JAX design matrix once, pass it to BOTH engines (nilearn via
`design_matrices=[dm_df]`). This isolates the AR(1) whitening implementation
from any design-matrix differences (drift basis, intercept, cosine count, etc.).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import jax.numpy as jnp
import nibabel as nib
import numpy as np
import pandas as pd

REPO = Path("/Users/mhough/Workspace/hippy-feat")
sys.path.insert(0, str(REPO / "scripts"))

from rt_glm_variants import (
    _variant_g_forward,
    build_design_matrix,
    make_glover_hrf,
)
from prereg_variant_sweep import load_rtmotion

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
import prereg_variant_sweep as P
P.RT3T = LOCAL / "rt3t" / "data"
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
P.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"


def main():
    SESSION, RUN, TR = "ses-03", 1, 1.5

    flat_brain = (nib.load(P.BRAIN_MASK).get_fdata() > 0).flatten()
    rel = np.load(P.RELMASK)
    ts = P.load_rtmotion(SESSION, RUN, flat_brain, rel)   # (V, T)
    events = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{RUN:02d}_events.tsv", sep="\t")
    onsets = (events["onset"].astype(float) - events["onset"].iloc[0]).values
    n_trs = ts.shape[1]

    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))
    PROBE = 0
    dm, probe_col = build_design_matrix(onsets, TR, n_trs, hrf, PROBE)
    print(f"design matrix: {dm.shape} (cols: probe, ref, intercept, cos_drift)")
    print(f"timeseries: {ts.shape}")

    # ---- JAX path ----
    max_trs = 200
    dm_pad = np.zeros((max_trs, dm.shape[1]), dtype=np.float32); dm_pad[:n_trs] = dm
    ts_pad = np.zeros((ts.shape[0], max_trs), dtype=np.float32); ts_pad[:, :n_trs] = ts
    b_jax, _ = _variant_g_forward(jnp.asarray(dm_pad), jnp.asarray(ts_pad),
                                   jnp.asarray(n_trs, dtype=jnp.int32),
                                   pp_scalar=0.0, rho_prior_mean=0.0, rho_prior_var=1e8)
    b_jax = np.asarray(b_jax)[:, probe_col]
    print(f"\nJAX β (probe col): {b_jax.shape}, mean={b_jax.mean():.4f}, std={b_jax.std():.4f}")

    # ---- nilearn with EXACTLY the same DM ----
    from nilearn.glm.first_level import FirstLevelModel
    bold_4d = []
    # Load 4D BOLD as nilearn-compatible Nifti1Image
    pattern = f"{SESSION}_run-{RUN:02d}_*_mc_boldres.nii.gz"
    vols = sorted(P.MC_DIR.glob(pattern))
    frames = [nib.load(v).get_fdata().astype(np.float32) for v in vols]
    arr = np.stack(frames, axis=-1)
    bold_4d_img = nib.Nifti1Image(arr, nib.load(vols[0]).affine)
    mask_img = nib.load(P.BRAIN_MASK)

    dm_df = pd.DataFrame(dm, columns=["probe", "ref", "intercept", "cos_drift"])
    glm = FirstLevelModel(t_r=TR, slice_time_ref=0,
                          drift_model=None, high_pass=0.0,
                          signal_scaling=False, smoothing_fwhm=None,
                          noise_model="ar1",
                          n_jobs=1, verbose=0,
                          minimize_memory=True, mask_img=mask_img)
    glm.fit(run_imgs=bold_4d_img, design_matrices=[dm_df])
    eff = glm.compute_contrast("probe", output_type="effect_size").get_fdata()
    b_nil = eff.flatten()[flat_brain][rel]
    print(f"nil β (probe col): {b_nil.shape}, mean={b_nil.mean():.4f}, std={b_nil.std():.4f}")

    # ---- compare ----
    diff = np.abs(b_jax - b_nil)
    print(f"\n=== parity (same DM, AR1 whitening) ===")
    print(f"  max abs diff: {diff.max():.6f}")
    print(f"  mean abs diff: {diff.mean():.6f}")
    print(f"  fraction within 1e-3: {(diff < 1e-3).mean():.4f}")
    print(f"  fraction within 1e-2: {(diff < 1e-2).mean():.4f}")
    print(f"  pearson r: {np.corrcoef(b_jax, b_nil)[0,1]:.6f}")
    print(f"  scale ratio (nil/jax): {b_nil.std() / b_jax.std():.4f}")


if __name__ == "__main__":
    main()
