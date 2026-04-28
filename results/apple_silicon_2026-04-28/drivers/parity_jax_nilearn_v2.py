#!/usr/bin/env python3
"""Parity test of the FIXED Prais-Winsten + iterated _variant_g_forward_pw
against nilearn FirstLevelModel(noise_model='ar1') with the same DM.
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
sys.path.insert(0, str(Path(__file__).parent))

from rt_glm_variants import (
    _variant_g_forward,
    build_design_matrix,
    make_glover_hrf,
)
from variant_g_pw import _variant_g_forward_pw

warnings.filterwarnings("ignore")

import prereg_variant_sweep as P
LOCAL = Path("/Users/mhough/Workspace/data/rtmindeye_paper")
P.RT3T = LOCAL / "rt3t" / "data"
P.MC_DIR = LOCAL / "motion_corrected_resampled"
P.BRAIN_MASK = LOCAL / "rt3t" / "data" / "sub-005_final_mask.nii.gz"
P.RELMASK = LOCAL / "rt3t" / "data" / "sub-005_ses-01_task-C_relmask.npy"
EVENTS_DIR = LOCAL / "rt3t" / "data" / "events"


def main():
    SESSION, RUN, TR = "ses-03", 1, 1.5
    flat_brain = (nib.load(P.BRAIN_MASK).get_fdata() > 0).flatten()
    rel = np.load(P.RELMASK)
    ts = P.load_rtmotion(SESSION, RUN, flat_brain, rel)
    events = pd.read_csv(EVENTS_DIR / f"sub-005_{SESSION}_task-C_run-{RUN:02d}_events.tsv", sep="\t")
    onsets = (events["onset"].astype(float) - events["onset"].iloc[0]).values
    n_trs = ts.shape[1]
    hrf = make_glover_hrf(TR, int(np.ceil(32.0 / TR)))
    PROBE = 0
    dm, probe_col = build_design_matrix(onsets, TR, n_trs, hrf, PROBE)

    max_trs = 200
    dm_pad = np.zeros((max_trs, dm.shape[1]), dtype=np.float32); dm_pad[:n_trs] = dm
    ts_pad = np.zeros((ts.shape[0], max_trs), dtype=np.float32); ts_pad[:, :n_trs] = ts

    # JAX (current, Cochrane-Orcutt single-pass)
    b_current, _ = _variant_g_forward(jnp.asarray(dm_pad), jnp.asarray(ts_pad),
                                       jnp.asarray(n_trs, dtype=jnp.int32),
                                       pp_scalar=0.0, rho_prior_mean=0.0, rho_prior_var=1e8)
    b_current = np.asarray(b_current)[:, probe_col]

    # JAX (fixed, Prais-Winsten + 3 iter)
    b_fixed, _ = _variant_g_forward_pw(jnp.asarray(dm_pad), jnp.asarray(ts_pad),
                                        jnp.asarray(n_trs, dtype=jnp.int32),
                                        pp_scalar=0.0, rho_prior_mean=0.0, rho_prior_var=1e8,
                                        n_iter=3)
    b_fixed = np.asarray(b_fixed)[:, probe_col]

    # nilearn with same DM
    from nilearn.glm.first_level import FirstLevelModel
    pattern = f"{SESSION}_run-{RUN:02d}_*_mc_boldres.nii.gz"
    vols = sorted(P.MC_DIR.glob(pattern))
    arr = np.stack([nib.load(v).get_fdata().astype(np.float32) for v in vols], axis=-1)
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

    def _report(name, a, b):
        diff = np.abs(a - b)
        print(f"\n=== {name} ===")
        print(f"  max diff: {diff.max():.6f}   mean diff: {diff.mean():.6f}")
        print(f"  within 1e-3: {(diff<1e-3).mean()*100:.2f}%   within 1e-2: {(diff<1e-2).mean()*100:.2f}%")
        print(f"  pearson r: {np.corrcoef(a, b)[0,1]:.6f}   scale ratio (a/b): {a.std()/b.std():.4f}")

    print(f"JAX current  std={b_current.std():.4f} mean={b_current.mean():.4f}")
    print(f"JAX fixed-pw std={b_fixed.std():.4f} mean={b_fixed.mean():.4f}")
    print(f"nilearn      std={b_nil.std():.4f} mean={b_nil.mean():.4f}")
    _report("nil vs JAX current", b_nil, b_current)
    _report("nil vs JAX fixed-pw", b_nil, b_fixed)
    _report("JAX fixed-pw vs JAX current", b_fixed, b_current)


if __name__ == "__main__":
    main()
