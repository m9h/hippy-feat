#!/usr/bin/env python
"""WAND resting-state FC gradients vs neuromaps reference maps.

Parcellates BOLD with Schaefer 400 atlas, computes FC gradients with
jaxoccoli, then correlates with neurotransmitter receptor, structural,
and functional reference maps from neuromaps.
"""

import sys
sys.path.insert(0, "/Users/mhough/dev/hippy-feat")

import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pkg_resources")
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import nibabel as nib
import jax
import jax.numpy as jnp

import jaxoccoli as joc

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
WAND = "/Users/mhough/dev/wand"
SUB = "sub-08033"
SES = "ses-03"
BOLD_MC = f"{WAND}/derivatives/fsl-fmri/{SUB}/{SES}/mc/task-rest_mc.nii.gz"
BOLD_JSON = f"{WAND}/{SUB}/{SES}/func/{SUB}_{SES}_task-rest_bold.json"
TR = 2.0

print("=" * 70)
print("WAND Neuromaps + Multi-Atlas FC Gradient Analysis")
print("=" * 70)

# ===================================================================
# PART 1: Multi-atlas parcellation
# ===================================================================
print("\n[PART 1] Multi-atlas parcellation of resting BOLD")
print("-" * 50)

from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets

bold_img = nib.load(BOLD_MC)
print(f"  BOLD: {bold_img.shape}, TR={TR}s")

# --- Schaefer 400 (Yeo 7 networks) ---
t0 = time.time()
schaefer400 = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
masker_s400 = NiftiLabelsMasker(
    labels_img=schaefer400['maps'],
    standardize='zscore_sample',
    resampling_target='data',
    t_r=TR, low_pass=0.1, high_pass=0.01, detrend=True,
)
ts_s400 = masker_s400.fit_transform(BOLD_MC)  # (300, 400)
print(f"  Schaefer 400: {ts_s400.shape}, time={time.time()-t0:.1f}s")

# --- Schaefer 100 (coarser) ---
t0 = time.time()
schaefer100 = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
masker_s100 = NiftiLabelsMasker(
    labels_img=schaefer100['maps'],
    standardize='zscore_sample',
    resampling_target='data',
    t_r=TR, low_pass=0.1, high_pass=0.01, detrend=True,
)
ts_s100 = masker_s100.fit_transform(BOLD_MC)  # (300, 100)
print(f"  Schaefer 100: {ts_s100.shape}, time={time.time()-t0:.1f}s")

# --- Harvard-Oxford cortical ---
t0 = time.time()
ho = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
masker_ho = NiftiLabelsMasker(
    labels_img=ho['maps'],
    standardize='zscore_sample',
    resampling_target='data',
    t_r=TR, low_pass=0.1, high_pass=0.01, detrend=True,
)
ts_ho = masker_ho.fit_transform(BOLD_MC)
n_ho = ts_ho.shape[1]
print(f"  Harvard-Oxford: {ts_ho.shape}, time={time.time()-t0:.1f}s")

# --- DiFuMo 64 (data-driven functional modes — 4D prob maps, needs NiftiMapsMasker) ---
t0 = time.time()
from nilearn.maskers import NiftiMapsMasker
difumo = datasets.fetch_atlas_difumo(dimension=64, resolution_mm=2)
masker_di = NiftiMapsMasker(
    maps_img=difumo['maps'],
    standardize='zscore_sample',
    resampling_target='data',
    t_r=TR, low_pass=0.1, high_pass=0.01, detrend=True,
)
ts_di = masker_di.fit_transform(BOLD_MC)
print(f"  DiFuMo 64: {ts_di.shape}, time={time.time()-t0:.1f}s")

# ===================================================================
# PART 2: FC and gradients per atlas with jaxoccoli
# ===================================================================
print("\n[PART 2] FC matrices and gradients per atlas")
print("-" * 50)

results = {}
for name, ts in [("Schaefer400", ts_s400), ("Schaefer100", ts_s100),
                  ("HarvardOxford", ts_ho), ("DiFuMo64", ts_di)]:
    t0 = time.time()
    ts_jax = jnp.array(ts.T, dtype=jnp.float32)  # (ROIs, T)
    n_rois = ts_jax.shape[0]

    # Full correlation
    fc = joc.corr(ts_jax)

    # Partial correlation
    fc_partial = joc.partial_corr(ts_jax, l2=0.1)

    # Affinity for embedding
    W = jnp.abs(fc) * (1 - jnp.eye(n_rois))

    # Laplacian eigenmaps
    eigvals_le, eigvecs_le = joc.laplacian_eigenmaps(W, k=10)

    # Diffusion mapping
    eigvals_dm, eigvecs_dm = joc.diffusion_mapping(W, k=10, alpha=0.5)

    # Spectral features
    sf = joc.spectral_features(W, k=10)

    # Modularity (5 communities from spectral)
    C_raw = eigvecs_le[:, :5]
    C_soft = jax.nn.softmax(C_raw * 10, axis=-1)
    Q = joc.relaxed_modularity(W, C_soft)

    elapsed = time.time() - t0

    results[name] = {
        'fc': fc, 'fc_partial': fc_partial,
        'eigvals_le': eigvals_le, 'eigvecs_le': eigvecs_le,
        'eigvals_dm': eigvals_dm, 'eigvecs_dm': eigvecs_dm,
        'spectral_features': sf, 'modularity': Q,
        'n_rois': n_rois,
    }

    print(f"  {name} ({n_rois} ROIs): Q={float(Q):.4f}, "
          f"lambda2={float(sf[-2]):.4f}, "
          f"spec_radius={float(sf[-1]):.4f}, "
          f"time={elapsed:.2f}s")

# ===================================================================
# PART 3: Neuromaps reference maps (MNI152 volumetric)
# ===================================================================
print("\n[PART 3] Fetching neuromaps reference maps")
print("-" * 50)

from neuromaps.datasets import fetch_annotation

# Monkey-patch neuromaps to work with newer nilearn (str -> Path compat)
import neuromaps.datasets.annotations as _nma
_orig_fetch = _nma._fetch_file
def _patched_fetch(url, data_dir, **kwargs):
    from pathlib import Path
    return _orig_fetch(url, Path(data_dir) if isinstance(data_dir, str) else data_dir, **kwargs)
_nma._fetch_file = _patched_fetch

# Fetch all MNI152 annotations we can directly parcellate
mni_annotations = [
    ('neurosynth', 'cogpc1', 'Neurosynth cognitive PC1'),
    ('alarkurtti2015', 'raclopride', 'D2/D3 receptor (raclopride)'),
    ('beliveau2017', 'cumi101', '5-HT1A receptor (CUMI-101)'),
    ('beliveau2017', 'cimbi36', '5-HT2A receptor (Cimbi-36)'),
    ('beliveau2017', 'dasb', 'SERT (DASB)'),
    ('dubois2015', 'abp688', 'mGluR5 (ABP688)'),
    ('finnema2016', 'ucbj', 'Synaptic density SV2A (UCB-J)'),
    ('gallezot2017', 'gsk189254', 'Histamine H3 (GSK189254)'),
    ('jaworska2020', 'fallypride', 'D2/D3 receptor (fallypride)'),
    ('kaller2017', 'sch23390', 'D1 receptor (SCH23390)'),
    ('kantonen2020', 'carfentanil', 'Mu-opioid receptor (carfentanil)'),
    ('satterthwaite2014', 'meancbf', 'Cerebral blood flow (ASL)'),
    ('norgaard2021', 'flumazenil', 'GABAa receptor (flumazenil)'),
    ('smart2019', 'abp688', 'mGluR5 (ABP688, Smart)'),
]

ref_maps = {}
for source, desc, label in mni_annotations:
    try:
        img_path = fetch_annotation(source=source, desc=desc, space='MNI152')
        ref_maps[label] = img_path
        print(f"  Fetched: {label}")
    except Exception as e:
        print(f"  SKIP {label}: {e}")

print(f"\n  Total reference maps: {len(ref_maps)}")

# ===================================================================
# PART 4: Parcellate reference maps with Schaefer 400
# ===================================================================
print("\n[PART 4] Parcellating reference maps (Schaefer 400)")
print("-" * 50)

from nilearn.maskers import NiftiLabelsMasker

# Simple masker for reference maps (no temporal processing)
ref_masker = NiftiLabelsMasker(
    labels_img=schaefer400['maps'],
    resampling_target='labels',
    strategy='mean',
)
ref_masker.fit()

ref_parcellated = {}
for label, img_path in ref_maps.items():
    try:
        # Load and parcellate
        vals = ref_masker.transform(img_path)  # (1, 400)
        parcel_vals = vals.squeeze()
        # Check for valid data
        n_nonzero = np.sum(parcel_vals != 0)
        if n_nonzero > 50:  # at least 50 parcels have data
            ref_parcellated[label] = parcel_vals
            print(f"  {label}: {n_nonzero}/400 parcels, "
                  f"range [{parcel_vals.min():.3f}, {parcel_vals.max():.3f}]")
        else:
            print(f"  SKIP {label}: only {n_nonzero} non-zero parcels")
    except Exception as e:
        print(f"  SKIP {label}: {e}")

print(f"\n  Parcellated maps: {len(ref_parcellated)}")

# ===================================================================
# PART 5: Correlate FC gradients with reference maps
# ===================================================================
print("\n[PART 5] FC gradient vs neuromaps correlations (Schaefer 400)")
print("-" * 50)

from scipy.stats import pearsonr, spearmanr

# Get Schaefer 400 gradients — may have fewer ROIs than 400 due to FOV
n_grad_rois = results['Schaefer400']['n_rois']
grad_dm = np.array(results['Schaefer400']['eigvecs_dm'])  # (n_grad_rois, 10)
grad_le = np.array(results['Schaefer400']['eigvecs_le'])  # (n_grad_rois, 10)

# The masker reports which labels it kept — align reference maps
kept_labels = masker_s400.labels_  # labels actually present in the data
# ref_parcellated has 400 values (indexed 0..399 for parcels 1..400)
# We need to select only the parcels that survived resampling
# Schaefer labels are 1-indexed; masker labels_ tells us which were kept
parcel_indices = np.array(kept_labels).astype(int) - 1  # 0-indexed

print(f"\n  Gradient ROIs: {n_grad_rois} (of 400 Schaefer parcels)")
print(f"\n  {'Reference Map':<40} {'Grad1 r':>8} {'Grad1 p':>10} "
      f"{'Grad2 r':>8} {'Grad2 p':>10} {'Grad3 r':>8}")
print("  " + "-" * 86)

correlations = {}
for label, ref_vals in ref_parcellated.items():
    # Select only parcels that exist in our gradient data
    ref_aligned = ref_vals[parcel_indices] if len(parcel_indices) == n_grad_rois else ref_vals[:n_grad_rois]

    # Mask NaN/zero
    valid = ~np.isnan(ref_aligned) & (ref_aligned != 0)
    if valid.sum() < 50:
        continue

    row = {'label': label}
    for gi, gname in enumerate(['grad1', 'grad2', 'grad3']):
        g = grad_dm[valid, gi]
        r_val = ref_aligned[valid]
        r, p = spearmanr(g, r_val)
        row[f'{gname}_r'] = r
        row[f'{gname}_p'] = p

    correlations[label] = row
    print(f"  {label:<40} {row['grad1_r']:>8.3f} {row['grad1_p']:>10.4f} "
          f"{row['grad2_r']:>8.3f} {row['grad2_p']:>10.4f} {row['grad3_r']:>8.3f}")

# ===================================================================
# PART 6: Cross-atlas gradient comparison
# ===================================================================
print(f"\n[PART 6] Cross-atlas gradient comparison")
print("-" * 50)

# Compare gradient 1 across parcellations
# For atlases with different numbers of parcels, compare spectral features
for name, res in results.items():
    sf = res['spectral_features']
    print(f"  {name:<16} ({res['n_rois']:>3} ROIs): "
          f"Q={float(res['modularity']):.4f}, "
          f"lambda2={float(sf[-2]):.4f}, "
          f"gap/radius={float(sf[-2]/sf[-1]):.4f}")

# ===================================================================
# PART 7: Dynamic FC with Schaefer 100 (manageable size)
# ===================================================================
print(f"\n[PART 7] Dynamic FC analysis (Schaefer 100)")
print("-" * 50)

ts_s100_jax = jnp.array(ts_s100.T, dtype=jnp.float32)

# Sliding window
t0 = time.time()
dfc = joc.sliding_window_corr(ts_s100_jax, window_size=30)
print(f"  Dynamic FC: {dfc.shape} (windows x ROIs x ROIs), time={time.time()-t0:.1f}s")

# FC temporal variability per edge
fc_std = jnp.std(dfc, axis=0)
print(f"  FC variability: mean={float(jnp.mean(fc_std)):.4f}, max={float(jnp.max(fc_std)):.4f}")

# Modularity over time
W_s100 = jnp.abs(joc.corr(ts_s100_jax)) * (1 - jnp.eye(100))
_, eigvecs_s100 = joc.laplacian_eigenmaps(W_s100, k=5)
C_s100 = jax.nn.softmax(eigvecs_s100[:, :5] * 10, axis=-1)

def _window_modularity(i):
    fc_w = dfc[i]
    W_w = jnp.abs(fc_w) * (1 - jnp.eye(100))
    return joc.relaxed_modularity(W_w, C_s100)

Q_over_time = jax.vmap(_window_modularity)(jnp.arange(dfc.shape[0]))
print(f"  Modularity over time: mean={float(jnp.mean(Q_over_time)):.4f}, "
      f"std={float(jnp.std(Q_over_time)):.4f}, "
      f"range=[{float(jnp.min(Q_over_time)):.4f}, {float(jnp.max(Q_over_time)):.4f}]")

# ===================================================================
# PART 8: Bayesian FC with Schaefer 100
# ===================================================================
print(f"\n[PART 8] Bayesian variance-aware FC (Schaefer 100)")
print("-" * 50)

from jaxoccoli.bayesian_beta import make_conjugate_glm_vmap

# Design matrix: cosine basis (DCT drift) + constant
T_pts = ts_s100.shape[0]
n_cosines = 5
t_vec = jnp.linspace(0, 1, T_pts)
X_design = jnp.column_stack([
    jnp.ones(T_pts),
    *[jnp.cos(2 * jnp.pi * k * t_vec) for k in range(1, n_cosines + 1)],
    *[jnp.sin(2 * jnp.pi * k * t_vec) for k in range(1, n_cosines + 1)],
])
print(f"  Design matrix: {X_design.shape} (DCT basis, {2*n_cosines+1} regressors)")

t0 = time.time()
params_glm, fwd_glm = make_conjugate_glm_vmap(X_design)
bm, bv, s2 = fwd_glm(params_glm, ts_s100_jax)
print(f"  Conjugate GLM: beta_mean={bm.shape}, sigma2 range=[{float(s2.min()):.3f}, {float(s2.max()):.3f}]")

# Standard FC vs posterior-corrected FC
fc_standard = joc.corr(ts_s100_jax)
fc_posterior = joc.posterior_corr(bm, bv)

# How different are they?
mask = 1 - jnp.eye(100)
diff = jnp.abs(np.array(fc_posterior) - np.array(fc_standard)) * np.array(mask)
print(f"  FC difference (standard vs posterior): "
      f"mean={float(jnp.mean(diff)):.4f}, max={float(jnp.max(diff)):.4f}")
print(f"  time={time.time()-t0:.1f}s")

# ===================================================================
# PART 9: MELODIC dimensionality comparison
# ===================================================================
print(f"\n[PART 9] Dimensionality estimation comparison")
print("-" * 50)

MELODIC_DIR = f"{WAND}/derivatives/fsl-fmri/{SUB}/{SES}/task-rest_melodic"
melodic_icstats = np.loadtxt(f"{MELODIC_DIR}/melodic_ICstats")
melodic_pcaE = np.loadtxt(f"{MELODIC_DIR}/melodic_pcaE")
n_melodic_ics = melodic_icstats.shape[0]

# Our eigenspectrum analysis per atlas
print(f"  MELODIC ICA dimensionality: {n_melodic_ics} components")
print(f"  MELODIC PCA eigenvalues: {melodic_pcaE.shape[0]} (one per timepoint)")
print()

for name, ts in [("Schaefer400", ts_s400), ("Schaefer100", ts_s100),
                  ("HarvardOxford", ts_ho), ("DiFuMo64", ts_di)]:
    # SVD eigenspectrum
    ts_centered = ts - ts.mean(axis=0)
    U, s, Vt = np.linalg.svd(ts_centered, full_matrices=False)
    n_sv = len(s)
    var_explained = (s ** 2) / np.sum(s ** 2)
    cum_var = np.cumsum(var_explained)

    # Broken stick model (use n_sv components, not n_rois)
    broken_stick = np.zeros(n_sv)
    for i in range(n_sv):
        broken_stick[i] = np.sum(1.0 / np.arange(i + 1, n_sv + 1)) / n_sv
    n_bs = np.sum(var_explained > broken_stick)

    # 90% variance
    n_90 = int(np.searchsorted(cum_var, 0.9)) + 1

    # Parallel analysis (simplified)
    n_perm = 20
    null_eigvals = np.zeros((n_perm, min(ts.shape)))
    for p in range(n_perm):
        ts_null = np.copy(ts_centered)
        for col in range(ts_null.shape[1]):
            np.random.shuffle(ts_null[:, col])
        _, s_null, _ = np.linalg.svd(ts_null, full_matrices=False)
        null_eigvals[p] = s_null ** 2
    null_95 = np.percentile(null_eigvals, 95, axis=0)
    actual_eigvals = s ** 2
    n_pa = int(np.sum(actual_eigvals > null_95))

    print(f"  {name:<16} ({n_rois:>3} ROIs): "
          f"broken_stick={n_bs}, parallel_analysis={n_pa}, "
          f"90%_var={n_90}, top_eigval={var_explained[0]:.3f}")

# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
Atlases used:
  Schaefer 400 (Yeo 7 networks) -- primary for neuromaps correlation
  Schaefer 100 (Yeo 7 networks) -- dynamic FC and Bayesian analysis
  Harvard-Oxford cortical ({n_ho} ROIs) -- anatomical reference
  DiFuMo 64 -- data-driven functional modes

Neuromaps reference maps parcellated: {len(ref_parcellated)}
  (PET receptors, metabolism, cognition -- all in MNI152 space)

Key FC gradient correlations (Spearman rho, Schaefer 400):""")

# Sort by absolute gradient 1 correlation
if correlations:
    sorted_corrs = sorted(correlations.values(), key=lambda x: abs(x.get('grad1_r', 0)), reverse=True)
    for row in sorted_corrs[:10]:
        sig = "***" if row['grad1_p'] < 0.001 else "**" if row['grad1_p'] < 0.01 else "*" if row['grad1_p'] < 0.05 else ""
        print(f"  {row['label']:<40} rho={row['grad1_r']:>7.3f} {sig}")

print(f"""
MELODIC dimensionality: {n_melodic_ics} ICs
Fieldmap correction: NOT applied (raw EPI with motion correction only)
""")
