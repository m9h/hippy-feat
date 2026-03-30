#!/usr/bin/env python
"""WAND FC gradients vs surface-space neuromaps (myelin, thickness, SA axis, etc.)

Requires: wb_command (Connectome Workbench 2.1.0)
Uses distortion-corrected BOLD from fieldmap_correction pipeline.
"""

import sys, os
sys.path.insert(0, "/Users/mhough/dev/hippy-feat")
os.environ['PATH'] = '/Applications/wb_view.app/Contents/usr/bin:' + os.environ['PATH']

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import time
import numpy as np
import nibabel as nib
import jax
import jax.numpy as jnp
from scipy.stats import spearmanr

import jaxoccoli as joc

# Monkey-patch neuromaps for nilearn compat
import neuromaps.datasets.annotations as _nma
_orig = _nma._fetch_file
def _patched(url, data_dir, **kw):
    from pathlib import Path
    return _orig(url, Path(data_dir) if isinstance(data_dir, str) else data_dir, **kw)
_nma._fetch_file = _patched

from neuromaps.datasets import fetch_annotation
from neuromaps.parcellate import Parcellater
from neuromaps.images import load_data

# ---------------------------------------------------------------------------
WAND = "/Users/mhough/dev/wand"
SUB = "sub-08033"
SES = "ses-03"
TR = 2.0

# Use distortion-corrected BOLD
BOLD_DC = f"{WAND}/derivatives/fsl-fmri/{SUB}/{SES}/fieldmap_correction/task-rest_mc_dc.nii.gz"
BOLD_MC = f"{WAND}/derivatives/fsl-fmri/{SUB}/{SES}/mc/task-rest_mc.nii.gz"

# Use corrected if available, otherwise uncorrected
BOLD = BOLD_DC if os.path.exists(BOLD_DC) else BOLD_MC
corrected = "CORRECTED" if BOLD == BOLD_DC else "UNCORRECTED"

print("=" * 70)
print(f"WAND Surface Neuromaps Analysis ({corrected})")
print("=" * 70)

# ===================================================================
# 1. Parcellate BOLD with Schaefer 400
# ===================================================================
print("\n[1] Parcellating BOLD with Schaefer 400...")
t0 = time.time()

from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets

schaefer400 = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
masker = NiftiLabelsMasker(
    labels_img=schaefer400['maps'],
    standardize='zscore_sample',
    resampling_target='data',
    t_r=TR, low_pass=0.1, high_pass=0.01, detrend=True,
)
ts = masker.fit_transform(BOLD)
n_rois = ts.shape[1]
print(f"  BOLD: {corrected}, {ts.shape} ({n_rois} parcels)")
print(f"  Time: {time.time()-t0:.1f}s")

# ===================================================================
# 2. FC + gradients with jaxoccoli
# ===================================================================
print("\n[2] Computing FC and gradients...")
t0 = time.time()

ts_jax = jnp.array(ts.T, dtype=jnp.float32)
fc = joc.corr(ts_jax)
W = jnp.abs(fc) * (1 - jnp.eye(n_rois))
eigvals_dm, eigvecs_dm = joc.diffusion_mapping(W, k=10, alpha=0.5)
eigvals_le, eigvecs_le = joc.laplacian_eigenmaps(W, k=10)
sf = joc.spectral_features(W, k=10)

print(f"  Diffusion map eigenvalues: {eigvals_dm[:5]}")
print(f"  Spectral gap: {float(sf[-2]):.4f}")
print(f"  Time: {time.time()-t0:.1f}s")

# ===================================================================
# 3. Fetch ALL surface neuromaps annotations
# ===================================================================
print("\n[3] Fetching surface neuromaps annotations...")

surface_annotations = [
    # Structural
    ('hcps1200', 'myelinmap', 'HCP Myelin (T1w/T2w)'),
    ('hcps1200', 'thickness', 'HCP Cortical Thickness'),
    # FC reference gradients
    ('margulies2016', 'fcgradient01', 'Margulies FC Gradient 1'),
    ('margulies2016', 'fcgradient02', 'Margulies FC Gradient 2'),
    ('margulies2016', 'fcgradient03', 'Margulies FC Gradient 3'),
    # Axes
    ('sydnor2021', 'SAaxis', 'Sensorimotor-Association Axis'),
    # Evolutionary / developmental
    ('hill2010', 'devexp', 'Developmental Expansion'),
    ('hill2010', 'evoexp', 'Evolutionary Expansion'),
    # Functional
    ('xu2020', 'FChomology', 'FC Homology'),
    ('xu2020', 'evoexp', 'Evolutionary Expansion (Xu)'),
    ('mueller2013', 'intersubjvar', 'Inter-subject Variability'),
    # Metabolism
    ('raichle', 'cbf', 'Cerebral Blood Flow (Raichle)'),
    ('raichle', 'cmrglc', 'Glucose Metabolism (Raichle)'),
    ('raichle', 'cmr02', 'Oxygen Metabolism (Raichle)'),
    # MEG
    ('hcps1200', 'megalpha', 'MEG Alpha Power'),
    ('hcps1200', 'megbeta', 'MEG Beta Power'),
    ('hcps1200', 'megdelta', 'MEG Delta Power'),
    ('hcps1200', 'megtheta', 'MEG Theta Power'),
    ('hcps1200', 'megtimescale', 'MEG Intrinsic Timescale'),
]

# Also fetch MNI152 PET maps
mni_annotations = [
    ('neurosynth', 'cogpc1', 'Neurosynth Cognitive PC1'),
    ('beliveau2017', 'cumi101', '5-HT1A (CUMI-101)'),
    ('beliveau2017', 'cimbi36', '5-HT2A (Cimbi-36)'),
    ('beliveau2017', 'dasb', 'SERT (DASB)'),
    ('finnema2016', 'ucbj', 'Synaptic Density SV2A'),
    ('kaller2017', 'sch23390', 'D1 (SCH23390)'),
    ('kantonen2020', 'carfentanil', 'Mu-Opioid (Carfentanil)'),
    ('norgaard2021', 'flumazenil', 'GABAa (Flumazenil)'),
    ('gallezot2017', 'gsk189254', 'Histamine H3'),
]

ref_data = {}

# Fetch surface maps
for source, desc, label in surface_annotations:
    try:
        files = fetch_annotation(source=source, desc=desc)
        ref_data[label] = {'files': files, 'space': 'fsLR', 'type': 'surface'}
        print(f"  OK: {label}")
    except Exception as e:
        print(f"  SKIP: {label} ({e})")

# Fetch MNI152 maps
for source, desc, label in mni_annotations:
    try:
        files = fetch_annotation(source=source, desc=desc, space='MNI152')
        ref_data[label] = {'files': files, 'space': 'MNI152', 'type': 'volume'}
        print(f"  OK: {label}")
    except Exception as e:
        print(f"  SKIP: {label} ({e})")

print(f"\n  Total: {len(ref_data)} reference maps")

# ===================================================================
# 4. Parcellate all reference maps to Schaefer 400
# ===================================================================
print("\n[4] Parcellating reference maps...")

# For MNI152 volumetric maps: use nilearn NiftiLabelsMasker
ref_masker_vol = NiftiLabelsMasker(
    labels_img=schaefer400['maps'],
    resampling_target='labels',
    strategy='mean',
)
ref_masker_vol.fit()

# For surface maps: manual parcellation using Schaefer GIFTI labels
# (bypasses neuromaps Parcellater which has a bug in 0.0.5 with newer nibabel)
ATLAS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'atlases')
schaefer_lh_path = os.path.join(ATLAS_DIR, 'Schaefer400_7Net_LH.label.gii')
schaefer_rh_path = os.path.join(ATLAS_DIR, 'Schaefer400_7Net_RH.label.gii')

has_surf_parc = False
surf_labels = None
if os.path.exists(schaefer_lh_path) and os.path.exists(schaefer_rh_path):
    lh_gii = nib.load(schaefer_lh_path)
    rh_gii = nib.load(schaefer_rh_path)
    lh_labels = lh_gii.darrays[0].data  # (32492,) int labels
    rh_labels = rh_gii.darrays[0].data  # (32492,) int labels
    # Offset RH labels so they don't overlap with LH
    rh_offset = int(lh_labels.max())
    rh_labels_offset = np.where(rh_labels > 0, rh_labels + rh_offset, 0)
    surf_labels = np.concatenate([lh_labels, rh_labels_offset])
    unique_labels = np.unique(surf_labels[surf_labels > 0])
    has_surf_parc = True
    print(f"  Surface parcellation: {len(unique_labels)} Schaefer parcels on {len(surf_labels)} vertices")
else:
    print(f"  Schaefer GIFTI files not found at {ATLAS_DIR}")

def parcellate_surface(data_files):
    """Manually parcellate surface GIFTI data with Schaefer labels."""
    lh_data = nib.load(data_files[0]).darrays[0].data  # (32492,)
    rh_data = nib.load(data_files[1]).darrays[0].data  # (32492,)
    vertex_data = np.concatenate([lh_data, rh_data])
    vals = np.zeros(len(unique_labels))
    for i, lab in enumerate(unique_labels):
        mask = surf_labels == lab
        if mask.sum() > 0:
            vals[i] = np.nanmean(vertex_data[mask])
    return vals

parcellated = {}
kept_labels = masker.labels_
parcel_indices = np.array(kept_labels).astype(int) - 1

for label, info in ref_data.items():
    try:
        if info['type'] == 'volume':
            vals = ref_masker_vol.transform(info['files']).squeeze()
            n_valid = np.sum((vals != 0) & ~np.isnan(vals))
            if n_valid > 50:
                parcellated[label] = vals
                print(f"  {label}: {n_valid}/{len(vals)} valid parcels")
        elif info['type'] == 'surface' and has_surf_parc:
            files = info['files']
            if isinstance(files, (str, os.PathLike)) or len(files) < 2:
                print(f"  SKIP {label}: need L/R pair")
                continue
            vals = parcellate_surface(files)
            n_valid = np.sum((vals != 0) & ~np.isnan(vals))
            if n_valid > 50:
                parcellated[label] = vals
                print(f"  {label}: {n_valid}/{len(vals)} valid parcels (surface)")
            else:
                print(f"  SKIP {label}: only {n_valid} valid parcels")
        else:
            print(f"  SKIP {label}: no surface parcellater")
    except Exception as e:
        print(f"  SKIP {label}: {e}")

print(f"\n  Parcellated: {len(parcellated)} maps")

# ===================================================================
# 5. Correlate gradients with ALL reference maps
# ===================================================================
print("\n[5] FC gradient vs reference map correlations")
print("=" * 90)

grad = np.array(eigvecs_dm)  # (n_rois, 10)

print(f"\n  {'Reference Map':<40} {'Grad1 rho':>10} {'p':>10} "
      f"{'Grad2 rho':>10} {'p':>10}")
print("  " + "-" * 84)

all_corrs = []
for label, vals in sorted(parcellated.items()):
    # Align parcels: ref has 400 values, gradients have n_rois (possibly 399)
    # parcel_indices maps from masker label IDs to 0-indexed positions
    if len(vals) > n_rois:
        # nilearn masker labels_ are 1-indexed parcel IDs
        # We need to select only parcels that exist in the BOLD data
        keep = np.array(kept_labels).astype(int) - 1  # 0-indexed into 400-element array
        keep = keep[keep < len(vals)]
        vals = vals[keep]
    if len(vals) != n_rois:
        # Last resort: truncate
        vals = vals[:n_rois]

    valid = ~np.isnan(vals) & (vals != 0)
    if valid.sum() < 50:
        continue

    g1 = grad[valid, 0]
    g2 = grad[valid, 1]
    rv = vals[valid]

    r1, p1 = spearmanr(g1, rv)
    r2, p2 = spearmanr(g2, rv)

    sig1 = "***" if p1 < 0.001 else "**" if p1 < 0.01 else "*" if p1 < 0.05 else ""
    sig2 = "***" if p2 < 0.001 else "**" if p2 < 0.01 else "*" if p2 < 0.05 else ""

    all_corrs.append({'label': label, 'r1': r1, 'p1': p1, 'r2': r2, 'p2': p2})
    print(f"  {label:<40} {r1:>9.3f}{sig1:<1} {p1:>10.4f} "
          f"{r2:>9.3f}{sig2:<1} {p2:>10.4f}")

# ===================================================================
# 6. Compare corrected vs uncorrected FC (if both exist)
# ===================================================================
if os.path.exists(BOLD_DC) and os.path.exists(BOLD_MC):
    print(f"\n[6] Corrected vs uncorrected FC comparison")
    print("-" * 50)

    ts_uc = masker.fit_transform(BOLD_MC)
    ts_uc_jax = jnp.array(ts_uc.T, dtype=jnp.float32)
    fc_uc = joc.corr(ts_uc_jax)

    mask = 1 - jnp.eye(n_rois)
    fc_diff = jnp.abs(np.array(fc) - np.array(fc_uc)) * np.array(mask)
    print(f"  FC difference (corrected vs uncorrected):")
    print(f"    Mean: {float(jnp.mean(fc_diff)):.4f}")
    print(f"    Max:  {float(jnp.max(fc_diff)):.4f}")
    print(f"    Correlation: r={float(np.corrcoef(fc[np.triu_indices(n_rois, k=1)], fc_uc[np.triu_indices(n_rois, k=1)])[0,1]):.6f}")

    # Compare gradient correlations
    W_uc = jnp.abs(fc_uc) * (1 - jnp.eye(n_rois))
    _, grad_uc = joc.diffusion_mapping(W_uc, k=10, alpha=0.5)
    grad_uc = np.array(grad_uc)

    # Procrustes-like alignment (sign flip)
    for i in range(min(5, grad.shape[1])):
        if np.corrcoef(grad[:, i], grad_uc[:, i])[0, 1] < 0:
            grad_uc[:, i] *= -1

    print(f"  Gradient alignment (corrected vs uncorrected):")
    for i in range(5):
        r = np.corrcoef(grad[:, i], grad_uc[:, i])[0, 1]
        print(f"    Gradient {i+1}: r={r:.4f}")

# ===================================================================
# Summary
# ===================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\n  BOLD: {corrected}")
print(f"  Parcellation: Schaefer 400 (Yeo 7)")
print(f"  Total reference maps: {len(parcellated)}")

# Top correlations sorted by |rho| for gradient 1
print(f"\n  Top gradient 1 correlations (|rho|):")
sorted_corrs = sorted(all_corrs, key=lambda x: abs(x['r1']), reverse=True)
for row in sorted_corrs[:15]:
    sig = "***" if row['p1'] < 0.001 else "**" if row['p1'] < 0.01 else "*" if row['p1'] < 0.05 else ""
    print(f"    {row['label']:<40} rho={row['r1']:>7.3f} {sig}")

print(f"\n  Top gradient 2 correlations (|rho|):")
sorted_corrs2 = sorted(all_corrs, key=lambda x: abs(x['r2']), reverse=True)
for row in sorted_corrs2[:10]:
    sig = "***" if row['p2'] < 0.001 else "**" if row['p2'] < 0.01 else "*" if row['p2'] < 0.05 else ""
    print(f"    {row['label']:<40} rho={row['r2']:>7.3f} {sig}")
