#!/usr/bin/env python
"""Generate publication-quality figures for WAND resting-state FC analysis.

Outputs to figures/ directory:
  fig1_fc_gradients_surface.png    - Gradient 1-3 on cortical surface
  fig2_neuromaps_correlations.png  - Bar chart of gradient vs reference maps
  fig3_fc_matrix_comparison.png    - FC matrices (corr, partial corr, dynamic)
  fig4_spectral_embedding.png     - Diffusion map scatter + eigenspectrum
  fig5_fieldmap_correction.png    - Corrected vs uncorrected gradient comparison
  fig6_receptor_gradient_scatter.png - Scatter plots of top receptor correlations
"""

import sys, os
sys.path.insert(0, "/Users/mhough/dev/hippy-feat")
os.environ['PATH'] = '/Applications/wb_view.app/Contents/usr/bin:' + os.environ['PATH']
os.environ['MPLBACKEND'] = 'Agg'

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import nibabel as nib
import jax
import jax.numpy as jnp
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import matplotlib.gridspec as gridspec

import jaxoccoli as joc

# Monkey-patch neuromaps
import neuromaps.datasets.annotations as _nma
_orig = _nma._fetch_file
def _patched(url, data_dir, **kw):
    from pathlib import Path
    return _orig(url, Path(data_dir) if isinstance(data_dir, str) else data_dir, **kw)
_nma._fetch_file = _patched

from neuromaps.datasets import fetch_annotation
from nilearn.maskers import NiftiLabelsMasker
from nilearn import datasets, plotting

FIGDIR = "/Users/mhough/dev/hippy-feat/figures"
WAND = "/Users/mhough/dev/wand"
SUB = "sub-08033"
SES = "ses-03"
BOLD_DC = f"{WAND}/derivatives/fsl-fmri/{SUB}/{SES}/fieldmap_correction/task-rest_mc_dc.nii.gz"
BOLD_MC = f"{WAND}/derivatives/fsl-fmri/{SUB}/{SES}/mc/task-rest_mc.nii.gz"
BOLD = BOLD_DC if os.path.exists(BOLD_DC) else BOLD_MC
TR = 2.0

ATLAS_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'atlases')

# Style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# ===================================================================
# Load and compute
# ===================================================================
print("Loading data and computing gradients...")

schaefer400 = datasets.fetch_atlas_schaefer_2018(n_rois=400, yeo_networks=7, resolution_mm=2)
masker = NiftiLabelsMasker(
    labels_img=schaefer400['maps'], standardize='zscore_sample',
    resampling_target='data', t_r=TR, low_pass=0.1, high_pass=0.01, detrend=True)
ts = masker.fit_transform(BOLD)
n_rois = ts.shape[1]
kept_labels = masker.labels_

ts_jax = jnp.array(ts.T, dtype=jnp.float32)
fc = joc.corr(ts_jax)
fc_partial = joc.partial_corr(ts_jax, l2=0.1)
W = jnp.abs(fc) * (1 - jnp.eye(n_rois))
eigvals_dm, eigvecs_dm = joc.diffusion_mapping(W, k=10, alpha=0.5)
eigvals_le, eigvecs_le = joc.laplacian_eigenmaps(W, k=10)
sf = joc.spectral_features(W, k=10)

grad = np.array(eigvecs_dm)

# Also load uncorrected for comparison
if os.path.exists(BOLD_DC) and os.path.exists(BOLD_MC):
    ts_uc = masker.fit_transform(BOLD_MC)
    ts_uc_jax = jnp.array(ts_uc.T, dtype=jnp.float32)
    fc_uc = joc.corr(ts_uc_jax)
    W_uc = jnp.abs(fc_uc) * (1 - jnp.eye(n_rois))
    _, grad_uc = joc.diffusion_mapping(W_uc, k=10, alpha=0.5)
    grad_uc = np.array(grad_uc)
    for i in range(5):
        if np.corrcoef(grad[:, i], grad_uc[:, i])[0, 1] < 0:
            grad_uc[:, i] *= -1
    has_comparison = True
else:
    has_comparison = False

# Load surface parcellation labels
schaefer_lh = nib.load(os.path.join(ATLAS_DIR, 'Schaefer400_7Net_LH.label.gii'))
schaefer_rh = nib.load(os.path.join(ATLAS_DIR, 'Schaefer400_7Net_RH.label.gii'))
lh_labels = schaefer_lh.darrays[0].data
rh_labels = schaefer_rh.darrays[0].data
rh_offset = int(lh_labels.max())
rh_labels_offset = np.where(rh_labels > 0, rh_labels + rh_offset, 0)
surf_labels = np.concatenate([lh_labels, rh_labels_offset])
unique_labels = np.unique(surf_labels[surf_labels > 0])

# Fetch neuromaps reference maps
print("Fetching neuromaps references...")
surface_maps = {}
for source, desc, label in [
    ('hcps1200', 'myelinmap', 'HCP Myelin'),
    ('hcps1200', 'thickness', 'Cortical Thickness'),
    ('margulies2016', 'fcgradient01', 'Margulies Grad 1'),
    ('sydnor2021', 'SAaxis', 'SA Axis'),
]:
    try:
        files = fetch_annotation(source=source, desc=desc)
        lh_data = nib.load(files[0]).darrays[0].data
        rh_data = nib.load(files[1]).darrays[0].data
        vertex_data = np.concatenate([lh_data, rh_data])
        vals = np.zeros(len(unique_labels))
        for i, lab in enumerate(unique_labels):
            mask = surf_labels == lab
            if mask.sum() > 0:
                vals[i] = np.nanmean(vertex_data[mask])
        surface_maps[label] = vals
    except Exception as e:
        print(f"  Skip {label}: {e}")

pet_maps = {}
ref_masker_vol = NiftiLabelsMasker(labels_img=schaefer400['maps'],
                                    resampling_target='labels', strategy='mean')
ref_masker_vol.fit()
for source, desc, label in [
    ('neurosynth', 'cogpc1', 'Neurosynth Cog PC1'),
    ('beliveau2017', 'cumi101', '5-HT1A'),
    ('beliveau2017', 'cimbi36', '5-HT2A'),
    ('beliveau2017', 'dasb', 'SERT'),
    ('kaller2017', 'sch23390', 'D1'),
    ('kantonen2020', 'carfentanil', 'Mu-Opioid'),
    ('gallezot2017', 'gsk189254', 'H3'),
    ('norgaard2021', 'flumazenil', 'GABAa'),
]:
    try:
        f = fetch_annotation(source=source, desc=desc, space='MNI152')
        vals = ref_masker_vol.transform(f).squeeze()
        pet_maps[label] = vals
    except Exception as e:
        print(f"  Skip {label}: {e}")

all_maps = {**surface_maps, **pet_maps}

# Align all maps to n_rois
keep = np.array(kept_labels).astype(int) - 1
for k in list(all_maps.keys()):
    v = all_maps[k]
    if len(v) > n_rois:
        all_maps[k] = v[keep[:n_rois]]

print(f"Loaded {len(all_maps)} reference maps")

# ===================================================================
# Figure 1: FC Gradients projected back to Schaefer volume
# ===================================================================
print("Generating Figure 1: FC gradients on brain...")

fig, axes = plt.subplots(3, 2, figsize=(10, 9))
for gi, gname in enumerate(['Gradient 1', 'Gradient 2', 'Gradient 3']):
    g_vals = grad[:, gi]
    # Map gradient values back to Schaefer atlas volume
    atlas_img = nib.load(schaefer400['maps'])
    atlas_data = atlas_img.get_fdata()
    grad_vol = np.zeros_like(atlas_data, dtype=np.float32)
    for pi, label_id in enumerate(kept_labels):
        grad_vol[atlas_data == label_id] = g_vals[pi] if pi < len(g_vals) else 0
    grad_nii = nib.Nifti1Image(grad_vol, atlas_img.affine)

    # Lateral and medial views
    for vi, (display_mode, cut_coords) in enumerate([('x', [-40, 40]), ('z', [5, 30])]):
        ax = axes[gi, vi]
        plotting.plot_stat_map(
            grad_nii, display_mode=display_mode, cut_coords=cut_coords,
            cmap='RdBu_r', colorbar=True, axes=ax,
            title=f'{gname} ({"lateral" if vi==0 else "axial"})',
            symmetric_cbar=True, annotate=False,
        )

fig.suptitle(f'WAND {SUB} Resting-State FC Gradients (Schaefer 400, distortion-corrected)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig1_fc_gradients_brain.png')
plt.close()
print(f"  -> {FIGDIR}/fig1_fc_gradients_brain.png")

# ===================================================================
# Figure 2: Neuromaps correlation bar chart
# ===================================================================
print("Generating Figure 2: Neuromaps correlation bar chart...")

corr_results = []
for label, vals in sorted(all_maps.items()):
    valid = ~np.isnan(vals) & (vals != 0)
    if valid.sum() < 50:
        continue
    r1, p1 = spearmanr(grad[valid, 0], vals[valid])
    r2, p2 = spearmanr(grad[valid, 1], vals[valid])
    corr_results.append({'label': label, 'r1': r1, 'p1': p1, 'r2': r2, 'p2': p2})

corr_results.sort(key=lambda x: abs(x['r1']), reverse=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gradient 1
labels_g1 = [r['label'] for r in corr_results]
rhos_g1 = [r['r1'] for r in corr_results]
colors_g1 = ['#d32f2f' if r['p1'] < 0.001 else '#ff9800' if r['p1'] < 0.01
              else '#ffeb3b' if r['p1'] < 0.05 else '#9e9e9e' for r in corr_results]
bars1 = ax1.barh(range(len(labels_g1)), rhos_g1, color=colors_g1, edgecolor='black', linewidth=0.5)
ax1.set_yticks(range(len(labels_g1)))
ax1.set_yticklabels(labels_g1, fontsize=8)
ax1.set_xlabel('Spearman rho')
ax1.set_title('Gradient 1 vs Reference Maps')
ax1.axvline(0, color='black', linewidth=0.5)
ax1.set_xlim(-0.5, 0.5)
ax1.invert_yaxis()

# Gradient 2
corr_results_g2 = sorted(corr_results, key=lambda x: abs(x['r2']), reverse=True)
labels_g2 = [r['label'] for r in corr_results_g2]
rhos_g2 = [r['r2'] for r in corr_results_g2]
colors_g2 = ['#d32f2f' if r['p2'] < 0.001 else '#ff9800' if r['p2'] < 0.01
              else '#ffeb3b' if r['p2'] < 0.05 else '#9e9e9e' for r in corr_results_g2]
bars2 = ax2.barh(range(len(labels_g2)), rhos_g2, color=colors_g2, edgecolor='black', linewidth=0.5)
ax2.set_yticks(range(len(labels_g2)))
ax2.set_yticklabels(labels_g2, fontsize=8)
ax2.set_xlabel('Spearman rho')
ax2.set_title('Gradient 2 vs Reference Maps')
ax2.axvline(0, color='black', linewidth=0.5)
ax2.set_xlim(-0.5, 0.5)
ax2.invert_yaxis()

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#d32f2f', label='p < 0.001'),
                   Patch(facecolor='#ff9800', label='p < 0.01'),
                   Patch(facecolor='#ffeb3b', label='p < 0.05'),
                   Patch(facecolor='#9e9e9e', label='n.s.')]
fig.legend(handles=legend_elements, loc='lower center', ncol=4, fontsize=9)

fig.suptitle('FC Gradient Correlations with Neuromaps Reference Maps (Spearman, uncorrected)',
             fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.05, 1, 0.95])
fig.savefig(f'{FIGDIR}/fig2_neuromaps_correlations.png')
plt.close()
print(f"  -> {FIGDIR}/fig2_neuromaps_correlations.png")

# ===================================================================
# Figure 3: FC matrices (full, partial, difference)
# ===================================================================
print("Generating Figure 3: FC matrices...")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

# Full correlation
im0 = axes[0].imshow(np.array(fc), cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
axes[0].set_title('Full Correlation')
axes[0].set_xlabel('Parcel')
axes[0].set_ylabel('Parcel')
plt.colorbar(im0, ax=axes[0], shrink=0.8)

# Partial correlation
im1 = axes[1].imshow(np.array(fc_partial), cmap='RdBu_r', vmin=-0.3, vmax=0.3, aspect='auto')
axes[1].set_title('Partial Correlation')
axes[1].set_xlabel('Parcel')
plt.colorbar(im1, ax=axes[1], shrink=0.8)

# Dynamic FC variability
dfc = joc.sliding_window_corr(ts_jax, window_size=30)
fc_std = np.array(jnp.std(dfc, axis=0))
im2 = axes[2].imshow(fc_std, cmap='hot', vmin=0, vmax=0.5, aspect='auto')
axes[2].set_title('Dynamic FC Variability (SD)')
axes[2].set_xlabel('Parcel')
plt.colorbar(im2, ax=axes[2], shrink=0.8)

fig.suptitle(f'Functional Connectivity Matrices ({n_rois} Schaefer Parcels)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig3_fc_matrices.png')
plt.close()
print(f"  -> {FIGDIR}/fig3_fc_matrices.png")

# ===================================================================
# Figure 4: Spectral embedding + eigenspectrum
# ===================================================================
print("Generating Figure 4: Spectral embedding...")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

# Yeo network colors for Schaefer parcels
network_names = [str(l) for l in schaefer400['labels'][1:]]  # skip background
network_colors = []
color_map = {
    'Vis': '#7B2D8E', 'SomMot': '#4682B4', 'DorsAttn': '#00A600',
    'SalVentAttn': '#C43AFA', 'Limbic': '#DCDC00', 'Cont': '#E69422',
    'Default': '#CD3E3E',
}
for name in network_names[:n_rois]:
    assigned = '#808080'
    for net, col in color_map.items():
        if net in name:
            assigned = col
            break
    network_colors.append(assigned)

# Diffusion map: Grad1 vs Grad2
axes[0].scatter(grad[:, 0], grad[:, 1], c=network_colors, s=8, alpha=0.7)
axes[0].set_xlabel('Gradient 1')
axes[0].set_ylabel('Gradient 2')
axes[0].set_title('Diffusion Map Embedding')

# Diffusion map: Grad1 vs Grad3
axes[1].scatter(grad[:, 0], grad[:, 2], c=network_colors, s=8, alpha=0.7)
axes[1].set_xlabel('Gradient 1')
axes[1].set_ylabel('Gradient 3')
axes[1].set_title('Diffusion Map (G1 vs G3)')

# Eigenspectrum
eigvals_all = np.array(eigvals_dm)
axes[2].bar(range(1, len(eigvals_all)+1), eigvals_all, color='steelblue', edgecolor='black', linewidth=0.5)
axes[2].set_xlabel('Component')
axes[2].set_ylabel('Eigenvalue')
axes[2].set_title('Diffusion Map Eigenspectrum')
axes[2].axhline(y=0, color='black', linewidth=0.5)

# Network legend
from matplotlib.lines import Line2D
legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=col,
                         markersize=6, label=net) for net, col in color_map.items()]
axes[0].legend(handles=legend_handles, fontsize=6, loc='upper left', framealpha=0.8)

fig.suptitle('Spectral Embedding of FC (Yeo 7 Network Coloring)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig4_spectral_embedding.png')
plt.close()
print(f"  -> {FIGDIR}/fig4_spectral_embedding.png")

# ===================================================================
# Figure 5: Fieldmap correction comparison
# ===================================================================
if has_comparison:
    print("Generating Figure 5: Fieldmap correction comparison...")

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    for gi in range(3):
        # Corrected gradient
        axes[0, gi].scatter(range(n_rois), grad[:, gi], c=network_colors, s=4, alpha=0.6)
        axes[0, gi].set_title(f'Gradient {gi+1} (corrected)')
        axes[0, gi].set_ylabel('Value')

        # Scatter: corrected vs uncorrected
        r = np.corrcoef(grad[:, gi], grad_uc[:, gi])[0, 1]
        axes[1, gi].scatter(grad_uc[:, gi], grad[:, gi], c=network_colors, s=4, alpha=0.6)
        lims = [min(grad[:, gi].min(), grad_uc[:, gi].min()),
                max(grad[:, gi].max(), grad_uc[:, gi].max())]
        axes[1, gi].plot(lims, lims, 'k--', linewidth=0.5, alpha=0.5)
        axes[1, gi].set_xlabel(f'Uncorrected G{gi+1}')
        axes[1, gi].set_ylabel(f'Corrected G{gi+1}')
        axes[1, gi].set_title(f'r = {r:.4f}')

    fig.suptitle('Effect of Fieldmap Distortion Correction on FC Gradients',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig.savefig(f'{FIGDIR}/fig5_fieldmap_correction.png')
    plt.close()
    print(f"  -> {FIGDIR}/fig5_fieldmap_correction.png")

# ===================================================================
# Figure 6: Top receptor-gradient scatter plots
# ===================================================================
print("Generating Figure 6: Receptor-gradient scatter plots...")

top_maps = [
    ('Margulies Grad 1', 0, 'Gradient 1'),
    ('Neurosynth Cog PC1', 0, 'Gradient 1'),
    ('HCP Myelin', 0, 'Gradient 1'),
    ('SA Axis', 0, 'Gradient 1'),
    ('5-HT2A', 0, 'Gradient 1'),
    ('H3', 1, 'Gradient 2'),
    ('Mu-Opioid', 1, 'Gradient 2'),
    ('SERT', 1, 'Gradient 2'),
    ('D1', 1, 'Gradient 2'),
]

n_plots = len(top_maps)
ncols = 3
nrows = (n_plots + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(12, nrows * 3.5))
axes = axes.flatten()

for idx, (map_name, gi, gname) in enumerate(top_maps):
    ax = axes[idx]
    if map_name not in all_maps:
        ax.set_visible(False)
        continue
    vals = all_maps[map_name]
    valid = ~np.isnan(vals) & (vals != 0)
    if valid.sum() < 50:
        ax.set_visible(False)
        continue
    g = grad[valid, gi]
    v = vals[valid]
    r, p = spearmanr(g, v)

    ax.scatter(v, g, c=[c for c, vl in zip(network_colors, valid) if vl],
               s=6, alpha=0.5)
    # Regression line
    z = np.polyfit(v, g, 1)
    x_line = np.linspace(v.min(), v.max(), 100)
    ax.plot(x_line, np.polyval(z, x_line), 'k-', linewidth=1.5, alpha=0.7)

    sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
    ax.set_xlabel(map_name)
    ax.set_ylabel(gname)
    ax.set_title(f'rho = {r:.3f}{sig}', fontsize=10)

# Hide unused axes
for idx in range(n_plots, len(axes)):
    axes[idx].set_visible(False)

fig.suptitle('FC Gradient vs Neurotransmitter/Structural Reference Maps',
             fontsize=12, fontweight='bold')
plt.tight_layout()
fig.savefig(f'{FIGDIR}/fig6_receptor_gradient_scatter.png')
plt.close()
print(f"  -> {FIGDIR}/fig6_receptor_gradient_scatter.png")

# ===================================================================
print(f"\nAll figures saved to {FIGDIR}/")
print("Done.")
