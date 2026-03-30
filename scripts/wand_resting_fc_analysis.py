#!/usr/bin/env python
"""Comprehensive resting-state FC analysis on WAND sub-08033 using jaxoccoli.

Computes all available connectivity measures and compares with MELODIC output.
"""

import sys
sys.path.insert(0, "/Users/mhough/dev/hippy-feat")

import json
import time
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
MELODIC_DIR = f"{WAND}/derivatives/fsl-fmri/{SUB}/{SES}/task-rest_melodic"
MOTION_PAR = f"{WAND}/derivatives/fsl-fmri/{SUB}/{SES}/mc/task-rest_mc.par"
MASK = f"{MELODIC_DIR}/mask.nii.gz"
APARC = f"{WAND}/derivatives/freesurfer/{SUB}/mri/aparc.DKTatlas+aseg.mgz"

TR = 2.0  # seconds

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print("=" * 70)
print("WAND Resting-State FC Analysis — jaxoccoli on sub-08033")
print("=" * 70)

print("\n[1] Loading data...")
t0 = time.time()

bold_img = nib.load(BOLD_MC)
bold_data = bold_img.get_fdata()  # (96, 96, 64, 300)
print(f"  BOLD shape: {bold_data.shape}, TR={TR}s, duration={bold_data.shape[3]*TR}s")

mask_img = nib.load(MASK)
mask = mask_img.get_fdata().astype(bool)
n_voxels = mask.sum()
print(f"  Brain mask: {n_voxels} voxels")

# Extract masked timeseries: (n_voxels, T)
ts = bold_data[mask].astype(np.float32)  # (V, 300)
T = ts.shape[1]
print(f"  Timeseries: {ts.shape}")

# Load motion parameters
motion = np.loadtxt(MOTION_PAR)  # (300, 6)
print(f"  Motion parameters: {motion.shape}")

# Compute framewise displacement
fd = np.zeros(T)
fd[1:] = np.sum(np.abs(np.diff(motion, axis=0)), axis=1)
mean_fd = np.mean(fd)
max_fd = np.max(fd)
print(f"  Framewise displacement: mean={mean_fd:.4f}, max={max_fd:.4f}")

print(f"  Loaded in {time.time()-t0:.1f}s")

# ---------------------------------------------------------------------------
# 2. MELODIC dimensionality estimation
# ---------------------------------------------------------------------------
print("\n[2] MELODIC dimensionality estimation...")

melodic_mix = np.loadtxt(f"{MELODIC_DIR}/melodic_mix")
melodic_icstats = np.loadtxt(f"{MELODIC_DIR}/melodic_ICstats")
n_ics = melodic_icstats.shape[0]
print(f"  MELODIC estimated {n_ics} independent components")
print(f"  Mixing matrix: {melodic_mix.shape} (timepoints x ICs)")

# Eigenvalues from MELODIC PCA
melodic_pcaE = np.loadtxt(f"{MELODIC_DIR}/melodic_pcaE")
print(f"  PCA eigenvalues: {melodic_pcaE.shape[0]} values")

# IC explained variance (col 0 of ICstats)
ic_var = melodic_icstats[:, 0]
total_ic_var = np.sum(ic_var)
print(f"  Total explained variance by {n_ics} ICs: {total_ic_var:.1f}%")
print(f"  Top-5 ICs explain: {np.sum(ic_var[:5]):.1f}%")
print(f"  Top-10 ICs explain: {np.sum(ic_var[:10]):.1f}%")

# ---------------------------------------------------------------------------
# 3. Simple parcellation via MELODIC ICs
# ---------------------------------------------------------------------------
print("\n[3] Using MELODIC ICs as parcellation (spatial maps)...")
t0 = time.time()

# IC timeseries from melodic_mix: (300, 67) -> use as ROI timeseries
ic_ts = jnp.array(melodic_mix.T, dtype=jnp.float32)  # (67, 300)
print(f"  IC timeseries: {ic_ts.shape}")

# ---------------------------------------------------------------------------
# 4. Covariance and correlation
# ---------------------------------------------------------------------------
print("\n[4] Computing connectivity measures with jaxoccoli...")

# 4a. Full correlation
t0 = time.time()
fc_corr = joc.corr(ic_ts)
print(f"  corr: {fc_corr.shape}, range [{float(jnp.min(fc_corr)):.3f}, {float(jnp.max(fc_corr)):.3f}], time={time.time()-t0:.3f}s")

# 4b. Covariance
t0 = time.time()
fc_cov = joc.cov(ic_ts)
print(f"  cov: {fc_cov.shape}, range [{float(jnp.min(fc_cov)):.3f}, {float(jnp.max(fc_cov)):.3f}], time={time.time()-t0:.3f}s")

# 4c. L2-regularised covariance
t0 = time.time()
fc_cov_reg = joc.cov(ic_ts, l2=0.1)
print(f"  cov(l2=0.1): {fc_cov_reg.shape}, time={time.time()-t0:.3f}s")

# 4d. Precision matrix
t0 = time.time()
fc_prec = joc.precision(ic_ts, l2=0.1)
print(f"  precision: {fc_prec.shape}, time={time.time()-t0:.3f}s")

# 4e. Partial correlation
t0 = time.time()
fc_pcorr = joc.partial_corr(ic_ts, l2=0.1)
print(f"  partial_corr: {fc_pcorr.shape}, range [{float(jnp.min(fc_pcorr)):.3f}, {float(jnp.max(fc_pcorr)):.3f}], time={time.time()-t0:.3f}s")

# ---------------------------------------------------------------------------
# 5. Spectral embedding
# ---------------------------------------------------------------------------
print("\n[5] Spectral embedding...")

# Use absolute correlation as affinity
W = jnp.abs(fc_corr) * (1 - jnp.eye(n_ics))

# 5a. Laplacian eigenmaps
t0 = time.time()
eigvals_le, eigvecs_le = joc.laplacian_eigenmaps(W, k=10)
print(f"  laplacian_eigenmaps: {eigvals_le.shape} eigenvalues, time={time.time()-t0:.3f}s")
print(f"    first 5 eigenvalues: {eigvals_le[:5]}")

# 5b. Diffusion mapping
t0 = time.time()
eigvals_dm, eigvecs_dm = joc.diffusion_mapping(W, k=10, alpha=0.5)
print(f"  diffusion_mapping: {eigvals_dm.shape} eigenvalues, time={time.time()-t0:.3f}s")
print(f"    first 5 eigenvalues: {eigvals_dm[:5]}")

# 5c. Spectral features
t0 = time.time()
spec_feats = joc.spectral_features(W, k=10)
print(f"  spectral_features: {spec_feats.shape}")
print(f"    spectral gap (lambda_2): {float(spec_feats[-2]):.6f}")
print(f"    spectral radius: {float(spec_feats[-1]):.6f}")

# ---------------------------------------------------------------------------
# 6. Graph analysis
# ---------------------------------------------------------------------------
print("\n[6] Graph analysis...")

# 6a. Graph Laplacian
L = joc.graph_laplacian(W, normalise=True)
print(f"  graph_laplacian: {L.shape}")

# 6b. Modularity (test with 5 communities via spectral clustering)
# Use first 5 eigenvectors as soft assignment
C_raw = eigvecs_le[:, :5]
C_soft = jax.nn.softmax(C_raw * 10, axis=-1)  # sharpen
Q = joc.relaxed_modularity(W, C_soft, gamma=1.0)
print(f"  relaxed_modularity (5 communities): Q={float(Q):.4f}")

# 6c. Chebyshev filter
t0 = time.time()
coeffs, cheb_fwd = joc.make_chebyshev_filter(L, K=8, key=jax.random.PRNGKey(0))
filtered = cheb_fwd(coeffs, ic_ts)  # (67, 300) — filter along node dim
print(f"  chebyshev_filter (K=8): {filtered.shape}, time={time.time()-t0:.3f}s")

# ---------------------------------------------------------------------------
# 7. Dynamic connectivity
# ---------------------------------------------------------------------------
print("\n[7] Dynamic connectivity (sliding window)...")

window_size = 30  # 30 TRs = 60 seconds
t0 = time.time()
dfc = joc.sliding_window_corr(ic_ts, window_size)
n_windows = dfc.shape[0]
print(f"  sliding_window_corr (ws={window_size}): {dfc.shape}, time={time.time()-t0:.3f}s")

# FC variability across time
fc_std = jnp.std(dfc, axis=0)
mean_fc_variability = float(jnp.mean(fc_std))
max_fc_variability = float(jnp.max(fc_std))
print(f"  FC temporal variability: mean={mean_fc_variability:.4f}, max={max_fc_variability:.4f}")

# Dynamic with covariance estimator
t0 = time.time()
dfc_cov = joc.dynamic_connectivity(ic_ts, window_size, estimator='cov')
print(f"  dynamic_connectivity (cov): {dfc_cov.shape}, time={time.time()-t0:.3f}s")

# ---------------------------------------------------------------------------
# 8. Fourier / frequency analysis
# ---------------------------------------------------------------------------
print("\n[8] Frequency-domain analysis...")

# 8a. Envelope of IC timeseries
t0 = time.time()
env = joc.envelope(ic_ts, axis=-1)
print(f"  envelope: {env.shape}, mean amplitude per IC: [{float(jnp.mean(env, axis=-1).min()):.2f}, {float(jnp.mean(env, axis=-1).max()):.2f}]")

# 8b. Instantaneous phase
phase = joc.instantaneous_phase(ic_ts, axis=-1)
print(f"  instantaneous_phase: {phase.shape}")

# 8c. Phase-locking value (PLV) — computed from instantaneous phase
phase_diff = phase[:, None, :] - phase[None, :, :]
plv = jnp.abs(jnp.mean(jnp.exp(1j * phase_diff), axis=-1))
print(f"  phase_locking_value: {plv.shape}, mean off-diag: {float(jnp.mean(plv * (1 - jnp.eye(n_ics)))):.4f}")

# 8d. Bandpass filter (0.01 - 0.1 Hz resting-state band)
n_freq = T // 2 + 1
bp_filter = joc.init_ideal_spectrum(n_freq, low=0.01, high=0.1, fs=1.0/TR)
# Apply to first IC as demo
filtered_ic0 = joc.product_filter(ic_ts[0], bp_filter)
print(f"  bandpass_filter (0.01-0.1 Hz): applied to IC0, power ratio: {float(jnp.var(filtered_ic0)/jnp.var(ic_ts[0])):.3f}")
print(f"  time={time.time()-t0:.3f}s")

# ---------------------------------------------------------------------------
# 9. Matrix / SPD operations
# ---------------------------------------------------------------------------
print("\n[9] SPD manifold operations...")

# Ensure FC is SPD
fc_spd = joc.ensure_spd(fc_cov_reg)
print(f"  ensure_spd: min eigenvalue = {float(jnp.linalg.eigvalsh(fc_spd).min()):.6f}")

# Tangent projection at identity
t0 = time.time()
ref = jnp.eye(n_ics)
tangent = joc.tangent_project_spd(fc_spd, ref)
print(f"  tangent_project_spd: {tangent.shape}, trace={float(jnp.trace(tangent)):.3f}, time={time.time()-t0:.3f}s")

# sym2vec (for classification/regression)
fc_vec = joc.sym2vec(fc_corr, offset=1)
print(f"  sym2vec: {fc_vec.shape} (upper triangle of {n_ics}x{n_ics} correlation)")

# ---------------------------------------------------------------------------
# 10. Temporal interpolation (simulate censoring)
# ---------------------------------------------------------------------------
print("\n[10] Temporal interpolation (simulated censoring)...")

# Censor high-motion timepoints (FD > 0.5mm)
fd_jnp = jnp.array(fd)
motion_mask = fd_jnp < 0.5
n_censored = int(T - jnp.sum(motion_mask))
print(f"  Censored {n_censored}/{T} timepoints (FD > 0.5mm)")

if n_censored > 0:
    t0 = time.time()
    ic0_interp = joc.hybrid_interpolate(
        ic_ts[0:1], motion_mask,
        max_consecutive_linear=3, sampling_period=TR
    )
    print(f"  hybrid_interpolate: recovered {n_censored} frames, time={time.time()-t0:.3f}s")
    rmse = float(jnp.sqrt(jnp.mean((ic0_interp[0, ~motion_mask] - ic_ts[0, ~motion_mask])**2)))
    print(f"  RMSE at censored timepoints: {rmse:.4f}")
else:
    print(f"  No timepoints exceed threshold — data is clean")

# ---------------------------------------------------------------------------
# 11. Optimal transport
# ---------------------------------------------------------------------------
print("\n[11] Optimal transport...")

# Split data into first/second half for test-retest
half = T // 2
fc_half1 = joc.corr(ic_ts[:, :half])
fc_half2 = joc.corr(ic_ts[:, half:])

# Wasserstein FC distance (1D sorted CDF)
t0 = time.time()
w_dist = joc.wasserstein_fc_distance(fc_half1, fc_half2)
print(f"  wasserstein_fc_distance (half1 vs half2): {float(w_dist):.6f}, time={time.time()-t0:.3f}s")

# Gromov-Wasserstein (structure comparison)
t0 = time.time()
T_plan, gw_cost = joc.gromov_wasserstein_fc(
    fc_half1, fc_half2, epsilon=1.0, max_iters=50, gw_iters=5
)
print(f"  gromov_wasserstein_fc: cost={float(gw_cost):.6f}, time={time.time()-t0:.3f}s")

# ---------------------------------------------------------------------------
# 12. Learnable components (demo)
# ---------------------------------------------------------------------------
print("\n[12] Learnable components (factory demos)...")

key = jax.random.PRNGKey(42)

# Learnable frequency filter
t0 = time.time()
params_ff, fwd_ff = joc.make_freq_filter(n_freq, key=key, n_filters=1,
                                          init_fn=lambda n: joc.init_butterworth_spectrum(
                                              n, order=4, low=0.01, high=0.1, fs=1.0/TR))
filtered_ts = fwd_ff(params_ff, ic_ts[0])
print(f"  make_freq_filter (Butterworth init): {filtered_ts.shape}, time={time.time()-t0:.3f}s")

# Learnable covariance
t0 = time.time()
params_cov, fwd_cov = joc.make_learnable_cov(n_ics, T, key=key, estimator='corr')
fc_learned = fwd_cov(params_cov, ic_ts)
print(f"  make_learnable_cov: {fc_learned.shape}, matches corr: {bool(jnp.allclose(fc_learned, fc_corr, atol=1e-3))}")

# ---------------------------------------------------------------------------
# 13. Bayesian beta estimation (on IC timeseries as demo)
# ---------------------------------------------------------------------------
print("\n[13] Bayesian conjugate GLM on IC timeseries...")

# Create simple design matrix (drift + constant)
t_vec = jnp.linspace(0, 1, T)
X_design = jnp.column_stack([
    jnp.ones(T),
    t_vec,
    jnp.cos(2 * jnp.pi * t_vec),
    jnp.sin(2 * jnp.pi * t_vec),
])
print(f"  Design matrix: {X_design.shape} (constant + linear + cos + sin)")

from jaxoccoli.bayesian_beta import make_conjugate_glm_vmap

t0 = time.time()
params_glm, fwd_glm = make_conjugate_glm_vmap(X_design)
bm, bv, s2 = fwd_glm(params_glm, ic_ts)  # treat ICs as "voxels"
print(f"  Conjugate GLM on {n_ics} ICs: beta_mean={bm.shape}, beta_var={bv.shape}")
print(f"  Noise variance (sigma^2): range [{float(s2.min()):.2f}, {float(s2.max()):.2f}]")
print(f"  Beta SE range: [{float(jnp.sqrt(bv).min()):.4f}, {float(jnp.sqrt(bv).max()):.4f}]")
print(f"  time={time.time()-t0:.3f}s")

# Posterior correlation (variance-aware)
t0 = time.time()
fc_posterior = joc.posterior_corr(bm, bv)
print(f"  posterior_corr: {fc_posterior.shape}, range [{float(jnp.min(fc_posterior)):.3f}, {float(jnp.max(fc_posterior)):.3f}]")
print(f"  time={time.time()-t0:.3f}s")

# ---------------------------------------------------------------------------
# 14. Comparison with MELODIC
# ---------------------------------------------------------------------------
print("\n[14] MELODIC comparison...")

# MELODIC dimensionality vs our spectral analysis
print(f"  MELODIC ICs: {n_ics}")
print(f"  Laplacian eigenmaps eigenvalue gap analysis:")
le_gaps = jnp.diff(eigvals_le)
largest_gap_idx = int(jnp.argmax(le_gaps))
print(f"    Largest eigenvalue gap at component {largest_gap_idx+1} "
      f"(gap={float(le_gaps[largest_gap_idx]):.6f})")

# Compare FC from MELODIC mixing matrix vs our computation
# MELODIC's mixing matrix columns are IC timeseries
melodic_fc = np.corrcoef(melodic_mix.T)
our_fc = np.array(fc_corr)
fc_agreement = np.corrcoef(melodic_fc[np.triu_indices(n_ics, k=1)],
                           our_fc[np.triu_indices(n_ics, k=1)])[0, 1]
print(f"  FC agreement (MELODIC vs jaxoccoli corr): r={fc_agreement:.6f}")

# ---------------------------------------------------------------------------
# 15. Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
Subject:               {SUB}
Session:               {SES}
BOLD dimensions:       {bold_data.shape}
Brain voxels:          {n_voxels}
TR:                    {TR}s
Duration:              {T*TR}s ({T} volumes)
Mean FD:               {mean_fd:.4f} mm

MELODIC ICs:           {n_ics}
Top-10 IC variance:    {np.sum(ic_var[:10]):.1f}%

Connectivity measures computed:
  - Full correlation          ({n_ics}x{n_ics})
  - Covariance                ({n_ics}x{n_ics})
  - L2-regularised covariance ({n_ics}x{n_ics})
  - Precision matrix          ({n_ics}x{n_ics})
  - Partial correlation       ({n_ics}x{n_ics})
  - Laplacian eigenmaps       (k=10)
  - Diffusion mapping         (k=10)
  - Spectral features         (gap, connectivity, radius)
  - Relaxed modularity        (Q={float(Q):.4f})
  - Chebyshev spectral filter (K=8)
  - Dynamic FC (sliding window, ws={window_size})
  - Phase-locking value       ({n_ics}x{n_ics})
  - Signal envelope           ({n_ics}x{T})
  - Bandpass filtering        (0.01-0.1 Hz)
  - SPD tangent projection    ({n_ics}x{n_ics})
  - Vectorised FC             ({fc_vec.shape[0]} features)
  - Wasserstein FC distance   (half1 vs half2)
  - Gromov-Wasserstein FC     (structural comparison)
  - Bayesian conjugate GLM    (beta_mean, beta_var)
  - Posterior correlation     (variance-aware FC)

FC agreement with MELODIC: r={fc_agreement:.6f}
Spectral gap (lambda_2):   {float(spec_feats[-2]):.6f}
""")
