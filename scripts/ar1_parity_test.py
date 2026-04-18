"""AR(1) vs OLS parity test for hippy-feat Variant A.

Generates synthetic single-run fMRI with AR(1) noise at multiple rho,
then fits probe betas with (a) OLS — what Variant A does — and
(b) nilearn ARModel with AR(1) prewhitening — what the notebook does via
FirstLevelModel(noise_model='ar1').

Design matrix uses the same structure as Variant A (build_design_matrix):
probe boxcar + lumped reference boxcar (both convolved with Glover),
intercept, 1st-order cosine drift. 192 TRs, TR=1.5s. 1000 voxels.

Reports per-rho mean abs relative diff, Pearson r, and RMSE of probe betas.
"""
import numpy as np

np.random.seed(0)

# Nilearn's Glover HRF (to match what FirstLevelModel uses internally with hrf_model='glover')
from nilearn.glm.first_level.hemodynamic_models import glover_hrf
from nilearn.glm.regression import ARModel, OLSModel

TR = 1.5
N_TRS = 192
N_VOX = 1000
OVERSAMPLING = 16  # nilearn default

# ---- Build one shared design matrix (probe/reference/intercept/cosine drift) ----
# Glover HRF at TR resolution
t_hrf = np.arange(0, 32, TR / OVERSAMPLING)
hrf = glover_hrf(TR, oversampling=OVERSAMPLING, time_length=32.0)

# 20 events: pick first as probe, rest lumped
n_events = 20
onsets_tr = np.linspace(5, N_TRS - 10, n_events).astype(int)
probe_trial = 0

def boxcar(idx_list):
    bc = np.zeros(N_TRS * OVERSAMPLING, dtype=np.float32)
    for i in idx_list:
        bc[i * OVERSAMPLING] = 1.0
    return bc

probe_bc = boxcar([onsets_tr[probe_trial]])
ref_bc = boxcar([o for i, o in enumerate(onsets_tr) if i != probe_trial])

probe_hi = np.convolve(probe_bc, hrf)[: N_TRS * OVERSAMPLING]
ref_hi = np.convolve(ref_bc, hrf)[: N_TRS * OVERSAMPLING]

# Downsample to TR
probe_reg = probe_hi[::OVERSAMPLING][:N_TRS]
ref_reg = ref_hi[::OVERSAMPLING][:N_TRS]

# Normalize columns (nilearn does not by default, but harmless)
intercept = np.ones(N_TRS, dtype=np.float32)
t = np.linspace(0, 1, N_TRS, dtype=np.float32)
drift = np.cos(2 * np.pi * t)

X = np.column_stack([probe_reg, ref_reg, intercept, drift]).astype(np.float64)
probe_col = 0

# ---- Simulate data ----
TRUE_BETA = 1.0
# Baseline signal (no noise): X @ true_beta_vec
true_beta_vec = np.zeros(X.shape[1])
true_beta_vec[0] = TRUE_BETA  # probe
true_beta_vec[1] = 0.5         # lumped reference
true_beta_vec[2] = 100.0       # intercept
true_beta_vec[3] = 0.3         # drift

signal = X @ true_beta_vec  # (N_TRS,)

def ar1_noise(n, rho, sigma, n_vox, rng):
    e = rng.standard_normal((n, n_vox)) * sigma
    y = np.zeros_like(e)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = rho * y[t - 1] + e[t]
    return y

results = []
rng = np.random.default_rng(42)
for rho in [0.0, 0.2, 0.4, 0.6]:
    noise = ar1_noise(N_TRS, rho, sigma=1.5, n_vox=N_VOX, rng=rng)
    Y = signal[:, None] + noise  # (N_TRS, N_VOX)

    # OLS (what Variant A does)
    ols = OLSModel(X)
    ols_res = ols.fit(Y)
    beta_ols = ols_res.theta[probe_col]  # (N_VOX,)

    # AR(1) prewhitening (what nilearn FirstLevelModel(noise_model='ar1') does per voxel)
    # Two-pass: OLS → estimate rho from residuals → prewhitened GLS
    resid_ols = Y - X @ ols_res.theta
    # Per-voxel AR(1) estimate
    num = (resid_ols[:-1] * resid_ols[1:]).sum(axis=0)
    den = (resid_ols[:-1] ** 2).sum(axis=0)
    rho_hat = np.clip(num / np.maximum(den, 1e-12), -0.99, 0.99)

    # nilearn-style: round rho_hat to quantized grid and group voxels (matches FirstLevelModel)
    q = np.round(rho_hat * 100) / 100.0
    beta_ar1 = np.empty(N_VOX)
    for rho_bin in np.unique(q):
        mask = q == rho_bin
        ar = ARModel(X, float(rho_bin))
        ar_res = ar.fit(Y[:, mask])
        beta_ar1[mask] = ar_res.theta[probe_col]

    # Compare
    mae = np.mean(np.abs(beta_ols - beta_ar1))
    rel_mae = np.mean(np.abs(beta_ols - beta_ar1) / np.maximum(np.abs(beta_ar1), 1e-6))
    rmse = np.sqrt(np.mean((beta_ols - beta_ar1) ** 2))
    r = np.corrcoef(beta_ols, beta_ar1)[0, 1]
    # Also report bias vs truth
    ols_bias = np.mean(beta_ols) - TRUE_BETA
    ar1_bias = np.mean(beta_ar1) - TRUE_BETA
    ols_sd = np.std(beta_ols)
    ar1_sd = np.std(beta_ar1)

    results.append((rho, mae, rel_mae, rmse, r, ols_bias, ar1_bias, ols_sd, ar1_sd))

print(f"{'rho':>5} {'MAE':>8} {'relMAE':>8} {'RMSE':>8} {'r':>7} {'OLSbias':>8} {'AR1bias':>8} {'OLSsd':>8} {'AR1sd':>8}")
for row in results:
    print(f"{row[0]:>5.2f} {row[1]:>8.4f} {row[2]:>8.3%} {row[3]:>8.4f} {row[4]:>7.4f} {row[5]:>+8.4f} {row[6]:>+8.4f} {row[7]:>8.4f} {row[8]:>8.4f}")
