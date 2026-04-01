# Bayesian First-Level fMRI Analysis in JAX

## Motivation

Current hippy-feat GLM variants (A-F) produce **point estimates** — a single beta per voxel per condition. This misses:

- **Uncertainty** on activation estimates (critical for clinical decisions)
- **HRF shape variability** with proper posterior (not just best-fit from a library)
- **Spatial context** (neighboring voxels should inform each other)
- **Proper noise modeling** (AR structure, not assumed white)

BROCCOLI (Eklund et al.) showed this is feasible on GPU via OpenCL. JAX makes it natural via autodiff + NUTS + vmap.

## Architecture: Variant G — Bayesian GLM

### Model

Per voxel, the generative model is:

```
y(t) = Σ_k β_k · [x_k(t) ⊛ h(t; θ_hrf)] + ε(t)

where:
  y(t)       = observed BOLD timeseries
  β_k        = activation amplitude for condition k
  x_k(t)     = stimulus function for condition k (from events.tsv)
  h(t; θ_hrf)= hemodynamic response function parameterized by θ_hrf
  ε(t)       = AR(p) colored noise: ε(t) = Σ_j a_j · ε(t-j) + η(t)
  η(t)       ~ N(0, σ²)
```

### Parameters (per voxel)

| Parameter | Description | Prior |
|---|---|---|
| `β_k` | Activation amplitude per condition | N(0, 10²) |
| `θ_hrf` | HRF shape parameters | See HRF models below |
| `a_1, ..., a_p` | AR noise coefficients (typically p=2) | N(0, 0.5²) |
| `σ²` | Noise variance | InvGamma(1, 1) |

### HRF Model Options

Three levels of physiological detail, selectable per analysis:

**Level 1: FLOBS (3 basis functions)**
```python
θ_hrf = [w1, w2, w3]  # weights on 3 FSL FLOBS basis functions
h(t) = w1*flobs1(t) + w2*flobs2(t) + w3*flobs3(t)
Prior: w1 ~ N(1, 0.5²), w2,w3 ~ N(0, 0.3²)
```

**Level 2: Parameterized double-gamma**
```python
θ_hrf = [peak_delay, peak_width, undershoot_ratio, undershoot_delay]
h(t) = gamma(t; peak_delay, peak_width) - undershoot_ratio * gamma(t; undershoot_delay, ...)
Prior: peak_delay ~ N(6, 1²), peak_width ~ N(1, 0.3²), etc.
```

**Level 3: vpjax physiological model (Riera)**
```python
θ_hrf = [neural_efficacy, signal_decay, autoregulation, transit_time, ...]
h(t) = vpjax.hemodynamics.riera_hrf(t, θ_hrf)
Prior: informed by MRS (GABA/Glu), qMRI (T1, T2*), angiography (vessel geometry)
```

Level 3 is the unique contribution — the "HRF" is the actual neurovascular coupling model, and the posterior is over physiological parameters, not abstract shape weights.

### Inference

**Blackjax NUTS (No-U-Turn Sampler)**
```python
import blackjax

def log_posterior(params, voxel_data, design_matrix):
    beta, theta_hrf, ar_coeffs, log_sigma = unpack(params)

    # HRF convolution (differentiable)
    hrf = make_hrf(theta_hrf)  # Level 1, 2, or 3
    predicted = convolve_design(design_matrix, hrf, beta)

    # AR(p) residual likelihood
    residuals = voxel_data - predicted
    sigma = jnp.exp(log_sigma)
    ll = ar_loglikelihood(residuals, ar_coeffs, sigma)

    # Priors
    lp = (log_prior_beta(beta) + log_prior_hrf(theta_hrf) +
          log_prior_ar(ar_coeffs) + log_prior_sigma(log_sigma))

    return ll + lp

# Vectorized over all voxels
@jax.vmap
def sample_voxel(voxel_data, key):
    kernel = blackjax.nuts(log_posterior, ...)
    states = blackjax.sample(kernel, key, n_samples=1000, n_warmup=500)
    return posterior_summary(states)

# Run on GPU: all voxels in parallel
results = sample_voxel(all_voxel_data, voxel_keys)
```

### Spatial Prior (optional, Level 2+)

Instead of independent voxel fitting, add a Markov Random Field (MRF) prior that encourages spatial smoothness of activation and HRF parameters:

```python
def spatial_log_prior(beta_map, adjacency, lambda_spatial):
    """MRF prior: penalize differences between neighboring voxels."""
    diff = beta_map[adjacency[:, 0]] - beta_map[adjacency[:, 1]]
    return -0.5 * lambda_spatial * jnp.sum(diff ** 2)
```

This replaces Gaussian spatial smoothing (which blurs activations) with a proper probabilistic constraint (which preserves edges).

### Outputs

Per voxel:
- `beta_mean` — posterior mean activation (like standard GLM)
- `beta_std` — posterior standard deviation (UNCERTAINTY — the key addition)
- `hrf_params_mean` — fitted HRF shape (or physiological parameters for Level 3)
- `ar_coeffs` — noise autocorrelation structure
- `log_evidence` — marginal likelihood for model comparison

Maps:
- `activation_map.nii.gz` — posterior mean β
- `uncertainty_map.nii.gz` — posterior σ_β
- `posterior_probability_map.nii.gz` — P(β > threshold | data)
- `hrf_delay_map.nii.gz` — spatial map of HRF peak latency
- `noise_ar1_map.nii.gz` — AR(1) coefficient map

## Performance Estimates

Based on hippy-feat Variant C benchmarks (76×90×74 volume):

| Component | Time per voxel | Time for whole brain (vmap) |
|---|---|---|
| Log posterior evaluation | ~0.1 ms | ~50 ms (GPU parallel) |
| NUTS step (leapfrog) | ~1 ms | ~500 ms |
| 1000 samples + 500 warmup | ~1.5 s | ~750 s (sequential per voxel) |
| With jax.vmap on A100 GPU | — | **~30-60 s** (all voxels parallel) |

For comparison: BROCCOLI reports ~1-2 minutes per subject on GPU. Our estimate of 30-60s is competitive, with the advantage of full posterior samples (not just Variational Bayes approximation).

## Comparison with Existing Tools

| Tool | Method | HRF | Noise | Spatial | GPU | Posterior |
|---|---|---|---|---|---|---|
| FSL FEAT | OLS/FLAME | FLOBS/Glover | Pre-whitened AR | Gaussian smooth | No | No |
| SPM | ReML | Canonical + derivatives | AR(1) | Gaussian smooth | No | No |
| BROCCOLI | Bayesian GLM | Canonical | AR(4) | MRF prior | OpenCL | VB approx |
| FSL FABBER | Bayesian (VB) | Flexible | AR | Optional spatial | No | VB approx |
| **hippy-feat G** | **Full Bayesian** | **FLOBS/param/vpjax** | **AR(p)** | **MRF optional** | **JAX GPU** | **NUTS samples** |

Key advantages of Variant G:
1. **Full posterior** via NUTS (not Variational Bayes approximation)
2. **Physiological HRF** from vpjax (Level 3) — posterior over vessel compliance, not basis weights
3. **JAX autodiff** — no manual gradient derivation, any model complexity
4. **vmap parallelism** — all voxels simultaneously on GPU
5. **Composable** — same framework for Level 1 (FLOBS), 2 (parametric), 3 (physiological)

## Implementation Plan

### Phase 1: Core Bayesian GLM (Variant G, Level 1)
- FLOBS HRF basis
- AR(2) noise model
- Blackjax NUTS per voxel
- jax.vmap parallelism
- Output: beta_mean, beta_std, posterior_probability maps
- Test on WAND categorylocaliser data
- **Compare against FSL FEAT on same data**

### Phase 2: Parameterized HRF (Level 2)
- Double-gamma parameterization
- HRF delay/width maps
- Spatial MRF prior on HRF parameters
- Test: does parameterized HRF improve activation detection?

### Phase 3: Physiological HRF via vpjax (Level 3)
- Riera neurovascular coupling as HRF generator
- Prior from MRS (GABA/Glu), qMRI (T1, vessel properties)
- Posterior over physiological parameters per voxel
- WAND validation: compare vpjax-derived CBF against pCASL CBF

### Phase 4: Real-time Bayesian (streaming)
- Incremental posterior update per TR (sequential Monte Carlo)
- For BCI / neurofeedback applications
- Builds on hippy-feat's existing per-TR pipeline

## Dependencies

- `blackjax` — NUTS/HMC sampling
- `vpjax` — physiological HRF model (Level 3)
- `jaxoccoli` — existing hippy-feat modules (motion, spatial, io)
- `jax`, `equinox`, `optax`

## References

- Eklund A et al. (2014). BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs. Front Neuroinform.
- Woolrich MW et al. (2004). Multilevel linear modelling for FMRI group analysis using Bayesian inference. NeuroImage. (FLAME)
- Penny WD et al. (2005). Bayesian fMRI time series analysis with spatial priors. NeuroImage.
- Chappell MA et al. (2009). Variational Bayesian inference for a nonlinear forward model. IEEE TMI. (FABBER)
- Riera JJ et al. (2006/2007). Nonlinear local electrovascular coupling. HBM.
- Hoffman MD, Gelman A (2014). The No-U-Turn Sampler. JMLR. (NUTS)
