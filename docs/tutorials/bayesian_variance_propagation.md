# Bayesian fMRI Pipelines: From GLM to Uncertainty-Aware Connectivity

This tutorial walks through the central innovation of hippy-feat: **end-to-end
variance propagation** from voxel-level BOLD time series through GLM estimation,
atlas parcellation, and functional connectivity. Every function is pure JAX,
compatible with `jax.jit`, `jax.grad`, and `jax.vmap`.

## Why propagate uncertainty?

Standard fMRI beta series correlation (Rissman et al. 2004) fits a GLM per
trial, extracts point-estimate betas, and computes Pearson correlation across
trials. The problem: **point estimates discard uncertainty**. A beta with
$\text{SE} = 0.1$ and a beta with $\text{SE} = 5.0$ are treated identically.

This matters because:

1. High-noise voxels produce unreliable betas that inflate connectivity estimates.
2. Averaging noisy betas into parcels without accounting for variance biases the
   parcel-level signal.
3. Correlating point estimates ignores the posterior width, giving
   overconfident connectivity matrices.

hippy-feat solves this by outputting `(beta_mean, beta_var)` at every stage and
propagating both through the pipeline.

## 1. Conjugate GLM: closed-form posterior on beta

The workhorse is `make_conjugate_glm()`, which implements a conjugate
normal-inverse-gamma GLM in closed form.

### The model

$$
y = X \beta + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, \sigma^2 I)
$$

$$
\beta \mid \sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2 \Lambda^{-1})
$$

$$
\sigma^2 \sim \text{InverseGamma}(a_0, b_0)
$$

where $X$ is the $(T \times P)$ design matrix, $\Lambda$ is the prior precision,
and $\mu_0$ is the prior mean.

### The posterior

The conjugate posterior is available in closed form:

$$
\beta \mid y, \sigma^2 \sim \mathcal{N}\!\bigl(\hat{\beta},\; \sigma^2 (X^\top X + \Lambda)^{-1}\bigr)
$$

$$
\sigma^2 \mid y \sim \text{InverseGamma}(a_n, b_n)
$$

where:

$$
\hat{\beta} = (X^\top X + \Lambda)^{-1}(X^\top y + \Lambda \mu_0)
$$

$$
a_n = a_0 + T/2, \qquad b_n = b_0 + \tfrac{1}{2}\bigl(\|y - X\hat{\beta}\|^2 + (\hat{\beta} - \mu_0)^\top \Lambda (\hat{\beta} - \mu_0)\bigr)
$$

### Usage

```python
import jax
import jax.numpy as jnp
from jaxoccoli.bayesian_beta import make_conjugate_glm

# Simulate data: 100 timepoints, 3 regressors
key = jax.random.PRNGKey(31)
k1, k2 = jax.random.split(key)
T, P = 100, 3
X = jax.random.normal(k1, (T, P))
true_beta = jnp.array([2.0, -1.0, 0.5])
y = X @ true_beta + 0.5 * jax.random.normal(k2, (T,))

# Create the GLM
params, forward = make_conjugate_glm(X)

# Fit: returns posterior mean, marginal variances, and noise variance
beta_mean, beta_var, sigma2 = forward(params, y)

print(f"beta_mean: {beta_mean}")    # ~ [2.0, -1.0, 0.5]
print(f"beta_var:  {beta_var}")     # positive variances per regressor
print(f"sigma2:    {sigma2:.4f}")   # ~ 0.25 (true sigma^2)
```

The factory pattern `make_conjugate_glm(X)` returns a `(params, forward_fn)`
pair. The design matrix $X$ is baked into `params` as precomputed matrices
$(X^\top X + \Lambda)^{-1}$ and $(X^\top X + \Lambda)^{-1} X^\top$, so the
forward pass is a single matrix-vector multiply -- fast enough for real-time.

```{note}
The returned `beta_var` is the **diagonal** of the full posterior covariance
$\sigma^2 (X^\top X + \Lambda)^{-1}$, i.e. the marginal variance per
regressor. This is what flows downstream through parcellation and connectivity.
```

### Controlling the prior

The prior precision $\Lambda$ controls regularisation. A strong prior shrinks
betas toward `prior_mean`:

```python
# Weak prior (nearly OLS)
params_weak, fwd_weak = make_conjugate_glm(X, prior_precision=0.01 * jnp.eye(P))
beta_weak, _, _ = fwd_weak(params_weak, y)

# Strong prior toward zero (heavy shrinkage)
params_strong, fwd_strong = make_conjugate_glm(
    X,
    prior_precision=100.0 * jnp.eye(P),
    prior_mean=jnp.zeros(P),
)
beta_strong, _, _ = fwd_strong(params_strong, y)

# Strong prior produces betas closer to zero
assert jnp.sum(beta_strong ** 2) < jnp.sum(beta_weak ** 2)
```

```{tip}
For fMRI, a weak prior ($\Lambda = 0.01 I$) is a good default. Use stronger
priors when you have informative expectations from previous sessions or
group-level results.
```

## 2. AR(1) prewhitened GLM: handling temporal autocorrelation

fMRI noise is temporally autocorrelated. Ignoring this inflates the apparent
precision of beta estimates. `make_ar1_conjugate_glm()` handles this with a
two-step procedure:

1. Estimate the AR(1) coefficient $\rho$ from OLS residuals (Yule-Walker +
   Bayesian shrinkage).
2. Prewhiten both $y$ and $X$, then apply the conjugate GLM.

### The AR(1) model

$$
\varepsilon_t = \rho\, \varepsilon_{t-1} + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, \sigma^2)
$$

Prewhitening transforms to independent residuals:

$$
y_t^* = y_t - \rho\, y_{t-1}, \qquad X_t^* = X_t - \rho\, X_{t-1}
$$

### Usage

```python
from jaxoccoli.bayesian_beta import make_ar1_conjugate_glm

# Generate data with AR(1) noise
T, P = 200, 2
k1, k2 = jax.random.split(key)
X = jax.random.normal(k1, (T, P))
true_beta = jnp.array([3.0, -2.0])
true_rho = 0.7

# AR(1) noise via jax.lax.scan
innovations = 0.5 * jax.random.normal(k2, (T,))
def ar_step(carry, inn):
    prev = carry
    curr = true_rho * prev + inn
    return curr, curr
_, noise = jax.lax.scan(ar_step, 0.0, innovations)
y = X @ true_beta + noise

# Fit AR(1)-prewhitened conjugate GLM
params, forward = make_ar1_conjugate_glm(X)
beta_mean, beta_var, sigma2, rho_est = forward(params, y)

print(f"beta_mean: {beta_mean}")      # ~ [3.0, -2.0]
print(f"rho:       {rho_est:.3f}")    # ~ 0.7
```

The returned `rho` is shrunk toward a prior mean of 0.5 with prior variance
0.09 by default. This prevents extreme estimates from short time series.

```{note}
The forward function returns a 4-tuple `(beta_mean, beta_var, sigma2, rho)`
compared to the standard GLM's 3-tuple. The extra `rho` is the estimated
AR(1) autocorrelation coefficient, bounded to $(-1, 1)$.
```

## 3. Vectorized multi-voxel fitting with `make_conjugate_glm_vmap()`

Fitting one voxel at a time is wasteful on GPU. `make_conjugate_glm_vmap()`
wraps the single-voxel forward pass in `jax.vmap` for batch processing:

```python
from jaxoccoli.bayesian_beta import make_conjugate_glm_vmap

# Simulated whole-brain data: 50 voxels, 100 timepoints, 3 regressors
T, P, V = 100, 3, 50
k1, k2, k3 = jax.random.split(key, 3)
X = jax.random.normal(k1, (T, P))
true_betas = jax.random.normal(k2, (V, P))
noise = 0.5 * jax.random.normal(k3, (V, T))
Y = jax.vmap(lambda b: X @ b)(true_betas) + noise  # (V, T)

# Fit all voxels in parallel
params, forward = make_conjugate_glm_vmap(X)
beta_means, beta_vars, sigma2s = forward(params, Y)

print(f"beta_means shape: {beta_means.shape}")  # (50, 3)
print(f"beta_vars shape:  {beta_vars.shape}")   # (50, 3)
print(f"sigma2s shape:    {sigma2s.shape}")      # (50,)
```

The vmapped forward function takes `(V, T)` input and returns `(V, P)` means
and variances plus `(V,)` noise variance estimates. The design matrix
precomputation is shared across all voxels.

```{tip}
`make_conjugate_glm_vmap` produces identical results to looping
`make_conjugate_glm` over voxels, but runs as a single fused XLA kernel.
On GPU, this is orders of magnitude faster for typical whole-brain sizes
($V \approx 100{,}000$).
```

## 4. Variance-aware parcellation with `make_atlas_linear_uncertain()`

Standard parcellation averages voxel betas within each parcel and discards
voxel-level variance. `make_atlas_linear_uncertain()` propagates both mean and
variance through a soft linear atlas:

$$
\mu_p = \sum_v w_{pv}\, \mu_v, \qquad
\sigma^2_p = \sum_v w_{pv}^2\, \sigma^2_v
$$

where $w_{pv} = \text{softmax}(\theta_p)_v$ are learnable, normalised
parcellation weights.

```python
from jaxoccoli.learnable import make_atlas_linear_uncertain

# From the conjugate GLM above:
# beta_means: (50, 3) -- 50 voxels, 3 parameters
# beta_vars:  (50, 3) -- corresponding posterior variances

V, P = beta_means.shape
n_parcels = 5

params_atlas, forward_atlas = make_atlas_linear_uncertain(V, n_parcels, key=key)
parc_mean, parc_var = forward_atlas(params_atlas, beta_means, beta_vars)

print(f"parc_mean shape: {parc_mean.shape}")  # (5, 3)
print(f"parc_var shape:  {parc_var.shape}")   # (5, 3)
```

The variance propagation formula $\sigma^2_p = \sum_v w^2_{pv} \sigma^2_v$
follows from the law of total variance for independent variables. Because the
weights are softmax-normalised, the parcel variance is always less than or equal
to the maximum voxel variance -- averaging reduces noise, and this is captured
quantitatively.

```{note}
The atlas weights are learnable via `jax.grad`. You can optimise the
parcellation to minimise a downstream objective (e.g. maximise modularity of
the connectivity matrix) while still propagating variance correctly.
```

## 5. Posterior correlation with `posterior_corr()`

The final piece: computing connectivity that accounts for beta uncertainty.
`posterior_corr()` from `jaxoccoli.covariance` implements a disattenuation
correction:

$$
r_{\text{post}}(i, j) = \frac{r_{\text{obs}}(i, j)}{\sqrt{\rho_i \cdot \rho_j}}
$$

where the reliability of each region is:

$$
\rho_i = \frac{\text{Var}(\mu_i)}{\text{Var}(\mu_i) + \mathbb{E}[\sigma^2_i]}
$$

Here $\text{Var}(\mu_i)$ is the variance of the posterior means across trials
(the "signal"), and $\mathbb{E}[\sigma^2_i]$ is the mean posterior variance
(the "noise"). Regions with high posterior uncertainty get larger corrections.

```python
from jaxoccoli.covariance import posterior_corr

# parc_mean: (5, 3), parc_var: (5, 3)
fc = posterior_corr(parc_mean, parc_var)

print(f"FC shape: {fc.shape}")  # (5, 5)
# Diagonal is always 1.0
# Off-diagonal entries are disattenuated correlations
```

```{tip}
When `beta_var` is zero everywhere, `posterior_corr` reduces to ordinary
Pearson correlation of the means. This means the variance-aware pipeline
is strictly more general -- you can always fall back to the classical
approach by setting variances to zero.
```

## 6. End-to-end pipeline: BOLD to posterior correlation

Putting it all together: BOLD time series in, uncertainty-aware connectivity
matrix out.

```python
import jax
import jax.numpy as jnp
from jaxoccoli.bayesian_beta import make_conjugate_glm_vmap
from jaxoccoli.learnable import make_atlas_linear_uncertain
from jaxoccoli.covariance import posterior_corr

key = jax.random.PRNGKey(31)

# --- Simulated data ---
T, P, V = 100, 3, 50
k1, k2, k3 = jax.random.split(key, 3)
X = jax.random.normal(k1, (T, P))          # design matrix
betas = jax.random.normal(k2, (V, P))      # true betas
noise = 0.5 * jax.random.normal(k3, (V, T))
Y = jax.vmap(lambda b: X @ b)(betas) + noise  # (V, T) BOLD data

# --- Step 1: Bayesian beta estimation ---
params_glm, fwd_glm = make_conjugate_glm_vmap(X)
beta_means, beta_vars, _ = fwd_glm(params_glm, Y)
# beta_means: (50, 3), beta_vars: (50, 3)

# --- Step 2: Variance-aware parcellation ---
n_parcels = 5
params_atlas, fwd_atlas = make_atlas_linear_uncertain(V, n_parcels, key=key)
parc_mean, parc_var = fwd_atlas(params_atlas, beta_means, beta_vars)
# parc_mean: (5, 3), parc_var: (5, 3)

# --- Step 3: Posterior correlation ---
fc = posterior_corr(parc_mean, parc_var)
# fc: (5, 5) uncertainty-aware connectivity matrix

assert fc.shape == (5, 5)
assert jnp.allclose(jnp.diagonal(fc), 1.0, atol=1e-5)
assert jnp.all(jnp.isfinite(fc))
```

This three-step pipeline replaces the classical workflow of OLS beta estimation,
hard atlas averaging, and naive Pearson correlation -- and it does so without
discarding any uncertainty information.

## 7. JAX JIT compilation and autograd compatibility

Every function in the pipeline is compatible with JAX transformations.

### JIT compilation

Wrap the forward passes in `jax.jit` for XLA compilation:

```python
# JIT the whole GLM forward pass
beta_means, beta_vars, sigma2s = jax.jit(fwd_glm)(params_glm, Y)

# JIT the atlas forward pass
parc_mean, parc_var = jax.jit(fwd_atlas)(params_atlas, beta_means, beta_vars)

# JIT posterior_corr
fc = jax.jit(posterior_corr)(parc_mean, parc_var)
```

### End-to-end gradients

Because the entire pipeline is differentiable, you can backpropagate a
downstream loss through connectivity, parcellation, and into the GLM inputs.
This enables learning atlas weights that optimise a connectivity objective:

```python
def pipeline_loss(atlas_params):
    bm, bv, _ = fwd_glm(params_glm, Y)
    pm, pv = fwd_atlas(atlas_params, bm, bv)
    fc = posterior_corr(pm, pv)
    return jnp.sum(fc ** 2)  # example: minimise total squared connectivity

# Gradient of the loss with respect to atlas parameters
grad_atlas = jax.grad(pipeline_loss)(params_atlas)
print(f"grad shape: {grad_atlas.weight.shape}")  # (5, 50)
assert jnp.all(jnp.isfinite(grad_atlas.weight))
```

You can also differentiate with respect to the BOLD data itself (useful for
sensitivity analysis) or through the conjugate GLM forward pass:

```python
def glm_loss(y):
    bm, bv, _ = make_conjugate_glm(X)[1](params_glm, y)
    return jnp.sum(bm ** 2)

grad_y = jax.grad(glm_loss)(Y[0])
assert grad_y.shape == (T,)
assert jnp.all(jnp.isfinite(grad_y))
```

```{note}
The gradient flows because the conjugate posterior is a smooth function of the
data. There are no discrete switches, sampling steps, or non-differentiable
operations in the real-time path.
```

## Summary

| Stage | Function | Input | Output |
|-------|----------|-------|--------|
| GLM (single voxel) | `make_conjugate_glm()` | $(T,)$ time series | `(beta_mean, beta_var, sigma2)` |
| GLM (AR(1)) | `make_ar1_conjugate_glm()` | $(T,)$ time series | `(beta_mean, beta_var, sigma2, rho)` |
| GLM (whole brain) | `make_conjugate_glm_vmap()` | $(V, T)$ all voxels | `(beta_means, beta_vars, sigma2s)` |
| Parcellation | `make_atlas_linear_uncertain()` | `(mean, var)` per voxel | `(mean, var)` per parcel |
| Connectivity | `posterior_corr()` | `(mean, var)` per parcel | $(R, R)$ correlation matrix |

The key insight: uncertainty is a **first-class citizen** at every stage. No
information is discarded, and the entire pipeline is differentiable end-to-end.
