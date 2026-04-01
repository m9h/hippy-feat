# Functional Connectivity: Correlation, Partial Correlation, and Uncertainty

This tutorial covers the `jaxoccoli.covariance` module -- a pure-JAX library of
functional connectivity estimators that are JIT-compilable, differentiable, and
vmappable. We progress from standard empirical covariance through partial
correlation, and then into the variance-aware methods that distinguish
hippy-feat from classical approaches.

## 1. Introduction to functional connectivity matrices

A functional connectivity (FC) matrix captures statistical dependencies between
brain regions (or voxels) from their time series. Given $C$ regions each
observed for $T$ time points, the FC matrix is a $(C \times C)$ symmetric matrix.

The `jaxoccoli.covariance` module provides several estimators, each making
different assumptions:

| Function | What it measures | When to use |
|----------|-----------------|-------------|
| `cov()` | Empirical covariance | Raw second-order statistics |
| `corr()` | Pearson correlation | Scale-free pairwise association |
| `partial_corr()` | Partial correlation | Direct connections (controlling for other regions) |
| `posterior_corr()` | Posterior correlation | Bayesian GLM outputs with uncertainty |
| `attenuated_corr()` | Disattenuated correlation | Known measurement reliability |
| `weighted_corr()` | Weighted correlation | Heterogeneous observation reliability |

All functions follow the convention that **rows are variables** by default
(`rowvar=True`), matching `numpy.cov`.

## 2. Empirical covariance and correlation

### Covariance with `cov()`

```python
import jax
import jax.numpy as jnp
import numpy as np
from jaxoccoli.covariance import cov, corr

key = jax.random.PRNGKey(42)
X = jax.random.normal(key, (5, 100))  # 5 regions, 100 time points

S = cov(X)
print(f"Shape: {S.shape}")  # (5, 5)
```

The function computes:

$$
\hat{\Sigma}_{ij} = \frac{1}{N - 1} \sum_{t=1}^{N} (x_{i,t} - \bar{x}_i)(x_{j,t} - \bar{x}_j)
$$

Key properties verified against NumPy:

```python
# Symmetry
assert jnp.allclose(S, S.T, atol=1e-6)

# Matches numpy.cov exactly
S_np = np.cov(np.array(X))
np.testing.assert_allclose(S, S_np, atol=1e-5)
```

#### Options

**Biased estimator** (divide by $N$ instead of $N-1$):

```python
S_biased = cov(X, bias=True)
S_biased_np = np.cov(np.array(X), bias=True)
np.testing.assert_allclose(S_biased, S_biased_np, atol=1e-5)
```

**L2 (Tikhonov) regularisation** adds $\lambda I$ to the diagonal, ensuring
invertibility for precision and partial correlation:

```python
S_reg = cov(X, l2=1.0)
# Diagonal increases by exactly l2
diag_diff = jnp.diagonal(S_reg) - jnp.diagonal(S)
np.testing.assert_allclose(diag_diff, 1.0, atol=1e-5)
```

**Observation layout**: use `rowvar=False` for $(T \times C)$ layout:

```python
X_T = X.T  # (100, 5)
S1 = cov(X, rowvar=True)
S2 = cov(X_T, rowvar=False)
np.testing.assert_allclose(S1, S2, atol=1e-5)
```

**Observation weights** for heterogeneous reliability:

```python
w = jnp.ones(100)
S_weighted = cov(X, weight=w)
np.testing.assert_allclose(S, S_weighted, atol=1e-4)  # uniform = unweighted
```

### Correlation with `corr()`

Pearson correlation normalises by standard deviations:

$$
r_{ij} = \frac{\hat{\Sigma}_{ij}}{\sqrt{\hat{\Sigma}_{ii}\, \hat{\Sigma}_{jj}}}
$$

```python
R = corr(X)

# Diagonal is always 1
np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-6)

# Bounded in [-1, 1]
assert jnp.all(R >= -1.0 - 1e-6)
assert jnp.all(R <= 1.0 + 1e-6)

# Matches numpy.corrcoef
R_np = np.corrcoef(np.array(X))
np.testing.assert_allclose(R, R_np, atol=1e-5)
```

```{tip}
Both `cov` and `corr` accept the same `weight` and `l2` parameters. Use
`l2` with `corr` when you need a well-conditioned correlation matrix (e.g.
before taking the matrix logarithm for tangent-space connectivity).
```

## 3. Partial correlation: direct connections

Pearson correlation conflates direct and indirect associations. If region A
connects to B and B connects to C, the A-C correlation will be nonzero even
with no direct connection. **Partial correlation** removes these indirect
effects by conditioning on all other variables.

### The math

From the precision matrix $P = \Sigma^{-1}$:

$$
\text{partial\_corr}(i, j) = -\frac{P_{ij}}{\sqrt{P_{ii}\, P_{jj}}}
$$

The diagonal is set to 1 by convention.

### Usage

```python
from jaxoccoli.covariance import partial_corr

pc = partial_corr(X)
print(f"Shape: {pc.shape}")  # (5, 5)

# Diagonal is 1
np.testing.assert_allclose(jnp.diagonal(pc), 1.0, atol=1e-5)

# Bounded in [-1, 1]
assert jnp.all(pc >= -1.0 - 1e-5)
assert jnp.all(pc <= 1.0 + 1e-5)
```

```{note}
`partial_corr` uses a default `l2=1e-6` for numerical stability. For
small sample sizes ($T < C$), increase this to prevent singular covariance
matrices. A common heuristic is `l2 = 0.1 / T`.
```

### Precision matrix

You can also access the precision matrix directly:

```python
from jaxoccoli.covariance import precision

P = precision(X, l2=0.01)
S = cov(X, l2=0.01)

# P is the inverse of S
eye = jnp.eye(5)
np.testing.assert_allclose(S @ P, eye, atol=1e-4)
```

## 4. Posterior correlation: the key innovation

`posterior_corr()` is the uncertainty-aware connectivity estimator at the heart
of hippy-feat. Given posterior distributions on beta parameters from the
Bayesian GLM, it computes a disattenuated correlation that accounts for
estimation uncertainty.

### The problem with naive correlation

When you correlate point-estimate betas across trials, you are computing:

$$
r_{\text{obs}}(i, j) = \text{corr}(\hat{\mu}_i, \hat{\mu}_j)
$$

But what you actually want is the correlation of the **true** betas:

$$
r_{\text{true}}(i, j) = \text{corr}(\beta_i, \beta_j)
$$

These differ because estimation noise attenuates the observed correlation
(Spearman 1904). The correction factor depends on the **reliability** of each
region's estimates:

$$
\rho_i = \frac{\text{Var}(\mu_i)}{\text{Var}(\mu_i) + \mathbb{E}[\sigma^2_i]}
$$

where $\text{Var}(\mu_i)$ is the across-trial variance of the posterior means
("signal") and $\mathbb{E}[\sigma^2_i]$ is the mean posterior variance
("noise"). The corrected correlation is:

$$
r_{\text{post}}(i, j) = \frac{r_{\text{obs}}(i, j)}{\sqrt{\rho_i \cdot \rho_j}}
$$

### Usage

```python
from jaxoccoli.covariance import posterior_corr

k1, k2 = jax.random.split(key)
beta_mean = jax.random.normal(k1, (5, 60))  # 5 regions, 60 trials
beta_var = jnp.abs(jax.random.normal(k2, (5, 60))) + 0.01

R = posterior_corr(beta_mean, beta_var)
print(f"Shape: {R.shape}")  # (5, 5)

# Diagonal is 1
np.testing.assert_allclose(jnp.diagonal(R), 1.0, atol=1e-5)
```

### Zero variance recovers ordinary correlation

When posterior variance is zero (i.e. perfect certainty), `posterior_corr`
reduces to standard Pearson correlation of the means:

```python
beta_mean = jax.random.normal(key, (5, 60))
beta_var = jnp.zeros((5, 60))

R1 = corr(beta_mean)
R2 = posterior_corr(beta_mean, beta_var)
np.testing.assert_allclose(R1, R2, atol=1e-5)
```

This is the expected behaviour: with no uncertainty, there is nothing to correct.

### High variance inflates naive correlation

This demonstrates **why uncertainty matters**. When posterior variance is high
relative to the variance of means, the naive correlation is attenuated and the
correction is larger:

```python
k1, k2 = jax.random.split(key)
beta_mean = jax.random.normal(k1, (5, 60))

# Low uncertainty
beta_var_low = 0.01 * jnp.ones((5, 60))
R_low = posterior_corr(beta_mean, beta_var_low)

# High uncertainty
beta_var_high = 10.0 * jnp.ones((5, 60))
R_high = posterior_corr(beta_mean, beta_var_high)

# High variance leads to larger correction (larger off-diagonal magnitudes)
mask = 1 - jnp.eye(5)
mean_abs_low = jnp.mean(jnp.abs(R_low) * mask)
mean_abs_high = jnp.mean(jnp.abs(R_high) * mask)
print(f"Mean |r| (low var):  {mean_abs_low:.4f}")
print(f"Mean |r| (high var): {mean_abs_high:.4f}")
# High-variance case applies larger correction
```

```{note}
The correction can push correlations toward $\pm 1$ when reliability is low.
`posterior_corr` clips the result to $[-1, 1]$ to maintain validity. This
is a known property of disattenuation corrections -- it signals that the
data may not support reliable connectivity estimation for those region pairs.
```

## 5. Attenuated correlation: measurement error correction

`attenuated_corr()` applies the classical Spearman (1904) correction for
attenuation when you have an external estimate of each variable's reliability
(e.g. from test-retest or split-half analysis):

$$
r_{\text{corrected}}(i, j) = \frac{r_{\text{observed}}(i, j)}{\sqrt{\text{rel}_i \cdot \text{rel}_j}}
$$

where $\text{rel}_i \in (0, 1]$ is the reliability of variable $i$.

```python
from jaxoccoli.covariance import attenuated_corr

# Perfect reliability: no correction
rel_perfect = jnp.ones(5)
R_obs = corr(X)
R_corrected = attenuated_corr(X, rel_perfect)
np.testing.assert_allclose(R_obs, R_corrected, atol=1e-6)
```

With imperfect reliability, off-diagonal magnitudes increase:

```python
# 50% reliability: substantial correction
rel_half = 0.5 * jnp.ones(5)
R_corrected = attenuated_corr(X, rel_half)

# Off-diagonal correlations are inflated (corrected upward)
mask = 1 - jnp.eye(5)
assert jnp.mean(jnp.abs(R_corrected) * mask) >= jnp.mean(jnp.abs(R_obs) * mask) - 1e-6

# Still bounded in [-1, 1]
assert jnp.all(R_corrected >= -1.0 - 1e-6)
assert jnp.all(R_corrected <= 1.0 + 1e-6)
```

```{tip}
`posterior_corr` is conceptually similar to `attenuated_corr`, but computes
the reliability internally from the posterior variances. Use `attenuated_corr`
when you have external reliability estimates; use `posterior_corr` when your
reliability comes from the Bayesian GLM.
```

## 6. Weighted correlation: reliability weighting

`weighted_corr()` downweights unreliable observations (trials) rather than
correcting after the fact. This is useful when different trials have different
noise levels:

```python
from jaxoccoli.covariance import weighted_corr

# Uniform weights reproduce ordinary correlation
w_uniform = jnp.ones(100)
R1 = corr(X)
R2 = weighted_corr(X, w_uniform)
np.testing.assert_allclose(R1, R2, atol=1e-4)

# Non-uniform weights: downweight noisy trials
w = jax.random.uniform(key, (100,)) + 0.1  # ensure positive
R_weighted = weighted_corr(X, w)
print(f"Shape: {R_weighted.shape}")  # (5, 5)
```

A common pattern is to use $w_t = 1 / \text{SE}_t$ from the Bayesian GLM, so
that high-uncertainty trials contribute less to the connectivity estimate.

## 7. Why uncertainty matters: a demonstration

Let us put the pieces together to see how ignoring variance leads to biased
connectivity.

```python
import jax
import jax.numpy as jnp
from jaxoccoli.covariance import corr, posterior_corr

key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)

# True signal: 5 regions, 60 trials, moderate correlation
true_signal = jax.random.normal(k1, (5, 60))

# Add heterogeneous noise (some regions are very noisy)
noise_scales = jnp.array([0.1, 0.1, 5.0, 5.0, 0.1])  # regions 2,3 are noisy
noise = noise_scales[:, None] * jax.random.normal(k2, (5, 60))

# Observed (point-estimate) betas
observed = true_signal + noise

# Naive correlation of noisy point estimates
R_naive = corr(observed)

# Posterior correlation with known uncertainty
# beta_var = noise_variance per region per trial
beta_var = jnp.broadcast_to(noise_scales[:, None] ** 2, (5, 60))
R_posterior = posterior_corr(observed, beta_var)

print("Naive FC (noisy regions attenuated):")
print(jnp.round(R_naive, 2))
print()
print("Posterior FC (uncertainty-corrected):")
print(jnp.round(R_posterior, 2))
```

The naive correlation between noisy regions (2-3) and clean regions (0-1, 4)
will be attenuated toward zero. The posterior correlation corrects for this,
recovering estimates closer to the true connectivity.

## 8. JAX transformations: JIT, grad, vmap

All covariance functions are compatible with JAX's function transformations.

### JIT compilation

```python
# JIT any function for XLA compilation
R_jit = jax.jit(corr)(X)
pc_jit = jax.jit(partial_corr)(X)
S_jit = jax.jit(cov)(X)
```

### Automatic differentiation

Gradients flow through all covariance estimators, enabling gradient-based
optimisation of connectivity objectives:

```python
def connectivity_loss(X):
    return jnp.sum(cov(X) ** 2)

grad_X = jax.grad(connectivity_loss)(X)
assert grad_X.shape == X.shape
assert jnp.all(jnp.isfinite(grad_X))
```

For `posterior_corr`, gradients flow with respect to both means and variances:

```python
def posterior_loss(bm, bv):
    return jnp.sum(posterior_corr(bm, bv) ** 2)

bm = jax.random.normal(k1, (5, 60))
bv = jnp.abs(jax.random.normal(k2, (5, 60))) + 0.01

g_mean, g_var = jax.grad(posterior_loss, argnums=(0, 1))(bm, bv)
assert g_mean.shape == bm.shape
assert g_var.shape == bv.shape
assert jnp.all(jnp.isfinite(g_mean))
assert jnp.all(jnp.isfinite(g_var))
```

### Batched computation with vmap

Process multiple subjects or sessions in parallel:

```python
# 10 subjects, each with 5 regions and 50 time points
batch = jax.random.normal(key, (10, 5, 50))
S_batch = jax.vmap(cov)(batch)
print(f"Batch covariance shape: {S_batch.shape}")  # (10, 5, 5)
```

```{note}
Combining `jit` and `vmap` is the recommended pattern for production use.
For example, `jax.jit(jax.vmap(posterior_corr))` compiles a single XLA kernel
that processes all subjects in parallel, achieving near-linear scaling on GPU.
```

## Summary

The `jaxoccoli.covariance` module provides a progression from standard to
uncertainty-aware connectivity:

| Method | Accounts for noise? | Requires uncertainty? | Key use case |
|--------|---------------------|-----------------------|-------------|
| `corr()` | No | No | Baseline FC |
| `partial_corr()` | No | No | Direct connections |
| `weighted_corr()` | Partially (downweights) | Weights per trial | Heterogeneous trial quality |
| `attenuated_corr()` | Yes (correction) | External reliability | Known test-retest reliability |
| `posterior_corr()` | Yes (correction) | From Bayesian GLM | End-to-end variance propagation |

The recommended pipeline for hippy-feat is:

1. Fit the conjugate GLM to get `(beta_mean, beta_var)` per voxel
2. Parcellate with `make_atlas_linear_uncertain()` preserving both quantities
3. Compute connectivity with `posterior_corr()` for uncertainty-corrected FC

See the companion tutorial {doc}`bayesian_variance_propagation` for the full
end-to-end pipeline.
