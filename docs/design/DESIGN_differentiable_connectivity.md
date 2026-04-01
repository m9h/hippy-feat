# Differentiable Connectivity Analysis in jaxoccoli

## Motivation

Standard fMRI connectivity analysis is a multi-stage pipeline where each stage discards information:

```
BOLD timeseries  -->  OLS GLM  -->  point-estimate betas  -->  Pearson correlation  -->  FC matrix
                      ^                    ^                        ^
                      |                    |                        |
                  noise model          variance lost           heteroscedasticity
                  assumed white        at this arrow            ignored
```

This creates three problems:

1. **Variance propagation gap** (Rissman et al. 2004; Mumford et al. 2012): Trial-wise beta estimates are noisy point estimates, but their standard errors are discarded before the correlation step. Noisy betas attenuate observed correlations (measurement error attenuation) and mask true connectivity.

2. **Non-differentiability**: The pipeline cannot be optimized end-to-end. Parcellation, covariance estimation, and downstream objectives (modularity, decoding) are separate stages with no gradient flow between them.

3. **Scalability**: Full eigendecomposition of the graph Laplacian is O(N^3), making vertex-wise spectral analysis on cortical surfaces (32k+ nodes) impractical.

## Solution

jaxoccoli implements a fully differentiable, variance-propagating connectivity pipeline in plain JAX:

```
BOLD timeseries
  |
  v
make_conjugate_glm  -->  (beta_mean, beta_var)      [closed-form Bayesian, ~0.5ms/voxel]
  |
  v
make_atlas_linear_uncertain  -->  (parc_mean, parc_var)  [W @ mean, W^2 @ var]
  |
  v
posterior_corr  -->  FC matrix                       [disattenuated by reliability]
  |
  v
modularity_loss / eigenmaps_loss / decoder           [differentiable objectives]
  |
  v
jax.grad  -->  gradients through entire pipeline
```

## Architecture

### Design pattern: vbjax factories

All learnable components follow the vbjax pattern:

```python
params, forward_fn = make_*(config, key=key)
output = forward_fn(params, input)
grads = jax.grad(loss)(params)
params = optax.apply_updates(params, optimizer.update(grads, opt_state))
```

Parameters are NamedTuples (immutable, JAX pytree compatible). No Equinox, no Flax.

### Module dependency graph

```
covariance.py    matrix.py     fourier.py       transport.py
     |               |              |                |
     v               v              v                |
  graph.py      interpolate.py                       |
     |                                               |
     v                                               |
 learnable.py  <--- (imports covariance, fourier) ---|
     |                                               |
     v                                               |
  losses.py    <--- (imports graph)                  |
     |                                               |
     v                                               v
bayesian_beta.py                         [cross-subject comparison]
```

No circular dependencies. Each module is independently testable.

## Variance-aware extensions

### The beta series variance gap

The Rissman/Mumford problem in concrete terms:

| Step | Input | Output | What is lost |
|------|-------|--------|-------------|
| GLM (OLS) | BOLD timeseries | beta (point estimate) | Standard error of beta |
| Parcellation | beta per voxel | beta per parcel | Voxel-level uncertainty |
| Correlation | beta series | FC matrix | Trial-level heteroscedasticity |

### Three solutions at increasing complexity

**1. `weighted_corr(X, weights)`** -- Simple reliability weighting

Downweights noisy trials using `weights = 1 / beta_std`. Implemented as weighted covariance → weighted correlation. Addresses heteroscedasticity across trials.

**2. `attenuated_corr(X, reliabilities)`** -- Spearman (1904) correction

Corrects observed correlation for measurement error attenuation:
```
r_corrected(i,j) = r_observed(i,j) / sqrt(reliability_i * reliability_j)
```
where `reliability = var(true) / var(observed)`. Analytically removes the bias introduced by noisy betas.

**3. `posterior_corr(beta_mean, beta_var)`** -- Full posterior marginalisation

Computes the expected correlation under the posterior distribution of betas:
```
reliability_i = var(mu_i across trials) / (var(mu_i) + E[sigma_i^2])
r_corrected = r_observed / sqrt(rel_i * rel_j)
```
This is the proper Bayesian solution: it uses the posterior variance from the conjugate GLM to estimate reliability per region, then corrects.

### Variance-aware parcellation

`make_atlas_linear_uncertain` propagates variance through linear parcellation:
```
parc_mean = W @ beta_mean        # standard parcellation
parc_var  = W^2 @ beta_var       # variance propagation (diagonal case)
```
where `W` is the (softmax-normalised) parcellation weight matrix. This is exact when trial-wise betas are independent (the typical assumption).

## Bayesian GLM

### Conjugate path (real-time compatible)

`make_conjugate_glm` implements BROCCOLI's Gibbs Block 1 as a single closed-form step:

**Model:**
```
y = X @ beta + epsilon,    epsilon ~ N(0, sigma^2 I)
beta | sigma^2 ~ N(mu_0, sigma^2 * Lambda^{-1})
sigma^2 ~ InverseGamma(a_0, b_0)
```

**Posterior:**
```
Omega_post = (X^T X + Lambda)^{-1}
beta_post  = Omega_post @ (X^T y + Lambda @ mu_0)
a_post     = a_0 + T/2
b_post     = b_0 + 0.5 * (RSS + prior quadratic form)
sigma^2    = b_post / (a_post - 1)
beta_var   = sigma^2 * diag(Omega_post)
```

The key matrices `(X^T X + Lambda)^{-1}` and `(X^T X + Lambda)^{-1} X^T` are precomputed once and reused across all voxels. The per-voxel computation is a matrix-vector multiply (~0.5ms).

### AR(1) prewhitened path

`make_ar1_conjugate_glm` extends the conjugate GLM with autocorrelation correction:

1. Estimate rho from OLS residuals (Yule-Walker)
2. Shrink toward prior: `rho = (rho_precision * rho_prior + data_precision * rho_ols) / total`
3. Update prewhitened cross-products analytically: `X_pw^T X_pw = S00 - 2*rho*S01 + rho^2*S11`
4. Apply conjugate GLM to prewhitened data

The S matrices are precomputed from the design matrix, following BROCCOLI's GPU trick of storing cross-products in constant memory.

### NUTS path (offline)

`make_bayesian_glm` uses blackjax NUTS for full posterior sampling with AR(p) noise, per-voxel HRF parameters, and complete posterior summaries. This is Variant G from the Bayesian first-level design doc.

## Spectral methods

### Chebyshev filtering (from hgx)

Full eigendecomposition of the Laplacian is O(N^3) and impractical for cortical surfaces (N=32k). The Chebyshev polynomial filter applies spectral operations without eigendecomposition:

```
h(L) @ x = sum_k a_k T_k(L_tilde) @ x
```

using the recurrence `T_{k+1} = 2 * L_tilde @ T_k - T_{k-1}`. This is O(K * nnz) where K is the polynomial order (typically 5-10). The coefficients `a_k` are learnable via `make_chebyshev_filter`.

### Sparse message passing

For vertex-wise analysis, dense N x N matrix operations are replaced by sparse `segment_sum` operations that are O(E) where E = number of edges:

```python
# Dense: O(N^2)
out = W @ x

# Sparse: O(E)
out = sparse_graph_conv(x, source_idx, target_idx, num_nodes, weights)
```

`adjacency_to_edge_index` converts dense connectivity matrices to COO-format edge indices.

## Optimal transport

### Wasserstein FC distance

Compares two functional connectivity matrices by treating upper-triangle entries as 1D distributions and computing the 1-Wasserstein distance via sorted CDFs (closed form, exact):

```python
d = wasserstein_fc_distance(fc1, fc2)
```

### Gromov-Wasserstein for cross-subject comparison

Compares the *structure* of two connectivity matrices without requiring node correspondence. This enables comparing subjects with different parcellations or vertex counts:

```python
T, cost = gromov_wasserstein_fc(fc_subject1, fc_subject2)
```

`T` is a soft alignment between the two sets of brain regions. Uses iterative linearisation with entropic Sinkhorn (log-domain stabilised).

## Fisher-Rao natural gradient

For atlas/parcellation optimization, the parcel weights live on the probability simplex (each row sums to 1 via softmax). Standard Euclidean gradients ignore the geometry of this manifold.

The Fisher-Rao natural gradient for categorical distributions is:
```
natural_grad = p * euclidean_grad
```

This gives geometry-aware updates that respect the simplex structure. `make_atlas_natural_grad` provides a factory with a built-in update function using this approach, adapted from hgx's information geometry module.

## Testing strategy

Each module has dedicated tests verifying:

1. **Shape correctness** -- output dimensions match expectations
2. **Numerical accuracy** -- comparison against numpy/scipy reference implementations
3. **JIT compatibility** -- `jax.jit(fn)` produces identical results
4. **Differentiability** -- `jax.grad(loss_fn)` produces finite, non-zero gradients
5. **vmap compatibility** -- batched operation matches sequential
6. **Mathematical properties** -- symmetry, positive definiteness, bounded eigenvalues, marginal constraints (Sinkhorn), simplex projection

The end-to-end integration test in `test_bayesian_beta.py::TestVariancePropagation` verifies that gradients flow through the complete pipeline: conjugate GLM -> uncertain atlas -> posterior correlation -> loss.

## Future work

- **Sparse Chebyshev on cortical surfaces**: Apply `chebyshev_filter` with sparse Laplacian on fsLR 32k mesh
- **Wavelet scattering for FC**: Multi-scale stable features from connectivity matrices (following hgx pattern)
- **Spatial 3D Matern priors**: Full Bayesian spatial smoothing (Siden et al. 2017)
- **Sequential Monte Carlo**: Incremental posterior updates per TR for streaming Bayesian analysis
- **Hypergraph parcellation**: Represent multi-voxel parcels as hyperedges using hgx primitives
