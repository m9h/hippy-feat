# Differentiable Parcellation: Learning Atlases with Uncertainty

Traditional fMRI analysis assigns each voxel to a parcel using a fixed
atlas -- Schaefer, Glasser, AAL -- and then averages within each region.
This is convenient but lossy: hard boundaries ignore partial-volume
effects, individual anatomical variability, and the fact that the
"optimal" grouping depends on the downstream task (e.g., connectivity
estimation vs. activation decoding).

hippy-feat's `jaxoccoli.learnable` module replaces fixed atlases with
**differentiable, learnable parcellations** that can be optimised
end-to-end alongside connectivity estimation.  Every component follows
the vbjax factory pattern -- `make_*() -> (params, forward_fn)` -- so
there are no framework dependencies beyond JAX itself.

This tutorial builds up the full differentiable parcellation pipeline,
from constraint primitives through variance-aware atlas operations.

## Prerequisites

```python
import jax
import jax.numpy as jnp
import numpy as np

from jaxoccoli.learnable import (
    make_simplex_constraint,
    make_spd_constraint,
    make_orthogonal_constraint,
    make_atlas_linear,
    make_atlas_linear_uncertain,
    make_learnable_cov,
)

KEY = jax.random.PRNGKey(13)
```

---

## 1. Soft Parcellation with `make_atlas_linear()`

### The factory pattern

`make_atlas_linear` returns a parameter object and a forward function that
maps voxel-level data to parcel-level data through a learnable soft
assignment matrix:

```python
params, forward = make_atlas_linear(
    n_voxels=100,    # input dimensionality (voxels or vertices)
    n_parcels=10,    # output dimensionality (parcels)
    key=KEY,         # JAX PRNG key for initialisation
)
```

The returned `params` is an `AtlasParams` named tuple containing a single
field `weight` of shape `(n_parcels, n_voxels)` -- raw logits that are
converted to normalised assignment probabilities via softmax in the forward
pass.

### Forward pass

The forward function applies the soft assignment:

$$
\mathbf{Y} = \mathbf{W}\,\mathbf{X}, \qquad
W_{pv} = \frac{\exp(w_{pv})}{\sum_{v'}\exp(w_{pv'})}
$$

where $\mathbf{X}$ is `(n_voxels, T)` voxel-level time series and
$\mathbf{Y}$ is `(n_parcels, T)` parcellated output.

```python
# Single-subject data: 100 voxels, 50 time points
data = jax.random.normal(KEY, (100, 50))
parcellated = forward(params, data)
assert parcellated.shape == (10, 50)

# Batched data: 8 subjects
batch_data = jax.random.normal(KEY, (8, 100, 50))
batch_parcellated = forward(params, batch_data)
assert batch_parcellated.shape == (8, 10, 50)
```

```{admonition} Automatic batching
:class: tip
The forward function uses `jnp.einsum('pv,...vt->...pt', w, data)`,
which handles arbitrary leading batch dimensions without explicit `vmap`.
```

### Softmax normalisation guarantees

Because the raw logits pass through `jax.nn.softmax(axis=-1)`, the
effective weight matrix always satisfies the simplex constraint -- each
row sums to one and all entries are non-negative:

```python
w = jax.nn.softmax(params.weight, axis=-1)
np.testing.assert_allclose(jnp.sum(w, axis=-1), 1.0, atol=1e-5)
assert jnp.all(w >= 0)
```

---

## 2. Simplex Constraints with `make_simplex_constraint()`

For general-purpose constrained optimisation on the probability simplex
(not just atlas weights), jaxoccoli provides a reusable constraint factory.

### Project and unproject

The `ConstraintFns` named tuple provides two functions:

- **`project(x)`**: Maps unconstrained logits to the simplex via softmax.
- **`unproject(p)`**: Approximate inverse via log-transform (maps simplex
  back to unconstrained space).

```python
constraint = make_simplex_constraint()
x = jax.random.normal(KEY, (10,))

# Project to simplex
p = constraint.project(x)
np.testing.assert_allclose(jnp.sum(p), 1.0, atol=1e-6)
assert jnp.all(p >= 0)

# Round-trip: project -> unproject -> project recovers the same distribution
x2 = constraint.unproject(p)
p2 = constraint.project(x2)
np.testing.assert_allclose(p, p2, atol=1e-5)
```

### Temperature control

The `temperature` parameter controls the sharpness of the resulting
distribution.  Mathematically, the projection is:

$$
p_k = \frac{\exp(x_k / \tau)}{\sum_{k'}\exp(x_{k'} / \tau)}
$$

- **High temperature** ($\tau \gg 1$): Nearly uniform distribution
  (soft assignment).
- **Low temperature** ($\tau \ll 1$): Nearly one-hot distribution
  (hard assignment).

```python
hot = make_simplex_constraint(temperature=10.0)
cold = make_simplex_constraint(temperature=0.1)

x = jax.random.normal(KEY, (10,))
p_hot = hot.project(x)
p_cold = cold.project(x)

# Cold temperature produces a more peaked distribution
assert jnp.max(p_hot) < jnp.max(p_cold)
```

```{admonition} Annealing strategy
:class: note
A common pattern is to start with high temperature (soft parcellation,
smooth gradients) and anneal toward low temperature (hard parcellation,
near-discrete assignments) over the course of training.
```

---

## 3. Variance-Aware Parcellation with `make_atlas_linear_uncertain()`

### The variance propagation problem

Bayesian GLM estimators (e.g., from `jaxoccoli.glm`) output posterior means
**and** variances for each voxel's beta coefficients.  Standard parcellation
discards the variance -- averaging the means but ignoring the uncertainty.
This is the "Rissman/Mumford beta series variance gap."

`make_atlas_linear_uncertain` closes this gap by propagating both quantities
through the linear atlas operation.

### Mathematical formulation

Given the soft weight matrix $\mathbf{W}$ (obtained via softmax), and
per-voxel posterior mean $\boldsymbol{\mu} \in \mathbb{R}^{V \times T}$
and variance $\boldsymbol{\sigma}^2 \in \mathbb{R}^{V \times T}$ (diagonal
approximation):

$$
\hat{\boldsymbol{\mu}}_p = \sum_v W_{pv}\,\mu_v
\qquad\text{(mean propagation)}
$$

$$
\hat{\sigma}^2_p = \sum_v W_{pv}^2\,\sigma^2_v
\qquad\text{(variance propagation)}
$$

The variance formula follows from the fact that
$\mathrm{Var}[\mathbf{a}^\top\mathbf{x}] = \mathbf{a}^\top\,\mathrm{diag}(\boldsymbol{\sigma}^2)\,\mathbf{a}$
for independent voxels.  Note the **squared** weights -- this means parcels
with more diffuse weights (spread across many voxels) accumulate less
variance than parcels concentrated on a few high-variance voxels.

### Usage

```python
params, forward = make_atlas_linear_uncertain(
    n_voxels=100,
    n_parcels=10,
    key=KEY,
)

k1, k2 = jax.random.split(KEY)
beta_mean = jax.random.normal(k1, (100, 50))
beta_var = jnp.abs(jax.random.normal(k2, (100, 50))) + 0.01

parc_mean, parc_var = forward(params, beta_mean, beta_var)
assert parc_mean.shape == (10, 50)
assert parc_var.shape == (10, 50)
```

### Key properties

**Non-negative variance**: Since $W_{pv}^2 \geq 0$ and $\sigma^2_v \geq 0$,
the output variance is guaranteed non-negative:

```python
assert jnp.all(parc_var >= 0)
```

**Zero-in, zero-out**: If all input variances are zero (perfectly known
betas), the output variance is exactly zero:

```python
bm = jax.random.normal(KEY, (50, 30))
bv = jnp.zeros((50, 30))

params_z, fwd_z = make_atlas_linear_uncertain(50, 5, key=KEY)
_, pv = fwd_z(params_z, bm, bv)
np.testing.assert_allclose(pv, 0.0, atol=1e-7)
```

**Mean consistency**: The mean output matches `make_atlas_linear` exactly
(same softmax weights, same einsum):

```python
k = KEY
params_u, fwd_u = make_atlas_linear_uncertain(50, 5, key=k)
params_s, fwd_s = make_atlas_linear(50, 5, key=k)

bm = jax.random.normal(KEY, (50, 30))
bv = jnp.abs(jax.random.normal(KEY, (50, 30))) + 0.01

pm_u, _ = fwd_u(params_u, bm, bv)
pm_s = fwd_s(params_s, bm)
np.testing.assert_allclose(pm_u, pm_s, atol=1e-5)
```

---

## 4. SPD Constraints with `make_spd_constraint()`

Covariance and precision matrices must be **symmetric positive definite**
(SPD).  When optimising over matrix-valued parameters, unconstrained
gradient descent can produce matrices with negative eigenvalues.

`make_spd_constraint` provides a projection onto the SPD cone via eigenvalue
clamping:

$$
\text{project}(\mathbf{X}) = \mathbf{U}\,\mathrm{diag}\bigl(\max(\lambda_i, \varepsilon)\bigr)\,\mathbf{U}^\top
$$

where $\mathbf{X} = \mathbf{U}\,\mathrm{diag}(\boldsymbol{\lambda})\,\mathbf{U}^\top$
is the eigendecomposition and $\varepsilon$ is a small positive floor
(default `1e-6`).

```python
spd = make_spd_constraint()
X = jax.random.normal(KEY, (5, 5))

S = spd.project(X)

# Symmetric
np.testing.assert_allclose(S, S.T, atol=1e-6)

# Positive definite
eigvals = jnp.linalg.eigvalsh(S)
assert jnp.all(eigvals > 0)
```

```{admonition} When to use SPD projection
:class: note
Use this constraint when learning precision matrices, Riemannian metric
tensors, or kernel matrices that must remain positive definite throughout
optimisation.
```

---

## 5. Orthogonal Constraints with `make_orthogonal_constraint()`

Orthogonality constraints arise in spectral embeddings, ICA, and
dimensionality reduction.  `make_orthogonal_constraint` projects an
arbitrary matrix onto the Stiefel manifold via QR decomposition:

$$
\text{project}(\mathbf{X}) = \mathbf{Q}, \qquad \mathbf{X} = \mathbf{Q}\mathbf{R}
$$

where $\mathbf{Q}^\top\mathbf{Q} = \mathbf{I}$.

```python
orth = make_orthogonal_constraint()
X = jax.random.normal(KEY, (10, 4))

Q = orth.project(X)
assert Q.shape == (10, 4)

# Verify orthonormality
np.testing.assert_allclose(Q.T @ Q, jnp.eye(4), atol=1e-5)
```

---

## 6. Learnable Connectivity with `make_learnable_cov()`

After parcellation, functional connectivity is estimated from the
parcel-level time series.  `make_learnable_cov` adds learnable
**per-observation weights** to the covariance (or correlation) estimator,
allowing the model to up-weight informative time points and down-weight
noisy ones:

```python
params, forward = make_learnable_cov(
    dim=10,           # number of parcels
    time_dim=50,      # number of time points
    key=KEY,
    estimator='corr', # 'cov' or 'corr'
)

data = jax.random.normal(KEY, (10, 50))
conn_matrix = forward(params, data)

assert conn_matrix.shape == (10, 10)
# Correlation matrix is symmetric
np.testing.assert_allclose(conn_matrix, conn_matrix.T, atol=1e-5)
```

The observation weights are stored as raw logits and normalised via softmax
internally, so they always sum to `time_dim` and remain non-negative.

---

## 7. Gradient-Based Optimisation of Atlas Parameters

Every factory function returns differentiable components.  Atlas weights,
covariance weights, and constraint parameters can all be optimised with
standard JAX gradient-based methods.

### Computing gradients through the atlas

```python
params, forward = make_atlas_linear(50, 5, key=KEY)
data = jax.random.normal(KEY, (50, 30))

def loss(params):
    return jnp.sum(forward(params, data) ** 2)

grads = jax.grad(loss)(params)

# Gradients have the same structure as params
assert grads.weight.shape == params.weight.shape
assert jnp.all(jnp.isfinite(grads.weight))
```

### Gradients through uncertain parcellation

The variance propagation path is also fully differentiable:

```python
params, forward = make_atlas_linear_uncertain(50, 5, key=KEY)
k1, k2 = jax.random.split(KEY)
bm = jax.random.normal(k1, (50, 30))
bv = jnp.abs(jax.random.normal(k2, (50, 30))) + 0.01

def loss(params):
    pm, pv = forward(params, bm, bv)
    return jnp.sum(pm ** 2) + jnp.sum(pv)

grads = jax.grad(loss)(params)
assert jnp.all(jnp.isfinite(grads.weight))
```

### Gradients through learnable connectivity

```python
params, forward = make_learnable_cov(10, 50, key=KEY)
data = jax.random.normal(KEY, (10, 50))

def loss(params):
    return jnp.sum(forward(params, data) ** 2)

grads = jax.grad(loss)(params)
assert grads.weight.shape == params.weight.shape
assert jnp.all(jnp.isfinite(grads.weight))
```

### JIT compilation

All forward functions are JIT-compatible:

```python
params, forward = make_atlas_linear(50, 5, key=KEY)
data = jax.random.normal(KEY, (50, 30))
out = jax.jit(forward)(params, data)
assert out.shape == (5, 30)
```

---

## 8. End-to-End: Learning Atlas and Computing Uncertain Connectivity

Here is a complete example that chains together the differentiable
parcellation pipeline: learn soft atlas weights, propagate variance through
parcellation, and estimate weighted connectivity -- all optimised jointly.

```python
import optax

# --- Setup ---
n_voxels, n_parcels, n_timepoints = 200, 10, 100

key = jax.random.PRNGKey(42)
k1, k2, k3 = jax.random.split(key, 3)

# Simulated Bayesian GLM output: voxel-level means and variances
beta_mean = jax.random.normal(k1, (n_voxels, n_timepoints))
beta_var = jnp.abs(jax.random.normal(k2, (n_voxels, n_timepoints))) + 0.01

# Target connectivity (e.g., from a group-level prior)
target_conn = jnp.eye(n_parcels)

# --- Build differentiable pipeline ---
atlas_params, atlas_fwd = make_atlas_linear_uncertain(
    n_voxels, n_parcels, key=k3,
)
cov_params, cov_fwd = make_learnable_cov(
    n_parcels, n_timepoints, key=k3, estimator='corr',
)

# Combined parameters (use a simple tuple for this example)
all_params = (atlas_params, cov_params)

def pipeline(params):
    a_params, c_params = params

    # Step 1: Variance-aware parcellation
    parc_mean, parc_var = atlas_fwd(a_params, beta_mean, beta_var)

    # Step 2: Learnable connectivity from parcellated means
    conn = cov_fwd(c_params, parc_mean)

    # Step 3: Loss = connectivity reconstruction + variance penalty
    conn_loss = jnp.sum((conn - target_conn) ** 2)
    var_penalty = jnp.sum(parc_var)  # encourage low-variance parcels
    return conn_loss + 0.01 * var_penalty

# --- Optimise ---
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(all_params)

for step in range(200):
    loss_val, grads = jax.value_and_grad(pipeline)(all_params)
    updates, opt_state = optimizer.update(grads, opt_state)
    all_params = optax.apply_updates(all_params, updates)
    if step % 50 == 0:
        print(f"Step {step:3d}  loss={loss_val:.4f}")
```

```{admonition} What is being learned?
:class: important
This optimisation jointly adjusts:

1. **Atlas weights** (`atlas_params.weight`) -- which voxels contribute to
   which parcels, reshaping the parcellation to minimise the downstream
   connectivity loss.
2. **Observation weights** (`cov_params.weight`) -- which time points are
   most informative for connectivity estimation, effectively learning a
   temporal weighting scheme.

The variance penalty encourages parcellations that group together
low-variance (high-confidence) voxels, producing more reliable
connectivity estimates.
```

---

## Summary

This tutorial covered the differentiable parcellation pipeline in
`jaxoccoli.learnable`:

| Component | Factory | Purpose |
|-----------|---------|---------|
| Soft atlas | `make_atlas_linear()` | Learnable voxel-to-parcel mapping |
| Uncertain atlas | `make_atlas_linear_uncertain()` | Propagate mean + variance |
| Simplex constraint | `make_simplex_constraint()` | Non-negative, sum-to-one |
| SPD constraint | `make_spd_constraint()` | Positive-definite matrices |
| Orthogonal constraint | `make_orthogonal_constraint()` | Stiefel manifold |
| Learnable connectivity | `make_learnable_cov()` | Weighted covariance/correlation |

All components are:
- **Differentiable**: Gradients flow through atlas weights, variance
  propagation, and connectivity estimation.
- **JIT-compilable**: Compatible with `jax.jit`, `jax.vmap`, and
  `jax.lax.scan` for high-performance training loops.
- **Framework-free**: Plain JAX + NamedTuples, following the vbjax
  `make_*() -> (params, forward_fn)` convention.

```{seealso}
- {doc}`/design/index` for the overall hippy-feat architecture.
- {doc}`motion_correction` for preprocessing steps that produce the
  aligned volumes fed into parcellation.
```
