# Motion Correction: Bilateral Filtering and Gauss-Newton Registration

Subject motion is the single largest source of artefact in functional MRI.
Even sub-millimetre head movements create spurious intensity changes that
corrupt activation maps and inflate functional connectivity estimates.
hippy-feat provides a fully differentiable, JIT-compiled motion correction
pipeline built on two complementary stages: **edge-preserving smoothing**
to suppress thermal noise while retaining anatomical boundaries, and
**rigid-body registration** to align every volume to a reference.

This tutorial walks through both stages using the `jaxoccoli.spatial` and
`jaxoccoli.motion` modules, demonstrating how to go from a noisy, misaligned
volume to a stabilised time series -- all on GPU in pure JAX.

## Prerequisites

```python
import jax
import jax.numpy as jnp
import numpy as np
from scipy.ndimage import gaussian_filter

from jaxoccoli.spatial import bilateral_filter_3d
from jaxoccoli.motion import RigidBodyRegistration, GaussNewtonRegistration
```

---

## 1. Edge-Preserving Smoothing with `bilateral_filter_3d()`

### Why not a simple Gaussian blur?

Standard Gaussian smoothing suppresses noise but also blurs tissue
boundaries -- grey/white matter edges, sulcal banks, ventricle walls.
The bilateral filter avoids this by weighting neighbours with **two**
Gaussian kernels: one in **space** and one in **intensity**:

$$
\hat{I}(\mathbf{x}) = \frac{
  \sum_{\mathbf{y} \in \mathcal{N}(\mathbf{x})}
    \underbrace{e^{-\|\mathbf{x}-\mathbf{y}\|^2 / 2\sigma_s^2}}_{\text{spatial weight}}
    \;\underbrace{e^{-(I(\mathbf{x})-I(\mathbf{y}))^2 / 2\sigma_r^2}}_{\text{range weight}}
    \;I(\mathbf{y})
}{
  \sum_{\mathbf{y} \in \mathcal{N}(\mathbf{x})}
    e^{-\|\mathbf{x}-\mathbf{y}\|^2 / 2\sigma_s^2}
    \;e^{-(I(\mathbf{x})-I(\mathbf{y}))^2 / 2\sigma_r^2}
}
$$

Voxels on the opposite side of an intensity edge receive near-zero range
weight and therefore contribute almost nothing to the average.

### Parameters

| Parameter | Meaning | Typical value |
|-----------|---------|---------------|
| `sigma_spatial` | Spatial kernel width (voxels) | 1.0 -- 2.0 |
| `sigma_range` | Intensity similarity scale | 10 -- 50 (or `None` for auto) |
| `kernel_radius` | Half-width of the cubic neighbourhood | 2 (gives a 5x5x5 kernel) |
| `mask` | Boolean brain mask; voxels outside are left unchanged | Optional |

### Example: preserving a step edge

Create a synthetic volume with a sharp boundary and additive noise, then
filter it:

```python
# Synthetic volume: left half = 100, right half = 200
vol = jnp.zeros((40, 40, 40), dtype=jnp.float32)
vol = vol.at[:20, :, :].set(100.0)
vol = vol.at[20:, :, :].set(200.0)

# Add Gaussian noise
key = jax.random.PRNGKey(0)
noisy_vol = vol + jax.random.normal(key, vol.shape) * 15.0

# Bilateral filter -- small sigma_range preserves the edge
filtered = bilateral_filter_3d(
    noisy_vol,
    sigma_spatial=1.5,
    sigma_range=10.0,   # tight range -> strong edge preservation
    kernel_radius=2,
)
```

```{admonition} Checking edge preservation
:class: tip
After filtering, voxels well inside each region (e.g. 5+ voxels from the
boundary) should stay within ~5 intensity units of their original value,
and the contrast across the edge should remain above 80% of the original.
```

### Example: noise reduction on a uniform region

When there is no edge to protect, a **large** `sigma_range` lets the filter
average aggressively -- reducing standard deviation by 30% or more while
preserving the mean:

```python
# Uniform volume with noise (mean=500, std=30)
rng = np.random.RandomState(123)
uniform_noisy = jnp.array(
    500.0 + rng.randn(40, 40, 40).astype(np.float32) * 30.0
)

filtered = bilateral_filter_3d(
    uniform_noisy,
    sigma_spatial=1.5,
    sigma_range=50.0,   # wide range -> smooth uniformly
    kernel_radius=2,
)

interior = slice(5, 35)
orig_std = jnp.std(uniform_noisy[interior, interior, interior])
filt_std = jnp.std(filtered[interior, interior, interior])
print(f"Noise std: {orig_std:.1f} -> {filt_std:.1f}")
# Mean is preserved to within ~2 intensity units
```

### Using a brain mask

Pass a boolean mask to restrict filtering to brain voxels.  Voxels outside
the mask are returned unchanged -- no boundary artefacts from air/skull:

```python
mask = jnp.ones((40, 40, 40), dtype=bool)
mask = mask.at[:2, :, :].set(False)   # exclude 2-voxel border
mask = mask.at[-2:, :, :].set(False)

filtered = bilateral_filter_3d(
    noisy_vol,
    sigma_spatial=1.5,
    sigma_range=10.0,
    kernel_radius=2,
    mask=mask,
)

# Verify: outside the mask, values are untouched
outside = ~mask
assert jnp.array_equal(filtered[outside], noisy_vol[outside])
```

---

## 2. Rigid Body Registration Basics

Head motion in the scanner is well-modelled by a **rigid body transform**:
3 translations $(t_x, t_y, t_z)$ and 3 rotations $(\theta_x, \theta_y, \theta_z)$
-- six degrees of freedom (6-DOF).

The transform is represented as a $4 \times 4$ homogeneous affine matrix
composed in extrinsic XYZ order:

$$
\mathbf{M}(\mathbf{p}) = \mathbf{T}(t_x,t_y,t_z) \; \mathbf{R}_z(\theta_z) \; \mathbf{R}_y(\theta_y) \; \mathbf{R}_x(\theta_x)
$$

where each rotation matrix takes the standard Euler form, e.g.:

$$
\mathbf{R}_x(\theta) = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & \cos\theta & -\sin\theta & 0 \\
0 & \sin\theta & \cos\theta & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

Both `RigidBodyRegistration` and `GaussNewtonRegistration` expose a static
method `make_affine_matrix(params)` that constructs this matrix from a
6-element parameter vector `[tx, ty, tz, rx, ry, rz]`.

---

## 3. Adam-Based Registration with `RigidBodyRegistration`

The first-order solver treats registration as gradient descent on the MSE
loss between the template and the resampled moving volume:

$$
\mathcal{L}(\mathbf{p}) = \frac{1}{N}\sum_{i=1}^{N}
  \bigl(I_\mathrm{ref}(\mathbf{x}_i) - I_\mathrm{mov}(\mathbf{M}^{-1}(\mathbf{p})\,\mathbf{x}_i)\bigr)^2
$$

The optimisation loop uses `optax.adam` and is fully unrolled inside
`jax.lax.scan`, so the entire registration compiles to a single XLA program.

### Signature

```python
reg = RigidBodyRegistration(
    template,         # (X, Y, Z) reference volume
    vol_shape,        # tuple (X, Y, Z)
    step_size=0.1,    # Adam learning rate
    n_iter=50,        # number of gradient steps
)

best_params, registered = reg.register_volume(moving_image)
# best_params: (6,) array [tx, ty, tz, rx, ry, rz]
# registered:  (X, Y, Z) aligned volume
```

### Example

```python
# Create a template with spatial features
rng = np.random.RandomState(123)
raw = rng.randn(76, 90, 74).astype(np.float32)
template = jnp.array(gaussian_filter(raw, sigma=5.0) * 500 + 1000)
vol_shape = template.shape

# Adam solver
adam_reg = RigidBodyRegistration(
    template=template,
    vol_shape=vol_shape,
    step_size=0.1,
    n_iter=50,
)

# Register template to itself -> should recover near-zero params
params, aligned = adam_reg.register_volume(template)
print("Identity params:", params)  # expect ~[0, 0, 0, 0, 0, 0]
```

```{admonition} When to use Adam
:class: note
Adam is simpler and more robust to poor initialisation, making it a good
fallback for large motions (> 5 mm translation).  For small motions typical
of compliant subjects, Gauss-Newton converges faster.
```

---

## 4. Gauss-Newton Registration with `GaussNewtonRegistration`

### Second-order convergence

The Gauss-Newton solver achieves superlinear convergence by solving the
**damped normal equations** at each iteration:

$$
\boldsymbol{\delta} = \bigl(\mathbf{J}^\top\mathbf{J} + \lambda\,\mathbf{I}\bigr)^{-1}\,\mathbf{J}^\top\,\mathbf{r}
$$

$$
\mathbf{p}_{k+1} = \mathbf{p}_k - \boldsymbol{\delta}
$$

where $\mathbf{r} \in \mathbb{R}^N$ is the residual vector
$I_\mathrm{ref} - I_\mathrm{mov} \circ \mathbf{M}^{-1}$,
$\mathbf{J} \in \mathbb{R}^{N \times 6}$ is the Jacobian of the residual
with respect to the 6 parameters, and $\lambda$ is the Levenberg-Marquardt
damping factor.

### Memory-efficient Jacobian computation

Materialising the full $N \times 6$ Jacobian for a typical fMRI volume
($N \approx 500{,}000$) would cost ~12 MB per iteration.  Instead,
hippy-feat computes the two required products without ever forming the full
matrix:

- **$\mathbf{J}^\top \mathbf{r}$** via a single VJP (reverse-mode AD) --
  equivalent to `jax.grad` of $\tfrac{1}{2}\|\mathbf{r}\|^2$.
- **$\mathbf{J}^\top \mathbf{J}$** via 6 JVP (forward-mode AD) passes,
  one per basis tangent vector $\mathbf{e}_i$.  Each JVP yields the $i$-th
  column of $\mathbf{J}$, and the $6 \times 6$ Gram matrix is assembled
  from their dot products.

This keeps memory at $O(N)$ rather than $O(6N)$.

### Signature

```python
gn_reg = GaussNewtonRegistration(
    template,          # (X, Y, Z) reference volume
    vol_shape,         # tuple (X, Y, Z)
    n_iter=10,         # GN iterations (5--15 is typical)
    damping=1e-4,      # Levenberg-Marquardt lambda
)

best_params, registered = gn_reg.register_volume(moving_image)
```

### Example: recovering a known translation

```python
# Build the registration object
gn_reg = GaussNewtonRegistration(
    template=template,
    vol_shape=vol_shape,
    n_iter=10,
    damping=1e-3,
)

# Simulate a 2-voxel shift along the X axis
true_params = jnp.array([2.0, 0.0, 0.0, 0.0, 0.0, 0.0])
moved = gn_reg.apply_transform(template, true_params)

# Recover the shift
recovered_params, registered = gn_reg.register_volume(moved)

# recovered_params are the INVERSE transform (moving -> template),
# so they should be approximately [-2, 0, 0, 0, 0, 0]
print("Recovered translations:", recovered_params[:3])
np.testing.assert_allclose(
    np.asarray(recovered_params[:3]),
    -np.asarray(true_params[:3]),
    atol=0.5,
)
```

```{admonition} Why the inverse?
:class: important
`register_volume` returns parameters that map the *moving* image back to
the *template*.  If you created the moving image by shifting +2 mm in X,
the recovered transform will be approximately -2 mm in X.
```

---

## 5. Applying Transforms with `apply_transform()`

Both registration classes expose `apply_transform(image, params)` for
resampling a volume under a given rigid-body transform.  Internally it:

1. Builds a centred homogeneous coordinate grid for the output volume.
2. Maps each output voxel back to source space via the inverse affine
   $\mathbf{M}^{-1}(\mathbf{p})$.
3. Interpolates the source volume with `jax.scipy.ndimage.map_coordinates`
   (first-order / linear interpolation, nearest-boundary extension).

```python
# Identity transform -> output equals input
identity = jnp.zeros(6)
result = gn_reg.apply_transform(template, identity)

np.testing.assert_allclose(
    np.asarray(result),
    np.asarray(template),
    atol=1e-4,
)
```

```{admonition} Precomputed grid in Gauss-Newton
:class: tip
`GaussNewtonRegistration` precomputes the coordinate grid once in
`__init__` and caches it as `self.coords_homo`.  This avoids redundant
grid construction across the iterative solver's inner loop -- a significant
saving when running 10+ GN steps per volume.
```

---

## 6. Gauss-Newton vs Adam: Iteration Efficiency

For small-to-moderate motions, Gauss-Newton reaches the same accuracy in
far fewer iterations thanks to its second-order curvature information.

```python
true_params = jnp.array([1.5, 0.0, 0.0, 0.0, 0.0, 0.0])
moved = gn_reg.apply_transform(template, true_params)
inv_true = -true_params

# Gauss-Newton: 10 iterations
gn_reg = GaussNewtonRegistration(
    template=template, vol_shape=vol_shape,
    n_iter=10, damping=1e-3,
)
gn_params, _ = gn_reg.register_volume(moved)
gn_error = float(jnp.sum((gn_params - inv_true) ** 2))

# Adam: same 10-iteration budget
adam_reg = RigidBodyRegistration(
    template=template, vol_shape=vol_shape,
    step_size=0.1, n_iter=10,
)
adam_params, _ = adam_reg.register_volume(moved)
adam_error = float(jnp.sum((adam_params - inv_true) ** 2))

print(f"GN error:   {gn_error:.4f}")
print(f"Adam error: {adam_error:.4f}")
# GN error will be substantially lower
```

---

## 7. JIT Compilation for Real-Time Performance

Every function in the motion correction pipeline is decorated with
`@partial(jax.jit, static_argnums=...)`, so the first call triggers XLA
compilation and all subsequent calls run at full speed.

### JIT-compiling the bilateral filter

```python
jitted_filter = jax.jit(
    lambda v: bilateral_filter_3d(
        v, sigma_spatial=1.5, sigma_range=50.0, kernel_radius=2
    )
)

# First call: compiles (~seconds)
result = jitted_filter(noisy_vol)

# Subsequent calls: ~milliseconds
result = jitted_filter(noisy_vol)
assert not jnp.any(jnp.isnan(result))
```

### JIT-compiled registration

Both `register_volume` and `apply_transform` are already JIT-compiled
through the class decorators.  The `jax.lax.scan`-unrolled optimisation
loop compiles to a single fused kernel:

```python
gn_reg = GaussNewtonRegistration(
    template=template, vol_shape=vol_shape, n_iter=5,
)

# First call compiles the full registration loop
params, registered = gn_reg.register_volume(template)
assert params.shape == (6,)
assert registered.shape == vol_shape
assert not jnp.any(jnp.isnan(params))
```

```{admonition} Compilation overhead
:class: warning
The first call to `register_volume` on a new volume shape will take several
seconds for XLA compilation.  Plan for this warm-up in real-time pipelines
by registering a dummy volume at startup.
```

---

## 8. Practical Tips

### Reference volume selection

The choice of reference (template) volume affects registration quality:

- **Middle volume**: Use the temporal midpoint of the time series.
  Motion tends to be smallest near the middle, reducing interpolation
  artefacts.
- **Mean volume**: Average all volumes first.  Provides higher SNR but
  is blurred if motion is large.
- **Minimum-motion volume**: Compute framewise displacement across a
  preliminary registration pass and pick the volume with the smallest FD.

### Motion thresholds

| Metric | Acceptable | Concerning | Exclude |
|--------|-----------|------------|---------|
| Max translation | < 1 mm | 1 -- 3 mm | > 3 mm |
| Max rotation | < 1 deg | 1 -- 2 deg | > 2 deg |
| Framewise displacement | < 0.5 mm | 0.5 -- 1.0 mm | > 1.0 mm |

### Choosing a solver

| Criterion | Adam | Gauss-Newton |
|-----------|------|--------------|
| Typical iterations | 30 -- 50 | 5 -- 15 |
| Memory | O(N) | O(N) |
| Large motion (> 5 mm) | More robust | May need warm start |
| Small motion (< 2 mm) | Converges slowly | Fast convergence |
| Damping parameter | N/A | `damping=1e-4` to `1e-2` |

### Bilateral filter parameter selection

- **`sigma_range=None`** (auto): Estimates `sigma_range` from the data
  using the median absolute deviation (MAD), scaled by 1.4826 for Gaussian
  equivalence.  Good default for most data.
- **Low `sigma_range`** (5 -- 15): Aggressive edge preservation.  Use for
  data with strong tissue contrast.
- **High `sigma_range`** (40 -- 80): Approaches a standard Gaussian smooth.
  Use for homogeneous regions or heavy noise.

---

## Summary

This tutorial covered the two-stage motion correction pipeline in hippy-feat:

1. **`bilateral_filter_3d()`** -- edge-preserving spatial smoothing that
   reduces noise without blurring tissue boundaries, fully JIT-compilable
   with optional brain mask support.

2. **`RigidBodyRegistration`** -- first-order Adam optimiser for 6-DOF
   rigid-body alignment, suitable as a robust baseline.

3. **`GaussNewtonRegistration`** -- second-order solver with
   Levenberg-Marquardt damping that achieves the same accuracy in 3--5x
   fewer iterations, using memory-efficient JVP-based Jacobian assembly.

All components follow the hippy-feat design philosophy: pure JAX, fully
differentiable, and JIT-compiled to a single XLA program for real-time
deployment.

```{seealso}
- {doc}`/design/index` for the overall hippy-feat architecture.
- {doc}`learnable_parcellation` for differentiable atlas operations on
  registered volumes.
```
