# Real-Time fMRI: End-to-End Streaming Analysis

Real-time fMRI (rt-fMRI) enables closed-loop neurofeedback, intraoperative
brain mapping, and quality-control monitoring by processing each acquired
volume before the next TR arrives. The hard constraint is **latency**: every
stage -- file I/O, motion correction, GLM fitting, and statistical inference
-- must complete within the repetition time (typically 1-2 s). This tutorial
shows how to build a complete rt-fMRI analysis pipeline with jaxoccoli,
exploiting JIT compilation and GPU parallelism to fit comfortably inside that
budget.

## Pipeline overview

The per-volume processing chain implemented here is:

```
Scanner  -->  File I/O  -->  Motion Correction  -->  Flatten  -->  GLM fit  -->  Permutation Test
  (NIfTI)      (nibabel)     (RigidBodyRegistration)               (GeneralLinearModel)   (PermutationTest)
```

Every compute-bound stage is a pure JAX function compiled once via `jax.jit`;
after warmup, each subsequent call dispatches to pre-compiled XLA code with
near-zero Python overhead.

## 1. Design matrix and GLM setup for streaming data

In offline analysis you build the design matrix once from the full session.
In a streaming context you typically maintain a **sliding window** of the most
recent *W* volumes and re-fit the GLM on every arrival. The
{class}`~jaxoccoli.glm.GeneralLinearModel` precomputes the pseudo-inverse
$(X'X)^{-1}X'$ at construction time, so fitting all voxels reduces to a
single `jnp.tensordot` call.

```python
import jax.numpy as jnp
from jaxoccoli import GeneralLinearModel, PermutationTest

WINDOW_SIZE = 50      # sliding-window length (volumes)
N_VOXELS    = 100_000 # whole-brain voxel count (flattened)

# Dummy two-column design (task + intercept).
# In practice, build from stimulus onsets convolved with an HRF.
design = jnp.ones((WINDOW_SIZE, 2))

glm      = GeneralLinearModel(design)
contrast = jnp.array([1.0, 0.0])   # task vs. baseline
```

`GeneralLinearModel.__init__` performs the Cholesky factorisation of $X'X$
and stores the pseudo-inverse as `glm.pinv` (shape `(P, T)`). Because this
matrix depends only on the design, it is computed **once** and reused for
every incoming volume.

### Data layout

jaxoccoli expects the time axis last: `(V, T)` for voxelwise data or
`(X, Y, Z, T)` for volumetric. A pre-allocated sliding buffer keeps the
most recent *W* volumes in GPU memory:

```python
buffer = jnp.zeros((N_VOXELS, WINDOW_SIZE))
```

When a new volume arrives, shift and append:

```python
buffer = buffer.at[:, :-1].set(buffer[:, 1:])
buffer = buffer.at[:, -1].set(new_volume_flat)
```

## 2. JIT-compiled processing for latency constraints

The key to real-time performance is eliminating Python-level overhead from
the hot path. Wrap the entire per-volume computation in a single
`@jax.jit` function so that JAX traces it once during warmup and dispatches
compiled XLA code thereafter.

```python
import jax

@jax.jit
def process_volume(data_buffer):
    """GLM fit + 200 permutations in one compiled call."""
    betas, residuals = glm.fit(data_buffer)

    key = jax.random.PRNGKey(0)
    pt  = PermutationTest(glm, contrast, seed=42)
    max_t = pt._run_batch(key, data_buffer, 200)

    return jnp.max(max_t)
```

:::{important}
The first call to a JIT-compiled function triggers **tracing** -- JAX walks
through the Python code symbolically and compiles the resulting XLA graph.
This can take several seconds for a complex pipeline. Always run a warmup
call with dummy data before entering the acquisition loop:

```python
dummy = jax.random.normal(jax.random.PRNGKey(0), (N_VOXELS, WINDOW_SIZE))
_ = process_volume(dummy).block_until_ready()
```
:::

After warmup, `process_volume` dispatches directly to the compiled kernel.
The `.block_until_ready()` call forces synchronous execution so that the
measured wall-clock time reflects actual compute, not just dispatch latency.

## 3. Permutation testing for real-time statistical inference

The {class}`~jaxoccoli.permutation.PermutationTest` implements the max-T
permutation approach of Nichols & Holmes (2002) with `jax.vmap`-batched GPU
parallelism.

**How it works:**

1. For each permutation, the columns of the pseudo-inverse are shuffled
   (equivalent to permuting design-matrix rows under the Freedman-Lane
   scheme).
2. Betas are re-estimated and a full voxelwise t-map is computed via
   {func}`~jaxoccoli.stats.compute_t_stat`.
3. The maximum absolute t-value across all voxels is recorded.
4. The resulting null distribution controls the family-wise error rate
   (FWER) without Gaussian assumptions.

In the streaming setting, running the full 5000-permutation test every TR
is infeasible. Instead, run a **reduced batch** (e.g. 200 permutations) to
obtain a rough threshold, and accumulate a proper null distribution over
multiple TRs if needed:

```python
pt = PermutationTest(glm, contrast, seed=42)

# Fast path: 200 permutations per TR via the internal batch runner
key = jax.random.PRNGKey(0)
max_t_values = pt._run_batch(key, data_buffer, 200)

# Threshold at alpha=0.05 (95th percentile of the null)
threshold = jnp.percentile(max_t_values, 95.0)
```

For offline post-hoc analysis, use the public `run()` method which
automatically partitions permutations into memory-safe batches:

```python
null_dist = pt.run(data, n_perms=5000, batch_size=100)
threshold = jnp.percentile(null_dist, 100 * (1 - 0.05))
```

## 4. Motion correction with RigidBodyRegistration

Head motion is the dominant source of artefact in real-time fMRI. jaxoccoli
provides two 6-DOF rigid-body registration solvers in
{mod}`jaxoccoli.motion`:

| Solver | Method | Typical iterations | Best for |
|---|---|---|---|
| {class}`~jaxoccoli.motion.RigidBodyRegistration` | Adam (first-order) | ~50 | Real-time fallback, smooth convergence |
| {class}`~jaxoccoli.motion.GaussNewtonRegistration` | Gauss-Newton + LM damping | 5-15 | Production: fewer iterations, O(N) memory |

Both solvers:

- Precompute the homogeneous coordinate grid once.
- Unroll the optimisation loop with `jax.lax.scan` so registration is a
  single JIT-compiled call.
- Use `jax.scipy.ndimage.map_coordinates` for linear interpolation.

### Initialising from the first volume

The first acquired volume serves as the template. On each subsequent volume,
pass the **previous volume's motion parameters** as the initial guess to
exploit temporal smoothness:

```python
from jaxoccoli.motion import RigidBodyRegistration

VOL_SHAPE = (100, 100, 10)

# Set template from the first volume
template   = first_volume_jax           # (X, Y, Z) jnp array
registrator = RigidBodyRegistration(
    template, VOL_SHAPE, step_size=0.01, n_iter=20
)

mc_params = jnp.zeros(6)  # [tx, ty, tz, rx, ry, rz]
```

For real-time use, reduce `n_iter` (e.g. 20 instead of the default 50) to
trade a small amount of registration accuracy for lower latency.

### Per-volume registration

```python
best_params, registered_vol = registrator.register_volume(
    new_volume_jax, mc_params
)

# Carry forward for next volume
mc_params = best_params
```

The `GaussNewtonRegistration` solver is preferred for production pipelines.
It converges in 5-15 iterations by solving damped normal equations, and its
memory-efficient implementation builds $J'J$ from 6 JVP forward-mode passes
rather than materialising the full $(N \times 6)$ Jacobian:

```python
from jaxoccoli.motion import GaussNewtonRegistration

registrator = GaussNewtonRegistration(
    template, VOL_SHAPE, n_iter=10, damping=1e-4
)
```

## 5. Latency measurement -- fitting within the TR budget

The critical performance metric is whether end-to-end processing completes
within one TR. The smoke tests measure this with wall-clock timing around
`.block_until_ready()`:

```python
import time

latencies = []

for i in range(n_volumes):
    start = time.time()

    result = process_volume(current_buffer)
    result.block_until_ready()

    elapsed = time.time() - start
    latencies.append(elapsed)

    status = "OK" if elapsed < TR else "LATE"
    print(f"TR {i+1}: {elapsed:.4f}s [{status}]")
```

:::{tip}
The project's `smoke_test_realtime.py` benchmarks 10 TRs with 100k voxels,
a 100-timepoint sliding window, and 200 permutations per volume. On GPU, the
target is sub-2-second processing (i.e. within a TR=2.0s budget). If average
latency exceeds the TR, reduce `n_perms` or `WINDOW_SIZE`.
:::

**Latency breakdown for a typical volume (GPU):**

| Stage | Approximate time |
|---|---|
| NIfTI load (nibabel) | 5-15 ms |
| Motion correction (20 Adam iters) | 10-30 ms |
| GLM fit (100k voxels, 50 timepoints) | 1-5 ms |
| Permutation batch (200 perms, 100k voxels) | 50-200 ms |
| **Total** | **~70-250 ms** |

These numbers leave substantial headroom inside a 2 s TR. The dominant cost
is the permutation batch; scaling to 1000 permutations per TR is feasible on
modern GPUs.

## 6. Cloud/streaming architecture pattern

The `smoke_test_rt_cloud.py` script demonstrates a **producer-consumer**
architecture that mirrors real scanner-to-analysis data flow:

```
Scanner Thread              Analysis Thread (main)
     |                            |
     |--- write vol_000.nii -->   |
     |    (every TR)              |--- poll directory
     |                            |--- load NIfTI
     |                            |--- motion correct
     |                            |--- update buffer
     |                            |--- GLM + permutation
     |                            |--- report latency
```

### Scanner simulator (producer)

A background thread writes synthetic NIfTI volumes to a temporary directory
at the TR cadence, embedding acquisition timestamps in filenames for
end-to-end latency measurement:

```python
import threading, tempfile, nibabel as nib, numpy as np

class ScannerSimulator(threading.Thread):
    def __init__(self, out_dir, tr=2.0, n_volumes=10):
        super().__init__()
        self.out_dir = out_dir
        self.tr = tr
        self.n_volumes = n_volumes

    def run(self):
        for i in range(self.n_volumes):
            start_acq = time.time()
            vol = np.random.randn(100, 100, 10)
            img = nib.Nifti1Image(vol, np.eye(4))
            fname = f"vol_{i:03d}_{start_acq:.6f}.nii"
            nib.save(img, os.path.join(self.out_dir, fname))

            elapsed = time.time() - start_acq
            time.sleep(max(0, self.tr - elapsed))
```

### Real-time analyser (consumer)

The analyser polls the watch directory for new `.nii` files, loads each with
nibabel, and runs the full JIT-compiled pipeline. The key design decisions:

1. **Template from first volume** -- the first arriving file becomes the
   motion-correction reference.
2. **Warm-start motion parameters** -- each volume's estimated parameters
   initialise the next, exploiting temporal continuity.
3. **Sliding buffer update** -- shift-and-append maintains the GLM's
   temporal window.
4. **Fast polling** -- 50 ms sleep between directory checks to minimise
   detection latency without burning CPU.

```python
class RealTimeAnalyzer:
    def __init__(self, watch_dir):
        self.watch_dir = watch_dir
        self.processed_files = set()

        # Initialise JAX pipeline components
        self.design = jnp.ones((WINDOW_SIZE, 2))
        self.glm = GeneralLinearModel(self.design)
        self.contrast = jnp.array([1.0, 0.0])
        self.pt = PermutationTest(self.glm, self.contrast, seed=42)
        self.mc_params = jnp.zeros(6)
        self.registrator = None  # set from first volume

    @partial(jax.jit, static_argnums=(0,))
    def register_and_process(self, new_vol_3d, current_buffer,
                              template, init_params):
        # 1. Motion correction
        best_params, reg_vol = self.registrator.register_volume(
            new_vol_3d, init_params
        )

        # 2. Flatten and update sliding buffer
        data_flat = reg_vol.flatten()
        updated = current_buffer.at[:, :-1].set(current_buffer[:, 1:])
        updated = updated.at[:, -1].set(data_flat)

        # 3. GLM fit
        betas, residuals = self.glm.fit(updated)

        # 4. Permutation test (200 perms)
        key = jax.random.PRNGKey(0)
        max_t = self.pt._run_batch(key, updated, 200)

        return best_params, jnp.max(max_t), updated
```

### Running the simulation

```python
with tempfile.TemporaryDirectory() as tmpdir:
    scanner = ScannerSimulator(tmpdir, tr=2.0, n_volumes=10)
    scanner.start()

    analyzer = RealTimeAnalyzer(tmpdir)
    latencies = analyzer.run_loop(timeout=25.0)

    scanner.join()

    print(f"Mean lag: {np.mean(latencies):.4f}s")
    print(f"Max lag:  {np.max(latencies):.4f}s")
```

The smoke test reports two timing metrics:

- **Compute time** -- wall-clock duration of `register_and_process` alone.
- **Total lag** -- time from the scanner's acquisition timestamp (encoded in
  the filename) to analysis completion. This includes file I/O and directory
  polling overhead.

## 7. Complete pipeline script

The following script is a self-contained real-time fMRI analysis pipeline
that combines all the components above. It uses the simpler in-memory
approach (no file I/O) for benchmarking pure compute performance.

```python
"""Real-time fMRI pipeline benchmark.

Measures per-TR latency for: GLM fit + 200 permutations on 100k voxels.
"""

import time
import jax
import jax.numpy as jnp
import numpy as np
from jaxoccoli import GeneralLinearModel, PermutationTest

# --- Parameters ---
TR                = 2.0       # seconds
N_VOXELS          = 100_000
WINDOW_SIZE       = 100       # sliding-window length
N_VOLUMES         = 10        # simulated TRs

# --- Build pipeline components ---
design   = jnp.ones((WINDOW_SIZE, 2))
contrast = jnp.array([1.0, 0.0])

glm = GeneralLinearModel(design)
pt  = PermutationTest(glm, contrast, seed=42)

@jax.jit
def process_volume(data_buffer):
    betas, residuals = glm.fit(data_buffer)
    key   = jax.random.PRNGKey(0)
    max_t = pt._run_batch(key, data_buffer, 200)
    return jnp.max(max_t)

# --- Warmup (JIT trace + compile) ---
print("Warming up JIT...")
dummy = jax.random.normal(jax.random.PRNGKey(0), (N_VOXELS, WINDOW_SIZE))
_ = process_volume(dummy).block_until_ready()
print("Warmup complete.\n")

# --- Acquisition loop ---
buffer    = jnp.zeros((N_VOXELS, WINDOW_SIZE))
latencies = []

for i in range(N_VOLUMES):
    t0 = time.time()
    result = process_volume(buffer)
    result.block_until_ready()
    elapsed = time.time() - t0

    latencies.append(elapsed)
    status = "OK" if elapsed < TR else "LATE"
    print(f"TR {i+1:3d}: {elapsed:.4f}s [{status}]")

# --- Report ---
print(f"\nAverage latency: {np.mean(latencies):.4f}s")
print(f"Max latency:     {np.max(latencies):.4f}s")

if np.mean(latencies) < TR:
    print("PASS -- system meets real-time constraint.")
else:
    print("FAIL -- processing exceeds TR budget.")
```

## 8. Practical considerations

**Choosing permutation count.** The number of permutations per TR controls
the trade-off between statistical resolution and latency. For neurofeedback
where you only need a binary above/below-threshold signal, 100-200
permutations may suffice. For research-grade maps, accumulate permutations
across TRs into a running null distribution.

**Memory.** The sliding buffer occupies `N_VOXELS * WINDOW_SIZE * 4` bytes
(float32). For 100k voxels and a 100-volume window, that is ~40 MB -- well
within GPU memory. The permutation batch temporarily allocates
`n_perms * N_VOXELS * 4` bytes for the vmapped t-maps.

**Design matrix updates.** If the task paradigm changes mid-run (e.g.
adaptive stimulus selection), you must reconstruct the `GeneralLinearModel`
with the updated design. This invalidates the JIT cache for
`process_volume`, triggering a re-trace. To avoid this, consider padding the
design matrix to a fixed size and masking unused rows.

**GaussNewtonRegistration for production.** The second-order solver
converges in 5-15 iterations versus ~50 for Adam, reducing motion correction
latency by roughly 3x. Its memory-efficient $J'J$ assembly via forward-mode
JVP columns keeps memory at $O(N)$ rather than $O(6N)$.

## References

- Nichols, T.E. & Holmes, A.P. (2002). Nonparametric permutation tests for
  functional neuroimaging: a primer with examples. *Human Brain Mapping*,
  15(1), 1-25.
- Winkler, A.M. et al. (2014). Permutation inference for the general linear
  model. *NeuroImage*, 92, 381-397.
- Jenkinson, M. et al. (2002). Improved optimization for the robust and
  accurate linear registration and motion correction of brain images.
  *NeuroImage*, 17(2), 825-841.
