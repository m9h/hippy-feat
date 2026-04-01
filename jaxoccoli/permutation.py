"""Non-parametric permutation testing with max-T family-wise correction.

Implements the Freedman-Lane approach to permutation-based inference for
the GLM: design-matrix rows (equivalently, pseudo-inverse columns) are
shuffled under the null hypothesis, and the maximum absolute t-statistic
across all voxels is recorded for each permutation.  The resulting
max-T null distribution controls the family-wise error rate (FWER)
without Gaussian assumptions.

Key class:
    ``PermutationTest`` -- wraps a :class:`~jaxoccoli.glm.GeneralLinearModel`
    and a contrast vector.  The ``run`` method partitions the requested
    number of permutations into JIT-compiled batches processed with
    ``jax.vmap`` for GPU-parallel execution.

Pipeline integration:
    This module consumes a fitted ``GeneralLinearModel`` and delegates
    per-permutation t-statistic computation to
    :func:`jaxoccoli.stats.compute_t_stat`.  It is the non-parametric
    counterpart to the closed-form p-values in :mod:`jaxoccoli.stats`.

References:
    Nichols & Holmes (2002) "Nonparametric permutation tests for
    functional neuroimaging: a primer with examples."
    Winkler et al. (2014) "Permutation inference for the general
    linear model."
"""

import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from .glm import GeneralLinearModel
from .stats import compute_t_stat

class PermutationTest:
    """Max-T permutation test with batched GPU-parallel execution.

    Shuffles the columns of the GLM pseudo-inverse (equivalent to
    permuting design-matrix rows under the Freedman-Lane scheme) and
    records the maximum absolute t-statistic across all voxels for
    each permutation.  The resulting null distribution is used for
    FWER-corrected thresholding.

    Batches of permutations are processed with ``jax.vmap`` and JIT
    compiled, giving near-linear GPU speedup.

    Args:
        glm: A fitted :class:`~jaxoccoli.glm.GeneralLinearModel`.
        contrast: (P,) contrast weight vector.
        seed: Integer seed for the JAX PRNG (default 42).
    """

    def __init__(self, glm: GeneralLinearModel, contrast: jnp.ndarray, seed: int = 42):
        self.glm = glm
        self.contrast = contrast
        self.key = random.PRNGKey(seed)
        
    def _run_batch(self, key, data, n_perms_batch):
        """Run a single batch of permutations via ``jax.vmap``.

        For each permutation: shuffle pseudo-inverse columns, refit
        betas, compute residuals, derive a t-map, and return the
        maximum absolute t-value.

        Args:
            key: JAX PRNG key.
            data: (..., T) voxel time series.
            n_perms_batch: Number of permutations in this batch
                (static for ``jax.random.split``).

        Returns:
            (n_perms_batch,) max-|t| values.
        """
        keys = random.split(key, n_perms_batch)
        
        # Define single permutation function
        def single_perm(k, d):
            # Generate random permutation of time indices
            # Or simpler: shuffle the pinv columns
            # pinv: (Regressors, Time)
            # We shuffle the Time axis.
            
            perm_idxs = random.permutation(k, self.glm.pinv.shape[1])
            permuted_pinv = self.glm.pinv[:, perm_idxs]
            
            # Re-fit
            # Betas = Y @ pinv.T
            # d: (..., Time)
            # permuted_pinv.T: (Time, Regressors)
            betas = jnp.tensordot(d, permuted_pinv.T, axes=(-1, 0))
            
            # Residuals?
            # residuals = Y - X_perm @ betas
            # X_perm: X[perm_idxs, :]
            X_perm = self.glm.X[perm_idxs, :]
            predicted = jnp.tensordot(betas, X_perm, axes=(-1, 1)) # (..., Time)
            residuals = d - predicted
            
            # T-stat
            # We assume X'X is invariant, so self.glm.XtX_inv is valid.
            df = self.glm.X.shape[0] - self.glm.X.shape[1]
            t_map = compute_t_stat(betas, residuals, self.glm.XtX_inv, self.contrast, df)
            
            # Max-T across volume
            return jnp.max(jnp.abs(t_map)) # Two-tailed max
            
        # Vmap over keys, keep data fixed (broadcasted or closure)
        # We need to reshape data to be (Voxels, Time) or just (..., Time)
        # single_perm takes (Key, Data)
        
        # To save memory, we might map over permutations but keep data as closure
        # However, jax.vmap requires mapping over args.
        
        batch_merger = jax.vmap(lambda k: single_perm(k, data))
        
        return batch_merger(keys)

    def run(self, data: jnp.ndarray, n_perms: int = 1000, batch_size: int = 100):
        """Execute the full permutation test and return the null distribution.

        Splits the requested number of permutations into JIT-compiled
        batches of size *batch_size* for controlled GPU memory usage.

        Args:
            data: (..., T) voxel time series (e.g. (V, T) for V voxels).
            n_perms: Total number of permutations (default 1000).
            batch_size: Permutations per JIT batch (default 100).

        Returns:
            (n_perms,) max-|t| null distribution.  To threshold a
            real t-map at FWER alpha, use
            ``jnp.percentile(null_dist, 100 * (1 - alpha))``.
        """
        n_batches = n_perms // batch_size
        max_t_dist = []
        
        # JIT the batch runner
        # n_perms_batch (index 2) must be static for random.split
        jit_batch = jax.jit(self._run_batch, static_argnums=(2,))
        
        for i in range(n_batches + 1):
            remaining = n_perms - len(max_t_dist)
            if remaining <= 0:
                break
            
            current_batch = min(remaining, batch_size)
            self.key, subkey = random.split(self.key)
            
            print(f"Running permutation batch {i+1}/{n_batches + 1} ({current_batch} perms)...")
            batch_max_t = jit_batch(subkey, data, current_batch)
            max_t_dist.append(batch_max_t)
            
        return jnp.concatenate(max_t_dist)
