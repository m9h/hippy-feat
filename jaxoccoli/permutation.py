import jax
import jax.numpy as jnp
from jax import random
from functools import partial
from .glm import GeneralLinearModel
from .stats import compute_t_stat

class PermutationTest:
    def __init__(self, glm: GeneralLinearModel, contrast: jnp.ndarray, seed: int = 42):
        self.glm = glm
        self.contrast = contrast
        self.key = random.PRNGKey(seed)
        
        # Precompute standard error components that don't change if X'X is constant
        # Note: If we just permute rows of X, X'X is constant.
        # So we only need to permute the pinv mapping or the data.
        # Permuting weights of pinv is equivalent to permuting rows of X.
        
        # pinv is (Regressors, Time).
        # We will apply permutation to the Time dimension of pinv.
        
    def _run_batch(self, key, data, n_perms_batch):
        """
        Runs a batch of permutations.
        Returns the Max-T statistic for each permutation in the batch.
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
        """
        Run the permutation test.
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
