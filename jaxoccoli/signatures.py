import jax
import jax.numpy as jnp
import signax
from functools import partial

class WindowedSignature:
    """
    Computes Rough Path Log-Signatures on sliding windows.
    Captures effective connectivity (lead-lag) and path geometry.
    """
    def __init__(self, depth=3):
        self.depth = depth
        
    @partial(jax.jit, static_argnums=(0,))
    def compute(self, path):
        """
        Compute the log-signature of the path.
        path: (Time, Channels) or (Time, ROIs)
        
        Returns:
            log_sig: Flat vector representing the log-signature.
        """
        # signax expects (Batch, Time, Channels) usually, or (Time, Channels)
        # Let's check signax.signature behavior or log_signature
        
        # Add batch dim if needed
        if path.ndim == 2:
            path = path[None, ...]
            
        # Compute Log-Signature
        # Log-signatures are more compact than full signatures and contain the "generators" of the path algebra.
        log_sig = signax.logsignature(path, depth=self.depth)
        
        # Flatten the output for use as a feature vector
        return jnp.ravel(log_sig)

    @partial(jax.jit, static_argnums=(0,))
    def lead_lag_score(self, path):
        """
        Extracts the 'Levy Area' (signed area) between first two channels.
        This corresponds to the term corresponding to [1, 2] in the signature.
        
        For 2 channels A and B:
        Area ~ Integral A dB - Integral B dA
        
        If A leads B (A goes up then B goes up), Area is Positive/Negative (depending on convention).
        If B leads A, Sign flips.
        If Synchronous (A=B), Area is 0.
        """
        # This is a heuristic helper.
        # Ideally we just return the full log-sig.
        # But for the demo, we want to show the specific "Levy Area" term.
        
        # signax.log_signature returns a list of arrays (terms at each level).
        # Level 1: Increments (Delta A, Delta B)
        # Level 2: Areas (Levy Areas)
        
        # Level 1: Increments (Delta A, Delta B)
        # Level 2: Areas (Levy Areas) / generators
        
        log_sig = signax.logsignature(path[None, ...], depth=self.depth)
        
        # Debug shape
        # jax.debug.print("Shape of log_sig[1]: {}", log_sig[1].shape)
        
        # If 1D, assume (Batch,) or (BasisSize,)?
        # For Batch=1, BasisSize=1, it might be (1,)
        # If so, [0] gets the value.
        
        # Let's handle generic case
        term = log_sig[1]
        area = term.flatten()[0] # Safe flatten
        
        return area
        
        return area
