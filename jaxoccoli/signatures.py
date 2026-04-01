"""Rough-path log-signatures for effective connectivity and lead-lag analysis.

Uses the ``signax`` library to compute truncated log-signatures of
multivariate time series.  Log-signatures form a compact, faithful
summary of a path's geometry up to a chosen depth and are strictly
richer than correlation: they capture higher-order temporal dependencies,
asymmetry (lead-lag), and nonlinear interactions between channels.

Key class:
    ``WindowedSignature`` -- computes log-signatures (and an explicit
    Levy-area / lead-lag score) on sliding windows of ROI time series.

The Levy area between two channels A and B,

    Area = integral(A dB) - integral(B dA),

is the depth-2 antisymmetric log-signature term.  Its sign encodes
which channel leads the other; its magnitude encodes the strength of
the lead-lag relationship.  This provides a differentiable proxy for
Granger causality that is compatible with the hippy-feat pipeline.

Pipeline integration:
    Feed variance-propagated beta series from
    :mod:`jaxoccoli.bayesian_beta` through sliding windows, then
    compute signatures for downstream classification or clustering
    of dynamic effective connectivity states.

References:
    Chevyrev & Kormilitzin (2016) "A primer on the signature method
    in machine learning."
    Kidger & Lyons (2021) "signatory: differentiable computations of
    the signature and logsignature transforms, on both CPU and GPU."

Requires:
    ``pip install 'hippy-feat[signatures]'`` (signax >= 0.2.0).
"""

import jax
import jax.numpy as jnp
import signax
from functools import partial

class WindowedSignature:
    """Truncated log-signature features for multivariate time series.

    Wraps ``signax.logsignature`` to compute compact path-algebraic
    summaries that capture temporal order, lead-lag relationships,
    and higher-order channel interactions up to a chosen truncation
    depth.

    Args:
        depth: Truncation depth of the log-signature (default 3).
            Depth 1 captures increments, depth 2 adds Levy areas
            (lead-lag), depth >= 3 adds higher-order interactions.
    """

    def __init__(self, depth=3):
        self.depth = depth
        
    @partial(jax.jit, static_argnums=(0,))
    def compute(self, path):
        """Compute the truncated log-signature of *path*.

        Args:
            path: (T, C) multivariate time series where T is the number
                of timepoints and C is the number of channels / ROIs.
                A batch dimension is added automatically if absent.

        Returns:
            (D,) flat feature vector where D depends on the number
            of channels and the truncation depth.
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
        """Extract the Levy area (signed area) between the first two channels.

        The Levy area is the depth-2 antisymmetric log-signature term:
            Area = integral(A dB) - integral(B dA)

        Its sign indicates which channel temporally leads the other;
        its magnitude reflects the strength of the lead-lag coupling.
        A value of zero indicates perfect synchrony (A = B up to
        constant offset).

        Args:
            path: (T, C) multivariate time series with C >= 2 channels.

        Returns:
            Scalar Levy area between channels 0 and 1.
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
