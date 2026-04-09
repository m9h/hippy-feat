"""DOT/fNIRS → cortical surface adapter for jaxoccoli.

Bridges dot-jax FEM mesh outputs (HbO/HbR per node) to jaxoccoli's
cortical analysis pipeline. The key operation is projecting from an
irregular FEM mesh to a standard cortical surface (e.g. fsaverage5)
via nearest-node interpolation.

Pipeline:
    dot-jax RealtimePipeline.process_frame()
      → (hbo, hbr) on FEM mesh (nn,)
      → make_mesh_to_cortex projection (nn → n_vertices)
      → make_cortical_projection parcellation (n_vertices → n_parcels)
      → jaxoccoli FC / graph / RSA tools

Factory:
    ``make_mesh_to_cortex(mesh_nodes, cortex_vertices)``
        → ``(MeshToCortexParams, forward_fn)``

Class:
    ``DOTFrameProcessor`` — per-frame consumer with sliding FC

References
----------
Eggebrecht AT et al. (2014) Mapping distributed brain function and
    networks with diffuse optical tomography. Nature Photonics 8:448-454.
Sherafati A et al. (2025) VHD-DOT. Imaging Neuroscience.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial import cKDTree

from .hf_encoder import make_cortical_projection
from .connectivity import SlidingWindowConnectivity


# ---------------------------------------------------------------------------
# Simulation helpers (for testing without dot-jax)
# ---------------------------------------------------------------------------

def simulate_dot_mesh_nodes(n_nodes: int, seed: int = 0) -> np.ndarray:
    """Generate simulated FEM mesh node positions in head space.

    Produces points on/near a head-shaped ellipsoid (~90×110×100 mm radii).

    Parameters
    ----------
    n_nodes : number of mesh nodes
    seed : random seed

    Returns
    -------
    (n_nodes, 3) float32 positions in mm
    """
    rng = np.random.RandomState(seed)
    # Ellipsoidal shell (head-like)
    theta = rng.uniform(0, 2 * np.pi, n_nodes)
    phi = rng.uniform(0, np.pi, n_nodes)
    r = rng.uniform(60, 90, n_nodes)
    x = r * np.sin(phi) * np.cos(theta)  # LR: ~90mm
    y = r * np.sin(phi) * np.sin(theta) * 1.2  # AP: ~110mm
    z = r * np.cos(phi) * 1.1  # SI: ~100mm
    return np.stack([x, y, z], axis=1).astype(np.float32)


def simulate_hbo_frame(n_nodes: int, seed: int = 0) -> jnp.ndarray:
    """Generate a simulated HbO frame with realistic amplitude.

    Parameters
    ----------
    n_nodes : FEM mesh node count
    seed : random seed

    Returns
    -------
    (n_nodes,) HbO change in micromolar
    """
    rng = np.random.RandomState(seed)
    # Background noise + focal activation
    hbo = rng.randn(n_nodes).astype(np.float32) * 0.5  # µM noise
    # Add a focal activation in a random region
    center = rng.randint(0, n_nodes)
    spread = max(n_nodes // 20, 10)
    start = max(0, center - spread)
    end = min(n_nodes, center + spread)
    hbo[start:end] += 3.0  # µM activation
    return jnp.array(hbo)


# ---------------------------------------------------------------------------
# Mesh-to-cortex projection
# ---------------------------------------------------------------------------

class MeshToCortexParams(NamedTuple):
    """Parameters for FEM mesh → cortical surface projection.

    Attributes
    ----------
    nearest_idx : (n_vertices,) index of nearest mesh node per cortex vertex
    distance_mask : (n_vertices,) 1.0 if within max_distance, else 0.0
    """
    nearest_idx: jnp.ndarray
    distance_mask: jnp.ndarray


def make_mesh_to_cortex(
    mesh_nodes: np.ndarray,
    cortex_vertices: np.ndarray,
    max_distance: float | None = None,
) -> tuple[MeshToCortexParams, callable]:
    """Create a nearest-node projection from FEM mesh to cortical surface.

    For each cortex vertex, finds the nearest FEM mesh node and assigns
    its value. Optionally masks vertices beyond ``max_distance`` from
    any mesh node (sets them to zero — outside DOT field of view).

    Parameters
    ----------
    mesh_nodes : (n_mesh, 3) FEM mesh node coordinates (mm)
    cortex_vertices : (n_cortex, 3) cortical surface coordinates (mm)
    max_distance : optional cutoff (mm); vertices farther get zero

    Returns
    -------
    (MeshToCortexParams, forward_fn)
        forward_fn(params, mesh_signal) → cortex_signal
        mesh_signal: (n_mesh,) → cortex_signal: (n_cortex,)
    """
    tree = cKDTree(np.asarray(mesh_nodes))
    distances, indices = tree.query(np.asarray(cortex_vertices))

    nearest_idx = jnp.array(indices, dtype=jnp.int32)

    if max_distance is not None:
        distance_mask = jnp.array(
            (distances <= max_distance).astype(np.float32)
        )
    else:
        distance_mask = jnp.ones(len(cortex_vertices), dtype=jnp.float32)

    params = MeshToCortexParams(
        nearest_idx=nearest_idx,
        distance_mask=distance_mask,
    )

    def forward_fn(
        params: MeshToCortexParams,
        mesh_signal: jnp.ndarray,
    ) -> jnp.ndarray:
        """Project mesh signal to cortical surface via nearest-node lookup."""
        projected = mesh_signal[params.nearest_idx]
        return projected * params.distance_mask

    return params, forward_fn


# ---------------------------------------------------------------------------
# DOTFrameProcessor
# ---------------------------------------------------------------------------

class DOTFrameProcessor:
    """Real-time DOT frame consumer with cortical projection and sliding FC.

    Processes each (hbo, hbr) frame from dot-jax:
        1. Project HbO/HbR from FEM mesh to cortical surface
        2. Parcellate (vertex → parcel)
        3. Update sliding window buffer
        4. Compute FC matrix

    Parameters
    ----------
    mesh_nodes : (n_mesh, 3) FEM node positions
    cortex_vertices : (n_cortex, 3) cortical surface positions
    n_parcels : output parcel count
    window_size : sliding window for FC
    max_distance : mesh-to-cortex distance cutoff (mm)
    key : JAX PRNG key
    """

    def __init__(
        self,
        mesh_nodes: np.ndarray,
        cortex_vertices: np.ndarray,
        n_parcels: int = 100,
        window_size: int = 20,
        max_distance: float | None = None,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        n_vertices = cortex_vertices.shape[0]
        self.n_parcels = n_parcels
        self.window_size = window_size
        self.frames_processed = 0

        # Mesh → cortex projection
        self.mesh_params, self.mesh_forward = make_mesh_to_cortex(
            mesh_nodes, cortex_vertices, max_distance=max_distance,
        )

        # Cortex → parcels projection
        self.proj_params, self.proj_forward = make_cortical_projection(
            n_vertices=n_vertices, n_parcels=n_parcels,
            key=key, init="block",
        )

        # Sliding FC
        self.swc = SlidingWindowConnectivity(n_parcels, window_size)
        self.buffer = jnp.zeros((n_parcels, window_size))

    def process_frame(
        self,
        hbo: jnp.ndarray,
        hbr: jnp.ndarray,
    ) -> dict:
        """Process one frame of HbO/HbR from dot-jax.

        Parameters
        ----------
        hbo : (n_mesh,) oxyhemoglobin change per mesh node
        hbr : (n_mesh,) deoxyhemoglobin change per mesh node

        Returns
        -------
        Dict with: hbo_cortex, hbr_cortex, parcellated_hbo, fc
        """
        # Step 1: mesh → cortex
        hbo_cortex = self.mesh_forward(self.mesh_params, hbo)
        hbr_cortex = self.mesh_forward(self.mesh_params, hbr)

        # Step 2: cortex → parcels
        parcellated_hbo = self.proj_forward(
            self.proj_params, hbo_cortex[None, :]
        )[0]

        # Step 3: sliding FC
        self.buffer, fc = self.swc.update_and_compute(
            self.buffer, parcellated_hbo,
        )

        self.frames_processed += 1

        return {
            "hbo_cortex": hbo_cortex,
            "hbr_cortex": hbr_cortex,
            "parcellated_hbo": parcellated_hbo,
            "fc": fc,
        }
