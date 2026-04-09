"""
Tests for jaxoccoli.dot_adapter — DOT/fNIRS → cortical surface adapter.

Bridges dot-jax FEM mesh outputs (HbO/HbR per node) to jaxoccoli's
cortical analysis pipeline (fsaverage5 vertices or parcels).

Covers:
  - make_mesh_to_cortex: nearest-node projection from FEM mesh to cortex
  - DOTFrameProcessor: per-frame HbO/HbR → cortical projection → FC
  - Integration with jaxoccoli connectivity tools
  - Compatibility with dot-jax RealtimePipeline output format
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from jaxoccoli.dot_adapter import (
    make_mesh_to_cortex,
    MeshToCortexParams,
    DOTFrameProcessor,
    simulate_dot_mesh_nodes,
    simulate_hbo_frame,
)


# ===========================================================================
# Fixtures
# ===========================================================================

N_MESH_NODES = 5000   # typical FEM mesh node count
N_VERTICES = 20484     # fsaverage5
N_PARCELS = 50


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def mesh_nodes():
    """Simulated FEM mesh node positions (N, 3) in mm."""
    return simulate_dot_mesh_nodes(N_MESH_NODES, seed=0)


@pytest.fixture
def cortex_vertices():
    """Simulated fsaverage5 vertex positions (N, 3) in mm."""
    return simulate_dot_mesh_nodes(N_VERTICES, seed=1)


@pytest.fixture
def hbo_frame():
    """Single frame of HbO data on FEM mesh."""
    return simulate_hbo_frame(N_MESH_NODES, seed=0)


# ===========================================================================
# make_mesh_to_cortex
# ===========================================================================

class TestMakeMeshToCortex:
    """Tests for the FEM mesh → cortical surface projection factory."""

    def test_returns_params_and_forward(self, mesh_nodes, cortex_vertices):
        params, forward_fn = make_mesh_to_cortex(
            mesh_nodes, cortex_vertices,
        )
        assert isinstance(params, MeshToCortexParams)
        assert callable(forward_fn)

    def test_forward_output_shape(self, mesh_nodes, cortex_vertices, hbo_frame):
        params, forward_fn = make_mesh_to_cortex(
            mesh_nodes, cortex_vertices,
        )
        projected = forward_fn(params, hbo_frame)
        assert projected.shape == (N_VERTICES,)

    def test_forward_preserves_signal(self, mesh_nodes, cortex_vertices):
        """Non-zero mesh signal → non-zero cortical signal."""
        params, forward_fn = make_mesh_to_cortex(
            mesh_nodes, cortex_vertices,
        )
        signal = jnp.ones(N_MESH_NODES) * 5.0
        projected = forward_fn(params, signal)
        assert jnp.mean(jnp.abs(projected)) > 0

    def test_forward_batch(self, mesh_nodes, cortex_vertices):
        """Should handle (T, n_nodes) batch input."""
        params, forward_fn = make_mesh_to_cortex(
            mesh_nodes, cortex_vertices,
        )
        batch = jnp.ones((10, N_MESH_NODES))
        # vmap over time
        projected = jax.vmap(lambda x: forward_fn(params, x))(batch)
        assert projected.shape == (10, N_VERTICES)

    def test_nearest_node_assignment(self):
        """Each cortex vertex should map to the nearest mesh node."""
        mesh = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=np.float32)
        cortex = np.array([[1, 0, 0], [11, 0, 0]], dtype=np.float32)
        params, forward_fn = make_mesh_to_cortex(mesh, cortex)
        signal = jnp.array([100.0, 200.0, 300.0])
        projected = forward_fn(params, signal)
        # Vertex at (1,0,0) → nearest mesh node (0,0,0) = 100
        # Vertex at (11,0,0) → nearest mesh node (10,0,0) = 200
        np.testing.assert_array_almost_equal(
            np.asarray(projected), [100.0, 200.0], decimal=3,
        )

    def test_max_distance_masking(self, mesh_nodes, cortex_vertices):
        """Vertices far from any mesh node should get zero signal."""
        params, forward_fn = make_mesh_to_cortex(
            mesh_nodes, cortex_vertices, max_distance=1.0,  # very tight
        )
        signal = jnp.ones(N_MESH_NODES)
        projected = forward_fn(params, signal)
        # Many vertices will be too far from any mesh node
        n_zero = jnp.sum(projected == 0.0)
        assert n_zero > 0, "With tight max_distance, some vertices should be masked"


# ===========================================================================
# DOTFrameProcessor
# ===========================================================================

class TestDOTFrameProcessor:
    """Tests for the real-time DOT frame consumer."""

    def test_process_frame_returns_dict(self, mesh_nodes, cortex_vertices, rng_key):
        proc = DOTFrameProcessor(
            mesh_nodes, cortex_vertices,
            n_parcels=N_PARCELS, key=rng_key,
        )
        hbo = simulate_hbo_frame(N_MESH_NODES, seed=0)
        hbr = simulate_hbo_frame(N_MESH_NODES, seed=1) * -0.3
        result = proc.process_frame(hbo, hbr)
        assert isinstance(result, dict)

    def test_result_has_hbo_cortex(self, mesh_nodes, cortex_vertices, rng_key):
        proc = DOTFrameProcessor(
            mesh_nodes, cortex_vertices,
            n_parcels=N_PARCELS, key=rng_key,
        )
        hbo = simulate_hbo_frame(N_MESH_NODES, seed=0)
        hbr = simulate_hbo_frame(N_MESH_NODES, seed=1) * -0.3
        result = proc.process_frame(hbo, hbr)
        assert "hbo_cortex" in result
        assert result["hbo_cortex"].shape == (N_VERTICES,)

    def test_result_has_parcellated(self, mesh_nodes, cortex_vertices, rng_key):
        proc = DOTFrameProcessor(
            mesh_nodes, cortex_vertices,
            n_parcels=N_PARCELS, key=rng_key,
        )
        hbo = simulate_hbo_frame(N_MESH_NODES, seed=0)
        hbr = simulate_hbo_frame(N_MESH_NODES, seed=1) * -0.3
        result = proc.process_frame(hbo, hbr)
        assert "parcellated_hbo" in result
        assert result["parcellated_hbo"].shape == (N_PARCELS,)

    def test_result_has_fc_after_window(self, mesh_nodes, cortex_vertices, rng_key):
        proc = DOTFrameProcessor(
            mesh_nodes, cortex_vertices,
            n_parcels=N_PARCELS, window_size=5, key=rng_key,
        )
        for i in range(6):
            hbo = simulate_hbo_frame(N_MESH_NODES, seed=i)
            hbr = simulate_hbo_frame(N_MESH_NODES, seed=i + 100) * -0.3
            result = proc.process_frame(hbo, hbr)
        assert "fc" in result
        assert result["fc"].shape == (N_PARCELS, N_PARCELS)

    def test_frame_counter(self, mesh_nodes, cortex_vertices, rng_key):
        proc = DOTFrameProcessor(
            mesh_nodes, cortex_vertices,
            n_parcels=N_PARCELS, key=rng_key,
        )
        for i in range(3):
            hbo = simulate_hbo_frame(N_MESH_NODES, seed=i)
            proc.process_frame(hbo, hbo * -0.3)
        assert proc.frames_processed == 3


# ===========================================================================
# Simulate helpers
# ===========================================================================

class TestSimulateHelpers:

    def test_mesh_nodes_shape(self):
        nodes = simulate_dot_mesh_nodes(100, seed=0)
        assert nodes.shape == (100, 3)

    def test_mesh_nodes_in_head_range(self):
        """Nodes should be within plausible head dimensions (mm)."""
        nodes = simulate_dot_mesh_nodes(1000, seed=0)
        assert np.all(np.abs(nodes) < 200)

    def test_hbo_frame_shape(self):
        hbo = simulate_hbo_frame(500, seed=0)
        assert hbo.shape == (500,)

    def test_hbo_frame_realistic_range(self):
        """HbO changes should be in micromolar range."""
        hbo = simulate_hbo_frame(500, seed=0)
        assert np.all(np.abs(hbo) < 100)  # µM


# ===========================================================================
# Integration: DOT → jaxoccoli connectivity
# ===========================================================================

class TestIntegration:

    def test_dot_to_corr(self, mesh_nodes, cortex_vertices, rng_key):
        """DOT frames → cortical projection → parcellation → FC."""
        from jaxoccoli.covariance import corr

        proc = DOTFrameProcessor(
            mesh_nodes, cortex_vertices,
            n_parcels=N_PARCELS, window_size=10, key=rng_key,
        )
        for i in range(12):
            hbo = simulate_hbo_frame(N_MESH_NODES, seed=i)
            result = proc.process_frame(hbo, hbo * -0.3)

        fc = result["fc"]
        assert fc.shape == (N_PARCELS, N_PARCELS)
        # Diagonal should be ~1
        diag = jnp.diag(fc)
        assert jnp.all(jnp.isfinite(diag))
