"""
Tests for smoke_test_tribe — simulated TRIBEv2 demo pipeline.

Covers:
  - Synthetic fsaverage5 BOLD generation with spatial patterns
  - Full pipeline: projection → sliding FC → spectral embedding → modularity
  - Latency measurement helpers
  - Output properties (shapes, ranges, symmetry)
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp

from smoke_test_tribe import (
    make_synthetic_bold,
    make_stimulus_schedule,
    run_tribe_pipeline,
    CORTICAL_REGIONS,
)


# ===========================================================================
# Fixtures
# ===========================================================================

N_VERTICES = 20484  # fsaverage5
N_PARCELS = 100
N_TIMESTEPS = 60  # 60 seconds of simulated stimulus


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def stimulus_schedule():
    return make_stimulus_schedule(n_timesteps=N_TIMESTEPS)


@pytest.fixture
def synthetic_bold(stimulus_schedule, rng_key):
    return make_synthetic_bold(
        stimulus_schedule,
        n_vertices=N_VERTICES,
        key=rng_key,
    )


# ===========================================================================
# Stimulus schedule
# ===========================================================================

class TestStimulusSchedule:
    """Tests for the synthetic stimulus timeline."""

    def test_shape(self):
        schedule = make_stimulus_schedule(n_timesteps=60)
        assert schedule.shape == (60,)

    def test_values_are_region_labels(self):
        """Each timestep should map to a known cortical region or rest."""
        schedule = make_stimulus_schedule(n_timesteps=60)
        valid_labels = set(CORTICAL_REGIONS.keys()) | {"rest"}
        for label in schedule:
            assert label in valid_labels, f"Unknown label: {label}"

    def test_has_multiple_conditions(self):
        """Schedule should contain at least visual and auditory blocks."""
        schedule = make_stimulus_schedule(n_timesteps=60)
        labels = set(schedule)
        assert len(labels) >= 2, f"Only found labels: {labels}"


# ===========================================================================
# Synthetic BOLD generation
# ===========================================================================

class TestSyntheticBold:
    """Tests for the simulated TRIBEv2 output."""

    def test_output_shape(self, synthetic_bold):
        assert synthetic_bold.shape == (N_TIMESTEPS, N_VERTICES)

    def test_output_dtype(self, synthetic_bold):
        assert synthetic_bold.dtype == jnp.float32

    def test_finite_values(self, synthetic_bold):
        assert jnp.all(jnp.isfinite(synthetic_bold))

    def test_temporal_variation(self, synthetic_bold):
        """Signal should change over time (not static)."""
        temporal_std = jnp.std(synthetic_bold, axis=0)
        # Most vertices should have nonzero temporal variation
        active_fraction = jnp.mean(temporal_std > 1e-6)
        assert active_fraction > 0.5

    def test_spatial_variation(self, synthetic_bold):
        """Different vertices should have different timecourses."""
        spatial_std = jnp.std(synthetic_bold, axis=1)
        assert jnp.all(spatial_std > 0)

    def test_visual_stimulus_activates_visual_region(self, stimulus_schedule, rng_key):
        """During visual blocks, visual cortex vertices should be more active."""
        bold = make_synthetic_bold(stimulus_schedule, n_vertices=N_VERTICES, key=rng_key)
        visual_times = [i for i, s in enumerate(stimulus_schedule) if s == "visual"]
        rest_times = [i for i, s in enumerate(stimulus_schedule) if s == "rest"]

        if len(visual_times) > 0 and len(rest_times) > 0:
            # Visual cortex is roughly the first ~20% of vertices in our layout
            v_start, v_end = CORTICAL_REGIONS["visual"]
            visual_during_stim = jnp.mean(jnp.abs(bold[jnp.array(visual_times)][:, v_start:v_end]))
            visual_during_rest = jnp.mean(jnp.abs(bold[jnp.array(rest_times)][:, v_start:v_end]))
            assert visual_during_stim > visual_during_rest


# ===========================================================================
# Full pipeline
# ===========================================================================

class TestTribePipeline:
    """Tests for the full simulated pipeline."""

    def test_returns_results_dict(self, synthetic_bold, rng_key):
        results = run_tribe_pipeline(
            synthetic_bold, n_parcels=N_PARCELS, key=rng_key,
        )
        assert isinstance(results, dict)

    def test_parcellated_shape(self, synthetic_bold, rng_key):
        results = run_tribe_pipeline(
            synthetic_bold, n_parcels=N_PARCELS, key=rng_key,
        )
        assert results["parcellated"].shape == (N_TIMESTEPS, N_PARCELS)

    def test_fc_matrix_shape(self, synthetic_bold, rng_key):
        results = run_tribe_pipeline(
            synthetic_bold, n_parcels=N_PARCELS, key=rng_key,
        )
        fc = results["fc"]
        assert fc.shape == (N_PARCELS, N_PARCELS)

    def test_fc_matrix_symmetric(self, synthetic_bold, rng_key):
        results = run_tribe_pipeline(
            synthetic_bold, n_parcels=N_PARCELS, key=rng_key,
        )
        fc = results["fc"]
        np.testing.assert_array_almost_equal(
            np.asarray(fc), np.asarray(fc.T), decimal=5,
        )

    def test_fc_diagonal_is_one(self, synthetic_bold, rng_key):
        results = run_tribe_pipeline(
            synthetic_bold, n_parcels=N_PARCELS, key=rng_key,
        )
        diag = jnp.diag(results["fc"])
        np.testing.assert_array_almost_equal(
            np.asarray(diag), np.ones(N_PARCELS), decimal=4,
        )

    def test_embedding_shape(self, synthetic_bold, rng_key):
        results = run_tribe_pipeline(
            synthetic_bold, n_parcels=N_PARCELS, key=rng_key,
        )
        emb = results["embedding"]
        assert emb.shape[0] == N_PARCELS
        assert emb.shape[1] <= 10  # k components

    def test_modularity_scalar(self, synthetic_bold, rng_key):
        results = run_tribe_pipeline(
            synthetic_bold, n_parcels=N_PARCELS, key=rng_key,
        )
        Q = results["modularity"]
        assert Q.shape == ()  # scalar
        assert jnp.isfinite(Q)

    def test_sliding_fc_shape(self, synthetic_bold, rng_key):
        results = run_tribe_pipeline(
            synthetic_bold, n_parcels=N_PARCELS, key=rng_key,
            window_size=20,
        )
        dfc = results["dynamic_fc"]
        n_windows = N_TIMESTEPS - 20 + 1
        assert dfc.shape == (n_windows, N_PARCELS, N_PARCELS)

    def test_latencies_recorded(self, synthetic_bold, rng_key):
        results = run_tribe_pipeline(
            synthetic_bold, n_parcels=N_PARCELS, key=rng_key,
        )
        assert "latencies" in results
        assert "projection_ms" in results["latencies"]
        assert "fc_ms" in results["latencies"]
        assert "embedding_ms" in results["latencies"]
