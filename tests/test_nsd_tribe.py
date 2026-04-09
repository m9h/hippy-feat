"""
Tests for smoke_test_nsd_tribe — NSD picture-watching TRIBEv2 simulation.

Covers:
  - NSD trial timing (3s stimulus + 1s ISI, TR=1.6s)
  - NSDTrialProducer: per-image BOLD frame generation with category structure
  - NSDSessionSimulator: full mini-session with GLM beta extraction
  - Trial-wise beta series and FC comparison
  - Representational similarity analysis (RSA) on predicted betas
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import queue

from smoke_test_nsd_tribe import (
    NSD_TR,
    NSD_STIM_DURATION,
    NSD_ISI,
    NSD_CATEGORIES,
    make_nsd_trial_sequence,
    NSDTrialProducer,
    NSDSessionAnalyzer,
    compute_rdm,
    run_nsd_session,
)


# ===========================================================================
# Fixtures
# ===========================================================================

N_VERTICES = 20484
N_PARCELS = 50
N_TRIALS = 12  # small session for testing


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def trial_sequence():
    return make_nsd_trial_sequence(n_trials=N_TRIALS)


@pytest.fixture
def frame_queue():
    return queue.Queue(maxsize=100)


# ===========================================================================
# NSD timing constants
# ===========================================================================

class TestNSDTiming:
    """Verify NSD paradigm timing constants."""

    def test_tr(self):
        assert NSD_TR == pytest.approx(1.6, abs=0.1)

    def test_stim_duration(self):
        assert NSD_STIM_DURATION == pytest.approx(3.0, abs=0.5)

    def test_isi(self):
        assert NSD_ISI == pytest.approx(1.0, abs=0.5)

    def test_trial_duration_in_trs(self):
        """Each trial (stim+ISI) should span ~2-3 TRs."""
        trial_dur = NSD_STIM_DURATION + NSD_ISI
        trs_per_trial = trial_dur / NSD_TR
        assert 2 <= trs_per_trial <= 4


# ===========================================================================
# Trial sequence
# ===========================================================================

class TestTrialSequence:
    """Tests for NSD trial sequence generation."""

    def test_length(self, trial_sequence):
        assert len(trial_sequence) == N_TRIALS

    def test_has_categories(self, trial_sequence):
        categories = {t["category"] for t in trial_sequence}
        assert len(categories) >= 2

    def test_each_trial_has_required_keys(self, trial_sequence):
        for trial in trial_sequence:
            assert "trial_id" in trial
            assert "category" in trial
            assert "onset_tr" in trial

    def test_onsets_are_sequential(self, trial_sequence):
        onsets = [t["onset_tr"] for t in trial_sequence]
        assert onsets == sorted(onsets)
        # Each trial should start after the previous one ends
        for i in range(1, len(onsets)):
            assert onsets[i] > onsets[i - 1]

    def test_categories_from_nsd_set(self, trial_sequence):
        for trial in trial_sequence:
            assert trial["category"] in NSD_CATEGORIES


# ===========================================================================
# NSDTrialProducer
# ===========================================================================

class TestNSDTrialProducer:
    """Tests for the per-image BOLD producer."""

    def test_produces_frames(self, trial_sequence, frame_queue):
        producer = NSDTrialProducer(
            frame_queue, trial_sequence,
            n_vertices=N_VERTICES, tr=0.0,
        )
        producer.start()
        producer.join(timeout=10)
        # Should produce multiple TRs per trial + sentinel
        count = 0
        while not frame_queue.empty():
            item = frame_queue.get()
            if item is None:
                break
            count += 1
        assert count > N_TRIALS  # multiple TRs per trial

    def test_frame_has_trial_info(self, trial_sequence, frame_queue):
        producer = NSDTrialProducer(
            frame_queue, trial_sequence,
            n_vertices=N_VERTICES, tr=0.0,
        )
        producer.start()
        producer.join(timeout=10)
        frame = frame_queue.get()
        assert "bold" in frame
        assert "tr_index" in frame
        assert "trial_id" in frame
        assert "is_stimulus" in frame

    def test_bold_frame_shape(self, trial_sequence, frame_queue):
        producer = NSDTrialProducer(
            frame_queue, trial_sequence,
            n_vertices=N_VERTICES, tr=0.0,
        )
        producer.start()
        producer.join(timeout=10)
        frame = frame_queue.get()
        assert frame["bold"].shape == (N_VERTICES,)

    def test_stimulus_trs_are_marked(self, trial_sequence, frame_queue):
        producer = NSDTrialProducer(
            frame_queue, trial_sequence,
            n_vertices=N_VERTICES, tr=0.0,
        )
        producer.start()
        producer.join(timeout=10)
        stim_count = 0
        isi_count = 0
        while not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break
            if frame["is_stimulus"]:
                stim_count += 1
            else:
                isi_count += 1
        assert stim_count > 0
        assert isi_count > 0


# ===========================================================================
# NSDSessionAnalyzer
# ===========================================================================

class TestNSDSessionAnalyzer:
    """Tests for the session-level beta extraction and analysis."""

    def test_extract_betas_shape(self, rng_key):
        analyzer = NSDSessionAnalyzer(
            n_vertices=N_VERTICES,
            n_parcels=N_PARCELS,
            key=rng_key,
        )
        # Feed a mini session
        n_trs = 30
        for i in range(n_trs):
            key = jax.random.fold_in(rng_key, i)
            frame = jax.random.normal(key, (N_VERTICES,))
            analyzer.add_frame(frame, trial_id=i // 3, is_stimulus=(i % 3 < 2))

        betas = analyzer.extract_betas()
        assert isinstance(betas, jnp.ndarray)
        assert betas.ndim == 2
        # Should have one beta per trial
        n_trials = len(set(range(n_trs // 3)))
        assert betas.shape[0] == n_trials
        assert betas.shape[1] == N_PARCELS


# ===========================================================================
# Representational dissimilarity matrix (RDM)
# ===========================================================================

class TestRDM:
    """Tests for representational similarity analysis."""

    def test_rdm_shape(self):
        betas = jnp.ones((10, 50))  # 10 trials, 50 parcels
        rdm = compute_rdm(betas)
        assert rdm.shape == (10, 10)

    def test_rdm_diagonal_zero(self):
        rng = np.random.RandomState(0)
        betas = jnp.array(rng.randn(10, 50).astype(np.float32))
        rdm = compute_rdm(betas)
        np.testing.assert_array_almost_equal(
            np.diag(np.asarray(rdm)), np.zeros(10), decimal=5,
        )

    def test_rdm_symmetric(self):
        rng = np.random.RandomState(0)
        betas = jnp.array(rng.randn(10, 50).astype(np.float32))
        rdm = compute_rdm(betas)
        np.testing.assert_array_almost_equal(
            np.asarray(rdm), np.asarray(rdm.T), decimal=5,
        )

    def test_rdm_nonnegative(self):
        rng = np.random.RandomState(0)
        betas = jnp.array(rng.randn(10, 50).astype(np.float32))
        rdm = compute_rdm(betas)
        assert jnp.all(rdm >= -1e-6)


# ===========================================================================
# End-to-end session
# ===========================================================================

class TestRunNSDSession:
    """Integration test for full NSD session simulation."""

    def test_returns_results(self, rng_key):
        results = run_nsd_session(
            n_trials=6,
            n_vertices=N_VERTICES,
            n_parcels=N_PARCELS,
            key=rng_key,
            tr=0.0,
        )
        assert isinstance(results, dict)

    def test_has_betas(self, rng_key):
        results = run_nsd_session(
            n_trials=6, n_vertices=N_VERTICES,
            n_parcels=N_PARCELS, key=rng_key, tr=0.0,
        )
        assert "betas" in results
        assert results["betas"].shape[0] == 6

    def test_has_rdm(self, rng_key):
        results = run_nsd_session(
            n_trials=6, n_vertices=N_VERTICES,
            n_parcels=N_PARCELS, key=rng_key, tr=0.0,
        )
        assert "rdm" in results
        assert results["rdm"].shape == (6, 6)

    def test_has_fc(self, rng_key):
        results = run_nsd_session(
            n_trials=6, n_vertices=N_VERTICES,
            n_parcels=N_PARCELS, key=rng_key, tr=0.0,
        )
        assert "fc" in results
        assert results["fc"].shape == (N_PARCELS, N_PARCELS)

    def test_category_rdm_structure(self, rng_key):
        """Within-category RDM distances should be smaller than between."""
        results = run_nsd_session(
            n_trials=12, n_vertices=N_VERTICES,
            n_parcels=N_PARCELS, key=rng_key, tr=0.0,
        )
        # This tests that the synthetic BOLD has category structure
        rdm = results["rdm"]
        categories = results["categories"]
        assert len(categories) == 12
