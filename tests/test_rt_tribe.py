"""
Tests for smoke_test_rt_tribe — RT-cloud TRIBEv2 streaming simulation.

Covers:
  - TribeProducer: generates BOLD frames with stimulus-driven dynamics
  - RTConnectivityAnalyzer: per-TR parcellation, sliding FC, modularity
  - Streaming state: buffer shift, FC history accumulation
  - Latency measurement per TR
  - End-to-end producer-consumer integration
"""

import pytest
import numpy as np
import jax
import jax.numpy as jnp
import queue
import threading

from smoke_test_rt_tribe import (
    TribeProducer,
    RTConnectivityAnalyzer,
    make_block_projection_params,
    CORTICAL_REGIONS,
)


# ===========================================================================
# Fixtures
# ===========================================================================

N_VERTICES = 20484
N_PARCELS = 50
WINDOW = 15
N_TRS = 10


@pytest.fixture
def rng_key():
    return jax.random.PRNGKey(42)


@pytest.fixture
def frame_queue():
    return queue.Queue(maxsize=20)


@pytest.fixture
def producer(frame_queue):
    return TribeProducer(
        frame_queue,
        n_vertices=N_VERTICES,
        n_trs=N_TRS,
        tr=0.0,  # no sleep in tests
    )


@pytest.fixture
def analyzer(rng_key):
    return RTConnectivityAnalyzer(
        n_vertices=N_VERTICES,
        n_parcels=N_PARCELS,
        window_size=WINDOW,
        key=rng_key,
    )


# ===========================================================================
# TribeProducer
# ===========================================================================

class TestTribeProducer:
    """Tests for the BOLD frame producer thread."""

    def test_produces_n_frames(self, producer, frame_queue):
        producer.start()
        producer.join(timeout=5)
        # N_TRS data frames + 1 None sentinel
        assert frame_queue.qsize() == N_TRS + 1

    def test_frame_shape(self, producer, frame_queue):
        producer.start()
        producer.join(timeout=5)
        frame = frame_queue.get()
        assert frame["bold"].shape == (N_VERTICES,)

    def test_frame_has_timestamp(self, producer, frame_queue):
        producer.start()
        producer.join(timeout=5)
        frame = frame_queue.get()
        assert "timestamp" in frame
        assert isinstance(frame["timestamp"], float)

    def test_frame_has_tr_index(self, producer, frame_queue):
        producer.start()
        producer.join(timeout=5)
        frame = frame_queue.get()
        assert "tr_index" in frame
        assert frame["tr_index"] == 0

    def test_frame_has_condition(self, producer, frame_queue):
        producer.start()
        producer.join(timeout=5)
        frame = frame_queue.get()
        assert "condition" in frame

    def test_frames_have_sequential_indices(self, producer, frame_queue):
        producer.start()
        producer.join(timeout=5)
        indices = []
        while not frame_queue.empty():
            item = frame_queue.get()
            if item is None:
                break
            indices.append(item["tr_index"])
        assert indices == list(range(N_TRS))

    def test_sentinel_signals_end(self, producer, frame_queue):
        """Producer should put a None sentinel after all frames."""
        producer.start()
        producer.join(timeout=5)
        items = []
        while not frame_queue.empty():
            items.append(frame_queue.get())
        assert items[-1] is None


# ===========================================================================
# RTConnectivityAnalyzer
# ===========================================================================

class TestRTConnectivityAnalyzer:
    """Tests for the real-time connectivity consumer."""

    def test_process_frame_returns_dict(self, analyzer, rng_key):
        frame = jax.random.normal(rng_key, (N_VERTICES,))
        result = analyzer.process_frame(frame)
        assert isinstance(result, dict)

    def test_result_has_fc(self, analyzer, rng_key):
        frame = jax.random.normal(rng_key, (N_VERTICES,))
        result = analyzer.process_frame(frame)
        assert "fc" in result
        assert result["fc"].shape == (N_PARCELS, N_PARCELS)

    def test_result_has_parcellated(self, analyzer, rng_key):
        frame = jax.random.normal(rng_key, (N_VERTICES,))
        result = analyzer.process_frame(frame)
        assert "parcellated" in result
        assert result["parcellated"].shape == (N_PARCELS,)

    def test_result_has_modularity(self, analyzer, rng_key):
        frame = jax.random.normal(rng_key, (N_VERTICES,))
        result = analyzer.process_frame(frame)
        assert "modularity" in result
        assert result["modularity"].shape == ()

    def test_buffer_fills_up(self, analyzer, rng_key):
        """Processing multiple frames should fill the sliding window."""
        for i in range(WINDOW):
            key = jax.random.fold_in(rng_key, i)
            frame = jax.random.normal(key, (N_VERTICES,))
            analyzer.process_frame(frame)
        assert analyzer.frames_processed == WINDOW

    def test_fc_improves_with_more_data(self, analyzer, rng_key):
        """FC should become non-trivial once buffer is full."""
        for i in range(WINDOW + 5):
            key = jax.random.fold_in(rng_key, i)
            frame = jax.random.normal(key, (N_VERTICES,))
            result = analyzer.process_frame(frame)
        fc = result["fc"]
        # Should have non-zero off-diagonal variance
        mask = 1.0 - jnp.eye(N_PARCELS)
        off_diag_std = jnp.std(fc * mask)
        assert off_diag_std > 0.01

    def test_fc_history_accumulates(self, analyzer, rng_key):
        for i in range(5):
            key = jax.random.fold_in(rng_key, i)
            frame = jax.random.normal(key, (N_VERTICES,))
            analyzer.process_frame(frame)
        assert len(analyzer.fc_history) == 5


# ===========================================================================
# Block projection params
# ===========================================================================

class TestBlockProjectionParams:

    def test_output_shape(self, rng_key):
        params = make_block_projection_params(N_VERTICES, N_PARCELS, rng_key)
        assert params.logits.shape == (N_VERTICES, N_PARCELS)


# ===========================================================================
# Integration: producer → consumer
# ===========================================================================

class TestEndToEnd:
    """Producer-consumer integration test."""

    def test_streaming_pipeline(self, rng_key):
        """Full streaming run: producer pushes frames, analyzer consumes."""
        q = queue.Queue(maxsize=20)
        n_trs = 8

        producer = TribeProducer(q, n_vertices=N_VERTICES, n_trs=n_trs, tr=0.0)
        analyzer = RTConnectivityAnalyzer(
            n_vertices=N_VERTICES, n_parcels=N_PARCELS,
            window_size=5, key=rng_key,
        )

        producer.start()
        results = []
        while True:
            frame = q.get(timeout=5)
            if frame is None:
                break
            result = analyzer.process_frame(frame["bold"])
            results.append(result)

        producer.join(timeout=5)
        assert len(results) == n_trs
        assert all("fc" in r for r in results)
        assert all("modularity" in r for r in results)

    def test_latency_under_tr(self, rng_key):
        """Per-TR processing should be fast (well under 1s on CPU)."""
        import time

        analyzer = RTConnectivityAnalyzer(
            n_vertices=N_VERTICES, n_parcels=N_PARCELS,
            window_size=5, key=rng_key,
        )
        # Warmup
        warmup_frame = jax.random.normal(rng_key, (N_VERTICES,))
        analyzer.process_frame(warmup_frame)

        # Timed run
        key = jax.random.fold_in(rng_key, 999)
        frame = jax.random.normal(key, (N_VERTICES,))
        t0 = time.time()
        result = analyzer.process_frame(frame)
        # Force computation
        result["fc"].block_until_ready()
        elapsed = time.time() - t0
        assert elapsed < 2.0, f"Per-TR latency {elapsed:.3f}s too slow"
