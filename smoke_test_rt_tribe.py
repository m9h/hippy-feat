"""RT-cloud streaming simulation: TRIBEv2 producer → jaxoccoli consumer.

Producer-consumer architecture where a TribeProducer thread generates
simulated BOLD frames at 1 Hz (matching TRIBEv2's temporal resolution),
and an RTConnectivityAnalyzer processes each frame in real-time:

    TribeProducer (1 frame/s)
      → queue →
    RTConnectivityAnalyzer:
      → parcellate (vertex → parcel)
      → shift sliding window buffer
      → update FC matrix
      → compute modularity
      → report latency

This demonstrates the full rt-cloud pipeline with zero scanner hardware.
With TRIBEv2 installed, the producer would call model.predict() per frame
instead of generating synthetic data.

Usage:
    python smoke_test_rt_tribe.py

Follows the same pattern as smoke_test_rt_cloud.py but for the
foundation-model → connectivity analysis path.
"""

import time
import queue
import threading
import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from jaxoccoli.hf_encoder import CorticalProjectionParams
from jaxoccoli.connectivity import SlidingWindowConnectivity
from jaxoccoli.graph import relaxed_modularity


# ---------------------------------------------------------------------------
# Cortical region layout (same as smoke_test_tribe.py)
# ---------------------------------------------------------------------------

CORTICAL_REGIONS = {
    "visual":   (0,     4096),
    "auditory": (4096,  6144),
    "language": (6144,  10240),
    "motor":    (10240, 14336),
    "dmn":      (14336, 20484),
}

_BLOCK_SCHEDULE = ["rest", "visual", "visual", "auditory", "language", "rest"]


# ---------------------------------------------------------------------------
# Helper: block-diagonal projection params
# ---------------------------------------------------------------------------

def make_block_projection_params(
    n_vertices: int,
    n_parcels: int,
    key: jax.Array,
) -> CorticalProjectionParams:
    """Create block-diagonal projection params (same as init='block')."""
    logits = jax.random.normal(key, (n_vertices, n_parcels)) * 0.01
    block_size = n_vertices // n_parcels
    for p in range(n_parcels):
        v_start = p * block_size
        v_end = min((p + 1) * block_size, n_vertices)
        logits = logits.at[v_start:v_end, p].add(5.0)
    return CorticalProjectionParams(logits=logits)


# ---------------------------------------------------------------------------
# TribeProducer
# ---------------------------------------------------------------------------

class TribeProducer(threading.Thread):
    """Simulated TRIBEv2 producer thread.

    Generates one BOLD frame per TR and pushes to a queue.
    Each frame is a dict: {bold, timestamp, tr_index, condition}.
    Sends None sentinel when finished.

    Parameters
    ----------
    frame_queue : queue.Queue to push frames into
    n_vertices : fsaverage5 vertex count
    n_trs : total number of frames to produce
    tr : repetition time in seconds (0 = no sleep, for testing)
    key : JAX PRNG key
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        n_vertices: int = 20484,
        n_trs: int = 60,
        tr: float = 1.0,
        key: jax.Array | None = None,
    ):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.n_vertices = n_vertices
        self.n_trs = n_trs
        self.tr = tr
        self.key = key if key is not None else jax.random.PRNGKey(0)

    def run(self):
        schedule_len = len(_BLOCK_SCHEDULE)
        block_len = max(1, self.n_trs // schedule_len)

        # Pre-build the schedule
        schedule = []
        for label in _BLOCK_SCHEDULE:
            schedule.extend([label] * block_len)
        while len(schedule) < self.n_trs:
            schedule.append("rest")
        schedule = schedule[:self.n_trs]

        # Pre-generate spatial patterns per region
        key = self.key
        region_patterns = {}
        for name, (v_start, v_end) in CORTICAL_REGIONS.items():
            key, subkey = jax.random.split(key)
            pattern = jnp.zeros(self.n_vertices) * 0.02
            pattern = pattern.at[v_start:v_end].set(1.0)
            # Add small noise to the pattern
            pattern = pattern + jax.random.normal(subkey, (self.n_vertices,)) * 0.02
            region_patterns[name] = pattern

        for i in range(self.n_trs):
            acq_start = time.time()
            condition = schedule[i]

            key, subkey = jax.random.split(key)
            noise = jax.random.normal(subkey, (self.n_vertices,)) * 0.1

            # Build signal: active region + DMN deactivation during task
            if condition in region_patterns:
                signal = region_patterns[condition] * 0.8 + noise
            else:
                signal = noise

            # DMN deactivation during non-rest
            if condition != "rest":
                dmn_start, dmn_end = CORTICAL_REGIONS["dmn"]
                signal = signal.at[dmn_start:dmn_end].add(-0.3)

            frame = {
                "bold": np.asarray(signal, dtype=np.float32),
                "timestamp": acq_start,
                "tr_index": i,
                "condition": condition,
            }
            self.frame_queue.put(frame)

            # Pace to TR
            if self.tr > 0:
                elapsed = time.time() - acq_start
                sleep_time = max(0, self.tr - elapsed)
                time.sleep(sleep_time)

        # Sentinel
        self.frame_queue.put(None)


# ---------------------------------------------------------------------------
# RTConnectivityAnalyzer
# ---------------------------------------------------------------------------

class RTConnectivityAnalyzer:
    """Real-time connectivity analyzer (consumer).

    Processes one BOLD frame per call:
        1. Parcellate (vertex → parcel via softmax projection)
        2. Shift sliding window buffer
        3. Compute FC matrix
        4. Compute modularity from FC

    Parameters
    ----------
    n_vertices : cortical vertex count
    n_parcels : number of output parcels
    window_size : sliding window length
    key : JAX PRNG key for projection init
    """

    def __init__(
        self,
        n_vertices: int = 20484,
        n_parcels: int = 100,
        window_size: int = 20,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        self.n_vertices = n_vertices
        self.n_parcels = n_parcels
        self.window_size = window_size
        self.frames_processed = 0

        # Projection weights (block-diagonal init)
        self.proj_params = make_block_projection_params(n_vertices, n_parcels, key)
        self._proj_weights = jax.nn.softmax(self.proj_params.logits, axis=0)

        # Sliding window buffer: (n_parcels, window_size)
        self.buffer = jnp.zeros((n_parcels, window_size))

        # Streaming FC tracker
        self.swc = SlidingWindowConnectivity(n_parcels, window_size)

        # History
        self.fc_history = []

    @partial(jax.jit, static_argnums=(0,))
    def _parcellate(self, bold_frame: jnp.ndarray) -> jnp.ndarray:
        """Project a single vertex-wise frame to parcels."""
        return bold_frame @ self._proj_weights  # (n_parcels,)

    @partial(jax.jit, static_argnums=(0,))
    def _update_and_analyze(
        self,
        buffer: jnp.ndarray,
        new_parcel_data: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Shift buffer, compute FC and modularity in one JIT call."""
        # Shift buffer and compute FC
        updated_buffer, fc = self.swc.update_and_compute(buffer, new_parcel_data)

        # Modularity from positive FC
        W = jnp.clip(fc, 0, None)
        W = W - jnp.diag(jnp.diag(W))

        # Soft community assignment from eigenvectors of FC
        # Use top-k eigenvectors of the positive FC as soft labels
        eigvals, eigvecs = jnp.linalg.eigh(W)
        # Take last 5 (largest eigenvalue) eigenvectors
        k = min(5, self.n_parcels)
        C = jax.nn.softmax(eigvecs[:, -k:], axis=-1)
        Q = relaxed_modularity(W, C)

        return updated_buffer, fc, Q

    def process_frame(self, bold_frame) -> dict:
        """Process a single BOLD frame (vertex-wise).

        Parameters
        ----------
        bold_frame : (n_vertices,) single TR of BOLD data

        Returns
        -------
        Dict with: parcellated, fc, modularity
        """
        bold_frame = jnp.asarray(bold_frame)

        # Step 1: parcellate
        parcellated = self._parcellate(bold_frame)

        # Step 2: update buffer and compute FC + modularity
        self.buffer, fc, Q = self._update_and_analyze(self.buffer, parcellated)

        self.frames_processed += 1
        self.fc_history.append(fc)

        return {
            "parcellated": parcellated,
            "fc": fc,
            "modularity": Q,
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("RT-Cloud Streaming: TRIBEv2 → jaxoccoli")
    print("(Simulated — no model download required)")
    print("=" * 60)

    N_VERTICES = 20484
    N_PARCELS = 100
    WINDOW = 20
    N_TRS = 30
    TR = 1.0

    key = jax.random.PRNGKey(42)

    # --- Setup ---
    frame_queue = queue.Queue(maxsize=10)
    producer = TribeProducer(
        frame_queue,
        n_vertices=N_VERTICES,
        n_trs=N_TRS,
        tr=TR,
        key=key,
    )
    key, subkey = jax.random.split(key)
    analyzer = RTConnectivityAnalyzer(
        n_vertices=N_VERTICES,
        n_parcels=N_PARCELS,
        window_size=WINDOW,
        key=subkey,
    )

    # --- JIT warmup ---
    print("\n[Analyzer] Warming up JIT...")
    warmup_frame = jax.random.normal(key, (N_VERTICES,))
    t0 = time.time()
    _ = analyzer.process_frame(warmup_frame)
    warmup_ms = (time.time() - t0) * 1000
    print(f"[Analyzer] Warmup: {warmup_ms:.0f}ms")

    # --- Start streaming ---
    print(f"\n[System] Starting RT stream: {N_TRS} TRs at TR={TR}s")
    print(f"[System] {N_VERTICES} vertices → {N_PARCELS} parcels, window={WINDOW}")
    print("-" * 60)

    producer.start()
    latencies = []
    conditions = []

    while True:
        try:
            frame = frame_queue.get(timeout=TR * 2)
        except queue.Empty:
            print("[Analyzer] Queue timeout — producer may have stalled.")
            break

        if frame is None:
            break

        t0 = time.time()
        result = analyzer.process_frame(frame["bold"])
        # Force sync for accurate timing
        result["fc"].block_until_ready()
        elapsed = time.time() - t0

        total_lag = time.time() - frame["timestamp"]
        latencies.append(elapsed)
        conditions.append(frame["condition"])

        Q = float(result["modularity"])
        fc_std = float(jnp.std(result["fc"]))

        status = "OK" if elapsed < TR else "LATE"
        print(
            f"  TR {frame['tr_index']+1:3d}/{N_TRS} "
            f"| {frame['condition']:8s} "
            f"| compute: {elapsed*1000:6.1f}ms "
            f"| lag: {total_lag*1000:6.0f}ms "
            f"| Q={Q:+.4f} "
            f"| FC_std={fc_std:.4f} "
            f"| [{status}]"
        )

    producer.join(timeout=5)

    # --- Summary ---
    print("-" * 60)
    if latencies:
        avg = np.mean(latencies) * 1000
        mx = np.max(latencies) * 1000
        print(f"\nLatency:  avg={avg:.1f}ms  max={mx:.1f}ms  budget={TR*1000:.0f}ms")

        if avg < TR * 1000:
            print(f"SUCCESS: Avg latency ({avg:.0f}ms) < TR ({TR*1000:.0f}ms)")
        else:
            print(f"WARNING: Avg latency ({avg:.0f}ms) >= TR ({TR*1000:.0f}ms)")

        # FC evolution summary
        fc_stds = [float(jnp.std(fc)) for fc in analyzer.fc_history]
        print(f"\nFC evolution over {len(fc_stds)} TRs:")
        print(f"  FC std range: [{min(fc_stds):.4f}, {max(fc_stds):.4f}]")

        # Modularity by condition
        from collections import defaultdict
        q_by_cond = defaultdict(list)
        for i, fc in enumerate(analyzer.fc_history):
            if i < len(conditions):
                W = jnp.clip(fc, 0, None) - jnp.diag(jnp.diag(jnp.clip(fc, 0, None)))
                eigvals, eigvecs = jnp.linalg.eigh(W)
                C = jax.nn.softmax(eigvecs[:, -5:], axis=-1)
                q_by_cond[conditions[i]].append(float(relaxed_modularity(W, C)))
        print("\nModularity by condition:")
        for cond in sorted(q_by_cond.keys()):
            vals = q_by_cond[cond]
            print(f"  {cond:8s}: Q={np.mean(vals):+.4f} (n={len(vals)})")
    else:
        print("FAILURE: No frames processed.")

    print("\n" + "=" * 60)
    print("With TRIBEv2, replace TribeProducer with:")
    print("  model = TribeModel.from_pretrained('facebook/tribev2')")
    print("  events = model.get_events_dataframe(video_path='stimulus.mp4')")
    print("  preds, _ = model.predict(events=events)")
    print("  # Then push preds[t, :] per TR into the queue")
    print("=" * 60)


if __name__ == "__main__":
    main()
