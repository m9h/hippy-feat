"""Simulated TRIBEv2 → jaxoccoli pipeline demo.

Demonstrates the full analysis path that TRIBEv2 predictions would take
through jaxoccoli, without requiring the actual model or GPU:

    Synthetic BOLD (fsaverage5)
      → Learnable cortical projection (vertex → parcel)
      → Functional connectivity matrix
      → Spectral embedding (diffusion maps)
      → Modularity analysis
      → Sliding-window dynamic FC

The synthetic BOLD mimics TRIBEv2's output: time-varying activation
on the fsaverage5 mesh with stimulus-driven spatial patterns that shift
from visual cortex → auditory cortex → language regions over time.

Usage:
    python smoke_test_tribe.py

Analogous to smoke_test_realtime.py but for the foundation-model path.
"""

import time
import numpy as np
import jax
import jax.numpy as jnp

from jaxoccoli.hf_encoder import make_cortical_projection
from jaxoccoli.covariance import corr
from jaxoccoli.graph import (
    diffusion_mapping,
    relaxed_modularity,
)
from jaxoccoli.connectivity import sliding_window_corr


# ---------------------------------------------------------------------------
# Cortical region layout (simplified fsaverage5 vertex ranges)
# ---------------------------------------------------------------------------

# Approximate vertex ranges for major cortical regions on fsaverage5.
# In a real application these come from an atlas (Schaefer, HCP-MMP, etc.).
# Here we partition the 20484 vertices into functional blocks for simulation.
CORTICAL_REGIONS = {
    "visual":   (0,     4096),    # V1, V2, V4, MT — posterior occipital
    "auditory": (4096,  6144),    # A1, belt, parabelt — superior temporal
    "language": (6144,  10240),   # STS, IFG, angular gyrus — left-lateralized
    "motor":    (10240, 14336),   # M1, premotor, SMA
    "dmn":      (14336, 20484),   # default mode — medial PFC, PCC, precuneus
}


# ---------------------------------------------------------------------------
# Stimulus schedule
# ---------------------------------------------------------------------------

def make_stimulus_schedule(n_timesteps: int = 60) -> np.ndarray:
    """Create a block-design stimulus schedule.

    Simulates a naturalistic stimulus (like TRIBEv2's Sintel demo):
    visual onset → audio/speech → language processing → rest, repeating.

    Parameters
    ----------
    n_timesteps : total duration in seconds (1 Hz, matching TRIBEv2 output)

    Returns
    -------
    (n_timesteps,) array of string labels from CORTICAL_REGIONS keys + "rest"
    """
    block_sequence = ["rest", "visual", "visual", "auditory", "language", "rest"]
    block_len = max(1, n_timesteps // len(block_sequence))

    schedule = []
    for label in block_sequence:
        schedule.extend([label] * block_len)

    # Pad or trim to exact length
    while len(schedule) < n_timesteps:
        schedule.append("rest")
    schedule = schedule[:n_timesteps]

    return np.array(schedule)


# ---------------------------------------------------------------------------
# Synthetic BOLD generator
# ---------------------------------------------------------------------------

def make_synthetic_bold(
    schedule: np.ndarray,
    n_vertices: int = 20484,
    key: jax.Array | None = None,
) -> jnp.ndarray:
    """Generate synthetic fsaverage5 BOLD mimicking TRIBEv2 output.

    Creates spatially structured, time-varying activation where each
    stimulus condition preferentially activates its cortical region.

    Parameters
    ----------
    schedule : (T,) string array of condition labels per timestep
    n_vertices : number of cortical vertices (20484 for fsaverage5)
    key : JAX PRNG key

    Returns
    -------
    (T, n_vertices) float32 array — simulated BOLD prediction
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    T = len(schedule)

    # Background noise (spatially correlated)
    key, subkey = jax.random.split(key)
    noise = jax.random.normal(subkey, (T, n_vertices)) * 0.1

    # Build activation signal per condition
    signal = jnp.zeros((T, n_vertices))

    for region_name, (v_start, v_end) in CORTICAL_REGIONS.items():
        # Find timesteps where this region is the active condition
        active_mask = np.array([1.0 if s == region_name else 0.0 for s in schedule])
        active_mask = jnp.array(active_mask, dtype=jnp.float32)

        # Temporal smoothing (simulate HRF-like delay)
        kernel = jnp.array([0.05, 0.1, 0.2, 0.3, 0.2, 0.1, 0.05])
        smoothed = jnp.convolve(active_mask, kernel, mode="same")

        # Spatial pattern: strong in target region, weak spillover elsewhere
        key, subkey = jax.random.split(key)
        spatial_weights = jax.random.normal(subkey, (n_vertices,)) * 0.02
        spatial_weights = spatial_weights.at[v_start:v_end].add(1.0)

        # Outer product: time × space
        signal = signal + smoothed[:, None] * spatial_weights[None, :]

    # Also add DMN anti-correlation during task (deactivation)
    task_mask = np.array([0.0 if s == "rest" else 1.0 for s in schedule])
    task_mask = jnp.array(task_mask, dtype=jnp.float32)
    dmn_start, dmn_end = CORTICAL_REGIONS["dmn"]
    signal = signal.at[:, dmn_start:dmn_end].add(-0.3 * task_mask[:, None])

    bold = signal + noise
    return bold.astype(jnp.float32)


# ---------------------------------------------------------------------------
# Analysis pipeline
# ---------------------------------------------------------------------------

def run_tribe_pipeline(
    bold: jnp.ndarray,
    n_parcels: int = 100,
    key: jax.Array | None = None,
    window_size: int = 20,
) -> dict:
    """Run the full jaxoccoli analysis pipeline on (simulated) TRIBEv2 output.

    Pipeline:
        1. Cortical projection (vertex → parcel)
        2. FC matrix (Pearson correlation)
        3. Spectral embedding (diffusion maps)
        4. Modularity (relaxed, differentiable)
        5. Sliding-window dynamic FC

    Parameters
    ----------
    bold : (T, n_vertices) BOLD data
    n_parcels : number of output parcels
    key : JAX PRNG key
    window_size : sliding window length for dynamic FC

    Returns
    -------
    Dict with: parcellated, fc, embedding, modularity, dynamic_fc, latencies
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    T, n_vertices = bold.shape
    latencies = {}

    # --- Step 1: Cortical projection ---
    t0 = time.time()
    proj_params, project_fn = make_cortical_projection(
        n_vertices=n_vertices, n_parcels=n_parcels, key=key,
        init="block",
    )
    parcellated = project_fn(proj_params, bold)
    parcellated.block_until_ready()
    latencies["projection_ms"] = (time.time() - t0) * 1000

    # --- Step 2: FC matrix ---
    t0 = time.time()
    fc = corr(parcellated.T)  # (n_parcels, n_parcels)
    fc.block_until_ready()
    latencies["fc_ms"] = (time.time() - t0) * 1000

    # --- Step 3: Spectral embedding ---
    t0 = time.time()
    # Convert FC to positive weights for the graph Laplacian
    W = jnp.clip(fc, 0, None)  # zero out negative correlations
    W = W - jnp.diag(jnp.diag(W))  # zero diagonal
    k = min(10, n_parcels - 1)
    eigvals, embedding = diffusion_mapping(W, k=k)
    embedding.block_until_ready()
    latencies["embedding_ms"] = (time.time() - t0) * 1000

    # --- Step 4: Modularity ---
    t0 = time.time()
    # Soft community assignment from the embedding (first 5 components → softmax)
    C_logits = embedding[:, :min(5, k)]
    C = jax.nn.softmax(C_logits, axis=-1)
    Q = relaxed_modularity(W, C)
    Q.block_until_ready()
    latencies["modularity_ms"] = (time.time() - t0) * 1000

    # --- Step 5: Sliding-window dynamic FC ---
    t0 = time.time()
    dynamic_fc = sliding_window_corr(parcellated.T, window_size=window_size)
    dynamic_fc.block_until_ready()
    latencies["dynamic_fc_ms"] = (time.time() - t0) * 1000

    return {
        "parcellated": parcellated,
        "fc": fc,
        "embedding": embedding,
        "modularity": Q,
        "dynamic_fc": dynamic_fc,
        "latencies": latencies,
    }


# ---------------------------------------------------------------------------
# Main: smoke test with printout
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("TRIBEv2 → jaxoccoli Pipeline Smoke Test")
    print("(Simulated — no model download required)")
    print("=" * 60)

    N_VERTICES = 20484
    N_PARCELS = 100
    N_TIMESTEPS = 60
    WINDOW = 20

    key = jax.random.PRNGKey(42)

    # --- Generate synthetic data ---
    print(f"\n[1/6] Generating stimulus schedule ({N_TIMESTEPS}s block design)...")
    schedule = make_stimulus_schedule(N_TIMESTEPS)
    conditions = {c: int(np.sum(schedule == c)) for c in np.unique(schedule)}
    print(f"       Conditions: {conditions}")

    print(f"[2/6] Generating synthetic fsaverage5 BOLD ({N_TIMESTEPS} × {N_VERTICES})...")
    t0 = time.time()
    bold = make_synthetic_bold(schedule, n_vertices=N_VERTICES, key=key)
    bold.block_until_ready()
    gen_time = (time.time() - t0) * 1000
    print(f"       Generated in {gen_time:.0f}ms")
    print(f"       Range: [{float(bold.min()):.3f}, {float(bold.max()):.3f}]")

    # --- JIT warmup ---
    print(f"\n[3/6] JIT warmup (tracing pipeline with {N_PARCELS} parcels)...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    _ = run_tribe_pipeline(bold[:5], n_parcels=N_PARCELS, key=subkey, window_size=3)
    warmup_time = (time.time() - t0) * 1000
    print(f"       Warmup complete in {warmup_time:.0f}ms")

    # --- Full pipeline ---
    print(f"\n[4/6] Running full pipeline ({N_TIMESTEPS}s BOLD → {N_PARCELS} parcels)...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    results = run_tribe_pipeline(
        bold, n_parcels=N_PARCELS, key=subkey, window_size=WINDOW,
    )
    total_time = (time.time() - t0) * 1000
    print(f"       Total pipeline: {total_time:.0f}ms")

    # --- Latency breakdown ---
    print(f"\n[5/6] Latency breakdown:")
    lat = results["latencies"]
    for stage, ms in lat.items():
        print(f"       {stage:20s} {ms:8.1f} ms")

    # --- Results summary ---
    print(f"\n[6/6] Results:")
    fc = results["fc"]
    emb = results["embedding"]
    Q = results["modularity"]
    dfc = results["dynamic_fc"]

    print(f"       Parcellated:   {results['parcellated'].shape}")
    print(f"       FC matrix:     {fc.shape}  (range: [{float(fc.min()):.3f}, {float(fc.max()):.3f}])")
    print(f"       Embedding:     {emb.shape}")
    print(f"       Modularity Q:  {float(Q):.4f}")
    print(f"       Dynamic FC:    {dfc.shape}  ({dfc.shape[0]} windows × {WINDOW}s)")

    # FC matrix statistics
    mask = 1.0 - jnp.eye(N_PARCELS)
    off_diag = fc * mask
    print(f"       Mean off-diag FC: {float(jnp.sum(off_diag) / jnp.sum(mask)):.4f}")
    print(f"       FC std:           {float(jnp.std(off_diag[mask > 0])):.4f}")

    # Dynamic FC temporal variability
    dfc_temporal_std = jnp.mean(jnp.std(dfc, axis=0))
    print(f"       dFC temporal std: {float(dfc_temporal_std):.4f}")

    print("\n" + "=" * 60)
    print("Pipeline complete.")
    print(f"Total wall time: {total_time:.0f}ms for {N_TIMESTEPS}s of simulated BOLD")
    print(f"Throughput: {N_TIMESTEPS / (total_time / 1000):.0f} simulated TRs/second")
    print("=" * 60)

    print("\nWith TRIBEv2 installed, replace make_synthetic_bold with:")
    print("  params, encode = make_hf_encoder('facebook/tribev2', key=key)")
    print("  bold = encode(params, {'video_path': 'stimulus.mp4'})")
    print("  results = run_tribe_pipeline(bold, n_parcels=100, key=key)")


if __name__ == "__main__":
    main()
