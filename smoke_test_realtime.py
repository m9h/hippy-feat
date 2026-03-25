import time
import jax
import jax.numpy as jnp
import numpy as np
from jaxoccoli import GeneralLinearModel, PermutationTest

def simulate_realtime_feed():
    print("Initializing Real-Time Smoke Test (MedARC-style simulation)...")
    
    # Simulation Parameters
    TR = 2.0  # Seconds (Time Repetition)
    n_voxels = 100000 
    n_timepoints_buffer = 100 # Window size for sliding GLM or accumulation
    n_new_volumes = 10 # Number of simulated TRs to process
    
    # 1. Setup Initial State
    # In a real-time setup, we often have a growing buffer or a sliding window.
    # For this test, let's assume we are re-running a GLM + Permutation on the current buffer
    # every time a new volume arrives. This is the "Incremental" or "Sliding Window" approach.
    
    print(f"TR: {TR}s | Voxels: {n_voxels} | Window: {n_timepoints_buffer}")
    
    # Pre-allocate buffer
    current_data = jnp.zeros((n_voxels, n_timepoints_buffer))
    design = jnp.ones((n_timepoints_buffer, 2)) # Dummy design
    contrast = jnp.array([1.0, 0.0])
    
    # JIT the core processing function to see if it's fast enough
    # The "Process" is: Fit GLM -> Run Permutation Test -> Get Threshold
    
    glm = GeneralLinearModel(design)
    pt = PermutationTest(glm, contrast, seed=42)
    
    @jax.jit
    def process_volume(data_buffer):
        # 1. Fit GLM
        betas, residuals = glm.fit(data_buffer)
        
        # 2. Run Permutation (Batched) - 1000 perms for speed check
        # We use a fixed key here for JIT simplicity in this smoke test
        # In reality, we'd pass a key.
        # Note: pt.run uses a loop, which might be slow to trace if unrolled.
        # But we jitted the internal batch runner. 
        # Let's call the internal batch runner directly for a "fast" check of e.g. 200 perms
        # to ensure we can do *some* stats in real-time.
        
        # Let's do 200 permutations per TR as a feasibility check
        key = jax.random.PRNGKey(0)
        max_t = pt._run_batch(key, data_buffer, 200) 
        
        return jnp.max(max_t)

    print("Warming up JAX (Tracing)...")
    dummy_data = jax.random.normal(jax.random.PRNGKey(0), (n_voxels, n_timepoints_buffer))
    _ = process_volume(dummy_data).block_until_ready()
    print("Warmup complete.")
    
    print("-" * 50)
    print("Starting Synthetic Scanner Feed...")
    print("-" * 50)
    
    latencies = []
    
    for i in range(n_new_volumes):
        # 1. "Wait" for scanner (Simulated)
        # In a real loop we would sleep, but here we just measure processing time.
        print(f"TR {i+1}: Volume Arrived...", end=" ")
        
        # 2. Update Buffer (Simulate incoming data)
        # Shift and append (or just overwrite for this test speed)
        # new_vol = jax.random.normal(...)
        # current_data = current_data.at[:, :-1].set(current_data[:, 1:])
        # current_data = current_data.at[:, -1].set(new_vol)
        
        # For pure compute benchmark, we just pass the buffer
        start_time = time.time()
        
        # PROCESS!
        result = process_volume(current_data)
        result.block_until_ready()
        
        end_time = time.time()
        latency = end_time - start_time
        latencies.append(latency)
        
        status = "OK" if latency < TR else "LATE"
        print(f"Processed in {latency:.4f}s. [{status}]")
        
        if latency > TR:
            print(f"CRITICAL WARNING: Processing took longer than TR ({TR}s)!")

    avg_latency = np.mean(latencies)
    print("-" * 50)
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"Max Latency: {np.max(latencies):.4f}s")
    
    if avg_latency < TR:
        print("SUCCESS: System is capable of Real-Time Processing.")
    else:
        print("FAILURE: System is too slow for Real-Time Processing.")

if __name__ == "__main__":
    simulate_realtime_feed()
