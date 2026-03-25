import time
import os
import shutil
import tempfile
import threading
import glob
import jax
import jax.numpy as jnp
import numpy as np
import nibabel as nib
from functools import partial
from jaxoccoli import GeneralLinearModel, PermutationTest

# Parameters
TR = 2.0
N_VOXELS = 100000
N_VOLUMES = 10
WINDOW_SIZE = 50

class ScannerSimulator(threading.Thread):
    def __init__(self, out_dir, tr=2.0, n_volumes=10):
        super().__init__()
        self.out_dir = out_dir
        self.tr = tr
        self.n_volumes = n_volumes
        self._stop_event = threading.Event()

    def run(self):
        print(f"[Scanner] Started. TR={self.tr}s")
        for i in range(self.n_volumes):
            if self._stop_event.is_set(): break
            
            start_acq = time.time()
            ts_str = f"{start_acq:.6f}"
            
            # Simulate Acquisition (compute noise or patterns)
            # We generate a numpy array and save via nibabel to mimic file I/O overhead
            vol_data = np.random.randn(100, 100, 10) # 100k voxels
            img = nib.Nifti1Image(vol_data, np.eye(4))
            
            filename = os.path.join(self.out_dir, f"vol_{i:03d}_{ts_str}.nii")
            
            # Simulate "Write time" slightly?
            nib.save(img, filename)
            
            print(f"[Scanner] Acquired Volume {i+1}/{self.n_volumes}")
            
            # Wait for next TR
            elapsed = time.time() - start_acq
            sleep_time = max(0, self.tr - elapsed)
            time.sleep(sleep_time)
            
        print("[Scanner] Finished protocol.")

    def stop(self):
        self._stop_event.set()

from jaxoccoli.motion import RigidBodyRegistration

class RealTimeAnalyzer:
    def __init__(self, watch_dir):
        self.watch_dir = watch_dir
        self.processed_files = set()
        
        # JAX Init
        print("[Analyzer] Initializing JAXoccoli...")
        self.design = jnp.ones((WINDOW_SIZE, 2))
        self.glm = GeneralLinearModel(self.design)
        self.contrast = jnp.array([1.0, 0.0])
        self.pt = PermutationTest(self.glm, self.contrast, seed=42)
        
        # Motion Correction Init
        # We need a template. For the first volume, we'll set it as template.
        self.registrator = None
        self.mc_params = jnp.zeros(6) # Keep track of estimation
        
        # Warmup
        print("[Analyzer] Warming up JIT...")
        dummy_data = jnp.zeros((N_VOXELS, WINDOW_SIZE))
        # self.process_volume calls... we need a new process method that includes motion
        
        # Warm up motion correction
        dummy_vol = jnp.zeros((46, 55, 46)) # Approximate MNI/Sample 100k voxels unflattened?
        # Actually our scanner gives flat data in this smoke test.
        # To test Motion Correction properly, we need 3D data.
        # The ScannerSimulator generates (100, 100, 10).
        self.vol_shape = (100, 100, 10)
        
        print("[Analyzer] Ready.")

    @partial(jax.jit, static_argnums=(0,))
    def register_and_process(self, new_vol_3d, current_buffer, template, init_params):
        # 1. Motion Correction
        # We create a temporary registrator here or pass it?
        # Ideally the class holds state, but for JIT we want functional.
        # Let's use the module class just as a container for JIT functions for now, 
        # or instantiate it inside if it's lightweight.
        # Actually `jaxoccoli/motion.py` uses a class but methods are partial JIT.
        # Let's assume we use the instance `self.registrator`.
        
        # NOTE: self.registrator.register_volume is JIT-ed.
        best_params, reg_vol = self.registrator.register_volume(new_vol_3d, init_params)
        
        # Flatten for GLM
        data_flat = reg_vol.flatten()
        
        # 2. Update Buffer
        # current_buffer: (Vocels, Time)
        updated_buffer = current_buffer.at[:, :-1].set(current_buffer[:, 1:])
        updated_buffer = updated_buffer.at[:, -1].set(data_flat)
        
        # 3. Fit GLM
        betas, residuals = self.glm.fit(updated_buffer)
        
        # 4. Permutation
        key = jax.random.PRNGKey(0)
        max_t = self.pt._run_batch(key, updated_buffer, 200)
        
        return best_params, jnp.max(max_t), updated_buffer

    def run_loop(self, timeout=30):
        print(f"[Analyzer] Watching {self.watch_dir}")
        start_loop = time.time()
        latencies = []
        
        # Sliding buffer (initialized with zeros)
        current_buffer = jnp.zeros((N_VOXELS, WINDOW_SIZE))
        
        # Template and Registrator
        self.template = None
        
        while True:
            # Check timeout
            if time.time() - start_loop > timeout:
                print("[Analyzer] Timeout reached.")
                break
                
            # Poll for files
            files = sorted(glob.glob(os.path.join(self.watch_dir, "*.nii")))
            new_files = [f for f in files if f not in self.processed_files]
            
            if not new_files:
                time.sleep(0.05) # FAST polling
                continue
            
            for f in new_files:
                # Parse timestamp
                try:
                    ts_str = f.split('_')[-1].replace('.nii', '')
                    acq_time = float(ts_str)
                except ValueError:
                    acq_time = time.time()
                
                # Load (Keep 3D structure for MC)
                t0_load = time.time()
                try:
                    img = nib.load(f)
                    vol_3d = img.get_fdata() # (100, 100, 10)
                except Exception as e:
                    print(f"[Analyzer] Error reading {f}: {e}")
                    continue
                    
                # Setup Template / Registrator if first volume
                vol_jax = jnp.array(vol_3d)
                
                if self.template is None:
                    print(f"[Analyzer] Setting template from {f}")
                    self.template = vol_jax
                    self.registrator = RigidBodyRegistration(self.template, self.vol_shape, step_size=0.01, n_iter=20)
                    # Warmup first run
                    print("[Analyzer] Compiling Motion Correction...")
                    # Unpack and block on one element
                    warmup_params, _, _ = self.register_and_process(vol_jax, current_buffer, self.template, self.mc_params)
                    warmup_params.block_until_ready()
                    print("[Analyzer] Compiled.")
                
                t0_process = time.time()
                
                # Run Pipeline (MC -> Flatten -> GLM -> Perm)
                # We pass current mc_params as initialization for this volume (assuming small motion)
                best_params, max_t, current_buffer = self.register_and_process(
                    vol_jax, current_buffer, self.template, self.mc_params
                )
                
                # Block
                best_params.block_until_ready()
                max_t.block_until_ready()
                
                t1_process = time.time()
                
                # Update MC params for next frame (continuity)
                self.mc_params = best_params
                
                analysis_time = t1_process - t0_process
                total_latency = t1_process - acq_time
                
                print(f"[Analyzer] Processed {os.path.basename(f)} | Compute: {analysis_time:.4f}s | Lag: {total_latency:.4f}s | Motion: {best_params[:3]}")
                latencies.append(total_latency)
                
                self.processed_files.add(f)
                
            if len(self.processed_files) >= N_VOLUMES:
                print("[Analyzer] All volumes processed.")
                return latencies

from functools import partial

def main():
    # Temp dir for handover
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Temporary RT-Cloud Directory: {tmpdir}")
        
        # Start Scanner Thread
        scanner = ScannerSimulator(tmpdir, tr=TR, n_volumes=N_VOLUMES)
        scanner.start()
        
        # Start Analyzer (Main Thread)
        analyzer = RealTimeAnalyzer(tmpdir)
        latencies = analyzer.run_loop(timeout=TR * N_VOLUMES + 5.0)
        
        scanner.join()
        
        if latencies:
            print("-" * 50)
            print(f"Mean Total Lag: {np.mean(latencies):.4f}s")
            print(f"Max Total Lag: {np.max(latencies):.4f}s")
            if np.mean(latencies) < TR:
                print("SUCCESS: RT-Cloud Simulation Passed.")
            else:
                print("FAILURE: System too slow.")
        else:
            print("FAILURE: No files processed.")

if __name__ == "__main__":
    main()
