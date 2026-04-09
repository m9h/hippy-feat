"""Simulated NSD picture-watching session with TRIBEv2 → jaxoccoli.

Simulates the Natural Scenes Dataset paradigm:
  - COCO images presented for 3s with 1s ISI, TR=1.6s
  - Each image belongs to a visual category (face, place, object, animal, food)
  - TRIBEv2 (simulated) predicts vertex-wise BOLD per image
  - jaxoccoli extracts per-trial betas via GLM, computes beta-series FC
    and representational dissimilarity matrices (RDMs)

The synthetic BOLD embeds category structure: images from the same category
produce similar activation patterns, enabling RSA to recover the category
structure from the predicted betas — just as it would with real NSD data.

With TRIBEv2 installed, each COCO image would be presented as a 3s static
video frame to model.predict(), and the real predictions would replace the
synthetic BOLD.

Usage:
    python smoke_test_nsd_tribe.py
"""

import time
import queue
import threading
import numpy as np

import jax
import jax.numpy as jnp
from functools import partial

from jaxoccoli.hf_encoder import make_cortical_projection
from jaxoccoli.covariance import corr


# ---------------------------------------------------------------------------
# NSD paradigm constants
# ---------------------------------------------------------------------------

NSD_TR = 1.6            # seconds (7T NSD acquisition)
NSD_STIM_DURATION = 3.0  # seconds per image
NSD_ISI = 1.0            # inter-stimulus interval
NSD_TRIAL_DURATION = NSD_STIM_DURATION + NSD_ISI  # 4.0s per trial
NSD_TRS_PER_TRIAL = round(NSD_TRIAL_DURATION / NSD_TR)  # ~2-3 TRs

# Simplified COCO supercategories for simulation
NSD_CATEGORIES = ["face", "place", "object", "animal", "food"]

# Cortical regions preferentially activated by each category
_CATEGORY_REGIONS = {
    "face":   (0,     2048),   # fusiform face area
    "place":  (2048,  4096),   # parahippocampal place area
    "object": (4096,  8192),   # lateral occipital complex
    "animal": (8192,  12288),  # animate-selective regions
    "food":   (12288, 16384),  # ventral temporal
}


# ---------------------------------------------------------------------------
# Trial sequence
# ---------------------------------------------------------------------------

def make_nsd_trial_sequence(
    n_trials: int = 50,
    seed: int = 0,
) -> list[dict]:
    """Generate an NSD-style trial sequence.

    Each trial has a category (cycling through NSD_CATEGORIES),
    a unique trial_id, and an onset in TR units.

    Parameters
    ----------
    n_trials : number of image presentations
    seed : random seed for category shuffling

    Returns
    -------
    List of dicts: {trial_id, category, onset_tr}
    """
    rng = np.random.RandomState(seed)
    categories = [NSD_CATEGORIES[i % len(NSD_CATEGORIES)] for i in range(n_trials)]
    rng.shuffle(categories)

    trials = []
    current_tr = 0
    for i in range(n_trials):
        trials.append({
            "trial_id": i,
            "category": categories[i],
            "onset_tr": current_tr,
        })
        current_tr += NSD_TRS_PER_TRIAL

    return trials


# ---------------------------------------------------------------------------
# NSDTrialProducer
# ---------------------------------------------------------------------------

class NSDTrialProducer(threading.Thread):
    """Producer thread: generates per-TR BOLD frames for an NSD session.

    Each image trial produces ~NSD_TRS_PER_TRIAL frames with
    category-specific spatial activation patterns. ISI TRs have
    only background noise (mimicking the HRF returning to baseline).

    Parameters
    ----------
    frame_queue : output queue for BOLD frames
    trial_sequence : list of trial dicts from make_nsd_trial_sequence
    n_vertices : fsaverage5 vertex count
    tr : pacing (0 = no sleep, for testing)
    key : JAX PRNG key
    """

    def __init__(
        self,
        frame_queue: queue.Queue,
        trial_sequence: list[dict],
        n_vertices: int = 20484,
        tr: float = 0.0,
        key: jax.Array | None = None,
    ):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.trials = trial_sequence
        self.n_vertices = n_vertices
        self.tr = tr
        self.key = key if key is not None else jax.random.PRNGKey(0)

    def run(self):
        key = self.key

        # Pre-build category spatial templates
        templates = {}
        for cat, (v_start, v_end) in _CATEGORY_REGIONS.items():
            key, subkey = jax.random.split(key)
            pattern = jax.random.normal(subkey, (self.n_vertices,)) * 0.05
            pattern = pattern.at[v_start:v_end].add(1.0)
            templates[cat] = np.asarray(pattern, dtype=np.float32)

        # Per-image variation: each image gets a unique sub-pattern
        image_offsets = {}
        for trial in self.trials:
            key, subkey = jax.random.split(key)
            offset = np.asarray(
                jax.random.normal(subkey, (self.n_vertices,)) * 0.15,
                dtype=np.float32,
            )
            image_offsets[trial["trial_id"]] = offset

        global_tr = 0
        stim_trs = max(1, round(NSD_STIM_DURATION / max(NSD_TR, 0.01)))
        isi_trs = max(1, NSD_TRS_PER_TRIAL - stim_trs)

        for trial in self.trials:
            cat = trial["category"]
            tid = trial["trial_id"]
            base_pattern = templates[cat] + image_offsets[tid]

            # Stimulus TRs (image on screen)
            for t in range(stim_trs):
                key, subkey = jax.random.split(key)
                noise = np.asarray(
                    jax.random.normal(subkey, (self.n_vertices,)) * 0.1,
                    dtype=np.float32,
                )
                # HRF ramp: signal builds over TRs
                hrf_weight = 0.5 + 0.5 * (t / max(stim_trs - 1, 1))
                bold = base_pattern * hrf_weight + noise

                self.frame_queue.put({
                    "bold": bold,
                    "tr_index": global_tr,
                    "trial_id": tid,
                    "category": cat,
                    "is_stimulus": True,
                    "timestamp": time.time(),
                })
                global_tr += 1
                if self.tr > 0:
                    time.sleep(self.tr)

            # ISI TRs (blank screen, signal decays)
            for t in range(isi_trs):
                key, subkey = jax.random.split(key)
                noise = np.asarray(
                    jax.random.normal(subkey, (self.n_vertices,)) * 0.1,
                    dtype=np.float32,
                )
                decay = 0.3 * (1.0 - t / max(isi_trs, 1))
                bold = base_pattern * decay + noise

                self.frame_queue.put({
                    "bold": bold,
                    "tr_index": global_tr,
                    "trial_id": tid,
                    "category": cat,
                    "is_stimulus": False,
                    "timestamp": time.time(),
                })
                global_tr += 1
                if self.tr > 0:
                    time.sleep(self.tr)

        self.frame_queue.put(None)


# ---------------------------------------------------------------------------
# NSDSessionAnalyzer
# ---------------------------------------------------------------------------

class NSDSessionAnalyzer:
    """Accumulates per-TR parcellated data and extracts per-trial betas.

    Parameters
    ----------
    n_vertices : cortical vertex count
    n_parcels : number of output parcels
    key : JAX PRNG key
    """

    def __init__(
        self,
        n_vertices: int = 20484,
        n_parcels: int = 100,
        key: jax.Array | None = None,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        self.n_parcels = n_parcels
        self.proj_params, self.project_fn = make_cortical_projection(
            n_vertices=n_vertices,
            n_parcels=n_parcels,
            key=key,
            init="block",
        )

        # Accumulate per-TR data grouped by trial
        self._trial_data: dict[int, list] = {}
        self._trial_stim_data: dict[int, list] = {}
        self.n_frames = 0

    def add_frame(
        self,
        bold_frame: jnp.ndarray,
        trial_id: int,
        is_stimulus: bool,
    ):
        """Add a single TR of data."""
        parcellated = self.project_fn(self.proj_params, bold_frame[None, :])
        parcel_vec = parcellated[0]  # (n_parcels,)

        if trial_id not in self._trial_data:
            self._trial_data[trial_id] = []
            self._trial_stim_data[trial_id] = []

        self._trial_data[trial_id].append(parcel_vec)
        if is_stimulus:
            self._trial_stim_data[trial_id].append(parcel_vec)

        self.n_frames += 1

    def extract_betas(self) -> jnp.ndarray:
        """Extract per-trial beta estimates by averaging stimulus TRs.

        Returns
        -------
        (n_trials, n_parcels) beta matrix
        """
        trial_ids = sorted(self._trial_stim_data.keys())
        betas = []
        for tid in trial_ids:
            stim_frames = self._trial_stim_data[tid]
            if stim_frames:
                beta = jnp.mean(jnp.stack(stim_frames), axis=0)
            else:
                # Fallback to all frames if no stimulus frames
                beta = jnp.mean(jnp.stack(self._trial_data[tid]), axis=0)
            betas.append(beta)

        return jnp.stack(betas)


# ---------------------------------------------------------------------------
# Representational dissimilarity matrix
# ---------------------------------------------------------------------------

def compute_rdm(betas: jnp.ndarray) -> jnp.ndarray:
    """Compute a representational dissimilarity matrix (1 - correlation).

    Parameters
    ----------
    betas : (n_trials, n_features) trial-wise activation patterns

    Returns
    -------
    (n_trials, n_trials) RDM, values in [0, 2]
    """
    fc = corr(betas.T)  # correlation between trials (treating features as "time")
    # Actually we want trial×trial correlation
    # corr expects (features, observations), so for trial×trial:
    r = corr(betas)  # (n_trials, n_trials) if betas is (n_trials, n_features)
    # corr in jaxoccoli takes (N, T) and computes N×N correlation
    # We want trial×trial, so we need betas as (n_trials, n_parcels)
    # and compute correlation across parcels dimension
    # corr(betas) with betas (n_trials, n_parcels) gives (n_trials, n_trials)
    rdm = 1.0 - r
    # Zero diagonal
    rdm = rdm - jnp.diag(jnp.diag(rdm))
    return rdm


# ---------------------------------------------------------------------------
# Full session runner
# ---------------------------------------------------------------------------

def run_nsd_session(
    n_trials: int = 50,
    n_vertices: int = 20484,
    n_parcels: int = 100,
    key: jax.Array | None = None,
    tr: float = 0.0,
    verbose: bool = False,
) -> dict:
    """Run a complete simulated NSD session.

    Parameters
    ----------
    n_trials : number of image presentations
    n_vertices : fsaverage5 vertex count
    n_parcels : output parcels
    key : JAX PRNG key
    tr : pacing (0 = no sleep)
    verbose : print per-TR info

    Returns
    -------
    Dict with: betas, rdm, fc, categories, trial_sequence, latencies
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    trial_sequence = make_nsd_trial_sequence(n_trials)
    frame_queue = queue.Queue(maxsize=100)

    key, k1, k2 = jax.random.split(key, 3)
    producer = NSDTrialProducer(
        frame_queue, trial_sequence,
        n_vertices=n_vertices, tr=tr, key=k1,
    )
    analyzer = NSDSessionAnalyzer(
        n_vertices=n_vertices, n_parcels=n_parcels, key=k2,
    )

    producer.start()
    latencies = []
    categories = [t["category"] for t in trial_sequence]

    while True:
        try:
            frame = frame_queue.get(timeout=10)
        except queue.Empty:
            break
        if frame is None:
            break

        t0 = time.time()
        analyzer.add_frame(
            jnp.asarray(frame["bold"]),
            trial_id=frame["trial_id"],
            is_stimulus=frame["is_stimulus"],
        )
        elapsed = time.time() - t0
        latencies.append(elapsed)

        if verbose:
            stim_flag = "STIM" if frame["is_stimulus"] else " ISI"
            print(
                f"  TR {frame['tr_index']:4d} | trial {frame['trial_id']:3d} "
                f"| {frame['category']:7s} | {stim_flag} | {elapsed*1000:.1f}ms"
            )

    producer.join(timeout=5)

    # Extract betas and compute metrics
    betas = analyzer.extract_betas()
    rdm = compute_rdm(betas)
    fc = corr(betas.T)  # parcel×parcel FC from beta series

    return {
        "betas": betas,
        "rdm": rdm,
        "fc": fc,
        "categories": categories,
        "trial_sequence": trial_sequence,
        "latencies": latencies,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 64)
    print("NSD Picture-Watching Simulation: TRIBEv2 → jaxoccoli")
    print("(Simulated — no model download or NSD data required)")
    print("=" * 64)

    N_TRIALS = 50
    N_VERTICES = 20484
    N_PARCELS = 100

    key = jax.random.PRNGKey(42)

    print(f"\nParadigm: {N_TRIALS} COCO images, {NSD_STIM_DURATION}s on / "
          f"{NSD_ISI}s ISI, TR={NSD_TR}s")
    print(f"Pipeline: {N_VERTICES} vertices → {N_PARCELS} parcels")
    print(f"Categories: {NSD_CATEGORIES}")

    # --- Warmup ---
    print("\n[Warmup] Compiling pipeline...")
    t0 = time.time()
    _ = run_nsd_session(n_trials=3, n_vertices=N_VERTICES,
                        n_parcels=N_PARCELS, key=key, tr=0.0)
    print(f"[Warmup] Done in {time.time()-t0:.1f}s")

    # --- Run session ---
    print(f"\n[Session] Running {N_TRIALS}-trial session...")
    key, subkey = jax.random.split(key)
    t0 = time.time()
    results = run_nsd_session(
        n_trials=N_TRIALS,
        n_vertices=N_VERTICES,
        n_parcels=N_PARCELS,
        key=subkey,
        tr=0.0,
        verbose=False,
    )
    session_time = time.time() - t0
    print(f"[Session] Complete in {session_time:.1f}s")

    # --- Results ---
    betas = results["betas"]
    rdm = results["rdm"]
    fc = results["fc"]
    categories = results["categories"]
    latencies = results["latencies"]

    print(f"\n{'='*64}")
    print("Results:")
    print(f"  Betas:       {betas.shape}")
    print(f"  RDM:         {rdm.shape}")
    print(f"  FC:          {fc.shape}")
    print(f"  Per-TR:      avg={np.mean(latencies)*1000:.1f}ms  "
          f"max={np.max(latencies)*1000:.1f}ms")
    total_trs = len(latencies)
    real_time = total_trs * NSD_TR
    print(f"  Throughput:  {total_trs} TRs in {session_time:.1f}s "
          f"({total_trs/session_time:.0f} TRs/s, "
          f"real-time would be {real_time:.0f}s)")

    # --- RSA: within vs between category distances ---
    print(f"\nRepresentational Similarity Analysis:")
    within_dists = []
    between_dists = []
    for i in range(len(categories)):
        for j in range(i + 1, len(categories)):
            d = float(rdm[i, j])
            if categories[i] == categories[j]:
                within_dists.append(d)
            else:
                between_dists.append(d)

    if within_dists and between_dists:
        w_mean = np.mean(within_dists)
        b_mean = np.mean(between_dists)
        print(f"  Within-category RDM:  {w_mean:.4f} (n={len(within_dists)})")
        print(f"  Between-category RDM: {b_mean:.4f} (n={len(between_dists)})")
        if w_mean < b_mean:
            print("  ✓ Category structure recovered: within < between")
        else:
            print("  ~ Category structure weak (expected with random projection)")

    # --- Per-category mean beta pattern ---
    print(f"\nPer-category mean activation (top-3 parcels):")
    for cat in NSD_CATEGORIES:
        cat_idx = [i for i, c in enumerate(categories) if c == cat]
        if cat_idx:
            mean_beta = jnp.mean(betas[jnp.array(cat_idx)], axis=0)
            top3 = jnp.argsort(mean_beta)[-3:][::-1]
            top3_vals = mean_beta[top3]
            parcels_str = ", ".join(
                f"P{int(p)}={float(v):.3f}" for p, v in zip(top3, top3_vals)
            )
            print(f"  {cat:7s} (n={len(cat_idx):2d}): {parcels_str}")

    # --- FC summary ---
    mask = 1.0 - jnp.eye(N_PARCELS)
    off_diag = fc * mask
    print(f"\nBeta-series FC:")
    print(f"  Range:    [{float(fc.min()):.3f}, {float(fc.max()):.3f}]")
    print(f"  Mean:     {float(jnp.sum(off_diag) / jnp.sum(mask)):.4f}")
    print(f"  Std:      {float(jnp.std(off_diag[mask > 0])):.4f}")

    print(f"\n{'='*64}")
    print("With TRIBEv2 + real NSD images:")
    print("  from tribev2 import TribeModel")
    print("  model = TribeModel.from_pretrained('facebook/tribev2')")
    print("  for img_path in nsd_image_paths:")
    print("      events = model.get_events_dataframe(video_path=img_path)")
    print("      preds, _ = model.predict(events=events)")
    print("      # preds[:, :] → feed into NSDSessionAnalyzer")
    print("=" * 64)


if __name__ == "__main__":
    main()
