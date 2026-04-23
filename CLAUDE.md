# hippy-feat / jaxoccoli

GPU-accelerated fMRI preprocessing and differentiable connectivity analysis in JAX.

## Project structure

- `jaxoccoli/` — core library (22 modules, ~5500 LOC)
- `tests/` — pytest suite, run with `python -m pytest` (uv has jaxlib platform issues on this mac)
- `scripts/` — analysis scripts (benchmarks, NSD validation, MindEye inference)
- `smoke_test_*.py` — demo scripts (realtime, rt-cloud, tribe, rt-tribe, nsd-tribe)
- `docs/` — Sphinx documentation with RTD config

## Key conventions

- **Factory pattern**: `make_*() → (params, forward_fn)` — params are NamedTuples, no Equinox
- **TDD**: red-green-refactor. Write failing tests first, then implement
- **Test runner**: `python -m pytest tests/` (uv run pytest fails due to jaxlib wheel unavailability on this macOS)
- **Pure JAX**: JIT/grad/vmap compatible everywhere. No Python control flow in hot paths

## Module groups

### Real-time preprocessing
`glm.py`, `spatial.py`, `motion.py`, `stats.py`, `permutation.py`, `io.py`, `signatures.py`, `fusion.py`

### Differentiable connectivity
`covariance.py`, `matrix.py`, `graph.py`, `interpolate.py`, `learnable.py`, `losses.py`, `connectivity.py`, `fourier.py`, `transport.py`, `bayesian_beta.py`, `multivariate.py`

### Foundation model integration (new)
- `hf_encoder.py` — HuggingFace adapter pattern: `make_hf_encoder(model_id)`, TribeV2Adapter, RaramuriAdapter (HTTP client for local Raramuri server), `make_cortical_projection` (vertex→parcel with block-diagonal init)
- `dot_adapter.py` — dot-jax FEM mesh → cortical surface bridge: `make_mesh_to_cortex`, `DOTFrameProcessor`
- `angiography.py` — TOF-MRA pipeline: Frangi→skeleton→radii→VesselTree (interface contract with vpjax)
- `nsd.py` — NSD validation: RSA (`rdm_from_betas`, `compare_rdms`), noise ceiling, category selectivity

### Task 2.1 — fMRIPrep vs GLMsingle contributions to RT gap

Discord-assigned task. See `TASK_2_1_STATUS.md` in the repo root for the current
state, the canonical paper checkpoint path, the finalmask derivation, and
resume instructions. Relevant scripts:
- `scripts/task_2_1_factorial.py` / `.sbatch` — produces per-trial betas under factorial conditions
- `scripts/mindeye_retrieval_eval.py` / `.sbatch` — retrieval-only inference using NGC PyTorch 26.03 arm64 SIF
- `scripts/download_*.sbatch` — HF data/checkpoint/stimuli pulls
- `scripts/pull_pytorch_ngc.sbatch` — builds `/data/derivatives/containers/pytorch_26.03.sif` (needs 64 GB RAM for mksquashfs)

## Cross-project interfaces

- **vpjax**: `angiography.py` outputs `{points, radii, branch_ids}` dict → consumed by `vpjax.vascular.angiography.VesselTree`
- **dot-jax**: `dot_adapter.py` consumes `RealtimePipeline.process_frame()` output `(hbo, hbr)` on FEM mesh
- **meeg-benchmark**: `hf_encoder.py` adapter pattern reusable for Zuna, REVE, BrainOmni EEG foundation models
- **TRIBEv2**: `TribeV2Adapter` registered for `facebook/tribev2` — video/audio/text → fsaverage5 BOLD

## Demo scripts

| Script | What | Requires |
|--------|------|----------|
| `smoke_test_realtime.py` | GLM + permutation within TR budget | jaxoccoli |
| `smoke_test_rt_cloud.py` | File I/O + motion correction + GLM (NIfTI) | jaxoccoli + nibabel |
| `smoke_test_tribe.py` | Offline: BOLD → FC → embedding → modularity | jaxoccoli |
| `smoke_test_rt_tribe.py` | Streaming: producer-consumer per-TR FC | jaxoccoli |
| `smoke_test_nsd_tribe.py` | NSD picture-watching: per-trial RSA | jaxoccoli |
| `scripts/nsd_tribe_validation.py` | Predicted vs actual NSD (DGX Spark) | jaxoccoli + nibabel + matplotlib |

## Infrastructure

- **DGX Spark**: `/data/3t/nsd_multisubject/` (8 NSD subjects), `/data/derivatives/`
- **VMTK**: `mhough/neuro/vmtk` brew formula for centerline extraction (Python bindings for python@3.14)
- **Containers**: `Dockerfile.mindeye-variants` (NGC PyTorch + JAX CUDA)

## Data paths (DGX Spark)

- NSD betas: `/data/3t/nsd_multisubject/{subj01..subj08}/betas_session{01..03}.nii.gz`
- NSD masks: `/data/3t/nsd_multisubject/{subj01..subj08}_nsdgeneral.nii.gz`
- Derivatives: `/data/derivatives/mindeye_variants/`, `/data/derivatives/tribe_validation/`
