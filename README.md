# 🧠 hippy-feat

**GPU-accelerated fMRI preprocessing in JAX — closing the gap between offline and real-time neuroimaging.**

[![Tests](https://img.shields.io/badge/tests-87%20passing-brightgreen)]()
[![JAX](https://img.shields.io/badge/JAX-0.9+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## 🎯 The Problem

State-of-the-art brain decoding models like [MindEye2](https://arxiv.org/abs/2403.11207) are trained on meticulously preprocessed fMRI data — hours of offline processing with [fMRIPrep](https://fmriprep.org) and [GLMsingle](https://glmsingle.readthedocs.io). But when deployed in real-time, the preprocessing is reduced to a bare-bones GLM, creating a **train/test mismatch** that degrades reconstruction quality.

| Stage | Offline (training) | Real-Time (current) |
|-------|-------------------|-------------------|
| Confound regression | CompCor (WM/CSF) | **None** |
| HRF model | Per-voxel (20-library) | **Single Glover** |
| Noise regression | GLMdenoise | **None** |
| Regularization | Ridge (cross-validated) | **None** |

**hippy-feat** bridges this gap with JAX/XLA-compiled preprocessing that runs the full pipeline in **~54ms per volume** on GPU — well within a 1.5s TR budget.

---

## 🏗️ Architecture

### jaxoccoli — JAX neuroimaging library

| Module | What | Speed (76×90×74 vol) |
|--------|------|---------------------|
| `glm.py` | JIT-compiled OLS General Linear Model | ~0.3ms |
| `spatial.py` | 3D bilateral filter (edge-preserving denoising) | **3.7ms** |
| `motion.py` | Gauss-Newton rigid-body registration (6 DOF) | **16.4ms** |
| `stats.py` | T-statistics, F-contrasts, p-values | <1ms |
| `signatures.py` | Log-signature streaming features via [signax](https://github.com/anh-tong/signax) | ~5ms |
| `permutation.py` | Non-parametric permutation testing | configurable |
| `connectivity.py` | Functional connectivity analysis | configurable |

### RT Preprocessing Variants

Eight interchangeable preprocessing strategies, all producing `(8627,)` z-scored beta vectors:

| Variant | Approach | Per-TR Time | Key Idea |
|---------|----------|-------------|----------|
| **A** | Glover HRF baseline | 163ms | Current RT-MindEye approach |
| **A+N** | + CSF/WM nuisance regression | ~170ms | Closes the biggest GLMsingle gap |
| **B** | FLOBS 3-basis HRF | 155ms | Captures HRF shape variability |
| **C** | Per-voxel HRF (GLMsingle-style) | **33ms** | 20-HRF library, pre-selected per voxel |
| **D** | Bayesian shrinkage | 237ms | Conjugate Gaussian with training priors |
| **E** | Spatial Laplacian regularization | 103ms | Graph-based smoothing preserving boundaries |
| **F** | Log-signature monitoring | 187ms | Streaming artifact/drift detection |
| **C+D** | Per-voxel HRF + Bayesian | **39ms** | Theoretically strongest combination |

### Proposed Real-Time Pipeline

```
Volume arrives (TR = 1.5s)
  │
  ├─ Gauss-Newton motion correction ────── 16ms
  ├─ Bilateral filter (edge-preserving) ── 4ms
  ├─ CSF/WM nuisance regressors ────────── 0.1ms
  ├─ Per-voxel HRF GLM (Variant C) ────── 33ms
  ├─ Bayesian shrinkage ────────────────── 1ms
  │                                    ═══════
  │                              Total: ~54ms
  │
  └─ Z-score → MindEye forward pass → reconstruction
```

**28× headroom** within the 1.5s TR budget.

---

## 📊 Dimensionality Estimation

Comprehensive analysis across the [Natural Scenes Dataset](https://naturalscenesdataset.org) (8 subjects, 7T) and 3T single-subject data (4 sessions):

- **Eigenspectrum analysis** with truncated SVD
- **Broken stick model** (conservative estimate)
- **Parallel analysis** with column-shuffled null distribution
- **MELODIC consensus** subsampling (Beckmann et al.)
- **FSL MELODIC** with all estimators (LAP, BIC, MDL, AIC, mean)

Key finding: **BIC estimates 11–18 components** consistently across 7/8 NSD subjects. 7T data has ~5× higher intrinsic dimensionality than 3T.

---

## 🔬 Anatomy-Informed Processing

Integration with [FastSurfer](https://deep-mi.org/research/fastsurfer/) for in-session anatomical processing:

- **60-second pathway**: FastSurfer CNN segmentation → WM/CSF masks → nuisance regression + tissue-informed Laplacian
- **45-minute pathway**: Full surface reconstruction → geodesic smoothing + surface-based registration
- Parcellation-based Bayesian priors from Desikan-Killiany atlas

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install jax[cuda12] signax nibabel scipy matplotlib pytest hypothesis

# Run tests (87 tests)
cd hippy-feat
python -m pytest tests/ scripts/tests/ -v

# Benchmark all variants on GPU
python scripts/benchmark_variants.py \
  --variants a_baseline c_pervoxel_hrf d_bayesian \
  --session ses-06 --runs 1 --no-trackio

# Dimensionality estimation
python scripts/dimensionality_analysis.py

# Multi-subject NSD analysis
python scripts/nsd_multisubject_dimest.py
```

### Container (recommended for DGX Spark)

```bash
docker build -f Dockerfile.mindeye-variants -t mindeye-variants .
docker run --gpus all -v /data:/data mindeye-variants
```

---

## 📁 Project Structure

```
hippy-feat/
├── jaxoccoli/                    # JAX neuroimaging library
│   ├── glm.py                    # General Linear Model (OLS, JIT)
│   ├── spatial.py                # Bilateral filter
│   ├── motion.py                 # Gauss-Newton + Adam registration
│   ├── stats.py                  # T/F statistics
│   ├── signatures.py             # Log-signature features (signax)
│   ├── permutation.py            # Permutation testing
│   └── connectivity.py           # Functional connectivity
├── scripts/
│   ├── rt_glm_variants.py        # 8 preprocessing variants + framework
│   ├── benchmark_variants.py     # Benchmark runner with trackio
│   ├── dimensionality_analysis.py
│   ├── nsd_multisubject_dimest.py
│   ├── run_melodic_dimest.py     # FSL MELODIC integration
│   └── run_mindeye_inference.py  # MindEye model inference (NGC container)
├── tests/
│   └── test_spatial.py           # Bilateral filter + GN MC tests
├── scripts/tests/
│   ├── conftest.py               # Shared fixtures
│   └── test_rt_glm_variants.py   # 73 variant tests (TDD)
├── smoke_test_realtime.py        # RT simulation benchmark
├── smoke_test_rt_cloud.py        # RT-Cloud file-watching pipeline
└── Dockerfile.mindeye-variants   # NGC-based container
```

---

## 🔗 Related Projects

| Project | Role |
|---------|------|
| [MindEye2](https://arxiv.org/abs/2403.11207) | Core brain decoding model (Scotti et al.) |
| [RT-Cloud MindEye](https://github.com/brainiak/rtcloud-projects) | Real-time pipeline (Iyer et al., 2026) |
| [GLMsingle](https://glmsingle.readthedocs.io) | Offline beta estimation gold standard |
| [fMRIPrep](https://fmriprep.org) | Offline fMRI preprocessing |
| [FastSurfer](https://deep-mi.org/research/fastsurfer/) | GPU-accelerated anatomical segmentation |
| [NSD](https://naturalscenesdataset.org) | Training dataset (8 subjects, 7T) |

---

## 📄 Citation

If you use hippy-feat in your research, please cite:

```bibtex
@software{hough2026hippyfeat,
  author = {Hough, Morgan G.},
  title = {hippy-feat: GPU-accelerated fMRI preprocessing for real-time brain decoding},
  year = {2026},
  url = {https://github.com/m9h/hippy-feat}
}
```

---

*Built with JAX on NVIDIA DGX Spark (GB10) by [Morgan G. Hough](https://github.com/m9h) — Center17 | OREL | Biopunk Lab | NeuroTechX*
