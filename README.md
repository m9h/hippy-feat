# hippy-feat

**GPU-accelerated fMRI preprocessing and differentiable connectivity analysis in JAX.**

[![Tests](https://img.shields.io/badge/tests-221%20passing-brightgreen)]()
[![JAX](https://img.shields.io/badge/JAX-0.9+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-lightgrey)]()

---

## The Problem

State-of-the-art brain decoding models like [MindEye2](https://arxiv.org/abs/2403.11207) are trained on meticulously preprocessed fMRI data -- hours of offline processing with [fMRIPrep](https://fmriprep.org) and [GLMsingle](https://glmsingle.readthedocs.io). But when deployed in real-time, preprocessing is reduced to a bare-bones GLM, creating a **train/test mismatch** that degrades reconstruction quality.

Beyond this, the standard beta series correlation pipeline (Rissman et al. 2004; Mumford et al. 2012) discards variance information at each stage: OLS beta estimates are point estimates with no uncertainty, and correlating noisy point estimates produces biased connectivity.

**hippy-feat** bridges both gaps:

1. **Real-time preprocessing**: JAX/XLA-compiled pipeline runs in ~54ms per volume on GPU (28x headroom within a 1.5s TR)
2. **Differentiable connectivity**: backpropagate through parcellation, covariance estimation, and spectral embedding
3. **Variance propagation**: Bayesian GLM outputs `(beta_mean, beta_var)` and uncertainty flows end-to-end through parcellation to connectivity

---

## Architecture

### jaxoccoli -- JAX neuroimaging library

19 modules, ~4,000 LOC, 221 tests. Pure JAX with vbjax-style factory functions (`make_*() -> (params, forward_fn)`).

#### Real-time preprocessing

| Module | What | Speed (76x90x74 vol) |
|--------|------|---------------------|
| `glm.py` | JIT-compiled OLS General Linear Model | ~0.3ms |
| `spatial.py` | 3D bilateral filter + spherical convolution | 3.7ms |
| `motion.py` | Gauss-Newton rigid-body registration (6 DOF) | 16.4ms |
| `stats.py` | T-statistics, F-contrasts, p-values | <1ms |
| `signatures.py` | Log-signature streaming features via [signax](https://github.com/anh-tong/signax) | ~5ms |
| `permutation.py` | Non-parametric permutation testing | configurable |
| `fusion.py` | EEG-fMRI balloon model fusion | experimental |
| `io.py` | NIfTI I/O via nibabel | -- |

#### Differentiable connectivity analysis

Reimplements the essential algorithms from [hypercoil](https://github.com/hypercoil/hypercoil) (Stanford/Poldrack Lab) without the Equinox dependency, following [vbjax](https://github.com/ins-amu/vbjax) architectural patterns.

| Module | What |
|--------|------|
| `covariance.py` | Covariance, correlation, precision, partial correlation + **variance-aware** extensions (`weighted_corr`, `attenuated_corr`, `posterior_corr`) |
| `matrix.py` | SPD manifold operations (tangent projection, Frechet mean, Cholesky inversion, sym2vec/vec2sym) |
| `fourier.py` | Analytic signal, Hilbert transform, envelope, instantaneous phase/frequency, frequency-domain filtering |
| `graph.py` | Graph Laplacian, modularity, Laplacian eigenmaps, diffusion maps, Chebyshev spectral filter, sparse message passing |
| `interpolate.py` | Linear, spectral, and hybrid interpolation for censored/scrubbed frames |
| `connectivity.py` | Sliding window correlation, dynamic connectivity, temporal windowing |

#### Learnable components (vbjax factory pattern)

All factories return `(params, forward_fn)` tuples -- no Equinox, no Flax.

| Module | What |
|--------|------|
| `learnable.py` | `make_atlas_linear` (differentiable parcellation), `make_atlas_linear_uncertain` (variance-aware), `make_atlas_natural_grad` (Fisher-Rao), `make_learnable_cov`, `make_freq_filter`, parameter constraints (simplex, SPD, orthogonal), Fisher-Rao natural gradient |
| `losses.py` | Entropy, KL, JS divergence, modularity loss, connectopy loss, eigenmaps loss, compactness/dispersion/tether, QC-FC, multivariate kurtosis, variance-aware losses |
| `transport.py` | Log-domain Sinkhorn, Wasserstein distance, Gromov-Wasserstein for cross-subject FC comparison without parcellation alignment |

#### Bayesian beta estimation

Addresses the Rissman/Mumford variance propagation gap, informed by [BROCCOLI](https://github.com/wanderine/BROCCOLI) (Eklund et al. 2014).

| Module | What |
|--------|------|
| `bayesian_beta.py` | `make_conjugate_glm` (closed-form real-time path, ~0.5ms/voxel), `make_ar1_conjugate_glm` (AR(1) prewhitened), `make_bayesian_glm` (full NUTS via blackjax, offline) |

### End-to-end variance-propagating pipeline

```
BOLD volume
  |
  v
make_conjugate_glm  -->  (beta_mean, beta_var)    # Bayesian, not OLS
  |
  v
make_atlas_linear_uncertain  -->  (parc_mean, parc_var)  # variance through parcellation
  |
  v
posterior_corr  -->  FC matrix                     # disattenuated connectivity
  |
  v
modularity_loss / eigenmaps_loss / decoder         # differentiable end-to-end
```

Gradients flow through the entire chain. The atlas weights, filter parameters, and loss objectives can all be jointly optimized.

### RT preprocessing variants

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
| **G** | Full Bayesian (planned) | ~30-60s | NUTS sampling, full posterior, AR(p) noise |

---

## Design Influences

### hypercoil (Stanford/Poldrack Lab)
Differentiable programming for fMRI connectivity (Ciric et al. 2022). We reimplemented the essential algorithms (~80% of useful functionality in ~15% of the LOC) without the Equinox dependency, using vbjax-style factory functions instead.

### vbjax (INS-AMU)
Architectural model: factory functions, namedtuples, plain JAX, minimal dependencies. Our `make_*()` pattern follows vbjax conventions.

### BROCCOLI (Eklund et al.)
GPU-accelerated Bayesian fMRI analysis. Our `make_conjugate_glm` implements BROCCOLI's Gibbs Block 1 (conjugate normal-inverse-gamma posterior) as a closed-form step, and `make_ar1_conjugate_glm` adds AR(1) prewhitening with precomputed S matrices.

### hgx (hypergraph neural networks)
Chebyshev spectral filtering (O(K*nnz) without eigendecomposition), log-domain Sinkhorn for optimal transport, Fisher-Rao natural gradient on the probability simplex, and sparse message passing via `segment_sum` were adapted from hgx for scalable cortical surface analysis.

### Beta series correlations (Rissman et al. 2004; Mumford et al. 2012)
The variance-aware extensions (`posterior_corr`, `make_atlas_linear_uncertain`, `make_conjugate_glm`) directly address the two-stage estimation problem where OLS betas discard uncertainty before the correlation step.

---

## Dimensionality estimation

Comprehensive analysis across the [Natural Scenes Dataset](https://naturalscenesdataset.org) (8 subjects, 7T) and 3T single-subject data (4 sessions):

- **Eigenspectrum analysis** with truncated SVD
- **Broken stick model** (conservative estimate)
- **Parallel analysis** with column-shuffled null distribution
- **MELODIC consensus** subsampling (Beckmann et al.)

Key finding: **BIC estimates 11-18 components** consistently across 7/8 NSD subjects. 7T data has ~5x higher intrinsic dimensionality than 3T.

---

## Quick start

```bash
# Install dependencies (prefer uv)
uv pip install jax[cuda12] signax nibabel scipy optax pytest hypothesis

# Run all tests (221 tests)
python -m pytest tests/ scripts/tests/ -v

# Run only the connectivity/learnable module tests
python -m pytest tests/test_covariance.py tests/test_graph.py tests/test_learnable.py tests/test_bayesian_beta.py tests/test_hgx_ports.py -v

# Benchmark RT variants on GPU
python scripts/benchmark_variants.py \
  --variants a_baseline c_pervoxel_hrf d_bayesian \
  --session ses-06 --runs 1 --no-trackio
```

### Container (recommended for DGX Spark)

```bash
docker build -f Dockerfile.mindeye-variants -t mindeye-variants .
docker run --gpus all -v /data:/data mindeye-variants
```

---

## Project structure

```
hippy-feat/
├── jaxoccoli/                        # JAX neuroimaging library (19 modules, ~4K LOC)
│   ├── glm.py                        # General Linear Model (OLS, JIT)
│   ├── spatial.py                    # Bilateral filter + spherical convolution
│   ├── motion.py                     # Gauss-Newton + Adam registration
│   ├── stats.py                      # T/F statistics
│   ├── signatures.py                 # Log-signature features (signax)
│   ├── permutation.py                # Permutation testing
│   ├── fusion.py                     # EEG-fMRI balloon model
│   ├── io.py                         # NIfTI I/O
│   ├── covariance.py                 # Covariance + variance-aware extensions
│   ├── matrix.py                     # SPD manifold operations
│   ├── fourier.py                    # Analytic signal, freq-domain filtering
│   ├── graph.py                      # Laplacian, eigenmaps, Chebyshev, sparse ops
│   ├── interpolate.py                # Temporal interpolation for censored frames
│   ├── connectivity.py               # Sliding window + dynamic FC
│   ├── learnable.py                  # Factory functions for differentiable components
│   ├── losses.py                     # Differentiable loss functions
│   ├── transport.py                  # Optimal transport (Sinkhorn, Wasserstein, GW)
│   └── bayesian_beta.py              # Bayesian GLM with variance propagation
├── scripts/
│   ├── rt_glm_variants.py            # 8 preprocessing variants + framework
│   ├── benchmark_variants.py         # Benchmark runner with trackio
│   ├── dimensionality_analysis.py
│   ├── nsd_multisubject_dimest.py
│   ├── run_melodic_dimest.py         # FSL MELODIC integration
│   └── run_mindeye_inference.py      # MindEye model inference
├── tests/                            # 221 tests
│   ├── test_covariance.py            # 32 tests
│   ├── test_matrix.py                # 27 tests
│   ├── test_fourier.py               # 21 tests
│   ├── test_graph.py                 # 22 tests
│   ├── test_interpolate.py           # 13 tests
│   ├── test_learnable.py             # 29 tests
│   ├── test_losses.py                # 27 tests
│   ├── test_bayesian_beta.py         # 16 tests
│   ├── test_hgx_ports.py             # 34 tests
│   ├── test_spatial.py               # Bilateral filter + GN motion tests
│   ├── test_geometry.py
│   ├── test_riemannian_ml.py
│   └── test_sklearn_compliance.py
├── scripts/tests/
│   ├── conftest.py                   # Shared fixtures
│   └── test_rt_glm_variants.py       # 73 variant tests (TDD)
├── docs/
│   ├── DESIGN_bayesian_first_level.md
│   └── DESIGN_differentiable_connectivity.md
├── smoke_test_realtime.py
├── smoke_test_rt_cloud.py
└── Dockerfile.mindeye-variants
```

---

## Dependencies

**Required**: `jax`, `optax`, `nibabel`, `scipy`, `numpy`

**Optional**:
- `signax` -- log-signature features (Variant F)
- `blackjax` -- full Bayesian NUTS sampling (Variant G)
- `hypothesis` -- property-based testing
- `matplotlib` -- visualization

No Equinox, no Flax, no TensorFlow, no PyTorch.

---

## References

- Ciric R, Thomas AW, Esteban O, Poldrack RA (2022). Differentiable programming for functional connectomics. *ML4H / PMLR* 193:419-455.
- Eklund A, Dufort P, Villani M, LaConte S (2014). BROCCOLI: Software for fast fMRI analysis on many-core CPUs and GPUs. *Front Neuroinform* 8:24.
- Rissman J, Gazzaley A, D'Esposito M (2004). Measuring functional connectivity during distinct stages of a cognitive task. *NeuroImage* 23(2):752-763.
- Mumford JA, Turner BO, Ashby FG, Poldrack RA (2012). Deconvolving BOLD activation in event-related designs for multivoxel pattern analysis. *NeuroImage* 59(3):2636-2643.
- Scotti PS et al. (2024). MindEye2: Shared-subject models enable fMRI-to-image with 1 hour of data. *ICML 2024*.

---

## Citation

```bibtex
@software{hough2026hippyfeat,
  author = {Hough, Morgan G.},
  title = {hippy-feat: GPU-accelerated fMRI preprocessing and differentiable connectivity analysis in JAX},
  year = {2026},
  url = {https://github.com/m9h/hippy-feat}
}
```

---

*Built with JAX on NVIDIA DGX Spark (GB10) by [Morgan G. Hough](https://github.com/m9h) -- Center17 | OREL | Biopunk Lab | NeuroTechX*
