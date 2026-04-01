# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-03-30

### Added
- Publication figures for resting-state FC gradient analysis
- `pyproject.toml` with hatchling build system and optional dependency groups
- WAND fieldmap correction support
- neuromaps FC analysis integration
- 8 new jaxoccoli modules for differentiable connectivity analysis:
  `covariance`, `matrix`, `fourier`, `graph`, `interpolate`, `learnable`,
  `losses`, `transport`
- Bayesian variance propagation pipeline (`bayesian_beta.py`)
- Variance-aware atlas parcellation (`make_atlas_linear_uncertain`)
- Posterior correlation (`posterior_corr`, `attenuated_corr`, `weighted_corr`)
- Optimal transport for FC comparison (Sinkhorn, Wasserstein, Gromov-Wasserstein)
- Fisher-Rao natural gradient for simplex-constrained parameters
- Chebyshev spectral filtering and sparse message passing (from hgx)
- Sliding window and dynamic functional connectivity
- Comprehensive test suite (221 tests)
- Design documents for Bayesian first-level analysis and differentiable connectivity

## [0.1.0] - 2026-03-25

### Added
- Initial MindEye RT preprocessing comparison framework
- Core jaxoccoli modules: `glm`, `spatial`, `motion`, `stats`, `signatures`,
  `permutation`, `fusion`, `io`, `connectivity`
- 8 RT preprocessing variants (A through G)
- Gauss-Newton rigid-body registration (6 DOF)
- JIT-compiled OLS General Linear Model
- 3D bilateral filter and spherical convolution
- Log-signature streaming features via signax
- EEG-fMRI balloon model fusion (experimental)
- Dockerfile for DGX Spark deployment
- README with architecture overview and benchmarks

[0.2.0]: https://github.com/m9h/hippy-feat/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/m9h/hippy-feat/releases/tag/v0.1.0
