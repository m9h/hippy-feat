# hippy-feat

**GPU-accelerated fMRI preprocessing and differentiable connectivity analysis in JAX.**

hippy-feat bridges the gap between offline fMRI preprocessing (fMRIPrep, GLMsingle) and
real-time deployment, providing a JAX/XLA-compiled pipeline that runs in ~54ms per volume
on GPU. Its core library, **jaxoccoli**, implements 19 modules (~4,000 LOC) of pure-JAX
primitives for the full fMRI analysis pipeline -- from motion correction and GLM estimation
through differentiable connectivity analysis with end-to-end variance propagation.

## Key Features

- **Real-time preprocessing**: 8 interchangeable strategies, all JIT-compiled
- **Differentiable connectivity**: Backpropagate through parcellation, covariance, and spectral embedding
- **Variance propagation**: Bayesian GLM outputs `(beta_mean, beta_var)` flowing end-to-end
- **vbjax factory pattern**: `make_*() -> (params, forward_fn)` -- no Equinox, no Flax

## Getting Started

```bash
uv pip install -e ".[all]"
python -m pytest tests/ -v
```

## Contents

```{toctree}
:maxdepth: 2

tutorials/bayesian_variance_propagation
tutorials/connectivity_analysis
design/index
tutorials/motion_correction
tutorials/learnable_parcellation
tutorials/realtime_pipeline
reference/jaxoccoli
```

## Indices and Tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
