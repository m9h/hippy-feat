# Contributing to hippy-feat

Thank you for your interest in contributing to hippy-feat! This project provides
GPU-accelerated fMRI preprocessing and differentiable connectivity analysis in
JAX, built around the jaxoccoli library.

---

## 1. Getting Started

**Prerequisites**: Python 3.11+, uv (preferred over pip), JAX with CUDA support
(optional but recommended).

```bash
# Clone and install in editable mode with all extras
git clone https://github.com/m9h/hippy-feat.git
cd hippy-feat
uv pip install -e ".[all]"

# Verify installation
python -m pytest tests/ -v --tb=short
```

We use **uv** for all package management. Do not use bare `pip install`.

---

## 2. Development Workflow

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feat/<short-description>
   ```
2. **Write tests first** (TDD). All new jaxoccoli modules need a corresponding
   `tests/test_<module>.py`.
3. **Run the test suite** before opening a PR:
   ```bash
   python -m pytest tests/ scripts/tests/ -v
   ```
4. **Keep commits atomic** -- one logical change per commit.
5. **Open a PR** against `main` with a clear description of the change.

---

## 3. Code Style

### Docstrings

Use **Google-style docstrings** throughout:

```python
def bilateral_filter_3d(volume, sigma_spatial=1.0, sigma_intensity=0.1):
    """Apply 3D bilateral filter to a volumetric image.

    Preserves edges while smoothing homogeneous regions. The filter
    combines spatial proximity and intensity similarity weighting.

    Args:
        volume: 3D array of shape ``(X, Y, Z)``.
        sigma_spatial: Spatial Gaussian kernel width.
        sigma_intensity: Intensity Gaussian kernel width.

    Returns:
        Filtered volume with same shape as input.
    """
```

### Type Annotations

Use `jaxtyping` annotations where applicable:

```python
from jaxtyping import Float, Array

def cov(x: Float[Array, "T N"]) -> Float[Array, "N N"]:
    ...
```

### Factory Pattern (vbjax style)

All learnable/configurable components follow the vbjax factory pattern:

```python
def make_atlas_linear(n_voxels, n_parcels, key):
    """Create a learnable linear atlas.

    Args:
        n_voxels: Number of input voxels.
        n_parcels: Number of output parcels.
        key: JAX PRNG key.

    Returns:
        Tuple of ``(params, forward_fn)`` where ``params`` is a
        NamedTuple and ``forward_fn(params, x)`` applies the atlas.
    """
    params = AtlasParams(weights=jax.random.normal(key, (n_voxels, n_parcels)))
    def forward(params, x):
        return x @ jax.nn.softmax(params.weights, axis=0)
    return params, forward
```

### General Conventions

- **Pure functions**: JIT/vmap/grad compatible, no hidden state.
- **No Equinox/Flax dependency**: Use plain JAX + NamedTuples.
- **Imports**: Absolute imports within jaxoccoli (`from jaxoccoli.covariance import cov`).
- **Module organization**: Each jaxoccoli module is self-contained with
  clear public API at the top of the file.

---

## 4. Testing

We use **pytest** with **hypothesis** for property-based testing.

```bash
# Full suite (221+ tests)
python -m pytest tests/ scripts/tests/ -v

# Single module
python -m pytest tests/test_covariance.py -v

# With coverage
python -m pytest tests/ --cov=jaxoccoli --cov-report=term-missing
```

### Testing Guidelines

- Every public function needs at least one test.
- Use `hypothesis` strategies for numerical properties (symmetry, positive
  definiteness, idempotency).
- Test with small arrays (e.g., 4x4 matrices) so tests run fast on CPU.
- For stochastic functions, fix the JAX PRNG key for reproducibility.
- Tolerance: use `jnp.allclose` with `atol=1e-5` for float32 operations.

---

## 5. Documentation

Documentation is built with Sphinx using the Furo theme.

```bash
# Install doc dependencies
uv pip install -e ".[doc]"

# Build HTML docs
cd docs && make html

# View locally
open _build/html/index.html
```

### Adding Documentation

- **API docs**: Generated automatically via `sphinx-apidoc`. Ensure your
  module has Google-style docstrings.
- **Design docs**: Place in `docs/design/` and add to the toctree in
  `docs/design/index.md`.
- **Math**: Use LaTeX in docstrings with ``:math:\`...\` `` inline or
  `.. math::` blocks. MathJax renders these in the HTML output.

After adding a new jaxoccoli module, regenerate API stubs:

```bash
sphinx-apidoc -o docs/reference jaxoccoli --separate --module-first --force
```

---

## 6. AI-Assisted Development

hippy-feat welcomes AI-assisted contributions. We follow a structured approach
to ensure quality.

### CLAUDE.md

The repository root may contain a `CLAUDE.md` with project-specific context for
Claude Code and similar AI coding assistants. Keep it updated when architecture
changes.

### Agent-Friendly Code

- Write clear docstrings that describe *what* and *why*, not just *how*.
- Use descriptive variable names (e.g., `beta_mean` not `bm`).
- Keep functions short and focused -- easier for both humans and agents to
  understand.
- Include type annotations; they serve as machine-readable documentation.

### Review Checklist for AI-Generated PRs

- [ ] All tests pass (`python -m pytest tests/ scripts/tests/ -v`)
- [ ] Docstrings follow Google style
- [ ] No Equinox/Flax imports introduced
- [ ] Factory functions return `(params, forward_fn)` tuples
- [ ] Numerical operations are JIT-compatible (no Python-level control flow
  on traced values)
- [ ] New modules have corresponding test files
- [ ] `sphinx-apidoc` regenerated if new modules added

### Attribution

AI-assisted commits should include a `Co-Authored-By` trailer:

```
Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
```

---

## 7. Scientific Standards

### Reproducibility

- Pin random seeds via explicit `jax.random.PRNGKey` passing.
- Document all hyperparameters with their default values and justification.
- Include references to the papers/methods implemented (BibTeX in
  `docs/references.bib`).

### Numerical Correctness

- Validate against reference implementations (e.g., nilearn for covariance,
  scipy for matrix operations).
- Property-based tests for mathematical invariants (symmetry of correlation
  matrices, SPD constraints, partition of unity for atlas weights).
- Test edge cases: single timepoint, single voxel, rank-deficient matrices.

### Domain-Specific Conventions

- **fMRI volumes**: Shape convention is `(X, Y, Z)` for 3D, `(T, V)` for
  timeseries where `T` = timepoints, `V` = voxels.
- **Connectivity matrices**: Always `(N, N)` symmetric, with `N` = number
  of parcels/ROIs.
- **Variance propagation**: Functions that accept uncertainty should take
  `(mean, var)` tuples and return `(mean, var)` tuples.
- **Units**: TR in seconds, frequencies in Hz, spatial coordinates in mm.

### Key References

- Ciric et al. (2022) -- Differentiable programming for functional connectomics
- Eklund et al. (2014) -- BROCCOLI: GPU-accelerated Bayesian fMRI
- Rissman et al. (2004) -- Beta series correlations
- Mumford et al. (2012) -- Deconvolving BOLD activation

---

## Questions?

Open an issue on [GitHub](https://github.com/m9h/hippy-feat/issues) or reach
out to Morgan Hough (@m9h).
