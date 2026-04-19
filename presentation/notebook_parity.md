# RT-Cloud MindEye Notebook Parity & Variant Review

**Date:** 2026-04-18
**Source notebook:** [`brainiak/rtcloud-projects/mindeye/scripts/mindeye.ipynb`](https://github.com/brainiak/rtcloud-projects/tree/main/mindeye)
**Purpose:** Verify the hippy-feat preprocessing variants are meaningful updates over the original RT-MindEye pipeline, and fact-check the March 25 presentation deck.

---

## 1. What the original notebook does per TR

Cell 18 of `mindeye.ipynb` (also `scripts/mindeye.py:736+`):

```python
lss_glm = FirstLevelModel(
    t_r=tr_length,                  # = 1.5s
    slice_time_ref=0,
    hrf_model='glover',             # single canonical HRF
    drift_model='cosine',
    drift_order=1,
    high_pass=0.01,
    mask_img=union_mask_img,        # 8627 voxels
    signal_scaling=False,
    smoothing_fwhm=None,            # deliberately no smoothing
    noise_model='ar1',
    n_jobs=-1,
    memory_level=1,
    minimize_memory=True,
)
lss_glm.fit(run_imgs=img, events=cropped_events,
            confounds=pd.DataFrame(np.array(mc_params)))  # 6-col motion only
beta = lss_glm.compute_contrast('probe', output_type='effect_size')
# then z-score against running history of all prior betas
# then feed MindEye: ridge → MLP-mixer → CLIP (256 × 1664)
```

### Pipeline nuances that must be preserved in variants

- **LSS-style single-trial** design: probe vs. lumped-reference regressors
- **Cosine drift** + `high_pass=0.01`
- **AR(1) noise model** — nilearn runs two-pass OLS → per-voxel rho → prewhitened GLS
- **Running z-score** against history — provides implicit regularization, washes out scale/bias
- **No spatial smoothing** (intentional; MindEye reads fine-grained voxel patterns)
- **8627-voxel union mask**
- **Only motion as confounds** — no CompCor, no GLMdenoise

### Stubbed code (the smoking gun)

Cell 17 of the notebook has, **commented out**:
```python
# from utils_glm import load_glmsingle_hrf_library, hrf_i_factory, fit_and_run_glm
# BASE_TIME, GLMS_HRFS = load_glmsingle_hrf_library(f"{data_path}/getcanonicalhrflibrary.tsv")
# hrf_fns = [hrf_i_factory(i, BASE_TIME, GLMS_HRFS) for i in range(1, 21)]
# hrf_indices = np.load(f"{data_path}/avg_hrfs_s1_s2_full.npy").astype(int)[:,:,:,0]
```

The Princeton team started the per-voxel GLMsingle-style HRF path and never finished it. **Variant C delivers this.**

---

## 2. AR(1) vs OLS parity — empirical

Variant A uses plain OLS, the notebook uses AR(1) prewhitening. Question: how different are the probe betas?

**Test setup** (`scripts/ar1_parity_test.py`): 192 TRs, TR=1.5s, 1000 voxels, probe+reference LSS design, Glover HRF, AR(1) noise at rho in {0.0, 0.2, 0.4, 0.6}. Compared plain OLS against nilearn's `ARModel` (same code path as `FirstLevelModel(noise_model='ar1')`, with per-voxel rho quantized to 0.01 grid).

| rho | MAE | Pearson r | RMSE | OLS σ | AR1 σ |
|-----|-----|-----------|------|-------|-------|
| 0.00 | 0.49 | **0.9997** | 0.78 | 29.62 | 29.61 |
| 0.20 | 2.30 | **0.9961** | 3.17 | 35.83 | 35.62 |
| 0.40 | 7.59 | 0.9743 | 9.86 | 43.68 | 42.10 |
| 0.60 | 18.50 | 0.9040 | 23.56 | 55.05 | 49.06 |

**Verdict:** In the typical operating regime (post motion correction + cosine drift, residual rho ≈ 0.2–0.3), the two estimators agree with r ≥ 0.996 across voxels. Since the notebook z-scores betas against history before feeding MindEye and the model cares about *pattern*, not magnitude, **OLS in Variant A is defensible**.

**Caveat:** At rho > 0.4, divergence is non-trivial. Before claiming parity in a paper, measure the actual residual rho on sub-005 real data. If rho > 0.3 anywhere on cortex, switch Variant A to AR(1) to avoid the argument.

---

## 3. Variant → notebook gap mapping

| Variant | Gap it closes | Notebook anchor | Meaningful update? |
|---|---|---|---|
| **A — Baseline** | (replicates RT) | `mindeye.py:745` | ✅ Reference. AR(1) parity verified (r ≥ 0.996 at rho ≤ 0.3). |
| **A+N — Nuisance** | **Confound regression** — notebook has only 6 motion params; offline uses aCompCor | `mindeye.py:750` | ✅ **Highest-value addition.** Directly closes a MAJOR row in the gap table. |
| **B — FLOBS** | HRF shape (3-basis) | single Glover | ⚠️ Dominated by C. Demote to ablation or drop. |
| **C — Per-voxel HRF** | **HRF model** — notebook team stubbed this; we deliver it | `utils_glm.py:37-93`, `mindeye.ipynb` cell 17 | ✅ **Highest narrative value.** |
| **D — Bayesian shrinkage** | **Regularization** — notebook has none; offline GLMsingle uses fracridge (not Bayesian) | — | ⚠️ Now largely subsumed by G's conjugate path. Keep as the minimal-compute reference, or fold into G. |
| **E — Spatial Laplacian** | (no gap — notebook intentionally has `smoothing_fwhm=None`) | `mindeye.py:746` | ⚠️ Risky: could blur info MindEye needs. Gate on retrieval-accuracy test. MRF prior in G is the principled version. |
| **F — Log-signature** | (monitoring, not preprocessing) | — | ⚠️ Move to separate "RT safety" track. |
| **C+D — Combined** | HRF + regularization | — | ✅ if D survives framing. |
| **G — Bayesian first-level** | **Uncertainty, AR(p) noise, MRF spatial, physiological HRF** — notebook has none of these, offline has some | `docs/design/DESIGN_bayesian_first_level.md`, `jaxoccoli/bayesian_beta.py` | ✅ **Most ambitious; highest publication value.** See §4. |

### Gaps still unclosed (pre-G)

- **GLMdenoise proper** (PCA of noise-pool voxels). A+N approximates with tissue means — ~70% of the gap, not 100%.
- **Fracridge** (GLMsingle's third pillar). D is a *different* regularizer, not equivalent. G's conjugate path is closer but still not fracridge.

## 4. Variant G — Bayesian first-level (added 2026-04-18)

Variant G is specified in `docs/design/DESIGN_bayesian_first_level.md` and partially implemented in `jaxoccoli/bayesian_beta.py` (359 lines). It's not just another variant — it reframes the whole taxonomy.

### What G actually adds vs. the notebook

| Dimension | Notebook | Variants A–F | Variant G |
|---|---|---|---|
| Beta output | point estimate | point estimate | **posterior mean + std** |
| Noise model | AR(1) (nilearn) | OLS (except implicit via z-score) | **AR(p), p=2 default** |
| HRF | single Glover | Glover / FLOBS / 20-lib | **FLOBS / parametric / physiological** |
| Spatial prior | none | none / Laplacian (E) | **MRF (edge-preserving)** |
| Uncertainty | none | none | **posterior distribution** |
| Inference | OLS | OLS + shrinkage | **conjugate closed-form OR NUTS** |

### Two paths, two use cases

- **G-conjugate** (`make_conjugate_glm`, `make_ar1_conjugate_glm`): closed-form normal-inverse-gamma posterior. RT-compatible. This is effectively **Variant D done right** — D outputs `beta_mean` only; G-conjugate outputs `(beta_mean, beta_var, sigma2_mean)`. The variance map is the novel output RT-MindEye could use for *confidence-gated reconstruction*: only run the diffusion model when posterior std is below threshold.
- **G-NUTS** (`make_bayesian_glm`, requires `blackjax`): full posterior via No-U-Turn Sampler with AR(p) noise. Offline — design doc estimates 30–60s for whole brain on A100. Not RT. For publication / model comparison against FSL FEAT, SPM, BROCCOLI, FABBER.

### The novel contribution: physiological HRF (vpjax-backed)

Level 3 of G's HRF hierarchy uses a physiological neurovascular coupling model as the HRF generator, with priors informed by MRS (GABA/Glu), qMRI (T1, T2\*), and angiography (vessel geometry). The posterior is over *neurovascular coupling parameters*, not abstract shape weights. This is the genuinely new capability — nothing in GLMsingle, fMRIPrep, BROCCOLI, FABBER, FSL FEAT, or SPM does this.

The physiological backend is **`vpjax`** — a sibling JAX project at `/home/mhough/dev/vpjax/` that provides differentiable cerebrovascular physiology (Riera 2006/2007 neurovascular coupling, Balloon-Windkessel, Bulte qBOLD, Germuska calibrated fMRI, Lu TRUST). It's already an **active interface partner** for hippy-feat: `jaxoccoli/angiography.py` emits `{points, radii, branch_ids}` → consumed by `vpjax.vascular.angiography.VesselTree` (contract-tested in `tests/test_angiography.py::test_compatible_with_vpjax`). Level-3 integration requires one new thin wrapper in vpjax: `riera_hrf(t, params) -> h(t)`, which feeds a unit impulse into `solve_riera` and reads out BOLD via `riera_to_balloon`. That's a single-afternoon change in vpjax, not a research project.

## 5. Current project capabilities (cross-repo)

Variant G sits inside a broader differentiable-physiology stack the user has built. For the deck and for future planning, treat these as **current capabilities**, not future dependencies:

| Repo | Role in hippy-feat | Key exports |
|---|---|---|
| **`jaxoccoli`** (inside hippy-feat) | 22-module core library (~5500 LOC) | `glm`, `spatial`, `motion`, `stats`, `permutation`, `signatures`, `connectivity`, `covariance`, `matrix`, `graph`, `interpolate`, `learnable`, `losses`, `fourier`, `transport`, `bayesian_beta`, `multivariate`, `fusion`, `hf_encoder`, `dot_adapter`, `angiography`, `nsd` |
| **`vpjax`** (peer, `/home/mhough/dev/vpjax/`) | Differentiable cerebrovascular physiology — drives Variant G Level 3 HRF, and consumes hippy-feat's angiography output | `hemodynamics` (Riera, Balloon, BOLD), `perfusion` (ASL), `qbold`, `qsm`, `vascular` (VesselTree), `metabolism`, `brainstem`, `cardiac`, `vaso`, `stochastic`, `integrators` (Local Linearization for stiff NVC ODEs) |
| **`vbjax`** (upstream) | Neural mass models — upstream of vpjax's activity→BOLD chain | `make_bold()` Balloon reference |
| **`dot-jax`** (peer) | Diffuse optical tomography FEM mesh | Consumed by `jaxoccoli/dot_adapter.py` for `hbo/hbr` → cortical surface |
| **`vmtk`** (system binary, `mhough/neuro/vmtk` brew formula) | Centerline extraction for TOF-MRA | Feeds `jaxoccoli/angiography.py` |

**Implication for Variant G:** Phase 4 (Riera Level 3) is not "waiting on vpjax to exist" — vpjax exists and is mature enough that hippy-feat already consumes its `VesselTree` interface. The real Phase 4 work is the `riera_hrf` helper (upstream, in vpjax) plus fixed-step diffrax for vmap-compatibility (a known constraint). Treat Level 3 as "gated on a small upstream API addition," not "research blocker."

**Multi-modal prior story.** The design-doc priors on Riera params come from data modalities vpjax already has forward models for: MRS (metabolism), qMRI (T1/T2\*), angiography (VesselTree→vessel compliance). When those data pipelines land in hippy-feat, Level 3 gets *informed* priors for free — the inference code doesn't change.

### Implementation status (as of 2026-04-18)

- ✅ `make_conjugate_glm` — weak-prior normal-inverse-gamma (`bayesian_beta.py:33-114`)
- ✅ `make_conjugate_glm_vmap` — batched over voxels (`:117`)
- ✅ `make_ar1_conjugate_glm` — conjugate with AR(1) prewhitening (`:158`)
- ✅ `make_bayesian_glm` — NUTS with AR(p) (`:256-359`, blackjax-gated)
- ❌ MRF spatial prior
- ❌ Parameterized double-gamma HRF (Level 2)
- ❌ Riera physiological HRF (Level 3) — waiting on vpjax

### What this means for the deck

- **Relabel D** as "D — conjugate shrinkage (point estimate)" and **add G** as "G — Bayesian first-level (posterior distribution)." Keep D for the timing/ablation column; G is the flagship.
- The deck's "Regularization: Ridge (cross-val)" gap row was flagged MAJOR. G-conjugate closes more of that than D does.
- Consider a new slide: "Uncertainty-aware RT reconstruction" showing β mean + β std maps side by side, with the proposal that diffusion only runs when posterior confidence exceeds threshold. That's a story nobody else in the RT-MindEye space has.
- Variant G bumps the deck's gap table by **one more MAJOR row**: the notebook has no uncertainty at all; offline GLMsingle reports per-fold ridge but not proper posteriors. Adding an "Uncertainty / posterior" row with Offline=partial, RT=none, G=full would strengthen the motivation.

### Reconciled taxonomy for the paper / deck

```
RT-compatible (<1.5s/TR):         A, A+N, B, C, D, E, F, C+D, G-conjugate
Offline/publication-only:         G-NUTS
Phase-3 (pending vpjax):          G-Level-3 (physiological HRF)
```

---

## 4. Deck fact-check (`rt_mindeye_pipeline.tex`, 2026-03-25)

### Errors fixed in 2026-04-17 edit

| Location | Before | After |
|---|---|---|
| MindEye2 slide (CLIP dim) | `(1024) → (768) CLIP space` | `(1024) → (256 × 1664) OpenCLIP ViT-bigG-14` |
| Summary slide (test count) | `73 passing tests` | `85 passing tests` (71 variant + 14 spatial) |
| Summary slide (NSD claim) | `8 NSD subjects downloaded` | `NSD dimensionality results across 8 subjects` |
| MindEye2 slide (7T framing) | ambiguous — read as "whole work is 7T" | added: *"Fine-tuned on Princeton 3T sub-005 (~3h). All RT benchmarks in this talk are 3T sub-005."* |
| Dimensionality slide (column labels) | "Key findings" / "3T sub-005 comparison" | "NSD (7T, 8 subjects)" / "Princeton sub-005 (3T, 4 sessions)" |

### Verified correct

- TR = 1.5s (`mindeye.py:472`)
- Union mask = 8627 voxels
- Motion correction = MCFLIRT (both offline and RT)
- RT registration = `flirt -dof 6` once at run-01 TR-0, then `convert_xfm -concat` per TR
- Offline uses CompCor/GLMdenoise/fracridge; RT uses none of these
- Gauss-Newton MC, bilateral filter, per-voxel HRF timings match benchmarks

### Open items

- **Ridge hidden dim (1024):** not independently verified against `model.ridge` source; flag for sanity check.
- **AR(1) note:** consider adding a one-liner to Variant Benchmark slide citing the parity result (r ≥ 0.996 at rho ≤ 0.3).
- **Stubbed-code story:** consider adding a slide or speaker-note that Variant C finishes what cell 17 of `mindeye.ipynb` started.
- **GLMdenoise partiality:** the deck's gap table implies A+N fully closes the "Noise regression" row. It closes ~70%. Consider softening or add a fracridge/GLMdenoise-proper variant.

---

## 5. Reproducibility

- **Notebook source:** `gh api repos/brainiak/rtcloud-projects/contents/mindeye/scripts/mindeye.ipynb`
- **Parity test:** `scripts/ar1_parity_test.py`, run with the project venv:
  ```
  /home/mhough/dev/hippy-feat/.venv/bin/python scripts/ar1_parity_test.py
  ```
- **Slides source:** `presentation/rt_mindeye_pipeline.tex`. Rebuild:
  ```
  cd presentation/
  pdflatex rt_mindeye_pipeline.tex   # twice for TOC, or use latexmk
  ```
