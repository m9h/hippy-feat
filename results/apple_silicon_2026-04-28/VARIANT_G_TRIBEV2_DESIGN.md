# Variant G + TRIBEv2-prior — Design document

> **Note 2026-05-03.** The "fold-10" reference below is the *deployment*
> ckpt (97.2% 2-AFC operating point), not the paper-Table-1 ckpt. Paper
> Table 1 uses **fold-0**. This design is orthogonal to that distinction —
> the structured-prior framework can wrap either fold. See `PIPELINE.md`
> "2026-05-03 — Major correction" for the fold-0 vs fold-10 details.

**Status**: design only. No experiments run yet. recon-all on sub-005 launched
2026-05-02 to make this approach feasible (its 3-6h runtime is the gating dependency).

**Author context**: closing out the realtime-mindeye Mac-side reproduction
(commits b133e69 → e689e63 on `results/apple-silicon-2026-04-28`). The
deployment pipeline is locked at FAST K=7 + HP + erode×1 aCompCor + Glover
+ AR(1) + cumulative z + MindEye2 fold-10. This document scopes a parallel
research direction: replace MindEye2's argmax retrieval with a Bayesian
formulation using TRIBEv2 as the structured prior.

## Why this exists as a separate research thread

MindEye2 produces argmax retrieval — a point estimate per trial with no
native confidence calibration. For closed-loop neurofeedback in the Norman
lab framing, what matters more than peak retrieval accuracy is **calibrated
single-trial uncertainty**: knowing when to trust the readout and when to
withhold feedback.

Variant G (per-voxel Bayesian AR(1) GLM with uninformative prior) gives us
the posterior `(β_mean, β_var)` per trial; in earlier rounds we showed it
delivers selective accuracy of 84-90% at high confidence thresholds even
when point-estimate top-1 sits at 60%. But Variant G's prior is uniform —
it ignores any prior knowledge of what a "plausible" β pattern looks like
for a given candidate image.

TRIBEv2 (facebook/tribev2 — video/audio/text → fsaverage5 BOLD encoder)
is exactly the missing ingredient: a foundation-model prior on the β
pattern expected for any candidate image, transferred from cross-subject
training data.

## Theoretical framework

For each test trial, we want the per-voxel β posterior given:
- the observed per-trial BOLD timeseries `y`,
- TRIBEv2's predicted β pattern for a candidate image `c` as a structured
  Gaussian prior.

Per-voxel Bayesian update (Variant G machinery, extended with structured
prior mean):

```
prior:        β | c  ~ N(μ_TRIBE(c), τ²)         ← TRIBEv2 prediction for c
likelihood:   y | β  ~ N(Xβ, σ²V)                 ← AR(1) prewhitened LSS
posterior:    β | y, c ~ N(μ_post(c), Σ_post(c))  ← closed-form Gaussian
```

For retrieval, this becomes Bayesian model selection over the 50 candidates:

```
P(image = c | y) ∝ ∫ P(y | β, X) · P(β | c) dβ
                = marginal likelihood under c's prior
```

The candidate `c*` with maximum marginal likelihood is the decode. The
framework gives us:
- **Posterior over β** per (trial, candidate) — calibrated uncertainty
- **Evidence ratio** between top-1 and runner-up — natural confidence metric
- **Selective accuracy** — accept the decode only when evidence ratio > threshold

This is what MindEye2 doesn't give natively.

## Required components

| Component | Status | Cost |
|---|---|---|
| TRIBEv2 inference: 50 stimuli → fsaverage5 BOLD predictions | not done | ~30 min on MPS once model loads |
| sub-005 FreeSurfer surfaces (recon-all output) | **launched 2026-05-02** | 3-6 hours on M5 Max |
| Morph fsaverage5 → sub-005 native surface | needs recon-all | ~10 min after recon-all |
| Surface → volume projection into the 2792-voxel relmask frame | needs recon-all | ~10 min |
| Variant G GLM with structured (non-uninformative) prior | **NEEDS NEW IMPLEMENTATION** | extend `_variant_g_forward` in `rt_glm_variants.py` to accept `prior_mean` parameter |
| Per-candidate marginal likelihood computation | new | ~15 LOC, closed-form for Gaussian conjugacy |
| Selective-accuracy evaluation harness | partial (have it for MindEye2) | adapt existing `score_unified_metrics.py` |

**Critical bottleneck**: recon-all on sub-005's T1. Without it, no sensible
fsaverage5 → native projection. **There is no MNI/FLIRT-volumetric
shortcut**: fsaverage5 → sub-005 is a surface-to-surface morph problem,
not a volumetric registration problem. Going through MNI volumetrically
would mash subcortical/cortical voxels and discard the surface topology
TRIBEv2's predictions live on.

### recon-all on Apple Silicon — known failures

`mri_synthstrip` and the related deep-learning segmentation steps OOM-crash
inside the FreeSurfer-arm container even with 32G allocated. The stock
`~/.local/bin/freesurfer` wrapper supports `FS_RECON_ALL_PATCH` to bind-mount
a patched recon-all that bypasses synthstrip. The patch at
`~/Workspace/dlbs/freesurfer-patch/recon-all` already had:

- Synthstrip workaround: replace mri_synthstrip with `mri_convert` of a
  pre-computed FSL-bet brain at `/data/derivatives/brainmask/<sub>/<sub>_brain.nii.gz`
- SynthSeg disabled: `if(0) then  # OOM workaround`

But that wasn't sufficient — recon-all 8.2.0 has FOUR additional deep-learning
segmentation steps that ALSO OOM on Mac:

1. **MCADura segmentation** (`mri_mcadura_seg` at line 2222) — calls synthstrip
   internally on `nu.mgz`. OOMs.
2. **VSinus segmentation** (`mri_vsinus_seg` at line 2243) — needs `synthseg.rca.mgz`
   which doesn't exist (synthseg disabled). Hard fail with "cannot find ..."
3. **EntoWM segmentation** (`mri_entowm_seg`/`mri_sclimbic_seg` at line 2865) — TF model OOMs.
4. (Possibly more in autorecon3 surface stages — not yet hit at the time of writing.)

We patched the local `recon-all` to disable all three:

```
2222: if(0) then # PATCHED: skip MCADura (synthstrip OOM on Mac)
2243: if(0) then # PATCHED: skip VSinus (depends on disabled synthseg)
2865: if(0) then # PATCHED: skip EntoWM (DL OOM)
3181: if(0) then # PATCHED: skip MCADura masking
3187: if(0) then # PATCHED: skip VSinus masking
3229: if(0) then # PATCHED: skip EntoWM downstream
3394: if(0) then # PATCHED: skip EntoWM downstream
```

These steps are refinement masks that improve surface quality at dural
attachments / venous sinuses / entorhinal WM. Skipping them produces
slightly less clean surfaces at those locations but adequate for our
purpose (fsaverage5 → native voxel projection of TRIBEv2 outputs).

Currently running v5 at 32G memory with these disables. Past the
failure points of v3 (synthseg.rca.mgz consumer) and v4 (mri_entowm_seg).

## Implementation plan

### Phase 1: prepare priors (post-recon-all)

1. Run TRIBEv2 inference on the 50 special515 images (image input → fsaverage5 BOLD).
2. For each image, project fsaverage5 vertex predictions → sub-005 native cortical
   surface via the recon-all-derived morphs.
3. Project surface predictions → volume (vertices to voxels) via FreeSurfer's
   `mri_surf2vol` or equivalent.
4. Mask to the 2792 relmask voxels. Result: a `(50, 2792)` matrix of per-image
   prior-mean β patterns `μ_TRIBE`.

### Phase 2: extend Variant G to accept structured prior

Modify `rt_glm_variants.py:_variant_g_forward` to accept `prior_mean`
parameter (currently only supports `prior_var` for the uninformative case).
The conjugate-Gaussian update is straightforward:

```python
def variant_g_with_prior(X, y, prior_mean, prior_var, sigma2):
    # prior:      β ~ N(prior_mean, diag(prior_var))
    # likelihood: y ~ N(Xβ, sigma2 * V)            ← V is AR(1) cov (prewhitened)
    # posterior:  β ~ N(post_mean, post_var)
    XtX = X.T @ X / sigma2
    Xty = X.T @ y / sigma2
    Lambda_post = XtX + np.diag(1.0 / prior_var)
    Sigma_post = np.linalg.inv(Lambda_post)
    post_mean = Sigma_post @ (Xty + prior_mean / prior_var)
    return post_mean, Sigma_post
```

Per-voxel implementation closes-form-ly via vectorization. Existing JAX
machinery handles the per-voxel AR(1) prewhitening before this step.

### Phase 3: per-candidate marginal likelihood

For each candidate image c, compute:

```python
# Marginal likelihood of y under prior μ_TRIBE(c):
log_evidence(c) = -0.5 * y' (X * diag(prior_var) * X' + sigma2 * V)^-1 y
                  - 0.5 * log|X * diag(prior_var) * X' + sigma2 * V|
                  + constants
```

Closed-form via Sherman-Morrison-Woodbury for efficiency. Done per-voxel
then summed in log-space.

Pick `c* = argmax log_evidence(c)`. Confidence = `log_evidence(c*) - log_evidence(c_2)`.

### Phase 4: evaluation

Compute on the same 50-image test set:

| Metric | What it measures |
|---|---|
| Top-1 retrieval (50-way) | Equivalent of argmax MindEye2 — for comparison |
| Top-5 retrieval (50-way) | Calibration of the top-5 candidates |
| 2-AFC pairwise | Pairwise discriminability |
| **Selective accuracy at confidence τ** | Accuracy on top-X% confident trials — **the headline metric** |
| **Brier score & ECE** | Calibration quality — explicit in this framework |
| **Coverage of 95% credible interval** | First principled posterior coverage measure |

The interesting comparison isn't "Variant G + TRIBEv2 vs MindEye2 on top-1"
(MindEye2 will win point estimates). It's "Variant G + TRIBEv2 vs MindEye2
on selective-accuracy at τ=0.9 and on calibration".

## Three actual research questions this would answer

### Q1: Does TRIBEv2's cross-subject prior carry useful information for sub-005?

Compare:
- Variant G with TRIBEv2 prior `μ_TRIBE(c)`
- Variant G with subject-mean prior `μ_subject = mean(β across training trials)`
- Variant G with uninformative prior (current baseline)

If TRIBEv2 wins, foundation-model priors transfer across subjects. If
subject-mean wins, priors need subject-specific calibration first.

### Q2: Does the structured prior help the selective-accuracy frontier?

Plot accuracy as a function of confidence-rejection rate. Variant G with
TRIBEv2 prior should dominate at the high-confidence end if the prior
contains useful information. If the curves are identical to Variant G with
flat prior, TRIBEv2 isn't doing real work.

### Q3: Does the prior τ² need to be voxel-specific?

The simplest version uses a global scalar τ². A more sophisticated version
uses voxel-specific τ² tuned via leave-one-image-out CV.

If global τ² works, the framework is simple and deployable. If voxel-specific
is needed, we're back to GLMsingle-style joint CV — and the win over MindEye2
is fragile to prior tuning.

## Compute budget

| Task | Time | Status |
|---|---|---|
| recon-all sub-005 | 3-6 hours | **running 2026-05-02** |
| TRIBEv2 inference on 50 images | 30 min | pending |
| Morph fsaverage5 → native | 10 min | pending |
| Variant G extension implementation | 1 hour | pending |
| Per-candidate marginal likelihood (50 candidates × 770 trials) | 1-2 hours on CPU | pending |
| Evaluation + selective accuracy curves | 30 min | pending |

**Total: ~1 day after recon-all completes.**

## Connection to the locked deployment recipe

This is **not** a replacement for the FAST K=7 + HP + erode×1 + MindEye2
champion pipeline. That recipe is locked for the published-paradigm
50-way retrieval task.

This **is** an extension for closed-loop neurofeedback where calibrated
confidence matters more than peak retrieval. Specifically:

- Norman-lab paradigms with selective feedback gating ("only update display
  when readout confidence > threshold")
- Mixed-target paradigms (decode-on-demand) where the target image isn't
  fixed at scan start
- Closed-loop adaptive stimulus paradigms where each upcoming stimulus
  depends on the previous trial's high-confidence decode

For these use cases, Variant G + TRIBEv2-prior plausibly produces a more
useful single-trial readout than MindEye2's argmax even at lower top-1
accuracy.

## Open methodological questions

1. **TRIBEv2 calibration**: TRIBEv2 was trained on a different cohort and
   different stimuli. Its predictions for our 50 special515 images need
   calibration (intercept + scale at minimum). Where to learn this calibration:
   the 543 ses-01 fine-tuning trials? Or pretrained NSD?

2. **Prior covariance structure**: simplest is `prior_var = τ² * I`. More
   realistic is `prior_var = τ² * Σ_subject` where `Σ_subject` is the
   per-voxel β variance from training. Even more realistic uses TRIBEv2's
   own predicted variance (if available — would need to inspect API).

3. **Joint vs per-trial inference**: in the streaming/RT case, can we update
   the posterior incrementally (sequential Bayes, EKF-style) rather than
   re-doing per-trial closed-form? See the `streaming_kalman_ar1` code in
   `prereg_variant_sweep.py:170-180` for a starting point.

4. **Evidence-ratio threshold**: how to set the confidence threshold τ for
   selective-accuracy gating? Cross-validation on training data. Plot the
   accuracy-coverage curve.

## Files

| Path | What |
|---|---|
| `~/Workspace/data/rtmindeye_paper/rt3t/data/freesurfer/sub-005/` | recon-all output (in progress) |
| `~/Workspace/hippy-feat/scripts/rt_glm_variants.py:_variant_g_forward` | Variant G JAX implementation to extend |
| `~/Workspace/hippy-feat/scripts/prereg_variant_sweep.py:run_glm_cell` | Existing Variant G cell driver |
| `~/Workspace/hippy-feat/results/apple_silicon_2026-04-28/drivers/score_unified_metrics.py` | Selective-accuracy harness to extend |
| (TBD) `drivers/run_variant_g_tribev2_prior.py` | New driver |
| (TBD) `drivers/score_variant_g_selective.py` | Specialized selective-accuracy scorer |
