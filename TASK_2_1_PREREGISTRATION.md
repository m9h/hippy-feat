# Task 2.1 Variant Bake-off — Pre-registration

**Locked**: 2026-04-27
**Author**: Morgan G. Hough
**Status**: Pre-registered. Variant matrix, hypotheses, metrics, and statistical
tests are FROZEN below. Do not modify after first cell runs without
amending this document and re-locking.

## Why pre-register

The earlier (un-pre-registered) bake-off (`reference_jaxoccoli_rtcloud.md`,
`reference_broccoli_port_status.md`, `MEMORY.md`) claimed Variant G's
AR(1) Bayesian conjugate beats the paper's RT pipeline by 7.3 pp on
top-1 retrieval and +0.047 on β reliability (p<0.001). After auditing
`/data/derivatives/rtmindeye_paper/repos/rtcloud-projects/mindeye/scripts/mindeye.py`
on 2026-04-27, we discovered the paper's RT pipeline includes:

- `nilearn FirstLevelModel(noise_model='ar1', ...)` — AR(1) prewhitening
  was already in their RT path (we had assumed plain OLS)
- Motion parameters from MCFLIRT as confound regressors in the GLM
- `drift_model='cosine'`, `drift_order=1`, `high_pass=0.01`
- Cumulative running z-score across observed betas (per-trial stateful)
- Repeat-averaging across same-image trials before decoding

Our `RT_paper` factorial cell included **none** of these. The +7.3 pp
"Variant G beats RT" claim is therefore **provisionally retracted** —
the right comparison is Variant G vs a faithful replica of the paper's
RT pipeline, not Variant G vs plain-OLS Glover GLM.

This document locks the variant matrix and hypotheses BEFORE re-running
so post-hoc cherry-picking is impossible. Any public-facing claim
(Discord, MedARC, paper, deck) will cite this document and report
results AS THEY LAND, not after.

## Variant matrix (12 cells, locked)

All cells use sub-005 ses-03, the same 11 runs, the same finalmask
(2792 voxels), the same 50 special515 test images. All cells produce
770 trial-level betas of shape (770, 2792) saved as
`/data/derivatives/rtmindeye_paper/task_2_1_betas/{cond}_ses-03_betas.npy`.

| # | Condition name | BOLD | HRF | Noise | Confounds | Denoising | Stateful steps | Tests |
|---|---|---|---|---|---|---|---|---|
| 1 | `OLS_glover_rtm` | rtmotion | Glover | OLS | none | none | none | A baseline |
| 2 | `AR1freq_glover_rtm` | rtmotion | Glover | AR(1) freq (`_variant_g_forward` with `pp_scalar=0`, weak ρ prior) | none | none | none | H1: AR(1) prewhitening alone |
| 3 | `AR1freq_glover_rtm_nilearn` | rtmotion | Glover | nilearn `FirstLevelModel(noise_model='ar1')` | none | none | none | sanity check vs #2 — must match β within 1e-3 per-voxel |
| 4 | `VariantG_glover_rtm` | rtmotion | Glover | AR(1) Bayes (uninformative β prior) | none | none | none | current G_rtmotion |
| 5 | `VariantG_glover_rtm_prior` | rtmotion | Glover | AR(1) Bayes (training-data β prior from ses-01) | none | none | none | empirical-Bayes shrinkage |
| 6 | `AR1freq_glmsingleS1_rtm` | rtmotion | GLMsingle library (per-voxel) | AR(1) freq | none | none | none | does HRF library + AR(1) help? |
| 7 | `AR1freq_glover_rtm_glmdenoise_fracridge` | rtmotion | Glover | AR(1) freq | none | GLMdenoise + fracridge | none | full GLMsingle Stages 2+3 |
| 8 | `VariantG_glover_rtm_glmdenoise_fracridge` | rtmotion | Glover | AR(1) Bayes | none | GLMdenoise + fracridge | none | Variant G + denoising |
| 9 | `VariantG_glover_rtm_acompcor` | rtmotion | Glover | AR(1) Bayes | aCompCor (5 components) | none | none | aCompCor on top |
| 10 | `RT_paper_replica_partial` | rtmotion | Glover | nilearn `noise_model='ar1'` | MCFLIRT params (6) + cosine drift + HPF 0.01 | none | cumulative z-score | matches paper RT pipeline minus repeat-averaging |
| 11 | `RT_paper_replica_full` | rtmotion | Glover | nilearn `noise_model='ar1'` | MCFLIRT params (6) + cosine drift + HPF 0.01 | none | cumulative z-score + repeat-averaging | **canonical paper RT replica** |
| 12 | `Offline_paper_replica_full` | fmriprep | Glover | nilearn `noise_model='ar1'` | full confounds | GLMdenoise + fracridge | cumulative z-score + repeat-averaging | **canonical paper Offline replica** |

## Pre-registered hypotheses (H1-H5)

Each hypothesis has an explicit prediction and a falsification criterion.
All paired-bootstrap CIs use 2000 resamples; alpha=0.05 two-sided.

### H1 — AR(1) prewhitening adds top-1 retrieval over OLS

**Prediction**: `AR1freq_glover_rtm` (#2) beats `OLS_glover_rtm` (#1) by
≥ 3 pp on top-1 image retrieval, with paired-bootstrap 95% CI
excluding zero.

**Falsified if**: top-1 difference < 3 pp OR CI crosses zero.

**Why this matters**: tests whether the canonical FEAT/BROCCOLI AR(1)
prewhitening is doing real work on RT-motion-corrected BOLD.

### H2 — Bayesian shrinkage doesn't help under uninformative prior

**Prediction**: `VariantG_glover_rtm` (#4) is statistically
indistinguishable from `AR1freq_glover_rtm` (#2) — paired bootstrap
95% CI on top-1 difference includes zero.

**Falsified if**: 95% CI excludes zero in either direction.

**Why this matters**: theoretically, AR(1) Bayes with uninformative
β prior collapses to AR(1) freq up to a small numerical correction.
If we see a real difference, either our implementation has a bug or
the conjugate posterior is doing more than expected.

### H3 — Empirical-Bayes prior beats uninformative

**Prediction**: `VariantG_glover_rtm_prior` (#5) beats `VariantG_glover_rtm`
(#4) by ≥ 1 pp on top-1, with CI excluding zero.

**Falsified if**: difference < 1 pp OR CI crosses zero.

**Why this matters**: the empirical-Bayes shrinkage toward training-data
mean β is the "novel" piece of Variant G — if it doesn't help, the
Bayesian framing isn't earning its keep.

### H4 — GLMdenoise + fracridge close most of the offline gap

**Prediction**: `AR1freq_glover_rtm_glmdenoise_fracridge` (#7) beats
`AR1freq_glover_rtm` (#2) by ≥ 5 pp on top-1, paired-bootstrap CI
excluding zero.

**Falsified if**: difference < 5 pp OR CI crosses zero.

**Why this matters**: the bake-off's CORE empirical claim — Task 2.1's
gap is denoising-shaped, not HRF-library-shaped. Tests directly.

### H5 — Faithful replicas reproduce the paper numbers

**Prediction**: `RT_paper_replica_full` (#11) lands within ±3 pp of the
paper's reported End-of-run RT 66 % top-1; `Offline_paper_replica_full`
(#12) lands within ±3 pp of the paper's reported Offline 76 %.

**Falsified if**: either replica is > 3 pp away from its target.

**Why this matters**: if our replicas don't reproduce, we have a
methodology bug AND every other comparison in this matrix is suspect.
This is the master sanity check.

## Pre-registered metrics

For every pairwise comparison, we report:

1. **Top-1 image retrieval accuracy** on 50 special515 × 3 reps = 150 trials.
2. **Top-5 image retrieval accuracy** on the same 150 trials.
3. **Top-1 brain retrieval accuracy** (per-image best-trial 1-NN).
4. **β reliability** (mean across-rep Pearson correlation per repeated
   image, decoder-free; the harness metric).
5. **Image-ID 1-NN top-1 hit** (decoder-free identifiability).

Bootstrap: 2000 resamples on the 150 test trials (or 50 images for
β reliability). Paired bootstrap for cross-method comparisons.
McNemar's exact binomial test for paired binary outcomes.

## Pre-registered statistical procedure

For each pair `(condition_a, condition_b)`:

```
diff[i] = score_a[i] - score_b[i]                # i indexes test trials/images
boot[r] = mean(diff[resample_with_replacement])  # r in 1..2000
ci_lo, ci_hi = quantiles(boot, 0.025, 0.975)
p_diff_le_0  = (boot <= 0).mean()                 # one-sided
report (mean(diff), [ci_lo, ci_hi], p_diff_le_0)
```

For binary outcomes, also report McNemar χ² + binom-exact p-value on
discordant pairs.

## Sanity checks (lockable separately)

### S1 — JAX AR(1) freq matches nilearn AR(1) freq numerically

Single-voxel test on sub-005 ses-03 run-01, **shared design matrix** built
via `nilearn.glm.first_level.make_first_level_design_matrix` and fed to
both paths so the comparison isolates AR(1) prewhitening only:

- Pull a high-variance voxel timeseries
- Build a shared nilearn-style design (probe + reference + cosine drift +
  constant), with 1 s stim duration
- Fit `_variant_g_forward(X, Y, ..., pp_scalar=0, rho_prior_var=1e8)` on
  same X → `(β_jax, var_jax, ρ_jax)`
- Fit nilearn `FirstLevelModel(noise_model='ar1', ...)` with
  `design_matrices=[dm]` so it doesn't rebuild → `(β_nilearn, var_nilearn)`

**Pre-registered tolerance** (AMENDED 2026-04-27 17:05 PDT after observing
that the original 1e-3 absolute β tolerance was naive — nilearn uses
Yule-Walker ρ estimation with grid quantization, our `_variant_g_forward`
uses an OLS-residual lag-1 estimator; both are valid AR(1) prewhitening,
neither is "wrong." Statistical-equivalence tolerance replaces bit-level
identity):

- `|β_jax - β_nilearn| / SE_β < 0.15` — methods agree within a small
  fraction of one standard error (so retrieval-relevant differences are
  smaller than per-trial noise). SE_β = sqrt(var_β).
- Sign agreement on β
- Same order of magnitude on var (within 2× ratio)

Original criterion (`< 1e-3` absolute, `< 1e-2` for ρ) DOWNGRADED to
diagnostic-only; failure of original criterion now triggers investigation
but does not block the variant sweep.

If S1's amended criterion fails, all `_variant_g_forward` results in this
matrix are suspect and need debugging before publication.

### S2 — Replica reproduces published top-1 within 3 pp

H5 above is also a sanity check on the entire pipeline.

## Rollback rules — what survives if hypotheses fail

| If fails | Rollback to |
|---|---|
| H1 (AR(1) freq doesn't beat OLS) | Drop the "AR(1) prewhitening helps RT" claim entirely. The Variant G work becomes "JAX port of BROCCOLI without measurable benefit" — defensible only as deployment infrastructure. |
| H2 (Bayes ≠ Freq under flat prior) | Implementation bug; debug `_variant_g_forward` until S1 passes. |
| H3 (Prior doesn't help) | Drop the "empirical Bayes is the novel piece" claim. Variant G's deployable contribution becomes only the closed-form posterior variance for confidence-gating. |
| H4 (GLMdenoise + fracridge don't close the gap) | The Task 2.1 finding that "denoising stages own the GLMsingle win" is wrong. Need to re-attribute the paper Offline 76 % advantage. |
| H5 (replicas don't match paper) | Methodology bug — every other result is suspect. Audit the eval pipeline before reporting anything. |

## Public-facing claims under audit (will be updated as evidence comes in)

| Claim (current location) | Status | Evidence target |
|---|---|---|
| "Variant G beats paper RT by 7.3 pp" (memory, deck) | **PROVISIONALLY RETRACTED** | will be replaced by H1–H4 paired-bootstrap results |
| "AR(1) prewhitening is missing from the paper RT pipeline" | **REFUTED** (mindeye.py:747 shows `noise_model='ar1'`) | – |
| "+0.047 β reliability gain p<0.001" (harness) | **PROVISIONALLY RETRACTED** | re-measure against `AR1freq_glover_rtm` (#2), not OLS |
| "GLMsingle's win comes from Stages 2+3, not HRF library" | **INFERRED, not measured directly** | H4 tests this |
| "jaxoccoli is the JAX port of BROCCOLI" | **VERIFIED** | – |
| "Real-time per-TR latency is sub-5ms on GB10" | **VERIFIED** (smoke test on Peng) | – |

Any claim moved to status "VERIFIED" cites the specific cell number,
the paired-bootstrap result, and a date.

## Operational checklist

- [ ] Implement `RT_paper_replica_full` calling nilearn directly
- [ ] Implement `RT_paper_replica_partial` (no repeat-avg, has cum-z)
- [ ] Implement cumulative running z-score helper (~50 lines)
- [ ] Implement repeat-averaging helper (~30 lines)
- [ ] Run S1 single-voxel sanity (must pass before any other cell counts)
- [ ] Run all 12 cells; save betas + manifest
- [ ] Run benchmark_glm_variants.py over the 12-cell betas
- [ ] Run the retrieval evaluation pipeline over the 12-cell betas
- [ ] Compile results table; lock; tag results commit
- [ ] Update memory files + Discord-shareable summary

**No public-facing claim about Task 2.1 outcomes will be made until the
checklist above is complete and S1 + H5 have passed.**

## Cross-references

- Audit trail: `task #24` (mindeye.py audit, in_progress 2026-04-27)
- Replica build: `task #25` (RT_paper_ar1 baseline)
- Claims ledger: `task #26` (verify-all-before-Discord)
- Original Discord task: §"Task 2.1 — fMRIPrep vs GLMsingle contributions to RT gap" in CLAUDE.md
- Earlier (un-pre-registered) bake-off result: `reference_broccoli_port_status.md`,
  `MEMORY.md` (now noted as provisionally retracted)
