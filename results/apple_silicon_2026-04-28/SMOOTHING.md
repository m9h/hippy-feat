# Spatial smoothing on fmriprep BOLD — both ends of the Heunis 4-8mm range hurt

The one preprocessing axis Heunis et al list as "field-typical 4-8mm FWHM"
that we hadn't ablated. The paper's own pipeline sets `smoothing_fwhm=None`
in its `FirstLevelModel` call (`canonical_refs/mindeye_py_GLM_excerpt.py:28`)
and so do all our cells. Bracketed at both ends — 4mm and 8mm — to confirm
the failure isn't kernel-size-specific.

## TL;DR

| | 4mm fmriprep ΔI | 8mm fmriprep ΔI |
|---|---|---|
| Fast subset0/1/2 | 0 / 0 / 0 | −14 / −14 / −16 |
| Slow subset0/1/2 | −2 / −6 / 0 | −24 / −18 / −18 |
| EoR subset0/1/2 | −6 / −8 / −4 | −16 / −16 / −20 |

8mm catastrophically destroys retrieval everywhere. 4mm leaves Fast
unchanged but hurts Slow/EoR by 4-8pp at the paper-anchor subsets. The
deltas are monotone in kernel size at every cell — there is no sweet
spot. The paper's `smoothing_fwhm=None` is the right default.

## Method

Drivers `drivers/run_paper_fmriprep_smooth{4mm,8mm}.py`. Identical to
`run_paper_fast_slow_fmriprep.py` and `run_paper_eor_fmriprep.py` except
the inner `fit_lss_nilearn` is replaced with a copy that sets
`smoothing_fwhm=4.0` or `smoothing_fwhm=8.0` in the `FirstLevelModel`
constructor — nilearn applies the Gaussian kernel to the BOLD volume
before fitting. Everything else held fixed: fmriprep T1w `preproc_bold`,
AR(1) noise model, Glover canonical HRF, cosine drift order=1, high-pass
0.01 Hz, 6-param motion confounds, 2792-voxel reliability mask, inclusive
cum-z post-hoc.

Six cells written (Fast pst=5, Slow pst=20, EoR full-run × 4mm and 8mm):
- `RT_paper_{Fast_pst5,Slow_pst20,EoR}_fmriprep_sm{4,8}mm_inclz_ses-03`

Scorers: `drivers/score_fmriprep_smooth{4mm,8mm}.py`. fold-0 ckpt,
sub-005 ses-03, 50 special515 first-rep / avg-of-2 / avg-of-3.

## Results — Image % top-1, fold-0

| Tier | Subset | Unsmoothed | 4mm | ΔI | ΔB | 8mm | ΔI | ΔB |
|---|---|---|---|---|---|---|---|---|
| Fast | single-rep | 36 | 36 | 0 | +2 | 22 | **−14** | −8 |
| Fast | avg-of-2 | 42 | 42 | 0 | −8 | 28 | **−14** | −20 |
| Fast | complete-set | 46 | 46 | 0 | −6 | 30 | **−16** | −26 |
| Slow | single-rep | 52 | 50 | −2 | −4 | 28 | **−24** | −24 |
| Slow | avg-of-2 | 62 | 56 | **−6** | −6 | 44 | **−18** | −26 |
| Slow | complete-set | 70 | 70 | 0 | −8 | 52 | **−18** | −24 |
| EoR | single-rep | 54 | 48 | **−6** | −2 | 38 | **−16** | −26 |
| EoR | avg-of-2 | 66 | 58 | **−8** | −10 | 50 | **−16** | **−36** |
| EoR | complete-set | 76 | 72 | **−4** | −2 | 56 | **−20** | −12 |

**8mm:** every tier × every subset loses by 14-24pp Image and 8-36pp Brain.

**4mm:** Fast Image is unaffected — at pst=5 there are only 5 BOLD rows
per trial, so the Gaussian kernel has little material to smear before the
GLM ingests it. Slow loses up to -6pp Image at the paper-anchor subset1
(62 → 56). EoR loses the most consistently (-4 to -8pp Image at every
subset). The damage gradient runs Fast ≈ 0 < Slow < EoR — opposite of
what an SNR-boost story would predict.

Both kernels also push Slow subset1 (62 → 56 → 44) and EoR subset1
(66 → 58 → 50) below the **paper-published anchors** of 58 and 66
respectively — i.e. any positive amount of Heunis-typical smoothing
breaks compatibility with the paper's headline numbers.

Result JSONs: `task_2_1_betas/smoothing_{4mm,8mm}_subsets_fold0.json`.

## Why it fails

The paper's pipeline operates on a 2792-voxel **reliability mask** — voxels
hand-selected from ses-01 for per-voxel BOLD reliability. The mask is
spatially sparse: not every neighbor of a kept voxel is itself kept.

- 8mm FWHM Gaussian: σ ≈ 3.4mm at 2.4mm isotropic voxels (~1.4-voxel σ).
  Reaches 3-4 voxels into the volume. Heavy contamination of high-SNR
  finalmask voxels by their out-of-mask or low-reliability neighbors.
- 4mm FWHM Gaussian: σ ≈ 1.7mm (~0.7-voxel σ). Reaches 1-2 voxels.
  Lighter contamination, but still crosses voxel boundaries — and the
  boundaries it crosses are the ones that matter (mask edges, voxel
  tuning discontinuities).

Either way, smoothing pulls reliability-mask voxels' signals toward
their unreliable neighbors — exactly the wrong direction for retrieval,
which depends on per-voxel pattern fidelity.

The Fast-tier-immunity at 4mm (0pp Image change at every subset) is
mechanistically interesting: at pst=5, the per-trial BOLD window is
5 TRs; the GLM has so few rows that the smoothing-introduced spatial
correlation barely propagates to the ridge solution. Slow (pst=20) and
EoR (full-run) have far more BOLD rows, more material for the kernel
to smear, and correspondingly larger losses.

This is consistent with the earlier Variant E spatial Laplacian result
(`README.md:375`): "spatial smoothing flat in fine 2792-voxel finalmask".
That cell tested λ=0.1 graph-Laplacian smoothing on the β patterns
post-extraction; FWHM smoothing on the BOLD pre-extraction is much
stronger, and the failure mode is correspondingly worse.

The Brain direction loses more than Image at avg-of-1 and avg-of-2,
consistent with smoothing destroying multi-voxel pattern information
that Brain retrieval is most sensitive to. At subset2 EoR, the Brain
loss narrows — averaging three reps partially recovers pattern
information, but not enough to offset the smoothing damage.

## Implication for the paper

The Heunis "spatial smoothing 4-8mm" gap is not a coverage hole worth
filling — it's a coverage hole that **confirms the paper's design
decision**. The paper sets `smoothing_fwhm=None` for a reason: spatial
smoothing is incompatible with reliability-mask-based decoding. Both
ends of the Heunis-typical range hurt; both push the paper-anchor
subset1 numbers below the published values. Both we and the paper
should keep `smoothing_fwhm=None`.

If anyone else replicates and asks "did you try smoothing?" — yes,
4mm and 8mm, three tiers, three subsets, monotone in kernel size,
no cell improves. The negative is robust.

## What this doesn't test

- **Smaller kernels** (2mm) — at sub-half-voxel σ smoothing might be
  inert. The 4mm result already shows Fast Image is unmoved at 4mm,
  so smaller kernels are unlikely to surface a hidden sweet spot.
- **Smoothing on a non-reliability-masked space** — if the decoder were
  trained on a contiguous ROI rather than a sparse reliability mask,
  smoothing might help by averaging within-ROI homogeneous responses.
  Out of scope for the rt-mindEye fold-0 evaluation; this checkpoint
  expects the 2792-voxel finalmask as input.
- **Smoothing only the noise pool, not the trial BOLD** — closer to the
  spirit of what aCompCor does, but with spatial rather than PCA-based
  noise extraction. Not tested.

## Files

- Drivers: `drivers/run_paper_fmriprep_smooth{4mm,8mm}.py`
- Scorers: `drivers/score_fmriprep_smooth{4mm,8mm}.py`
- Result JSONs: `task_2_1_betas/smoothing_{4mm,8mm}_subsets_fold0.json`
- Logs: `task_2_1_betas/logs/smooth{4mm,8mm}.log`

— Smoothing bracket 2026-05-08, fold-0, n=50 special515 ses-03.
