# 8mm spatial smoothing on fmriprep BOLD — catastrophic at every tier

The one preprocessing axis Heunis et al list as "field-typical 4-8mm FWHM"
that we hadn't ablated. The paper's own pipeline sets `smoothing_fwhm=None`
in its `FirstLevelModel` call (`canonical_refs/mindeye_py_GLM_excerpt.py:28`)
and so do all our cells. Filling the gap.

## Method

Driver: `drivers/run_paper_fmriprep_smooth8mm.py`. Identical to
`run_paper_fast_slow_fmriprep.py` and `run_paper_eor_fmriprep.py` except
the inner `fit_lss_nilearn` is replaced with a copy that sets
`smoothing_fwhm=8.0` in the `FirstLevelModel` constructor — nilearn applies
the Gaussian kernel to the BOLD volume before fitting. Everything else
held fixed: fmriprep T1w `preproc_bold`, AR(1) noise model, Glover
canonical HRF, cosine drift order=1, high-pass 0.01 Hz, 6-param motion
confounds, 2792-voxel reliability mask, inclusive cum-z post-hoc.

Three cells written:
- `RT_paper_Fast_pst5_fmriprep_sm8mm_inclz_ses-03`
- `RT_paper_Slow_pst20_fmriprep_sm8mm_inclz_ses-03`
- `RT_paper_EoR_fmriprep_sm8mm_inclz_ses-03`

Scorer: `drivers/score_fmriprep_smooth8mm.py`. fold-0 ckpt, sub-005 ses-03
50 special515 first-rep / avg-of-2 / avg-of-3.

## Results — Image % top-1, fold-0

| Tier | Subset | Unsmoothed fmriprep | **8mm fmriprep** | ΔI | ΔB |
|---|---|---|---|---|---|
| Fast | single-rep | 36 | **22** | **−14** | −8 |
| Fast | avg-of-2 | 42 | 28 | **−14** | −20 |
| Fast | complete-set | 46 | 30 | **−16** | −26 |
| Slow | single-rep | 52 | 28 | **−24** | −24 |
| Slow | avg-of-2 | 62 | 44 | **−18** | −26 |
| Slow | complete-set | 70 | 52 | **−18** | −24 |
| EoR | single-rep | 54 | 38 | **−16** | −26 |
| EoR | avg-of-2 | 66 | 50 | **−16** | **−36** |
| EoR | complete-set | 76 | 56 | **−20** | −12 |

Every tier × every subset loses, by 14-24pp Image and 8-36pp Brain. There
is no slice of the factorial where 8mm helps, and the magnitude of the
loss grows with the latency budget — Slow loses more than Fast at
single-rep, and EoR loses the most at subset1 Brain (-36pp).

Result JSON: `task_2_1_betas/smoothing_8mm_subsets_fold0.json`.

## Why it fails

The paper's pipeline operates on a 2792-voxel **reliability mask** — voxels
hand-selected from ses-01 for per-voxel BOLD reliability. The mask is
spatially sparse: not every neighbor of a kept voxel is itself kept. An
8mm FWHM Gaussian (σ ≈ 3.4mm at 2.4mm isotropic voxels = ~1.4 voxel
σ) reaches into 3-4 voxels in each direction. Most of those neighbors
are either outside the reliability mask (zero or low-SNR voxels) or
have entirely different voxel-wise tuning. Smoothing pulls high-SNR
voxels' signals toward their unreliable neighbors — exactly the wrong
direction for retrieval, which depends on per-voxel pattern fidelity.

This is consistent with the earlier Variant E spatial Laplacian result
(`README.md:375`): "spatial smoothing flat in fine 2792-voxel finalmask".
That cell tested λ=0.1 graph-Laplacian smoothing on the β patterns
post-extraction; 8mm FWHM smoothing on the BOLD pre-extraction is much
stronger, and the failure mode is correspondingly worse.

The Brain direction loses more than Image at avg-of-1 and avg-of-2,
consistent with smoothing destroying multi-voxel pattern information
that Brain retrieval is most sensitive to. At subset2 EoR, the Brain
loss narrows to -12pp — averaging three reps partially recovers
pattern information, but not enough to offset the smoothing damage.

## Implication for the paper

The Heunis "spatial smoothing 4-8mm" gap is not a coverage hole worth
filling — it's a coverage hole that **confirms the paper's design
decision**. The paper sets `smoothing_fwhm=None` for a reason:
spatial smoothing is incompatible with reliability-mask-based decoding.
Both we and the paper should keep it off.

If anyone else replicates and asks "did you try smoothing?" — yes, 8mm
FWHM, three tiers, three subsets, every cell loses. The negative is
robust.

## What this doesn't test

- **Smaller kernels** (2mm, 4mm) — at half-voxel σ smoothing might be
  too gentle to damage pattern info. We didn't test; bracketing this
  would cost another ~3hr of nilearn fits.
- **Smoothing on a non-reliability-masked space** — if the decoder were
  trained on a contiguous ROI rather than a sparse reliability mask,
  smoothing might help by averaging within-ROI homogeneous responses.
  Out of scope for the rt-mindEye fold-0 evaluation; this checkpoint
  expects the 2792-voxel finalmask as input.
- **Smoothing only the noise pool, not the trial BOLD** — closer to the
  spirit of what aCompCor does, but with spatial rather than PCA-based
  noise extraction. Not tested.

## Files

- Driver: `drivers/run_paper_fmriprep_smooth8mm.py`
- Scorer: `drivers/score_fmriprep_smooth8mm.py`
- Result JSON: `task_2_1_betas/smoothing_8mm_subsets_fold0.json`
- Log: `task_2_1_betas/logs/smooth8mm.log`

— 8mm smoothing ablation 2026-05-08, fold-0, n=50 special515 ses-03.
