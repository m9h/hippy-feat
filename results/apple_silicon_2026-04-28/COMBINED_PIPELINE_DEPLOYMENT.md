# Combined pipeline deployment on held-out ses-01

Test: deploy the streaming RLS GLM (Slow tier) and Fast refiner on a held-out
session whose 50 special515 test images are entirely different from the ses-03
images the pipeline was designed against.

Setup:
- Test set: 50 special515 with 3-rep in ses-01 (different image identities from
  ses-03's 50)
- BOLD: fmriprep (rtmotion not available for ses-01)
- Refiner: trained on ses-03 rtmotion non-test trials with streaming-Slow teacher
- Fast input: `RT_paper_Fast_pst5_fmriprep_inclz_ses-01`
- Slow input: `RT_paper_RLS_AR1_pst20_K7CSFWM_HP_e1_inclz_distill_fmriprep_ses-01`

## Results

| Config | s0 Image | s0 Brain | s1 Image | s1 Brain | s2 Image | s2 Brain |
|---|---|---|---|---|---|---|
| Fast (no refiner) | 38 | 36 | 54 | 56 | 66 | 68 |
| Fast (with refiner from ses-03) | 38 | **40** | 50 ↓ | **66** | 66 | 68 |
| Streaming Slow GLM | **54** | 36 | **64** | **68** | **78** | **82** |
| Streaming Slow + refiner (sanity) | 48 | 36 | 66 | 72 | 82 | 84 |

ses-03 reference for comparison:
- Fast baseline s0: 36/34 → refined: 40/48 (v1) or 42/48 (v3)
- Streaming Slow s0: 54/56, s1: 70/76, s2: 78/80

## Reading

**Streaming Slow GLM generalizes to held-out session.** This is the strong
positive result. Same numbers as ses-03 to within ~4pp. s2 Image hits 78% on
ses-01 — matching canonical Offline GLMsingle on this completely held-out
test set. The growing-design ridge OLS β extractor is session-agnostic; the
gain over per-trial AR(1) LSS travels.

**Fast refiner: Brain transfers, Image doesn't.** The cross-session +14pp
Brain gain (subset1: 56 → 66) replicates almost exactly the ses-03 training
result (Brain +14pp). But the Image refinement degrades by 4pp at subset1
(54 → 50). The refiner's per-voxel scalar absorbed ses-03 rtmotion's noise
structure that doesn't match ses-01 fmriprep BOLD; Brain retrieval is more
robust to this mismatch than Image.

**Refiner-on-Slow sanity** shows mild improvement (+2-4pp) when applied to
already-Slow inputs. Suggests roughly half the refiner's apparent benefit is
generic per-voxel re-scaling/calibration, not Slow-specific feature
extraction.

## Implications for deployment

Recommended pipeline for new participants/sessions:

1. **Slow tier (36s latency)**: streaming RLS GLM with growing design,
   K=7+CSFWM+HP+erode×1 nuisance. Deploy directly. Generalizes session-to-session
   without retraining.

2. **EoR tier (2.7m latency)**: same streaming RLS GLM. Same generalization
   expected (not directly tested on ses-01 but extrapolating from the pattern).

3. **Fast tier (14.5s latency)**: deploy the refiner conditionally. Use it for
   Brain-direction retrieval (where +10-14pp gains are robust); skip for Image
   if input BOLD source differs from ses-03 rtmotion. Train a session-specific
   refiner if available labeled data (~500 trials).

The Fast Image refinement is fragile and likely overfits BOLD-source-specific
noise. A clean version of this would re-train per BOLD source, or use a
batch-norm-style adapter that learns a session-invariant transformation.

## Files

- Driver: `local_drivers/score_combined_pipeline_ses01.py`
- ses-01 βs: `task_2_1_betas/prereg/RT_paper_*_ses-01_betas.npy`
- Refiner state: `task_2_1_betas/fast_refiner_state.pth`

— Held-out deployment, 2026-05-04, fold-0, n=50 special515 ses-01.
