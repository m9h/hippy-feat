# Neurofeedback metrics — what to optimize for closed-loop fMRI (Norman lab framing)

> **Partial deprecation 2026-05-03.** Methodology framing in this doc is
> sound. Empirical numbers cited as "fold-10, single-rep filter" should
> be interpreted as one specific operating point, NOT the paper-faithful
> baseline. Paper Table 1 uses **fold-0** + `filter_and_average_repeats`
> (avg-of-3-reps); see `PIPELINE.md` "2026-05-03 — Major correction".
> The closed-loop deployment champion (97.2% 2-AFC, fold-10 + first-rep)
> remains the right operating point for actual neurofeedback — different
> goal than reproducing Table 1.


The paper's headline metrics (50-way top-1 retrieval, multi-metric averaged
reconstruction quality) are **research metrics**, not deployment metrics. For
closed-loop fMRI as practiced by the Norman lab and other RT-fMRI groups, a
different metric stack matters.

## Why 50-way top-1 is the wrong target for closed-loop

A closed-loop neurofeedback experiment makes a *binary* or *small-pool*
decision per trial: "is the current brain state on-target or off-target?",
"which of these N candidate stimuli is the participant attending to?", or
"how confident am I in my current readout, should I update the feedback
display?". The question is rarely "out of 50 candidate images, which one is
the participant seeing" — that's a cognitive-decoding evaluation framing.

Concretely:
- **DeBettencourt et al. 2015** (closed-loop attention training) decoded 1
  bit per TR (face vs. scene attention).
- **Mennen et al. 2019** (RT-fMRI review) describes typical neurofeedback
  protocols as 2-class decoding with a confidence-gated update rule.
- **deBettencourt, Norman, Turk-Browne, Wallace 2015** Nature Neuroscience
  used MVPA logistic-regression classifier accuracy on a 2-class problem
  (face vs scene), reporting fwd-acc, false-positive rate, and detection
  latency.

For Norman-lab-style closed-loop, **2-AFC discriminability + calibrated
confidence + selective accuracy** are the relevant metrics, not 50-way
retrieval.

## The closed-loop metric stack

For each candidate processing pipeline, compute:

| Metric | What it measures | Why it matters for closed-loop |
|---|---|---|
| **2-AFC pairwise accuracy** | For each pair (i, j), does query i score higher on candidate i than j? | Direct analog of "which of two candidate stimuli is currently being attended" — the actual deployment decision |
| **Merge/separate ROC-AUC** | Distribution of same-image vs different-image cosine distances | Single-trial discriminability; the headline number for whether closed-loop is feasible at all |
| **Cohen's d (merge/separate)** | Effect size of same vs different distance distributions | Threshold-independent magnitude of the discriminability signal |
| **Brier score** | Mean squared difference between predicted probability and outcome | Calibration: are the model's confidence estimates trustworthy? |
| **Expected Calibration Error (ECE)** | Average gap between confidence and accuracy in confidence bins | Calibration without making distributional assumptions |
| **Selective accuracy at τ** | Accuracy on the subset of trials where max-prob ≥ τ | The deployment-realistic "only update feedback on high-confidence trials" frame |
| **β-reliability (rep-rep r)** | Pearson r between repeated presentations of the same image | Single-trial signal stability; the floor on what neurofeedback can resolve |

50-way top-1 is included for paper comparability, not deployment guidance.

## A note on naming: "GLMdenoise K=N" vs "aCompCor K=N"

Cells in our `prereg/` directory named `RT_paper_EoR_K{N}_CSFWM_inclz` are
**aCompCor** (Behzadi et al. 2007), not canonical GLMdenoise (Kay et al.
2013). Both methods share machinery (PCA on a noise pool → top-K
components as nuisance regressors) but differ in the noise pool
definition:

- **aCompCor**: anatomical noise pool (CSF + WM tissue masks)
- **GLMdenoise**: data-driven noise pool (voxels with low task-R²)

We use the file-naming `K{N}_CSFWM` for historical reasons; semantically
these are aCompCor variants. Canonical GLMdenoise on this dataset (run as
GLMsingle Stage 2 with task-R²-based noise pool + CV) selects
**`pcnum=0`** — i.e., zero noise components are useful. The +1.2pp 2-AFC
lift from K=7 comes from anatomical (aCompCor-style) noise modeling, not
from GLMdenoise.

## Empirical ranking on this dataset (fold-10, single-rep filter)

From `unified_metrics.json` (top of leaderboard by 2-AFC):

| Cell | Method | top-1 | brain | **2-AFC** | AUC | d | β-rel | Notes |
|---|---|---|---|---|---|---|---|---|
| `RTmotion_GLMsingle_singleRep` | Full GLMsingle (HRF lib + GLMdenoise CV-K=0 + fracridge) | 62.0% | 70.0% | **96.2%** | 0.963 | **2.51** | 0.000* | Offline-style; canonical |
| **`RT_paper_EoR_K7_CSFWM_inclz`** | **aCompCor K=7 + Glover + AR(1)** | 52.0% | 62.0% | **96.3%** | **0.951** | **2.38** | **0.243** | **Best RT-deployable** |
| `RT_paper_EoR_K3_CSFWM_inclz` | aCompCor K=3 | 54.0% | 66.0% | 96.2% | 0.948 | 2.33 | 0.240 | (within sampling of K=7) |
| `RT_paper_EoR_K5_CSFWM_inclz` | aCompCor K=5 | 48.0% | 66.0% | 96.1% | 0.950 | 2.35 | 0.243 | |
| `RT_paper_EoR_K10_CSFWM_inclz` | aCompCor K=10 | 52.0% | 62.0% | 95.7% | 0.944 | 2.29 | 0.238 | (prior baseline) |
| `RT_paper_EndOfRun_pst_None_inclz` | No GLMdenoise (motion only) | 50.0% | 62.0% | 95.1% | 0.945 | 2.28 | 0.232 | aCompCor-free baseline |
| `RT_paper_EoR_fmriprep_inclz` | fMRIPrep BOLD + Glover + AR(1) | 48.0% | 64.0% | 94.1% | 0.934 | 2.11 | 0.215 | not RT-deployable |
| `RT_paper_EoR_OLS_hrflib_inclz` | Per-voxel HRF library + OLS | 42.0% | 48.0% | 92.8% | 0.900 | 1.84 | 0.152 | HRF lib alone |
| `RT_paper_Slow_pst25_inclz` | Slow tier (37s window) | 50.0% | 52.0% | 90.9% | 0.912 | 1.94 | 0.192 | |
| `RT_paper_Fast_pst5_inclz` | Fast tier (7.5s window) | 44.0% | 26.0% | 87.8% | 0.839 | 1.41 | 0.198 | |

**Bootstrap caveat (n=50):** the K=3..K=10 aCompCor cells are statistically
indistinguishable from each other and from full GLMsingle at α=0.05.
Only K=20 (collapsed) and full GLMsingle vs K=0 (no aCompCor) reach
significance. K=7's empirical edge is real on point estimate but within
sampling noise.

*`RTmotion_GLMsingle_singleRep` β-reliability is 0.000 because the saved
betas are already first-rep-filtered to 50 trials; reliability needs the
raw 693-trial output to compute.

## Headline takeaways for closed-loop deployment

1. **All RT-deployable cells hit 88-96% 2-AFC.** This is the relevant number
   for binary or small-pool closed-loop tasks, and it is excellent for
   deployment. The 50-way top-1 numbers (36-62%) are misleadingly low for
   judging deployment readiness.

2. **CSF/WM K=10 is currently the best RT-deployable processing.** Within
   0.5pp of full canonical GLMsingle on the deployment-relevant 2-AFC
   metric, despite using only RT-streamable operations (PCA noise-pool
   regression + nilearn LSS + Glover + AR(1) + cumulative z).

3. **Cohen's d > 2.0** on the best cells — a strong effect size, far above
   the d > 0.8 "large effect" threshold typical for fMRI MVPA.

4. **Latency-vs-accuracy tradeoff is shallow on 2-AFC.** Fast (7.5s window)
   gets 87.8%, EoR (full-run, ~150s) gets 95.1%. Roughly 1pp 2-AFC per 20s
   of additional BOLD. For most closed-loop applications this argues for
   the Fast tier — 87.8% binary accuracy at 7.5s latency is excellent.

5. **β-reliability is 0.15-0.24 across cells.** This is a hard ceiling on
   single-trial decode quality — even with perfect downstream processing,
   no decoder can do better than the underlying signal repeatability.

## What we'd recommend testing next

To improve the RT-deployable ceiling (95.7% 2-AFC):

- **CSF/WM K-sweep** (K ∈ {0, 3, 5, 7, 10, 15, 20}) — identify whether K=10 is the right shrinkage level
- **aCompCor variant** — high-pass-filtered CSF/WM PCs, the canonical ANTs/fMRIPrep noise model
- **Per-voxel HRF library + CSF/WM K=10 jointly** — both stages are RT-streamable; their interaction wasn't tested in isolation
- **Voxel selection refinement** — sweep the relmask r threshold (currently r > 0.2) or use beta-reliability-derived selection
- **Variant G prior with calibrated posterior** — for closed-loop applications where confidence quantification matters more than point-estimate top-1

To improve top-1/brain retrieval beyond `RTmotion_GLMsingle_singleRep`:

- The current win (62% top-1 / 70% brain) is essentially the GLMsingle
  ceiling on this checkpoint. Major lifts would require either:
  - More fine-tuning data (paper appendix shows 2-session > 1-session)
  - A different checkpoint family (rt_ft variants per Rishab Discord)
  - Architectural changes downstream (different ridge regularization,
    diffusion prior tuning)

## References for Norman-lab-style closed-loop

- deBettencourt MT, Cohen JD, Lee RF, Norman KA, Turk-Browne NB.
  Closed-loop training of attention with real-time brain imaging.
  *Nat Neurosci* 18:470-475 (2015).
- Mennen AC, Norman KA, Turk-Browne NB. Attentional bias in depression:
  Understanding mechanisms to improve training and treatment.
  *Curr Opin Psychol* 29:266-273 (2019).
- Heunis et al. Quality and denoising in real-time fMRI neurofeedback:
  A methods review. *Hum Brain Mapp* 41:3439-3467 (2020).
  → catalog of 128 RT-fMRI NF studies and their methodological choices.
