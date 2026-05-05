# RT-fMRI preprocessing coverage vs. Heunis et al (2020) taxonomy

Heunis et al (2020) *Quality and denoising in real-time fMRI neurofeedback: A methods review* organizes RT-fMRI denoising into ~7 categories. This document maps each of our 136 prereg cells against those categories, stating coverage, best result, and verdict.

All numbers below are 50-way single-rep Image retrieval (top-1) on sub-005 ses-03 special515 first-occurrences, fold-0 ckpt. Baseline (rtmotion AR(1) LSS β + Glover + cum-z): **36% Fast / 44% Slow / 56% EoR** (paper anchors 36 / 58 / 66 at the paper's reporting subset).

## 1. Geometric corrections

| Subcategory | What we tested | Cells | Verdict |
|---|---|---|---|
| **Motion correction** | MCFLIRT (rtmotion), fMRIPrep MC | all rtmotion cells; `*_fmriprep_inclz` cells | fmriprep MC = +8pp on Slow at single-rep over rtmotion MC; +2pp at EoR; ~0pp at Fast. fmriprep is the better motion-corrected source when the post-stim window is long enough |
| **Motion regressors** | 6-param mc_params from MCFLIRT | every cell uses these as nuisance regressors | included in baseline |
| **Slice timing correction** | NOT TESTED | — | nilearn `slice_time_ref=0` matches paper exactly; STC was not applied or ablated |
| **Distortion correction (top-up)** | implicit in fmriprep BOLD | `*_fmriprep_inclz` | folded into the BOLD-source comparison |
| **24-param Friston motion** | NOT TESTED | — | only 6-param used; squared/derivative motion regressors not added |

## 2. Spatial preprocessing

| Subcategory | What we tested | Cells | Verdict |
|---|---|---|---|
| **Brain extraction** | finalmask (19,174 vox) from canonical | all cells | held fixed |
| **Subject registration** | T1w-space throughout (no MNI) | all cells | matches paper |
| **Reliability mask** | ses-01 relmask 2792 vox | all cells | held fixed at the paper's 2792-voxel input |
| **Spatial smoothing (Gaussian)** | NOT TESTED | — | nilearn `smoothing_fwhm=None`. Heunis lists FWHM 4-8mm as common; we ablated this nowhere |
| **Spatial normalization (template)** | NOT TESTED | — | the paper's pipeline is subject-native; templating was outside scope |

## 3. GLM specification

| Subcategory | What we tested | Cells | Verdict |
|---|---|---|---|
| **HRF — Glover canonical** | baseline | most cells | wins at subset2; ties on subset0 |
| **HRF — temporal derivative** | `*_glov_td_inclz` | 1 cell | +0pp on subset0, +2pp on subset1 vs Glover-only. Marginal |
| **HRF — temporal + dispersion** | `*_glov_tddisp_inclz` | 1 cell | -2pp subset0, -6pp subset1. Hurts |
| **HRF — SPM canonical** | `*_spm_inclz` | 1 cell | -2pp subset0, -4pp subset1. Worse than Glover |
| **HRF — SPM + temporal+disp** | `*_spm_tddisp_inclz` | 1 cell | -4pp subset0, -10pp subset1 |
| **HRF — 20-HRF library (per-voxel)** | `*_hrflib_*` cells | 2 cells | -10pp subset0. Hurts when not jointly fit with ridge |
| **HRF — FLOBS basis (3-fn)** | `VariantB_FLOBS_glover_rtm`, `VariantB_FLOBS_fitted_glover_rtm` | 2 cells | -18 to -36pp subset0. Catastrophic |
| **HRF — adaptive peak detection** | `RT_paper_Slow_adaptive_n{15,20}_inclz` | 2 cells | +2pp subset0 vs pst=20. Marginal |
| **Boxcar regressor duration** | `OLS_K10_dur{0,1,2,3}_glover_rtm` | 4 cells | dur=0 vs 3 makes ~0pp difference at Glover-canonical HRF |
| **Drift model — cosine order=1, hp=0.01** | baseline | all cells | held fixed |
| **Drift model — alternatives** | NOT ABLATED | — | polynomial drift, cosine higher orders not tested |

## 4. Statistical regression (nuisance)

| Subcategory | What we tested | Cells | Verdict |
|---|---|---|---|
| **aCompCor (CSF+WM PCs)** | K-sweep 0,3,5,7,10,15,20 with FAST PVE-derived noise pool, with/without HP filter, with/without erosion | `RT_paper_EoR_K*_CSFWM_*_inclz` (~15 cells) | K=3-7 sweet spot at subset1, K>10 hurts. Erosion×1 helps subset2 by +4pp. K=0 (no aCompCor) ties at subset2 = 78% — aCompCor helps subset0/1 but is neutral at avg-of-3 |
| **GLMdenoise (PCA, task-driven pool)** | K=0..15 with relmask noise pool | `OLS_glover_rtm_denoiseK*` | K=10 with relmask hurts 6pp (task-leakage in pool); K=0 with `_softFrac_only` matches OLS baseline |
| **Joint GLMdenoise + fracridge** | `*_glmdenoise_fracridge` | 4 cells | reaches subset2=72-74; doesn't beat aCompCor stack |
| **Segmentation source for noise pool** | FAST (default), DeepMriPrep, SynthSeg | `RT_paper_EoR_K7_*_HP_e1_inclz` × 3 | FAST=DeepMriPrep at subset1 (66); SynthSeg lags 8-14pp. Liberal noise-pool definition hurts |
| **Stein shrinkage on per-trial βs** | λ ∈ {0.7, 0.85, 0.95, 1.0} | `*_stein*_inclz` × 4 | shrinkage hurts; λ=1.0 (no shrinkage) is best |
| **fracridge (per-voxel)** | frac ∈ {0.3, 0.5, 0.7, 0.9, 1.0, CV-Fratio} | `OLS_glover_rtm_denoiseK0_fracR*` × 6, `RT_paper_EoR_OLS_glover_frac*_inclz` × 3 | catastrophic standalone (2-30%); only works inside GLMsingle's joint stack |
| **RETROICOR / RVHRcor / physio** | NOT TESTED | — | physiological recordings not available in dataset |
| **ICA-AROMA / ICA-FIX** | NOT TESTED | — | batch methods; not RT-compatible without modification |
| **Global signal regression (GSR)** | NOT TESTED — and the pBOLD literature argues against it for closed-loop | — | Bruzdiak et al 2026 (bioRxiv 2026.03.19.712948) shows GSR improves tSNR but *lowers* pBOLD (the probability that signal change is BOLD-dominated). For closed-loop where downstream classification depends on BOLD fidelity, GSR is contra-indicated despite its tSNR gain. |
| **Scrubbing / frame censoring** | `OLS_K10_FrameCensor_glover_rtm` | 1 cell | -10pp subset0. Censoring hurts more than the bad TRs cost |

## 5. Filtering

| Subcategory | What we tested | Cells | Verdict |
|---|---|---|---|
| **High-pass filter (drift)** | hp=0.01 Hz on BOLD via cosine drift | every cell | held fixed |
| **High-pass on noise pool** | hp=0.01 Hz applied to CompCor regressors before SVD | `*_HP_*` cells | -2pp subset1 alone, +2pp combined with erosion |
| **Band-pass filter** | `OLS_K10_BandPass_glover_rtm` | 1 cell | -8pp subset0. Hurts |
| **Temporal smoothing** | `OLS_K10_TempSmooth_glover_rtm` | 1 cell | -8pp subset0 |
| **Low-pass / cardiac-resp band** | NOT TESTED | — | requires physio frequency targeting |

## 6. Trial-level estimation

| Subcategory | What we tested | Cells | Verdict |
|---|---|---|---|
| **LSS (per-trial refit)** | nilearn `FirstLevelModel` with `probe`/`reference` contrast | every nilearn cell | baseline |
| **LSA per-run (single joint fit)** | `OLS_persistentLSA_K0_glover_rtm`, `*_K10_*` | 2 cells | -8pp subset2 vs AR(1) LSS |
| **LSA cross-run (block-diagonal)** | `OLS_persistentLSA_crossrun_K{0,10}_glover_rtm` | 2 cells | identical to per-run LSA |
| **Streaming/incremental RLS GLM** | growing-design ridge OLS at decode time, K7+CSFWM+HP+e1 nuisance | `RT_paper_RLS_{Fast_pst5,Slow_pst20,EoR}_K7CSFWM_HP_e1_inclz` (3 cells) | **+10pp Slow subset0, +14pp Slow subset1, +4pp EoR subset1 vs LSS baseline.** Beats paper Slow 58% by +12pp and paper EoR 66% by +4pp at the paper's subset1 anchors. Hurts Fast (-12pp) due to underdetermined design. See `STREAMING_RLS_GLM.md` |
| **GLMsingle (HRF-lib + GLMdenoise + fracridge joint)** | local on rtmotion, local on fmriprep, canonical on fmriprep | `RTmotion_GLMsingle_*`, `*_FMRIPREP_LOCAL/`, canonical Offline cells | local-fmriprep-GLMsingle subset2=78%, canonical=76%, local-rtmotion-GLMsingle=50% subset0 only |

## 7. Real-time-specific

| Subcategory | What we tested | Cells | Verdict |
|---|---|---|---|
| **Causal cumulative z-score** | `(arr[i] - mean(arr[:i])) / std` | non-`_inclz` cells | baseline form |
| **Inclusive cum-z** | `(arr[i] - mean(arr[:i+1])) / std` (matches `mindeye.py:771`) | `*_inclz` cells | +2-4pp subset0, neutral elsewhere; matches paper code |
| **Running average over repeats** | subset semantics from `mindeye.py:947-955`: subset0/1/2 | post-hoc evaluation across all cells | the deployment-relevant analysis axis |
| **Cross-run AR(1) ρ** | `HybridOnline_AR1freq_glover_rtm` | 1 cell | +0pp vs per-run AR(1). Frozen ρ doesn't help |
| **Online drift estimation** | NOT TESTED | — | drift is fit per-run via cosine basis; online drift state untested |
| **Linear Dynamical System** | `OLS_LDS_glover_rtm`, `AR1freq_LDS_glover_rtm`, `OLS_denoiseK10_LDS_glover_rtm` | 3 cells | LDS adds ~0pp |
| **Bayesian per-voxel AR(1)** | `VariantG_*` family | 9 cells | structured prior reaches 52-66% subset1; matches but doesn't beat AR(1) LSS+aCompCor |
| **Cross-trial Bayesian update** | `SameImagePrior_VariantG_glover_rtm`, `VariantG_NUTSprior_glover_rtm` | 2 cells | NUTSprior chance-level (broken); SameImagePrior 40% subset0 |

## 8. Quality control / spike detection

| Subcategory | What we tested | Cells | Verdict |
|---|---|---|---|
| **DVARS / FD framewise displacement** | NOT TESTED explicitly | — | motion params used as confounds but no FD-based censoring decision |
| **Spike detection** | NOT TESTED | — | |
| **Frame censoring** | `OLS_K10_FrameCensor_glover_rtm` | 1 cell | hurts (see §4) |

## 9. Feature-level (post-β)

Things tested AT THE β LEVEL (after extraction, before model):

| Subcategory | What we tested | Cells | Verdict |
|---|---|---|---|
| **Per-voxel reliability mask** | ses-01 relmask 2792 vox | held fixed | |
| **HOSVD spatial denoising** | `RT_streaming_pst8_HOSVD_K{5,10}_*`, `*_ResidHOSVD_*` | 6 cells | -6 to -16pp subset0. Hurts |
| **Log-signature features (depth-1, depth-2 Lévy area)** | Phase A/B/C runs from 2026-05-04 | post-hoc tests | +0 to +2pp; within sampling noise (see `LOGSIG_RESULTS.md`) |
| **Cross-voxel signatures (multivariate)** | NOT BUILT | — | scope: pairwise Lévy area on top-K reliable voxels |

---

## Coverage summary

**Categories thoroughly covered**: motion correction (BOLD source), HRF specification (8 variants), aCompCor + GLMdenoise + segmentation source (~25 cells), fracridge sweep, trial-level LSS vs LSA persistent, streaming z-score variants, post-β denoising (HOSVD, signatures, Stein).

**Categories with light coverage**:
- Boxcar duration (4 cells)
- Drift model alternatives (only cosine order=1 tested)
- Streaming pst sweep (Fast 4-6, Slow 18-30)

**Categories NOT TESTED**:

1. **Spatial smoothing (Gaussian FWHM)** — common in offline pipelines; never ablated
2. **Slice timing correction** — held fixed at `slice_time_ref=0`; not ablated
3. **24-param Friston motion / squared+derivative motion regressors** — only 6-param used
4. **RETROICOR / physiological regression** — no physio data in this dataset
5. **ICA-AROMA / ICA-FIX denoising** — batch; would need RT-compatible variant
6. **Global signal regression (GSR)** — never added as confound; per Bruzdiak 2026 we shouldn't, since GSR improves tSNR while lowering pBOLD (BOLD-dominance probability). Contra-indicated for closed-loop.
7. **DVARS / FD-based prospective scrubbing** — only 1 frame-censoring cell
8. ~~Streaming/incremental RLS GLM~~ — **BUILT 2026-05-04**, see `STREAMING_RLS_GLM.md`. Beats paper Slow by +12pp and paper EoR by +4pp at the paper's subset1 anchors.
9. **Multivariate cross-voxel signatures** — pairwise Lévy area unbuilt
10. **Distortion correction ablation** — fmriprep does it; rtmotion's lack of top-up not isolated
11. **Drift model variants** (polynomial, higher cosine orders, online state-space drift)
12. **Online detrending** (per-TR rather than per-run)

---

## Honest assessment

We covered the core preprocessing categories from Heunis et al thoroughly: HRF, noise regression (aCompCor, GLMdenoise), trial estimation (LSS vs LSA), regularization (fracridge, ridge, Stein), real-time z-scoring, and segmentation source. The factorial gives strong evidence that:

- **Glover canonical HRF + AR(1) LSS + aCompCor (K=3-7, CSF+WM, eroded) + inclusive cum-z** is at or near the ceiling for this β-extraction class.
- **fmriprep BOLD source helps at Slow latency** by ~8pp single-rep — the only positive BOLD-source effect we found.
- **GLMsingle's stages 2+3 (denoise PCA + fracridge) are essentially redundant** with AR(1) LSS + aCompCor at avg-of-3 latencies.

What remains genuinely under-tested are categories that either need additional data (physio), require a different mechanism (streaming RLS — now built), or fall outside the paper's reported pipeline (smoothing, GSR, ICA). None of these are obviously load-bearing for the Fast/Slow gap — but they're the honest gaps in our coverage relative to the Heunis taxonomy.

## Multi-echo EPI as the highest-leverage acquisition upgrade

The Fast tier ceiling at 36% Image is largely SNR-limited at 7.5s post-stim BOLD windows. Multi-echo EPI (Posse 1999, Kundu 2012, Heunis 2021) addresses this on three independent axes:

1. **Optimal TE-weighted combination per voxel** — universal +30-50% tSNR gain via tedana's optimal combination
2. **ME-ICA denoising (TE-dependence as BOLD signature)** — beats ICA-AROMA for separating BOLD from non-BOLD components
3. **pBOLD as a deployment-grade QA metric (Bruzdiak 2026, bioRxiv 2026.03.19.712948)** — quantifies "probability that signal change is BOLD-dominated" using the linear-in-TE BOLD model. Validated on N=439 scans; higher pBOLD predicts better whole-brain FC phenotype prediction. Specifically, pBOLD distinguishes BOLD-fidelity-preserving denoising (which raises pBOLD) from BOLD-suppressing denoising like GSR (which raises tSNR while lowering pBOLD) — a distinction tSNR alone cannot make.

For closed-loop deployment with ME data, **per-trial pBOLD is the right primary signal-quality alert**, not tSNR. tSNR can be misleadingly high under GSR-style denoising while BOLD-fidelity collapses.

Recommended for the Princeton group's next ME-collected data:
- Sequence: ME-EPI with TE = [12, 28, 44]ms at MB=4 (or similar)
- Combination: tedana optimal-combination per-voxel before downstream GLM
- QA: pBOLD per-voxel per-run as the primary alert metric (alongside per-trial decoder confidence)
- **Skip GSR** — Bruzdiak shows it actively worsens BOLD-fidelity

— Mapping completed 2026-05-04, fold-0, n=50 special515 ses-03.
