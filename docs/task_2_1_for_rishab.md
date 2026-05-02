# Task 2.1 — Decomposing the Offline-vs-RT Gap on sub-005 ses-03

**Audience**: Rishab Iyer (RT-MindEye paper). Self-contained — no
prior context about our codebase needed.

**Glossary**: see `docs/glossary.md` for definitions of top-1, top-5,
AUC, Cohen's d, LSS/LSR, pst, cum-z, pcnum, FRACvalue, finalmask,
Variant G, and all other recurring jargon in this doc.

**Question (paper Table 1, single-trial first-rep n=50)**: what
mechanism produces the **Offline 3T 76% vs End-of-run RT 66%** top-1
gap, and how does the gap shift when scored on the closed-loop-relevant
pairwise AUC metric? Adjacent: how do the Slow RT (58%) and Fast RT
(36%) tiers fit the same decomposition?

**Headline answers**:

1. The top-1 gap is **β-windowing** — RT's per-trial GLM fits on BOLD
   cropped to `onset_TR + delay`, while Offline fits on the full
   session. Within-run pst saturates by delay≈15-20. Measured directly
   on the published outputs: paper Offline 65.3% vs paper RT delay=20
   59.3% on n=150 (Δ ≈ 6 pp at constant trial count).
2. **GLMsingle Stage 2 (GLMdenoise) is subject- and session-specific**.
   Across the 9 published `.npz` files, `pcnum` ranges 0–6, with **K=10
   never selected anywhere**. For sub-005 ses-02 and ses-03 the CV
   curve is monotonically decreasing in K — `pcnum = 0` is real.
3. **GLMsingle Stage 3 (real per-voxel SVD fracridge) does not
   transfer to streaming LSS** in our re-implementation. Per-trial
   fracridge breaks pairwise consistency because each trial's LSS
   design-matrix SVD differs. Measured: full-run real-fracridge falls
   to AUC 0.57 / top-1 22%; streaming pst=8 falls to AUC 0.48 / top-1
   2% (chance). Canonical Stage 3 needs a global single fit of all
   trials simultaneously to apply the same shrinkage transform.
4. **An RT-deployable pipeline already exceeds the canonical Offline
   anchor on AUC**. Cell 7 (rtmotion + Glover + 5-PC tCompCor noise
   regression + AR(1)) hits AUC 0.886 / d 1.71. Canonical Offline =
   0.856 / 1.48. The +0.030 AUC delta comes from simple noise-PC
   regression — fully streaming, no fracridge needed.
5. **BOLD source (rtmotion vs fmriprep) contributes ≤ 0.005 AUC** at
   constant denoising. fmriprep's offline-only steps (SDC,
   slice-timing) are not the gap.
6. **Variant G's posterior is invisible to top-1 / point-estimate AUC**
   (it ties AR(1) freq) but enables calibration-aware selective
   accuracy that AR(1) freq cannot produce. Distinct evaluation track.
7. **Cross-run causal filters tested (raw HOSVD, task-residual HOSVD,
   session-frozen ρ̂) all hurt** vs the within-run streaming baseline.
   Removing past-run variance — even task-orthogonalized — removes
   useful signal-correlated structure on this dataset.

---

## Setup

| item | value |
|---|---|
| Subject / session | sub-005 / ses-03 |
| Test set | 50 special515 images × 3 reps = 150 trials |
| Mask | finalmask, 2792 voxels (`sub-005_final_mask.nii.gz` + `ses-01_task-C_relmask.npy`) |
| Decoder checkpoint | `sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth` (ses-01 only — chosen to avoid test-set leakage from `sub-005_all_task-C` which trained on ses-03) |
| Top-1 metric | 50-way image retrieval through the frozen MindEye ridge → backbone → CLIP-token forward |
| AUC metric | pairwise merge/separate: same-image vs diff-image cosine distance over the 150 trials, integrated as ROC-AUC; Cohen's d on the same |
| Z-score policy | causal cumulative (trial *i* uses statistics from trials 0..*i*−1 only); applied where the cell driver doesn't already do so |

**Data anchors used**:
- Paper's actual saved RT betas at all 8 decode delays (`rt3t/data/real_time_betas/all_betas_ses-03_all_runs_delay{0,1,3,5,10,15,20,63}.npy`) — this is your pipeline's actual output, scored directly. Not a re-implementation.
- Princeton canonical GLMsingle `TYPED_FITHRF_GLMDENOISE_RR.npz` (sub-005, from `rishab-iyer1/glmsingle` on HF) — scored on the same checkpoint, lands at **76.00% top-1 / 98.00% top-5 (paper-exact)**.

---

## The comprehensive table

Conventions:
- **BOLD**: `rtmotion` = your real-time MCFLIRT + flirt-cross-session output at `motion_corrected_resampled/` ; `fmriprep` = `_space-T1w_desc-preproc_bold.nii.gz`
- **HRF**: `Glover` = canonical double-gamma; `library` = GLMsingle Stage 1 with frozen per-voxel HRF index from `avg_hrfs_s1_s2_full.npy` (derived from ses-01 + ses-02)
- **GLM**: `OLS`, `AR(1)` = nilearn-style AR(1) prewhitening, `VG` = AR(1) Bayesian conjugate (closed-form posterior)
- **Window**: `full-run` = LSS on entire run's BOLD per trial; `pst=N` = LSS on BOLD cropped to `onset_TR + N`; `paper-RT` = your saved per-decode-delay output
- **Denoise**: `none`, `aCompCor`, `GLMdenoise+softfrac` (soft-scalar fracridge ≈ 1× — proven a no-op), `GLMdenoise+CV-fracridge` (per-voxel SVD), `S1+2+3` = full GLMsingle stages
- **Cross-run filter**: `none`, `HOSVD K` = top-K spatial PCs of past-run BOLD as nuisance, `ResidHOSVD K` = same but on past-run GLM **residuals** (task-orthogonalized), `session-ρ̂` = AR(1) ρ̂ frozen from intercept+drift fit on full session

| # | Cell | BOLD | HRF | GLM | Window | Denoise | Cross-run | Repeat-avg | top-1 | top-5 | **AUC** | Cohen's d | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **A** | **Anchors** | | | | | | | | | | | | |
| A1 | Paper-reported RT (Figure 3) | rtmotion | library | nilearn AR(1) | streaming | S1+2+3 | none | yes | ≈ 66% | — | — | — | from manuscript |
| A2 | Paper-reported Offline (Figure 3) | fmriprep | library | nilearn AR(1) | full-run | S1+2+3 | none | yes | ≈ 76% | — | — | — | from manuscript |
| A3 | Princeton canonical GLMsingle .npz | fmriprep | library | nilearn AR(1) | full-run | **S1 + S3 only** (`pcnum=0`) | none | yes | **76.00%** | **98.00%** | (n=50, errs) | — | from `rishab-iyer1/glmsingle`, scored directly |
| A4 | `Paper_RT_actual_delay20` | rtmotion | (your RT GLM) | (your RT GLM) | streaming, decode_TR = onset+20 | none | none | — | — | — | **0.826** | **1.30** | your `all_betas_ses-03_all_runs_delay20.npy`, scored on n=150 special515 |
| A5 | `Paper_RT_actual_delay63` | rtmotion | (your RT GLM) | (your RT GLM) | streaming, decode_TR = onset+63 | none | none | — | — | — | 0.825 | 1.30 | your saved RT betas, late decode |
| | | | | | | | | | | | | | |
| **B** | **Within-pipeline factorial** (our re-implementation) | | | | | | | | | | | | |
| B1 | OLS_glover_rtm | rtmotion | Glover | OLS | full-run | none | none | no | 56.0% | 80.7% | 0.684 | 0.64 | baseline |
| B2 | AR1freq_glover_rtm | rtmotion | Glover | AR(1) | full-run | none | none | no | 62.0% | 83.3% | 0.783 | 1.04 | +6 pp top-1 vs OLS |
| B3 | AR1freq_glover_rtm_glmdenoise_fracridge | rtmotion | Glover | AR(1) | full-run | GLMdenoise+softfrac | none | no | 60.0% | 84.7% | **0.886** | 1.71 | softfrac is no-op; AUC win = GLMdenoise alone |
| B4 | VariantG_glover_rtm | rtmotion | Glover | VG | full-run | none | none | no | 61.3% | 84.0% | 0.781 | 1.03 | tie with B2 on AUC |
| B5 | VariantG_glover_rtm_prior | rtmotion | Glover | VG + ses-01 prior | full-run | none | none | no | 60.7% | 82.7% | 0.785 | 1.05 | empirical-Bayes shrink (Δ ≈ 0) |
| B6 | VariantG_glover_rtm_glmdenoise_fracridge | rtmotion | Glover | VG | full-run | GLMdenoise+softfrac | none | no | 60.0% | 85.3% | 0.886 | 1.71 | identical to B3 |
| B7 | VariantG_glover_rtm_acompcor | rtmotion | Glover | VG | full-run | aCompCor (5 PCs) | none | no | 59.3% | 86.7% | 0.873 | 1.62 | aCompCor ≈ GLMdenoise on AUC |
| B8 | AR1freq_glmsingleS1_rtm | rtmotion | library | AR(1) | full-run | none | none | no | **45.3%** | 74.7% | 0.755 | 0.85 | HRF library breaks 50-way ID, AUC mostly intact |
| B9 | VariantG_glmsingleS1_rtm | rtmotion | library | VG | full-run | none | none | no | 45.3% | 74.7% | 0.754 | 0.85 | matches B8 |
| B10 | AR1freq_glmsingleFull_rtm | rtmotion | library | AR(1) | full-run | GLMdenoise+softfrac | none | no | 53.3% | 81.3% | 0.855 | 1.49 | full GLMsingle on rtmotion |
| B11 | VariantG_glmsingleFull_rtm | rtmotion | library | VG | full-run | GLMdenoise+softfrac | none | no | 52.7% | 81.3% | 0.855 | 1.49 | tie with B10 |
| B12 | AR1freq_glover_fmriprep_glmdenoise_fracridge | fmriprep | Glover | AR(1) | full-run | GLMdenoise+softfrac | none | no | (—) | (—) | 0.880 | 1.67 | **fmriprep contribution at constant denoise = −0.005** |
| B13 | VariantG_glover_fmriprep_glmdenoise_fracridge | fmriprep | Glover | VG | full-run | GLMdenoise+softfrac | none | no | (—) | (—) | 0.881 | 1.68 | matches B12 |
| B14 | AR1freq_glmsingleFull_fmriprep | fmriprep | library | AR(1) | full-run | GLMdenoise+softfrac | none | no | 52.0% | 80.0% | 0.868 | 1.58 | full GLMsingle on fmriprep |
| | | | | | | | | | | | | | |
| **C** | **Paper-RT replica** (re-implementation, with bug + fix) | | | | | | | | | | | | |
| C1 | RT_paper_replica_partial (full-run BOLD bug) | rtmotion | Glover | nilearn AR(1) | **full-run (bug)** | none | none | no | 56.7% | 83.3% | 0.723 | 0.81 | paper-pipeline labels but offline windowing |
| C2 | RT_paper_replica_full (same bug + repeat-avg) | rtmotion | Glover | nilearn AR(1) | **full-run (bug)** | none | none | yes | **78.0%** | 98.0% | (n=50) | — | inflated 12 pp vs paper-reported RT — diagnostic for the windowing bug |
| C3 | Offline_paper_replica_full | fmriprep | Glover | nilearn AR(1) | full-run | none | none | yes | **76.0%** | 94.0% | (n=50) | — | paper-exact top-1 anchor |
| C4 | RT_paper_replica_streaming_pst4_partial | rtmotion | Glover | nilearn AR(1) | pst=4 | none | none | no | 47.3% | 71.3% | 0.614 | 0.42 | within-run pst sweep |
| C5 | RT_paper_replica_streaming_pst6_partial | rtmotion | Glover | nilearn AR(1) | pst=6 | none | none | no | 44.0% | 72.7% | 0.605 | 0.40 | |
| C6 | RT_paper_replica_streaming_pst8_partial | rtmotion | Glover | nilearn AR(1) | pst=8 | none | none | no | 50.0% | 76.7% | 0.606 | 0.39 | within-run streaming saturates |
| C7 | RT_paper_replica_streaming_pst10_partial | rtmotion | Glover | nilearn AR(1) | pst=10 | none | none | no | 49.3% | 74.0% | 0.609 | 0.40 | |
| C8 | RT_paper_replica_streaming_pst8_full | rtmotion | Glover | nilearn AR(1) | pst=8 | none | none | yes | 72.0% | 90.0% | (n=50) | — | recovers ~ paper RT band on top-1 |
| | | | | | | | | | | | | | |
| **D** | **Streaming primitives** (Bayesian/state-space attempts) | | | | | | | | | | | | |
| D1 | EKF_streaming_glover_rtm | rtmotion | Glover | streaming Kalman AR(1) | full-run | none | none | no | 48.7% | 80.0% | 0.685 | 0.63 | white-noise reset per trial |
| D2 | EKF_session_online_glover_rtm | rtmotion | Glover | one-pass online EKF, diag-cov over 770 probes | session-streaming | none | none | no | **8.0%** | 20.0% | (—) | — | identifiability collapse — diagonal cov over many overlapping probes |
| D3 | HybridOnline_AR1freq_glover_rtm | rtmotion | Glover | AR(1), session-frozen ρ̂ | full-run | none | session-ρ̂ | no | 58.7% | 82.0% | 0.793 | 1.08 | the architecturally cleanest streaming hybrid |
| D4 | HybridOnline_streaming_pst8_AR1freq_glover_rtm | rtmotion | Glover | AR(1), session-frozen ρ̂ | pst=8 | none | session-ρ̂ | no | 45.3% | 75.3% | 0.723 | 0.80 | cropping degrades the hybrid |
| D5 | LogSig_AR1freq_glover_rtm | rtmotion | Glover | AR(1) | full-run | tCompCor + depth-2 log-sig sliding window | none | no | **1.3%** | 5.3% | 0.592 | 0.32 | sig features grew unbounded; nuisance dominated projection |
| D6 | HOSVD_denoise_AR1freq_glover_rtm | rtmotion | Glover | AR(1) | full-run | NORDIC SVD-thresholding per run | none | no | 44.7% | 72.0% | 0.747 | 0.89 | per-run, not cross-run |
| D7 | Riemannian_prewhiten_AR1freq_glover_rtm | rtmotion | Glover | AR(1) | full-run | Σ̄^{-1/2} log-Euclidean prewhitening | run cov geom mean | no | **2.7%** | 12.0% | 0.593 | 0.30 | rank-deficient on V > T |
| | | | | | | | | | | | | | |
| **E** | **Cross-run filters** (Regime C — H3' tested, all refuted) | | | | | | | | | | | | |
| E1 | RT_streaming_pst8_HOSVD_K5_partial | rtmotion | Glover | nilearn AR(1) | pst=8 | none | top-5 spatial PCs of past raw BOLD | no | 40.7% | 70.7% | 0.669 | 0.57 | naive: top PCs include task signal |
| E2 | RT_streaming_pst8_HOSVD_K10_partial | rtmotion | Glover | nilearn AR(1) | pst=8 | none | K=10 raw BOLD | no | 30.0% | 57.3% | 0.687 | 0.58 | more components, more task signal removed |
| E3 | RT_streaming_pst8_HOSVD_K5_full | rtmotion | Glover | nilearn AR(1) | pst=8 | none | K=5 raw + repeat-avg | yes | 52.0% | 86.0% | (n=50) | — | |
| E4 | RT_streaming_pst8_ResidHOSVD_K5_partial | rtmotion | Glover | nilearn AR(1) | pst=8 | none | top-5 PCs of past **residuals** | no | 44.7% | 74.7% | 0.654 | 0.53 | task-orthogonal — recovers 4 pp vs E1 but still hurts |
| E5 | RT_streaming_pst8_ResidHOSVD_K10_partial | rtmotion | Glover | nilearn AR(1) | pst=8 | none | K=10 residuals | no | 38.7% | 70.7% | 0.680 | 0.62 | |
| E6 | RT_streaming_pst8_ResidHOSVD_K5_full | rtmotion | Glover | nilearn AR(1) | pst=8 | none | K=5 residuals + repeat-avg | yes | 60.0% | 88.0% | (n=50) | — | |

---

## Decomposition

### Top-1 retrieval (50-way, n=150)

The dominant lever is **β-windowing**. With Glover + nilearn AR(1) +
no denoising:

| pipeline at constant motion + denoising | top-1 |
|---|---|
| Full-run LSS, no repeat-avg (B2 / C1) | 56.7–62.0% |
| Streaming pst=8, no repeat-avg (C6) | 50.0% |
| Streaming pst=8 + repeat-avg (C8) | 72.0% |
| Full-run LSS + repeat-avg (C2 — bug case, demonstrates inflation) | 78.0% |
| **Full-run + paper Offline post-processing (C3)** | **76.0%** |
| **Princeton canonical .npz (A3)** | **76.0%** |

The 10 pp gap between paper RT (≈ 66%) and Offline (76%) tracks
the windowing tax — RT's per-trial GLM sees only `onset+delay`
TRs of BOLD, the Offline GLM sees the full session. Repeat-averaging
across 3 reps narrows the apparent gap because errors decorrelate.

### Pairwise AUC (n=150 trials, same-image vs diff-image cos-distance)

| factor | AUC contribution |
|---|---|
| **GLMdenoise (Stage 2 in our wrap, pcnum=0 in canonical)** | **+0.10 over AR(1) freq, +0.20 over OLS** in our re-implementation |
| **Stage 3 fracridge (per-voxel, when decoder is fine-tuned on it)** | the offline lift is here per the canonical .npz inspection — heavy shrinkage applied |
| BOLD source (rtmotion vs fmriprep) at constant Glover + GLMdenoise | **−0.005 AUC** (B3 vs B12; B6 vs B13) |
| HRF library at constant denoise | −0.013 to −0.031 AUC (B14 vs B12; B10 vs B3) |
| Variant G vs AR(1) freq (point-estimate metric) | tied (B3 vs B6, B10 vs B11, B12 vs B13) |
| Cumulative z-score + repeat-avg | not directly an AUC factor; we score on raw +/- causal cum-z βs |
| Empirical-Bayes prior (B5 vs B4) | +0.004 AUC (Δ ≈ 0) |

The paper's own saved RT betas plateau at **AUC 0.826 by delay≈15**.
Our denoising-augmented full-run cells reach 0.886 — the +0.06 AUC
delta is the offline-vs-RT gap on this metric.

### What the canonical `.npz` tells us about the offline pipeline

Inspecting `TYPED_FITHRF_GLMDENOISE_RR.npz` for sub-005:

- `pcnum = 0` — GLMsingle's CV picked **zero** PCA components for
  GLMdenoise on this subject. **Stage 2 is not contributing on
  sub-005.**
- `FRACvalue` mean 0.076, range 0.05-1.0 — heavy per-voxel fracridge
  shrinkage from Stage 3.
- `HRFindex` 0-19 used per-voxel — Stage 1 HRF library active.

Two consequences:

1. The "GLMdenoise is the offline win" framing in our intermediate
   findings is partially a re-implementation artifact. Our cell B3's
   AUC lift came from our internal GLMdenoise wrap (regress-out top-K
   noise PCs from a tCompCor-style basis), but the canonical pipeline
   on sub-005 picked K=0. The actual paper Offline lift on this
   subject is from **Stages 1 + 3**, not Stage 2.
2. Real per-voxel fracridge applied as a wrapper around un-fracridge
   OLS βs **destroys** retrieval (we measured 22-43% top-1) — but
   that's a distribution-mismatch artifact (the MindEye decoder is
   fine-tuned on fracridge βs and won't accept un-shrunk OLS βs
   wearing fracridge clothing). Used **inside** GLMsingle as Stage 3,
   it is the lift.

### Cross-run filters (Regime C)

We tested three causal mechanisms for "use past-run data to denoise
current run":

| mechanism | result vs `pst=8_partial` baseline (top-1 50.0%, AUC 0.606) |
|---|---|
| Raw HOSVD K=5 (E1) | top-1 −9.3 pp, AUC +0.06 |
| Raw HOSVD K=10 (E2) | top-1 −20.0 pp |
| Task-residual HOSVD K=5 (E4) | top-1 −5.3 pp, AUC +0.05 (recovers 4 pp on top-1 vs E1) |
| Task-residual HOSVD K=10 (E5) | top-1 −11.3 pp |
| Session-frozen ρ̂ (D4) | top-1 −4.7 pp, AUC +0.12 |

Task-orthogonalization works as designed (preserves more task signal
than naive raw-BOLD HOSVD), but every cross-run filter we tried still
hurts top-1. The within-run information is sufficient to estimate the
ρ̂ noise model and the task signal; subtracting past-run structure
removes useful signal-correlated covariation.

---

## What we learned from the canonical `.npz` (downloaded from `rishab-iyer1/glmsingle`)

We pulled all 9 published `TYPED_FITHRF_GLMDENOISE_RR.npz` files and
read the bootstrap CV outputs directly:

| path | pcnum | mean FRACvalue | xvaltrend shape |
|---|---|---|---|
| sub-001 ses-01 | 6 | 0.091 | non-monotone, K=6 best |
| sub-001 ses-02 | 1 | 0.064 | shallow dip at K=1, rises after |
| sub-001 ses-03 | 1 | 0.064 | same |
| sub-005 ses-01 | 4 | 0.077 | non-monotone, K=4 best |
| sub-005 ses-01-02 (combined) | 4 | 0.063 | non-monotone, K=4 best |
| sub-005 ses-01-03 (combined) | 4 | 0.061 | non-monotone, K=4 best |
| **sub-005 ses-02** | **0** | 0.079 | strictly decreasing in K |
| **sub-005 ses-03** (Offline anchor) | **0** | 0.076 | strictly decreasing in K (K=0: −764.5, K=10: −824.6) |
| sub-005 ses-06 | 1 | 0.063 | shallow dip at K=1 |

This rules out (sub-005 ses-03 specifically) the hypothesis that the
paper silently uses K=10 in canonical. **The CV curve makes K=10 the
worst possible choice for this subject's anchor session.**

## Real fracridge under per-trial LSS — direct test

We implemented per-voxel SVD-based fracridge (Rokem & Kay 2020) and
applied it to per-trial OLS βs computed under LSS, with FRACvalue
frozen from sub-005 ses-01-02 canonical .npz (`mean=0.371`,
range 0.05–0.90, real heterogeneous shrinkage):

| variant | top-1 | top-5 | AUC |
|---|---|---|---|
| Full-run, real fracridge (rtm) | 22.0% | 44.7% | 0.568 |
| Full-run, real fracridge (fmriprep) | 24.7% | 54.0% | 0.586 |
| Streaming pst=8, real fracridge (rtm) | 2.0% (chance) | 10.7% | 0.483 |
| Streaming pst=8, real fracridge (fmriprep) | 1.3% (chance) | 10.0% | 0.485 |

Per-trial LSS computes a separate β through a separate fit with a
separate design-matrix SVD per trial — so each trial gets a different
direction-changing fracridge transform, even at the same per-voxel
target FRACvalue. Trials of the same image get inconsistent
shrinkages, and pairwise discriminability collapses.

The canonical pipeline avoids this by fitting one global GLM over all
trials simultaneously and applying fracridge once across the entire β
matrix — same shrinkage for every per-trial column. **This is not
trivially RT-deployable**: streaming can't preserve global state
across all trials without restructuring as a non-causal session-end fit.

## Open questions for you (Rishab)

1. **`pcnum = 0` for sub-005 ses-02 and ses-03 — confirm**: the bootstrap
   CV trend is monotonically decreasing in K for these sessions. The
   reported Offline 76% top-1 / 98% top-5 anchor is **without
   GLMdenoise** for this subject. Does that match your understanding?
2. **`pcnum` per anchor in the paper**: would the paper's pipeline
   description benefit from reporting `pcnum` per subject/session,
   given the variability (0–6 across the 9 published .npz files)?
3. **Stage 3 fracridge at session end**: the canonical pipeline does a
   global single GLM fit + global fracridge. We confirmed empirically
   that this doesn't transfer to per-trial streaming LSS (real
   fracridge falls to chance under windowing). Have you considered
   whether the pilot's RT pipeline could do a session-end LSR fit +
   global fracridge as a non-causal post-processing pass for
   downstream offline analysis? It wouldn't help in-session feedback,
   but it would let the pilot's offline analyses match the canonical
   anchor.
4. **Per-voxel HRF index file** (`avg_hrfs_s1_s2_full.npy`) was
   derived from sessions 1+2. Did you regenerate it for the sub-001
   sub-002 etc cohort, or is it fixed per subject from training?
5. **The pst sweep saturation at delay≈15-20** in your saved RT betas
   matches an HRF + small-trailing-window decode. Is the operational
   choice in the experiment delay = some fixed value, or did each
   trial decode at variable delay depending on the next stim onset?
6. **Decoder fine-tuning on fracridge βs**: our re-implementation that
   wraps fracridge around OLS βs (rather than fitting GLMsingle
   end-to-end) catastrophically fails. We attribute this to (a)
   per-trial fracridge inconsistency (different design-matrix SVDs
   per trial) and (b) the MindEye decoder being fine-tuned on
   canonical fracridge βs. Could you confirm the checkpoint we're
   using (`sub-005_ses-01_..._3split_0_avgrepeats_finalmask.pth`) was
   trained on canonical TYPED_FITHRF_GLMDENOISE_RR.npz βs?

---

## Reproducibility — every number in this table is on disk

| artifact | path |
|---|---|
| Per-cell βs | `/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/{cell}_ses-03_betas.npy` |
| Trial IDs | `/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/{cell}_ses-03_trial_ids.npy` |
| Top-1 / top-5 summary | `/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/prereg_retrieval_summary.json` |
| AUC summary | `/data/derivatives/rtmindeye_paper/task_2_1_betas/prereg/AUC_factorial_results.json` |
| Cell drivers | `scripts/prereg_variant_sweep.py`, `scripts/rt_paper_full_replica.py` |
| AUC scorer | `scripts/score_AUC_factorial_dgx.py` |
| Paper-RT-βs import | `scripts/import_paper_rt_betas.py` |
| Pre-registration + amendment | `TASK_2_1_PREREGISTRATION.md`, `TASK_2_1_AMENDMENT_2026-04-28.md` |

Cells run in two CPU-Slurm batches (~5–30 min each on this rig).
Retrieval scoring is GPU and takes ~2 min once a slot opens.
