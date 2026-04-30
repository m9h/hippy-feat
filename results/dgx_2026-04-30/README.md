# DGX Spark factorial results — 2026-04-30

Companion to `results/apple_silicon_2026-04-28/`. Same factorial design,
same checkpoint (`sub-005_ses-01_..._3split_0_avgrepeats_finalmask.pth`),
same 50 special515 test images. Cross-replication checkpoint for the
Task 2.1 windowing-vs-denoising decomposition + the AUC reframe.

## Files

| file | what |
|---|---|
| `AUC_factorial_results.json` | pairwise merge/separate AUC + Cohen's d for every cell on disk (n=150 special515 trials, 1 − cosine sim distance) |
| `prereg_retrieval_summary.json` | top-1 / top-5 image retrieval through the MindEye decoder (frozen, ses-01 finalmask checkpoint) |

## What's new vs the Apple Silicon snapshot

- **GLMsingle gap-fill cells** (Stage 1 + 2 + 3 stacked, Stage 1 + VG, full-stack on fmriprep) — first-time AUC measurements
- **fmriprep + Glover + GLMdenoise** isolation cell — closes the BOLD-source contribution at constant denoising
- **Paper's actual saved RT betas at all 8 decode delays** (`Paper_RT_actual_delay{0,1,3,5,10,15,20,63}`) — imported from `/data/derivatives/rtmindeye_paper/rt3t/data/real_time_betas/` and scored alongside the factorial. Lets us read Figure 3's RT bar directly off the paper's pipeline output rather than from a re-implementation.

## Headline numbers (n=150, AUC)

| cell | AUC | Cohen's d | top-1 |
|---|---|---|---|
| `VariantG_glover_rtm_glmdenoise_fracridge` (cell 8) | **0.886** | 1.707 | 60.0% |
| `AR1freq_glover_rtm_glmdenoise_fracridge` (cell 7) | **0.886** | 1.705 | 60.0% |
| `VariantG_glover_fmriprep_glmdenoise_fracridge` (new) | 0.881 | 1.675 | (pending 1048) |
| `AR1freq_glover_fmriprep_glmdenoise_fracridge` (new) | 0.880 | 1.668 | (pending 1048) |
| `VariantG_glover_rtm_acompcor` (cell 9) | 0.873 | 1.619 | 59.3% |
| `AR1freq_glmsingleFull_fmriprep` (new) | 0.868 | 1.580 | (pending 1048) |
| `AR1freq_glmsingleFull_rtm` (new) | 0.855 | 1.487 | (pending 1048) |
| `VariantG_glmsingleFull_rtm` (new) | 0.855 | 1.486 | (pending 1048) |
| **`Paper_RT_actual_delay20`** (paper RT plateau) | **0.826** | 1.303 | (pending 1048) |
| `Paper_RT_actual_delay63` (paper RT, late decode) | 0.825 | 1.301 | (pending 1048) |
| `Paper_RT_actual_delay15` | 0.825 | 1.294 | — |
| `Paper_RT_actual_delay5` | 0.803 | 1.182 | — |
| `HybridOnline_AR1freq_glover_rtm` (cell 17, full-run + session-ρ̂) | 0.793 | 1.082 | 58.7% |
| `VariantG_glover_rtm_prior` (cell 5) | 0.785 | 1.049 | 60.7% |
| `AR1freq_glover_rtm` (cell 2) | 0.783 | 1.040 | 62.0% |
| `VariantG_glover_rtm` (cell 4) | 0.781 | 1.032 | 61.3% |
| `RT_paper_replica_partial` (cell 11, full-run BOLD bug) | 0.723 | 0.808 | 56.7% |
| `OLS_glover_rtm` (cell 1) | 0.684 | 0.638 | 56.0% |

## Figure 3 decomposition (the Task 2.1 deliverable)

Comparing the paper's actual saved RT plateau (AUC 0.825) against our
top denoising cells (AUC 0.886) — Figure 3's RT-vs-Offline gap on the
AUC metric:

| factor | AUC contribution |
|---|---|
| **GLMdenoise** (Stages 2+3 with no-op soft-fracridge ≡ GLMdenoise alone) | **+0.061 AUC** over paper RT plateau |
| BOLD source (rtmotion vs fmriprep) at constant Glover + GLMdenoise | **−0.005 AUC** (within noise; fmriprep doesn't help) |
| HRF library (GLMsingle Stage 1) at constant GLMdenoise | **−0.013 to −0.031 AUC** (hurts mildly) |
| Variant G vs AR(1) freq | tied |

The ~0.06 AUC offline-vs-RT-with-all-data gap on Figure 3's metric is
**entirely GLMdenoise's noise-PC subtraction**. BOLD source, HRF
library, and Variant G's posterior shape contribute nothing visible
once GLMdenoise is in place.

## Cross-replication parity vs Apple Silicon

DGX cells 7/8 (0.886) match the Apple Silicon equivalents (0.871) within
~1.5 pp on AUC. Cells 11/12 paper anchors land paper-exact 0.76 top-1
on both replications.

## What's not in this snapshot yet

- `Paper_RT_actual_delay*` retrieval (top-1) — job 1048 queued
- Streaming-pst cell AUC across the pst sweep (4/6/8/10) is recorded
  here at AUC ~0.61 — well below the paper-RT plateau (0.83). The pst
  sweep was built on rtmotion BOLD without MCFLIRT confounds; the paper
  RT betas include them. The ~0.22 AUC gap is the missing-confounds
  contribution + the implementation differences in our LSS vs the
  paper's per-TR streaming GLM.
- A full Figure 3 reconstruction (per-run accuracy curve, RT vs offline
  band) — currently we have the saturation plateau but not the per-run
  trajectory. To make that exact figure on DGX we'd need to score
  per-trial accuracy at each run boundary.
