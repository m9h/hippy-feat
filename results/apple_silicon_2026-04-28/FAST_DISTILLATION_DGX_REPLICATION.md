# Fast distillation — DGX replication finding

DGX port of `drivers/train_fast_distill_from_slow.py` ran successfully but
produced a **chance-level teacher**, not the 54%/58% upper bound the Mac
result reports. Sharing for triage.

## Setup (DGX)

- Script: `scripts/train_fast_distill_from_slow.py` (DGX port)
- Job: 1099 (2026-05-08), GB10 GPU, 1 min training run after container setup
- Fold-0 ckpt: `/data/3t/data/model/sub-005_ses-01_task-C_bs24_MST_rishab_repeats_3split_0_avgrepeats_finalmask.pth`
  (8.6 GB, byte-for-byte identical filename to Mac side)
- Fast student β: `Paper_RT_actual_delay0_ses-03_betas.npy` + `session_zscore()`
  (= Rishab's pre-saved delay=0 LSS, then mu/std normalized over 693 trials —
  matches mindeye.py:770-784 at-session-end semantics)
- Slow teacher β: `RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz_ses-03_betas.npy`
  (job 1090, 2026-05-04, our streaming RLS GLM extraction, Mac-champion port)

## Result — discrepancy

| Config | DGX (1099) | Apple-silicon (anchor) |
|---|---|---|
| baseline (fold-0 on Fast β) | **Image=36.0%  Brain=34.0%** ✓ | Image=36% Brain=34% — exact match |
| **teacher (fold-0 on Slow β)** | **Image= 4.0%  Brain= 2.0%** ✗ | Image=54% Brain=58% — chance vs upper-bound |
| student (Fast β + refiner, best-val) | Image= 4.0% Brain= 6.0% | Image=40% Brain=48% |
| Δ vs baseline | Image=−32 pp Brain=−28 pp | Image=+4 Brain=+14 |

The **fast baseline replicates exactly** (36% Image / 34% Brain), so fold-0
is loading correctly and the Fast input pipeline is sound. But the
**streaming-Slow βs through fold-0 give chance-level retrieval** on DGX.
The student naturally tracks the broken teacher.

## Sanity checks that PASSED

- `fast_b.shape == slow_b.shape == (693, 2792)`
- `(fast_ids == slow_ids).all()` ✓ — row-aligned
- Slow β value distribution: `mean=0.017 std=2.27 range=[-261, 972]` —
  consistent with z-scored βs (no obvious sign flip or scaling issue)
- Slow β config: `{"tier": "Slow", "pst": 20, "z": "inclusive_causal", ...}`
- Voxel space: relmask-applied 2792 voxels, same indexing as fold-0 expects
  (`run_streaming_rls_glm.py:49-53` applies `where(final_mask)[0][relmask]`)

## Hypotheses for triage

1. **Streaming-RLS β extraction differs.** DGX `RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz`
   from `scripts/run_streaming_rls_glm.py` may not produce βs equivalent to
   Mac's `local_drivers/run_streaming_rls_glm.py`. Both follow the
   STREAMING_RLS_GLM.md spec but a subtle implementation difference (HP filter
   order, design-matrix construction, ridge λ, residualization step) could
   produce βs that look distributionally normal but lose the image-content
   signal fold-0 was trained against.
2. **Voxel-axis ordering.** Both sides claim relmask-aligned 2792 voxels, but
   if Mac used `relmask` from a different file or a different boolean indexing
   convention, the 2792 voxels on the two sides could be the same set in a
   permuted order — that would zero out fold-0 retrieval entirely.
3. **K=7+CSFWM+HP+e1 nuisance regressors** on DGX side may absorb task-related
   variance the Mac side leaves intact.

## What the Apple agent could share to break the tie

- A scoring run of `RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz_ses-03_betas.npy`
  through fold-0 on Mac (just baseline retrieval, no refiner) — confirming
  the 54%/58% number for *that exact filename* in their own derivatives.
- A 5-trial slice of the Mac βs to compare voxel-by-voxel against ours
  (or md5sum on the .npy if they're meant to be byte-identical from a synced run).
- The exact ridge λ, HP cutoff, and nuisance-design construction order in
  Mac's streaming RLS GLM script.

## Files committed in this push

- `scripts/train_fast_distill_from_slow.py` (DGX port)
- `scripts/train_fast_distill_from_slow.sbatch`
- `scripts/design_matrix_power_analysis.py` (separate FEAT-style design audit)
- This findings doc

DGX result JSON: `/data/derivatives/rtmindeye_paper/task_2_1_betas/fast_distill_results_dgx.json`
DGX log: `/data/derivatives/rtmindeye_paper/logs/fast-distill-1099.out`
