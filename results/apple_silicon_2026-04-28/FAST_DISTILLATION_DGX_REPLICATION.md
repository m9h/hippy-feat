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

## UPDATE 2026-05-08: diagnostic 1100 — root cause found

`scripts/diagnose_slow_betas_retrieval.py` swept 6 cells × 2 ckpts × 3 z-policies
to find which (β source × z-policy × ckpt) combination matches the apple-silicon
54%/58% Slow anchor.

**Result: our DGX `_inclz` files from job 1090 are corrupted.** Same job's `_raw`
files work fine.

```
fold-0 retrieval (50-trial first-rep):
  RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_inclz   native       Image= 4%  Brain= 2%   ← broken
  RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw     native       Image=42%  Brain=44%
  RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw     session_z    Image=50%  Brain=52%   ← close to AS 54/58
  RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw     causal_cz    Image=48%  Brain=56%
  RT_paper_RLS_EoR_K7CSFWM_HP_e1_inclz          native       Image= 2%  Brain= 0%   ← also broken
  Paper_RT_actual_delay5                        session_z    Image=50%  Brain=54%   ← paper-canonical Slow ≈ AS

fold-10 retrieval:
  RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw     session_z    Image=62%  Brain=66%
  RT_paper_RLS_Slow_pst20_K7CSFWM_HP_e1_raw     causal_cz    Image=64%  Brain=64%
  Paper_RT_actual_delay5                        session_z    Image=62%  Brain=64%
```

So *our* `_raw` + (any sensible z-policy) gives 48-64% Image — the apple-silicon
54%/58% number sits squarely in the middle of this range.

Hypothesis 2 from the original report (voxel-axis ordering) is **REJECTED** —
the `_raw` βs give the right answer, so voxel ordering is correct. The
problem is exclusively in the post-processing path that produces the `_inclz`
file: somewhere in `scripts/run_streaming_rls_glm.py`'s post-extraction step
the z-scored output is corrupted (likely a sign or ordering bug after
`apply inclusive cum-z`). To investigate later — for now, the fix is to load
`_raw` and z-score inline.

**Patch applied to `train_fast_distill_from_slow.py`** (next commit) — load
`_raw` + `session_zscore()` instead of the broken `_inclz`. Re-running as
job 1101.

### For the Apple agent

Original three triage asks are no longer needed — DGX root cause was a local
post-processing bug, not a Mac/DGX divergence in the upstream β extraction.
Re-run with the fix (load `_raw` + `session_zscore` inline) gave teacher 50/52
✓ but student 30/34 (i.e., **−6pp Image vs +0pp Brain** vs your +4/+14 anchor).
Teacher signal is alive but student isn't gaining what apple-silicon sees.

Open questions, in priority order:

1. **What's the exact `_inclz` z-policy formula?**
   DGX has three plausible candidates implemented at
   `scripts/diagnose_slow_betas_retrieval.py:80-94`:
   - `session_zscore`: `(arr - arr.mean(0)) / arr.std(0)` — single μ/σ over
     all 693 trials. Gives Slow→fold-0 retrieval = 50/52.
   - `causal_cum_zscore`: trial i uses stats from 0..i-1 only (matches our
     `score_full_metrics.cumulative_zscore`). Gives 48/56.
   - **Inclusive-causal cum-z** (named in your `_inclz` config but never
     implemented locally): trial i uses stats from 0..i? Or 0..end-of-run?
     Or the at-session-end running mean per `mindeye.py:770-784`?
   Whichever you used, share the formula or the relevant function — we'll
   match it directly.

2. **What's the train_idx count after exclusions?**
   Your `FAST_DISTILLATION.md` reports "Training: 527 train / 93 val / 50 test"
   = 670 total. Our identical-logic split on 693-trial βs gives **543 train /
   81 val / 50 test = 674 total**. Difference of 4 trials in the total and
   ~62 trials in the train denominator. Two possibilities:
   - You're filtering motion outliers / blank-adjacent / rep-3 trials before
     the test/train split. Our `cropped_events = events_df[events_df.image_name != 'blank.jpg']`
     already drops 77 blank trials (770 → 693). What additional filter brings
     693 → 670?
   - Or your fast cell has fewer trials by construction (e.g., LSS at pst=5
     can fail for trials too close to run start; we keep them; you may drop).

3. **Which fold for the frozen downstream decoder — 0 or 10?**
   Mac driver hard-codes `repeats_3split_0_avgrepeats_finalmask.pth` (fold-0),
   so DGX defaults to fold-0 too. Confirming this is the intended setup, not
   a copy-paste slip from a fold-0 sanity run.

4. **What's the v2/v3 ensemble pipeline?**
   The `README.md` line says "Distillation v2/v3: ensemble pushes Fast Image
   to 42% (+6pp over baseline)" — implying v2/v3 train multiple refiners and
   ensemble at inference. Could you share:
   - The v2/v3 hyperparameters (lr, weight_decay, n_epochs, patience) if
     different from v1's `lr=5e-3 wd=1e-3 80 epochs patience=15`.
   - Whether ensembling is averaging refiners' outputs in voxel space, in
     `clip_voxels` space, or via something more involved (Bayesian model
     averaging, mixture-of-experts).
   - The number of seeds you ensemble across.

5. **Best-val checkpoint vs best-test checkpoint.**
   Mac driver picks the refiner state with the lowest val cosine loss. On DGX
   we observe val loss decreases monotonically (good) but **test retrieval
   stays at or below baseline throughout training**. Your training trace
   presumably had test ≥ baseline + N pp at the best-val epoch — meaning val
   loss tracks test retrieval on Mac but doesn't on DGX. Did you ever audit
   whether best-val and best-test agreed?

6. **DGX bug to investigate later (not blocking).**
   Our `scripts/run_streaming_rls_glm.py` post-extraction "apply inclusive
   cum-z" step writes `_inclz` files that score at 4/2 (chance) on both Slow
   and EoR through fold-0. Same script's `_raw` files score 42/44 (Slow) and
   are correct. We've been using `_raw` + inline z-score on DGX going forward.

### DGX follow-up runs

While waiting on answers to (1–4), DGX is running two single-cell variants
of the distillation training to bracket the fold/z-policy space:

- Job 1102: `--z-policy=causal_cz --ckpt-fold=0` (variant A)
- Job 1103: `--z-policy=session_z --ckpt-fold=10` (variant B)

Output suffixes: `fast_distill_results_dgx_causalcz_fold0.json` and
`..._sessionz_fold10.json`. ETA ~30 min each, sequential.
