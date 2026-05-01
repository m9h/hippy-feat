# Glossary — terminology pinned to operational definitions

Worked out 2026-04-30 to disambiguate terms used during the Mac/DGX
factorial discussions. Each term gets:
- A precise definition
- A concrete pipeline-level operation
- Cells we've already run that test it
- Where it gets confused with other terms

## GLM-fitting strategies (per-trial vs joint)

### LSS — Least Squares Separate
Per-trial refit. For each trial *i*, build a 4-column design matrix:
`[probe_i, ref_lump, intercept, drift]`. Fit OLS independently. β_i = probe column.
- 70 separate GLM fits per run (one per trial)
- Each trial's β is independent of the others
- **Cells**: all our default cells (`OLS_glover_rtm`, `AR1freq_glover_rtm`, etc.)

### LSA — Least Squares All
Single joint GLM. Build one design matrix per run with N+2 columns:
`[trial_1, trial_2, ..., trial_N, intercept, drift]`. Fit OLS once, β_i = column *i*.
- 1 GLM fit per run, 70 βs come out simultaneously
- β estimates are correlated by design (regressors overlap)
- **Cells**: `OLS_persistentLSA_K0_glover_rtm` (per-run LSA),
  `OLS_persistentLSA_crossrun_K0_glover_rtm` (block-diagonal cross-run LSA)

### Persistent GLM (Ernest Lo's term, ambiguous)
Could mean any of:
1. **LSA per-run** (single GLM that "persists" across all 70 trials of one run)
2. **LSA cross-run with block-diagonal nuisance** (one GLM design, but still per-run intercept/drift)
3. **Incremental/streaming GLM** (single model that updates online as new TRs arrive — recursive least squares, RLS)

(1) and (2) are tested. (3) is the open follow-up — different mechanism, not yet built.

### CONFIRMED via canonical source: persistent GLM is NOT in canonical mindeye.py

After fetching `brainiak/rtcloud-projects/mindeye/scripts/mindeye.py` directly (not in the `--depth 1` clone we had), the canonical RT path is **per-trial LSS**: each trial → `FirstLevelModel(...).fit(...)` from scratch. Zero references to `persistent`, `online`, `recursive`, `RLS`, `streaming GLM`, `Kalman` in the source. So Ernest's "persistent GLM" is HIS proposal/extension, not part of the reference rt-cloud-projects mindeye implementation.

### Incremental / Streaming / Online GLM (untested — Ernest's likely true target)
At each new TR *t*:
- Append a new row to the design matrix
- Update β estimate via recursive least squares (or Kalman update)
- No batch refitting — single growing model
- New trials extend the design with new columns column-by-column
- Result: at end of run, β_i for trial *i* used all BOLD from trial-*i*'s
  onset onward (causal); for cross-run, all BOLD from session start
- **Cells**: NONE BUILT YET

This is what Ernest actually means when he says "stack across runs", almost
certainly. The block-diagonal LSA we tested gave near-identical results to
per-run LSA, so it doesn't capture the cross-run mechanism.

## What "cross-run" can mean

| Meaning | Operation | Cells |
|---|---|---|
| Cross-run noise pool | Single PCA on concat(all 11 runs' BOLD), K components used as confounds across runs | NOT TESTED — our `_extract_noise_components_per_run` is per-run |
| Cross-run AR(1) ρ | Single ρ̂ averaged across runs, frozen for per-trial prewhitening | `HybridOnline_AR1freq_glover_rtm` (cell 17) |
| Cross-run trial regressors | LSA design with all 770 trials in one fit | `OLS_persistentLSA_crossrun_K10_glover_rtm` (block-diagonal) |
| Cross-run nuisance | Per-run intercept/drift COMMON to all runs (no block diagonal) | NOT TESTED |
| Cross-run β prior | Past-trials' β feeds current trial's prior (via Bayes update) | `SameImagePrior_VariantG_glover_rtm` |
| Cross-run state (nuisance) | Single Kalman state spans the session, captures session-wide drift | NOT TESTED |
| Cross-run HOSVD template | Top-K spatial PCs of past-runs BOLD, regressed out of current run | `RT_streaming_pst8_HOSVD_K{5,10}_*` |

Most "cross-run" cells we've tested are **partial** — share one quantity
(noise pool, ρ̂, trial design) across runs but not others. A "fully
cross-run persistent GLM" would share *everything*.

## What "delay" can mean

| Meaning | What changes | Cells |
|---|---|---|
| HRF peak time | Glover (peak ~5s) vs FLOBS (3-basis) vs library (per-voxel canonical) | All Glover cells / Variant B / Variant C |
| Boxcar duration in regressor | Δ at onset_TR vs 1s / 2s / 3s box convolved with HRF | `OLS_K10_dur{0,1,2,3}_glover_rtm` |
| Post-stimulus decode TR | At what TR offset from onset is β extracted | `RT_paper_replica_streaming_pst{4,6,8,10}` |
| Streaming decode window | BOLD cropped to `[onset_TR : onset_TR + pst]` before fit | `RT_paper_replica_full_streaming_pst8` |
| Per-voxel HRF index | GLMsingle library index → per-voxel preferred HRF shape | `AR1freq_glmsingleS1_rtm` (cell 6) |

When Ernest says "stimulus delay improves over longer timespan than BOLD
response", he's specifically talking about the 3rd row: post-stimulus
decode TR. Optimal pst extends past the canonical HRF peak (~5-6s) to
~10+ s post-onset because session-shared evidence keeps accumulating.

## What "real-time / streaming / online / incremental" mean

| Term | Property tested |
|---|---|
| Real-time | Per-volume processing latency ≤ TR (1.5s) — see `bench_rt_mc.py` |
| Streaming | Causal — only past BOLD visible at decode time |
| Online | Model parameters update with new data (recursive) — different from offline batch |
| Incremental | Same as online; sometimes implies design matrix grows row/col |

**Mechanically**:
- `RT_paper_replica_full_streaming_pst8` is **streaming** (causal BOLD crop) but **batch GLM** (per-trial LSS refit).
- The untested "Ernest persistent GLM" would be both **streaming AND online** —
  causal BOLD AND recursive parameter updates.

## Implications for next cell

When I build the "actually-incremental streaming persistent GLM":
- **Streaming**: at each new TR, only data from `[0:t]` visible
- **Online RLS**: β updated via Sherman-Morrison rank-1 update, not batch refit
- **Cross-run pooling**: same model state persists across run boundaries
  (no per-run reset of intercept/drift/noise model)
- **Per-trial β**: each trial's β read off the model at `decode_TR_i = onset_TR_i + pst`

That's a 4-quadrant change vs LSS:

|  | Causal (streaming) | Non-causal (full BOLD)  |
|---|---|---|
| **Per-trial fit (LSS)** | `RT_paper_streaming_pst8` | `OLS_glover_rtm` |
| **Joint fit (LSA / persistent)** | NOT BUILT (← Ernest's target) | `OLS_persistentLSA_K10_glover_rtm` |

The bottom-left quadrant is Ernest's mechanism. We've tested every other
quadrant.
