# Streaming GLM: AR(1) vs OLS ablation + cross-session rtQA

Two extensions to the streaming RLS GLM result:
1. **AR(1) ablation**: disambiguates whether the +12pp Slow gain came from the joint growing-design or from missing AR(1) prewhitening (which the LSS baseline has).
2. **Cross-session rtQA**: extends the rtQA computation to all 4 sub-005 sessions where data is available, in line with OpenNFT's rtQA design.

## 3-way β-extractor comparison (ses-03, fold-0, n=50 special515)

All cells use the same K7+CSFWM+HP+e1 nuisance stack, Glover canonical HRF, inclusive cum-z, rtmotion BOLD. The only difference is the β estimator at decode time.

| Tier | Subset | LSS AR(1) | Stream OLS | Stream AR(1) | Paper |
|---|---|---|---|---|---|
| Fast | s0 (single-rep) | **36** | 24 | 24 | 36 |
| Fast | s1 | 44 | 36 | 36 | — |
| Fast | s2 | 44 | 34 | 32 | — |
| Slow | s0 | 44 | 54 | 52 | — |
| **Slow** | **s1 (paper anchor)** | 56 | **70** | **70** | **58** |
| Slow | s2 | 72 | 72 | 74 | — |
| EoR | s0 | 54 | 54 | 50 | — |
| EoR | s1 (paper anchor) | 66 | **70** | 66 | **66** |
| EoR | s2 | 76 | 78 | 78 | — |

**Reading:**
- **Slow s1 (paper anchor 58%): Stream OLS = Stream AR(1) = 70%. Both gain +14pp over LSS AR(1).** AR(1) prewhitening contributes nothing on top of the growing joint design.
- **EoR s1 (paper anchor 66%): Stream OLS = 70 wins. Stream AR(1) = 66 ties LSS AR(1).** AR(1) actually *hurts* the streaming GLM by 4pp at this subset.
- **Fast s0**: both streaming variants lose to LSS AR(1) by 12pp. Joint design is the wrong tool at short windows; LSS's tight 2-column design is the right model.
- **Subset2 (avg-of-3, complete-set)**: all three converge to 72-78%. Joint design helps slightly (+2-6pp), AR(1) is neutral.

**Why AR(1) prewhitening doesn't help streaming GLM:**

The growing trial design absorbs the relevant temporal autocorrelation through HRF-shaped trial regressors. Once those are in the model, residual autocorrelation is small and well-modeled by the cosine drift + aCompCor structure. AR(1) on top is redundant or counterproductive.

Additionally, the per-voxel ρ̂ values came out *negative* (median -0.42 across all sessions, see rtQA below). Negative AR(1) is an artifact of motion-correction interpolation — not the classical positive-correlation noise model AR(1) is designed for. Applying AR(1) prewhitening with negative ρ̂ in this regime gives no benefit.

**Conclusion**: The +12pp Slow / +4pp EoR gain at the paper's subset1 anchors is attributable entirely to the **joint growing-design**, not to AR(1) noise modeling. Streaming OLS GLM with K7+CSFWM+HP+e1 nuisance is the simplest correct form; adding AR(1) prewhitening adds nothing.

## Cross-session rtQA (4 sub-005 sessions)

Computed per OpenNFT rtQA conventions: tSNR (whole-brain + relmask), DVARS, FD (where motion params available), spike counts, drift coefficient per run, global ρ̂ from session residuals.

| Session | BOLD | tSNR relmask | tSNR brain | DVARS spikes | FD>0.5 spikes | ρ̂ | drift mean |
|---|---|---|---|---|---|---|---|
| ses-01 | fmriprep | **43.76** | 48.66 | 21 | n/a | -0.42 | -4.75 |
| ses-02 | fmriprep | 32.13 | 37.65 | 25 | n/a | -0.48 | -4.63 |
| ses-03 | rtmotion | 39.92 | **48.41** | **40** | **20** | -0.42 | 0.83 |
| ses-06 | fmriprep | 38.60 | 42.70 | 27 | n/a | -0.39 | 3.04 |

**Per-session signal-quality observations:**

- **ses-02 has the lowest tSNR** by ~10 units in both relmask and whole-brain. Would be flagged for re-acquisition in an online dashboard (tSNR < 35 in occipital is borderline).
- **ses-03 has the most DVARS spikes** (40 vs 21-27 elsewhere). Likely related to it being the active retrieval session — subject doing the task may produce more transient signal events.
- **All four sessions show negative ρ̂** (-0.39 to -0.48), pointing to motion-correction interpolation artifact rather than physiological autocorrelation. This is a session-level data property, not a per-session anomaly.
- **Drift signatures differ across sessions**: ses-01/02 strongly negative (initial scanner warm-up?), ses-06 positive (different acquisition state). ses-03 minimal drift, suggesting the test session was acquired under stable scanner conditions.

**FD limitation**: ground-truth motion params (.par files) only available for ses-03 (the only session with rtmotion BOLD pipeline). For ses-01/02/06, motion-derived metrics would require either:
- Running MCFLIRT-equivalent ourselves on fmriprep BOLD (~5 min/session)
- Downloading fmriprep confounds_timeseries.tsv files from HF (not currently local)

Either fix is straightforward if cross-session FD is needed.

## Implications for an experimenter dashboard

In the spirit of OpenNFT rtQA, a real-time dashboard for this paper's pipeline would surface:

1. **Per-TR**: FD, DVARS, current spike count
2. **Per-trial**: tCNR median over relmask, design-matrix conditioning (tr(XᵀX)/K)
3. **Per-run**: tSNR, drift coefficient, completed trials, decoder confidence (cosine sim of top-1)
4. **Cross-run**: tSNR drift over time, ρ̂ stability, session-level spike rate

Alert thresholds we'd recommend based on this analysis:
- tSNR relmask drops > 20% from session start → scanner instability flag
- DVARS spike rate > 2% of TRs → motion-related signal corruption
- FD sustained > 0.5mm over 5+ TRs → subject motion alarm
- ρ̂ deviation > 2σ from baseline → preprocessing pipeline anomaly
- Per-trial decoder confidence below chance for 10+ trials → decoder failure flag

## Files

Drivers:
- `local_drivers/run_streaming_ar1_with_rtqa.py` — single-pass driver, configurable via env vars
- `local_drivers/run_ar1_rtqa_chain.sh` — chain runner across sessions
- `local_drivers/score_streaming_ar1_vs_ols.py` — 3-way scoring
- `local_drivers/refresh_rtqa_with_fd_fix.py` — post-hoc FD column-order fix

Outputs:
- `task_2_1_betas/prereg/RT_paper_RLS_AR1_*.npy` — βs per tier per session
- `task_2_1_betas/rtqa/*.json` — rtQA per cell (per-TR FD/DVARS, summary stats)
- `task_2_1_betas/streaming_ar1_vs_ols_fold0.json` — 3-way scoring results

— Streaming GLM AR(1) ablation + cross-session rtQA, 2026-05-04, fold-0.
