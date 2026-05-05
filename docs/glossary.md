# Glossary — RT-MindEye / Task 2.1 / hippy-feat

Definitions of recurring jargon in `docs/task_2_1_for_rishab.md` and
`docs/princeton_pilot_recommendations.md`. Grouped by topic; cross-referenced.

---

## Evaluation metrics

**top-1 (image retrieval, 50-way)**
For each test trial's β: forward through the MindEye decoder
(ridge → backbone → CLIP-token head) to get a predicted CLIP embedding.
Cosine-similarity that prediction against the 50 candidate special515
image embeddings; pick the single highest. `top-1` = fraction of
trials whose top pick matches the stimulus actually shown.
**Chance = 1/50 = 2%.**

**top-5**
Same pipeline, pick the 5 highest-similarity images. `top-5` =
fraction of trials whose true stimulus is among those 5.
**Chance = 5/50 = 10%.**

**Pairwise AUC** (merge/separate AUC)
Closed-loop neurofeedback metric. For each pair of trials, compute the
cosine distance between their βs. Two distributions:
- **same-image**: pairs of trials where the same stimulus was shown
- **diff-image**: pairs with different stimuli

`AUC` = probability that a randomly drawn diff-image pair has greater
distance than a randomly drawn same-image pair. **Chance = 0.5.**
Equivalent to the Mann-Whitney U statistic between same and diff
distributions.

**Cohen's d** (on the same pairwise distance distributions)
`(mean(diff) − mean(same)) / pooled_sd`. Effect-size measure of how
separable same-image and diff-image distance distributions are.
Roughly: 0.2 = small, 0.5 = medium, 0.8+ = large.

**Selective accuracy at τ**
For confidence-gated decoding using Variant G's posterior `(β_mean,
β_var)`. Trial-level SNR = `|β_mean| / √β_var`. Threshold τ on SNR;
report `accuracy` and `coverage`:
- `coverage` = fraction of trials with SNR > τ
- `selective accuracy` = retrieval accuracy on that subset

Higher τ → fewer trials decoded but each at higher confidence.

**Brier score**
Mean squared error of predicted-class-probability against the
one-hot true label. Lower = better-calibrated. Range [0, 2] for K-way
classification (Brier ≤ 0.5 is decent).

**ECE (Expected Calibration Error)**
Weighted gap between predicted probability and observed accuracy
across confidence bins. Lower = better-calibrated. Range [0, 1].

---

## GLM mechanics

**β / beta**
Regression coefficient for a single trial's stimulus regressor. The
(V-voxel) vector of beta values *is* the trial's neural signature
fed to the decoder. Everything in our pipeline is "estimate this β
better".

**LSS (Least Squares Separate)**
Per-trial GLM design: each trial gets its own GLM fit where one
event is the "probe" regressor and all other events are a single
"reference" regressor. Repeated for every trial. Used by paper RT.

**LSR (Least Squares Refit / Reduced)**
Single-fit design: every trial gets its own column in a single global
design matrix; the fit produces all per-trial βs at once. Used by
canonical GLMsingle. Lets fracridge apply the same shrinkage transform
across all trials simultaneously.

**OLS**
Ordinary Least Squares — `(X'X)^-1 X'Y`. No noise assumption beyond
i.i.d. Gaussian.

**AR(1) prewhitening**
Models the BOLD residual as a first-order autoregressive process:
`ε_t = ρ ε_{t-1} + η_t`. Estimates `ρ` from residuals, prewhitens
both `X` and `Y` by the operator `I − ρL` (where `L` is the lag),
then refits OLS on prewhitened data. Closer to GLS for typical fMRI
noise. Used by both nilearn `noise_model='ar1'` and our own
`_variant_g_forward` (with weak prior on ρ).

**ρ̂ (rho-hat)**
The estimated AR(1) coefficient. Per-voxel scalar in [-1, 1].

**Variant G (VG)**
AR(1)-prewhitened **Bayesian conjugate** GLM. Closed-form
Normal-Inverse-Gamma posterior over (β, σ²) per voxel given prior
hyperparameters (prior_mean, prior_var, a, b). Output: per-trial
`(β_mean, β_var)` posterior. Same forward-pass cost as AR(1) freq
(1.6–4.8 ms/TR JIT'd on GB10).

**HRF (hemodynamic response function)**
The temporal kernel mapping neural activity to BOLD signal. Stimulus
regressor is convolved with HRF before fitting.
- **Glover canonical**: double-gamma kernel from Glover (1999),
  peak ~5 s, undershoot ~12 s.
- **GLMsingle library**: 20 candidate HRFs derived from a normative
  population. Each voxel picks the best-fit HRF index from this
  library. Per-voxel index file: `avg_hrfs_s1_s2_full.npy`.

**Drift regressors**
Slowly-varying nuisance terms in the design matrix. Cosine basis
(`drift_model='cosine'`) at high-pass cutoff 0.01 Hz is standard.

**aCompCor**
Anatomical CompCor: 5 PCs extracted from a noise pool defined by
WM/CSF tissue masks (from FreeSurfer or FAST segmentation). Used as
nuisance regressors. Captures physiological / drift noise without
removing task signal.

**tCompCor**
Temporal CompCor: same idea, but the noise pool is defined as the
top-N-percentile-variance voxels (no tissue mask needed). Riskier
because high-variance voxels can include task-driven voxels.

---

## GLMsingle stages

**Stage 1 — Per-voxel HRF library**
For each voxel, pick the HRF (from the 20-element library) that gives
the highest single-trial GLM `R²`. Stored as `HRFindex` (an integer
0–19 per voxel).

**Stage 2 — GLMdenoise**
Run a leave-one-out cross-validation over candidate K = 0..10
GLMdenoise PCA components extracted from a noise pool. Pick the K
that maximizes a CV criterion. Apply that K as additional nuisance
regressors in a refit. Stored as scalar `pcnum` per pipeline run.
**For sub-005 ses-03 the canonical pipeline picked `pcnum=0`** —
GLMdenoise contributes nothing on this anchor session.

**Stage 3 — Per-voxel SVD fracridge**
For each voxel, ridge-regress with regularization `λ_v` chosen so
that `‖β_ridge(λ_v)‖ / ‖β_OLS‖ = FRACvalue[v]`. Stored as `FRACvalue`
per voxel (value in [0.05, 1.0]; mean ~0.1 across the dataset means
heavy shrinkage). **Direction-changing operation**: NOT equivalent to
`β *= FRACvalue` (which is a no-op on cosine distances). Requires
global single fit + global fracridge to apply consistently across all
trial βs simultaneously.

**TYPED_FITHRF_GLMDENOISE_RR**
GLMsingle's "Type D" output struct (a `.npz` per pipeline run):
HRF library + GLMdenoise + ridge regression. Contains `betasmd`
(per-trial βs), `HRFindex`, `pcnum`, `FRACvalue`, `xvaltrend` (the CV
curve over K=0..10), and several diagnostic fields.

---

## Masks and voxel spaces

**finalmask** (`sub-005_final_mask.nii.gz`)
The brain mask used as the *outer* envelope for the MindEye decoder.
19174 voxels, defined in T1w-aligned BOLD-reference space.

**relmask** (`sub-005_ses-01_task-C_relmask.npy`)
A bool selector inside finalmask (length 19174, true count 2792)
that picks the voxels actually fed to the MindEye decoder. **2792 is
the input dimension of the MindEye ridge layer** for the
`...finalmask.pth` checkpoints.

**unionmask** (`union_mask_from_ses-01-02.npy`)
Larger mask (8627 True voxels) — used by a different family of
checkpoints (e.g., `..._unionmask_finetune.pth`). Not what we use.

**fmriprep BOLD**
`_space-T1w_desc-preproc_bold.nii.gz`. fmriprep's full-pipeline
output: motion corrected + slice-timing-corrected + susceptibility-
distortion-corrected + co-registered to T1w. Available offline only
(needs the T1w + fieldmap + full session BOLD).

**rtmotion BOLD**
`motion_corrected_resampled/` — paper's RT pipeline output: per-TR
MCFLIRT against `rt_vol0` + flirt-based cross-session registration
to fmriprep's T1w-aligned reference. Available in real time.

---

## Paper latency tiers (Iyer et al. ICML 2026 Table 1)

The paper distinguishes **four** decode-latency operating points on the
same MindEye2 architecture, all on n=50 single-trial first-rep eval:

| Tier | Latency | Paper top-1 | Maps to our `Paper_RT_actual_delay` |
|---|---|---|---|
| Offline 3T (avg 3 reps) | 1 day | **90%** | — (offline post-processing tier) |
| Offline 3T | 1 day | **76%** | Canonical `.npz` (n=50 first-rep) |
| End-of-run RT | 2.7 min | **66%** | `delay=63` |
| Slow RT | 36 s | **58%** | `delay=20` (we measure 59.3% — within 1.3 pp) |
| Fast RT | 14.5 s | **36%** | `delay=5` |

**Stimulus delay** in this taxonomy = wall-clock seconds elapsed after
stimulus onset before the GLM is fit. ~7.9 s = ~5 TRs at TR = 1.5 s.

The "10 pp gap" the Discussion focuses on is **Offline 3T 76% − End-of-Run RT 66%**.
Both have all the within-run BOLD; the difference is
fmriprep + GLMsingle vs rtmotion + nilearn AR(1) LSS at constant
within-run data availability.

---

## Stimuli, trials, sessions

**special515**
Set of 515 NSD images selected by the paper for repeated
presentation. 50 of these appear in ses-03 (3 reps each = 150 trials).

**MST_pairs**
Memory-set-test image pairs (similar images presented for
same/different judgement). Used for behavioral 1-back task.

**unchosen_nsd_1000_images**
Filler images for non-test trials. The other ~620 trials per session.

**ses-01 / ses-02 / ses-03**
sub-005's three task-C sessions. ses-01 + ses-02 = training data for
decoder + GLMsingle library. **ses-03 is the held-out test session
that anchors the paper's reported retrieval numbers.**

**ses-06**
The RT-pilot session — different stimulus set, behavioral component.
Out of scope for Task 2.1.

**Trial count notes**
- 770 total events per session (from BIDS events.tsv)
- 693 non-blank trials (after dropping `blank.jpg` rows)
- 150 special515 trials in ses-03 (50 unique × 3 reps)
- 50 unique-image trials after repeat-averaging

---

## Windowing & post-processing

**pst (post-stim TRs)**
The decode delay used in the streaming GLM: each trial's β is fit on
BOLD cropped to `onset_TR + pst`. `pst=8` is the saturation point
in the within-run pst sweep (HRF peak + small trailing window).

**cum-z (causal cumulative z-score)**
Per-trial z-score where trial *i* uses statistics (mean, std) from
trials 0..*i*−1 only. Strictly causal — no test-set leakage from
future trials. Replaces session-level z-score in the paper's RT path.
Implemented in `cumulative_zscore_with_optional_repeat_avg` in
`scripts/rt_paper_full_replica.py`.

**repeat-avg**
After all 3 reps of an image have been observed, average their
post-cum-z βs into a single per-image β. Reduces `n_test_trials` from
150 to 50 and improves accuracy by averaging out trial-level noise.

**delay (paper RT saved betas)**
Rishab's published `all_betas_ses-03_all_runs_delay{N}.npy` files use
delay = 0, 1, 3, 5, 10, 15, 20, 63 — TRs after onset at which the
GLM was refit. **delay=20 saturates at AUC 0.826** as the paper RT
plateau.

---

## Pipeline stages we tested (cross-run filters)

**HOSVD**
Higher-Order SVD. Used as a **cross-run causal filter**: top-K
spatial PCs extracted from concatenated past-run BOLD, projected onto
current run as additional nuisance regressors. Tested with K=5 and
K=10 — both **hurt** retrieval (the top spatial PCs include task
signal).

**ResidHOSVD** (task-orthogonal HOSVD)
Same construction, but applied to past-run **GLM residuals** (after
subtracting the canonical Glover fit) instead of raw BOLD. Removes
the task subspace before SVD. Recovers some of the loss vs raw
HOSVD but **still hurts** vs the within-run baseline.

**HybridOnline (session-frozen ρ̂)**
AR(1) noise model where ρ̂ is computed once per voxel from the full
session's BOLD with intercept+drift-only design, then frozen and
applied to every trial's GLM. Cell 17 in our re-implementation.
**Hurts** vs per-trial AR(1) freq when used with per-trial design
matrices that have additional regressors.

---

## Architecture / amendment terminology

**Regime A** (offline)
GLM fit on the full session's BOLD per trial. Non-causal, not
RT-deployable. Paper Offline anchor.

**Regime B** (within-run streaming)
GLM fit on BOLD cropped to onset+pst per trial. Causal, RT-deployable.
Paper RT pipeline operates here.

**Regime C** (cross-run streaming)
Streaming pst + filters that draw on past-run information (HOSVD,
session-ρ̂, etc.). Causal, RT-deployable. **The actually-deployable
goal**, but every cross-run mechanism we tested in this regime hurt.

---

## Acronyms quick-ref

| acronym | full |
|---|---|
| AUC | Area Under the ROC Curve (here: pairwise merge/separate) |
| AR(1) | First-order autoregressive |
| BOLD | Blood-Oxygen-Level-Dependent (signal) |
| CV | Cross-validation |
| ECE | Expected Calibration Error |
| EPI | Echo-Planar Imaging |
| FAST | FSL's tissue segmentation tool |
| GLM | General Linear Model |
| HOSVD | Higher-Order Singular Value Decomposition |
| HRF | Hemodynamic Response Function |
| LSR | Least Squares Refit/Reduced |
| LSS | Least Squares Separate |
| MCFLIRT | FSL's per-volume motion correction tool |
| OFC | Orbito-frontal cortex |
| PVE | Partial-volume estimate (FAST tissue probability map) |
| pst | Post-stim TRs |
| ρ̂ (rho-hat) | Estimated AR(1) coefficient |
| SDC | Susceptibility Distortion Correction (fieldmap-based) |
| SNR | Signal-to-Noise Ratio |
| SVD | Singular Value Decomposition |
| TR | Repetition Time (volume acquisition interval, 1.5 s here) |
| VG | Variant G (Bayesian conjugate AR(1) GLM) |
