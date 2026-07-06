# Naturalistic-fMRI FM audit — SOTA landscape + proposed protocol

**Origin:** deep-research sweep (2026-07-01, 103 agents, 24/25 claims verified) + the decision
"is processing HBN movie fMRI through our FMs worth pursuing?" **Verdict: not for a SOTA
encoding/decoding leaderboard — yes as the substrate for an FM *audit*.** This doc is the cited
landscape + a concrete, compute-honest protocol.

---

## 1. Verified SOTA (2024–2026)

### Encoding (stimulus → brain) — the flagship
- **Algonauts 2025** switched to naturalistic multimodal movies on **CNeuroMod**: 4 deeply-sampled
  subjects, ~65 h (Friends S1–6 + 4 films), Schaefer-1000 parcels, MNI, ~1.5 s TR, scored by mean
  Pearson r on **out-of-distribution** movies; top-3 must open-source (arXiv:2501.00504,
  2508.10784, 2507.18104; github courtois-neuromod/algonauts_2025.competitors). *[high]*
- **TRIBE** (Meta FAIR, arXiv:2507.22229) **won 1st** — a trimodal (text+audio+video pretrained
  features → transformer) whole-brain encoder. Key lesson: **multimodal fusion beats unimodal
  specifically in high-level associative cortex** (1st-place-by-margin is 2-1/self-reported; top
  teams differed mainly by ensembling). *[high]*
- Naturalistic encoders show a real **ID→OOD drop** (~0.305→0.199 Pearson r) and **complexity does
  not guarantee OOD transfer** (a linear model was more robust OOD than the ID-winning attention
  model, +18% over baseline) — arXiv:2507.18104, 2507.19052. *[high]*

### Decoding (brain → stimulus)
- **Video reconstruction:** diffusion-based — **NeuroClips** (NeurIPS'24, arXiv:2410.19452; +128%
  SSIM / +81% spatiotemporal vs MinD-Video) and **Mind-Animator** (ICLR'25). Author-reported vs
  single baselines. *[high]*
- **Language/semantic:** **Tang et al. 2023** (Nat Neuro, 10.1038/s41593-023-01304-9) — meaning-
  level, cross-modal/amodal; a **2025 Current Biology** successor (10.1016/j.cub.2025.01.054) adds
  **zero-training cross-participant transfer via functional alignment**, working even with
  movie-based (non-linguistic) alignment. *[high]*

### FM representation on *naturalistic* fMRI — thin
- **No verified evidence** that BrainLM / Brain-JEPA / CortexMAE / SwiFT / NeuroSTORM is
  *benchmarked* on naturalistic/movie fMRI. Nuance: CortexMAE's *pretraining* mix
  (arXiv:2510.13768) **includes HCP movie runs** (a fraction; never benchmarked naturalistically).

### The audit literature (validates our angle — and it's in top venues)
- **Spurious/hallucinated reconstruction** (Kamitani, arXiv:2405.10078; Neural Networks 2025):
  "SOTA" image reconstructions are largely classification-into-training-categories + diffusion
  hallucination; collapse on category-non-overlapping splits. *[high]*
- **Cross-subject data leakage** in brain-to-text (EMNLP 2025, arXiv:2312.10987): *all* existing
  cross-subject splits leak, inflating results. *[high]*
- **Third-order statistics beat billion-param FMs** at cognition prediction (arXiv:2606.04010) —
  big fMRI FMs miss simple signal.

---

## 2. Why HBN movies are the right substrate for an *audit* (not a leaderboard)
- **Deep-phenotyping beats breadth** (arXiv:2505.21304): multi-subject training didn't beat
  single-subject in limited data. HBN is **many subjects × short shallow runs = the opposite of
  CNeuroMod** → it *cannot* win encoding/decoding SOTA. That turf is CNeuroMod/TRIBE's.
- **But the same shape is the audit's strength:** many subjects × short runs is *precisely* the
  regime where **subject-identity leakage is most dangerous** — the ideal setting to expose the
  identity trap. Plus HBN carries **age / sex / psychopathology labels** → a concrete, high-value
  question: *is movie-FM "clinical prediction" actually just subject identity?*
- **Movie = time-locked shared stimulus** across all subjects → a clean **(subject × movie-time)
  factorial**, the naturalistic analog of the NSD shared-1000 design (and *better* than rest, which
  has no shared stimulus axis).

**Unclaimed, paper-shaped question:** *Do naturalistic-fMRI foundation models pass the identity-trap
/ weight-spectrum audit?* Extends our CortexMAE-WW/RG + FMScope work into naturalistic territory
where SOTA models exist but **zero critical audit** does.

---

## 3. Proposed protocol (compute-honest; no heavy launch until greenlit)

**Assets in hand:** CortexMAE-volume, NeuroSTORM, SwiFT (all validated on HBN rest); WW spectral
coords for each; FMScope erasure primitives (`fmscope.diagnostics.erasure`); the embed pipeline
(`hbn_full_volume.py` + `cmae_volume_enrich.py`); TRIBEv2 adapter.

**Substrate:** HBN `_scan_movieDM` (537) + `_scan_movieTP` (559), CPAC MNI volumes, on
`/data/raw/hbn-cpac`. Movie = time-locked shared timeline across subjects.

**Prong A — identity-trap (FMScope) on naturalistic FM latents**
1. Embed each subject's movie run through the 3 volume FMs (windowed → per-window latent).
2. Build the **(subject × movie-timepoint) latent tensor** (shared timeline = stimulus axis).
3. **Variance decomposition** → naturalistic **null-subject-variance ratio** (stimulus-time vs
   subject vs residual) — the movie analog of the NSD-betas 0.07 and the EEG 13–89×.
4. **Subject-axis erasure (LEACE):** does erasing subject identity kill inter-subject synchrony /
   movie-timepoint decoding? (survival = how much was identity.)
5. **The clinical question:** age / sex / psychopathology decoding from the movie-FM latent, **pre
   vs post** subject-axis erasure. If it doesn't survive erasure, the "clinical signal" was identity.

**Prong B — weight-spectrum × naturalistic**
6. We already have α / %α<2 / φ₁ / M_tr / traps per FM. Test whether the **α-health→downstream
   ordering** we found on rest (CortexMAE ≫ NeuroSTORM ≫ SwiFT on age) **holds on movie-derived
   downstream** — a rest-vs-naturalistic RG consistency check.
7. **ISC** (inter-subject correlation) of FM latents as a naturalistic quality metric vs spectral
   health.

**Pilot first (small, controlled, checkpointed):** the subjects with **both rest and movieDM** →
a **within-subject rest-vs-naturalistic identity-trap contrast** (same subjects, 3 FMs). Strong,
cheap first result; answers "does naturalistic viewing raise or lower the identity trap vs rest?"

**Compute discipline (lessons learned):** per-subject checkpointing from the start (movie runs are
longer than rest → heavier); detached-docker or harness-workflow execution (host bg jobs get
SIGTERM'd on session teardown); idempotent skip-if-cached.

**Deliverable:** white-paper section / standalone note: *"An identity-trap + spectral audit of
naturalistic-fMRI foundation models (HBN movies)."*

---

## 3b. Lit-check verdict (2026-07-02, focused 105-agent adversarial sweep)
Both target niches **APPARENTLY OPEN** (absence-of-evidence under adversarial search, not proof):
- **CLAIM 1 (HT-SR weight-spectrum of fMRI FMs): OPEN, low collision risk.** Canonical HT-SR is
  CV/NLP only (Martin & Mahoney arXiv:1901.08278/1901.08276; Nat Commun 2021 arXiv:2002.06716;
  FARMS ICML'25 arXiv:2506.06280). No neuro-FM weight-spectrum work. BrainLM (ICLR'24) = prime
  unaudited target.
- **CLAIM 2 (identity-trap audit of fMRI-FM latents): OPEN for fMRI, identity half flanked.** EEG
  FMScope (arXiv:2606.06647) + companion leakage audit (arXiv:2606.09189) are **EEG-only** → open
  fMRI lane. MUST cite/out-position: Finn 2015 fingerprinting (nn.4135), movie-fingerprinting
  (Vanderwal 2017; Wei/Guan 2023 — subjects MORE identifiable during movies), connectome-ML subject
  leakage (Orlichenko 2023 arXiv:2308.01451, 61%→86%; Rosenblatt 2024 Nat Commun).

**Defensible novelty:** two-axis combination ported to a new modality — first HT-SR weight audit +
first FMScope-style identity-trap latent audit of fMRI FMs. **Reviewer risk** on CLAIM 2's identity
half ("fMRI ID is known"): mitigate by framing as FM-latent-specific (frozen reps as the audit
boundary, not raw FC), method-transfer with no fMRI analog, and confound-dominates-clinical-
prediction — NOT "fMRI is identifiable."

**Cost-saving finding:** the identity-trap is demonstrable on RESTING-STATE FM latents too — so the
core result can be built from **cached assets already on disk** (n=1121 rest CortexMAE-vol/NeuroSTORM/
SwiFT latents + WW spectral coords), sidestepping the flaky movie-embed compute. Movies become the
naturalistic *enhancement* (cleaner subject×time factorial, higher identifiability), not a prereq.

## 3c. RESULT — rest FM-latent identity trap, 3 models (2026-07-02, n=399, 8025 windows)
`wwj/benchmarks/{cmae_window_embed,nstorm_swift_window_embed,hbn_fmscope_rest_audit}.py` →
`results/fmscope/rest_fmscope_audit.json`. Per-window latents (~20 win/subj), null-subject-variance
ratio + 399-way subject fingerprint BA.

| Model | objective | null-subj-var ratio | fingerprint BA | ×chance | subj-subspace rank/dim |
|---|---|---|---|---|---|
| CortexMAE-vol | MAE | **12.3×** | 0.939 | 375× | 398/768 |
| NeuroSTORM | MAE+Mamba | **9.3×** | **0.989** | 395× | 288/288 |
| SwiFT | contrastive | **2.1×** | 0.087 | 35× | 288/288 |

**Findings:**
1. **The identity trap is real and severe in fMRI FMs** — CortexMAE & NeuroSTORM latents are
   9–12× identity-dominated and **~94–99% subject-fingerprintable among 399 subjects**, landing in
   the EEG FMScope 13–89× range. Read straight off the frozen pretrained latent (not raw FC).
2. **Objective matters — SwiFT (contrastive) is the outlier**: only 2.1× identity variance and
   fingerprint BA 0.087. Contrastive pretraining produced a markedly **less subject-identity-
   dominated** latent than the two MAE models.
3. **Three-axis cross (identity × downstream × spectrum):** the two most identity-trapped models
   (CortexMAE, NeuroSTORM) are also the **best downstream** (age r 0.808 / 0.706); the least
   identity-trapped (SwiFT) is **worst downstream** (0.593). Spectrally, SwiFT is the over-
   parameterized outlier (α-med ~7.3) while CortexMAE sits in the healthy band (α~2–3). So identity-
   domination, downstream accuracy, and MAE-vs-contrastive objective all separate SwiFT from the MAE
   pair — the on-rest signature that FM downstream skill is entangled with subject-identity encoding
   (the clinical-survival test that would causally confirm this needs the movie/stimulus axis).

## 3d. RESULT — cross-run fingerprint (HBN rest_run-1 → run-2, 2026-07-03, n≈293 preview→400)
`wwj/benchmarks/cross_run_embed.py` + `cross_run_fingerprint.py` → `results/fmscope/cross_run_fingerprint.json`.
Mean-pooled FM latent per (subject, run); Finn-style re-ID + Amico-Goñi I_diff. NOVEL — connectome
fingerprinting is raw-FC (Finn 2015); FM-latent cross-run fingerprint not previously reported (cite
adjacent: CT-embedding demographic leakage arXiv:2412.00110).

| Model | cross-run re-ID | ×chance | I_diff | I_self/I_others |
|---|---|---|---|---|
| CortexMAE | **65.2%** | 191× | 7.3 | 0.984/0.911 |
| NeuroSTORM | 60.4% | 177× | 1.5 | 0.993/0.979 |
| SwiFT | 6.5% | 19× | 0.1 | 0.999/0.998 |

Findings: (1) fMRI FM latents ARE cross-run fingerprints (CortexMAE 65.9%, n=400, 264× chance) —
identity persists across separate rest sessions in the frozen latent. (2) **RECALIBRATED against a
CC200 raw-FC baseline computed ON HBN** (melodic_refine.py, hbn_cc200_fc.npz fc_run1 vs fc_run2, n=753):
raw-FC (Finn method) cross-run re-ID is **65.9%** on HBN (I_diff 20.4) — NOT the ~92-94% Finn reports
on pristine HCP. So CortexMAE's FM latent (65.9%) **exactly matches raw-FC fingerprinting on the same
data** — it inherits the FULL subject-identifiability of the underlying connectome, not less. (The
earlier "below Finn / capacity artifact" caveat is thus resolved: Finn's 92-94% is an HCP ceiling;
HBN's shorter/noisier runs cap raw-FC fingerprinting at ~66%, which the FM latent meets.) (3) The within-session (~0.94, §3c) → cross-run (0.65) drop = the PERMANENCE GAP, same
shape as EEG (EER 3%→10-14% across sessions). (4) SwiFT (contrastive) outlier again — 6.5%, I_diff
0.1, latents ~collapsed (I_others 0.998); MAE-vs-contrastive story consistent across rest (2.1×),
movie (1.2×), and cross-run (I_diff 0.1) analyses.

## 4. Caveats (honest)
- The "unexplored" claim is **low-confidence** (absence-of-evidence, not proof) — do a focused
  lit-check on the exact niche before committing to a paper framing.
- CortexMAE *saw* movie data in pretraining → frame as **benchmarking/auditing**, not "first FM on
  movies."
- HBN won't win encoding SOTA — don't frame the deliverable as an encoding result.
- Headline SOTA metrics cited here are largely author-reported vs single baselines, not independent
  leaderboard audits.
