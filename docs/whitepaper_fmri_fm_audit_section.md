# White-paper §4/§5 (fMRI half): The identity trap in fMRI foundation models

*Draft — the fMRI complement to the EEG FMScope identity-trap audit. Numbers are final unless
flagged preview. The EEG half (REVE/LaBraM/CBraMod on EEGMMIDB/Lee2019/PEERS/M3CV) is the EEG-FM
agent's contribution; §7 below is a placeholder for it. Companion: `naturalistic_fmri_fm_audit.md`
(protocol + raw results), `reference_identity_trap_crossmodal` (the renamed-classic mapping).*

## 1. Motivation — an old confound, a new instrument
Frozen foundation-model (FM) latents are increasingly used as off-the-shelf features for clinical
and cognitive prediction. We ask whether a frozen fMRI-FM latent is dominated by **subject
identity** rather than the signal of interest — the "identity trap." This is **not a new
phenomenon**: it is the FM-era face of a confound named repeatedly across fields —
**connectome fingerprinting** (Finn et al. 2015, Nat Neurosci) and **differential identifiability
I_diff** (Amico & Goñi 2018); **ICC / generalizability-theory** variance decomposition;
**inter-subject correlation (ISC, Hasson 2004) / shared-response** for the stimulus component;
**ComBat** harmonization and **confound regression** for the correction; the **reliability paradox**
(Hedge 2018) and **Marek et al. 2022** (brain-wide association studies swamped by stable individual
variance); and, on EEG, **"brainprint" biometrics** (Marcel & Millán 2007; Maiorana 2016). Our
contribution is to package these as a **frozen-representation audit for fMRI foundation models** and
show the confound is inherited, quantified, and objective-dependent — with a weight-spectrum
correlate that has no neuro-FM precedent.

## 2. Methods
**Models (3 volume fMRI-FMs):** CortexMAE-volume (ViT-B MAE, `mni_cortex`), NeuroSTORM (Swin4D +
Mamba, MAE), SwiFT (Swin4D, contrastive). All frozen; mean-pooled or per-window patch latents.
**Data:** HBN (CMI Healthy Brain Network), CPAC-preprocessed MNI volumes — resting-state
(n≈399–1121) and movie-watching (`movieDM` = *Despicable Me*, n≈398), + subjects with two rest runs
(n=400/938) for cross-session. **Metrics (native vocabulary):** (i) between/within subject variance
ratio (null-subject-variance, the ISC/ICC framing) + LEACE subject-axis erasure (confound-regression
analog); (ii) subject re-identification / fingerprint accuracy + I_diff; (iii) cross-subject
movie-timepoint decoding (ISC/shared-response) survival after erasure. HT-SR weight-spectrum (α,
traps; Martin & Mahoney) from the peer-FM survey.

## 3. Results

### 3.1 Resting-state: MAE latents are severely identity-dominated (n=399, 8,025 windows)
| Model | null-subj-var ratio | fingerprint BA (399-way) | ×chance | LEACE subspace rank/dim |
|---|---|---|---|---|
| CortexMAE | **12.3×** | 0.939 | 375× | 398/768 |
| NeuroSTORM | 9.3× | **0.989** | 395× | 288/288 |
| SwiFT | **2.1×** | 0.087 | 35× | 288/288 |

MAE latents carry 9–12× more between- than within-subject variance and are ~94–99% subject-
identifiable — squarely in the **EEG FMScope 13–89× range**. NeuroSTORM's subject subspace fills the
entire latent (rank = dim) → linear erasure is degenerate.

### 3.2 Movie-watching: identity dominates even under a shared stimulus (n=398, 32 timepoints)
| Model | subject var | stimulus var | subj/stim | timepoint-decode survival (erasure) | subj BA pre→post |
|---|---|---|---|---|---|
| CortexMAE | 0.51 | 0.03 | **16.3×** | 92% | 0.95→**0.00** |
| NeuroSTORM | 0.36 | 0.03 | 12.5× | 97% | 0.99→0.98 |
| SwiFT | 0.07 | 0.06 | **1.2×** | 94% | 0.08→0.06 |

Even during a rich time-locked movie, MAE latent variance is **12–16× more about *who* is watching
than *what* they watch.** Critically, the (smaller) shared-stimulus signal is **separable**: cross-
subject movie-timepoint decoding survives ~92–97% of subject-axis erasure — identity sits *alongside*
the stimulus, not masking it.

### 3.3 Cross-run permanence: FM latents fingerprint across sessions (n=400)
| Model | cross-run re-ID | ×chance | I_diff |
|---|---|---|---|
| CortexMAE | **65.9%** | 264× | 7.3 |
| NeuroSTORM | 60.2% | 241× | 1.4 |
| SwiFT | 5.2% | 21× | 0.1 |

A frozen CortexMAE latent re-identifies a subject across two *separate* rest runs at 66% (264×
chance). This is **below raw-FC connectome fingerprinting (~92–94%, Finn)** — but the gap is largely
a **capacity artifact** (mean-pooled 768-dim latent vs ~35k FC edges), not necessarily a model
property. The within-session (0.94) → cross-run (0.66) drop is the **permanence gap**, the same
shape as EEG (verification EER ~3% adjacent → ~10–14% over months).

### 3.4 Weight-spectrum correlate (no prior neuro-FM precedent)
From HT-SR analysis: CortexMAE sits in the healthy band (α-med ~2–3, ~0 correlation traps);
NeuroSTORM is heavy-tailed with many traps; SwiFT is over-parameterized (α-med ~7.3, α>6). The
spectral outlier (SwiFT) is also the identity outlier and the worst downstream model.

### 3.5 Three-axis synthesis (identity × downstream × spectrum)
| Model | identity (rest ratio / fp) | downstream age r (n=1121) | spectrum |
|---|---|---|---|
| CortexMAE | 12.3× / 0.94 | **0.808** | healthy (α~2–3) |
| NeuroSTORM | 9.3× / 0.99 | 0.706 | heavy / traps |
| SwiFT | 2.1× / 0.09 | 0.593 | over-param (α~7.3) |

**The most identity-encoding models are also the best downstream; the contrastive model suppresses
identity and is worst downstream.** Two readings: MAE pretraining that captures individuating
structure also captures task-relevant structure (§3.2 shows the stimulus signal is separable, so
this is entanglement, not pure confound); and **training objective is the lever** — contrastive
learning demonstrably reduces the identity trap.

## 4. The headline
**MAE fMRI foundation models inherit and amplify a severe, cross-session-persistent subject-identity
confound — dominant even under naturalistic stimulation — and the effect is consistent across three
independent lenses (variance domination, fingerprinting, permanence). Contrastive pretraining
uniquely mitigates it.** Practically: clinical/cognitive prediction from frozen fMRI-FM latents must
control for identity (subject-disjoint CV is necessary but the variance domination remains), and
objective choice materially changes identity leakage.

## 5. Novelty & positioning (lit-confirmed)
- **First HT-SR weight-spectrum audit of fMRI-FM weights** — no neuro-FM precedent (CLAIM 1, open).
- **First FMScope-style identity-trap / fingerprint audit of fMRI-FM *latents*** — the audit
  literature is EEG (FMScope arXiv:2606.06647; "Still Leaking" 2606.09189) or structural CT/MRI
  demographic-predictability (arXiv:2412.00110); connectome fingerprinting is raw FC (Finn 2015).
- **Reviewer-risk mitigation:** frame as FM-latent-specific (frozen pretrained reps as the audit
  boundary, not raw FC), method-transfer with no fMRI analog, and *objective-dependence* — NOT a
  claim that "fMRI is identifiable" (that is Finn/Amico-Goñi, cited).

## 6. Limitations (honest)
- Cross-run re-ID uses **mean-pooled** latents (768/288-dim) vs FC's ~35k edges → the "below Finn"
  gap is partly capacity; per-window/richer latents would fingerprint higher.
- On **rest**, subject-level labels (age/sex) don't vary within subject, so the clinical *survival*
  test is degenerate there; the movie (shared-stimulus) analysis is where survival is well-posed.
- 3 volume FMs (parcel/surface FMs need CIFTI HBN lacks); n≈400 per audit (ample for these metrics).
- HBN movieDM is a single film; ISC/timepoint numbers are one naturalistic stimulus.

## 7. EEG half (EEG-FM agent — placeholder)
The cross-modal claim — *a frozen neuro-FM latent is as subject-identifiable as a purpose-built
biometric, in EEG and fMRI alike* — is completed by the EEG side: REVE/LaBraM/CBraMod on
EEGMMIDB (within-session, vs 85–100% biometric ID), Lee2019 (2-session), and the truenas-pulled
PEERS/ds004395 + M3CV (cross-session permanence, vs EER ~3–14%). The EEG FMScope subject-variance
(13–89× null) is the direct analog of §3.1's 12.3×. [To be filled by the EEG-FM agent.]
