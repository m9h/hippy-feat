# Handoff: fMRI-FMScope — the identity-trap audit ported to fMRI (NSD multisubject)

**Owner:** outsourced to the **fmri-fm agent**. **Data sourcing:** via the **truenas agent**.
**Origin:** EMEG-FM / NeuroTechX white paper thread (2026-06-29). Feeds white paper §4/§5
(the fMRI-side complement to the EEG FMScope identity-trap audit).

## What it is
fMRI-FMScope = the literal fMRI analog of the EEG **FMScope** audit (arXiv:2606.06647).
FMScope asks whether a frozen FM representation's downstream metric secretly exploits
**subject identity** (a confound) rather than the task/stimulus — via subject-axis erasure
(pooled + per-trial) / variance decomposition. On EEG it found frozen FM reps carry
**13–89× null subject-variance** in 12/12 model-dataset pairs. The fMRI question: **how much
of an fMRI representation's image-decoding is subject identity vs stimulus?**

This is genuinely beyond a covariate GLM (the cross-modal EEG→fMRI cases reduce to one) —
subject identity here is a latent representational subspace + a train/test partition artifact,
exactly FMScope's native regime.

## Why NSD multisubject is the textbook substrate
- 8 subjects; the **shared-1000** images were seen by **ALL 8** → a clean (subject × stimulus)
  factorial. Same stimulus across subjects ⇒ subject-identity is separable from stimulus
  (exactly FMScope's (subject, condition) design; the EEG side had to *pool* conditions, here
  they're matched by construction — a cleaner test).
- **Data already on disk:** `/data/3t/nsd_multisubject/{subj01..08}/betas_session{01..03}.nii.gz`
  + `{subj}_nsdgeneral.nii.gz` masks.

## Representations to audit (in increasing interest)
1. **nsdgeneral-masked betas** (per-trial voxel patterns) — the raw fMRI representation.
2. **MindEye embedding** (ridge/CLIP-aligned) — `/home/mhough/dev/hippy-feat/scripts/mindeye_retrieval_eval.py`,
   `/home/mhough/dev/hippy-feat/scripts/run_mindeye_inference.py`.
3. **fmri-FM latent** (flat-map ViT-MAE, arXiv:2510.13768) — the fmri-fm agent's own model;
   the most interesting target (does the FM latent carry *more or less* subject identity than
   raw betas?).

## Existing scaffold to reuse (hippy-feat)
- `/home/mhough/dev/hippy-feat/scripts/nsd_multisubject_dimest.py` — **already loads** nsdgeneral-masked betas for all 8
  subjects + runs MELODIC/eigenspectrum dimensionality. Loader + representation done; bolt the
  erasure on.
- `/home/mhough/dev/hippy-feat/jaxoccoli/nsd.py` — RSA (`rdm_from_betas`, `compare_rdms`), noise ceiling, category
  selectivity. Cross-subject RSA on shared-1000 is directly relevant.
- `/home/mhough/dev/hippy-feat/scripts/mindeye_retrieval_eval.py` — the image-retrieval metric (the "performance" to audit).

## Method (port FMScope)
`subject_axis_erasure` / the audit cell are re-exported in emeg-fm
(`neurojax.bench.foundation_models` → `emeg_fm`). Steps:
1. On shared-1000: build the (subject × image) representation tensor (betas / embedding / FM latent).
2. **Variance decomposition** — partition representation variance into stimulus (image) vs
   subject vs residual ⇒ the FMScope **null-subject-variance ratio** (the fMRI analog of 13–89×).
3. **Subject-axis erasure** (LEACE; pooled vs per-trial) — remove the subject-identity linear
   subspace, re-measure image-retrieval / cross-subject RSA. How much "performance" was identity?
4. Report the null-subject-variance ratio + the erasure-survival of retrieval, per representation.

## Principled choices (use the ecosystem, don't hardcode) — see `reference_donoho_rmt_wavelet_stack`
- Representation rank → **Gavish–Donoho** (`smni-cmi gavish_donoho_rank` for the PCA/MIGP step;
  Artoni-2018 true-rank for any ICA). `/home/mhough/dev/hippy-feat/scripts/nsd_multisubject_dimest.py` already runs MELODIC dim —
  reconcile MELODIC's estimate with the GD rank.
- Beta/BOLD denoising → **NORDIC** (`/home/mhough/dev/hippy-feat/jaxoccoli/nordic.py`, Marchenko–Pastur).

## Data needs from the truenas agent
- **Design mapping is ALREADY LOCAL:** `/data/3t/data/all_stimuli/nsd_stim_info_merged.csv` has
  the **shared-1000 flag** + per-subject `subjectN_repK` trial indices ⇒ the trial→image mapping
  + shared-image identification is solved *without a download*.
- **MORE BETA SESSIONS are the real truenas need:** only sessions **01–03** are on `/data/3t`
  (~2250 trials/subject), so the shared-1000 images seen by **all 8** subjects *within* those 3
  sessions is a small subset. For a well-populated (subject × shared-image) factorial, supply
  more NSD sessions per subject (ideally the full run, or enough for solid shared-1000
  cross-subject coverage). **First check the shared-image overlap in sessions 01–03** (from the
  stim_info csv) to size the download.
- (Optional) MindEye checkpoints / precomputed fmri-FM embeddings if auditing those reps.

## Deliverable
The fMRI-FMScope null-subject-variance ratio + erasure-survival on NSD shared-1000, per
representation (betas / MindEye / fmri-FM latent) → white paper §4/§5 as the literal fMRI
complement to the EEG identity-trap audit.
