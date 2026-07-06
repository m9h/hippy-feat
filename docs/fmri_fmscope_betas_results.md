# fMRI-FMScope — Result 1: raw nsdgeneral betas (NSD shared-1000)

**Deliverable:** white-paper §4/§5, the fMRI complement to the EEG FMScope identity-trap audit
(arXiv:2606.06647). Question: *how much of an fMRI representation's structure is subject identity
(a confound) vs stimulus?* This is the **raw-betas** representation (the foundational baseline);
MindEye-embedding and fMRI-FM-latent reps are the higher-interest follow-ups.

**Code:** `scripts/nsd_fmscope_audit.py` · **Result:** `results/fmscope/betas_audit.json`
**Method reuse:** `fmscope.diagnostics.erasure.subject_axis_erasure` — the *exact* LEACE primitive
the EEG side uses (Belrose 2023), so the subject-decodability number is apples-to-apples.

## Substrate  (n=748 — powered re-run, 2026-06-30, NSD sessions 01–20)
- **748 shared-1000 images × 8 subjects**, fully crossed, local NSD sessions 01–20 (truenas pulled
  04–20; see coverage curve). Mean **1.99 reps/image** (494 imgs w/ ≥2 reps → real noise ceiling).
- Betas are **per-subject native functional space** (12.7k–17.9k nsdgeneral voxels, different
  grids) ⇒ no common voxel space → audit identity in **representational geometry (RSA)**.
- *A first pass at n=145 (sessions 01–03) gave the same qualitative story; the n=748 run below
  sharpens every number and is the reportable result.*

## Headline numbers
| Quantity | n=748 | (n=145) | Reading |
|---|---|---|---|
| within-subject rep noise ceiling | **0.423** | 0.483 | reliable signal (1.99 reps; 494 imgs ≥2 rep) |
| between-subject RDM consistency | **0.396**, z=**199** | 0.430, z=47 | shared, stimulus-driven geometry |
| idiosyncratic (subject) geometry | **0.027** | 0.054 | reliable-but-not-shared = subject identity |
| **null-subject-variance ratio** (idio/shared) | **0.07** | 0.12 | stimulus dominates geometry |
| subject-decode BA — **linear** pre-erase | **0.214** (chance 0.125) | 0.053 | weak linear identity |
| subject-decode BA — linear **post-LEACE** | **0.209** | 0.050 | linear erasure removes ~nothing |
| subject-decode BA — **MLP** post-LEACE | **0.870** (chance 0.125) | 0.547 | identity **strongly nonlinear**, survives erasure |

## Two findings (sharpened at n=748)

**1. Raw nsdgeneral betas are stimulus-dominated, not identity-trapped — at the geometry level.**
Between-subject RDM consistency (0.396) reaches **94% of the rep noise ceiling** (0.423), z=199 vs
permutation null — nearly all *reliable* representational geometry is shared across subjects
(natural-image structure), leaving only 0.027 idiosyncratic. The null-subject-variance ratio is
**0.07** — the **opposite regime** to the EEG FM reps, which carried 13–89× *more* subject than
task variance. As a stimulus representation, raw fMRI is clean.

**2. But subject identity is strongly *nonlinear*, and linear erasure barely touches it.** The
linear subject probe is only 0.214 and LEACE's rank-7 linear axis drops it to 0.209 (removes
~nothing) — yet an MLP recovers subject identity at **0.870 BA (7× chance)** from the *same* erased
features. The identity signature the MLP reads is exactly what RSA's correlation metric normalizes
away (subject-specific scale / fine structure), so it is **invisible to standard geometry analysis
and to a linear deconfounder**. This is the FMScope point made concrete for fMRI: **linear
subject-regression deconfounding is insufficient**; the individuating signal (cf. Charest et al.
2014, characteristic individual RDMs) lives in a nonlinear subspace.

## Caveats (honest)
- **1.99 reps** ⇒ noise ceiling 0.42; full 40-session NSD (3 reps) would push it higher. Still, the
  n=145→n=748 trend confirms the findings are robust (both sharpen in the same direction).
- The subject probe uses the per-(subject,image) **RDM-row** feature; the RDM-row construction
  shares each subject's geometry across its rows, so the MLP's absolute 0.87 is representation-
  dependent — the robust claim is the **nonlinear ≫ linear gap and its survival of LEACE**, not the
  exact BA. The clean "how much *retrieval* was identity" number comes from the common-space reps
  (MindEye / fmri-FM latent), still data-blocked.
- Diagonal self-distance encodes *image* (not subject), so subject decoding is not diagonal leakage.

## Result 2 — MindEye/CLIP representation (DONE 2026-07-06, n=225 common shared imgs)
Per-subject betas→CLIP ridge (train non-shared sessions-01–05 trials → predicted CLIP for the 225
shared imgs seen by all 8 subjects), then FMScope on the common-space predicted-CLIP.
`scripts/nsd_fmscope_mindeye.py` → `results/fmscope/mindeye_audit.json`.
- image retrieval top-1: **0.077 pre → 0.078 post subject-axis erasure = 101% survival** (17× chance).
- subject-id BA: **0.562 → 0.081** after LEACE (identity present, cleanly erased); MLP-post 0.182.

**Finding:** the MindEye/CLIP representation *carries* subject identity (linearly decodable, 4.5×
chance) **but its image-retrieval is fully separable from it** — erasing the subject axis leaves
retrieval intact. So MindEye-style decoding is **NOT an identity trap**, in sharp contrast to the
raw MAE FM latents (rest 12×, movie 16× identity-dominated). **Representation determines trap
severity:** a task-aligned CLIP decoder makes identity a removable nuisance; a raw MAE latent makes
it the dominant signal. Caveat: linear ridge on 5 sessions with OpenCLIP ViT-L/14 (retrieval r≈0.29)
— not the full MindEye diffusion model, so survival is on a weak-but-real retrieval baseline.

## Next (the interesting comparison)
Run the same audit on the **MindEye embedding** and the **fMRI-FM latent** (CortexMAE flat-map) —
both common feature spaces, so the full FMScope treatment applies including **image-retrieval
survival after subject-axis erasure**. The key question: *does the learned FM latent amplify or
suppress the subject-identity subspace relative to raw betas?* These need GPU (MindEye inference /
CortexMAE encoder) → deferred until the DGX Spark GPU clears (smri-fm + helmet job running).
