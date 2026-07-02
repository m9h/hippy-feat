# Handoff: data sourcing → the truenas agent

**From:** the fmri-fm agent. **Origin:** EMEG-FM / NeuroTechX white-paper thread (2026-06-29).
**Why:** two fMRI workstreams are data-limited; this is the concrete, sized pull list to unblock
them. Companion docs: `docs/eeg_to_fmri_prediction_survey.md` (model survey),
`docs/fmri_fmscope_betas_results.md` (the audit this data sharpens).

**Storage convention (from `fmri_threads_handoff.md`):** heavy raw data → **`/mnt/t9`**, *not*
`/data` NFS (GPU-heavy reads over NFS are slow and hammer the shared mount). New datasets below
land on `/mnt/t9`.

---

## ★ NEED D (CURRENT priority, 2026-07-02) — TUSZ v2.0.3 seizure corpus (epilepsy benchmark)

For the **NeuroTechX-Atlas epilepsy section** ([[project_neurotechx_atlas]]). We already have
**TUEG v2.0.2** (general corpus) at `/mnt/tank/shared/datasets/tuh_eeg/v2.0.2/`, but the epilepsy
benchmark needs the **seizure corpus TUSZ with the `.csv_bi` event annotations** — a *separate*
Temple pull (same ISIP mirror, same/no auth as the v2.0.2 wget). Anchors: the Sci-Rep 2026
"transparent AI assurance" TUSZ benchmark (official Train/Dev/Eval split, continuous-EEG eval,
Event-Sens@FA) + NeuroAtlas's TUSZ arm. **Get** (mirror the existing `download_tuh.sh` pattern):

```bash
cd /data/datasets/tuh_eeg || exit 1        # = NAS /mnt/tank/shared/datasets/tuh_eeg
# --cut-dirs=4 (NOT 5) keeps the `tuh_eeg_seizure/` component so TUSZ doesn't collide
# with the general-corpus v2.0.2/ at the same level; + v2.0.2's retry-hardening flags.
wget -r -l inf -N -c -np -nH --cut-dirs=4 -nv -e robots=off \
  --reject "index.html*" --reject-regex='\?C=' -t 0 --waitretry=15 --read-timeout=120 \
  -a /data/datasets/tuh_eeg/wget_tusz_v2.0.3.log \
  "https://isip.piconepress.com/projects/nedc/data/tuh_eeg/tuh_eeg_seizure/v2.0.3/"
```

Lands at `/data/datasets/tuh_eeg/tuh_eeg_seizure/v2.0.3/` (distinct from general `v2.0.2/`).
~60 GB. Keep the official `edf/{train,dev,eval}/` split intact + the per-recording `.csv_bi`
seizure event files (start/stop/label) — the scorer (`emeg-fm/scripts/epilepsy_scorer.py`) needs
them for the patient-disjoint Event-Sens@FA. If the ISIP mirror now requires the nedc/nedc_resources
credentials (v2.0.2 was open), add `--user nedc --password <pw>` (in Temple's DUA email).

---

## ★ NEED A (priority) — open SIMULTANEOUS EEG-fMRI datasets
For the EEG→fMRI BOLD-prediction priority (NeuroBOLT-class models). HBN EEG is **non-simultaneous**
and cannot train/test these — these datasets are the only substrate. Land under
`/mnt/t9/eeg_fmri/<id>/` (BIDS where available).

| Pri | id | Source | n | get |
|---|---|---|---|---|
| 1 | **NKI NatView** | `s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/` (CC BY 4.0, public, no creds) | 22 | `aws s3 sync --no-sign-request` |
| 2 | **Oddball ds000116** | OpenNeuro `ds000116` | 17 | `openneuro-py download` / datalad |
| 3 | **NODDI** | OSF `osf.io/94c5t` | 16–17 | `osf -p 94c5t clone` |

D2+D3 are the **canonical benchmark pair** (NT-ViT/E2fNet/Calhas report on them) → pulling them
lets us publish directly-comparable numbers. **Phase-1 = these three** (≈ tens of GB; NatView is
the bulk — 5 kHz EEG + 2.1 s fMRI × 22 subj). Scale later if needed: `ds003768` (33, sleep),
`ds006040` (28, task+rest), `ds002336`+`ds002338` (Donoso's neurofeedback data, for his contrast),
EBRAINS TVB-50 (needs free registration), Mendeley `crhybxpdy6` (CC BY 4.0).
*Note:* NeuroBOLT's own Vanderbilt data is **not** openly released → reproduce on NatView + NODDI.

## ✅ NEED B — DELIVERED 2026-06-30 (NSD sessions 04–20, all 8 subjects, on /data/3t)
Sessions **01–20** now local → factorial grew 145→**748 images @ 1.99 reps**; FMScope Result 1
(raw betas) re-run at n=748 (see `fmri_fmscope_betas_results.md`). Below kept for record.

## ★ NEED C (CURRENT priority) — NSD stimulus images, for the MindEye-rep FMScope
The MindEye-embedding representation (FMScope's 2nd, richer target — enables the **image-retrieval
survival after subject-axis erasure** the raw-betas rep can't do) needs the STIMULUS IMAGES to
build CLIP targets for a per-subject betas→CLIP ridge (train on sessions 01–20 unique-image trials,
test on the 748 shared). This is the one remaining data block for FMScope; everything else is coded.
- **Pull:** `s3://natural-scenes-dataset/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5`
  (73000 images, 3×425×425 uint8, ~37 GB, public `--no-sign-request`).
- **Land at:** `/mnt/t9/nsd_stimuli/nsd_stimuli.hdf5` (heavy → /mnt/t9, off NFS; I read it once on
  the GPU to compute OpenCLIP embeddings for the images in sessions 01–20, then cache small `.npz`).
- *Alternative if available:* precomputed NSD OpenCLIP embeddings (skips the 37 GB image pull), but
  the raw hdf5 is the sure thing and lets me pick the CLIP variant.
- **On arrival I run:** betas→CLIP fracridge per subject → predicted-CLIP (subject×748-image common
  space) → `subject_axis_erasure` + retrieval-survival → FMScope Result 2 (MindEye rep).

## NEED B — more NSD beta sessions (fMRI-FMScope factorial + rep noise ceiling)  [historical]
Only sessions **01–03** are local → the shared-1000 factorial is **145 images × 8 subj at 1.39
reps** (rep noise ceiling only 0.48). More sessions grow both the image count and the reps/image.
**Exact coverage curve** (computed from `nsd_stim_info_merged.csv`; size = 8 subj × ~0.55 GB/session):

| sessions | total size | shared imgs (all-8) | imgs w/ 3 reps | mean reps |
|---|---|---|---|---|
| 01–03 (have) | 13 GB | 145 | 12 | 1.39 |
| → 01–10 | 44 GB | 416 | 76 | 1.68 |
| **→ 01–20 (recommended)** | **88 GB** | **748** | **249** | **1.99** |
| → 01–40 (full) | 176 GB | 1000 | 1000 | 3.00 |

**Recommended: pull sessions 04–20 for all 8 subjects** (we already have 01–03) → 748 images, ~2
reps, 249 with a full 3-rep noise ceiling — a 5× bigger factorial and a real ceiling, at 88 GB.
04–10 (+31 GB) is a cheaper first increment if bandwidth is tight; full 01–40 is the ideal.

- **Source (S3, public):** `s3://natural-scenes-dataset/nsddata_betas/ppdata/<subj>/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session<NN>.nii.gz`
  (the local files are exactly this `func1pt8mm` / `b3` product — shape 81×104×83×750 per session).
- **Land at:** `/mnt/t9/nsd_multisubject/<subj>/betas_session<NN>.nii.gz` and copy/symlink the
  existing `<subj>_nsdgeneral.nii.gz` masks there too. (I'll point the FMScope loader's `NSD_DIR`
  at `/mnt/t9` so the GPU learned-rep audit reads off `/mnt/t9`, not NFS.)
- *Subjects vary in completed sessions* (subj04/06/08 fewer than 40); pull what exists per subject.

---

## Priority order (updated 2026-07-01)
0. **NEED C** (`nsd_stimuli.hdf5`, ~37 GB → /mnt/t9) — **current block**: the only thing gating
   FMScope Result 2 (MindEye rep). Small, single file, everything downstream is already coded.
1. **NEED A phase-1** (NatView + ds000116 + NODDI) — unblocks the ★ EEG→fMRI-prediction priority +
   the REVE-vs-bespoke experiment. Highest leverage.
2. **NEED B sessions 04–20** — sharpens the fMRI-FMScope rep ceiling + grows the factorial 145→748.

Ping the fmri-fm agent when A-phase-1 lands (I'll scope the REVE adapter experiment) and when the
NSD sessions land (I'll re-run the FMScope audit at the larger factorial + point the loader at
`/mnt/t9`).
