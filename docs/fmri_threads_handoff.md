# Handoff: all fMRI threads → the fmri-fm agent

**Owner:** the **fmri-fm agent**. **Data sourcing:** the **truenas agent**.
**Origin:** EMEG-FM / NeuroTechX white paper thread (2026-06-29). This session (EEG-FM core)
is handing off the entire fMRI-side apparatus; results feed white paper §5
(`/home/mhough/dev/emeg-fm/docs/neurotechx_dl_eeg_whitepaper.md`).

**Companion brief:** `/home/mhough/dev/hippy-feat/docs/fmri_fmscope_handoff.md` (fMRI-FMScope,
already handed off — the NSD-multisubject subject-axis erasure audit).

**Standing constraints for all of these:** absolute pathnames everywhere; heavy GPU on
`/mnt/t9` + Docker NGC, never `/data` NFS; FSL is installed locally
(`/home/mhough/fsl`); use the ecosystem's principled rank/threshold functions, don't
hardcode (`reference_donoho_rmt_wavelet_stack`: `gavish_donoho_rank` for PCA/MIGP/CCA,
Artoni true-rank for ICA, NORDIC/Marchenko-Pastur for BOLD denoise); don't re-derive
well-known FC findings (Murphy GSR, Smith partial>full, Power motion, NARPS).

---

## ★ Priority — EEG→fMRI PREDICTION models (the user's actual intent)
NOT the cross-subject FC-association work below — **models that predict/synthesize fMRI
BOLD *from* EEG**: **NeuroBOLT** (Neuro-to-BOLD Transformer, 2024/25), **Calhas &
Henriques** EEG→fMRI synthesis (2025 bespoke-NN vs frozen-FM-adapter fork). These
**require SIMULTANEOUS EEG-fMRI data** — HBN is non-simultaneous and CANNOT support them
(this is why the HBN cross-subject work below could only ever be a weak FC association).
**Tasks:** (1) survey the model class (architectures; per-TR BOLD vs network amplitudes vs
whole-volume; eval); (2) find usable **open simultaneous EEG-fMRI datasets** (truenas to
source); (3) the EEG-FM hook — does a frozen **REVE** embedding (`/mnt/t9/reve_hbn_emb.npz`
exists for HBN; would need a simultaneous-cohort extraction) beat a bespoke net at BOLD
prediction? The EEG-FM side is ours to provide; the fMRI + data side is the fmri-fm agent's.

## 1. MELODIC / dual-regression FC backbone (the Oxford default)
Replace the CC200 parcellation FC with the FMRIB approach (user preference,
`feedback_oxford_melodic_fc`): MELODIC group-ICA (MIGP for scale; **MIGP rank →
`gavish_donoho_rank`**, **ICA dim → Artoni true-rank**) → `dual_regression` → FSLnets
**partial-correlation netmats + node amplitudes**. Pull the C-PAC denoised MNI BOLD
(`bandpassed_demeaned_filtered_antswarp.nii.gz`) from FCP-INDI S3 to `/mnt/t9`. Keep CC200
as the cross-method robustness column. Node amplitudes (esp. a vigilance/global IC) are
plausibly the most EEG-coupled feature — and ICA isolating the global IC is the principled
alternative to the GSR fight (our C-PAC FC is non-GSR aCompCor, CC200-only).

## 2. Cross-subject EEG↔FC cases (scaffolded; lower priority per user)
`/home/mhough/dev/emeg-fm/examples/eeg_to_fmri_hbn.py` — 11 ranked cross-subject cases +
primitives. Runner `/home/mhough/dev/emeg-fm/scripts/run_eeg_to_fmri_hbn_case2.py` (case #2
DONE: confound-mediated null; principled GD rank). **NOTE:** the user clarified this
cross-subject-association framing was a misread of "EEG→fMRI" — keep only as robustness
context, not the headline. Case #1 (1/f aperiodic exponent → fALFF/ReHo; NEOBA features at
`/mnt/t9/neoba_hbn_feats.npz`; fALFF/ReHo derived from C-PAC BOLD; ΔR² over
age+motion+sex+site + spin null) is the one with a real physiological prior.

## 3. FC-methods literature sweep (settled vs contested map)
Multi-lens survey: GSR debate + EEG-fMRI vigilance/global-signal coupling; FC estimation
(Smith-2011/FSLnets axis); denoising pipelines (Ciric-2017/XCP axis); atlas dependence
(CC200 vs Brainnetome vs Schaefer); reliability + dynamic-FC caveats; NARPS analytical
flexibility. Deliverable: "settled / contested / canonical-refs / how to position our work"
→ white paper §4/§7. Keeps us off well-known ground.

## 4. Van De Ville fMRI wavelet gaps
We already have a deep wavelet/JTFA stack (`reference_donoho_rmt_wavelet_stack`); the
fMRI-specific gaps to build: **wavelet coherence / wavelet-CPSD → dynamic FC** (extend
`/home/mhough/dev/neurojax/src/neurojax/analysis/multitaper.py` +
`connectivity_spectral.py`); **iCAPs** (innovation-driven co-activation patterns,
Karahanoğlu & Van De Ville 2015); wavelet scattering; innovation/sparsity metrics. Donoho-
Johnstone wavelet shrinkage (`/home/mhough/Workspace/smni-cmi/src/smni_cmi/clean.py`) is
the denoise layer under these.

---

## Data inventory (already local — no re-pull)
- NSD multisubject betas (sessions 01-03): `/data/3t/nsd_multisubject/subj0{1..8}/`
- NSD design/shared-1000: `/data/3t/data/all_stimuli/nsd_stim_info_merged.csv`
- HBN cached: `/mnt/t9/reve_hbn_emb.npz`, `/mnt/t9/neoba_hbn_feats.npz`,
  `/mnt/t9/hbn_cc200_fc.npz`, `/mnt/t9/hbn_reve_ids.txt`, `/mnt/t9/hbn_cohort.txt`
- HBN C-PAC BOLD volumes: FCP-INDI S3 `s3://fcp-indi/data/Projects/HBN/CPAC_preprocessed/`
- Truenas needs: more NSD beta sessions (for fMRI-FMScope shared-image coverage);
  open **simultaneous EEG-fMRI** datasets (for the priority EEG→fMRI-prediction task).
