# EEG → fMRI BOLD Prediction: Model-Class Survey (NeuroTechX White Paper §5)

**Scope.** This survey covers models that **predict / synthesize fMRI BOLD signal *from* EEG** (cross-modal generation), not cross-subject EEG↔fMRI functional-connectivity association. Every model in this class requires **simultaneous (concurrent) EEG-fMRI** recordings for training and validation. The user's HBN EEG is non-simultaneous and therefore cannot train or test these models, so §2 is a concrete shortlist of open simultaneous EEG-fMRI datasets to source.

**Key terminology note / attribution correction.** The task brief attributed the "bespoke net vs frozen-foundation-model-adapter" contrast to *Calhas & Henriques*. In fact two distinct lines exist and should not be conflated:
- **Calhas & Henriques** = the **bespoke neural-transcoding** line (Fourier/attentional-graph autoencoders → whole fMRI volume). See models C, and their data benchmarks (NODDI/Oddball).
- The explicit **"two strategies: bespoke NN vs frozen foundation-model adapter"** paper is **Donoso (2025), Frontiers in Systems Biology** (model I) — and the frozen models it adapts are generic LLMs/VLMs (Gemma, Llama, PaliGemma), not an EEG-FM. This distinction matters for the REVE hook (§3): NeuroBOLT, not Donoso, is the real precedent for "frozen *EEG* foundation model + adapter."

---

## 1. Model-class table

| # | Model | Year / venue | ID | Architecture (1-line) | **What it predicts** | Train data (n / EEG / TR) | Headline result | Code |
|---|-------|-------------|----|------------------------|----------------------|----------------------------|-----------------|------|
| A | **NeuroBOLT** | 2024, NeurIPS | arXiv:2410.05341 | Frozen **LaBraM-base** EEG-FM (spatiotemporal) + multi-scale FFT spectral module + MLP head; sequence-to-one | **Per-ROI, per-TR scalar BOLD** (7 ROIs: cuneus, Heschl's, precuneus, frontal gyri, putamen, thalamus + global); DiFuMo-64 | 22 subj / 29 scans, 26 ch, TR 2.1 s, 3T, eyes-closed rest 20 min (+auditory task) | Pearson **r = 0.531 intra**, **0.473 inter**-subject; zero-shot→task r=0.38 (0.42 fine-tuned) | [soupeeli/NeuroBOLT](https://github.com/soupeeli/NeuroBOLT) |
| B | **E2fNet** ("Brainwaves→Brain Scans") | 2025 | arXiv:2502.08025 | EEG-spectrogram encoder + U-Net + fMRI decoder | **Whole 3D fMRI volume** (e.g. 30×64×64) | NODDI(15)+Oddball(17)+CN-EPFL(20) | SSIM **0.605 / 0.631 / 0.674** — beats NT-ViT on SSIM, simpler | [kgr20/E2fNet](https://github.com/kgr20/E2fNet) |
| C | **Calhas & Henriques** (attentional-graph synth) | 2022 arXiv / 2023 MLHC; 2025 follow-up | arXiv:2203.03481; arXiv:2504.10752 | Fourier features + attentional graph of electrode relations + shared latent + style transfer (autoencoder) | **Whole 3D fMRI volume** | NODDI, Oddball, CN-EPFL | RMSE/SSIM "SOTA at time"; 2025 paper addresses statistical significance & generalizability | [DCalhas/eeg_to_fmri](https://github.com/DCalhas/eeg_to_fmri) (MIT) |
| D | **NT-ViT** | 2024, ICIAP (Springer) | arXiv:2409.11836 | Generator (EEG→Mel-spectrogram→ViT enc-dec) + Domain-Matching module (train-only fMRI latent alignment) | **Whole 3D fMRI volume** | NODDI (15) + Oddball (17) | RMSE ↓ up to ~10×, SSIM ↑ up to ~3× vs prior SOTA | [rom42pla/ntvit](https://github.com/rom42pla/ntvit) |
| E | **CATD** | 2024 | arXiv:2408.00777 | Latent **diffusion** (DiT) + condition-aligned block + dynamic time-freq segmentation + VAE | **Whole-brain BOLD** + 3× temporal super-resolution | Motor-imagery (10) + NODDI (17) + Parkinson's (8), 64 ch | SSIM > 0.67, CCC ≈ 0.8, RMSE < 0.1; downstream classif. gains | none found |
| F | **Spec2VolCAMU-Net** | 2025 | arXiv:2505.09521 | Multi-directional time-freq conv-attention encoder + **Vision-Mamba U-Net** | **Whole 3D fMRI volume** | NODDI / Oddball / CN-EPFL | Reports SOTA SSIM on the three benchmarks | repo referenced in paper |
| G | **SIREN-EEG→fMRI** | 2023, SPIE Med. Imaging | arXiv:2311.04234 | Sinusoidal representation network (implicit neural rep.) + encoder-decoder | fMRI signal (no explicit feature engineering) | simultaneous EEG-fMRI (proof-of-concept) | preliminary feasibility | n/a |
| H | **Liu & Sajda — hierarchical deep transcoding** | 2023, Brain Informatics | DOI:10.1007/978-3-031-43075-6_6; arXiv:2010.02167, arXiv:2212.02226 | Cyclic CNN transcoder, **bidirectional EEG↔fMRI** via shared latent neural source space | High spatiotemporal-resolution **latent source space** (and reconstructed fMRI) | NODDI / Oddball | recovers subcortical/latent sources; qualitative + correlation | partial |
| I | **Donoso — "two strategies"** | 2025, Front. Syst. Biol. | DOI:10.3389/fsysb.2025.1715692; bioRxiv:2025.05.06.652346 | (A) bespoke MLP/CNN/RNN/Transformer from scratch vs (B) **frozen LLM/VLM adapters** (Gemma-2-2B, Llama-3.2-3B, PaliGemma2-3B) | **Harvard-Oxford 49 cortical ROI** activity | ds002336 neurofeedback, 9 subj, 64 ch | Classif. ~53–54%; frozen LLMs gave only marginal gains, deep nets failed regression | Jupyter notebooks (suppl.) |

---

## 1b. Per-model notes

**A. NeuroBOLT** (arXiv:2410.05341, NeurIPS 2024; project page https://soupeeli.github.io/NeuroBOLT, code https://github.com/soupeeli/NeuroBOLT). The most directly relevant model and the strongest precedent for the REVE hook. It is **not** a from-scratch net: its spatiotemporal stream **is a frozen/pretrained EEG foundation model (LaBraM-base)**, patch-tokenized like a ViT, fused with a *trainable* multi-scale FFT spectral module and a small projection head. It is a **sequence-to-one regressor**: a ~16 s EEG window (chosen to approximate HRF duration, so no fixed hemodynamic delay is assumed) → one scalar BOLD value per ROI per TR. Predicts 7 representative ROIs spanning primary sensory (cuneus, Heschl's), high-level cognitive (precuneus, frontal), deep subcortical (putamen, **thalamus**), plus global signal, under a DiFuMo-64 parcellation. Trained on 22 subjects / 29 scans (eyes-closed rest, 20 min, 26 EEG ch, TR 2.1 s, 3T Vanderbilt) plus a separate auditory-task set. Beats from-scratch BIOT and even outperforms its own LaBraM backbone alone — i.e. **"frozen EEG-FM + trainable adapter" already beats a bespoke from-scratch transformer in this exact task.**

**B. E2fNet** (arXiv:2502.08025; Roos, Fukuda, Cap; code https://github.com/kgr20/E2fNet). Encoder–U-Net–decoder producing a **full 3D volume per sample**. Notable because it is the cleanest **3-benchmark comparison** (NODDI/Oddball/CN-EPFL) and currently the **best published SSIM** on all three, beating the heavier NT-ViT with a simpler design. Good baseline/target for whole-volume prediction.

**C. Calhas & Henriques** (arXiv:2203.03481, 2022; MLHC 2023; follow-up arXiv:2504.10752, 2025; code https://github.com/DCalhas/eeg_to_fmri, MIT). The original **bespoke neural-transcoding** line: an autoencoder using Fourier features, a learned **attentional graph over electrode relationships**, a cross-modal shared latent space, and style injection, to synthesize the **whole fMRI volume**. The 2025 follow-up explicitly tackles statistical-significance and generalizability concerns of EEG→fMRI synthesis (a recurring critique of the whole-volume papers — high SSIM can be driven by anatomical priors rather than genuine moment-to-moment BOLD prediction).

**D. NT-ViT** (arXiv:2409.11836; Lanzino et al., Sapienza/KTH; code https://github.com/rom42pla/ntvit). EEG→Mel-spectrogram→ViT generator with a **Domain-Matching** module that, at train time only, aligns the EEG latent with an fMRI-volume latent. Large RMSE/SSIM gains over prior whole-volume SOTA on NODDI and Oddball; superseded on SSIM by E2fNet.

**E. CATD** (arXiv:2408.00777; Yao et al.). A **latent diffusion (DiT)** approach that additionally claims **3× temporal super-resolution** of BOLD and uses the generated BOLD for downstream classification (motor imagery, Parkinson's). Represents the generative-diffusion branch of the field. No public code found.

**F. Spec2VolCAMU-Net** (arXiv:2505.09521). Spectrogram-to-volume with a Vision-Mamba U-Net backbone; another whole-volume model benchmarked on the canonical NODDI/Oddball/CN-EPFL triple — useful as a recent SSOTA reference point.

**G. SIREN→fMRI** (arXiv:2311.04234, SPIE 2023). Early implicit-neural-representation proof of concept; predicts fMRI signal from EEG without hand-engineered features. Mostly of historical/architectural interest.

**H. Liu & Sajda — deep transcoding** (Brain Informatics 2023, DOI:10.1007/978-3-031-43075-6_6; arXiv:2010.02167, arXiv:2212.02226). A **bidirectional cyclic CNN transcoder** that maps EEG↔fMRI through a shared high-resolution latent "neural source space"; emphasizes recovery of **subcortical/latent sources** rather than raw volume SSIM. The conceptual ancestor of the modern synthesis work and the canonical reference for "transcoding."

**I. Donoso — two strategies** (Front. Syst. Biol. 2025, DOI:10.3389/fsysb.2025.1715692; bioRxiv:2025.05.06.652346). Directly compares (A) bespoke MLP/CNN/RNN/Transformer trained from scratch against (B) **frozen general-purpose foundation models** (Gemma-2-2B-IT, Llama-3.2-3B-Instruct, PaliGemma2-3B) with a light readout, predicting Harvard-Oxford 49-ROI activity on the ds002336 neurofeedback set (9 subj, 64 ch). **Finding: frozen generic LLM/VLM adapters gave only marginal gains and deep nets failed at regression** — i.e. *generic* foundation models do not transfer well to BOLD. This is the cautionary counterpoint to the REVE hypothesis and the reason a *domain-specific EEG-FM* (REVE), not a generic LLM, is the right bet (§3).

**Field-level caveat.** Two output granularities dominate and are not directly comparable: **(i) per-ROI/per-TR timeseries regression** (NeuroBOLT, Donoso) reported as Pearson *r* (~0.4–0.6), vs **(ii) whole-volume synthesis** (Calhas, NT-ViT, E2fNet, CATD, Spec2Vol) reported as SSIM/PSNR/RMSE. High SSIM in (ii) can reflect anatomical priors more than genuine dynamic BOLD prediction — a point the 2025 Calhas paper explicitly raises. A serious benchmark should report *r* on held-out subjects, not just SSIM.

---

## 2. Open simultaneous EEG-fMRI dataset shortlist (truenas download list)

All are openly downloadable (some require free registration/DUA). Concrete accessions and access paths below.

| # | Dataset | Repository / URL | n | Paradigm | EEG | TR | License / access |
|---|---------|------------------|---|----------|-----|----|------------------|
| D1 | **NKI Naturalistic Viewing** (Telesford 2023) | S3 `s3://fcp-indi/data/Projects/NATVIEW_EEGFMRI/`; INDI page https://fcon_1000.projects.nitrc.org/indi/retro/nat_view.html; preproc code [NATVIEW_EEGFMRI](https://github.com/NathanKlineInstitute/NATVIEW_EEGFMRI) | 22 | **rest + Inscapes + movies** (Despicable Me, The Present, Monkey clips) + checkerboard | 61 cortical+3 (EOG/ECG) @ 5000 Hz | 2.1 s | **CC BY 4.0** (open S3) — DOI:10.1038/s41597-023-02458-8 |
| D2 | **Oddball / CWL** (Walz 2013) | OpenNeuro **ds000116** https://openneuro.org/datasets/ds000116 | 17 | auditory + visual oddball | 43 ch @ 1000 Hz, 3T Philips | 2.0 s | open (OpenNeuro/OpenfMRI) |
| D3 | **NODDI** (Deligianni 2014) | OSF **osf.io/94c5t** https://osf.io/94c5t | 16–17 | eyes-open resting state | 64 ch @ 250 Hz, 1.5T Siemens | 2.16 s | open (OSF) |
| D4 | **EEG-fMRI sleep** (Penn State) | OpenNeuro **ds003768** https://openneuro.org/datasets/ds003768 | 33 | rest (2×10 min) + sleep | 32 ch MR-compatible (Brain Products) | ~2.1 s | open — DOI:10.1016/j.dib.2023.109131 |
| D5 | **gradCPT sustained attention** | OpenNeuro **ds006040** https://openneuro.org/datasets/ds006040 | 28 | rest (EO/EC) + checkerboard + gradCPT + imagery (+DWI) | 64 ch (BrainCap MR), 3T Prisma | — | open — DOI:10.1038/s41597-026-06616-6 |
| D6 | **Neurofeedback motor imagery XP1/XP2** | OpenNeuro **ds002336** + **ds002338** | 10 / 16 | right-hand motor-imagery neurofeedback (block) | 64 ch | — | open — DOI:10.1038/s41597-020-0498-3 |
| D7 | **TVB / Charité-Berlin** (Ritter 2024) | **EBRAINS** dataset (search.kg.ebrains.eu) | 50 | 22-min resting state (+dwMRI, sMRI, connectomes) | EEG (BIDS) | — | EBRAINS (free registration) — bioRxiv:2024.04.17.589718 |
| D8 | **CN-EPFL** (speeded visual discrimination) | distributed via E2fNet/NT-ViT preproc repos (kgr20/E2fNet) | 20 | speeded visual discrimination | EEG, 3T | — | open via repos |
| D9 | **Spectral-power EEG-fMRI** | Mendeley Data **crhybxpdy6** https://data.mendeley.com/datasets/crhybxpdy6 | 20 | eyes-open/closed + EO/EC block task | 32 ch (Geodesic) | — | **CC BY 4.0** |

**Recommended priority to pull (best 3):**
1. **D1 NKI Naturalistic Viewing** — richest, fully open on S3 (CC BY 4.0), 22 subjects, rest **plus** movies/Inscapes → built-in naturalistic generalization condition; high-density EEG. Best single dataset for a modern EEG-FM→BOLD study.
2. **D2 Oddball (ds000116)** and **3. D3 NODDI (OSF 94c5t)** — the **canonical benchmark pair** used by Calhas, NT-ViT, E2fNet, Spec2Vol. Pulling these lets you report numbers directly comparable to published SSIM/r.

**Scale extensions:** D4 (ds003768, 33 subj sleep — good for low-vigilance/global-signal prediction), D5 (ds006040, 28 subj, task+rest), D6 (ds002336/8 — same data Donoso used, lets you reproduce his contrast). D7 (TVB-50) is large but resting-only and needs EBRAINS registration. **NeuroBOLT's own Vanderbilt data does not appear to be openly released** — reproduce it on D1/D3 instead.

---

## 3. The REVE-vs-bespoke experiment proposal

**REVE** (arXiv:2510.21585, NeurIPS 2025; project + **public code & weights** at https://brain-bzh.github.io/reve/) is a masked-autoencoder EEG foundation model with a **4D spatio-temporal positional encoding** built from true 3D electrode coordinates, pretrained on ~60,000 h / 92 datasets / 25,000 subjects. It is montage-agnostic (handles arbitrary electrode layouts and lengths) and yields strong frozen-feature linear-probe performance — exactly the properties that matter for transferring across simultaneous EEG-fMRI rigs with non-standard in-scanner montages.

### One-line research question
**Does a frozen REVE embedding + a lightweight adapter match or beat a bespoke end-to-end EEG→BOLD network (e.g. NeuroBOLT / E2fNet) at predicting fMRI BOLD — and does REVE transfer better than the LaBraM backbone NeuroBOLT already uses?**

### Why it's well-posed (and what's already known)
- NeuroBOLT (model A) **already validates the "frozen EEG-FM + trainable adapter" recipe** with LaBraM-base, beating a from-scratch transformer (BIOT). The novel test is **swapping LaBraM → REVE** — REVE's larger, montage-aware pretraining is the hypothesized advantage, especially cross-site/cross-montage.
- Donoso (model I) shows **generic** frozen LLMs barely help. So the bet is specifically on a **domain EEG-FM**, not a generic foundation model — REVE is the right candidate.

### What's needed to run it (the existing npz is unusable here)
The precomputed `/mnt/t9/reve_hbn_emb.npz` is **HBN = non-simultaneous EEG**, so it has **no time-locked BOLD target** and cannot be used for this experiment. REVE must be re-extracted on a **simultaneous** cohort (D1 NKI, or the D2/D3 benchmark pair).

1. **Cohort + targets.** Take D1 (or D2+D3). fMRI: preprocess → ROI timeseries (DiFuMo-64 / Schaefer, include thalamus + V1) for NeuroBOLT-style per-TR *r*; optionally keep whole-volume for E2fNet-style SSIM.
2. **REVE extraction (frozen).** Slide REVE over ~16 s EEG windows (HRF-duration, NeuroBOLT convention) aligned to each fMRI TR; feed true electrode coords into REVE's 4D positional encoding. Output: one embedding per window. Backbone weights frozen.
3. **Adapter.** Lightweight head only — linear probe → small MLP → shallow transformer — mapping REVE embedding → BOLD target. (Optional small decoder for whole-volume output.)
4. **Baselines.** (a) Bespoke end-to-end: NeuroBOLT and/or E2fNet trained on the same cohort. (b) Frozen-FM ablation: **REVE vs LaBraM vs CBraMod**, identical adapter — isolates which EEG-FM transfers best to hemodynamics.
5. **Metrics & target.** Pearson *r* per ROI, **intra- and inter-subject** splits (SSIM/PSNR if whole-volume). Bar to beat: **NeuroBOLT inter-subject r ≈ 0.47**.
6. **Compute.** Frozen backbone + MLP head + ROI targets is light — fits a single GB10 on the DGX Spark. No foundation-model pretraining required.

**Expected contribution.** Either (i) REVE+adapter ≥ bespoke nets, confirming that a strong EEG-FM is a sufficient front-end for BOLD prediction (cheaper, montage-portable), or (ii) it underperforms, sharpening the question of what hemodynamics-relevant information bespoke spectral/spatial modules add on top of an EEG-FM. Both outcomes are publishable for §5.

---

## References (IDs)

- NeuroBOLT — arXiv:2410.05341 (NeurIPS 2024)
- E2fNet / "Brainwaves→Brain Scans" — arXiv:2502.08025
- Calhas & Henriques (attentional graphs) — arXiv:2203.03481; follow-up arXiv:2504.10752
- NT-ViT — arXiv:2409.11836
- CATD — arXiv:2408.00777
- Spec2VolCAMU-Net — arXiv:2505.09521
- SIREN→fMRI — arXiv:2311.04234
- Liu & Sajda transcoding — DOI:10.1007/978-3-031-43075-6_6; arXiv:2010.02167; arXiv:2212.02226
- Donoso "two strategies" — DOI:10.3389/fsysb.2025.1715692; bioRxiv:2025.05.06.652346
- REVE EEG-FM — arXiv:2510.21585
- Datasets: NKI NatView DOI:10.1038/s41597-023-02458-8; Oddball OpenNeuro ds000116; NODDI OSF 94c5t; sleep ds003768 (DOI:10.1016/j.dib.2023.109131); gradCPT ds006040 (DOI:10.1038/s41597-026-06616-6); neurofeedback ds002336/ds002338 (DOI:10.1038/s41597-020-0498-3); TVB-50 bioRxiv:2024.04.17.589718 (EBRAINS); Mendeley crhybxpdy6
