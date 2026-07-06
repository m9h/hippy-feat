#!/usr/bin/env python3
"""fMRI-FMScope: the identity-trap audit ported to fMRI on NSD multisubject.

The literal fMRI analog of the EEG FMScope audit (arXiv:2606.06647): how much of an
fMRI representation's structure is *subject identity* (a confound) vs *stimulus*?
Substrate = NSD shared-1000 seen by ALL 8 subjects -> a clean (subject x image) factorial.
Locally (sessions 01-03) the factorial is 145 images x 8 subjects (see shared1000 sizing).

This module does the RAW-BETAS representation (the foundational, hardest case). NSD betas are
in PER-SUBJECT native functional space (different shapes/voxel counts), so there is no common
voxel space to stack across subjects. The principled cross-subject-comparable representation is
representational geometry (RSA), so we audit identity in geometry space:

  (A) RSA variance decomposition (Charest et al. 2014, individual differences in object
      representation): within-subject RDM reliability (split-half over voxels) = total reliable
      signal; between-subject RDM consistency = SHARED stimulus geometry; the reliable-but-NOT-
      shared remainder = IDIOSYNCRATIC (subject-identity) geometry. The FMScope null-subject-
      variance analog = idiosyncratic / shared, and vs a subject-label-permutation null.

  (B) Subject-axis LEACE erasure (Belrose 2023) on the per-(subject,image) RDM-row features,
      reusing the EXACT emeg-fm primitive (fmscope.diagnostics.erasure.subject_axis_erasure) so
      the number is apples-to-apples with the EEG side: linear subject-identity BA pre/post
      erasure + the nonlinear residual. (Image-decoding survival is leak-free only in a common
      feature space -> deferred to the MindEye-embedding / fmri-FM-latent reps, next.)

Pure numpy/sklearn — NO jax/torch/GPU import (the box GPU is contended). CPU + a few NFS reads.

    python scripts/nsd_fmscope_audit.py            # full betas audit -> docs/ + results json
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/mhough/dev/fmscope")  # the FMScope diagnostics package

NSD_DIR = Path("/data/3t/nsd_multisubject")
STIM_CSV = "/data/3t/data/all_stimuli/nsd_stim_info_merged.csv"
OUT_DIR = Path("/home/mhough/dev/hippy-feat/docs")
RES_DIR = Path("/home/mhough/dev/hippy-feat/results/fmscope"); RES_DIR.mkdir(parents=True, exist_ok=True)
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
TRIALS_PER_SESSION = 750
# sessions available locally (truenas pulled 04-20 on 2026-06-30; all 8 subjects have 01-20)
LOCAL_SESSIONS = [f"{i:02d}" for i in range(1, 21)]
LOCAL_MAX_TRIAL = TRIALS_PER_SESSION * len(LOCAL_SESSIONS)   # 20 sessions -> 15000


# --------------------------------------------------------------------------- #
# trial -> image map for the local shared-1000 factorial                      #
# --------------------------------------------------------------------------- #
def build_trial_map():
    """Return (image_ids, per_subject_trials) where per_subject_trials[s][img] is the list of
    1-indexed global trials (<=2250) at which subject s saw shared-1000 image `img`."""
    import pandas as pd
    df = pd.read_csv(STIM_CSV)
    sh = df[df["shared1000"] == True].reset_index(drop=True)
    img_ids = sh["nsdId"].to_numpy()
    per_subj_all = {}
    present = []
    for si, subj in enumerate(SUBJECTS, start=1):
        reps = sh[[f"subject{si}_rep0", f"subject{si}_rep1", f"subject{si}_rep2"]].to_numpy()
        local = [(reps[r][(reps[r] >= 1) & (reps[r] <= LOCAL_MAX_TRIAL)]).tolist()
                 for r in range(len(sh))]
        per_subj_all[subj] = {int(img_ids[r]): [int(t) for t in local[r]] for r in range(len(sh))}
        present.append(np.array([len(local[r]) > 0 for r in range(len(sh))]))
    keep = np.all(np.vstack(present), axis=0)            # images with >=1 local rep in ALL subjects
    image_ids = [int(img_ids[r]) for r in range(len(sh)) if keep[r]]
    per_subj = {subj: {img: per_subj_all[subj][img] for img in image_ids} for subj in SUBJECTS}
    return image_ids, per_subj


def load_image_reps(subj, image_ids, trials):
    """Per-image list of per-rep nsdgeneral-masked beta vectors for one subject.
    Returns reps: list (len n_images) of (n_local_reps, n_vox) arrays. Reads each needed
    session file once; uses `> 0` (NSD nsdgeneral mask is {-1,0,1}, -1 = out-of-brain)."""
    import nibabel as nib
    mask = (nib.load(str(NSD_DIR / f"{subj}_nsdgeneral.nii.gz")).get_fdata().reshape(-1) > 0)
    nvox = int(mask.sum())
    by_ses = {ses: [] for ses in LOCAL_SESSIONS}             # ses -> list of (row, image_idx)
    for ii, img in enumerate(image_ids):
        for g in trials[img]:
            ses = LOCAL_SESSIONS[(g - 1) // TRIALS_PER_SESSION]
            by_ses[ses].append(((g - 1) % TRIALS_PER_SESSION, ii))
    reps = [[] for _ in image_ids]
    for ses in LOCAL_SESSIONS:
        items = by_ses[ses]
        if not items:
            continue
        img = nib.load(str(NSD_DIR / subj / f"betas_session{ses}.nii.gz"))
        data = np.asarray(img.dataobj, dtype=np.float32)             # (X,Y,Z,750) full read once
        data = data.reshape(-1, data.shape[-1])[mask].T              # (750, n_vox)
        for r, ii in items:
            reps[ii].append(np.nan_to_num(data[r]))
        del data
    return [np.vstack(r) if r else np.zeros((0, nvox), np.float32) for r in reps]


def zscore_voxels(betas):
    """voxel z-score across images (standard RSA pre-step)."""
    return (betas - betas.mean(0)) / (betas.std(0) + 1e-6)


def rep_noise_ceiling(reps):
    """Split-rep noise ceiling: on images with >=2 local reps, split reps in half, build a
    correlation RDM from each half over that common image set, Spearman-correlate the two,
    Spearman-Brown correct to full length. Returns (ceiling, n_images_used)."""
    idx = [i for i, r in enumerate(reps) if r.shape[0] >= 2]
    if len(idx) < 10:
        return float("nan"), len(idx)
    h1, h2 = [], []
    for i in idx:
        r = reps[i]; k = r.shape[0] // 2
        h1.append(r[:k].mean(0)); h2.append(r[k:].mean(0))
    h1 = zscore_voxels(np.vstack(h1)); h2 = zscore_voxels(np.vstack(h2))
    rho = spearman(utri(corr_rdm(h1)), utri(corr_rdm(h2)))
    sb = (2 * rho) / (1 + rho) if rho > -1 else rho      # Spearman-Brown (half -> full)
    return float(sb), len(idx)


# --------------------------------------------------------------------------- #
# RSA primitives (pure numpy)                                                 #
# --------------------------------------------------------------------------- #
def corr_rdm(betas):
    """1 - Pearson correlation RDM over images. betas (n_img, n_vox)."""
    c = betas - betas.mean(1, keepdims=True)
    c /= (np.linalg.norm(c, axis=1, keepdims=True) + 1e-12)
    return 1.0 - c @ c.T


def utri(m):
    iu = np.triu_indices(m.shape[0], k=1)
    return m[iu]


def spearman(a, b):
    from scipy.stats import rankdata
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra @ rb) / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-12))


# --------------------------------------------------------------------------- #
# main audit                                                                  #
# --------------------------------------------------------------------------- #
def main():
    print("=== fMRI-FMScope: raw-betas identity-trap audit (NSD shared-1000) ===", flush=True)
    image_ids, per_subj = build_trial_map()
    n_img = len(image_ids)
    print(f"local shared factorial: {n_img} images x {len(SUBJECTS)} subjects", flush=True)

    betas = {}; rdms = {}; reliab = {}; nc_n = {}
    for subj in SUBJECTS:
        reps = load_image_reps(subj, image_ids, per_subj[subj])
        b = zscore_voxels(np.vstack([r.mean(0) for r in reps]))     # rep-averaged, (n_img, n_vox)
        nrep = np.mean([len(per_subj[subj][img]) for img in image_ids])
        nc, n_used = rep_noise_ceiling(reps)                        # split-REP noise ceiling
        betas[subj] = b; rdms[subj] = corr_rdm(b); reliab[subj] = nc; nc_n[subj] = n_used
        print(f"  {subj}: betas {b.shape}  mean_reps={nrep:.2f}  "
              f"rep-noise-ceiling={nc:.3f} (n>=2rep={n_used})", flush=True)

    # (A) RSA variance decomposition --------------------------------------------------------
    pairs = [(i, j) for i in range(len(SUBJECTS)) for j in range(i + 1, len(SUBJECTS))]
    between = [spearman(utri(rdms[SUBJECTS[i]]), utri(rdms[SUBJECTS[j]])) for i, j in pairs]
    between_mean = float(np.mean(between))
    within_mean = float(np.nanmean(list(reliab.values())))   # reliable-signal ceiling (rep split)
    # idiosyncratic (subject-identity) geometry = reliable - shared
    idiosyncratic = max(within_mean - between_mean, 0.0)
    nsv_ratio = idiosyncratic / between_mean if between_mean > 0 else float("nan")
    # permutation null for between-subject consistency (shuffle image labels per subject)
    rng = np.random.RandomState(0); null = []
    for _ in range(200):
        i, j = pairs[rng.randint(len(pairs))]
        perm = rng.permutation(n_img)
        null.append(spearman(utri(rdms[SUBJECTS[i]]), utri(rdms[SUBJECTS[j]][perm][:, perm])))
    null_mean, null_sd = float(np.mean(null)), float(np.std(null))
    z_between = (between_mean - null_mean) / (null_sd + 1e-12)

    print(f"\n[RSA decomposition]", flush=True)
    print(f"  within-subject rep noise ceiling (reliable signal): {within_mean:.3f}", flush=True)
    print(f"  between-subject RDM consistency (shared stimulus): {between_mean:.3f} "
          f"(perm null {null_mean:.3f}+/-{null_sd:.3f}, z={z_between:.1f})", flush=True)
    print(f"  idiosyncratic (subject-identity) geometry        : {idiosyncratic:.3f}", flush=True)
    print(f"  >>> null-subject-variance ratio (idio/shared)    : {nsv_ratio:.2f}", flush=True)

    # (B) LEACE subject-axis erasure on RDM-row features (reuse emeg-fm primitive) ----------
    from fmscope.diagnostics.erasure import subject_axis_erasure
    X, subj_ids = [], []
    for si, subj in enumerate(SUBJECTS):
        X.append(rdms[subj])                  # (n_img, n_img) rows = RDM-row features
        subj_ids.append(np.full(n_img, si))
    X = np.vstack(X); subj_ids = np.concatenate(subj_ids)
    # column z-score (per-feature) so the probe isn't dominated by scale
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    er = subject_axis_erasure(X, subj_ids, label=None)
    print(f"\n[LEACE subject-axis erasure on RDM-row features]  (chance={er.chance:.3f})", flush=True)
    print(f"  subject-identity BA   pre-erase : {er.subj_ba_linear_pre:.3f}", flush=True)
    print(f"  subject-identity BA  post-erase : {er.subj_ba_linear_post:.3f} (linear axis removed)",
          flush=True)
    print(f"  subject-identity BA  post (MLP) : {er.subj_ba_mlp_post:.3f} (nonlinear residual)",
          flush=True)
    print(f"  subject subspace rank/dim       : {er.rank_subject_axis}/{er.embed_dim} "
          f"(degenerate={er.degenerate})", flush=True)

    # save -----------------------------------------------------------------------------------
    out = {
        "n_images": n_img, "n_subjects": len(SUBJECTS),
        "mean_reps_per_image": float(np.mean([len(per_subj[s][i]) for s in SUBJECTS for i in image_ids])),
        "rsa": {"within_subject_rep_noise_ceiling": within_mean, "between_subject_consistency": between_mean,
                "idiosyncratic_geometry": idiosyncratic, "null_subject_variance_ratio": nsv_ratio,
                "perm_null_mean": null_mean, "perm_null_sd": null_sd, "z_between": z_between,
                "per_subject_noise_ceiling": reliab, "per_subject_n_2rep_images": nc_n},
        "leace": {"chance": er.chance, "subj_ba_linear_pre": er.subj_ba_linear_pre,
                  "subj_ba_linear_post": er.subj_ba_linear_post, "subj_ba_mlp_post": er.subj_ba_mlp_post,
                  "rank_subject_axis": er.rank_subject_axis, "embed_dim": er.embed_dim,
                  "degenerate": er.degenerate, "cond_shrunk": er.cond_shrunk},
    }
    (RES_DIR / "betas_audit.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {RES_DIR / 'betas_audit.json'}", flush=True)
    return out


if __name__ == "__main__":
    main()
