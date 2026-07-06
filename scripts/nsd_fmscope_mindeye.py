#!/usr/bin/env python3
"""fMRI-FMScope Result 2 — the MindEye-style (CLIP-aligned) representation.

The richer FMScope target the raw-betas rep couldn't do: a betas->CLIP ridge maps each subject's
nsdgeneral betas into a COMMON CLIP image-embedding space (shared across subjects by construction),
so we can measure not just subject-identity decodability but **image-retrieval survival after
subject-axis erasure** — the literal "how much of the retrieval was subject identity?" number.

Pipeline:
  clip  : OpenCLIP (ViT-L/14) embed the NSD images seen in sessions 01-20 -> cache (GPU, container).
  audit : per subject, betas->CLIP ridge (train on non-shared-1000 trials, test on the 748 shared,
          rep-averaged) -> predicted-CLIP tensor (subject x shared-image, common space). Then
          fmscope.subject_axis_erasure (subject-identity BA pre/post) + image-retrieval top-1
          (predicted-CLIP vs true-CLIP over the 748) before vs after applying the subject eraser.

    python scripts/nsd_fmscope_mindeye.py clip     # GPU (container): cache CLIP targets
    python scripts/nsd_fmscope_mindeye.py audit     # CPU (host): ridge + FMScope Result 2
"""
from __future__ import annotations
import argparse, csv, sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, "/home/mhough/dev/fmscope")

NSD_DIR = Path("/data/3t/nsd_multisubject")
STIM_CSV = "/data/3t/data/all_stimuli/nsd_stim_info_merged.csv"
HDF5 = "/data/3t/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
CLIP_CACHE = Path("/data/derivatives/peer_fm_ww/fmscope/nsd_clip_vitL14.npz")
RES_DIR = Path("/home/mhough/dev/hippy-feat/results/fmscope"); RES_DIR.mkdir(parents=True, exist_ok=True)
CLIP_CACHE.parent.mkdir(parents=True, exist_ok=True)
SUBJECTS = [f"subj{i:02d}" for i in range(1, 9)]
TRIALS_PER_SESSION = 750
# 5 sessions: ~225 shared test images + ~3700 ridge-train trials — ample for the betas->CLIP ridge,
# and fast enough per subject (~4 min) that each checkpoints before the (MindEye-specific) reaper hits.
LOCAL_SESSIONS = [f"{i:02d}" for i in range(1, 6)]
LOCAL_MAX_TRIAL = TRIALS_PER_SESSION * len(LOCAL_SESSIONS)   # 3750
CLIP_MODEL = ("ViT-L-14", "openai")


# --------------------------------------------------------------------------- #
# design: trial<->image maps                                                  #
# --------------------------------------------------------------------------- #
def load_design():
    """Return (df-derived) per-subject trial->nsdId maps + the shared-1000 set + needed nsdIds.
    trial map: per subject, dict global_trial(1..15000) -> nsdId (row index in the 73k stim table)."""
    import pandas as pd
    df = pd.read_csv(STIM_CSV)
    nsd_id = df["nsdId"].to_numpy()
    shared = df["shared1000"].to_numpy().astype(bool)
    per_subj = {}
    needed = set()
    for si, subj in enumerate(SUBJECTS, start=1):
        reps = df[[f"subject{si}_rep0", f"subject{si}_rep1", f"subject{si}_rep2"]].to_numpy()
        t2img = {}
        for row in range(len(df)):
            for k in range(3):
                g = int(reps[row][k])
                if 1 <= g <= LOCAL_MAX_TRIAL:
                    t2img[g] = int(nsd_id[row])
                    needed.add(int(nsd_id[row]))
        per_subj[subj] = t2img
    shared_ids = set(int(nsd_id[r]) for r in range(len(df)) if shared[r])
    return per_subj, shared_ids, sorted(needed)


# --------------------------------------------------------------------------- #
# phase clip: OpenCLIP embed the needed images                                #
# --------------------------------------------------------------------------- #
def phase_clip(batch=256):
    import h5py, torch, open_clip
    from torchvision.transforms import v2
    _, _, needed = load_design()
    print(f">>> CLIP targets needed: {len(needed)} images", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(CLIP_MODEL[0], pretrained=CLIP_MODEL[1])
    model = model.to(dev).eval()
    mean = (0.48145466, 0.4578275, 0.40821073); std = (0.26862954, 0.26130258, 0.27577711)
    tf = v2.Compose([v2.Resize(224, antialias=True), v2.CenterCrop(224),
                     v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)])
    brick = h5py.File(HDF5, "r")["imgBrick"]
    ids = np.array(needed, dtype=np.int64)
    out = np.zeros((len(ids), 768), np.float32)
    for i in range(0, len(ids), batch):
        bids = ids[i:i + batch].tolist()                       # sorted -> valid h5py fancy index
        imgs = brick[bids]                                     # (B,425,425,3) uint8
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2)         # (B,3,425,425)
        x = tf(x).to(dev)
        with torch.inference_mode(), torch.autocast("cuda", torch.float16, enabled=dev == "cuda"):
            e = model.encode_image(x)
        e = (e / e.norm(dim=-1, keepdim=True)).float().cpu().numpy()
        out[i:i + len(bids)] = e
        if (i // batch) % 20 == 0:
            print(f"  {i+len(bids)}/{len(ids)}", flush=True)
    np.savez(CLIP_CACHE, ids=ids, emb=out)
    print(f">>> saved {out.shape} CLIP targets -> {CLIP_CACHE}", flush=True)


# --------------------------------------------------------------------------- #
# betas loader (all sessions-01-20 trials, per-trial, with nsdId)             #
# --------------------------------------------------------------------------- #
def load_all_trial_betas(subj, t2img):
    """Return (X, img_ids) for all local trials of one subject: X (n_trials, n_vox) voxel-z-scored
    across trials, img_ids (n_trials,) nsdId per trial. Reads each session file once."""
    import nibabel as nib
    mask = (nib.load(str(NSD_DIR / f"{subj}_nsdgeneral.nii.gz")).get_fdata().reshape(-1) > 0)
    rows, imgs = [], []
    for si, ses in enumerate(LOCAL_SESSIONS):
        img = nib.load(str(NSD_DIR / subj / f"betas_session{ses}.nii.gz"))
        data = np.asarray(img.dataobj, dtype=np.float32).reshape(-1, 750)[mask].T   # (750, n_vox)
        base = si * TRIALS_PER_SESSION
        for r in range(750):
            g = base + r + 1
            if g in t2img:
                rows.append(np.nan_to_num(data[r])); imgs.append(t2img[g])
        del data
    X = np.vstack(rows).astype(np.float32)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)                    # voxel z-score
    return X, np.array(imgs)


# --------------------------------------------------------------------------- #
# phase audit: ridge -> predicted-CLIP -> FMScope                              #
# --------------------------------------------------------------------------- #
def _ridge_fit_predict(Xtr, Ytr, Xte, alpha=None):
    """Closed-form ridge with a small alpha sweep (PCA-reduced for speed/stability)."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=min(512, Xtr.shape[0] - 1, Xtr.shape[1]), random_state=0).fit(sc.transform(Xtr))
    Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    best = None
    for a in ([alpha] if alpha else [1e2, 1e3, 1e4]):
        # quick self-split score
        m = Ridge(alpha=a).fit(Ztr[: int(0.8 * len(Ztr))], Ytr[: int(0.8 * len(Ztr))])
        s = np.mean([np.corrcoef(Ytr[int(0.8*len(Ztr)):][:, d], m.predict(Ztr[int(0.8*len(Ztr)):])[:, d])[0, 1]
                     for d in range(0, Ytr.shape[1], 64)])
        if best is None or s > best[0]:
            best = (s, a)
    m = Ridge(alpha=best[1]).fit(Ztr, Ytr)
    return m.predict(Zte)


def phase_audit():
    from fmscope.diagnostics.erasure import subject_axis_erasure
    per_subj, shared_ids, _ = load_design()
    clip = np.load(CLIP_CACHE)
    _emb, _ids = clip["emb"], clip["ids"]     # materialize ONCE — NpzFile re-reads the 167MB array
    id2emb = {int(i): _emb[k] for k, i in enumerate(_ids)}   # on every access (was → 78GB blowup)
    # shared imgs seen by ALL subjects in the loaded sessions (common factorial) — avoids empty
    # rep-means (NaN) for shared imgs a subject didn't see, and keeps the cross-subject tensor aligned
    _seen = [{img for img in per_subj[s].values() if img in shared_ids} for s in SUBJECTS]
    shared_sorted = sorted(set.intersection(*_seen) & set(id2emb))
    print(f">>> audit: {len(shared_sorted)} shared images, CLIP dim={clip['emb'].shape[1]}", flush=True)
    true_clip = np.stack([id2emb[i] for i in shared_sorted])           # (748, D)

    # per-subject checkpointing (resumable across kills/reboots): each subject's predicted-CLIP is
    # saved as soon as its ridge finishes, so a SIGTERM mid-run loses at most one subject.
    PRED_DIR = CLIP_CACHE.parent / "mindeye_pred"; PRED_DIR.mkdir(parents=True, exist_ok=True)
    pred, subj_ids, img_ids = [], [], []
    for si, subj in enumerate(SUBJECTS):
        pp = PRED_DIR / f"{subj}.npz"
        if pp.exists():
            Yhat = np.load(pp)["yhat"]
            print(f"  {subj}: cached pred {Yhat.shape}", flush=True)
        else:
            X, imgs = load_all_trial_betas(subj, per_subj[subj])
            Y = np.stack([id2emb[i] for i in imgs])                        # (n_trials, D)
            is_shared = np.array([i in shared_ids for i in imgs])
            Xtr, Ytr = X[~is_shared], Y[~is_shared]                        # train on non-shared
            Xte = np.stack([X[imgs == i].mean(0) for i in shared_sorted])  # rep-avg shared (748,vox)
            Yhat = _ridge_fit_predict(Xtr, Ytr, Xte)                       # (748, D) predicted CLIP
            np.savez(pp, yhat=Yhat.astype(np.float32))                     # CHECKPOINT immediately
            r = np.mean([np.corrcoef(true_clip[:, d], Yhat[:, d])[0, 1] for d in range(0, Yhat.shape[1], 32)])
            print(f"  {subj}: train {Xtr.shape} -> pred CLIP (r~{r:.3f}) [checkpointed]", flush=True)
            del X, Y
        pred.append(Yhat); subj_ids.append(np.full(len(shared_sorted), si)); img_ids.append(np.arange(len(shared_sorted)))
    P = np.vstack(pred); subj_ids = np.concatenate(subj_ids); img_ids = np.concatenate(img_ids)

    # retrieval top-1 (predicted-CLIP vs the 748 true-CLIP), pooled, BEFORE erasure
    def retrieval_top1(feats):
        f = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        t = true_clip / (np.linalg.norm(true_clip, axis=1, keepdims=True) + 1e-9)
        sim = f @ t.T                                                  # (N, 748)
        return float(np.mean(sim.argmax(1) == img_ids))

    from fmscope.diagnostics.erasure import whiten, subject_eraser, apply_eraser
    mu, Xc, W, Wp, _ = whiten(P, shrinkage=True)
    _, P_perp, rank = subject_eraser(Xc, W, Wp, subj_ids)
    Pe = apply_eraser(P, mu, P_perp)
    top1_pre, top1_post = retrieval_top1(P), retrieval_top1(Pe)
    er = subject_axis_erasure(P, subj_ids, label=None)

    chance = 1.0 / len(shared_sorted)
    print(f"\n[FMScope Result 2 — MindEye/CLIP rep]", flush=True)
    print(f"  image retrieval top-1  PRE-erase : {top1_pre:.3f} (chance {chance:.4f})", flush=True)
    print(f"  image retrieval top-1 POST-erase : {top1_post:.3f}  "
          f"(survival {top1_post/max(top1_pre,1e-9)*100:.0f}% -> identity share {(1-top1_post/max(top1_pre,1e-9))*100:.0f}%)", flush=True)
    print(f"  subject-id BA linear pre/post    : {er.subj_ba_linear_pre:.3f} / {er.subj_ba_linear_post:.3f}  (chance {er.chance:.3f})", flush=True)
    print(f"  subject-id BA MLP post           : {er.subj_ba_mlp_post:.3f}", flush=True)
    print(f"  subject subspace rank/dim        : {er.rank_subject_axis}/{er.embed_dim}", flush=True)
    out = {"n_shared": len(shared_sorted), "clip_model": CLIP_MODEL,
           "retrieval_top1_pre": top1_pre, "retrieval_top1_post": top1_post,
           "retrieval_survival": top1_post / max(top1_pre, 1e-9),
           "subj_ba_linear_pre": er.subj_ba_linear_pre, "subj_ba_linear_post": er.subj_ba_linear_post,
           "subj_ba_mlp_post": er.subj_ba_mlp_post, "rank_subject_axis": er.rank_subject_axis,
           "embed_dim": er.embed_dim, "chance_retrieval": chance}
    (RES_DIR / "mindeye_audit.json").write_text(json.dumps(out, indent=2))
    print(f"\n[saved] {RES_DIR/'mindeye_audit.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("phase", choices=["clip", "audit"])
    a = ap.parse_args()
    (phase_clip if a.phase == "clip" else phase_audit)()


if __name__ == "__main__":
    main()
