#!/usr/bin/env python3
"""
Peng 2024 dataset inventory + Variant G feasibility check.

Walks every subject/session, reports per-run BOLD shapes, mask sizes,
behavioral trial counts. Then runs Variant G (AR(1) Bayesian conjugate
GLM) on a sample feedback run to confirm the variance-aware decoder
produces sensible posterior variance distributions on this dataset.

Output: /data/datasets/peng_2024_neurofeedback/inventory.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

ROOT = Path("/data/datasets/peng_2024_neurofeedback/subjects")
OUT = Path("/data/datasets/peng_2024_neurofeedback/inventory.json")


def scan_session(sub_dir: Path, ses: str) -> dict:
    sd = sub_dir / ses
    if not sd.exists():
        return {}
    info = {"session": ses}
    for branch in ("recognition", "feedback"):
        b = sd / branch
        if not b.exists():
            continue
        runs = sorted(b.glob("run_*_bet.nii*"))
        per_run = []
        for r in runs[:1]:  # just first run for shape probe
            img = nib.load(r)
            per_run.append({
                "name": r.name,
                "shape": list(img.shape),
                "vox_mm": [float(x) for x in img.header.get_zooms()[:3]],
            })
        info[f"{branch}_n_runs"] = len(runs)
        info[f"{branch}_first"] = per_run
    # Masks
    mask_dir = sd / "recognition" / "mask"
    if mask_dir.exists():
        chosen = mask_dir / "chosenMask.npy"
        if chosen.exists():
            arr = np.load(chosen)
            info["chosenMask_shape"] = list(arr.shape)
            info["chosenMask_dtype"] = str(arr.dtype)
            if arr.dtype == bool:
                info["chosenMask_true"] = int(arr.sum())
            elif arr.dtype.kind in "iu":
                info["chosenMask_nonzero"] = int((arr != 0).sum())
        roi_files = sorted(mask_dir.glob("*_FreeSurfer.nii"))
        info["roi_masks"] = [r.stem for r in roi_files]
        hippo = sorted(mask_dir.glob("lfseg_corr_usegray_*.nii"))
        info["hippocampal_subfields"] = len(hippo)
    return info


def variant_g_sanity(sub_dir: Path) -> dict:
    """Quick Variant G probe on the first feedback run we find.

    Loads BOLD, applies chosenMask, runs the conjugate AR(1) GLM via
    jaxoccoli on a synthetic event design. Reports beta/var stats so
    we can confirm the variance-aware decoder behaves sensibly on
    Peng data structure.
    """
    sys.path.insert(0, "/home/mhough/dev/hippy-feat")
    sys.path.insert(0, "/home/mhough/dev/hippy-feat/jaxoccoli")
    try:
        from jaxoccoli.bayesian_beta import make_ar1_conjugate_glm
        import jax.numpy as jnp
    except ImportError as e:
        return {"variant_g_error": f"jaxoccoli import failed: {e}"}

    # Find the first feedback run with an associated chosenMask
    fb = None
    mask_arr = None
    for ses in ("ses1", "ses2", "ses3", "ses4", "ses5"):
        runs = sorted((sub_dir / ses / "feedback").glob("run_*_bet.nii*"))
        mask_path = sub_dir / ses / "recognition" / "mask" / "chosenMask.npy"
        if runs and mask_path.exists():
            fb = runs[0]
            mask_arr = np.load(mask_path)
            session_used = ses
            break
    if fb is None:
        return {"variant_g_error": "no feedback run + chosenMask found"}

    img = nib.load(fb)
    bold = img.get_fdata()  # (X, Y, Z, T)
    # chosenMask is typically 3D with same XYZ. Coerce to bool.
    if mask_arr.dtype != bool:
        mask_arr = mask_arr != 0
    if mask_arr.shape != bold.shape[:3]:
        return {
            "variant_g_error": f"mask {mask_arr.shape} != bold {bold.shape[:3]}",
        }
    masked = bold[mask_arr].astype(np.float32)  # (n_vox, T)

    # Synthetic event design: stimulus every ~7 TRs (paper-like cadence)
    T = masked.shape[1]
    tr = float(img.header.get_zooms()[3]) if len(img.header.get_zooms()) > 3 else 1.5
    onsets = np.arange(6.0, T * tr, 7.0 * tr)
    # Glover HRF
    from scipy.stats import gamma as g
    t = np.arange(int(np.ceil(32.0 / tr))) * tr
    hrf = g.pdf(t, 6.0, scale=1.0) - (1/6) * g.pdf(t, 16.0, scale=1.0)
    hrf = (hrf / np.abs(hrf).max()).astype(np.float32)
    box = np.zeros(T, dtype=np.float32)
    for o in onsets:
        i = int(round(o / tr))
        if 0 <= i < T:
            box[i] = 1.0
    reg = np.convolve(box, hrf)[:T]
    drift = np.cos(2 * np.pi * np.arange(T) / max(T - 1, 1))
    intercept = np.ones(T, dtype=np.float32)
    X = np.column_stack([reg, intercept, drift]).astype(np.float32)

    # Run G on a sample of 200 voxels (full whole-mask too slow for inventory)
    n_sample = min(200, masked.shape[0])
    idx = np.linspace(0, masked.shape[0] - 1, n_sample).astype(int)
    Y = masked[idx]                                            # (n_sample, T)
    params, fwd = make_ar1_conjugate_glm(jnp.array(X))
    import jax
    betas, vars_, sigma2, rho = jax.vmap(lambda y: fwd(params, y))(jnp.array(Y))
    betas = np.asarray(betas[:, 0])  # probe column
    vars_ = np.asarray(vars_[:, 0])

    return {
        "variant_g_session_tested": session_used,
        "variant_g_run_tested": fb.name,
        "variant_g_n_TRs": int(T),
        "variant_g_TR_seconds": tr,
        "variant_g_n_voxels_total": int(masked.shape[0]),
        "variant_g_n_voxels_sampled": int(n_sample),
        "variant_g_beta_mean": float(np.mean(betas)),
        "variant_g_beta_std": float(np.std(betas)),
        "variant_g_var_mean": float(np.mean(vars_)),
        "variant_g_var_min": float(np.min(vars_)),
        "variant_g_var_max": float(np.max(vars_)),
        "variant_g_var_strictly_positive": bool(np.all(vars_ > 0)),
        "variant_g_no_nan": bool(not np.any(np.isnan(vars_))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+",
                    help="subset of subjects to scan; default all")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    sub_dirs = []
    if args.subjects:
        for s in args.subjects:
            d = ROOT / s / s
            if d.exists():
                sub_dirs.append(d)
    else:
        for top in sorted(ROOT.iterdir()):
            inner = top / top.name
            if inner.exists():
                sub_dirs.append(inner)

    print(f"scanning {len(sub_dirs)} subjects")

    out = {"scanned_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
           "n_subjects": len(sub_dirs),
           "subjects": {}}

    for d in sub_dirs:
        sub = d.name
        print(f"  {sub}")
        out["subjects"][sub] = {}
        for ses in ("ses1", "ses2", "ses3", "ses4", "ses5"):
            si = scan_session(d, ses)
            if si:
                out["subjects"][sub][ses] = si

    # Variant G probe on the first subject only — characterize behavior
    if sub_dirs:
        print(f"variant_g sanity on {sub_dirs[0].name}")
        out["variant_g"] = variant_g_sanity(sub_dirs[0])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {args.out}")
    print(json.dumps(out.get("variant_g", {}), indent=2))


if __name__ == "__main__":
    main()
