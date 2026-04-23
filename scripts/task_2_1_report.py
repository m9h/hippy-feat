#!/usr/bin/env python3
"""
Task 2.1 decomposition report.

Reads retrieval_<condition>.json outputs produced by
scripts/mindeye_retrieval_eval.py and prints a one-page summary that
attributes the paper's 10 pp RT gap to fMRIPrep-motion vs GLMsingle-HRF
contributions.

Paper references (Iyer et al. ICML 2026 Table 1):
    End-of-run RT   : 66% image / 62% brain retrieval
    Offline 3T      : 76% image / 64% brain retrieval
    gap             : 10pp image / 2pp brain

Factorial:
    paper-RT    = RT motion    + Glover HRF     ← 66% baseline
    condition A = fMRIPrep mc  + Glover HRF     ← isolates fMRIPrep
    condition B = RT motion    + GLMsingle HRF  ← isolates GLMsingle
    paper-full  = fMRIPrep mc  + GLMsingle HRF  ← 76% paper-offline
"""
import argparse
import json
from pathlib import Path


PAPER = {
    "RT_baseline": {"top1_image_retrieval": 0.66, "top1_brain_retrieval": 0.62},
    "Offline":     {"top1_image_retrieval": 0.76, "top1_brain_retrieval": 0.64},
}


def load(d: Path, name: str) -> dict | None:
    p = d / f"retrieval_{name}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir",
                    default="/data/derivatives/rtmindeye_paper/task_2_1_betas")
    args = ap.parse_args()

    d = Path(args.results_dir)
    A = load(d, "A_fmriprep_glover")
    B = load(d, "B_rtmotion_glmsingle")

    print("=" * 72)
    print("Task 2.1 decomposition — fMRIPrep vs GLMsingle contributions to RT gap")
    print("=" * 72)

    print(f"\nPaper (Iyer et al. ICML 2026 Table 1, sub-005 ses-03 test):")
    print(f"  paper-RT baseline  (RT mc + Glover)    : img={PAPER['RT_baseline']['top1_image_retrieval']:.0%}  brain={PAPER['RT_baseline']['top1_brain_retrieval']:.0%}")
    print(f"  paper-offline      (fMRIPrep + GLMsingle): img={PAPER['Offline']['top1_image_retrieval']:.0%}  brain={PAPER['Offline']['top1_brain_retrieval']:.0%}")
    print(f"  gap to close                            : {PAPER['Offline']['top1_image_retrieval']-PAPER['RT_baseline']['top1_image_retrieval']:+.0%} img  {PAPER['Offline']['top1_brain_retrieval']-PAPER['RT_baseline']['top1_brain_retrieval']:+.0%} brain")

    rows = [("paper-RT baseline", PAPER["RT_baseline"]["top1_image_retrieval"],
             PAPER["RT_baseline"]["top1_brain_retrieval"])]
    if A:
        rows.append(("condition A (fMRIPrep + Glover)",
                     A["top1_image_retrieval"], A["top1_brain_retrieval"]))
    else:
        rows.append(("condition A", None, None))
    if B:
        rows.append(("condition B (RT + GLMsingle)",
                     B["top1_image_retrieval"], B["top1_brain_retrieval"]))
    else:
        rows.append(("condition B", None, None))
    rows.append(("paper-offline", PAPER["Offline"]["top1_image_retrieval"],
                 PAPER["Offline"]["top1_brain_retrieval"]))

    print(f"\n{'condition':<38} {'img-ret':>10} {'brain-ret':>10}")
    print("-" * 72)
    for name, i, b in rows:
        if i is None:
            print(f"{name:<38} {'pending':>10} {'pending':>10}")
        else:
            print(f"{name:<38} {i:>10.1%} {b:>10.1%}")

    if A and B:
        rt = PAPER["RT_baseline"]["top1_image_retrieval"]
        full = PAPER["Offline"]["top1_image_retrieval"]
        dA = A["top1_image_retrieval"] - rt       # fMRIPrep contribution
        dB = B["top1_image_retrieval"] - rt       # GLMsingle contribution
        total = full - rt
        interaction = total - dA - dB
        print(f"\nDecomposition (image retrieval, pp):")
        print(f"  fMRIPrep motion contribution  (A - RT)       : {dA*100:+.1f} pp")
        print(f"  GLMsingle HRF contribution    (B - RT)       : {dB*100:+.1f} pp")
        print(f"  Interaction (non-additive residual)          : {interaction*100:+.1f} pp")
        print(f"  Sum vs observed total gap   ({total*100:+.1f} pp)   : ({(dA+dB+interaction)*100:+.1f} pp)")
        dom = "fMRIPrep" if abs(dA) > abs(dB) else "GLMsingle"
        print(f"\nDominant driver of the 10 pp RT gap: {dom}")
    else:
        missing = [n for n, v in [("A", A), ("B", B)] if v is None]
        print(f"\nWaiting on condition(s) {missing} — rerun once retrieval JSONs land.")


if __name__ == "__main__":
    main()
