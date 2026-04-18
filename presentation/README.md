# Presentation: Closing the Preprocessing Gap for Real-Time MindEye

**Deck:** `rt_mindeye_pipeline.tex` (Beamer, `metropolis` theme, 16:9).
**Originally compiled:** 2026-03-25 (PDF not checked in — rebuild locally).
**Last source edit:** 2026-04-17 (fact-check corrections, see `notebook_parity.md`).

## Build

```bash
cd presentation/
pdflatex rt_mindeye_pipeline.tex
pdflatex rt_mindeye_pipeline.tex   # second pass for TOC / cross-refs
```

Or with `latexmk`:

```bash
latexmk -pdf rt_mindeye_pipeline.tex
```

Requires a LaTeX distribution with the `metropolis` beamer theme (TeX Live full, or `tlmgr install beamertheme-metropolis`).

## Contents

- `rt_mindeye_pipeline.tex` — source
- `*.png` — 7 figures referenced by the deck
- `notebook_parity.md` — notebook analysis, AR(1) parity results, variant-to-gap mapping, fact-check log
