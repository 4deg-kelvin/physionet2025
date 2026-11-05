#!/bin/bash
# Bundle the manuscript for Overleaf: sources, generated numbers/tables/figures, bibliography, and the
# IOP class + ORCID logo. Rebuild first (bash paper/build.sh) so the generated files are current.
#   bash paper/make_overleaf_zip.sh   -> paper/overleaf_<date>.zip (ignored by git)
set -euo pipefail
cd "$(dirname "$0")"
OUT="overleaf_$(date +%Y%m%d).zip"
rm -f "$OUT"
cp -f iop/iopjournal.cls iop/orcid.pdf .
zip -q "$OUT" main.tex supplement.tex refs.bib numbers.tex numbers_r3.tex numbers_ext.tex iopjournal.cls orcid.pdf tables/*.tex figures/*.pdf
unzip -l "$OUT" | tail -1
echo "wrote $PWD/$OUT  (Overleaf: New Project -> Upload Project; compiler pdfLaTeX; main document main.tex; compile supplement.tex separately or as a second project)"
