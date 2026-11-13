#!/bin/bash
# Bundle the manuscript for Overleaf / IOP submission: sources, generated numbers/tables/figures,
# bibliography, and the IOP class + ORCID logo. Rebuild first (bash paper/build.sh) so the generated
# files are current.
#   bash paper/make_overleaf_zip.sh   -> paper/overleaf_<date>.zip (ignored by git)
# IOP asks for no subfolders in the source upload, so figures/ and tables/ are flattened into the zip
# root with figures_ / tables_ prefixes, and the \input / \includegraphics paths are rewritten in
# temporary copies of main.tex and supplement.tex (the working files are not touched).
set -euo pipefail
cd "$(dirname "$0")"
HERE=$PWD
OUT="overleaf_$(date +%Y%m%d).zip"
rm -f "$OUT"
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT
cp -f iop/iopjournal.cls iop/orcid.pdf refs.bib numbers.tex numbers_r3.tex numbers_ext.tex "$STAGE"/
for f in figures/*.pdf; do cp -f "$f" "$STAGE/figures_$(basename "$f")"; done
for f in tables/*.tex; do cp -f "$f" "$STAGE/tables_$(basename "$f")"; done
for t in main.tex supplement.tex; do
  sed -e 's#\\input{tables/#\\input{tables_#g' \
      -e 's#\\includegraphics\(\[[^]]*\]\)\{0,1\}{figures/#\\includegraphics\1{figures_#g' "$t" > "$STAGE/$t"
done
# Fail if any non-comment line still points into a subfolder.
if grep -nE '(figures|tables)/' "$STAGE/main.tex" "$STAGE/supplement.tex" | grep -vE '^[^:]+:[0-9]+:\s*%'; then
  echo "error: unrewritten subfolder path in the staged sources" >&2
  exit 1
fi
(cd "$STAGE" && zip -q "$HERE/$OUT" ./*)
unzip -l "$OUT" | tail -1
echo "wrote $HERE/$OUT  (Overleaf: New Project -> Upload Project; compiler pdfLaTeX; main document main.tex; compile supplement.tex separately or as a second project)"
