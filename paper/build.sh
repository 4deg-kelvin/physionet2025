#!/bin/bash
# Build the manuscript PDF: regenerate figures/tables/numbers from saved results, then compile.
#   bash paper/build.sh            (no model is run; needs experiments/results/R2_TABLE_{A,C}.csv)
# LaTeX: tectonic, a self-contained engine (https://tectonic-typesetting.github.io); set TECTONIC to
# its path.  The IOP class (iop/iopjournal.cls) is IOP Publishing's author template, downloaded from
# https://publishingsupport.iopscience.iop.org/questions/latex-template/ .
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python}
TECTONIC=${TECTONIC:-/juice2/scr2/kelvinkn/other_work/edwards/tools/tectonic}
export TECTONIC_CACHE_DIR=${TECTONIC_CACHE_DIR:-/juice2/scr2/kelvinkn/other_work/edwards/tools/tectonic_cache}
cd "$HERE"
"$PY" make_assets.py
cp -f iop/iopjournal.cls iop/orcid.pdf .
"$TECTONIC" -X compile main.tex
echo "wrote $HERE/main.pdf"
