#!/bin/bash
#SBATCH --job-name=chagas_rescore_reports
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/rescore_reports.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/rescore_reports.err
#SBATCH --time=0-04:00:00

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Re-score every per-run evaluation report (report.md, summary.csv, ...) whose summary has
# the round-2 views (and therefore bootstrap CIs) from the run's SAVED predictions, in place:
# eval_report.py --from-saved <run> [--saved-suffix tta].  The predictions files are not
# rewritten.  Covers results/<run>/, the WiSE-FT mixtures results/<run>/alpha*/ and the
# test-time-window pass (report_tta*).  Smoke runs are skipped.
#
# First checks that the --from-saved path reproduces an existing record-level summary.csv
# byte for byte (r3_B0_s42, against the copy in results/_record_level_backup/), so the only
# change in place is the resampling unit.
# CPU-only: bootstrap resampling over the test patients, no model is run.
#   sbatch experiments/batch_rescore_reports.sh
set -euo pipefail
WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
RES="$WT/experiments/results"
cd "$WT"
export CUDA_VISIBLE_DEVICES=
# Bootstrap resampling unit: patient groups (cluster bootstrap, default) or records.
UNIT=${BOOTSTRAP_UNIT:-patient}
echo "===== ENV ====="; hostname; date; echo "bootstrap unit: $UNIT"

echo "===== check: --from-saved --bootstrap-unit record reproduces the old summary ====="
REF="$RES/_record_level_backup/runs/r3_B0_s42/summary.csv"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
cp "$RES/r3_B0_s42/predictions_A.csv" "$RES/r3_B0_s42/predictions_C.csv" "$TMP/"
cp "$REF" "$TMP/summary_ref.csv"
$PY experiments/eval_report.py --from-saved "$TMP" --name r3_B0_s42 --bootstrap-unit record > "$TMP/log.txt"
if cmp -s "$TMP/summary.csv" "$TMP/summary_ref.csv"; then
  echo "CHECK OK: record-level summary reproduced byte for byte"
else
  echo "CHECK FAILED: record-level re-score differs from the saved summary"; diff "$TMP/summary.csv" "$TMP/summary_ref.csv" || true
  exit 1
fi

echo "===== re-score every run ====="
n=0
for S in "$RES"/*/summary.csv "$RES"/*/alpha*/summary.csv; do
  D=$(dirname "$S")
  REL=${D#"$RES"/}
  case "$REL" in _record_level_backup*|*smoke*) continue ;; esac
  grep -q '^S7,' "$S" || continue
  $PY experiments/eval_report.py --from-saved "$D" --bootstrap-unit "$UNIT" > /dev/null
  echo "rescored $REL"
  if [ -f "$D/predictions_A_tta.csv" ]; then
    $PY experiments/eval_report.py --from-saved "$D" --saved-suffix tta --bootstrap-unit "$UNIT" > /dev/null
    echo "rescored $REL (tta)"
  fi
  n=$((n + 1))
done
echo "===== DONE: $n run folders ====="; date
