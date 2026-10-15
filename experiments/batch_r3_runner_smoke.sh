#!/bin/bash
#SBATCH --job-name=chagas_r3_runner
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r3_runner_smoke.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r3_runner_smoke.err
#SBATCH --time=0-03:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# run_r3.sh end to end on the 400-record set (train, score A and C, test-time windows,
# run_info), so a runner bug cannot stall the array.  Gates batch_r3.sh (afterok).
#
#   sbatch --dependency=afterok:<r3 smoke id> experiments/batch_r3_runner_smoke.sh
set -euo pipefail
WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
cd "$WT"
hostname; date
bash experiments/run_r3.sh "$WT/experiments/configs/r3_smoke.yaml" "$FIX/testdata" "$FIX/testdata" "$FIX/testdata"
R="$WT/experiments/results/r3_smoke"
for f in predictions_A.csv predictions_C.csv predictions_A_tta.csv predictions_C_tta.csv report.md report_tta.md run_info.json; do
  test -s "$R/$f" || { echo "FAIL: missing $R/$f"; exit 1; }
done
date; echo "RUNNER SMOKE OK"
