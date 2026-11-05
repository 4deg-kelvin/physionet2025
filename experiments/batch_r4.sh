#!/bin/bash
#SBATCH --job-name=chagas_r4
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r4_%a.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r4_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --array=0-7%4

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
CONFIGS=(
  r3_W1R_s45 r3_W1R_s46
  r4_MWR_s42 r4_MWR_s43 r4_MWR_s44
  r4_W1RS_s42 r4_W1RS_s43 r4_W1RS_s44
)
CFG="$WT/experiments/configs/r4/${CONFIGS[$SLURM_ARRAY_TASK_ID]}.yaml"
mkdir -p "$WT/experiments/logs" "$WT/experiments/results"
cd "$WT"
test -f "$CFG" || { echo "missing $CFG"; exit 1; }
test -d "$FIX/data_train_full" || { echo "missing $FIX/data_train_full"; exit 1; }
test -f "$WT/experiments/data/record_durations.csv" || { echo "missing record_durations.csv"; exit 1; }
echo "array task $SLURM_ARRAY_TASK_ID -> $(basename "$CFG")"

bash experiments/run_r3.sh "$CFG" "$FIX/data_train_full"
