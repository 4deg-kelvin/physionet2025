#!/bin/bash
#SBATCH --job-name=chagas_r3
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r3_%a.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r3_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --array=0-13%4

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# Round-3 experiment matrix: every config is finalA_negcap + the round-3 protocol (fixed
# natural-prevalence validation split carved before the cap, PTB-XL grouped by patient,
# negatives capped to 6.5% prevalence, checkpoints selected on validation S7 TPR@5%), then
#   B0  protocol only                          seeds 42-44
#   W1  + 5 s windows                          seeds 42-46
#   W1R + 5 s windows + ranking loss           seeds 42-44
#   W1S + 5 s windows, PTB-XL excluded         seeds 42-44
# At most 4 tasks at once (%4).  Each task trains on the full folder through the
# UNMODIFIED train_model.py, scores Test A and Test C with predict_batched.py (and again
# with 2.5 s test-time window strides for the 5 s-window configs) and writes the report
# and run_info.json -- experiments/run_r3.sh.
#
#   sbatch experiments/batch_r3.sh
#
# Then the table: sbatch --dependency=afterany:<array id> experiments/batch_r3_table.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
CONFIGS=(
  r3_B0_s42 r3_B0_s43 r3_B0_s44
  r3_W1_s42 r3_W1_s43 r3_W1_s44 r3_W1_s45 r3_W1_s46
  r3_W1R_s42 r3_W1R_s43 r3_W1R_s44
  r3_W1S_s42 r3_W1S_s43 r3_W1S_s44
)
CFG="$WT/experiments/configs/r3/${CONFIGS[$SLURM_ARRAY_TASK_ID]}.yaml"
mkdir -p "$WT/experiments/logs" "$WT/experiments/results"
cd "$WT"
test -f "$CFG" || { echo "missing $CFG"; exit 1; }
test -d "$FIX/data_train_full" || { echo "missing $FIX/data_train_full"; exit 1; }
test -f "$WT/experiments/data/record_durations.csv" || { echo "missing record_durations.csv"; exit 1; }
echo "array task $SLURM_ARRAY_TASK_ID -> $(basename "$CFG")"

bash experiments/run_r3.sh "$CFG" "$FIX/data_train_full"
