#!/bin/bash
#SBATCH --job-name=chagas_r2
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r2_%a.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r2_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --array=0-8%4

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# Round-2 experiment matrix (PLAN_R2.md section 4): M1 (length-matched negatives),
# W1 (5 s windows), MW (both + WiSE-FT alpha in {0, 0.5, 1}) x seeds 42/43/44, on top of
# the shipped config B0 (= finalA_negcap, lrsweep_lr2e5_s43/44, which already exist).
# At most 4 tasks at once (%4).  Each task trains on the full 292,885-record folder
# through the UNMODIFIED train_model.py, scores Test A and Test C with
# predict_batched.py (once per alpha for MW) and writes the round-2 report and
# run_info.json -- experiments/run_r2.sh.
#
#   sbatch experiments/batch_r2.sh
#
# Then the table (PLAN_R2.md section 5): see experiments/README.md, "Round 2".
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
CONFIGS=(
  r2_M1_s42 r2_M1_s43 r2_M1_s44
  r2_W1_s42 r2_W1_s43 r2_W1_s44
  r2_MW_s42 r2_MW_s43 r2_MW_s44
)
CFG="$WT/experiments/configs/r2/${CONFIGS[$SLURM_ARRAY_TASK_ID]}.yaml"
mkdir -p "$WT/experiments/logs" "$WT/experiments/results"
cd "$WT"
test -f "$CFG" || { echo "missing $CFG"; exit 1; }
test -d "$FIX/data_train_full" || { echo "missing $FIX/data_train_full"; exit 1; }
test -f "$WT/experiments/data/record_durations.csv" || { echo "missing record_durations.csv"; exit 1; }
echo "array task $SLURM_ARRAY_TASK_ID -> $(basename "$CFG")"

bash experiments/run_r2.sh "$CFG" "$FIX/data_train_full"
