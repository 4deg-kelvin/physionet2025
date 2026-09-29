#!/bin/bash
#SBATCH --job-name=chagas_lrsweep
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/lrsweep_%a.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/lrsweep_%a.err
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

# Stage-2 learning-rate sweep on the SHIPPED config (finalA_negcap), 3 seeds per rate
# (DECISIONS.md D-2026-09-29-4).  2e-5 seed 42 is the existing finalA_negcap result, so
# the array holds 2e-5 x {43,44}, 5e-6 x {42,43,44} and 1e-6 x {42,43,44}.  At most 4
# tasks run at once (%4) to leave jag-standard GPUs for others.  Each task trains on the
# full 292,885-record folder through the UNMODIFIED train_model.py (negatives capped by
# the config, as in the shipped default), then scores Test A and Test C.
#
#   sbatch experiments/batch_lr_sweep.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
CONFIGS=(
  lrsweep_lr2e5_s43 lrsweep_lr2e5_s44
  lrsweep_lr5e6_s42 lrsweep_lr5e6_s43 lrsweep_lr5e6_s44
  lrsweep_lr1e6_s42 lrsweep_lr1e6_s43 lrsweep_lr1e6_s44
)
CFG="$WT/experiments/configs/lrsweep/${CONFIGS[$SLURM_ARRAY_TASK_ID]}.yaml"
mkdir -p "$WT/experiments/logs" "$WT/experiments/results"
cd "$WT"
test -f "$CFG" || { echo "missing $CFG"; exit 1; }
test -d "$FIX/data_train_full" || { echo "missing $FIX/data_train_full"; exit 1; }
echo "array task $SLURM_ARRAY_TASK_ID -> $(basename "$CFG")"

bash experiments/run_ladder.sh "$CFG" "$FIX/data_train_full"
