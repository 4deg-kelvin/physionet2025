#!/bin/bash
#SBATCH --job-name=chagas_finalA
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/finalA.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/finalA.err
#SBATCH --time=2-00:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# Final candidate A: 5c config, all positives + capped negatives (data.max_records=100000) -- DECISIONS.md D-2026-09-29-2
#
# Trains on the FULL 292,885-record patient-grouped training split, as the grader will
# (it retrains our code on the whole public set), through the UNMODIFIED train_model.py,
# infers with run_model.py on Test A and Test C, and writes
# experiments/results/finalA_negcap/report.md.  T4 memory cap applied inside run_ladder.sh.
#
#   sbatch experiments/batch_finalA.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
mkdir -p "$WT/experiments/logs" "$WT/experiments/results"
cd "$WT"
test -d "$FIX/data_train_full" || { echo "missing $FIX/data_train_full (built by batch_full_final.sh)"; exit 1; }

bash experiments/run_ladder.sh "$WT/experiments/configs/finalA_negcap.yaml" "$FIX/data_train_full"
