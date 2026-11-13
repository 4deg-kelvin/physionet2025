#!/bin/bash
#SBATCH --job-name=chagas_row6
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/row6.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/row6.err
#SBATCH --time=2-00:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# PLAN section 2 row 6: + auxiliary RBBB/1dAVb/LBBB/AF/SB heads (optional).
#
# titanrtx is Turing SM 7.5, the same architecture as the grader's T4; the CUDA
# allocator is capped to 14.5 GiB inside run_ladder.sh so OOM behaviour is
# representative.  Trains through the UNMODIFIED train_model.py, infers through the
# UNMODIFIED run_model.py on Test A and Test C, then writes
# experiments/results/row6_aux_heads/report.md.
#
#   sbatch experiments/batch_ladder_row6.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
mkdir -p "$WT/experiments/logs" "$WT/experiments/results"
cd "$WT"

bash experiments/run_ladder.sh "$WT/experiments/configs/row6_aux_heads.yaml"
