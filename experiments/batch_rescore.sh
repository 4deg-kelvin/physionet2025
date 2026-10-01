#!/bin/bash
#SBATCH --job-name=chagas_rescore
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/rescore.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/rescore.err
#SBATCH --time=0-03:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# PLAN_R2.md section 2: re-score every saved run with the round-2 views into
# experiments/results/RESCORED.md, then check the tools against the hand-computed
# acceptance numbers.  CPU only; reads saved predictions, runs no model.
#
#   sbatch experiments/batch_rescore.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
cd "$WT"
export CUDA_VISIBLE_DEVICES=
echo "===== ENV ====="; hostname; date; git -C "$WT" rev-parse HEAD
echo "===== acceptance checks ====="
$PY experiments/acceptance_r2.py || echo "ACCEPTANCE: at least one check differs (see above)"
echo "===== rescore every run ====="
$PY experiments/rescore_all.py
echo "===== DONE ====="; date
