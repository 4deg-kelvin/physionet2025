#!/bin/bash
#SBATCH --job-name=chagas_durations
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/build_durations.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/build_durations.err
#SBATCH --time=0-03:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=30g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# PLAN_R2.md section 2: experiments/data/record_durations.csv from every WFDB header in
# split.csv (366,185 records), cross-checked in full against physionet2025/signal_lengths.csv.
# CPU only (header I/O over NFS); no GPU requested.
#
#   sbatch experiments/batch_build_durations.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
mkdir -p "$WT/experiments/logs" "$WT/experiments/data"
cd "$WT"
echo "===== ENV ====="; hostname; date
export CUDA_VISIBLE_DEVICES=
$PY experiments/build_durations.py --workers 32
echo "===== DONE ====="; date
