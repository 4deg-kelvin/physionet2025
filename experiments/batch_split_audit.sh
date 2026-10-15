#!/bin/bash
#SBATCH --job-name=chagas_split_audit
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/split_audit.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/split_audit.err
#SBATCH --time=0-00:30:00

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=16g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Audit the held-out split (experiments/check_split.py): no patient group in both splits,
# no duplicate stems, every record has a duration; counts and prevalence by source x split
# x length group.  CPU-only, reads two CSVs.
#   sbatch experiments/batch_split_audit.sh
set -euo pipefail
WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
cd "$WT"
hostname
date
$PY experiments/check_split.py --out experiments/results/SPLIT_AUDIT
echo "===== DONE ====="
