#!/bin/bash
#SBATCH --job-name=chagas_r3_table
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r3_table.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r3_table.err
#SBATCH --time=0-03:00:00

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Build the round-3 comparison table from saved predictions, decided by the round-3 rule
# (aggregate.py --rule r3: primary S10+ AUROC, secondary S7 TPR@5%).  W1x5 is W1 over all
# five seeds; W1tta is W1's first three seeds re-scored with 2.5 s test-time windows
# ("<run>:tta" -> predictions_<A|C>_tta.csv).
# CPU-only: bootstrap resampling over the test records, no model is run.
#   sbatch --dependency=afterany:<array id> experiments/batch_r3_table.sh
set -euo pipefail
WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
cd "$WT"
for W in A C; do
  $PY experiments/aggregate.py \
    --group B0=r3_B0_s42,r3_B0_s43,r3_B0_s44 \
    --group W1=r3_W1_s42,r3_W1_s43,r3_W1_s44 \
    --group W1x5=r3_W1_s42,r3_W1_s43,r3_W1_s44,r3_W1_s45,r3_W1_s46 \
    --group W1R=r3_W1R_s42,r3_W1R_s43,r3_W1R_s44 \
    --group W1S=r3_W1S_s42,r3_W1S_s43,r3_W1S_s44 \
    --group W1tta=r3_W1_s42:tta,r3_W1_s43:tta,r3_W1_s44:tta \
    --baseline B0 --rule r3 --which $W --out experiments/results/R3_TABLE_$W
done
echo "===== DONE ====="
