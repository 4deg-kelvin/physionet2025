#!/bin/bash
#SBATCH --job-name=chagas_r2_table
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r2_table.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r2_table.err
#SBATCH --time=0-03:00:00

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=32g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Build the round-2 comparison table (PLAN_R2.md section 5) from saved predictions.
# CPU-only: bootstrap resampling over the test records, no model is run.
#   sbatch --dependency=afterany:<array id> experiments/batch_r2_table.sh
set -euo pipefail
WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
cd "$WT"
for W in A C; do
  $PY experiments/aggregate.py \
    --group B0=finalA_negcap,lrsweep_lr2e5_s43,lrsweep_lr2e5_s44 \
    --group M1=r2_M1_s42,r2_M1_s43,r2_M1_s44 \
    --group W1=r2_W1_s42,r2_W1_s43,r2_W1_s44 \
    --group MW=r2_MW_s42,r2_MW_s43,r2_MW_s44 \
    --group MW_a0.00=r2_MW_s42/alpha0.00,r2_MW_s43/alpha0.00,r2_MW_s44/alpha0.00 \
    --group MW_a0.50=r2_MW_s42/alpha0.50,r2_MW_s43/alpha0.50,r2_MW_s44/alpha0.50 \
    --group MW_a1.00=r2_MW_s42/alpha1.00,r2_MW_s43/alpha1.00,r2_MW_s44/alpha1.00 \
    --baseline B0 --which $W --out experiments/results/R2_TABLE_$W
done
echo "===== DONE ====="
