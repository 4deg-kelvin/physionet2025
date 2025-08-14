#!/bin/bash

# Grid of auxiliary loss weights
RBBB_WEIGHTS="0.0 0.1 0.25 0.5"
D1AVB_WEIGHTS="0.0 0.1 0.25 0.5"

# Fixed training params
DATA_DIR="../training_data"
EPOCHS=15
BATCH_SIZE=32
LR=1e-4
WD=1e-5

for w_rbbb in $RBBB_WEIGHTS; do
  for w_d1avb in $D1AVB_WEIGHTS; do
    # only consider combos whose sum ≤ 1.0
    sum=$(echo "$w_rbbb + $w_d1avb" | bc -l)
    valid=$(echo "$sum <= 1.0" | bc -l)
    if [ "$valid" -eq 1 ]; then
      echo "Submitting job: RBBB=$w_rbbb, 1dAVb=$w_d1avb"
      sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=se_R${w_rbbb}_A${w_d1avb}
#SBATCH --output=logs/se_R${w_rbbb}_A${w_d1avb}.out
#SBATCH --error=logs/se_R${w_rbbb}_A${w_d1avb}.err
#SBATCH --account=nlp
#SBATCH --time=2-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu
#SBATCH --partition=sphinx
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1

# Activate environment & cd
source ~/juice/miniconda3/etc/profile.d/conda.sh
# conda activate physionet
cd /sailhome/kelvinkn/scr2_juice/other_work/edwards/physionet2025/se_blocks

# Launch training with multi-task weights
srun python train.py \\
    --data-dir "$DATA_DIR" \\
    --epochs "$EPOCHS" \\
    --batch-size "$BATCH_SIZE" \\
    --lr "$LR" \\
    --weight-decay "$WD" \\
    --do-multitask \\
    --rbbb-loss-weight "$w_rbbb" \\
    --d1avb-loss-weight "$w_d1avb"
EOF
    else
      echo "Skipping invalid combo: RBBB=$w_rbbb, 1dAVb=$w_d1avb"
    fi
  done
done

echo "All grid‐search jobs submitted."
