#!/bin/bash
#SBATCH --job-name=chagas_mem_smoke
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/memory_smoke.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/memory_smoke.err
#SBATCH --time=0-01:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# PLAN.md section 5 item 11 -- stage-2 memory smoke test.
#
# One stage-2 optimiser step (all 12 transformer layers unfrozen), batch 16 x 5000
# samples, fp16 AMP with GradScaler, grad-accumulated x4, under a 14.5 GiB CUDA
# allocator cap that emulates the grader's T4.  Reports
# torch.cuda.max_memory_allocated().  titanrtx is Turing SM 7.5, the same architecture
# as the T4; bf16 is deliberately NOT used because SM 7.5 has no native bfloat16.
#
# Run this BEFORE batch_ladder_row5.sh.  If the plain run exceeds the cap, the
# gradient-checkpointing run below is the fallback and
# `model.gradient_checkpointing: true` goes into the shipped config.
#
#   sbatch experiments/batch_memory_smoke.sh
set -uo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
mkdir -p "$WT/experiments/logs"
cd "$WT"

echo "===== ENV ====="
hostname
date
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY -c "import torch; print('torch', torch.__version__, '| sm', torch.cuda.get_device_capability(0))"

echo
echo "===== A. stage 2, NO gradient checkpointing ====="
$PY experiments/memory_smoke.py \
    --config "$WT/experiments/configs/row5_two_stage_ft.yaml" \
    --cap-gib 14.5 --batch-size 16 --samples 5000 --accumulate 4 --steps 3
PLAIN_RC=$?
echo "PLAIN_RC=$PLAIN_RC"

echo
echo "===== B. stage 2, WITH gradient checkpointing (the fallback) ====="
$PY experiments/memory_smoke.py \
    --config "$WT/experiments/configs/row5_two_stage_ft.yaml" \
    --cap-gib 14.5 --batch-size 16 --samples 5000 --accumulate 4 --steps 3 \
    --gradient-checkpointing
CKPT_RC=$?
echo "CKPT_RC=$CKPT_RC"

echo
echo "===== SUMMARY ====="
if [ "$PLAIN_RC" -eq 0 ]; then
  echo "stage 2 fits in 14.5 GiB without gradient checkpointing"
  exit 0
elif [ "$CKPT_RC" -eq 0 ]; then
  echo "stage 2 needs gradient checkpointing; set model.gradient_checkpointing: true"
  exit 0
else
  echo "FAIL: stage 2 does not fit even with gradient checkpointing"
  exit 1
fi
