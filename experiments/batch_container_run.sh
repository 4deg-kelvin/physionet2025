#!/bin/bash
#SBATCH --job-name=chagas_ctr_run
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/container_run.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/container_run.err
#SBATCH --time=0-04:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# Container check, part 2: run the entry exactly as the organisers do, inside the image
# built by batch_container_build.sh, on the 400-record smoke set with the DEFAULT config
# (no CHAGAS_CONFIG, no run_capped.py):
#   python train_model.py -d training_data -m model -v
#   python run_model.py   -d holdout_data  -m model -o holdout_outputs -v
#   python evaluate_model.py -d holdout_data -o holdout_outputs
# The container filesystem is read-only, so model/ and outputs/ are bind-mounted.
#
#   sbatch --dependency=afterok:<build job> experiments/batch_container_run.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
CTR=/juice2/scr2/kelvinkn/other_work/edwards/containers
RUN=$CTR/run_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RUN/model" "$RUN/outputs"
export APPTAINER_CACHEDIR=$CTR/cache
export APPTAINER_TMPDIR=$CTR/tmp

echo "host $(hostname)  gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)  run $RUN"
date
EXEC="apptainer exec --nv --containall --cleanenv --env PYTHONNOUSERSITE=1 --pwd /challenge \
  --bind $FIX/testdata:/challenge/training_data:ro \
  --bind $FIX/testdata:/challenge/holdout_data:ro \
  --bind $RUN/model:/challenge/model \
  --bind $RUN/outputs:/challenge/holdout_outputs \
  $CTR/chagas.sif"
$EXEC python train_model.py -d training_data -m model -v
date
$EXEC python run_model.py -d holdout_data -m model -o holdout_outputs -v
$EXEC python evaluate_model.py -d holdout_data -o holdout_outputs
date
n_out=$(ls "$RUN/outputs"/*.txt | wc -l)
echo "outputs: $n_out / 400"
[ "$n_out" -eq 400 ] && echo "RUN OK"
