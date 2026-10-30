#!/bin/bash
#SBATCH --job-name=chagas_layer_weights
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/layer_weights.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/layer_weights.err
#SBATCH --time=0-00:20:00

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=16g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2

# Read the learned 12-layer softmax aggregation weights out of every saved checkpoint
# (entry members first, then the round-3 B0 10 s-window models, then the rest) and write
# experiments/results/entry_final_r3/layer_weights.csv.  CPU only: torch.load on each
# ~362 MB checkpoint, no model is built and no forward is run.
set -euo pipefail
WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
cd "$WT"
echo "node: $(hostname)  torch: $($PY -c 'import torch; print(torch.__version__)')"
$PY experiments/layer_weights.py
echo "===== DONE ====="
