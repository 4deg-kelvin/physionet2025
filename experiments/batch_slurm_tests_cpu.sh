#!/bin/bash
#SBATCH --job-name=chagas_pytest_cpu
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/pytest_cpu.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/pytest_cpu.err
#SBATCH --time=0-01:30:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10

# The whole pytest suite on a CPU-only node (no GPU requested, CUDA hidden).  Every test
# runs on CPU by design, and CPU jobs start quickly when the GPUs are busy.  Same suite
# as batch_slurm_tests.sh, which runs it on a TitanRTX node.
#
#   sbatch experiments/batch_slurm_tests_cpu.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
mkdir -p "$WT/experiments/logs"
cd "$WT"
export CUDA_VISIBLE_DEVICES=

echo "===== ENV ====="
hostname
date
git -C "$WT" rev-parse HEAD
$PY -c "import torch, numpy, scipy, lightning; print('torch', torch.__version__, '| numpy', numpy.__version__, '| lightning', lightning.__version__, '| cuda', torch.cuda.is_available())"

echo "===== pytest ====="
set +e
$PY -m pytest "$WT/tests" -v --tb=short -p no:cacheprovider -rs
RC=$?
set -e
echo "PYTEST_RC=$RC"
date
exit $RC
