#!/bin/bash
#SBATCH --job-name=chagas_pytest
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/pytest.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/pytest.err
#SBATCH --time=0-01:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# The whole pytest suite (PLAN.md section 5 items 1-10 and 13) on a GPU node.
#
# All of it runs on CPU and passes on the login node too -- the two tests that touch the
# real 85M-parameter backbone do one tiny forward on two short records.  This script
# exists so the suite can also be verified on the *same* hardware the ladder runs on,
# with CUDA present, before a long GPU job is queued behind it.
#
# Items 11 and 12 are NOT pytest cases: they need a GPU by construction and live in
# batch_memory_smoke.sh and batch_e2e_smoke.sh.  They are run here too, in report-only
# mode, so one sbatch verifies everything that can be verified quickly.
#
#   sbatch experiments/batch_slurm_tests.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
mkdir -p "$WT/experiments/logs"
cd "$WT"

echo "===== ENV ====="
hostname
date
$PY -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', '| sm', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '-')"
$PY -c "import pytest, lightning, scipy, numpy; print('pytest', pytest.__version__, '| lightning', lightning.__version__, '| scipy', scipy.__version__, '| numpy', numpy.__version__)"

echo
echo "===== pytest (PLAN section 5 items 1-10, 13) ====="
$PY -m pytest "$WT/tests" -v --tb=short -p no:cacheprovider
PYTEST_RC=$?
echo "PYTEST_RC=$PYTEST_RC"

echo
echo "===== config round trip for every ladder row ====="
$PY - <<'PYEOF'
import glob, sys
sys.path.insert(0, '/juice2/scr2/kelvinkn/other_work/edwards/improve_wt')
from chagas.config import ExperimentCfg
for path in sorted(glob.glob('/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/configs/*.yaml')):
    cfg = ExperimentCfg.from_yaml(path)
    assert ExperimentCfg.from_dict(cfg.to_dict()).to_dict() == cfg.to_dict(), path
    print(f'OK  {cfg.name:<28} {cfg.description}')
PYEOF

echo
echo "===== DONE ====="
date
exit $PYTEST_RC
