#!/bin/bash
#SBATCH --job-name=chagas_full_final
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/full_final.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/full_final.err
#SBATCH --time=4-00:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# The final configuration on the FULL patient-grouped training split (292,885 records),
# run twice with different seeds so the report can quote mean +/- sd (PLAN.md section 2:
# "Final config: >= 2 seeds; report mean +/- sd").
#
# Run this only after the ladder has justified the configuration.  Each seed gets the
# full 71 h budget internally (TimeLimitCallback), and the SLURM wall clock allows for
# both plus inference.
#
#   sbatch experiments/batch_full_final.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
BASE_CFG=$WT/experiments/configs/final_full.yaml

mkdir -p "$WT/experiments/logs"
cd "$WT"

echo "===== ENV ====="
hostname
date
$PY -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '| sm', torch.cuda.get_device_capability(0))"

# The full training split lives in split.csv, not in data_train (which is the 100k
# subsample).  Materialise it once if it is not there yet.
FULL_TRAIN=$FIX/data_train_full
if [ ! -d "$FULL_TRAIN" ]; then
  echo "===== materialising the full training split ====="
  $PY - "$FIX" "$FULL_TRAIN" <<'PYEOF'
import os, sys
import pandas as pd
fix, dest = sys.argv[1], sys.argv[2]
df = pd.read_csv(f'{fix}/split/split.csv', low_memory=False)
train = df[df.split == 'train']
def subdir(source):
    if str(source).startswith('CODE'): return 'code15'
    if str(source).startswith('PTB'):  return 'ptb'
    return 'samitrop'
for d in ('code15', 'ptb', 'samitrop'):
    os.makedirs(f'{dest}/{d}', exist_ok=True)
n = 0
for r in train.itertuples():
    out = f'{dest}/{subdir(r.source)}'
    stem = os.path.basename(r.path)
    for ext in ('.hea', '.mat', '.dat'):
        src = r.path + ext
        if os.path.exists(src):
            dst = f'{out}/{stem}{ext}'
            if not os.path.lexists(dst):
                os.symlink(src, dst)
    n += 1
print(f'materialised {n:,} training records into {dest}', flush=True)
PYEOF
fi
echo "full train records: $(find "$FULL_TRAIN" -name '*.hea' | wc -l)"

for SEED in 42 43; do
  CFG=$WT/experiments/results/final_full_seed${SEED}.yaml
  mkdir -p "$WT/experiments/results"
  $PY - "$BASE_CFG" "$CFG" "$SEED" <<'PYEOF'
import sys
sys.path.insert(0, '/juice2/scr2/kelvinkn/other_work/edwards/improve_wt')
from chagas.config import ExperimentCfg
base, out, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
cfg = ExperimentCfg.from_yaml(base)
cfg.name = f'final_full_seed{seed}'
cfg.train.seed = seed
cfg.to_yaml(out)
print(f'wrote {out}', flush=True)
PYEOF
  echo "===== SEED $SEED ====="
  bash experiments/run_ladder.sh "$CFG" "$FULL_TRAIN"
done

echo "===== SUMMARY ACROSS SEEDS ====="
$PY - <<'PYEOF'
import glob, os
import pandas as pd
rows = []
for path in sorted(glob.glob('/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/results/final_full_seed*/summary.csv')):
    df = pd.read_csv(path)
    df['run'] = os.path.basename(os.path.dirname(path))
    rows.append(df)
if rows:
    allr = pd.concat(rows)
    print(allr.to_string(index=False))
    print()
    print(allr.groupby('view')[['challenge', 'auroc', 'auprc']].agg(['mean', 'std']).to_string())
PYEOF

echo "===== DONE ====="
date
