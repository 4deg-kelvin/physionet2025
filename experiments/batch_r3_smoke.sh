#!/bin/bash
#SBATCH --job-name=chagas_r3_smoke
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r3_smoke.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r3_smoke.err
#SBATCH --time=0-03:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# Round-3 contract smoke on the 400-record set, configs/r3_smoke.yaml (fixed validation
# split, prevalence cap at 0.4 so it thins the smoke negatives, S7 monitor, 5 s
# multi-window), tiny budget:
#
#   1. unmodified train_model.py  -> checkpoint + manifest: split protocol, monitor (or
#      its val_challenge fallback), per-view metrics in the history, prevalence cap;
#      the fixed validation split recomputed from the headers is the same for two
#      training seeds and two caps;
#   2. unmodified run_model.py    -> 400/400 outputs, binary == prob > stored threshold;
#   3. unmodified evaluate_model.py;
#   4. predict_batched.py on the same records == run_model.py outputs (max |dprob| < 1e-4);
#   5. predict_batched.py --window-stride 2.5 (the run_r3.sh test-time windows) runs;
#   6. run_info.json start/end.
#
#   sbatch experiments/batch_r3_smoke.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
DATA=$FIX/testdata
CFG=$WT/experiments/configs/r3_smoke.yaml
RUN=$WT/experiments/results/r3_smoke
MODEL=$RUN/model
OUT=$RUN/outputs
CAP=14.5

mkdir -p "$WT/experiments/logs"
rm -rf "$RUN"
mkdir -p "$MODEL" "$OUT"
cd "$WT"

echo "===== ENV ====="
hostname
date
git -C "$WT" rev-parse HEAD
$PY -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '| sm', torch.cuda.get_device_capability(0))"
N_RECORDS=$(find "$DATA" -name '*.hea' | wc -l)
echo "smoke records: $N_RECORDS"
$PY experiments/run_info.py --out "$RUN/run_info.json" --phase start --config "$CFG"

echo
echo "===== 1. train_model.py (unmodified) ====="
export CHAGAS_CONFIG=$CFG
$PY experiments/run_capped.py --cap "$CAP" --label train --script train_model.py -- \
    -d "$DATA" -m "$MODEL" -v
unset CHAGAS_CONFIG
ls -la "$MODEL"
test -f "$MODEL/chagas_model.ckpt" || { echo "FAIL: no checkpoint"; exit 1; }
test ! -d "$MODEL/_work" || { echo "FAIL: _work was not cleaned up"; exit 1; }
STRAY=$(find "$WT" -name '*.ckpt' -newermt '-1 day' -not -path "$WT/experiments/results/*" || true)
if [ -n "$STRAY" ]; then echo "FAIL: stray checkpoints:"; echo "$STRAY"; exit 1; fi
$PY - "$MODEL" "$CFG" "$DATA" <<'PYEOF'
import copy, json, os, sys
from chagas import labels as L
from chagas.config import ExperimentCfg
from chagas.dataset import find_records, scan_records
from chagas.model import VIEW_METRICS
from chagas.training import _ptbxl_patients, _subsample, negative_budget, split_entries
model, cfg_path, data = sys.argv[1:4]
man = json.load(open(os.path.join(model, 'chagas_manifest.json')))
print('split:', man['split'])
assert man['split']['val_protocol'] == 'fixed' and man['split']['val_seed'] == 12345
assert man['split']['monitor'] in ('val_challenge_s7', 'val_challenge'), man['split']
for rec in man['training'][0]['history']:
    assert all(k in rec for k in VIEW_METRICS), rec
    print('history:', {k: rec.get(k) for k in ('stage', 'epoch', 'val_challenge') + tuple(VIEW_METRICS)})
print('best_by_stage:', man['training'][0]['best_by_stage'])
counts = {}
for row in man['data_summary']:
    counts[(row['split'], row['label'])] = counts.get((row['split'], row['label']), 0) + row['n']
    print('  ', row)
cfg = ExperimentCfg.from_yaml(cfg_path)
expected = negative_budget(counts[('train', 1)], cfg.data.cap_prevalence)
print(f"train: {counts[('train', 1)]} pos, {counts[('train', 0)]} neg (prevalence budget {expected})")
assert counts[('train', 0)] == expected or expected > counts[('train', 0)], 'prevalence cap not applied'
# The fixed validation split does not depend on the training seed or the cap.
table = L.Code15Table.from_csv(cfg.code15_exams_path, cfg.labels.aux_columns)
entries = scan_records(find_records(data), cfg, table=table, ptbxl_patients=_ptbxl_patients(cfg, data, True))
vals = []
for seed, p in ((42, 0.4), (43, 0.2)):
    c = copy.deepcopy(cfg)
    c.train.seed, c.data.cap_prevalence = seed, p
    tr, va = split_entries(entries, c.train.val_fraction, c.train.val_seed, protocol='fixed')
    _subsample(tr, c.data.max_records, c.train.seed, c.data.match_negatives_on,
               cap_mode=c.data.cap_mode, cap_prevalence=c.data.cap_prevalence)
    vals.append(sorted(e.record_id for e in va))
assert vals[0] == vals[1], 'fixed validation split moved with the seed or the cap'
n_val = counts.get(('val', 0), 0) + counts.get(('val', 1), 0)
assert len(vals[0]) == n_val, (len(vals[0]), n_val)
print(f'OK: fixed validation split ({n_val} records) identical across seeds and caps; '
      f'{len({e.group for e in entries if e.source == "ptbxl"})} PTB-XL groups for '
      f'{sum(e.source == "ptbxl" for e in entries)} PTB-XL records')
PYEOF

echo
echo "===== 2. run_model.py (unmodified, no --allow_failures) ====="
$PY experiments/run_capped.py --cap "$CAP" --label infer --script run_model.py -- \
    -d "$DATA" -m "$MODEL" -o "$OUT" -v | tail -5
N_OUT=$(find "$OUT" -name '*.txt' | wc -l)
echo "outputs written: $N_OUT / $N_RECORDS"
test "$N_OUT" -eq "$N_RECORDS" || { echo "FAIL: missing outputs"; exit 1; }

echo
echo "===== 3. evaluate_model.py (unmodified) ====="
$PY evaluate_model.py -d "$DATA" -o "$OUT" -s "$RUN/scores.csv"
cat "$RUN/scores.csv"

echo
echo "===== 4/5. predict_batched.py vs run_model.py; 2.5 s test-time windows ====="
$PY experiments/predict_batched.py -m "$MODEL" -d "$DATA" -o "$RUN/batched.csv" --cap-gib "$CAP"
$PY experiments/predict_batched.py -m "$MODEL" -d "$DATA" -o "$RUN/batched_tta.csv" --cap-gib "$CAP" --window-stride 2.5
$PY - "$MODEL" "$OUT" "$RUN" <<'PYEOF'
import glob, os, re, sys
import numpy as np, pandas as pd, torch
model, out, run = sys.argv[1:4]
threshold = float(torch.load(os.path.join(model, 'chagas_model.ckpt'), map_location='cpu', weights_only=False)['chagas']['threshold'])
rows = []
for path in glob.glob(os.path.join(out, '**', '*.txt'), recursive=True):
    text = open(path).read()
    rows.append((os.path.basename(path)[:-4],
                 float(re.search(r'probability:\s*(\S+)', text, re.I).group(1)),
                 int(float(re.search(r'label:\s*(\S+)', text, re.I).group(1)))))
ref = pd.DataFrame(rows, columns=['stem', 'prob_run_model', 'binary_run_model'])
assert np.array_equal((ref.prob_run_model > threshold).astype(int), ref.binary_run_model), 'binary != prob > threshold'
print(f'threshold {threshold:.6f}; binary column == prob > threshold; positive rate {ref.binary_run_model.mean():.4f}')
batched = pd.read_csv(os.path.join(run, 'batched.csv'), dtype={'stem': str})
m = ref.merge(batched, on='stem', how='outer')
assert len(m) == len(ref) == len(batched) == 400, (len(m), len(ref), len(batched))
delta = np.abs(m['prob'] - m['prob_run_model']).max()
print(f'predict_batched vs run_model.py: n={len(m)}  max |dprob| = {delta:.3e}')
assert delta < 1e-4, 'FAIL: batched inference differs from run_model.py'
tta = pd.read_csv(os.path.join(run, 'batched_tta.csv'), dtype={'stem': str}).set_index('stem')['prob']
assert len(tta) == 400 and np.isfinite(tta).all(), 'FAIL: test-time-window scoring lost records'
d = (tta - batched.set_index('stem')['prob']).abs()
print(f'2.5 s test-time windows vs default windows: mean |dprob| = {d.mean():.3e}, max = {d.max():.3e}')
print('OK: batched == run_model.py; test-time windows scored every record')
PYEOF

$PY experiments/run_info.py --out "$RUN/run_info.json" --phase end --model-folder "$MODEL"
echo
echo "===== DONE: round-3 smoke passed ====="
date
