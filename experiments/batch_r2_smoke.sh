#!/bin/bash
#SBATCH --job-name=chagas_r2_smoke
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r2_smoke.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/r2_smoke.err
#SBATCH --time=0-03:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# Round-2 contract smoke (PLAN_R2.md section 7 / brief item 6), on the 400-record set,
# with every new flag ON (configs/smoke_r2.yaml: length-matched negatives, 5 s
# multi-window, WiSE-FT select with one folder per alpha), tiny budget:
#
#   0. stage-2 memory at 5 s windows (batch 16 x 2500) under the 14.5 GiB cap;
#   1. unmodified train_model.py  -> checkpoint, manifest, wise_ft/member0/alpha*/;
#   2. unmodified run_model.py    -> 400/400 outputs, binary == prob > stored threshold;
#   3. unmodified evaluate_model.py;
#   4. predict_batched.py on the same records == run_model.py outputs (max |dprob| < 1e-4);
#   5. predict_batched.py on the selected alpha folder == the shipped model;
#   6. run_info.json start/end.
#
#   sbatch experiments/batch_r2_smoke.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
DATA=$FIX/testdata
CFG=$WT/experiments/configs/smoke_r2.yaml
RUN=$WT/experiments/results/r2_smoke
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
echo "===== 0. stage-2 memory with 5 s windows (r2_MW_s42 config) ====="
$PY experiments/memory_smoke.py --config "$WT/experiments/configs/r2/r2_MW_s42.yaml" \
    --samples 2500 --batch-size 16 --steps 3 --cap-gib "$CAP"

echo
echo "===== 1. train_model.py (unmodified) ====="
export CHAGAS_CONFIG=$CFG
$PY experiments/run_capped.py --cap "$CAP" --label train --script train_model.py -- \
    -d "$DATA" -m "$MODEL" -v
unset CHAGAS_CONFIG
ls -la "$MODEL" "$MODEL"/wise_ft/member0
test -f "$MODEL/chagas_model.ckpt" || { echo "FAIL: no checkpoint"; exit 1; }
test ! -d "$MODEL/_work" || { echo "FAIL: _work was not cleaned up"; exit 1; }
STRAY=$(find "$WT" -name '*.ckpt' -newermt '-1 day' -not -path "$WT/experiments/results/*" || true)
if [ -n "$STRAY" ]; then echo "FAIL: stray checkpoints:"; echo "$STRAY"; exit 1; fi
for A in alpha0.00 alpha0.50 alpha1.00; do
  test -f "$MODEL/wise_ft/member0/$A/chagas_model.ckpt" || { echo "FAIL: missing $A"; exit 1; }
done
$PY - "$MODEL" <<'PYEOF'
import json, os, sys, torch
model = sys.argv[1]
man = json.load(open(os.path.join(model, 'chagas_manifest.json')))
wise = man['training'][0]['wise_ft']
print('WiSE-FT per-alpha validation:')
for r in wise['alphas']:
    print(f"  alpha={r['alpha']:.2f} val_challenge={r['val_challenge']:.4f} val_auroc={r['val_auroc']:.4f} "
          f"threshold={r['threshold']:.4f} windows/record={r.get('windows_per_record')}")
print('selected alpha:', wise['selected_alpha'])
meta = torch.load(os.path.join(model, 'chagas_model.ckpt'), map_location='cpu', weights_only=False)['chagas']
assert meta['wise_ft_alpha'] == wise['selected_alpha'], 'checkpoint and manifest disagree on alpha'
print('data_summary (split, source, label, length_group, n):')
for row in man['data_summary']:
    print('  ', row)
neg = {}; pos = {}
for row in man['data_summary']:
    d = pos if row['label'] == 1 else neg
    d[row['length_group']] = d.get(row['length_group'], 0) + row['n']
print('positives by length group:', pos, '| negatives by length group:', neg)
for line in man['training'][0]['history'][:3]:
    print('history:', line)
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
echo "===== 4/5. predict_batched.py vs run_model.py; selected alpha folder vs shipped model ====="
$PY experiments/predict_batched.py -m "$MODEL" -d "$DATA" -o "$RUN/batched.csv" --cap-gib "$CAP"
SELECTED=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print('alpha%.2f' % m['training'][0]['wise_ft']['selected_alpha'])" "$MODEL/chagas_manifest.json")
for A in alpha0.00 alpha0.50 alpha1.00; do
  $PY experiments/predict_batched.py -m "$MODEL/wise_ft/member0/$A" -d "$DATA" -o "$RUN/batched_$A.csv" --cap-gib "$CAP"
done
$PY - "$MODEL" "$OUT" "$RUN" "$SELECTED" <<'PYEOF'
import glob, os, re, sys
import numpy as np, pandas as pd, torch
model, out, run, selected = sys.argv[1:5]
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
sel = pd.read_csv(os.path.join(run, f'batched_{selected}.csv'), dtype={'stem': str})
d2 = np.abs(sel.set_index('stem')['prob'] - batched.set_index('stem')['prob']).max()
print(f'selected mixture {selected} folder vs shipped model: max |dprob| = {d2:.3e}')
assert d2 < 1e-6, 'FAIL: the shipped model is not the selected mixture'
for a in ('alpha0.00', 'alpha0.50', 'alpha1.00'):
    p = pd.read_csv(os.path.join(run, f'batched_{a}.csv'))['prob']
    print(f'  {a}: mean prob {p.mean():.4f}, sd {p.std():.4f}')
print('OK: batched == run_model.py, shipped == selected mixture')
PYEOF

$PY experiments/run_info.py --out "$RUN/run_info.json" --phase end --model-folder "$MODEL"
echo
echo "===== DONE: round-2 smoke passed ====="
date
