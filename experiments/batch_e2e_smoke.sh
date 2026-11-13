#!/bin/bash
#SBATCH --job-name=chagas_e2e_smoke
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/e2e_smoke.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/e2e_smoke.err
#SBATCH --time=0-02:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# PLAN.md section 5 item 12 -- end-to-end smoke test on the 400-record set.
#
# The organisers' contract, unmodified and in order:
#     python train_model.py    -d testdata -m model -v
#     python run_model.py      -d testdata -m model -o outputs -v
#     python evaluate_model.py -d testdata -o outputs -s scores.csv
#
# Checks: training finishes with a tiny config (2 short epochs, both stages), no stray
# .ckpt outside model_folder, 400/400 records produce output files, the calibrated
# threshold (not a hard 0.5) is what the binary column uses, and evaluate_model.py runs.
# Everything under the 14.5 GiB T4 allocator cap.
#
#   sbatch experiments/batch_e2e_smoke.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
DATA=$FIX/testdata
RUN=$WT/experiments/results/e2e_smoke
MODEL=$RUN/model
OUT=$RUN/outputs

mkdir -p "$WT/experiments/logs"
rm -rf "$RUN"
mkdir -p "$MODEL" "$OUT"
cd "$WT"

echo "===== ENV ====="
hostname
date
$PY -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '| sm', torch.cuda.get_device_capability(0))"
N_RECORDS=$(find "$DATA" -name '*.hea' | wc -l)
echo "smoke records: $N_RECORDS"

export CHAGAS_CONFIG=$WT/experiments/configs/smoke.yaml
export T4_CAP_GIB=14.5

echo
echo "===== 1. train_model.py (unmodified) ====="
$PY experiments/run_capped.py --cap "$T4_CAP_GIB" --label train --script train_model.py -- \
    -d "$DATA" -m "$MODEL" -v

echo "--- model folder ---"
ls -la "$MODEL"
test -f "$MODEL/chagas_model.ckpt" || { echo "FAIL: no checkpoint produced"; exit 1; }

echo "--- stray .ckpt outside model_folder (must be empty) ---"
STRAY=$(find "$WT" -name '*.ckpt' -newermt '-1 day' -not -path "$MODEL/*" -not -name 'mimic_iv_ecg_finetuned.pt' | grep -v '/results/' || true)
if [ -n "$STRAY" ]; then echo "FAIL: stray checkpoints:"; echo "$STRAY"; exit 1; fi
echo "(none)"

echo "--- model folder must not contain the temporary work dir ---"
test ! -d "$MODEL/_work" || { echo "FAIL: _work was not cleaned up"; exit 1; }

echo
echo "===== 2. run_model.py (unmodified, no --allow_failures) ====="
$PY experiments/run_capped.py --cap "$T4_CAP_GIB" --label infer --script run_model.py -- \
    -d "$DATA" -m "$MODEL" -o "$OUT" -v | tail -20

N_OUT=$(find "$OUT" -name '*.txt' | wc -l)
echo "outputs written: $N_OUT / $N_RECORDS"
test "$N_OUT" -eq "$N_RECORDS" || { echo "FAIL: missing outputs"; exit 1; }

echo
echo "===== 3. threshold check ====="
$PY - "$MODEL" "$OUT" <<'PYEOF'
import glob, os, re, sys
import numpy as np, torch
model_folder, out_folder = sys.argv[1], sys.argv[2]
meta = torch.load(os.path.join(model_folder, 'chagas_model.ckpt'),
                  map_location='cpu', weights_only=False)['chagas']
threshold = float(meta['threshold'])
print('threshold stored in the checkpoint :', threshold)
print('learned layer weights              :', np.round(meta['layer_weights'], 4))
probs, binaries = [], []
for path in glob.glob(os.path.join(out_folder, '**', '*.txt'), recursive=True):
    text = open(path).read()
    probs.append(float(re.search(r'probability:\s*(\S+)', text, re.I).group(1)))
    binaries.append(int(float(re.search(r'label:\s*(\S+)', text, re.I).group(1))))
probs, binaries = np.array(probs), np.array(binaries)
print(f'probabilities: n={len(probs)} min={probs.min():.4f} max={probs.max():.4f} mean={probs.mean():.4f}')
print(f'positive rate in the binary column : {binaries.mean():.4f}')
expected = (probs > threshold).astype(int)
assert np.array_equal(expected, binaries), 'binary column does not match the stored threshold'
assert threshold != 0.5, 'threshold was never calibrated (still the hard-coded 0.5)'
print('OK: the binary column is exactly `prob > stored threshold`')
PYEOF

echo
echo "===== 4. evaluate_model.py (unmodified) ====="
$PY evaluate_model.py -d "$DATA" -o "$OUT" -s "$RUN/scores.csv"
cat "$RUN/scores.csv"

echo
echo "===== DONE: end-to-end smoke passed ====="
date
