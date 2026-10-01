#!/bin/bash
# One round-2 run end to end (PLAN_R2.md section 4).
#
#   experiments/run_r2.sh <config.yaml> [train_dir] [test_dir] [testC_dir]
#
# 1. run_info.json (start): git commit/dirty, config SHA-256, seed, host, GPU, versions.
# 2. Train through the UNMODIFIED train_model.py (config via $CHAGAS_CONFIG), under the
#    14.5 GiB T4 allocator cap; fail on stray checkpoints or a leftover _work dir.
# 3. Score Test A and Test C with experiments/predict_batched.py (batched, same
#    probabilities as run_model.py -- tests/test_predict_batched.py), then
#    eval_report.py with the round-2 views (S7 / S10 / S10+, CIs, length-cue gap).
#    With WiSE-FT (model/wise_ft/member0/alpha*/), every mixture is scored into
#    results/<run>/alpha<a>/ and the shipped (selected) mixture's predictions are
#    copied to results/<run>/ and reported there.
# 4. run_info.json (end): end time, train/val counts by source x label x length group.
#
# Called by experiments/batch_r2.sh; runnable directly on a GPU node too.
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python

CONFIG="${1:?usage: run_r2.sh <config.yaml> [train_dir] [test_dir] [testC_dir]}"
TRAIN_DIR="${2:-$FIX/data_train_full}"
TEST_A="${3:-$FIX/data_test}"
TEST_C="${4:-$FIX/data_test_C}"

ROW="$(basename "$CONFIG" .yaml)"
RESULTS="$WT/experiments/results/$ROW"
MODEL="$RESULTS/model"
CAP=14.5

echo "===== ENV ====="
hostname
date
echo "run        : $ROW"
echo "config     : $CONFIG"
echo "train dir  : $TRAIN_DIR"
$PY -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# Idempotent: every run starts from a clean output tree.
rm -rf "$RESULTS"
mkdir -p "$MODEL"
cd "$WT"
$PY experiments/run_info.py --out "$RESULTS/run_info.json" --phase start --config "$CONFIG" --name "$ROW"

echo "===== TRAIN (unmodified train_model.py) ====="
export CHAGAS_CONFIG="$CONFIG"
$PY experiments/run_capped.py --cap "$CAP" --label train --script train_model.py -- \
    -d "$TRAIN_DIR" -m "$MODEL" -v
unset CHAGAS_CONFIG

ls -la "$MODEL"
STRAY=$(find "$WT" -name '*.ckpt' -newermt '-2 days' -not -path "$WT/experiments/results/*" || true)
if [ -n "$STRAY" ]; then echo "FAIL: stray checkpoints outside model_folder:"; echo "$STRAY"; exit 1; fi
test ! -d "$MODEL/_work" || { echo "FAIL: temporary _work dir was not cleaned up"; exit 1; }

# score <model_folder> <out_dir>: batched inference on A and C, then the full report.
score() {
  local model="$1" out="$2"
  mkdir -p "$out"
  $PY experiments/predict_batched.py -m "$model" -d "$TEST_A" -o "$out/raw_predictions_A.csv" --cap-gib "$CAP"
  $PY experiments/predict_batched.py -m "$model" -d "$TEST_C" -o "$out/raw_predictions_C.csv" --cap-gib "$CAP"
  $PY experiments/eval_report.py --name "$ROW${3:+ $3}" --preds-a "$out/raw_predictions_A.csv" \
      --preds-c "$out/raw_predictions_C.csv" --data "$TEST_A" --out-dir "$out" > "$out/eval_report.log"
  sed -n '1,30p' "$out/report.md"
}

WISE="$MODEL/wise_ft/member0"
if [ -d "$WISE" ]; then
  echo "===== SCORE every WiSE-FT mixture ====="
  for ALPHA_DIR in "$WISE"/alpha*; do
    ALPHA="$(basename "$ALPHA_DIR")"
    echo "--- $ALPHA ---"
    score "$ALPHA_DIR" "$RESULTS/$ALPHA" "($ALPHA)"
  done
  SELECTED=$($PY -c "import json,sys; m=json.load(open(sys.argv[1])); print('alpha%.2f' % m['training'][0]['wise_ft']['selected_alpha'])" "$MODEL/chagas_manifest.json")
  echo "===== shipped mixture: $SELECTED (selected on validation) ====="
  cp "$RESULTS/$SELECTED/raw_predictions_A.csv" "$RESULTS/$SELECTED/raw_predictions_C.csv" "$RESULTS/"
  $PY experiments/eval_report.py --name "$ROW (selected $SELECTED)" \
      --preds-a "$RESULTS/raw_predictions_A.csv" --preds-c "$RESULTS/raw_predictions_C.csv" \
      --data "$TEST_A" --out-dir "$RESULTS" > "$RESULTS/eval_report.log"
  echo "$SELECTED" > "$RESULTS/selected_alpha.txt"
else
  echo "===== SCORE ====="
  score "$MODEL" "$RESULTS"
fi

$PY experiments/run_info.py --out "$RESULTS/run_info.json" --phase end --model-folder "$MODEL"
echo "===== DONE: $ROW ====="
date
