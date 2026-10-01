#!/bin/bash
# Run one rung of the ablation ladder end to end.
#
#   experiments/run_ladder.sh <config.yaml> [train_dir] [test_dir] [testC_dir]
#
# Trains through the UNMODIFIED train_model.py, infers through the UNMODIFIED
# run_model.py on Test A (data_test) and Test C (data_test_C), then calls
# eval_report.py.  Everything runs under the 14.5 GiB T4 allocator cap so an OOM here
# means an OOM on the grader's T4.
#
# The config reaches team_code through $CHAGAS_CONFIG; train_model.py takes no config
# argument and is never edited.
#
# Called by experiments/batch_ladder_row*.sh; runnable directly on a GPU node too.
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python

CONFIG="${1:?usage: run_ladder.sh <config.yaml> [train_dir] [test_dir] [testC_dir]}"
TRAIN_DIR="${2:-$FIX/data_train}"
TEST_A="${3:-$FIX/data_test}"
TEST_C="${4:-$FIX/data_test_C}"

ROW="$(basename "$CONFIG" .yaml)"
RESULTS="$WT/experiments/results/$ROW"
MODEL="$RESULTS/model"
OUT_A="$RESULTS/outputs_A"
OUT_C="$RESULTS/outputs_C"

echo "===== ENV ====="
hostname
date
echo "row        : $ROW"
echo "config     : $CONFIG"
echo "train dir  : $TRAIN_DIR  ($(find "$TRAIN_DIR" -name '*.hea' | wc -l) records)"
echo "test A dir : $TEST_A     ($(find "$TEST_A" -name '*.hea' | wc -l) records)"
$PY -c "import torch; print('torch', torch.__version__, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', '| sm', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else '-')"

# Idempotent: every run starts from a clean output tree.
rm -rf "$MODEL" "$OUT_A" "$OUT_C"
mkdir -p "$MODEL" "$OUT_A" "$RESULTS"

export CHAGAS_CONFIG="$CONFIG"
export T4_CAP_GIB=14.5
cd "$WT"
# Provenance record (PLAN_R2.md section 6): commit, config hash, seed, host, versions.
$PY experiments/run_info.py --out "$RESULTS/run_info.json" --phase start --config "$CONFIG" --name "$ROW"

echo "===== TRAIN (unmodified train_model.py) ====="
$PY experiments/run_capped.py --cap "$T4_CAP_GIB" --label train --script train_model.py -- \
    -d "$TRAIN_DIR" -m "$MODEL" -v

echo "--- model folder ---"
ls -la "$MODEL"
echo "--- stray .ckpt outside the model folder (must be empty) ---"
# The judged container runs with WORKDIR /challenge; an implicit Lightning
# ModelCheckpoint there writes a full checkpoint per epoch into the repo root.
STRAY=$(find "$WT" -name '*.ckpt' -newermt '-2 days' -not -path "$WT/experiments/results/*" || true)
if [ -n "$STRAY" ]; then
  echo "FAIL: stray checkpoints outside model_folder:"
  echo "$STRAY"
  exit 1
fi
echo "(none)"
test ! -d "$MODEL/_work" || { echo "FAIL: temporary _work dir was not cleaned up"; exit 1; }

echo "===== INFERENCE: Test A (unmodified run_model.py) ====="
# No --allow_failures: a systematic error must surface, not silently become nan.
$PY experiments/run_capped.py --cap "$T4_CAP_GIB" --label infer_A --script run_model.py -- \
    -d "$TEST_A" -m "$MODEL" -o "$OUT_A"

if [ -d "$TEST_C" ]; then
  echo "===== INFERENCE: Test C (organiser-style perturbations) ====="
  mkdir -p "$OUT_C"
  $PY experiments/run_capped.py --cap "$T4_CAP_GIB" --label infer_C --script run_model.py -- \
      -d "$TEST_C" -m "$MODEL" -o "$OUT_C"
else
  echo "WARNING: $TEST_C does not exist; run experiments/batch_build_test_C.sh first."
  OUT_C=""
fi

echo "===== EVALUATE (organisers' evaluate_model.py, Test A) ====="
$PY evaluate_model.py -d "$TEST_A" -o "$OUT_A" -s "$RESULTS/scores_A.csv"
cat "$RESULTS/scores_A.csv"

echo "===== REPORT (A / B / C, per-source, per-age-decile, C-A gap) ====="
$PY experiments/eval_report.py --name "$ROW" \
    --outputs-a "$OUT_A" ${OUT_C:+--outputs-c "$OUT_C"} \
    --data "$TEST_A" --out-dir "$RESULTS"

$PY experiments/run_info.py --out "$RESULTS/run_info.json" --phase end --model-folder "$MODEL"

echo "===== DONE: $ROW ====="
date
