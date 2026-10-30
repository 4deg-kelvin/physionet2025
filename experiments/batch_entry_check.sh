#!/bin/bash
#SBATCH --job-name=chagas_entry
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/entry_check.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/entry_check.err
#SBATCH --time=1-06:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:titanrtx:1

# Time-budget and end-to-end check of the ENTRY (PLAN_R3.md section 9.2): the unmodified
# train_model.py with the DEFAULT config (no CHAGAS_CONFIG) on the full training split, then
# run_model.py on the held-out set, exactly as the organisers run it, under the 14.5 GiB cap.
# Records per-member wall time so the T4 projection (x2.5) can be checked against 72 h.
#
#   sbatch --dependency=afterok:<tests> experiments/batch_entry_check.sh
set -euo pipefail
WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
RUN=$WT/experiments/results/entry_final_r3
MODEL=$RUN/model
OUT=$RUN/outputs
rm -rf "$RUN"; mkdir -p "$MODEL" "$OUT"
cd "$WT"
hostname; date; git rev-parse HEAD
$PY experiments/run_info.py --out "$RUN/run_info.json" --phase start --config experiments/configs/final_r3.yaml
T0=$(date +%s)
$PY experiments/run_capped.py --cap 14.5 --label train --script train_model.py -- -d "$FIX/data_train_full" -m "$MODEL" -v
T1=$(date +%s)
echo "TRAIN_SECONDS $((T1-T0))  members: $(ls "$MODEL"/chagas_model*.ckpt | wc -l)"
$PY experiments/predict_batched.py -m "$MODEL" -d "$FIX/data_test" -o "$RUN/raw_predictions_A.csv" --cap-gib 14.5
$PY experiments/predict_batched.py -m "$MODEL" -d "$FIX/data_test_C" -o "$RUN/raw_predictions_C.csv" --cap-gib 14.5
$PY experiments/eval_report.py --name entry_final_r3 --preds-a "$RUN/raw_predictions_A.csv" \
    --preds-c "$RUN/raw_predictions_C.csv" --data "$FIX/data_test" --out-dir "$RUN" > "$RUN/eval_report.log"
sed -n '1,30p' "$RUN/report.md"
$PY experiments/run_info.py --out "$RUN/run_info.json" --phase end
T2=$(date +%s)
echo "INFER_SECONDS $((T2-T1))"
$PY - "$MODEL" <<'PYEOF'
import json, sys
m = json.load(open(sys.argv[1] + "/chagas_manifest.json"))
print("members:", m["members"]); print("split:", m.get("split"))
for i, t in enumerate(m["training"]):
    print("member", i, "best_by_stage:", {k: round(v["val_challenge_s7"], 4) for k, v in t["best_by_stage"].items()})
PYEOF
echo "ENTRY CHECK OK"
