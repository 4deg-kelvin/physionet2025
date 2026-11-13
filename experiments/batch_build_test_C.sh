#!/bin/bash
#SBATCH --job-name=chagas_build_test_C
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/build_test_C.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/build_test_C.err
#SBATCH --time=0-08:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=60g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16

# Build Test C once (PLAN.md section 2): the held-out set perturbed the way the
# organisers perturbed the hidden cohorts -- random rate round trip, a random device
# band-pass, and Gaussian noise / baseline wander / powerline each with p=0.5.
#
# CPU only (pure DSP + WFDB I/O), but still launched through sbatch: it writes ~73,300
# records and must not run on the login node.  The perturbation is seeded from each
# record's own stem, so the output is reproducible and independent of worker count.
#
#   sbatch experiments/batch_build_test_C.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python
DEST=$FIX/data_test_C

mkdir -p "$WT/experiments/logs"
cd "$WT"

echo "===== ENV ====="
hostname
date
$PY -c "import torch, scipy, wfdb; print('torch', torch.__version__, '| scipy', scipy.__version__, '| wfdb', wfdb.__version__)"
echo "source records: $(find "$FIX/data_test" -name '*.hea' | wc -l)"

# Idempotent: rebuild from scratch so a partial previous run cannot leak through.
rm -rf "$DEST"
mkdir -p "$DEST"

$PY experiments/build_test_C.py \
    --source "$FIX/data_test" \
    --dest   "$DEST" \
    --workers 16

echo "===== VERIFY ====="
N_SRC=$(find "$FIX/data_test" -name '*.hea' | wc -l)
N_DST=$(find "$DEST" -name '*.hea' | wc -l)
echo "source $N_SRC -> dest $N_DST"
test "$N_SRC" -eq "$N_DST" || { echo "FAIL: record count mismatch"; exit 1; }

# The '#' header block must survive, or evaluate_model.py loses the labels.
$PY - "$FIX/data_test" "$DEST" <<'PYEOF'
import os, random, sys
src_root, dst_root = sys.argv[1], sys.argv[2]
stems = []
for root, _d, files in os.walk(dst_root):
    stems += [os.path.join(root, f)[:-4] for f in files if f.endswith('.hea')]
random.Random(0).shuffle(stems)
for dst in stems[:200]:
    rel = os.path.relpath(dst, dst_root)
    src = os.path.join(src_root, rel)
    a = [l for l in open(src + '.hea') if l.startswith('#')]
    b = [l for l in open(dst + '.hea') if l.startswith('#')]
    assert a == b, f'header comments differ for {rel}'
print('OK: header comment blocks match on a 200-record sample')
PYEOF

echo "===== DONE ====="
date
