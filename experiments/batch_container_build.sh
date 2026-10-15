#!/bin/bash
#SBATCH --job-name=chagas_ctr_build
#SBATCH --output=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/container_build.out
#SBATCH --error=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt/experiments/logs/container_build.err
#SBATCH --time=0-05:00:00

#SBATCH --mail-type=ALL
#SBATCH --mail-user=kelvinkn@stanford.edu

#SBATCH --account=nlp
#SBATCH --partition=jag-standard
#SBATCH --mem=48g
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8

# Container check, part 1 (PLAN_R3.md section 9.3).  No Docker daemon is reachable from
# this cluster (permission denied on docker.sock, also for Apptainer's buildkit: frontend),
# so the image is built rootless by Apptainer from experiments/container/chagas.def, a
# step-for-step rendering of the unmodified Dockerfile.  The build context is a staged
# copy of the worktree filtered with .dockerignore.  If fakeroot emulation cannot run
# apt-get, the pytorch-base fallback definition is built instead and the log says so.
# --ignore-fakeroot-command: the host fakeroot library needs glibc 2.38 (Ubuntu 22.04 has
# 2.35), so %post runs in a root-mapped user namespace instead.
#
#   sbatch experiments/batch_container_build.sh
set -euo pipefail

WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt
CTR=/juice2/scr2/kelvinkn/other_work/edwards/containers
export APPTAINER_CACHEDIR=$CTR/cache
export APPTAINER_TMPDIR=$CTR/tmp
export TMPDIR=$CTR/tmp
mkdir -p "$CTR/cache" "$CTR/tmp"

echo "host $(hostname)  apptainer $(apptainer --version)  commit $(git -C "$WT" rev-parse --short HEAD) dirty=$(git -C "$WT" status --porcelain | grep -v '^??' | wc -l)"
date
# Stage the build context the way `COPY ./ /challenge` + .dockerignore would see it.
rm -rf "$CTR/stage"
grep -vE '^\s*(#|$)' "$WT/.dockerignore" > "$CTR/stage_exclude.txt"
rsync -a --exclude-from="$CTR/stage_exclude.txt" "$WT/" "$CTR/stage/"
du -sh "$CTR/stage"
rm -f "$CTR/chagas.sif" "$CTR/chagas.variant"

if apptainer build --fakeroot --ignore-fakeroot-command "$CTR/chagas.sif" "$WT/experiments/container/chagas.def"; then
    echo "dockerfile-equivalent" > "$CTR/chagas.variant"
else
    echo "=== fakeroot build failed; building the pytorch-base fallback ==="
    rm -f "$CTR/chagas.sif"
    apptainer build --fakeroot --ignore-fakeroot-command "$CTR/chagas.sif" "$WT/experiments/container/chagas_nofakeroot.def"
    echo "pytorch-base-fallback" > "$CTR/chagas.variant"
fi
date
ls -la "$CTR/chagas.sif"; cat "$CTR/chagas.variant"
apptainer exec "$CTR/chagas.sif" bash -c 'cd /challenge && python -c "import torch, lightning, chagas, fairseq_signals; print(torch.__version__, lightning.__version__)" && ls -la mimic_iv_ecg_finetuned.pt'
echo "BUILD OK ($(cat "$CTR/chagas.variant"))"
