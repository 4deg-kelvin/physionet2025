# Reproducing every table in `experiments/README.md`

From a clean clone of branch `improve/robust-finetune` to every number we report.  Paths are the cluster's; change
`WT` / `FIX` in the scripts if you move them.  Everything heavy runs through SLURM (`jag-standard`, `nlp` account);
nothing below is meant for a login node except `make_configs.py` and `git`.

```bash
WT=/juice2/scr2/kelvinkn/other_work/edwards/improve_wt          # this worktree
FIX=/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test      # data folders and the split
PY=/juice2/scr2/kelvinkn/conda_envs/physionet/bin/python         # torch 2.7.1+cu118, lightning 2.5.2
cd $WT
```

The ECG-FM checkpoint `mimic_iv_ecg_finetuned.pt` must sit in the repo root (McKeen et al. 2025; the Dockerfile
symlinks it from `/opt/ecgfm`).  `code15_exams.csv` is tracked.

## 1. Data: the split, the folders, Test C, the durations

| Step | Command | Output |
|---|---|---|
| Patient-grouped train/test split of all 366,185 records | `sbatch $FIX/build_split.sh` | `$FIX/split/split.csv` |
| 100k training subsample + full held-out test (symlink trees) | `sbatch $FIX/make_folders.sh` | `$FIX/data_train`, `$FIX/data_test` (73,300) |
| Full training folder (292,885) | created on first use by `experiments/batch_full_final.sh` (inline block) | `$FIX/data_train_full` |
| Test C (organiser-style perturbations, seeded per record) | `sbatch experiments/batch_build_test_C.sh` | `$FIX/data_test_C` |
| Record durations from every WFDB header, cross-checked against `physionet2025/signal_lengths.csv` | `sbatch experiments/batch_build_durations.sh` | `experiments/data/record_durations.csv` (committed; SLURM 17682272, 0 mismatches) |
| 400-record smoke set | already at `$FIX/testdata` | |

## 2. Configs and tests

```bash
$PY experiments/make_configs.py               # rewrites experiments/configs/*.yaml and configs/r2/*.yaml
git diff --stat experiments/configs           # must be empty: tests/test_config_r2.py enforces it
sbatch experiments/batch_slurm_tests_cpu.sh   # whole suite, CPU node (~5 min); or batch_slurm_tests.sh on a GPU
sbatch experiments/batch_memory_smoke.sh      # stage 2 fits the 14.5 GiB T4 cap
sbatch experiments/batch_e2e_smoke.sh         # round-1 contract: train_model -> run_model -> evaluate_model
sbatch experiments/batch_r2_smoke.sh          # round-2 contract with every new flag on + batched-inference equivalence
```

The 8 `configs/lrsweep/*.yaml` were written by hand in commit 74512a6 (shipped config + `stage2_lr`/`seed`), not by
`make_configs.py` (DECISIONS.md D-2026-10-01-1).

## 3. Round 1 tables (README sections "Ladder results" ... "Stage-2 learning-rate sweep")

Each batch script trains through the unmodified `train_model.py`, infers through the unmodified `run_model.py`
(Test A and Test C), runs `evaluate_model.py` and `eval_report.py`, and writes `results/<run>/`:

```bash
sbatch experiments/batch_ladder_row0.sh   # ... row1, row1b, row2, row2b, row3, row4, row5, row5b, row5c, row5d
sbatch experiments/batch_full_final.sh    # final_full_seed42/43 (full 292,885 records)
sbatch experiments/batch_finalA.sh        # finalA_negcap  (= the shipped config, B0 seed 42)
sbatch experiments/batch_finalB.sh        # finalB_stepval
sbatch experiments/batch_lr_sweep.sh      # lrsweep_* (B0 seeds 43/44 are lrsweep_lr2e5_s43/44)
```

## 4. The re-scored table (README "Re-scored within record-length groups") and the acceptance checks

```bash
sbatch experiments/batch_rescore.sh       # acceptance_r2.py, then rescore_all.py
```

Writes `results/RESCORED.md` (+ `RESCORED_runs.csv`, `RESCORED_aggregate_{A,C}.csv`, `RESCORED_gap_{A,C}.csv`)
from the saved `predictions_{A,C}.csv`; no model is re-run.  Bootstrap: 1,000 resamples, fixed seed per view.

## 5. Round 2 (PLAN_R2.md §4)

```bash
sbatch experiments/batch_r2.sh            # 9 tasks (M1, W1, MW x seeds 42/43/44), at most 4 at once
```

Each task runs `experiments/run_r2.sh`: training via the unmodified `train_model.py` on `data_train_full`,
`predict_batched.py` on Test A and Test C (for MW once per WiSE-FT alpha, into `results/<run>/alpha<a>/`), the
round-2 report, and `run_info.json`.  Then the table and the pre-registered decision rule (PLAN_R2.md §5):

```bash
for W in A C; do
$PY experiments/aggregate.py \
  --group B0=finalA_negcap,lrsweep_lr2e5_s43,lrsweep_lr2e5_s44 \
  --group M1=r2_M1_s42,r2_M1_s43,r2_M1_s44 \
  --group W1=r2_W1_s42,r2_W1_s43,r2_W1_s44 \
  --group MW=r2_MW_s42,r2_MW_s43,r2_MW_s44 \
  --group MW_a0.00=r2_MW_s42/alpha0.00,r2_MW_s43/alpha0.00,r2_MW_s44/alpha0.00 \
  --group MW_a0.50=r2_MW_s42/alpha0.50,r2_MW_s43/alpha0.50,r2_MW_s44/alpha0.50 \
  --group MW_a1.00=r2_MW_s42/alpha1.00,r2_MW_s43/alpha1.00,r2_MW_s44/alpha1.00 \
  --baseline B0 --which $W --out experiments/results/R2_TABLE_$W
done
```

B0's predictions came from the per-record `run_model.py` path, round 2's from `predict_batched.py`; the two are
asserted equal to max |Δprob| < 1e-4 (`tests/test_predict_batched.py`, `batch_r2_smoke.sh`).

## 6. Provenance of a run

Every run since round 2 has `results/<run>/run_info.json` (commit, dirty flag, config SHA-256, seed, host, GPU,
versions, `pip freeze` SHA-256, start/end, train/val counts by source × label × length group) and
`results/<run>/model/chagas_manifest.json` (selection history per validation, WiSE-FT per-alpha metrics) and
`results/<run>/model/metrics/member0/stage*/metrics.csv` (per-step loss and LR, every validation metric).

## 7. Round 3 (PLAN_R3.md §4)

```bash
sbatch experiments/batch_split_audit.sh                       # results/SPLIT_AUDIT.{md,csv}: the held-out split re-checked
S1=$(sbatch --parsable experiments/batch_r3_smoke.sh)         # contract smoke, 400 records, configs/r3_smoke.yaml
S2=$(sbatch --parsable --dependency=afterok:$S1 experiments/batch_r3_runner_smoke.sh)   # run_r3.sh end to end
A=$(sbatch  --parsable --dependency=afterok:$S2 experiments/batch_r3.sh)                # 14 tasks, at most 4 at once
sbatch --dependency=afterany:$A experiments/batch_r3_table.sh                           # results/R3_TABLE_{A,C}.*
```

Configs come from `make_configs.py` (`r3_configs()`): the shipped configuration plus `train.val_protocol: fixed`
(validation carved before the cap, `val_seed` 12345, strata source × label × length group, PTB-XL grouped by
patient from `data.ptbxl_database_csv`), `train.monitor: val_challenge_s7`, `data.cap_mode: prevalence` at 0.065;
then B0 (nothing else), W1 (5 s multi-window; seeds 42–46), W1R (+ `ranking_loss`), W1S (+ `exclude_sources:
[ptbxl]`).  Each task runs `experiments/run_r3.sh`: `train_model.py` on `data_train_full`, `predict_batched.py`
on Test A and Test C, the report, and, for multi-window configs, a second scoring pass with `--window-stride 2.5`
into `predictions_{A,C}_tta.csv` (test-time windows; the judged `run_model.py` path is untouched).  The table
job applies the pre-registered round-3 rule (`aggregate.py --rule r3`: primary S10+ AUROC, guard S7 TPR@5%)
against B0; the variant-vs-W1 comparisons of PLAN_R3 §5 are a second call with `--baseline W1`.

The CPU test suite that gated the chain: `sbatch experiments/batch_slurm_tests_cpu.sh` (173 tests).

## 8. Container check (PLAN_R3.md §9.3)

No Docker daemon is reachable from the cluster, so `experiments/container/chagas.def` renders the unmodified
`Dockerfile` for Apptainer (same base image, packages, pins and paths; the 1 GB checkpoint is copied from the
local file instead of gdown).  `batch_container_build.sh` stages the worktree through `.dockerignore`, builds the
SIF rootless, and `batch_container_run.sh` (afterok) runs `train_model.py`, `run_model.py` and
`evaluate_model.py` inside it on the 400-record set with the default config, exactly as the organisers do.
