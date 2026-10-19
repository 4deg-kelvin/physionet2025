# Submission package — PhysioNet Challenge 2025 (post-Challenge evaluation)

**Entry:** `final_r3` (DECISIONS.md D-2026-10-16-1/2): ECG-FM backbone → learnable 12-layer aggregation → MLP head, fine-tuned in
two stages; fixed 5 s input windows (mean of window logits at inference); hard-negative margin-ranking loss (Hong et al.,
CinC 2025); three-seed ensemble trained inside one `train_model.py` call; negatives capped to 6.5% training prevalence;
validation carved before the cap, stratified by source × label × record length, checkpoints selected on the validation
TPR@5% within 7.34 s CODE-15% records. The configuration is the package default (`chagas/config.py` ==
`experiments/configs/final_r3.yaml`); the organisers set no environment variable and pass no config.

**What to submit:** the GitHub repository `4deg-kelvin/physionet2025`, branch/tag below. The organisers clone it, run
`docker build`, then `train_model.py -d training_data -m model -v`, `run_model.py -d holdout_data -m model -o outputs -v`,
`evaluate_model.py`. The four organiser files (`train_model.py`, `run_model.py`, `evaluate_model.py`, `helper_code.py`)
and the `Dockerfile` block they mark "DO NOT EDIT" are unchanged.

## Steps (run by the repository owner; the cluster shell cannot push)

```bash
cd /juice2/scr2/kelvinkn/other_work/edwards/improve_wt
git push origin improve/robust-finetune              # the entry branch
git push origin r3-submission                        # the tag on the submitted commit
# If the submission form takes only a repository URL (default branch), fast-forward main to the entry:
git push origin improve/robust-finetune:main         # origin/main is an ancestor; this is a fast-forward
```

Submission form text (adjust to the form's fields):

> Repository: https://github.com/4deg-kelvin/physionet2025 — branch `improve/robust-finetune`, tag `r3-submission`
> (the commit the tag points to: `git rev-parse r3-submission`). Entry point: unmodified `train_model.py` / `run_model.py`.
> Training trains three models sequentially in one call; measured 4.1 h on an NVIDIA Titan RTX for the full public
> training set (292,885 records), projected ~10 h on a T4; peak GPU memory 5.0 GiB reserved in training, 2.7 GiB at
> inference. Inference over 73,300 records took 26 min. No internet access is needed after `docker build`
> (the build downloads the public ECG-FM checkpoint, 1 GB, with gdown).

## Evidence that it runs as the organisers run it

| Check | Where | Result |
|---|---|---|
| Container build from the Dockerfile rendering (Apptainer, rootless; no Docker daemon on the cluster), hardened Dockerfile incl. checkpoint sha256 check | `experiments/logs/container_build.out`, SLURM 17967832 | BUILD OK, 16 min |
| `train_model.py` → `run_model.py` → `evaluate_model.py` inside the image, default config, 400 records | `experiments/logs/container_run.out`, SLURM 17967833 | 400/400 outputs, RUN OK |
| Default config on the full training split, then inference on 73,300 held-out records, 14.5 GiB allocator cap | `experiments/logs/entry_check.out`, SLURM 17902758 | train 14,896 s (3 members), infer 3,139 s for 2 × 73,300 |
| Held-out scores of that entry (patient-grouped split, natural prevalence 2.27%) | `experiments/results/entry_final_r3/report.md` | pooled TPR@5% 0.454 / AUROC 0.845; 7.34 s records 0.416 / 0.808; 10 s format AUROC 0.854 |
| Unit tests (174) | `experiments/logs/pytest_cpu.out`, SLURM 17967831 | passed |
| Submission conformance and Docker audits (two independent read-only audits, no FAIL; WARNs fixed) | `SUBMISSION_AUDIT.md` | see file |

Not exercised: the organisers' own `docker build` (no daemon reachable here). The Apptainer definition follows the
Dockerfile step for step, with the checkpoint copied from a local file instead of gdown.

## Known properties the organisers should not be surprised by

- `train_model.py` prints per-validation metrics and writes `model/chagas_manifest.json` (selection history) and
  `model/metrics/` (CSV logs) next to the three checkpoints; `run_model.py` reads only the checkpoints and manifest.
- The training-set composition the grader sees (~366k records) differs from our split; the prevalence cap keeps the
  negative ratio the same (D-2026-10-15-3).
- `ptbxl_database.csv` is not part of the Challenge data: PTB-XL records are grouped individually for validation
  (all PTB-XL records are negative; no label is affected).
- Time budget: `train.max_hours` 71 is split across the three members; at the measured speed it is never reached.

## What the hidden-cohort score will test

Our internal result is that record length is a label shortcut in CODE-15% (paper/main.pdf, section 3.1) and that the
entry does not use it (length-cue gap +0.01). The hidden cohorts are all ~10 s recordings; whether removing the cue
improves external performance is untested until the organisers score this entry (paper, Limitations).
