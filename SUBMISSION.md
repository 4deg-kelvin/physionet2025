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

## Where it lives

`main` is the entry (fast-forwarded to the tagged commit on 2026-10-19); `improve/robust-finetune` is kept as the
development branch and points at the same commit. Clone cost for the organisers: 526 files, 59 MB checked out
(Docker build context ~61 MB after `.dockerignore`); the history reachable from `main` holds 160 MB of blobs
uncompressed, ~300 MB packed with every branch. No file in the tree exceeds 50 MB.

```bash
git clone https://github.com/4deg-kelvin/physionet2025.git     # default branch main == tag r3-submission
```

Submission form text (adjust to the form's fields):

> Repository: https://github.com/4deg-kelvin/physionet2025 — default branch `main`, tag `r3-submission`
> (the commit the tag points to: `git rev-parse r3-submission`). Entry point: unmodified `train_model.py` / `run_model.py`.
> Training trains three models sequentially in one call; measured 4.1 h on an NVIDIA Titan RTX for the full public
> training set (292,885 records), projected ~10 h on a T4; peak GPU memory 5.0 GiB reserved in training, 2.7 GiB at
> inference. Inference over 73,300 records took 26 min. No internet access is needed after `docker build`
> (the build downloads the public ECG-FM checkpoint, 1 GB, with gdown).

## Evidence that it runs as the organisers run it

The logs cited below are committed under `experiments/logs/` (the SLURM job ids are the cluster's).

| Check | Where | Result |
|---|---|---|
| Container build from the Dockerfile rendering (Apptainer, rootless; no Docker daemon on the cluster), hardened Dockerfile incl. checkpoint sha256 check | `experiments/logs/container_build.out`, SLURM 17969157 | BUILD OK, 15 min |
| `train_model.py` → `run_model.py` → `evaluate_model.py` inside the image, default config, 400 records | `experiments/logs/container_run.out`, SLURM 17969158 | 400/400 outputs, RUN OK |
| Default config on the full training split, then inference on 73,300 held-out records, 14.5 GiB allocator cap | `experiments/logs/entry_check.out`, SLURM 17902758 | train 14,896 s (3 members), infer 3,139 s for 2 × 73,300 |
| Held-out scores of that entry (patient-grouped split, natural prevalence 2.27%) | `experiments/results/entry_final_r3/report.md` | pooled TPR@5% 0.454 / AUROC 0.845; 7.34 s records 0.416 / 0.808; 10 s format AUROC 0.854 |
| Unit tests (175) | `experiments/logs/pytest_cpu.out`, SLURM 17969156 | passed |
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

## Checked against the organisers' instructions (moody-challenge.physionet.org/2025/submissions, python-example-2025; 2026-10-19)

| Requirement | Status |
|---|---|
| Code in the root directory of the `main` or `master` branch | `main` == tag `r3-submission` |
| `train_model.py`, `run_model.py`, `helper_code.py` unmodified (the organisers substitute their originals) | `train_model.py` and `run_model.py` are byte-identical to python-example-2025 `main`; `helper_code.py` differs in one line of `compute_challenge_score` (`np.argsort(..., kind='stable')` here, `stable=True` upstream — the upstream form needs numpy ≥ 2, our image pins 1.26.4; same ordering). `team_code.py` signatures unchanged |
| Dockerfile: keep the three "DO NOT EDIT" lines, pin versions, add system packages as needed | kept byte-for-byte; base `nvidia/cuda:12.1.0-devel-ubuntu22.04`; torch/torchvision/transformers/numpy/pandas/lightning pinned |
| `requirements.txt` with versions, `scikit-learn` not `sklearn` | our Dockerfile installs `requirements_linux.txt` (same role); uses `scikit-learn` |
| Environment: g4dn.4xlarge, 16 vCPU, 60 GB RAM usable, optional T4; 72 h training, 24 h validation (scaled for test) | measured 4.1 h training on a Titan RTX (~10 h T4 projection), 5.0 GiB GPU; `num_workers` 10 ≤ 16 vCPU; inference 0.4 h per 73k records |
| Move repeated setup from `run_model` into `load_model` | done: `load_model` builds the three members once; `run_model` only preprocesses and predicts |
| Build the image and test on at least one training file before submitting | Apptainer rendering built; the three scripts ran on 400 records (SUBMISSION_AUDIT.md) |
| Leave out files not needed to create and run the code (e.g. training data) | no data in the repo; `code15_exams.csv` (34 MB) is needed for labels; two unused CSVs (7 MB) and legacy modules remain, harmless |
| Private repository shared with `physionetchallengeshelper`, or public | the repository is public; if it is made private, share it with that user |
| Robustness tests the organisers may run (altered/missing values, 15% or 150% training set, randomised labels) | signal NaNs are zeroed in preprocessing; training-set size is free (the prevalence cap keeps the negative ratio; the time budget scales); labels are read per record and a header without a label is skipped with a count, not fatal |
| `evaluate_model.py` | the organisers' evaluation runs with their own code; ours is the earlier upstream version (same metric) |
