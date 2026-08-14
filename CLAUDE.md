# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A submission to the **George B. Moody PhysioNet Challenge 2025** —
<https://moody-challenge.physionet.org/2025/> — which asks for detection of **Chagas disease from
12-lead ECG**. Every model here is a binary classifier over a 12-lead ECG record, emitting a
`(binary_label, probability)` pair.

The ranking metric is **not** AUROC: it is the true-positive rate among the top 5% highest-probability
predictions, modelling limited serological testing capacity. `compute_challenge_score(labels, outputs,
fraction_capacity=0.05, ...)` in `helper_code.py` / `custom_helper_code.py` / `utils.py` implements it.
Optimizing ranking at the head of the distribution matters more than global calibration — this is why
`fm.py` carries `PercentileRankingLoss`, `TopKRankingLoss`, `ChallengeScoreLoss`, and `DAM_Momentum`
alongside plain BCE.

Training data are heterogeneous by design: CODE-15% (~300k Brazilian records, 400 Hz), SaMi-Trop
(1,631 confirmed positives, 400 Hz), PTB-XL (21,799 European records at 500 Hz, assumed negative).
Sampling-rate and cohort shift are real, not hypothetical — hence the unconditional resample to 500 Hz
and the powerline/temporal augmentations.

## The organizer contract

The organizers replace and run three wrapper scripts; they are the only entry points that execute:

```
python train_model.py    -d data -m model -v
python run_model.py      -d data -m model -o outputs -v
python evaluate_model.py -d data -o outputs -s scores.csv
```

These wrappers, plus `helper_code.py`, are **do-not-edit** files (each says so in its header). Local
edits are discarded at judging time, so changing them is worse than useless — it hides drift. All team
logic reaches the wrappers through exactly three functions in `team_code.py`:

| Function | Signature | Current behaviour |
|---|---|---|
| `train_model` | `(data_folder, model_folder, verbose)` | delegates to `fm.train_model(data_folder, model_folder)` |
| `load_model` | `(model_folder, verbose)` | loads `model_folder/foundation_model_finetuned.ckpt`, returns an `FMChagasClassifier` on cuda:0 if available |
| `run_model` | `(record, model, verbose)` | returns `(pred, prob)`; note the arg order is **record first, model second** |

Note the wrapper calls all three positionally, so any added parameter must have a default.
`run_model.py` catches per-record failures only when invoked with `-f/--allow_failures`; without it, one
bad record aborts the entire pass. `team_code.run_model` therefore does its own try/except, returning
`(None, None)` on generic failure — but deliberately **re-raises `NotImplementedError`**, which is the
codebase's signal for "preprocessing is wrong, do not emit predictions."

### Runtime environment you are targeting

A `g4dn.4xlarge` (16 vCPU, 60 GB RAM, 100 GB disk, one **NVIDIA T4 / 16 GB**), with a **72-hour training
limit** and 24-hour validation limit. `fm.TimeLimitCallback(max_hours=71.0)` exists specifically to stop
training just under that ceiling — keep it wired up in any new training path. Assume **no internet at
train/inference time**: anything downloaded must be fetched in the Dockerfile, not at runtime.

## Commands

There is **no test suite and no linter configured.** `AUGMENTATION_README.md` and
`IMPLEMENTATION_SUMMARY.md` describe `test_augmentations.py`, `test_simple.py`, and
`test_dataloader_integration.py` — none of these files exist in the repo. Treat both documents as
stale design notes, not as a description of the current tree.

```bash
# Container (this is what the organizers actually build)
docker build -t physionet2025 .
docker run -it --gpus all \
  -v /path/to/training_data:/challenge/training_data \
  -v /path/to/model:/challenge/model \
  -v /path/to/test_data:/challenge/test_data \
  -v /path/to/outputs:/challenge/outputs \
  physionet2025 bash

# Inside the container (WORKDIR is /challenge)
python train_model.py -d training_data -m model -v
python run_model.py   -d test_data -m model -o outputs -v
python evaluate_model.py -d test_data -o outputs -s scores.csv

# Local dev setup (Linux/GPU box; the dev machine here is Windows, the target is not)
pip install -r requirements_linux.txt          # or: conda env create -f requirements_linux.yml
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -e fairseq-signals/                # required: provides the ECG-FM backbone
pip install "transformers==4.53.0"
gdown 1uI2J_gMk0eh0vu3MbKPBEoupgakZ06j2        # -> mimic_iv_ecg_finetuned.pt (not committed)

# Research loop, bypassing the organizer wrappers (fm.py has its own __main__ with splits,
# wandb logging, checkpoint monitoring, and k-fold that fm.train_model deliberately omits)
python fm.py --debug                           # subsample records
python fm.py --evaluate --predictions-output validation_predictions_fm.csv
python fm.py --grad-cam                        # saliency plots; expects a hardcoded ckpt path, edit first
python train_se.py                             # SE-ResNet branch has its own arg parser
```

`fm.py`'s `__main__` block and `fm.train_model` are **separate code paths with separate
hyperparameters**. `__main__` is the experimentation harness (`BATCH_SIZE=32`, `windowing_method='random'`,
5-fold, wandb, val/test splits, `val_challenge_score` checkpoint monitoring). `fm.train_model` is the
submission path (`BATCH_SIZE=16`, `'entire_recording'`, no validation split — it trains on *all* records,
`logger=False`, `num_sanity_val_steps=0`, saves one weights-only checkpoint). Changing a hyperparameter in
one does not change the other; a result you measured via `python fm.py` is not what the submission trains.

## Architecture

### Signal pipeline (the part every model shares)

`utils.preprocess_signal(record_path, windowing_method, is_training)` is the single funnel for turning a
WFDB record into model input, and it is used at both train and inference time:

1. `custom_helper_code.load_signals` → reorder leads to `utils.ECG_FM_LEAD_ORDER` → transpose to
   `(leads, samples)` → reject anything that isn't 12-lead by returning `(None, None)`.
2. If `is_training`, drop records shorter than `MIN_SIGNAL_DURATION` (2 s) by returning `(None, None)`.
3. Parse age/sex (and label, **only** when `is_training=True` — the `get_label=False` branch at inference
   is what keeps held-out labels from leaking in).
4. Resample to `UNIFIED_FREQUENCY = 500` Hz; `nk.ecg_clean` per lead.
5. `utils.zscore_per_record` — per-lead z-score over the **whole recording**, before windowing.
6. Window to `WINDOW_SIZE = 5000` samples (10 s) via `utils.get_windows`.
7. Return `(signal, wide_feats)` where `wide_feats = [z-scored age, sex ∈ {0.0, 0.5, 1.0}]` — the two
   "info features" the classifier head concatenates onto the ECG embedding.

**Steps 1, 5 and 6 are load-bearing and must stay identical in `dataloader.ECGDataset.__getitem__`,
which reimplements this pipeline rather than calling it.** They diverged once already: training applied
a global per-lead z-score from an `ecg_statistics.csv` of unknown provenance while inference applied
per-record min-max, so the model was served a distribution it never saw and nothing crashed to say so.

Per-record z-score is not a free choice — it is what the ECG-FM backbone was pretrained on
(`fairseq-signals/scripts/preprocess/ecg/preprocess.py`, standardize-then-segment). The encoder's first
conv block is `Fp32GroupNorm(dim, dim)` with `conv_bias=False`, so it absorbs a global rescale and a
per-lead DC offset, but **not** per-lead relative rescaling. Do not "helpfully" add a normalization step.

Age normalization uses fixed training-set constants (`MEAN_AGE_TRAIN`, `STD_AGE_TRAIN` in `utils.py`).
Missing age → 0.0 (the mean), unknown sex → 0.5. Any change to these constants silently invalidates
existing checkpoints, since inference must z-score identically to training.

**`helper_code.py` vs `custom_helper_code.py`:** the former is the frozen organizer file; the latter is a
superset the team owns, adding `find_records_abs`, `get_patient_info`, and variants. New helpers go in
`custom_helper_code.py`. Both define `compute_challenge_score` — they are separate copies.

### Data layer

`dataloader.ECGDataset` / `ECGDataModule` wrap the above for Lightning. Notable behaviour:

- `__getitem__` returns `None` for unusable records; `collate_fn_skip_none` filters them out, so batches
  can be smaller than `batch_size` — and it returns `None` for the whole batch when every record failed,
  which every consumer must check before unpacking.
- `train_dataloader` shuffles by default, and uses a `WeightedRandomSampler` when `ECGDataModule` is given
  `sample_weights`. Neither is optional in practice: records arrive grouped by source dataset, so without
  this the model sees a long run of all-negative CODE-15 records then a run of SaMi-Trop positives, in
  the same order every epoch. `fm.train_model` passes inverse-frequency weights built from the label
  column of `utils.prepare_stratification`.
- `val_dataloader`/`test_dataloader` return `None` when their dataset is unset. Disabling validation also
  needs `limit_val_batches=0` on the Trainer — `num_sanity_val_steps=0` only skips the sanity check.
- Augmentation is opt-in and training-only: passing `aug_config` constructs `augmentations.ECGAugmentations`
  (powerline 50/60 Hz + harmonics at 10–30 dB SNR, random crop/shift), applied after `nk.ecg_clean` but
  before windowing. When an augmenter is active it guarantees exactly 5000 samples, so the usual
  `utils.USE_ONE_WINDOW` windowing step is skipped. `fm.train_model` currently passes **no** `aug_config`.
- Optional multi-task labels (`RBBB`, `1dAVb`, `AF`, `SB`) come from `code15_exams.csv`. That default is a
  **bare relative filename** resolved against the process CWD — it works because Docker's `WORKDIR` is
  `/challenge` and the CSV sits at the repo root. Run training from anywhere else and it silently falls
  back to "not found".

### Model branches

`team_code.py` currently dispatches to `fm.py`, but the other backbones are live alternatives that get
swapped in, not dead code. When switching, the change is usually confined to `team_code.py`'s three
functions plus the checkpoint filename.

| Module | Backbone | Entry points |
|---|---|---|
| `fm.py` | **Current submission.** ECG-FM (fairseq-signals 12-layer transformer) via `ECGFMFeatureExtractor` → `FMChagasClassifier` | `train_model`, `load_finetuned_model` |
| `se_model.py` + `wavelnet.py` | SE-ResNet 1D/2D with squeeze-excitation; wavelet-conv variants (`WaveletSE_ECGNet`, `Wavelet_Pooled_SE_ECGNet`) | `train_se.py` (`train_se`, `load_model`, `run_model`) |
| `ensemble.py` | Weighted blend of the FM and SE probabilities, with `search_best_weights` grid search | `train_ensemble`, `run_ensemble` |
| `transformer.py` | From-scratch RoPE transformer (`ChagasTransformer`); carries its own `ECGDataset`/`ECGDataModule` copies | `train`, `load_model` |
| `mae_vit_ecg.py` | ViT-style 1D masked autoencoder, pretrain → finetune | `pretrain_mae_vit_ecg.py`, `finetuning_mae_vit_ecg.py` |
| `mae.py` + `finetuning.py` | Earlier conv-MAE variant | `mae.train_model`, `finetuning.train_finetune_model` |

Because `transformer.py` duplicates `ECGDataset`/`ECGDataModule`/`collate_fn_skip_none` rather than
importing `dataloader.py`, a fix to the shared data path needs applying in both places.

### How `fm.py` uses ECG-FM

`ECGFMFeatureExtractor` is the non-obvious piece. It loads a fairseq-signals checkpoint
(`mimic_iv_ecg_finetuned.pt`) and then:

- Registers a forward hook on **each of the 12 transformer layers** and combines their outputs with a
  learnable softmax-weighted sum (`layer_weights`, initialized to zeros → uniform), rather than using only
  the final layer.
- Calls `ecg_fm_model.encoder` (the `ECGTransformerModel`) **directly**, bypassing the classification
  wrapper's `forward()` because that wrapper `.detach()`es the encoder output and would block gradients.
- Masked-mean-pools over time using the encoder's `padding_mask`.
- Emits a 768-d embedding; `FMChagasClassifier` concatenates the 2 wide features and runs a
  `Dropout → Linear(770, 256) → ReLU → Dropout → Linear(256, 1)` head.

Encoder freezing is a first-class control: `freeze_encoder=True` at construction, with `unfreeze()` and an
`unfreeze_after_epoch` hparam for staged fine-tuning. `fm.train_model` ships with the encoder **frozen**.

The `mimic_iv_ecg_finetuned.pt` path is a bare relative filename in both `fm.train_model` (line ~1474) and
`fm.load_finetuned_model` (line ~1684). It is not committed, and exists only because the Dockerfile
`gdown`s Drive ID `1uI2J_gMk0eh0vu3MbKPBEoupgakZ06j2` into `/challenge`. This is the single most fragile
external dependency in the repo — if that ID rots or the downloaded filename changes, the image still
builds and training fails only on the organizers' machine. `docker-submission-audit` exists to test it.

## Pre-submission audits

Two Claude Skills in this repo gate submission readiness. Run **both** before shipping; they are read-only
by contract and report findings rather than applying fixes.

- **`docker-submission-audit/`** — Dockerfile and dependency conformance: base image vs. actual runtime
  needs, the three `DO NOT EDIT` lines, referenced requirements files exist, every `gdown` ID is commented
  *and still reachable*, every third-party import has an install path, layer ordering for cache reuse. It
  exists because a prior submission shipped with a typo'd Google Drive ID that failed only when the
  organizers built the image.
- **`submission-conformance-audit/`** — the runtime code `team_code.py` reaches: protected files unchanged
  (SHA-256), signatures still match the wrappers, and the user-authored call-path modules free of absolute
  paths, runtime network calls, stray writes, T4-memory risk, and fatal-on-one-record gaps.

Both were originally written for a later year's repo and have been retargeted to this one; if a check's
concrete example no longer matches the tree, the *check* is still valid — re-derive the specifics.

## Known gaps

Documented, not fixed — resolve deliberately rather than assuming they are intentional.

- **No validation split or model selection on the submission path.** `fm.train_model` trains on every
  record and saves whatever the last of 16 epochs produced — no `ModelCheckpoint`, no early stopping, no
  `gradient_clip_val`, no `seed_everything`. The `__main__` harness has all of these; the shipped path
  does not.
- **The loss does not target the metric.** `LOSS_TYPE='bce'` with no `pos_weight`. Of the four custom
  ranking losses only `PercentileRankingLoss` is usable — `TopKRankingLoss` returns a bare Python `int`
  when a batch has no pos/neg pair (crashing `.backward()`) and its "missed positives" term has zero
  gradient; `ChallengeScoreLoss` inherits that at 0.7 weight; `DAM_Momentum` never uses its dual variable,
  reducing to variance minimisation that collapses to a constant output. Only `'bce'` and `'percentile'`
  are selectable at all.
- **Input is 10 s / 5000 samples but ECG-FM pretrained on 5 s / 2500.** Architecturally safe — positional
  encoding is a depthwise conv, not a learned table, and no `max_sample_size` applies — but out of
  distribution. Cheap ablation via `utils.WINDOW_TIME`.
- **Sub-10 s records are zero-padded and those pad frames enter the unweighted mean pool** as if they were
  signal. Mitigated (pads are now at the per-lead mean, not an arbitrary value) but not eliminated.
- **Validation-only metric bugs:** raw logits passed to `Accuracy`, `labels.int()` truncating soft labels,
  float targets into AUROC. Harmless on the submission path, which runs no validation.
- **The Dockerfile is missing the three mandated `DO NOT EDIT` lines** (`RUN mkdir /challenge`,
  `COPY ./ /challenge`, `WORKDIR /challenge`). It reaches an equivalent layout via `WORKDIR /challenge` +
  `COPY . .`, but the organizers' instructions require the exact block.
- **Missing required repository files:** the 2025 submission instructions list `requirements.txt`,
  `AUTHORS.txt`, `LICENSE.txt`, and `README.md`. This repo has none of them — dependencies live in
  `requirements_linux.txt` / `requirements_linux.yml`.
- Large CSVs are committed at the repo root (`code15_exams.csv` is ~36 MB, `train_val_test_sets.csv` ~5 MB,
  `code15_label_issues.csv` ~2 MB) and ship inside the image.
- `fm.py`'s `--grad-cam` branch contains a hardcoded absolute `/sailhome/...` checkpoint path. It is inside
  `if __name__ == '__main__'`, so it never executes on the submission path, but it will not run as-is for
  anyone else.
- The `code15_label_issues.csv` exclusion logic in `fm.train_model` is commented out — mislabelled CODE-15
  records are currently trained on.
