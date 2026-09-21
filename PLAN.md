# Improvement plan: ECG-FM Chagas model (PhysioNet 2025)

Author of this plan: Claude (2026-09-21), from a code audit of branch
`fix/submission-blockers-and-model-defects` @ a72c51c and a literature review. Everything
below that is not the team's own idea is attributed in §7 (Provenance) and must be cited
in code docstrings. The team's core idea is kept unchanged: **ECG-FM backbone → learnable
aggregation over all 12 transformer layers → pooled 768-d feature → MLP head, with age/sex
as side information.**

## 0. What we are optimising, and why the current number is misleading

The challenge score is TPR@5%: recall among the top 5% of records ranked by probability,
**averaged over three hidden cohorts** (REDS-II, SaMi-Trop 3, ELSA-Brasil), all Brazilian,
all serology-labelled, ~2% prevalence, sampled at 300/500/600/1000 Hz, on different ECG
machines. The organisers *also perturbed the test records*: "we added small amounts of
various forms of noise, applied filters that are representative of different ECG devices,
and resampled the data to different sampling frequencies" (Reyna et al. 2025, §2.2).

Every team's in-distribution score collapsed out-of-distribution. Median hidden-validation
0.279 → ELSA-Brasil 0.100 (−64%). The winner went CV 0.490 → test 0.323. Our own
patient-grouped held-out score (0.440 challenge / 0.858 AUROC, run 16878337) is in the same
in-distribution band as the top teams' CV numbers, so **the remaining problem is
out-of-distribution robustness, not in-distribution fit.** Every change below is chosen
because there is published evidence it helped on the *hidden* sets, or because it removes a
concrete defect found in the audit.

Leaderboard for reference (41 ranked, median 0.217): 1st 0.323, 2nd 0.283, 3rd 0.280.

## 1. Defects found in the audit that are fixed as part of this work

| # | Defect (file:line at a72c51c) | Consequence | Fix |
|---|---|---|---|
| D1 | No validation split, no model selection, no seed (`fm.py:1665-1690`) | We ship epoch 16 blind; no early stop; non-reproducible | Patient-grouped internal holdout; checkpoint on val challenge score; early stop; `seed_everything` |
| D2 | Encoder permanently frozen; `unfreeze_after_epoch` never passed (`fm.py:1657-1662`) | Only ~197k head params train | Two-stage fine-tuning (§3.1) |
| D3 | Masked mean pooling is dead code: `padding_mask` is always `None` (`fm.py:590-633`) | Zero-padded frames of short (7.3 s) records are averaged in as signal | Carry `valid_len` from windowing; build token-level mask; apply in attention and pooling |
| D4 | Deterministic first-10 s truncation, no crop (`utils.py:514-524`) | No temporal augmentation; long records donate nothing | Random crop+shift in training (§3.4) |
| D5 | Plain BCE + ~50/50 `WeightedRandomSampler` (`dataloader.py:347-356`) | Far more aggressive than any top team; harmful under label noise (Physiol Meas 2026 ablation) | Weighted BCE w_pos=5 + 5× oversampling (winner); ablate no-oversampling |
| D6 | No augmentation on submission path (`fm.py:1631-1648`) | Zero robustness to the organisers' perturbations | Augmentation suite (§3.4) |
| D7 | `scipy.signal.resample` (FFT) for rate conversion (`utils.py:156`) | Edge ringing on non-periodic signals; slow for awkward ratios | `scipy.signal.resample_poly` (anti-aliased polyphase) |
| D8 | `nk.ecg_clean` per lead ×12, 50 Hz-only notch (`utils.py:160`) | Inference bottleneck (~11 rec/s); Brazil mains is 60 Hz | Vectorised Butterworth band-pass + 50/60 Hz notches (§3.5) |
| D9 | Hard-coded 0.5 threshold on a 50%-prior-calibrated model (`team_code.py:79`) | Binary column wildly over-positive at 2% prevalence | Threshold = 95th percentile of validation scores (the metric's operating point) |
| D10 | Validation-metric bugs: raw logits into `Accuracy`, `labels.int()` truncates soft labels (`fm.py:786-797`) | Would corrupt model selection the moment a val split exists | Metrics on sigmoid probs and hard labels |
| D11 | Two hand-synchronised preprocessing copies (`utils.preprocess_signal` vs `ECGDataset.__getitem__`) plus a third in `transformer.py` | Train/inference skew is one edit away | ONE preprocessing function used by both paths, with a parity test |
| D12 | `team_code.py` imports `pretrain_mae_vit_ecg`, `finetuning_mae_vit_ecg`, `transformer`, `wandb` (via dataloader) it never uses | Import-time failure surface in the judged container | Thin `team_code.py` importing only the new package |
| D13 | `except TypeError: model(sig_t)` fallback can never succeed with `info_features=2` (`team_code.py:72-76`) | Hides real errors | Remove |
| D14 | `assert label is not None and label == 0 or label == 1` precedence bug (`dataloader.py:174`) | Wrong intent | Fix |
| D15 | `sex.lower()` without `isinstance` guard on inference path only (`utils.py:148`) | Train/inference asymmetry | Unified code path |
| D16 | Missing age encoded as 0.0 == the training mean (`utils.py:143`) | A missing age is indistinguishable from a 53.8-year-old | Add `age_missing` indicator |

## 2. Evaluation protocol (build this FIRST; every change is judged against it)

All runs on SLURM `jag-standard`, `gpu:titanrtx:1` (Turing SM 7.5 = the grader's T4), with
the CUDA allocator capped at 14.5 GiB to emulate the T4 (see `fixbranch_test/run_heldout.sh`).
Never run training on the login node.

**Split**: the existing patient-grouped split at
`/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/split/split.csv` (zero patient
overlap across 253,720 groups; train 292,885 / test 73,300 at natural 2.27% prevalence).
Train on the 100k subsample (`data_train`, all 6,528 positives + 93,472 negatives) for
ablations; the final config is also run on the full 292,885.

**Three test views, one inference run:**

| View | What | Why |
|---|---|---|
| **A** | Full held-out (73,300, natural composition) | In-distribution score; comparable to top teams' CV numbers |
| **B** | CODE-15-only subset of A (68,575; 1,336 pos) | Source shortcut impossible (positives and negatives share a source) |
| **C** (new) | A, perturbed the way the organisers perturbed the hidden sets: for each record, resample to a random fs ∈ {300, 500, 600, 1000} Hz and back with `resample_poly`; apply a random "device" band-pass (HP 0.05–0.67 Hz, LP 40–150 Hz, Butterworth order 1–2); add Gaussian noise at SNR 20–35 dB, baseline wander (0.1–0.5 Hz, ≤0.1 of lead std) and 50/60 Hz powerline at SNR 15–30 dB, each with p=0.5. Fixed seed, generated once, cached as WFDB. | Direct estimate of robustness to the documented test-time perturbations |

Report for every run: challenge score, AUROC, AUPRC on A/B/C; per-source mean probability;
score by age decile (age-shortcut diagnostic); and the C−A gap. Baseline to beat (a72c51c,
100k subsample): **A 0.434 / B 0.440 / AUROC 0.858**; C to be measured in step 0.

**Acceptance rule for every change**: A must not decrease by more than 0.005, and the C−A gap
must not widen; otherwise the change is reverted. Final config: ≥2 seeds; report mean ± sd.

**Ablation ladder** (each row adds to the previous, like Hong et al. Table 2):
0. baseline a72c51c (rerun under the new eval harness, gives C)
1. + D1/D3/D7/D8/D9/D10 defect fixes, val split, model selection, seed
2. + soft labels and w_pos=5 weighted BCE (with 5× oversampling; separately: no oversampling)
3. + augmentation suite
4. + FiLM demographics with age-missing indicator
5. + two-stage fine-tuning with LLRD (memory-cap test must pass first)
6. + auxiliary heads (RBBB/1dAVb/LBBB/AF/SB) — optional
7. + margin-ranking loss on hard negatives — optional
8. + domain-adversarial source head — optional, judged on C and B
9. + seed/fold ensemble within the time budget

## 3. Design

### 3.1 Two-stage fine-tuning with layer-wise LR decay (D2)
Winner: head-only on a frozen backbone gave CV 0.381; unfreezing everything gave 0.490
(Van Santvliet et al. 2025). ECG-FM's authors also report full fine-tuning > linear probing
(McKeen et al. 2025).

- **Stage 1** (frozen encoder): train aggregation weights + FiLM + head, lr 2e-4, batch 16,
  grad-accumulate ×4 (effective 64, as the winner), ~2,000 optimiser steps (winner: 2,000).
- **Stage 2**: unfreeze the 12 transformer layers (keep the conv frontend frozen — the
  checkpoint's `feature_grad_mult=0.0` already does this; expose a flag to override). Base
  lr 2e-5 (winner) with **layer-wise LR decay** 0.85: layer 12 gets base lr, layer k gets
  base·0.85^(12−k), head/FiLM get 1e-4 (Howard & Ruder 2018; Clark et al. 2020). Cosine
  schedule with 5% warm-up. Encoder dropout: keep the checkpoint's `activation_dropout=0.1`
  active in train mode (the existing `train()` override must switch behaviour once unfrozen);
  `encoder_layerdrop` stays 0.0 so the 12-layer aggregation invariant holds.
- **Memory**: AMP fp16 (T4 has fp16 tensor cores; **no bf16** on SM 7.5 — assert this),
  batch 16, and gradient checkpointing as an automatic fallback if the memory-cap smoke test
  fails. Measure peak with `torch.cuda.max_memory_allocated()` under the 14.5 GiB cap.
- **Time budget**: the grader gives 72 h on a T4 (~2× slower than TitanRTX). Keep the
  existing `TimeLimitCallback`, but make it budget-aware: stage 1 ≤ 5%, stage 2 ≤ 80% of
  `max_hours`, ensemble members split the remainder; always save the best checkpoint.

### 3.2 Labels and loss (D5)
- **Source-aware soft labels** (Van Santvliet 2025; Physiol Meas 2026 `ae98e0`; Hong 2025):
  SaMi-Trop → 1.0, PTB-XL → 0.0, CODE-15 positive → 0.8, CODE-15 negative → 0.2. Source is
  read from the `# Source:` header line; unknown source → treat as CODE-15-like (0.8/0.2).
- **Multi-report consistency** (Hong 2025 eq. 1; Ahus AIM 2025): for CODE-15 patients with
  >1 exam (patient_id from the shipped `code15_exams.csv`), replace the constant 0.8/0.2 by
  `c = 0.51·y + 0.49·(n_pos/n_total)`, so a patient with 3/5 positive reports gets 0.804 for
  positive exams and 0.294 for negative ones. Unit-test that exact example.
- **Optional RBBB-informed bump** for CODE-15 negatives (Hong 2025: +0.15 if RBBB, +0.25 if
  RBBB with AVb/AF, capped 0.49). The team's own `_compute_soft_label` in `dataloader.py`
  is a stronger version of this idea; keep theirs as `soft_label_scheme: "team_rbbb"` and
  Hong's as `"hong_rbbb"`; default off (ablation row 2b).
- **Loss**: `BCEWithLogitsLoss(pos_weight=5.0)` on the soft targets (winner; Physiol Meas
  2026). Hard labels are kept alongside for metrics.
- **Sampling**: `WeightedRandomSampler` with positives weighted 5× (winner), *not*
  inverse-frequency (~50/50). Ablate `oversample: 1.0` (Physiol Meas 2026 found upsampling
  harmful under noise; the two top papers disagree, so we test it).
- **Optional margin-ranking loss on hard negatives** (Hong 2025 eq. 2–3): pairwise hinge
  between positives and the top-20% highest-scoring negatives in the batch, margin 0.5 scaled
  by the soft-label gap, added with weight α=2.5 to the BCE. Matches the metric (it is a
  ranking task). The repo's `TopKRankingLoss`/`ChallengeScoreLoss` are unreachable and
  numerically unsafe (audit); replace, do not extend.

### 3.3 Demographics via FiLM (D16)
Replace concat with a FiLM conditioner (Perez et al. 2018), as both the winner and the
Physiol Meas paper did (+0.022 there): `MLP([age_centuries, age_missing, sex_f, sex_m,
sex_unknown]) → (γ, β)`; feature ← γ ⊙ feature + β, γ initialised to 1, β to 0 so training
starts as the identity. Age >89 is already clipped to 90 by the organisers; do the same on
our side for symmetry. Keep `demographics: "concat" | "film" | "none"` — `"none"` is the
age-shortcut ablation.

### 3.4 Augmentation suite (D4, D6)
Two stages, each aug a pure function `(x: np.ndarray[12, T], fs, rng, cfg) → x'` with its
own probability, composable, seeded, unit-tested for shape/finiteness/lead-consistency.

Stage **R** (raw, before rate conversion — simulates acquisition):
- `resample_roundtrip`: to fs' ∈ {300, 400, 500, 600, 1000} and back (`resample_poly`).
  This is exactly what the organisers did to the test data (Reyna 2025 §2.2).
- `device_filter`: random Butterworth HP 0.05–0.67 Hz and/or LP 40–150 Hz, order 1–2
  ("filters representative of different ECG devices", ibid.).

Stage **S** (after cleaning, before z-score — signal-level nuisance):
- `powerline`: 50 or 60 Hz ± 0.2 Hz, random phase, SNR 15–30 dB, + 2nd/3rd harmonics at
  ½ and ⅓ amplitude, identical across leads, p=0.5 (Van Santvliet 2025 Table 1; the team's
  `augmentations.py` already implements this — reuse it).
- `baseline_wander`: `a·sin(2πft)`, f ∈ [0.1, 0.5] Hz, a = 0.1 (Hong 2025).
- `gaussian_noise`: σ = 0.01 (Hong 2025).
- `amplitude_scale`: global 0.8–1.2 (Hong 2025) **and** per-lead 0.9–1.1 — the global
  factor is neutralised by the per-record z-score, the per-lead one is not.
- `time_warp`: linear resample by 0.9–1.1 (Hong 2025).
- `random_mask`: zero a random segment ≤1 s; `edge_mask`: zero 0–1.5 s at start and end
  (Hong 2025).
- `crop_shift`: crop to L1 ∈ [5.65, 10] s, shift, zero-pad to 10 s (Van Santvliet 2025;
  team's `augmentations.py`).
Then z-score per record, window, and return `valid_len` for the padding mask (D3).

Evidence this matters more OOD than in-distribution: the winner's augmentations gave
+5% on the hidden validation set but only +2.7% on their CV (Van Santvliet 2025 §3).

### 3.5 Preprocessing, unified (D7, D8, D11, D15)
One function `preprocess(record_or_array, fs, *, train: bool, aug: AugConfig|None, rng)`
returning `Sample(signal[12,5000] float32, valid_len, age_centuries, age_missing, sex_onehot,
label_hard, label_soft, source, aux_labels)`, used by the dataset **and** by inference.
- Lead reorder (existing `reorder_signal`, case-insensitive) and 12-lead guard (existing).
- `resample_poly` to 500 Hz (ECG-FM's pretraining rate, McKeen 2025).
- Cleaning: vectorised `sosfiltfilt` Butterworth band-pass 0.5–100 Hz order 2 over all
  leads at once + IIR notches at 50 and 60 Hz (Q=30). Keep `clean: "neurokit"` as the
  legacy option for the ablation. Rationale: 12 sequential `nk.ecg_clean` calls are the
  inference bottleneck; the legacy notch is 50 Hz only while Brazil's mains is 60 Hz;
  ECG-FM was pretrained on unfiltered z-scored signal (McKeen 2025), so minimal filtering
  stays closest to pretraining.
- `zscore_per_record` (existing; keep).
- Windowing to 10 s = 5000 samples, zero-pad at the end, **return `valid_len`**.

### 3.6 Model (core idea preserved)
- `ECGFMFeatureExtractor`: keep the team's hook-based capture of all 12 layers and the
  learnable softmax `layer_weights`. Add (a) a real `padding_mask` built from `valid_len`
  (token length = `valid_len // 16` for the 4×stride-2 conv frontend; verify against
  fairseq-signals' own length computation) passed to the encoder and used in pooling; (b)
  an optional input-dependent gate `aggregation: "static_softmax" | "pma"` (Nguyen et al.
  2025, arXiv:2509.00102) — default stays the team's static softmax; (c) log the learned
  layer weights at the end of training (the paper predicts mass on layers ~6–8).
- `DemographicFiLM` (§3.3), head `Dropout(0.1) → Linear(768,256) → ReLU → Dropout(0.1) →
  Linear(256,1)` unchanged.
- Optional auxiliary heads off the 256-d hidden: RBBB, 1dAVb, LBBB, AF, SB from
  `code15_exams.csv`, masked BCE (loss only where the label exists), weight 0.25. The team's
  own multitask idea (`fm/train_multitask_fm.py` in the dev tree); motivated by Jidling et
  al. 2023 (the network's Chagas evidence is RBBB/LAFB-type conduction disease) and
  Caruana 1997.
- Optional domain head with gradient reversal (Ganin et al. 2016) on the pooled feature,
  3-way source, λ ramped 0→0.1. The only top-15 team that used domain-adversarial training
  had the best ELSA-Brasil score of the top 15 (0.164, "CinCo Amigos", CinC 2025).

### 3.7 Training loop
- Patient-grouped 90/10 internal split (CODE-15 grouped by `patient_id` from the shipped
  CSV; other sources by record), stratified by source×label, fixed seed.
- `ModelCheckpoint(monitor="val_challenge", mode="max")`, `EarlyStopping(patience=3)`
  (Physiol Meas 2026 used patience 3 on the challenge score), `seed_everything`,
  `gradient_clip_val=1.0`, AMP fp16, deterministic dataloader seeding per worker.
- After training: compute `threshold = quantile(val_probs, 0.95)` and the learned layer
  weights; store both in the checkpoint `hparams`.
- Optional `ensemble.n_members` (seeds or folds): average sigmoid outputs; each member gets
  an equal share of the remaining time budget. Jidling et al. 2023 averaged 15 seeds; Ahus
  AIM used 5 folds. Inference cost scales linearly with members — with the vectorised
  cleaner the per-record cost is dominated by the encoder, so budget ≤3 members.

### 3.8 Inference
`team_code.run_model` → `chagas.inference.predict(record)`: same `preprocess` (no aug),
model(s) forward with padding mask, average over members, `pred = prob > threshold`. Keep
`NotImplementedError` re-raise semantics. Add an optional batched path for our own
evaluation (the grader's `run_model.py` is per-record and must keep working).

## 4. Code layout (maintainability)
```
chagas/                 new package; team_code.py becomes a thin adapter
  config.py             dataclasses DataCfg, AugCfg, ModelCfg, TrainCfg (+ yaml load/dump); every default carries a citation comment
  preprocessing.py      the ONE pipeline (§3.5)
  augment.py            pure functions + Compose (§3.4)
  labels.py             source parsing, soft labels, multi-report, aux labels, patient groups
  dataset.py            Dataset/DataModule/sampler over preprocessing+augment+labels
  model.py              feature extractor (+mask, +gate), FiLM, heads, LightningModule
  losses.py             weighted soft BCE, hard-negative margin ranking, masked aux BCE, GRL
  training.py           split, two-stage schedule, LLRD param groups, callbacks, budget, ensemble, threshold
  inference.py          load_model / predict
  metrics.py            challenge score (vectorised), AUROC, AUPRC
tests/                  pytest (§5)
experiments/            SLURM scripts, ablation yamls, build_test_C.py, eval_report.py
PROVENANCE.md           idea → source table (§7)
```
Rules: no bare relative paths (resolve against `Path(__file__).parent`); no module-level
side effects; type hints; docstrings cite the source of every non-team idea; `fm.py`,
`dataloader.py`, `utils.py` stay in the tree untouched for history but nothing on the
submission path imports them. `run_model.py`, `train_model.py`, `helper_code.py`,
`evaluate_model.py` and the three Dockerfile `DO NOT EDIT` lines are never modified.

## 5. Tests (pytest; must pass before any GPU run)
1. **Preprocessing parity**: 50 real records; `preprocess(train=False)` == the dataset path
   with aug off, bit-exact (`np.array_equal`).
2. **Resample invariance**: 400 Hz record → `resample_poly` to 300/600/1000 Hz → pipeline;
   max |Δ signal| after z-score < 0.05 and model |Δ logit| reported (assert < 0.25 with
   the baseline checkpoint; record the actual number).
3. **Padding mask**: a 7 s record zero-padded to 8 s vs 10 s yields identical pooled
   features (atol 1e-5); token length formula matches fairseq-signals.
4. **Labels**: source parsing; soft labels per source; multi-report 3/5 → 0.804 / 0.294;
   aux labels masked when exam_id unknown; patient grouping never splits a patient.
5. **Augment**: every aug preserves shape/dtype, finite, seeded-deterministic; lead-identical
   augs are identical across leads; `resample_roundtrip` returns exactly the input length.
6. **Losses**: weighted soft BCE == hand computation; ranking loss 0 when margin satisfied;
   all-negative batch → finite loss and gradient; GRL reverses gradient sign.
7. **Metric**: `chagas.metrics.challenge_score` == `utils.compute_challenge_score` ==
   organisers' `evaluate_model.py` on toy vectors, including ties.
8. **LLRD/two-stage**: param-group LRs follow base·decay^(12−k); after stage 1 the encoder
   has `requires_grad=False`; after the switch the 12 layers are trainable and the conv
   frontend is not; `train()`/`eval()` set encoder dropout correctly in both stages.
9. **Threshold/serialisation**: checkpoint round-trips threshold, layer weights, config.
10. **Time budget**: `TimeLimitCallback(max_hours≈0.002)` stops training and a valid
    best checkpoint still exists.
11. **Memory smoke (SLURM, titanrtx, 14.5 GiB cap)**: one stage-2 optimiser step at batch 16
    × 5000 samples in fp16 → report peak; must be < 14.5 GiB, else enable checkpointing.
12. **End-to-end smoke (SLURM)**: unmodified `train_model.py` → `run_model.py` →
    `evaluate_model.py` on the 400-record set at
    `/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/testdata`; no stray `.ckpt`
    outside `model_folder`; 400/400 outputs; threshold applied.
13. **Test-C builder**: perturbed copy is deterministic under the seed; all records still
    load through `preprocess`.

## 6. Risks and what would make us revert
- **T4 memory/time**: stage 2 may not fit or may be too slow on a T4 (2× slower than our
  TitanRTX). Mitigation: fp16, grad checkpointing, budget-aware callback; if stage 2 cannot
  complete ≥2 epochs of the full data within 80% of 72 h on TitanRTX×2, ship stage-1-only.
- **Age shortcut across cohorts** (ELSA-Brasil is a broader, asymptomatic population):
  measure with `demographics: "none"` and the per-age-decile table; if FiLM's gain is
  entirely explained by age, prefer `"none"` or clip the FiLM's γ range.
- **Soft labels + AUROC**: metrics must use hard labels (D10).
- **Bigger ≠ better OOD**: Ahus AIM found larger models improved CV but not the hidden
  test. The ensemble is therefore last in the ladder and judged on C, not A.

## 7. Provenance (what is whose)
| Idea | Origin |
|---|---|
| ECG-FM backbone, 12-layer hook capture, static learnable softmax aggregation, masked mean pooling, MLP head, age/sex side input | **The team** (fm.py) |
| Powerline / crop / shift augmentation implementation | **The team** (`augmentations.py`); parameters coincide with Van Santvliet et al. 2025 Table 1 — cite both |
| RBBB/1dAVb soft labels and multitask heads | **The team** (`dataloader._compute_soft_label`, dev-tree `train_multitask_fm.py`) |
| Soft labels 0.8/0.2 for CODE-15; w_pos=5 + 5× oversampling; two-stage frozen→full FT (2e-4 then 2e-5, eff. batch 64); FiLM demographics; select by top5%-TPR | Van Santvliet, Nguyen, Vandenberk, De Vos, *Detecting Chagas Disease Using a Vision Transformer-based ECG Foundation Model*, CinC 2025, DOI 10.22489/CinC.2025.019 |
| Multi-report soft labels; RBBB-informed bump; margin-ranking loss on hard negatives; baseline-wander / Gaussian / amplitude / time-warp / masking aug parameters; Butterworth band-pass | Hong, Chou, González, *Managing Label Uncertainty in the Detection of Chagas Disease from the ECG*, CinC 2025, DOI 10.22489/CinC.2025.100 |
| Source-stratified label smoothing; "upsampling is harmful under label noise"; early stop on challenge score | *Reliability-aware hierarchical learning for Chagas disease screening from 12-lead ECGs*, Physiol. Meas. 2026, doi:10.1088/1361-6579/ae98e0 |
| Input-dependent layer gate (PMA); middle layers most informative | Nguyen et al., *Exploiting a Mixture-of-Layers in an ECG Foundation Model*, arXiv:2509.00102 |
| 500 Hz / z-score pretraining recipe; full FT > linear probe; AUPRC checkpointing | McKeen et al., *ECG-FM: an open electrocardiogram foundation model*, JAMIA Open 2025, arXiv:2408.05178 |
| RBBB/LAFB as the ECG signature; seed ensembling; CCC-stage finding | Jidling et al., PLoS NTD 2023, 17(7):e0011118 |
| Layer-wise LR decay / discriminative fine-tuning | Howard & Ruder 2018 (ULMFiT); Clark et al. 2020 (ELECTRA) |
| FiLM | Perez et al., AAAI 2018 |
| Domain-adversarial training (GRL) | Ganin et al., JMLR 2016; applied to Chagas by "CinCo Amigos", CinC 2025 (rank 7, best ELSA score in top 15) |
| Test-time perturbation types; cohort facts; limits | Reyna et al., *Detection of Chagas Disease from the ECG: The George B. Moody PhysioNet Challenge 2025*, CinC 2025 / arXiv:2510.02202; https://moody-challenge.physionet.org/2025/ |
| Multitask regularisation | Caruana, *Multitask Learning*, Mach. Learn. 1997 |
| `resample_poly`, `sosfiltfilt` | scipy.signal (standard DSP; no paper) |
