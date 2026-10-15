# Methods: what changed relative to commit `a72c51c`, and why

Branch `improve/robust-finetune`.  The plan this implements is `PLAN.md`; the origin of
every borrowed idea is in `PROVENANCE.md`.

## The problem being solved

The challenge score is TPR@5% — recall among the top 5% of records ranked by
probability — averaged over three hidden Brazilian cohorts (REDS-II, SaMi-Trop 3,
ELSA-Brasil) at ~2% prevalence, recorded at 300/500/600/1000 Hz on different machines.
The organisers additionally perturbed the hidden records: *"we added small amounts of
various forms of noise, applied filters that are representative of different ECG
devices, and resampled the data to different sampling frequencies"* (Reyna et al.,
CinC 2025, §2.2).

Every team's in-distribution score collapsed out of distribution: median hidden
validation 0.279 → ELSA-Brasil 0.100 (−64%); the winner went CV 0.490 → test 0.323.
Our own patient-grouped held-out score (Test A 0.434 / Test B 0.440 / AUROC 0.858, run
16878337, 100k subsample, 16 epochs, frozen encoder) sits in the same in-distribution
band as the top teams' CV numbers. **The remaining problem is out-of-distribution
robustness, not in-distribution fit**, so every change below is either a fix for a
concrete defect found in the audit or something with published evidence that it helped
on the *hidden* sets.

## Structural change

`team_code.py` is now three functions over a new `chagas/` package.  The previous
version imported `transformer`, `pretrain_mae_vit_ecg`, `finetuning_mae_vit_ecg` and
`fm` (and through it `wandb`, `matplotlib` and `neurokit2`) without using most of them —
each one an import-time failure surface inside the judged container (defect **D12**).
`fm.py`, `dataloader.py`, `utils.py` and `augmentations.py` stay in the tree for history;
nothing on the judged path imports them.

```
chagas/config.py         dataclass configs + YAML; every default cites its source
chagas/preprocessing.py  the ONE pipeline, used by training and inference
chagas/augment.py        pure, seeded augmentation functions
chagas/labels.py         header parsing, soft labels, aux labels, patient groups
chagas/dataset.py        Dataset / DataModule / sampler
chagas/model.py          feature extractor, FiLM, heads, LightningModule
chagas/losses.py         weighted soft BCE, ranking loss, masked aux BCE, GRL
chagas/training.py       split, two-stage schedule, budget, threshold, ensemble
chagas/inference.py      load_model / run_model
chagas/metrics.py        challenge score, AUROC, AUPRC
```

## The core idea is unchanged

ECG-FM backbone → forward-hook capture of **all 12** transformer layers → learnable
**static softmax** aggregation → masked mean pool → 768-d feature → MLP head, with
age/sex side information.  That is the team's design and it is preserved exactly; the
input-dependent PMA gate (Nguyen et al., arXiv:2509.00102) is implemented but **off by
default**.

## Defects fixed

| # | Defect at `a72c51c` | Fix |
|---|---|---|
| D1 | No validation split, no model selection, no seed — epoch 16 shipped blind | Patient-grouped 90/10 holdout stratified by source×label; `ModelCheckpoint(monitor="val_challenge", mode="max")`; `EarlyStopping(patience=3)`; `seed_everything`; `gradient_clip_val=1.0` |
| D2 | Encoder permanently frozen; only ~197k head parameters trained | Two-stage fine-tuning: frozen stage 1 at lr 2e-4, then the 12 transformer layers unfrozen at base lr 2e-5 with layer-wise decay 0.85 and head/FiLM at 1e-4 |
| D3 | Masked mean pooling was dead code — `padding_mask` was always `None`, so a 7.3 s record's zero padding was averaged in as signal | `valid_len` is carried out of preprocessing, turned into a sample-level prefix mask and passed to `ECGTransformerModel(source=…, padding_mask=…)`; the token-level mask it returns drives both attention and pooling |
| D4 | Deterministic first-10 s truncation, no crop | Random crop to 5.65–10 s with a random start (the crop start *is* the shift) |
| D5 | Plain BCE with ~50/50 inverse-frequency `WeightedRandomSampler` (≈43× positive re-weighting at 2% prevalence) | `BCEWithLogitsLoss(pos_weight=5)` on soft targets plus 5× oversampling (the winner's setting); `oversample: 1.0` is the Physiol. Meas. 2026 ablation |
| D6 | No augmentation at all on the submission path | Two-stage augmentation suite (below) |
| D7 | FFT `scipy.signal.resample` | `scipy.signal.resample_poly` everywhere |
| D8 | 12 sequential `nk.ecg_clean` calls, 50 Hz notch only | One vectorised `sosfiltfilt` pass over all 12 leads: Butterworth band-pass 0.5–100 Hz order 2 plus **50 and 60 Hz** notches (Q=30).  `clean: "neurokit"` is retained for the ablation |
| D9 | Hard-coded 0.5 threshold on a heavily positive-re-weighted model | Threshold = 95th percentile of the validation probabilities, stored in the checkpoint and applied at inference |
| D10 | Raw logits into `Accuracy`, `labels.int()` truncating soft labels, float targets into AUROC | All metrics take sigmoid probabilities and **hard** labels |
| D11 | Three hand-synchronised copies of the preprocessing pipeline | One `preprocess_record`, called by the dataset and by inference; `tests/test_preprocessing.py` asserts bit-identity on 50 real records |
| D12 | Unused heavy imports in `team_code.py` | Thin `team_code.py` |
| D13 | `except TypeError: model(sig_t)` fallback that could never succeed | Removed |
| D14 | `assert label is not None and label == 0 or label == 1` precedence bug | Gone with the rewrite; label parsing is now in `chagas/labels.py` with tests |
| D15 | `sex.lower()` without an `isinstance` guard on the inference path only | One code path; `demographics_vector` guards and is tested with a non-string sex |
| D16 | Missing age encoded as 0.0 == the training mean | `[age/100, age_missing, sex_f, sex_m, sex_unknown]`, with the organisers' >89→90 clipping mirrored |

## New capability

**Soft labels (§3.2).** SaMi-Trop → 1.0, PTB-XL → 0.0, CODE-15 → 0.8/0.2, unknown source
treated as CODE-15-like.  For CODE-15 patients with more than one exam, Hong et al.'s
multi-report consistency replaces the flat value with `0.51·y + 0.49·(n_pos/n_total)`
(3 of 5 positive → 0.804 / 0.294, unit-tested).  The team's own RBBB/1dAVb scheme and
Hong's RBBB bump are available as `rbbb_scheme: "team_rbbb" | "hong_rbbb"`, both off by
default.

**Augmentation (§3.4).** Stage **R** runs at the record's native rate and reproduces the
organisers' own perturbations: a resample round trip through {300, 400, 500, 600, 1000}
Hz and a random Butterworth "device" filter (HP 0.05–0.67 Hz, LP 40–150 Hz, order 1–2).
Stage **S** runs after cleaning and before the z-score: powerline at 50/60 Hz with
harmonics, baseline wander, Gaussian noise, global and per-lead amplitude scaling, time
warp, interior and edge masking, and the crop.  Every augmentation is a pure function of
`(x, fs, rng, …)` and is unit-tested for shape, finiteness, purity, seed-determinism and
(where it is common-mode interference) lead-identity.

**FiLM demographics (§3.3).** `MLP([age/100, age_missing, sex_f, sex_m, sex_unknown]) →
(γ, β)`, zero-initialised so training starts at the identity.  `demographics: "concat"`
reproduces the old behaviour and `"none"` is the age-shortcut ablation.

**Two-stage fine-tuning with LLRD (§3.1).** Stage 1 trains the aggregation weights, FiLM
and the head on a frozen backbone for ~2,000 optimiser steps at lr 2e-4, batch 16 with
×4 accumulation (effective 64).  Stage 2 unfreezes the 12 transformer layers at base lr
2e-5 with layer-wise decay 0.85 — layer 12 gets 2e-5 and layer 1 gets 2e-5·0.85¹¹ ≈
2.8e-6 — and head/FiLM at 1e-4, on a cosine schedule with 5% warm-up.  The conv frontend
stays out of the graph because the checkpoint ships `feature_grad_mult = 0.0`;
`train_conv_frontend: true` overrides that.  `encoder_layerdrop` is asserted to be 0.0 at
construction, because layer aggregation needs every layer to run on every forward.
Crucially, the `train()` override that keeps a *frozen* encoder in `eval()` now gets out
of the way once the backbone is unfrozen, so the checkpoint's `activation_dropout = 0.1`
actually regularises in stage 2.

**Budget.** `TimeLimitCallback` is now budget-aware: stage 1 gets 5% and stage 2 80% of
`max_hours`, ensemble members split the total, and there is an absolute deadline shared
across stages.  It sets `trainer.should_stop` rather than raising, so Lightning finishes
the epoch, runs validation and lets `ModelCheckpoint` save.

**Precision.** fp16 AMP (`precision: "16-mixed"`).  bf16 is rejected by
`TrainCfg.__post_init__` and by `chagas.training._assert_no_bf16`: the grader's T4 and
our TitanRTX are both SM 7.5, which has fp16 tensor cores but no native bfloat16.

**Optional, all default-off and behind config flags:** auxiliary RBBB/1dAVb/LBBB/AF/SB
heads with a masked BCE, margin-ranking loss on the batch's hardest negatives, a
domain-adversarial source head with a ramped gradient-reversal layer, the PMA layer
gate, and a seed ensemble.

## Evaluation harness

`experiments/` contains one YAML per rung of the ablation ladder, `run_ladder.sh` (which
trains through the **unmodified** `train_model.py` and infers through the **unmodified**
`run_model.py`, both under a 14.5 GiB CUDA allocator cap that emulates the T4), and
`eval_report.py`, which scores three views of a single inference run:

* **A** — the full held-out set at natural composition;
* **B** — the CODE-15-only subset (positives and negatives share a source, so the
  device/filter signature cannot be used as a shortcut);
* **C** — A perturbed the way the organisers perturbed the hidden cohorts
  (`build_test_C.py`, seeded per record from its own stem, cached as WFDB).

plus per-source mean probability, score by age decile, and the **C − A gap**.

The challenge score is computed in closed form.  The organisers' reference
implementation averages over 10,000 random permutations, each a full argsort, which is
~15 minutes on 73,300 records and unusable as a per-epoch validation metric.  For a
record with probability `p`, `P(it lands in the top c) = clip(c − #{q > p}, 0,
#{q == p}) / #{q == p}`, so the expectation is exact and vectorised.  Without ties it is
bit-identical to `helper_code.compute_challenge_score`; with ties it is the exact value
their Monte-Carlo estimates (both asserted in `tests/test_metrics.py`).

## Measured findings worth recording

Two of PLAN.md's test thresholds turned out not to be attainable, for reasons that are
properties of the backbone rather than bugs.  Both tests were kept, with measured
tolerances and the explanation in their docstrings.

**1. ECG-FM's conv frontend is not padding-aware.**  Its first block is
`Fp32GroupNorm(256, 256)` — a per-channel instance norm **over the time axis** — and the
k=128 convolutional positional encoding is unmasked as well.  Changing how much trailing
zero padding a record carries therefore changes the normalisation statistics of every
channel and moves the encoder's inputs at *valid* positions too.  Measured on a 7 s
record padded to 8 s versus 10 s: max |Δ pooled feature| = **0.78 with the padding mask**
versus **1.29 without** (feature scale 0.60).  The mask does its job — it removes padded
tokens from attention and pooling, and the masked-versus-unmasked pooled feature differs
by 2.42 on a padded record — but exact pad-length invariance is unreachable without
modifying the vendored backbone, which is out of scope here.  Practical consequence:
because every record is padded to the same 5000 samples, the distortion is a systematic
function of record length, and record length correlates with source.  The `crop_shift`
and `edge_mask` augmentations, which vary `valid_len` during training, are the mitigation
that is in place.

**2. A rate round trip moves the signal more than 0.05 in z-score units.**  Measured
over 8 real 400 Hz records × {300, 600, 1000} Hz, with 0.5 s dropped at each end: worst
max |Δ| = **1.03**, worst RMS |Δ| = **0.092**, worst correlation **0.9965**.  Two causes:
downsampling to 300 Hz genuinely removes QRS content between 100 and 150 Hz that a
second-order 100 Hz band-pass only attenuates by ~24 dB/octave, and the per-record
z-score divides each lead by its own standard deviation, so a low-amplitude precordial
lead (V2 at 0.95 mV in record 1000087) multiplies a tiny absolute difference into a large
one.  What actually matters is the model-level number, and that is small: worst
**|Δ logit| = 0.0102** over the same round trips (against the plan's < 0.25 bound), with
the shipped backbone and a seeded random head.

**3. The 12-layer aggregation is safe in train mode.**  `encoder_layerdrop` is 0.0 in
this checkpoint, and fairseq-signals only applies layerdrop when `self.training` is true.
The construction-time assertion makes that a hard invariant rather than a coincidence.

## What was deliberately *not* done

* The legacy FFT `scipy.signal.resample` is not selectable — D7 is fixed
  unconditionally — so the row-0 config reproduces the baseline in every respect except
  that one and the "train on all records, ship epoch 16" policy.  The authoritative
  `a72c51c` numbers remain the recorded run 16878337.
* The ablation ladder itself was not run; it is hours of GPU time per rung.  The batch
  scripts are written, verified in shape, and listed in `experiments/README.md`.

## Round 2: the record-length shortcut (PLAN_R2.md)

**What round 1 got wrong.**  In CODE-15 the recording length predicts the label: records are 7.34 s (2,934
samples at 400 Hz) or 10.24 s (4,096), with Chagas prevalence 3.05% vs 0.24% (12.7x).  A score that knows only the
length reaches AUROC 0.681 on the CODE-15 test set, and every round-1 model scores short negatives higher than long
ones (shipped model: 0.62 vs 0.39).  The hidden cohorts are ~10 s, so the cue is absent there.  Views A, B and C
are all inflated by it (view B removed only the *source* cue).  This discovery and the within-length evaluation
below are this project's own (DECISIONS.md D-2026-09-30-2).

**Evaluation (`experiments/scoring.py`, `eval_report.py`, `aggregate.py`).**  Durations come from every record's
WFDB header (`experiments/build_durations.py`; all 366,185 checked against `signal_lengths.csv`, 0 mismatches).
New views, CODE-15 only so the source cue is gone too: **S7** (< 7.5 s; 1,259 test positives; primary), **S10**
(>= 9.9 s; 77 positives; the hidden cohorts' format) and **S10+** (S10 negatives plus every >= 9.9 s positive,
CODE-15 and SaMi-Trop; ~240 positives; secondary).  Every view has a 1,000-resample bootstrap 95% CI; every report
also gives the duration-only baseline and the **length-cue gap** (mean probability of CODE-15 negatives, short
minus long).  `aggregate.py` gives mean ± sd over seeds, the seed ensemble, and paired-bootstrap differences against
a baseline (every model scored on the same resamples).  The bootstrap uses integer resampling weights on scores
sorted once, which is exactly the metric on the resampled set because both metrics treat ties exactly.
All 22 saved round-1 runs are re-scored in `experiments/results/RESCORED.md`.

**Changes under test (all flags; defaults = round 1, asserted by tests):**

* **Length-matched negatives** (`data.match_negatives_on: duration`).  When negatives are capped, their
  distribution over length groups (short / mid / long, read from each record's own header) is made equal to the
  positives' — frequency matching of controls to cases on a confounder, the "subsample groups" remedy for shortcut
  learning (Idrissi et al. 2022; Sagawa et al. 2020; Geirhos et al. 2020).  Seeded; a group short of negatives is
  taken whole and the rest redistributed.  Needs headers only, so it works in the grader's container.
* **Fixed 5 s windows** (`preprocess.window_seconds: 5`, `window_mode: multi`).  ECG-FM was pretrained on 5 s at
  500 Hz (McKeen et al. 2025); feeding 10 s padded 56% of training records by 27%.  Training takes one random 5 s
  crop from the valid region (after the whole-record z-score); validation and inference take
  `K = min(4, ceil(T / 5 s))` evenly spaced windows (2 for 7.34 s and 10 s, 3 for 10.24 s) and average their
  **logits** (FjordNet, CinC 2025).  No record >= 5 s is padded, which removes the GroupNorm padding distortion
  recorded above and the visibility of record length.  `crop_shift` is replaced by the random crop in this mode.
* **WiSE-FT** (`train.wise_ft: select`).  After stage 2, `θ(α) = (1 − α)·θ_stage1 + α·θ_stage2` is evaluated on
  validation for α ∈ {0, 0.5, 1}; the best validation challenge score ships (ties towards smaller α).  α = 0 is the
  frozen-encoder model, α = 1 the fine-tuned one (Wortsman et al., CVPR 2022; Kumar et al., ICLR 2022).
  Experiments also write each α as a model folder.
* **Seed ensemble** (unchanged code, now tested): members' probabilities are averaged and each member gets
  `max_hours / members` of the budget.

**Logging and reproducibility.**  Every validation line and manifest record now carries the global optimiser
step, the within-length validation scores (CODE-15 short and long), windows per record, and the WiSE-FT α it
scored; the CSV log gets the same columns.  `chagas_manifest.json` records train/val counts by source × label ×
length group.  Every run writes `results/<run>/run_info.json` (commit, dirty flag, config SHA-256, seed, host, GPU,
library versions, `pip freeze` hash, times).  `experiments/REPRODUCE.md` goes from a clean clone to every table.

**Batched inference for experiments** (`experiments/predict_batched.py`): the same model loading, preprocessing
and batching code as `team_code.run_model`, with a DataLoader and batches of 64; asserted equal to the
`run_model.py` path (max |Δprob| < 1e-4).  The unmodified `run_model.py` stays the contract test.

**Decision rule (pre-registered, PLAN_R2.md §5).**  Primary: 3-seed mean TPR@5% on S7; co-primary S7 AUROC; S10+
AUROC must not drop by more than its SE; the length-cue gap must shrink.  `aggregate.py` applies it mechanically.

## Round 3: a validation protocol that cannot see the shortcut (PLAN_R3.md)

- **Validation split** (`train.val_protocol: fixed`): carved before the negative cap, at natural prevalence,
  stratified by source × label × length group, shuffled by a fixed `val_seed` with one generator per stratum
  (so removing a source leaves every other stratum's holdout unchanged), PTB-XL grouped by patient as in the
  held-out split.  Legacy order (cap, then split, seeded by the training seed) remains the default.
- **Per-view validation metrics**: `val_challenge_s7`, `val_auroc_s7`, `val_auroc_s10p` from the same
  predictions as the pooled metrics; any of them can be `train.monitor`; a view with one class falls back to
  `val_challenge` with a warning (small smoke sets).
- **Prevalence cap** (`data.cap_mode: prevalence`): negatives = n_pos·(1 − p)/p, uniform or length-matched.
- **Test-time windows** (`preprocess.window_stride_seconds`): overlapping 5 s windows at inference, last one
  right-aligned, thinned to `max_windows` keeping first and last; the judged path is unchanged unless a config sets it.
- **Source exclusion** (`data.exclude_sources`), used by W1S to drop PTB-XL.
- **Split audit** (`experiments/check_split.py`) re-verifies the held-out split and writes `SPLIT_AUDIT.md`.
- **Container**: Apptainer definitions mirroring the Dockerfile, because no Docker daemon is reachable here.
