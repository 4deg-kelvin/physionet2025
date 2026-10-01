# Round 2 plan: remove the record-length shortcut, then re-measure

Author: Claude (2026-09-30). Follows `PLAN.md` (round 1) and `DECISIONS.md`. The team's core idea is
unchanged: ECG-FM backbone → learnable aggregation over 12 layers → pooled feature → MLP head.
Every borrowed method is attributed in §8 and must be cited in code docstrings and `PROVENANCE.md`.

## 1. What round 1 got wrong, and the evidence

**Finding (2026-09-30).** In CODE-15 the recording length predicts the label. Records are either
7.34 s (2,934 samples at 400 Hz; 204,522 records) or 10.24 s (131,734). Chagas prevalence is 3.05% in
the short group and 0.24% in the long group, a 12.7× difference: 6,236 of 6,559 CODE-15 positives are
short records. 56% of all training records are 7.34 s and therefore sit in our 10 s window with 27%
zero padding. All three hidden cohorts are ~10 s recordings (Reyna et al. 2025), so the cue is absent
at test time.

- A model that knows *only* the record length scores AUROC 0.681 on the CODE-15 test set (view B).
- Every model trained so far uses the cue. The shipped model's mean probability for CODE-15
  **negatives** is 0.62 for 7.3 s records and 0.39 for 10.2 s records.
- Scored *within* the 7.3 s group (1,259 test positives; bootstrap SE ±0.0125 TPR@5%, ±0.007 AUROC):
  original row 0 = 0.377 / 0.807; shipped mean of 3 seeds = 0.384 / 0.787; shipped 3-seed ensemble =
  0.404 / 0.801. So round 1's single-model gain over the original is inside the noise, and only the
  ensemble holds an edge.
- Within the 10.2 s group (77 positives; SE ±0.037 AUROC) models score AUROC 0.62–0.74, the same range
  top teams reported on the hidden cohorts (0.57–0.74).
- Round 1 called view B "source shortcut removed". It removed the *source* cue only. Views A, B and C
  are all inflated by the length cue. See `experiments/README.md` (re-scored table).

**Corroboration from the literature (8 CinC 2025 papers, read in full).** Internal scores moved opposite
to hidden scores in five of them (158, 416, 091, 037, 132), with no cause identified. The winner's
random-crop augmentation and the rank-3 team's edge masking "to simulate shorter signals" both blur
record length, and both helped more on hidden data than internally. The team's own paper (416) lists
"random padding or random windowing" as untested future work. No paper reports the length–label
association.

## 2. Evaluation overhaul (do this first; everything else is judged by it)

All scoring is from saved predictions, so re-scoring old runs costs no GPU time.

**New views** (CODE-15 only, so the source cue is also absent):
| View | Records | Test positives | Role |
|---|---|---|---|
| **S7** | CODE-15, duration < 7.5 s | 1,259 | **Primary.** Powered; no length or source cue. |
| **S10** | CODE-15, duration ≥ 9.9 s | 77 | Closest format to the hidden cohorts; low power; always with a bootstrap CI. |
| **S10+** | CODE-15 ≥ 9.9 s negatives + all ≥ 9.9 s positives (CODE-15 and SaMi-Trop) | ~240 | More power for the 10 s format; SaMi-Trop positives carry a source cue, so secondary only. |
| A, B, C | as in round 1 | | Reported for continuity. **Not decision metrics**: they should fall when the cue is removed. |

**Diagnostics on every report:** the duration-only baseline (AUROC and TPR@5% of a score that is 1 for
short records); mean predicted probability for CODE-15 negatives by length group (the "length-cue
gap", which should go to ~0); bootstrap 95% CIs (1,000 resamples) for every view.

**Tools:**
- `experiments/data/record_durations.csv` (stem, n_samples, fs, duration_s for every record), built by
  a committed script from the WFDB headers and cross-checked against the existing
  `physionet2025/signal_lengths.csv`.
- `eval_report.py` gains the views, diagnostics and CIs above.
- `experiments/aggregate.py`: given run names, prints and writes (markdown + CSV) per-config mean ± sd
  across seeds, the seed-ensemble score (mean of probabilities), and paired-bootstrap differences
  against a named baseline (resampling test records, paired across models).
- `experiments/predict_batched.py`: our own batched inference (DataLoader, workers, batch 64) writing
  the same predictions CSV as the `run_model.py` path. Must match the `run_model.py` outputs on the
  400-record smoke set to max |Δprob| < 1e-4 (tested). Used for experiments only; the unmodified
  `run_model.py` path stays the contract test and is used for the final candidate.
- Re-score every existing run into `experiments/results/RESCORED.md`.

## 3. Changes to test

All new behaviour sits behind config flags that default to the current shipped behaviour, so the
validated default does not move until §5's rule says so.

### 3.1 Length-matched negative sampling (`data.match_negatives_on: "none" | "duration"`)
When capping negatives (`data.max_records`), choose them so their distribution over length groups
(short < 7.5 s, mid, long ≥ 9.9 s; read from each record's own header) equals the positives'
distribution. Record length then carries no label information in the training or internal-validation
data, so neither the loss nor checkpoint selection can reward the cue. This is frequency matching of
controls to cases on a confounder, the standard epidemiological design, and the "subsample groups"
remedy for shortcut learning (Idrissi et al. 2022; Sagawa et al. 2020; Geirhos et al. 2020). Seeded and
deterministic; if a group lacks negatives, take all of them and redistribute the remainder. It works in
the grader's container because it needs headers only. Costs nothing: we already discard two thirds of
the negatives.

### 3.2 Fixed 5 s windows (`preprocess.window_seconds: 5.0`, `preprocess.window_mode: "multi"`)
ECG-FM was pretrained on 5 s segments at 500 Hz (McKeen et al. 2025); we feed it 10 s. With 5 s
windows no record ≥ 5 s is ever padded (99.6% of records), which removes both the padding distortion
found in round 1 (GroupNorm over time) and the *visibility* of record length.
- Training: one random 5 s crop from the valid region per sample per epoch (seeded).
- Validation and inference: K = ceil(T / 5 s) evenly spaced windows from 0 to T − 5 s (two for both
  7.34 s and 10 s records; cap K at 4), run as one batch, **mean of logits**.
- z-score over the whole record before windowing, as in ECG-FM's pretraining and as now.
- Records shorter than 5 s keep the padding mask. `crop_shift` is replaced by the random crop in this
  mode; the other augmentations are unchanged.
- Random 5 s training crops with window averaging at inference were used by FjordNet (CinC 2025, 091).

### 3.3 WiSE-FT weight interpolation (`train.wise_ft_alphas: [0.0, 0.5, 1.0]`, `train.wise_ft: "off" | "select"`)
Stage-2 fine-tuning peaks at its first validation and then drifts, strengthens source cues, and on the
hidden sets a frozen encoder beat a fine-tuned one for CinCo Amigos (158) and for the team's own entry
(416). WiSE-FT (Wortsman et al., CVPR 2022) interpolates θ(α) = (1 − α)·θ_stage1 + α·θ_stage2 and is the
standard remedy for fine-tuning that loses out-of-distribution robustness (see also Kumar et al., ICLR
2022). α = 0 is the frozen-encoder model and α = 1 the fine-tuned one, so one training run measures the
whole path. For experiments, write a checkpoint per α; `"select"` ships the α with the best validation
challenge score. All per-α validation metrics go to the manifest and the CSV log.

### 3.4 Seed ensemble (`train.ensemble_members`, already implemented)
Measured offline on saved predictions: three seeds of the shipped config score 0.4525 on view A against
a single-model mean of 0.436, and 0.404 vs 0.384 on S7 (deep ensembles: Lakshminarayanan et al. 2017;
5 members by ISIBrno, rank 4; 15 seeds by Jidling et al. 2023). Verify the inference path averages
members and that the time budget is split. The final candidate is the 3-seed ensemble of the winning
config.

## 4. Experiment matrix
Seeds 42/43/44. Train from `fixbranch_test/data_train_full` through the unmodified `train_model.py`
with the shipped negative cap. Score A and C with `predict_batched.py`, then all views.

| Name | Change vs shipped | Runs |
|---|---|---|
| B0 | none (exists: `finalA_negcap`, `lrsweep_lr2e5_s43/44`) | 0 new |
| M1 | + length-matched negatives | 3 |
| W1 | + 5 s windows | 3 |
| MW | both, with WiSE-FT checkpoints at α ∈ {0, 0.5, 1} | 3 (9 scored models) |

One SLURM array, at most 4 concurrent, `jag-standard`, `gpu:titanrtx:1`, 14.5 GiB cap.

## 5. Pre-registered decision rule
Primary: 3-seed mean TPR@5% on **S7**; co-primary: S7 AUROC. A configuration replaces B0 if its S7
TPR@5% is higher by more than the paired-bootstrap SE of the difference, or its S7 AUROC is higher by
more than 0.010 with S7 TPR@5% not lower by more than one SE; and in either case its S10+ AUROC is not
lower by more than that view's SE. If several qualify, take the highest S7 TPR@5%. Among WiSE-FT α
values, the same rule applied to MW's α set. The length-cue gap must shrink for M1/MW/W1 or the change
is not doing what it claims. Views A/B/C are reported and are expected to fall.

## 6. Metrics logging and reproducibility
- CSV logger (round 1, merged): per-step loss and learning rate, all validation metrics.
- Add to validation logging: global optimiser step on every validation line; validation AUROC and
  TPR@5% within the short and long length groups; number of windows per record; per-α WiSE-FT metrics.
- Every run writes `results/<name>/run_info.json`: git commit and dirty flag, config path and SHA-256,
  seed, hostname, GPU, torch/lightning/numpy versions, SHA-256 of `pip freeze`, start and end time,
  train/val record counts by source × label × length group.
- `experiments/REPRODUCE.md`: from a clean clone to every table in `experiments/README.md` — build the
  split, Test C, the durations file, the configs; the exact `sbatch` lines; the `aggregate.py` calls.
- All configs are generated by `make_configs.py`; historical configs keep pinning what they ran with.

## 7. Tests (must pass on SLURM before any GPU run)
1. Durations: script output equals header-derived values on a sample; matches `signal_lengths.csv`.
2. Matched sampling: negatives' length distribution equals positives' within rounding; every positive
   kept; seeded-deterministic; no-op when `"none"`; graceful when a group has too few negatives.
3. Windows: K and start positions for 7.34 s, 10 s, 4 s and 23 s records; no padding for records ≥ 5 s;
   training crops lie inside the valid region and are seeded; inference logit equals the mean of the
   per-window logits; `window_mode: "single"` reproduces the current behaviour bit-exactly.
4. WiSE-FT: α = 0 and α = 1 reproduce the stage-1 and stage-2 state dicts exactly; α = 0.5 is the
   elementwise mean; non-float buffers handled; selected α is stored and reloaded.
5. Batched predictor equals the `run_model.py` path on the smoke set (max |Δprob| < 1e-4).
6. `eval_report.py` views: hand-built toy predictions with known within-group scores; duration-only
   baseline; CIs contain the point estimate.
7. `aggregate.py`: ensemble of identical models equals the single model; paired difference of a model
   with itself is exactly 0.
8. Existing 89 tests still pass; the shipped default config is unchanged (asserted against
   `finalA_negcap.yaml`).

## 8. Provenance
| Idea | Origin |
|---|---|
| Discovery of the length–label association in CODE-15 and the within-length evaluation | This project (analysis of `signal_lengths.csv` and saved predictions, 2026-09-30) |
| ECG-FM backbone, 12-layer aggregation, MLP head | **The team** (`fm.py`); "random padding or random windowing" named as future work in the team's paper, Nguyen et al., CinC 2025, DOI 10.22489/CinC.2025.416 |
| CODE records are 7–10 s | Ribeiro et al., Nat. Commun. 2020, 11:1760 |
| 5 s segments at 500 Hz as ECG-FM's input | McKeen et al., JAMIA Open 2025; arXiv:2408.05178 |
| Random 5 s training crops, window averaging at inference | FjordNet, CinC 2025, DOI 10.22489/CinC.2025.091 |
| Shortcut learning; balancing/subsampling groups to remove a spurious cue | Geirhos et al., Nat. Mach. Intell. 2020; Idrissi et al., CLeaR 2022; Sagawa et al., ICLR 2020 |
| WiSE-FT weight interpolation; LP-FT analysis | Wortsman et al., CVPR 2022; Kumar et al., ICLR 2022 |
| Frozen encoder better than fine-tuned on hidden data | Wang et al. (CinCo Amigos), CinC 2025, DOI 10.22489/CinC.2025.158; Nguyen et al. 2025 (416) |
| Deep ensembles; multi-member ensembles for Chagas | Lakshminarayanan et al., NeurIPS 2017; Nejedly et al. (ISIBrno), CinC 2025, DOI 10.22489/CinC.2025.132; Jidling et al., PLoS NTD 2023 |
| Hidden cohorts are ~10 s; cohort facts | Reyna et al., CinC 2025 / arXiv:2510.02202 |

## 9. Not in this round (recorded so they are not lost)
Distilling ECG-FM's own 17-label MIMIC head into the student to limit feature drift (Learning without
Forgetting; the multi-pathology distillation of ISIBrno, rank 4); early-learning regularisation (Liu et
al. 2020; used but never ablated by CinCo Amigos); hard-negative ranking loss (row 7) and auxiliary
heads (row 6), both configs ready; a re-split that puts more 10 s positives in the test set.
