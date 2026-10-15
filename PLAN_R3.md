# Round 3 plan: a validation protocol that cannot see the shortcut, and the submission candidate

Author: Claude (2026-10-15), approved by the user. Follows `PLAN.md` (round 1), `PLAN_R2.md` (round 2)
and `DECISIONS.md`. The team's core idea is unchanged: ECG-FM backbone → learnable aggregation over
12 layers → pooled feature → MLP head. Borrowed methods are attributed in §8 and in the code.

## 1. Where round 2 left us, and what this round is for

Round 2 found the record-length shortcut in CODE-15 and showed that fixed 5 s windows (W1) remove it,
with a large gain on 10 s recordings (S10+ AUROC 0.779 → 0.835, +0.056 ± 0.010) and no significant
change within 7.34 s records (S7 TPR@5% −0.004 ± 0.004). The round-2 rule took S7 as primary and S10+
as a guard, so it adopted nothing (D-2026-10-01-5).

The goal now is a single entry that the Challenge organisers can score on the hidden cohorts, all of
which are ~10 s recordings. Three things must happen before that external test is spent:

1. **Validation must not reward the shortcut and must not change with the seed.** Today the
   validation set is carved *after* the negative cap and shuffled with the training seed
   (`chagas/training.py`, `_subsample` then `split_entries`), so each seed has its own validation set
   at 6.5% prevalence (the test set and the hidden data are at natural prevalence, 2.27% here), it is
   length-matched whenever training is, and checkpoints are selected on *pooled* `val_challenge`,
   which the length cue inflates. PTB-XL is grouped per record at training time but per patient in
   the held-out split.
2. **The entry must train its own seed ensemble inside the organisers' 72 h on a T4.** The
   mechanism exists (`ensemble_members`, per-member budgets, probability averaging); it has not been
   exercised as a submission.
3. **One more controlled round** of the cheapest changes with a plausible 10 s-format gain, judged by
   a rule written down here before launch.

The held-out test split is **kept unchanged**: 20% of patient groups stratified by source × label
(seed 12345), no group in both splits, no duplicate stems, length represented proportionally
(S7 1,259 / S10 77 / S10+ 240 positives). Every one of the 31 earlier runs is scored on it, and a
re-split cannot create more 10 s positives (CODE-15 has 317 in total). The audit is re-run and
written to `experiments/results/SPLIT_AUDIT.md`.

## 2. Validation protocol (`train.val_protocol: fixed`)

| | Legacy (rounds 1–2) | Fixed (round 3 and the entry) |
|---|---|---|
| Carved | after the negative cap | before it; never capped, never length-matched |
| Shuffle seed | `train.seed` (changes per seed) | `train.val_seed` = 12345, independent of the training seed |
| Strata | source × label | source × label × length group (short / mid / long) |
| PTB-XL grouping | per record | per patient (`ptbxl_database.csv`), as in the held-out split |
| Prevalence | ≈ 6.5% | natural (≈ 2.27%) |
| Monitor | pooled `val_challenge` | `val_challenge_s7` (CODE-15 < 7.5 s), with `val_auroc_s7` and `val_auroc_s10p` logged |

`val_fraction` stays 0.1: about 29k validation records and 650 positives. Seeds now differ only in
initialisation and sampling, so seed-to-seed spread measures model variance, and ensemble members are
selected on the same data. Legacy behaviour is the default so every committed config still reproduces.

## 3. Changes to test

All round-3 configs start from the shipped configuration (`finalA_negcap`) plus the fixed protocol and
a prevalence-based negative cap (`data.cap_mode: prevalence`, `cap_prevalence: 0.065`, the ratio the
100,000-record cap gives on our split; on the organisers' full set the count cap would give 8.2%).

| Key | Variant | Why |
|---|---|---|
| B0 | shipped config under the new protocol | required re-baseline; also measures the protocol change itself against round-2 B0 |
| W1 | + fixed 5 s windows (round-2 flags) | the candidate carried over from round 2 |
| W1R | W1 + hard-negative ranking loss (Hong et al., CinC 2025; weight 2.5, margin 0.5, top 20% of negatives) | acts on the top-5% ranking the score measures; +0.02 internal in the cited paper; implemented in round 1, never run with seeds |
| W1S | W1 without PTB-XL | PTB-XL is European, 500 Hz, exactly 10.0 s, all negative and absent from the hidden data; CODE-15 already has 105k ≥ 9.9 s negatives. Tests whether the presumed negatives help or only teach a source cue |
| W1 ×5 | W1 with seeds 45 and 46 added | ensemble size 3 vs 5 for the entry |
| W1-TTA | inference only: the W1 members scored with overlapping windows (2.5 s stride) | free test-time averaging |

Not in this round (same reasons as PLAN_R2 §9): distillation of the MIMIC head, early-learning
regularisation, domain-adversarial heads; and lower-LR / frozen-layer regularisation of stage 2, which
is deferred until the above are settled.

## 4. Experiment matrix

Seeds 42/43/44 (W1 also 45/46). Train from `fixbranch_test/data_train_full` through the unmodified
`train_model.py`; score A and C with `predict_batched.py`, then views A/B/C/S7/S10/S10+ with the
round-2 diagnostics (length-cue gap, duration-only baseline, per-source mean probability).

| Name | Runs |
|---|---|
| B0 | 3 |
| W1 | 5 |
| W1R | 3 |
| W1S | 3 |
| W1-TTA | 0 (re-scores W1) |

14 runs, one SLURM array `0-13%4`, `jag-standard`, `gpu:titanrtx:1`, 14.5 GiB cap, tests gate the
array (`afterok`), the table job follows (`afterany`). Expected 1.4–2.3 h per run.

## 5. Pre-registered decision rule

The submission target is 10 s data, so the primary view changes from round 2. This is a deliberate
departure, recorded as a decision, not a post-hoc one.

- **Primary:** S10+ AUROC, 3-seed mean, paired bootstrap against B0 (round-3 baseline).
  **Guard:** S7 TPR@5% not lower than B0 by more than one paired SE.
- **Candidate selection.** W1 is the candidate carried in. A variant (W1R, W1S) replaces it if it
  beats W1 on S10+ AUROC by more than one SE while passing the guard against B0, or beats W1 on S7
  TPR@5% by more than one SE without losing S10+ AUROC by more than one SE. Ties go to the simpler
  configuration. If W1 itself fails the guard against B0 under the new protocol, the candidate is B0
  and the 10 s result is reported as unconfirmed.
- **Ensemble size.** 5 members if the 5-seed W1 ensemble beats the 3-seed ensemble on S10+ AUROC by
  more than that difference's SE and the projected T4 time stays under 48 h; otherwise 3.
- **TTA.** Adopted if it loses on none of S7 TPR@5%, S7 AUROC, S10+ AUROC and gains on at least one by
  more than one SE.
- **Diagnostics that must hold.** The length-cue gap stays near zero for every W1-based variant. For
  W1S the per-source mean probability of SaMi-Trop positives minus CODE-15 positives must not grow;
  if it grows, W1S is not doing what it claims regardless of its score.
- Views A/B/C and S10 (77 positives) are reported, not decided on.

## 6. Metrics logging and reproducibility

- All validation metrics (pooled, S7, S10+) per validation in the CSV logger and `best_by_stage`.
- `run_info.json` per run as in round 2 (commit, config hash, seed, host, GPU, versions, times).
- `chagas_manifest.json["data_summary"]` must show validation counts identical across seeds of a
  variant and validation prevalence equal to the natural prevalence of the training split.
- `experiments/results/R3_TABLE_{A,C}.{md,csv,_gap.csv,_rule.csv}` from `aggregate.py --rule r3`.
- `experiments/results/SPLIT_AUDIT.{md,csv}` from `experiments/check_split.py` (SLURM, CPU).

## 7. Tests

Fixed split identical across seeds and independent of cap and matching; legacy split unchanged;
strata complete and validation prevalence natural; PTB-XL patient grouping; prevalence cap count;
monitor validation and empty-view fallback; S7/S10+ masks equal `experiments/scoring.view_mask`;
overlapping windows cover the record and reduce to the old placement at the default stride; defaults
unchanged; committed configs round-trip; split audit on a synthetic frame; aggregate `--rule r3`.

## 8. Provenance

- Hard-negative ranking loss: Hong, Chou and González, CinC 2025 (doi 10.22489/CinC.2025.100).
- Test-time averaging over windows: standard test-time augmentation; window averaging as in FjordNet,
  CinC 2025 (091) and `PLAN_R2.md` §3.2.
- Stratified, grouped validation at natural prevalence; monitoring on the shortcut-free view: this
  project (follows from D-2026-09-30-2).
- Removing a source that exists only as negatives: the group-shortcut literature cited in round 2
  (Geirhos et al. 2020; Idrissi et al. 2022); the decision to test it is this project's.
- Seed ensembles: Lakshminarayanan et al. 2017; Jidling et al. 2023.

## 9. After the rule

1. Set `chagas/config.py` defaults to the chosen entry (`ensemble_members` 3 or 5, fixed protocol,
   prevalence cap, S7 monitor); `experiments/configs/final_r3.yaml` equals the default.
2. Time-budget check: the exact entry path (`train_model.py` then `run_model.py`) on SLURM with
   per-member wall time recorded; at a 2.5× T4 slowdown the budget must keep a 30% margin.
3. Container check with Docker or Apptainer (both exist on the login node; the build must run where
   the daemon is reachable) on the 400-record smoke set.
4. Tag `r3-submission`; record the decision; update README, REPRODUCE §7, METHODS, PROVENANCE, and the
   manuscript (Methods 2.1/2.2, Results 3.6).
