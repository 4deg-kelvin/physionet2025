# Decision log — branch `improve/robust-finetune`

One entry per decision that shaped the work, newest last. Each records what was decided,
why, the evidence at the time, what was rejected, and status. Results live in
`experiments/README.md`; method provenance in `PROVENANCE.md`.

Format: **D-<date>-<n>**. Status: `active` | `superseded by D-…` | `open`.

---

## D-2026-08-14-1 — Test on `jag-standard` with `gpu:titanrtx:1`, allocator capped to 14.5 GiB
- **Decision:** All GPU jobs use partition `jag-standard` (not `jag-important`), a TitanRTX, and
  `torch.cuda.set_per_process_memory_fraction(14.5 / total)`.
- **Why:** User asked not to use `jag-important`. The grader's T4 is Turing SM 7.5; TitanRTX is the
  only Turing card on jag-*, so it takes the same `get_device_capability()[0] >= 8` (TF32) branch
  as the T4 and lacks native bf16 like the T4. It has 24 GB vs the T4's 16 GB, hence the cap.
- **Rejected:** a6000ada (SM 8.9; different TF32 path, bf16 available, would hide T4 failures);
  titanv (12 GB, stricter than the T4, risks false OOM).
- **Status:** active.

## D-2026-08-14-2 — Train on a 100k patient-grouped subsample for all ablations
- **Decision:** Ablation rows train on all 6,528 positives + 93,472 sampled negatives from the
  patient-grouped train split; the full 292,885 split is reserved for the final config.
- **Why:** Full training was ~10 h/run; the encoder is frozen or lightly tuned and the sampler
  draws positives at 5×, so positives bind learning. User chose "subsampled".
- **Rejected:** full data per row (~10 h each, 10 rows); 40k "fast" (directional only).
- **Status:** active.

## D-2026-08-14-3 — Patient-grouped split, natural test prevalence, three views A/B/C
- **Decision:** Split on CODE-15 `patient_id` (1.48 exams/patient); test keeps 2.27% prevalence;
  score view A (full), B (CODE-15 only) and C (organiser-style perturbation of A).
- **Why:** The repo's `train_val_test_sets.csv` leaks 45.5% of test exams' patients into train.
  TPR@5% is capped at 0.05/prevalence, so a rebalanced test set is uninterpretable. B removes the
  SaMi-Trop=positive / PTB-XL=negative source shortcut; C targets the perturbations the
  organisers documented (noise, device filters, resampling; Reyna et al. 2025 §2.2).
- **Rejected:** the repo's record-level split; rebalanced test set; leave-one-source-out (SaMi-Trop
  is all-positive, so it cannot be ranked).
- **Status:** active.

## D-2026-09-21-1 — New `chagas/` package with a thin `team_code.py`; legacy files kept
- **Decision:** Implement the plan as a new package; `fm.py`, `dataloader.py`, `utils.py`,
  `augmentations.py` stay in the tree but nothing on the judged path imports them.
- **Why:** The audit found three hand-synchronised preprocessing copies and a 74 KB `fm.py` with
  four unreachable loss classes; maintainability was an explicit user requirement. Keeping the
  legacy files preserves history and the team's authorship.
- **Rejected:** editing `fm.py` in place (skew risk, unreviewable diff); deleting legacy files.
- **Status:** active.

## D-2026-09-21-2 — Keep the team's core idea; borrow only cited, evidence-backed methods
- **Decision:** ECG-FM backbone + hook-captured 12 layers + static softmax aggregation + MLP head
  + age/sex remain the model. Every added method is one a top-3 team or a peer-reviewed paper
  measured on the *hidden* sets (two-stage FT, soft labels, w_pos=5 + 5× oversampling, FiLM,
  augmentation, model selection on the challenge score).
- **Why:** User instruction ("keep the core idea"; "cite … make clear if it's not mine"). The
  hidden-set gap (median −64% to ELSA-Brasil) is the problem, not in-distribution fit.
- **Rejected:** replacing the backbone with the winner's ST-MEM ViT (would abandon the core idea);
  uncited heuristics.
- **Status:** active.

## D-2026-09-21-3 — Default config = ladder row 5; optional methods behind flags, default off
- **Decision:** FiLM, soft labels, augmentation and two-stage FT are on by default; aux heads,
  ranking loss, GRL and the PMA gate are off until an ablation row justifies them.
- **Why:** The four defaults each have hidden-set evidence from ≥2 sources; the optional four have
  one source each (or none for PMA on this task).
- **Status:** active.

## D-2026-09-21-4 — Departures from cited augmentation recipes
- **Decision:** `crop_shift` randomises the crop *start* instead of prepending zeros;
  `baseline_wander`/`gaussian_noise` amplitudes are relative to record std.
- **Why:** The padding mask requires the valid region to be a prefix; the augmentations run before
  the per-record z-score, so absolute amplitudes from Hong et al. would not transfer.
- **Status:** active (documented in `PROVENANCE.md`).

## D-2026-09-21-5 — Two PLAN §5 tolerances replaced by measured bounds, not loosened silently
- **Decision:** Pad-invariance (item 3) asserts exact token counts and mask behaviour and *reports*
  the pooled-feature delta (0.78 masked vs 1.29 unmasked); resample invariance (item 2) asserts
  RMS < 0.15, corr > 0.99, and |Δ logit| < 0.25 (measured 0.0102).
- **Why:** ECG-FM's conv frontend uses `Fp32GroupNorm` over time, so trailing zeros change valid
  positions; 300 Hz genuinely removes >100 Hz QRS content. Both are backbone properties.
- **Open risk:** padding distortion is a function of record length, which correlates with source.
- **Status:** active.

## D-2026-09-22-1 — Interpret "run both" as ladder rows 2 and 3
- **Decision:** After rows 0/5, the user said "yes run both"; the immediately preceding list named
  "rows 2 and 3" first. Rows 2 (+soft labels/weighting) and 3 (+augmentation) are submitted;
  second-seed repeats and the source-independent soft-label variant are deferred.
- **Why:** Rows 2 and 3 attribute row 5's gain and its widened source separation
  (PTB-XL/CODE-15-negative probability ratio ~5× → ~80×), which is the open question.
- **Status:** active.

## D-2026-09-22-2 — Add per-epoch validation print and per-stage best score to the manifest
- **Decision:** `ChagasModule.on_validation_epoch_end` prints one line per epoch;
  `chagas_manifest.json` gains `training[].best_by_stage` and the full history.
- **Why:** The progress bar is disabled for SLURM logs, so rows 0/5 were only observable through
  checkpoint mtimes, and the manifest did not record which epoch was selected. Logging only; no
  effect on training. Gated by the test suite on SLURM before rows 2/3 start.
- **Status:** active.

## D-2026-09-22-3 — Rows 2/3 run only after the test suite passes (SLURM `afterok`)
- **Decision:** `batch_slurm_tests.sh` is submitted first; rows 2 and 3 depend on it with
  `--dependency=afterok`.
- **Why:** The harness edit above touches the training loop; a failed test cancels the chain
  instead of burning two GPU runs on a broken loop. Mirrors the `afterok` guard that already
  saved a wasted run when the first split job timed out on 2026-08-14.
- **Status:** active.

## Open decisions
- Whether the source-valued soft labels (PTB-XL 0.0 vs CODE-15-neg 0.2) cost anything on the
  hidden cohorts. Row 2 vs row 0 answers the in-distribution part; a variant with a single
  negative target is the proposed follow-up.
- Whether Test C is strong enough to discriminate robustness (both rows moved < 0.01). If rows
  2/3 also show no C−A separation, strengthen C or accept it as a sanity check only.
- Second seeds for rows 0 and 5 (TPR@5% sd ≈ 0.012 at 1,662 positives; the observed +0.023 is ~2σ).

## D-2026-09-25-1 — Rows 2/3 read as inconclusive on attribution; row 1 must run next
- **Decision:** Do not accept or reject the soft-label/weighting change on rows 2/3 alone. Run row 1
  (padding mask + butterworth cleaner + early stopping, nothing else) to attribute the C collapse.
- **Why:** Row 0→1 changes three things and row 1 was never run (D-2026-09-22-1 jumped to rows 2/3).
  Row 2's C fell from 0.408 to 0.351 (gap −0.004 → −0.076), far beyond the ~0.012 noise; the cause is
  either the new cleaner (0.5–100 Hz passes more of the injected noise than neurokit's default) or
  the sharper, source-valued targets. These need different fixes.
- **Status:** open — awaiting user go-ahead for row 1 (~5 h GPU).

## D-2026-09-25-2 — Augmentation judged by Test C and the PTB-XL probability, not by Test A
- **Decision:** Keep augmentation on in the default config despite A −0.020 vs row 2.
- **Why:** It halves the C−A gap and raises PTB-XL mean probability from 0.078 to 0.316, i.e. it
  removes the sampling-rate fingerprint (PTB-XL 500 Hz vs 400 Hz elsewhere). Hidden SaMi-Trop 3 is
  partly 500 Hz; a model that has learned "500 Hz ⇒ negative" would score those records as negatives.
- **Rejected:** dropping augmentation to recover A (optimises the wrong distribution).
- **Status:** active.

## D-2026-09-25-3 — Row 5's source separation (PTB-XL 0.008) is an open risk, not a win
- **Decision:** Before the full-data final, run row 4 (+FiLM only) and a `demographics: none` variant of
  row 5 to find which component re-learns the source cue that augmentation had removed.
- **Why:** Row 3 → 5 adds FiLM and unfreezing; PTB-XL is European with a different age/sex mix, so
  demographics alone could identify it. The hidden cohorts are all Brazilian: a demographic source
  cue learned here transfers as an age prior, not as disease signal.
- **Status:** open.

## D-2026-09-25-4 — "run both" = row 1 + (row 4 and row 5b), three jobs, test-gated
- **Decision:** Submit row 1 (skipped defect-fix step), row 4 (+FiLM only) and a new row 5b
  (row 5 with `demographics: none`), each `afterok` the test suite. Row 5b is added to
  `make_configs.py` so it regenerates with the other rows.
- **Why:** The user's "run both" answered a two-item list whose second item contained two runs.
  Row 1 attributes the Test-C collapse (D-2026-09-25-1); rows 4 and 5b bracket FiLM's role in the
  source separation (D-2026-09-25-3).
- **Status:** active.

## D-2026-09-26-1 — The Test-C collapse is attributed to the defect-fix step, most likely the cleaner
- **Decision:** Treat the 0.5–100 Hz butterworth cleaner (audit D8) as the prime suspect for the C
  collapse (row 1: A 0.442 but C 0.341, gap −0.101; row 0 gap −0.004). Soft labels are cleared
  (row 2 recovers +0.010 C). Prepare three variants to isolate and fix it: row 1 with
  `clean: neurokit` (isolates the cleaner from padding mask/early stop), row 5 with
  `bandpass_high_hz: 45` (fast vectorised cleaner, cutoff from Physiol. Meas. 2026 / Hong et al.
  used 60), and row 5 with `clean: neurokit` (the legacy cleaner, 12× slower at inference).
- **Why:** The 100 Hz cutoff was my choice (to keep high-frequency QRS content); the evidence says it
  passes injected noise into fixed backbone features. A cutoff in the 45–60 Hz range is what the two
  cited Chagas papers used. Padding-mask and early-stop are retained unless row 1b says otherwise.
- **Status:** open — configs prepared; launch awaits user go-ahead (~5 h each).

## D-2026-09-26-2 — FiLM demographics confirmed; row 5's source separation attributed to unfreezing
- **Decision:** Keep FiLM (row 4: +0.027 A, +0.035 C vs row 3). Row 4's PTB-XL mean probability
  (0.243) is far from row 5's (0.008), so the strong source cue comes from stage-2 fine-tuning.
- **Status:** active; row 5b (no demographics) pending to confirm.

## D-2026-09-26-3 — Final config uses `demographics: none`; cleaner variants re-based on row 5b
- **Decision:** Row 5b (no demographics) replaces row 5 as the base for the final config and for the
  cleaner variants 5c/5d (`make_configs.py` updated; `final_full` likewise). The shipped default in
  `chagas/config.py` is NOT changed until the cleaner question is settled.
- **Why:** Row 5b matches row 5 on every view (A +0.003, B +0.006, C +0.005; noise ≈ 0.012) while
  cutting the PTB-XL source separation 9× (0.008 → 0.072). D-2026-09-25-3 said: if FiLM's gain is
  explained by demographics, prefer none. It is. The hidden cohorts are all Brazilian, so a
  demographic PTB-XL identifier can only transfer as an age prior.
- **Rejected:** keeping FiLM for its row-4 gain (a frozen-encoder effect that fine-tuning subsumes).
- **Status:** active; open items: cleaner (5c vs 5d vs 5b), then second seed, then full-data final.

## D-2026-09-26-4 — Batch 1 (rows 1b, 5c, 5d) then one full-data final; winner pre-registered
- **Decision:** User approved "the 3 experiments and one final". Rows 1b/5c/5d are submitted now,
  test-gated. The final (`final_full`, 292,885 records, 2 seeds) is launched by me as soon as batch 1
  reports, with the cleaner chosen by this rule, fixed in advance:
  1. Candidates: 5b (0.5–100 Hz, C 0.430), 5c (0.5–45 Hz), 5d (neurokit).
  2. Pick the highest Test C among candidates whose Test A is within 0.005 of the best A.
  3. Ties within 0.012 (one-seed noise) go to the vectorised butterworth (5b/5c) over neurokit,
     because the grader's 48 h inference limit on ~81k records favours the 12× faster cleaner.
  4. Row 1b is diagnostic only (confirms the cleaner attribution); it does not enter the pick.
- **Why:** Pre-registering avoids choosing the cleaner on a post-hoc reading of three noisy numbers,
  and lets the final start without another round-trip.
- **Status:** active.

## D-2026-09-28-1 — Cleaner = butterworth 0.5–45 Hz (row 5c); shipped defaults flipped; final launched
- **Decision:** Apply D-2026-09-26-4: 5c wins (A 0.437 within 0.005 of 5b's 0.438; C 0.434 highest;
  gap −0.003; B 0.448 best in ladder). `chagas/config.py` defaults now `demographics="none"`,
  `bandpass_high_hz=45.0`, so the judged container (no `CHAGAS_CONFIG`) trains exactly the final
  config. `final_full` builds on row 5c. Final submitted (2 seeds, 292,885 records, 71 h budget each).
- **Correction:** the "neurokit is 12× slower at inference" claim (D-2026-09-26-1) is not supported by
  batch 1 wall times (confounded by co-scheduling; 5d 70 min vs 5b 40 min vs 5c 93 min). Speed was not
  needed to break a tie: 5d lost on Test A outright (0.418).
- **Noise caveat:** 5c vs 5b differences are inside one-seed noise (~0.012). The pick is principled
  (pre-registered) rather than statistically decisive; the two-seed final reports mean ± sd.
- **Status:** active.

## D-2026-09-28-2 — Test gate caught the default flip; final resubmitted
- **What happened:** `test_two_stage_freezing_llrd_and_dropout_modes` asserted FiLM parameters are
  trainable under `default_config()`; the shipped default became `demographics="none"` in
  D-2026-09-28-1, so the gate failed (84/85) and the dependent final (17641239) was held
  (`DependencyNeverSatisfied`) and cancelled.
- **Decision:** Pin `cfg.model.demographics = "film"` inside that test. Its subject is the
  freezing / LLRD / dropout mechanics, which are independent of the shipped default; the FiLM
  path still needs coverage even though it is off by default. No assertion was weakened.
- **Status:** active. Final resubmitted behind a fresh test gate.

## D-2026-09-29-1 — Full-data final rejected as shipped; stage-2 schedule is scale-dependent
- **Finding:** `final_full` (2 seeds) scores TPR@5% 0.427 ± 0.011 on A (≈ ladder 5c) but AUROC 0.791 vs 0.842 on
  the same test set and 0.792 vs 0.847 on source-free view B; PTB-XL cue returns (0.02).
- **Cause:** in every two-stage run stage 2 peaks at its first validation and then drifts. Validation/checkpointing
  is per epoch, so the first checkpoint arrives after ~1.4k optimiser steps on 100k but ~4.6k on 292k — past the
  peak. The shipped procedure's quality therefore depends on training-set size, and the grader trains on the full set.
- **Decision:** do not submit the current default. Candidate fixes (need user go-ahead, full-data runs):
  (A) keep all positives, cap negatives at ~94k inside `train_model` — reproduces the validated 5c regime exactly;
  (B) make stage 2 step-based: validate every ~500 optimiser steps and cap stage 2 at a fixed step count, so
      checkpoint timing is independent of data size;
  (C) lower stage-2 LR toward McKeen et al. 2025's full-FT rate (1e-6) to widen the useful window.
  Recommendation: A and B in parallel, one seed each on full data; A is the safe fallback, B tests whether the
  extra negatives help once the peak is caught. C only if B is unstable.
- **Status:** open — awaiting user decision.

## D-2026-09-29-2 — Run final candidates A and B on the full training split (user approved "do both")
- **A (`finalA_negcap`):** config-only. `data.max_records = 100000` uses the existing `_subsample()` (all
  positives kept, negatives thinned, seeded) inside `train_model`, so a full-data retrain reproduces the
  validated 100k ladder regime (5c: A 0.437 / AUROC 0.842). No code change.
- **B (`finalB_stepval`):** new `TrainCfg.stage2_val_every_steps` (validate every N optimiser steps;
  Lightning's `val_check_interval` is converted to training batches) and `stage2_max_steps`, both default 0
  = legacy per-epoch behaviour, so every earlier config and result is unchanged (existing YAMLs only gain
  the two zero-valued fields). B uses 500 / 6000 with early-stop patience 3 (stops ≤1.5k steps past the
  last improvement); the ladder peak sat at ~1.4k stage-2 steps.
- **Tests added:** trainer wiring for both stages and both settings; `_subsample` keeps every positive,
  is seeded-deterministic, and is a no-op at 0 or above size. Gate = full suite on SLURM.
- **Both** train on `fixbranch_test/data_train_full` (292,885 records), seed 42, evaluated on A/B/C.
- **Pick rule (pre-registered):** B replaces A only if its Test A AUROC ≥ A's − 0.005 AND its TPR@5% on
  A and C are within noise (0.012) of A's or better; otherwise ship A. Rationale: A reproduces a measured
  result; B must earn the extra complexity.
- **Status:** active.

## D-2026-09-29-3 — Ship candidate A (negative cap); B rejected by the pre-registered rule
- **Result:** A: TPR@5% 0.427 / AUROC 0.835 on A, C 0.414. B: 0.440 / 0.789, C 0.439.
- **Decision:** Ship A. Rule (D-2026-09-29-2) required B's AUROC ≥ A's − 0.005; B is 0.046 lower. The shipped default
  in `chagas/config.py` is now `data.max_records = 100000` and was verified identical to `finalA_negcap.yaml` apart from
  name/description; historical ladder configs pin `max_records = 0` so they still describe what ran.
- **The tension, recorded honestly:** B wins the challenge metric (+0.013 on A ≈ 1σ, +0.025 on C ≈ 2σ) and has the
  smallest C−A gap. The rule was fixed in advance precisely so this would not be decided post hoc. B's AUROC matches the
  rejected full final (0.791), showing full-data training degrades global ranking regardless of checkpoint timing, and B
  has the study's strongest source cue (PTB-XL 0.007) — both risk factors on unseen cohorts.
- **Known caveat:** the cap is an absolute record count. On our 292k folder it gives 6.5% prevalence (validated). The
  grader trains on the full public set (≈366k records, ≈8,190 positives), where the same cap gives ≈8.2%. Untested but
  a modest shift; a ratio-based cap would be size-invariant and is the natural follow-up.
- **Status:** active. Remaining before submission: Docker build-and-run verification (not possible on the cluster).

## D-2026-09-29-4 — Stage-2 learning-rate sweep on the shipped config, 3 seeds per rate
- **Trigger:** user asked "can you not test the stage 2 learning rate sweep" after I proposed it; read as
  "can't you test it?" and run (stated to the user, with an offer to cancel).
- **Hypothesis:** stage 2 barely helps (val AUROC ~0.81 → ~0.83, peaking at its first check and then drifting)
  versus the winner's +29% from full fine-tuning. 2e-5 came from the winner's different backbone; ECG-FM's
  authors fine-tuned at 1e-6 (McKeen et al. 2025).
- **Design:** `finalA_negcap` (the shipped default) with only `train.stage2_lr` ∈ {2e-5, 5e-6, 1e-6} and
  seed ∈ {42, 43, 44} changed (verified structurally). 2e-5/s42 = the existing finalA result, so 8 new runs,
  one SLURM array (`batch_lr_sweep.sh`, at most 4 concurrent), full folder with the shipped negative cap.
- **Pre-registered rule:** per LR take 3-seed means of Test-A TPR@5%, Test-A AUROC and C−A. Adopt a lower LR only
  if (1) its mean TPR@5% beats 2e-5's by more than the SE of the difference (≈ 0.011·√(2/3) ≈ 0.009), (2) its mean
  AUROC ≥ 2e-5's − 0.005, and (3) its mean C−A is not worse by > 0.012. If both qualify, take the higher mean
  TPR@5%. Otherwise keep 2e-5. Diagnostics reported regardless: stage-2 best epoch and stage-2 val gain over stage 1.
- **Status:** active.

## D-2026-09-29-5 — Persist training loss, learning rate and validation metrics to CSV
- **Decision:** `_make_trainer` attaches a Lightning `CSVLogger` and a `LearningRateMonitor` (per step) when
  `train.csv_log` (default true) and a log dir are given. Output: `model_folder/metrics/member<k>/stage<n>/metrics.csv`
  with `train_loss_step`, `train_loss_epoch`, one `lr-*` column per param group (so LLRD is visible), and all
  `val_*` metrics. Logged every `train.log_every_n_steps` (default 50) optimiser steps.
- **Why:** previously the trainer ran with `logger=False`, so loss and LR were computed and discarded; only
  validation metrics survived (manifest history). User asked for per-epoch loss/LR logging.
- **Placement:** inside model_folder (grader rules), outside `_work` (which is deleted after training). Test asserts
  no fallback `lightning_logs/` directory is created.
- **Process:** built on side branch `improve/csv-logger` (worktree `improve_csvlog_wt`) so the running LR sweep
  (array 17656806, which imports code from `improve_wt` as each task starts) stays on one code version. Tests:
  89/89 on SLURM job 17657002 (CPU-only; GPUs were occupied by the sweep). Merge into `improve/robust-finetune`
  after the sweep completes.
- **Status:** active; merge pending.
