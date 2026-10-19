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

## D-2026-09-30-1 — LR sweep: keep stage2_lr = 2e-5; my hypothesis was wrong
- **Result (3 seeds each):** 2e-5: 0.436 ± 0.010; 5e-6: 0.423 ± 0.002; 1e-6: 0.404 ± 0.004 (Test-A TPR@5%).
  AUROC flat (0.829 / 0.834 / 0.830). C−A: −0.007 / −0.001 / +0.001.
- **Decision:** keep 2e-5 per the pre-registered rule (lower rates had to win by > 0.009; they lose by 0.013 and 0.032).
  No config change.
- **What the sweep did and did not show:** predicted — the stage-2 peak moves later at lower rates (0.7 → 1.7 → 5.7
  epochs). Not predicted — stage 2's validation gain is ~+0.02 at every rate and the challenge score drops. So "stage 2
  barely helps" is real but is not caused by the learning rate. Remaining candidates: the 10 s input vs ECG-FM's 5 s
  pretraining segments, the frozen conv frontend, or a label-noise ceiling.
- **Side findings:** lower LR → weaker PTB-XL source cue (0.029 → 0.068 → 0.113) and ~5× lower seed variance. A
  trade-off, not adopted: it costs the metric.
- **By-product:** the shipped config's estimate is now 3-seed: A 0.436 ± 0.010 (the earlier single-seed 0.427 was the
  low draw), AUROC 0.829, B 0.438, C 0.429.
- **Merged** `improve/csv-logger` (D-2026-09-29-5) into this branch now that the sweep has finished.
- **Status:** active.

## D-2026-09-30-2 — Record length is a label shortcut in CODE-15; round 1's gains are mostly not real
- **Finding:** CODE-15 prevalence 3.05% (7.34 s records) vs 0.24% (10.24 s), 12.7×; duration alone gives AUROC 0.681 on
  view B; every model (incl. row 0) scores CODE-15 negatives higher when they are short (shipped: 0.62 vs 0.39). Hidden
  cohorts are all ~10 s. Within the 7.3 s group: row 0 = 0.377 / 0.807, shipped (3 seeds) = 0.384 / 0.787,
  shipped ensemble = 0.404 / 0.801 (TPR@5% / AUROC; SE ±0.0125 / ±0.007).
- **Consequences:** (1) view B was mislabelled "shortcut removed" — only the source cue was. (2) The round-1 ladder's
  single-model gain over the original is inside the noise once length is controlled. (3) D-2026-09-29-3 (ship candidate
  A) stands only as "no worse than the original"; it is not a demonstrated improvement.
- **Decision:** round 2 per `PLAN_R2.md`: within-length views (S7 primary, S10/S10+ secondary), length-matched
  negative sampling, fixed 5 s windows (ECG-FM's native input), WiSE-FT, seed ensemble; pre-registered rule in §5.
  New features default off; the shipped default does not move until the rule says so.
- **Also from the literature review (8 CinC 2025 papers):** internal scores moved opposite to hidden scores in five
  papers; a frozen encoder beat a fine-tuned one on hidden data for rank 7 and for the team's own entry; ELR was never
  ablated; heterogeneous ensembles with a weak member hurt; nobody reported the length–label association.
- **Status:** active.

## D-2026-10-01-1 — Round 2 implemented behind flags; shipped default unchanged
- **Decision:** every PLAN_R2.md §3 change is a config flag whose default is round-1 behaviour:
  `data.match_negatives_on: none`, `preprocess.window_mode: single` (+ `max_windows: 4`), `train.wise_ft: off`
  (+ `wise_ft_alphas`, `wise_ft_save_all`). `tests/test_config_r2.py` asserts `default_config()` ==
  `configs/finalA_negcap.yaml` (name/description aside), that each flag defaults to round 1, that every committed
  config equals `make_configs.py` output, and that the 9 r2 configs differ from finalA only in their flags (+ seed).
  `window_mode: single` is asserted bit-exact against a verbatim copy of the round-1 pipeline (train and eval, with
  augmentation), `match_negatives_on: none` against a verbatim copy of round-1 `_subsample`.
- **Historical configs:** regenerating added only the 9 new default-valued lines to each of the 19 YAMLs; nothing
  needed pinning. The 8 `configs/lrsweep/*.yaml` were hand-made in 74512a6 (not by `make_configs.py`); they are left
  untouched as the record of what ran (they also lack the later `csv_log` fields).
- **Status:** active.

## D-2026-10-01-2 — Round-2 implementation choices the plan did not fix
- **Durations** are read from all 366,185 headers (not derived) and cross-checked in full against
  `signal_lengths.csv` (SLURM 17682272: 0 unreadable, 0 missing, 0 mismatches). The 9 MB CSV is committed.
- **Matched sampling:** groups short/mid/long plus `unknown` (unparsable header). Quotas by largest remainder; a
  group short of negatives is taken whole and the shortfall is redistributed by the positives' proportions; groups
  with no positives receive negatives only when all others are exhausted.
- **Observation, not acted on (plan fixes the design):** matching is marginal over all sources, as specified. On the
  train split it leaves a residual length cue *within* CODE-15: CODE-15 positives are 95% short, SaMi-Trop positives
  ~51% long, PTB-XL negatives all long, so the matched negatives are ~85% short overall and CODE-15 prevalence among
  kept records becomes ~5.9% (short) vs ~2.1% (long), a 2.8x ratio instead of 12.7x (my arithmetic from the split
  counts, not measured on a run). Matching within source (CODE-15 negatives to CODE-15 positives) would remove it.
  The length-cue gap of M1/MW will show whether this matters; a `"source_duration"` variant is the natural follow-up.
- **Windows:** `K = min(4, ceil(T/5 s))` exactly as written, so 10.24 s CODE-15 records get 3 windows (7.34 s and
  10.0 s get 2). Window logits (and aux/domain logits) are averaged per record; ensemble members are still averaged
  on probabilities. In multi mode `crop_shift` is skipped without drawing from the RNG. Validation uses the same
  multi-window path as inference.
- **WiSE-FT:** ties in validation challenge go to the smaller alpha (towards the frozen encoder, the more robust end
  in the cited evidence); integer/bool buffers come from the nearer endpoint (alpha >= 0.5 -> stage 2); alpha 0/1 are
  copies, not arithmetic. Each alpha's threshold is calibrated in its own validation pass, so the extra final
  calibration pass is skipped when WiSE-FT runs. WiSE passes are not counted against the time budget (3 validations).
  Per-alpha folders (`model/wise_ft/member<k>/alpha<a>/`) are written only with `wise_ft_save_all` (experiments);
  the selected alpha is stored in the checkpoint (`chagas.wise_ft_alpha`) and the manifest.
- **Evaluation:** bootstrap = plain resampling of test records (not label-stratified), 1,000 resamples, seed fixed
  per view, computed with integer weights on pre-sorted scores (tested equal to explicit resampling). The
  duration-only baseline is the indicator "record < 7.5 s". S-views are also reported on Test C (`S7·C` etc.).
  Within-length *validation* metrics use CODE-15 records only (source id 0, which also covers `unknown`).
- **Batched inference:** `predict_batched.py` writes `stem, record, binary, prob`; `eval_report.py --preds-a/-c`
  reads it. For MW the shipped (selected) model is not re-inferred: the selected alpha folder's predictions are
  copied to the run's top level; the smoke asserts the two are identical.
- **RESCORED.md** lives under `experiments/results/`, which `.gitignore` excludes; it and its CSVs are force-added.
- **Status:** active.

## D-2026-10-01-3 — run_info.json is written by the experiment runners, counts by training
- **Decision:** `experiments/run_info.py` writes `results/<run>/run_info.json`, called at start and end by
  `run_ladder.sh` and `run_r2.sh`. It records git commit + dirty flag (tracked files, ignoring `experiments/logs`
  and `experiments/results`), config path + SHA-256, seed, host, SLURM ids, GPU, python/torch/lightning/numpy
  versions, SHA-256 of `pip freeze`, start/end times. The train/val counts by source x label x length group are
  computed inside `chagas.training` (`data_summary`) and written to `chagas_manifest.json`, which run_info copies.
- **Why:** the judged container must not shell out to `git`/`pip`, and only the training entry knows the split.
- **Status:** active.

## D-2026-10-01-4 — Acceptance: the duration-only 0.681 is an all-CODE-15 number, not a view-B number
- **Finding:** with the tools (`experiments/acceptance_r2.py`, SLURM 17682392) every hand-computed target of
  2026-09-30 reproduces to its last digit except the duration-only AUROC "on view B": the score "1 for records
  < 7.5 s" gives **0.6767** on view B (CODE-15 test records, 68,575) but **0.6805** on all 342,755 CODE-15 records and
  0.6815 on the training split. No other simple definition gives 0.681 on view B (−duration 0.671, per-exact-length
  training prevalence 0.680, ordinal short/mid/long 0.676). So the 0.681 in PLAN_R2.md §1 was computed on all CODE-15
  records; the conclusion (length alone is a strong cue) is unchanged.
- **Decision:** keep the definition (indicator of < 7.5 s, scored on the view); report 0.677 for view B and note the
  discrepancy. The acceptance script prints both lines.
- **Status:** active.

## D-2026-10-01-5 — Round-2 verdict: the pre-registered rule adopts nothing; 10 s-format gain flagged for the user
- **Result (3 seeds; paired bootstrap vs B0):** S7 TPR@5%: M1 −0.002 ± 0.005, W1 −0.004 ± 0.004, MW(α=1) −0.009 ± 0.005.
  S7 AUROC: W1 +0.008 ± 0.002, MW +0.008 ± 0.003 (bar was 0.010). S10+ AUROC: M1 +0.037 ± 0.009, W1 +0.056 ± 0.010,
  MW +0.062 ± 0.011. Length-cue gap: 0.170 → 0.036 (M1), 0.097 (W1), 0.009 (MW). Pooled A falls for M1/MW (−0.035/−0.033),
  as predicted. Frozen (α=0) is clearly worse than fine-tuned (S7 0.343 vs 0.375).
- **Decision:** by PLAN_R2 §5, applied mechanically, no configuration replaces B0. The shipped default is unchanged.
- **Flag (not acted on):** the rule made S10+ a guard, not a criterion. The hidden cohorts are all ~10 s recordings, and on
  that format W1 and MW improve AUROC by 4–5.5 SE while S7 moves by ≤ 2 SE. Adopting W1 or MW would be a post-hoc
  departure from the rule, so it is the user's call, not mine. If chosen: MW(α=1) as a 3-seed ensemble is the candidate that
  both removes the cue and scores best on S10/S10+; W1 is the lower-risk option (pooled A −0.008 only).
- **Infrastructure note:** array task 6 (MW seed 42) hit NODE_FAIL three times on jagupard27 and completed on its fourth
  attempt on jagupard26; the table job was also held after a failed launch and re-run with `--exclude=jagupard27`.
- **Status:** active; adoption question open.

## D-2026-10-01-6 — Manuscript: lead with the record-length shortcut, generate every number from saved results
- **Request:** a LaTeX/PDF write-up of the work so far in the form expected by the *Physiological Measurement* focus
  collection "Detection of Neglected Tropical Diseases from the Cardiac Signals" (IOP; deadline 31 Dec 2026).
- **Decision (framing):** the paper's claim is the record-length label shortcut in CODE-15% and its effect on
  evaluation, not a better model. The evidence supports that claim (prevalence 3.05% vs 0.24%; duration-only AUROC
  0.677; pooled gain 0.412 → 0.436 vanishing within length groups) and does not support "our model improved": the
  within-length single-model score is unchanged and the pre-registered round-2 rule adopted nothing. Negative results
  (lower stage-2 LR, all negatives, WiSE-FT) are reported as such. The 10 s-format AUROC gain is stated as a
  hypothesis that the hidden cohorts would have to confirm.
- **Decision (format):** IOP `iopjournal` class, structured abstract (Objective / Approach / Main results /
  Significance, 245 words; limit 250), author-year references, about 4,200 words of body text (limit 8,000).
- **Decision (reproducibility):** `main.tex` holds no hand-typed result. `paper/make_assets.py` writes `numbers.tex`,
  `tables/*.tex` and `figures/*.pdf` from `experiments/results`; `bash paper/build.sh` rebuilds the PDF without
  running a model. Built with tectonic because the cluster has no TeX installation.
- **Not mine to decide (left as `TODO(authors)`):** author list and order (placeholder: the CinC 2025 team), funding,
  CRediT roles, repository URL, and the wording of the generative-AI disclosure in the acknowledgements.
- **Corrections made while writing:** W1 S7 TPR@5% standard error is 0.004 (0.0045), not 0.005; M1 S10+ AUROC
  difference is +0.037 (0.0375), not +0.038. README and D-2026-10-01-5 updated to match the table.
- **Status:** draft; needs author review before submission.

## D-2026-10-15-1 — Round 3: fixed validation protocol (pre-cap, seed-independent, length-stratified, S7 monitor)
- **Finding:** the validation set was carved after the negative cap and shuffled with `train.seed`, so each seed
  validated on different records at 6.5% prevalence (test and hidden data are at natural prevalence), validation was
  length-matched whenever training was, checkpoints were selected on pooled `val_challenge` (inflated by the length
  cue, D-2026-09-30-2), and PTB-XL was grouped per record at training time but per patient in the held-out split.
- **Decision:** `train.val_protocol: fixed` for round 3 and the entry: split before the cap, `val_seed` 12345
  independent of the training seed, strata source × label × length group, PTB-XL grouped by patient, monitor
  `val_challenge_s7` with S7/S10+ metrics logged. Legacy remains the default so rounds 1–2 reproduce.
- **Why:** seeds should differ only in initialisation and sampling; ensemble members should be selected on the same
  data; checkpoint selection must not reward the shortcut. See PLAN_R3.md §2.
- **Not changed:** the held-out test split (20% of patient groups, source × label strata, seed 12345). 31 runs are
  scored on it and a re-split cannot create more 10 s positives (317 in CODE-15). Re-audited (SPLIT_AUDIT.md).
- **Status:** active.

## D-2026-10-15-2 — Round 3: S10+ AUROC becomes the primary metric, S7 TPR@5% the guard
- **Decision:** PLAN_R3.md §5. The entry is to be scored on hidden cohorts that are all ~10 s recordings, so the 10 s
  format is the primary view; S7 guards against losing within-format performance. Round 2 had S7 primary and S10+ as
  a guard (PLAN_R2 §5), which adopted nothing although W1 gained +0.056 ± 0.010 S10+ AUROC.
- **Why recorded:** changing the primary metric after seeing round-2 results is exactly the kind of move a
  pre-registered rule exists to prevent. It is made before round 3 is launched, applies to new runs under a new
  validation protocol with a re-run baseline, and the round-2 verdict stands as written (D-2026-10-01-5).
- **Caveat carried:** S10+ includes SaMi-Trop positives, so a source cue can contribute; the per-source
  mean-probability diagnostic is reported alongside and W1S is judged on it.
- **Status:** active.

## D-2026-10-15-3 — Prevalence-based negative cap for the entry
- **Finding:** `data.max_records: 100000` is an absolute count. On our 292,885-record training split it gives 6.5%
  training prevalence; on the organisers' full set (~366k records) it would give 8.2%, a setting never validated.
- **Decision:** `data.cap_mode: prevalence`, `cap_prevalence: 0.065` for all round-3 runs and the entry; `count` stays
  the default so committed configs reproduce. The negative budget is n_pos·(1−p)/p, drawn uniformly or length-matched
  exactly as before.
- **Status:** active.

## D-2026-10-15-4 — Round 3 matrix: B0, W1 (5 seeds), W1R, W1S, W1-TTA
- **Decision:** 14 training runs (PLAN_R3 §3–4). Chosen for expected gain per GPU-hour on the 10 s format: the ranking
  loss acts on the ranking the score measures and already exists; PTB-XL removal tests the remaining source cue with
  no new code; two extra W1 seeds settle the ensemble size; overlapping-window inference costs no training.
- **Deferred:** stage-2 regularisation (lower LLRD factor, frozen bottom layers), distillation, ELR, domain heads.
- **Infrastructure:** jagupard27 is healthy today (state allocated, no reason); it is not excluded. SLURM requeues on
  NODE_FAIL as it did in round 2.
- **Status:** active; outcome to be recorded after `aggregate.py --rule r3`.

## D-2026-10-15-5 — Round-3 protocol implemented behind flags; implementation choices
- **Flags (all default to round-2 behaviour; `default_config()` still equals `finalA_negcap.yaml`):**
  `train.val_protocol: legacy|fixed` (+ `train.val_seed: 12345`), `train.monitor` now validated against
  `config.MONITOR_METRICS` (adds `val_challenge_s7`, `val_auroc_s7`, `val_auroc_s10p`), `data.cap_mode: count|prevalence`
  (+ `cap_prevalence: 0.065`), `data.exclude_sources: []`, `data.ptbxl_database_csv: null`,
  `preprocess.window_stride_seconds: null`. Regenerating added only these 7 default-valued lines to the 29 generated YAMLs.
- **Fixed split:** scan -> split (from all scanned records, `val_seed`) -> cap the training half only. Strata are
  (source, any-positive, length group of the group's first record); each stratum shuffles with its own generator seeded by
  `(val_seed, crc32(stratum))`, so excluding a source (W1S) removes exactly that source's records from validation and moves
  nothing else. Legacy keeps the shared generator and the old key (pinned partition in `tests/test_val_protocol.py`).
- **PTB-XL patient groups are gated to `fixed`.** Under legacy the strata are shuffled in sorted order by one generator;
  PTB-XL's stratum comes before SaMi-Trop's, and `Generator.shuffle` of n items consumes n-1 draws, so grouping PTB-XL by
  patient (17,400 train records -> 15,095 patients) would also change which SaMi-Trop positives are validated. The table is
  not in `data_train_full` (WFDB files only), so r3 configs point `data.ptbxl_database_csv` at
  `physionet2025/training_data/ptb-xl-.../ptbxl_database.csv`; with no path, `ptbxl_database.csv` is looked for at the top
  of the data folder and one level down. Key = stem without `_hr` and leading zeros, as `build_split.py`.
- **Prevalence cap** ignores `max_records`; negatives = `round(n_pos (1-p)/p)`, drawn as `match_negatives_on` says; if
  fewer exist, all are kept.
- **Per-view validation metrics** mirror `experiments/scoring.view_mask` in `chagas.model.view_mask` (scoring imports pandas);
  equality is tested. They are NaN (logged) / `null` (manifest) when a view lacks a class. `resolve_monitor` checks the
  validation entries before training: an unscorable view falls back to `val_challenge` with a `RuntimeWarning`, and the
  manifest's new `split` block records the monitor actually used. `best_by_stage` is chosen by that monitor. WiSE-FT alpha
  selection still uses `val_challenge` (unchanged; no r3 config uses WiSE-FT).
- **Window stride lives in `preprocess`, not `model`** (all window settings and the only consumer, `preprocess_array`, are
  there). Strided starts every `stride` from 0 plus a right-aligned last window; more than `max_windows` are thinned to evenly
  spaced ones keeping the first and last. Test-time use is `predict_batched.py --window-stride` (not an environment variable
  in `load_model`), so the judged path reads nothing new. `run_r3.sh` writes `predictions_{A,C}_tta.csv`, `report_tta.md`,
  `report_tta_*.csv`; `aggregate.py` reads them with the run spec `run:tta` (per-run suffix rather than a global
  `--predictions-suffix` flag, so one table can mix normal and TTA groups).
- **`aggregate.py --rule r3`:** primary = S10+ AUROC delta > 1 SE with S7 TPR@5% delta >= -1 SE; secondary = S7 TPR@5% delta
  > 1 SE with S10+ AUROC delta >= -1 SE; `--rule r2` stays the default.
- **Status:** implemented and unit-tested; round-3 smoke and array not yet run.

## D-2026-10-16-1 — Round-3 verdict: W1R (5 s windows + hard-negative ranking loss), 3-seed ensemble, is the entry
- **Result (fixed validation protocol; 3 seeds; paired bootstrap; array 17885830, tables 17885831 / 17899069):**
  vs B0 — S10+ AUROC: W1 +0.043 ± 0.010, W1R +0.046 ± 0.012, W1S +0.038 ± 0.011, W1-TTA +0.041 ± 0.010; S7 TPR@5%:
  W1 −0.002 ± 0.005, W1R +0.004 ± 0.005, W1S −0.001 ± 0.005. All four pass the primary criterion against B0.
  vs W1 — S10+ AUROC: W1R +0.003 ± 0.007, W1S −0.005 ± 0.008, TTA −0.001 ± 0.001 (none > 1 SE); S7 TPR@5%: W1R
  +0.006 ± 0.005 (> 1 SE, S10+ not lower → secondary criterion), W1S +0.001 ± 0.004, TTA 0. Ensembles: W1R S7 0.410
  (+0.021 ± 0.007 vs W1 ensemble), S10+ AUROC 0.841; 5-seed W1 ensemble vs 3-seed: S10+ +0.003 ± 0.003 (not > SE).
  Length-cue gap: B0 +0.15, W1 +0.08, W1R +0.01, W1S +0.07. Source gap (SaMi-Trop − CODE-15 positives): B0 −0.04,
  W1 −0.02, W1S −0.01 (did not grow).
- **Decision (PLAN_R3 §5 applied as written):** candidate = **W1R**; ensemble size **3**; TTA **not** adopted (no gain);
  W1S not adopted. The entry is `final_r3`: shipped config + fixed validation protocol + prevalence cap 0.065 +
  S7 monitor + 5 s multi-window + ranking loss (Hong et al. 2025; 2.5 / 0.5 / top 20%) + `ensemble_members: 3`.
- **Caveat recorded:** W1R beat W1 on S7 by 1.2 SE, the weakest margin that satisfies the rule. Its stronger evidence is
  indirect: it removed the residual length-cue gap (+0.08 → +0.01) and gave the best ensemble on both views. W1
  remains the fallback if the entry's time-budget check fails.
- **Protocol effect:** B0 under the fixed protocol scored S7 0.386 / S10+ 0.781 vs 0.384 / 0.779 in round 2 — unchanged.
- **Status:** active; time-budget check and container check pending.

## D-2026-10-16-2 — Entry checks pass: time budget, container, end-to-end
- **Time budget (SLURM 17902758, Titan RTX, 14.5 GiB cap, `data_train_full` 292,885 records):** the unmodified
  `train_model.py` with the default config trained the three members in 14,896 s (4.1 h; 1.4 h per member), and
  inference over 2 × 73,300 records took 3,139 s. At a 2.5× T4 slowdown the training projection is ~10.3 h against the
  organisers' 72 h, a margin far beyond the 30% required (PLAN_R3 §9.2). The TimeLimitCallback split (71 h / 3 members)
  is never reached.
- **Entry scores (3-member ensemble from one call):** S7 TPR@5% 0.416 / AUROC 0.808; S10+ AUROC 0.854; S10 AUROC 0.765;
  pooled A 0.454 / 0.845 — consistent with the round-3 W1R seed ensemble (0.410 / 0.841 on S7 / S10+).
- **Container (17902785 / 17902786):** the Apptainer rendering of the unmodified Dockerfile builds (22 min, rootless) and
  `train_model.py` → `run_model.py` → `evaluate_model.py` run inside it on the 400-record set with the default config
  (400/400 outputs). The organisers' Docker build itself is not exercised here (no daemon on the cluster); the
  rendering follows the Dockerfile step for step.
- **Decision:** `final_r3` (commit 3103ff9) is the entry to submit for external scoring; no further change before that.
- **Status:** active.

## D-2026-10-16-3 — Manuscript revised after an independent review; PTB-XL grouping difference in the entry is deliberate
- **Review (Fable, read-only) and rewrite (Opus):** 15 required and 4 optional changes applied: full round-3 rule stated in
  Methods (W1R qualified by its second clause), S10-only round-3 gain reported (+0.084 ± 0.027) against the SaMi-Trop
  source-cue objection, why W1 rather than MW was carried in and that the entry has no length-matched sampling, the
  held-out split named as the selection set of three rounds, "pre-registered" replaced by "fixed in writing before the
  runs", rounds named, original vs post-Challenge entry disambiguated, T4 time stated as a projection with its factor,
  source-cue diagnostics defined in Methods. Abstract 250 words; 13 pages.
- **Entry grouping:** the organisers' data folder has no `ptbxl_database.csv`, so the entry groups PTB-XL records
  individually (validation 29,273 records) while the round-3 runs grouped them by patient (29,250). All PTB-XL records are
  negative; the paper states the difference. Not a defect; no re-run.
- **Status:** active; `TODO(authors)` items remain (authors, funding, CRediT, AI disclosure, three literature numbers).

## D-2026-10-19-1 — Submission package frozen: tag `r3-submission`
- **Audits:** two independent read-only audits (runtime conformance, Dockerfile) found no FAIL (SUBMISSION_AUDIT.md). Their
  WARNs were fixed in 3213fa1: checkpoint download falls back to the Hugging Face mirror and is sha256-checked, pytorch-
  lightning pinned to 2.5.2 (the version every result used), pip cache off, `.dockerignore` patterns match at any depth,
  offline env vars, `run_model -v` logs records that cannot be preprocessed. Left as is: unused `exclude_label_issues`
  option, legacy modules and two unused CSVs in the tree (not imported).
- **Re-verification after the edits:** CPU tests 17967831 (174 passed); Apptainer rebuild 17967832 (BUILD OK, sha256 check
  executed); in-container entry run 17967833 (400/400, RUN OK).
- **Decision:** the commit tagged `r3-submission` is what goes to the organisers; no further change to the judged path
  without re-running the three checks above. Not exercised: the organisers' own `docker build` (no daemon reachable).
- **Status:** active.
