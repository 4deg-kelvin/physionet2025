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
