# External (organiser-scored) results

Scores computed by the Challenge organisers on their hidden validation and test cohorts. Source rows in
`EXTERNAL_SCORES.csv`. The post-Challenge entry is commit 355ff92 (tag `r3-submission` at the time of submission;
docs-only commits followed), focus-issue submission F28, scored 2026-10-19; it was the first of two focus-issue entries.

| Cohort | Original entry (official phase, 2025) | Post-Challenge entry (final_r3) | Δ |
|---|---|---|---|
| Hidden validation (REDS-II subset) | 0.323 | **0.477** (AUROC 0.794) | +0.154 |
| Test: REDS-II | 0.329 | **0.371** (AUROC 0.745) | +0.042 |
| Test: SaMi-Trop 3 | 0.281 | **0.371** (AUROC 0.780) | +0.090 |
| Test: ELSA-Brasil | 0.125 | 0.125 (AUROC 0.582) | 0.000 |
| **Test mean (ranking metric)** | **0.245** (rank 8/40) | **0.289** | **+0.044** |

Context: the official 2025 leaderboard top was 0.323 (Biomed-Cardio: 0.468 / 0.376 / 0.125), second 0.283, median 0.217;
0.289 would sit between first and second. ELSA-Brasil is 0.125 for the winner, for our original entry and for the
post-Challenge entry alike, which suggests the score on that cohort is quantised by a very small number of positives and
is not where entries separate. Internal held-out scores of the same post-Challenge entry: 0.416 within 7.34 s CODE-15%
records (AUROC 0.808), AUROC 0.854 on the 10 s format, 0.454 pooled — the pooled internal number remains an optimistic
selection-set estimate, as the paper's Limitations say.

What changed between the two entries (paper sections 2.3–2.7): the encoder is fine-tuned in a second stage instead of
frozen; soft labels, class weighting and augmentation; demographics removed; 0.5–45 Hz band-pass; negatives capped by
prevalence; fixed 5 s input windows (removing the record-length shortcut); hard-negative ranking loss; three-seed ensemble;
validation carved before the cap and scored within the shortcut-free format.
