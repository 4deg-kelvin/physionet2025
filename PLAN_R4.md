# Round 4 plan: what to do with the last hidden-test evaluation

Author: Claude (2026-11-05), at the user's request to "see if there are any other tests we should run as we have
one more evaluation on the hidden test set". Follows PLAN_R3.md and DECISIONS.md D-2026-10-30-1.

## 1. Situation

The post-Challenge entry (W1R: 5 s windows + hard-negative ranking loss, three seeds, fixed validation protocol,
commit 355ff92) was scored by the organisers: test mean 0.289 (REDS-II 0.371, SaMi-Trop 3 0.371, ELSA-Brasil
0.125) against 0.245 for the original entry. One focus-issue entry remains. Two things it could buy:

- **A better score**, if a configuration can be shown internally to beat the scored entry on the views closest to
  the hidden format by more than the noise.
- **Attribution**, if it differs from the scored entry in one change, so that the paper can say which change moved
  the hidden score (Discussion 4.3 says this is what is missing).

ELSA-Brasil is 0.125 for the winner and for both of our entries; TPR@5% is quantised there and no candidate is
expected to move it. Gains, if any, come from REDS-II and SaMi-Trop 3.

## 2. Candidates and why

All derived from W1R under the round-3 protocol, three seeds unless stated; scored on A, B, S7, S10, S10+, C.

| Key | Change vs W1R | Why | Cost |
|---|---|---|---|
| W1R×5 | two more seeds (45, 46) → five-member W1R | the ensemble is the most reliable gain we have; five W1 seeds gained +0.003 ± 0.003 on S10+, not tested for W1R | 2 runs |
| MWR | + length-matched negatives | the only round-2 remedy never combined with the ranking loss; in round 2 MW had the best S10 AUROC (0.773) | 3 runs |
| W1RS | − PTB-XL | W1S had the best S10+ ensemble TPR@5% (0.483) and a smaller source gap; combined with the ranking loss untested | 3 runs |
| mixes | ensembles across existing W1R / W1S / W1 members | diversity at zero training cost; scored from saved predictions (`batch_r3_mix.sh`) | 0 runs |

8 training runs (`experiments/batch_r4.sh`, ~2 h each, two Titan RTX at a time → ~8 h) plus one CPU table.

## 3. Decision rule (written before the runs)

Reference: the W1R three-seed ensemble that was scored externally (group `W1R` in `R3_TABLE_A`).

- **Primary:** S10 AUROC of the candidate ensemble (CODE-15% only; the hidden format without the source cue; 77
  positives, so the bar is high) must exceed the reference by more than one paired SE, **and** S10+ AUROC must not be
  lower by more than one SE.
- **Guard:** S7 TPR@5% of the ensemble not lower than the reference by more than one SE.
- **Diagnostics:** length-cue gap ≤ +0.02; source gap not larger than the reference's.
- If several qualify, the one with the larger S10 AUROC gain; ties → fewer members / simpler change.
- **If none qualifies, the second entry is spent on attribution, not on score:** submit W1 (three seeds, round-3
  runs' configuration, no ranking loss), which differs from the scored entry only in the ranking loss. Its hidden
  score tells whether the loss mattered externally; the paper reports it either way. This is a user decision
  (D-2026-11-05-1 records the recommendation; the user chooses).

## 4. What is not done here

No new architecture; no change to preprocessing; no training on the organisers' validation feedback (the single
validation score cannot be used for selection without overfitting to it). Pinning the floating requirement versions
is left as is (verified image).
