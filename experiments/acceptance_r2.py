#!/usr/bin/env python
"""Check the round-2 scoring tools against the numbers computed by hand on 2026-09-30.

The targets come from the analysis that found the length shortcut (DECISIONS.md
D-2026-09-30-2, PLAN_R2.md section 1).  Each is recomputed here with
``experiments/scoring.py`` / ``aggregate.py`` from the saved predictions and
``experiments/data/record_durations.csv``; the script prints target vs measured and
exits non-zero if any differs by more than half a unit in the target's last digit.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
for _p in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import pandas as pd  # noqa: E402

import aggregate as AG  # noqa: E402
import scoring as S  # noqa: E402
from chagas.metrics import challenge_score, compute_auc  # noqa: E402

RESULTS = EXPERIMENTS_DIR / "results"
SPLIT = "/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/split/split.csv"
B0 = ("finalA_negcap", "lrsweep_lr2e5_s43", "lrsweep_lr2e5_s44")


def _scores(df, view: str) -> Tuple[float, float]:
    sub = df[S.view_mask(df, view)]
    y = sub["label"].to_numpy(dtype=float)
    p = sub["prob_filled"].to_numpy(dtype=float)
    return float(challenge_score(y, p)), float(compute_auc(y, p)[0])


def checks() -> List[Tuple[str, float, float, int]]:
    """``(name, target, measured, decimals)``."""
    durations = S.load_durations()
    final_a = AG.load_run(RESULTS, "finalA_negcap", "A", durations)
    b0 = [AG.load_run(RESULTS, r, "A", durations) for r in B0]
    ens = AG.mean_ensemble(b0)
    fa_s7 = _scores(final_a, "S7")
    fa_s10 = _scores(final_a, "S10")
    ens_a = _scores(ens, "A")
    ens_s7 = _scores(ens, "S7")
    dur_b = S.duration_only(final_a, "B")
    test_c15 = final_a[S.view_mask(final_a, "B")]
    pos_short = int(((test_c15["length_group"] == "short") & (test_c15["label"] == 1)).sum())
    pos_long = int(((test_c15["length_group"] == "long") & (test_c15["label"] == 1)).sum())
    # The 2026-09-30 hand number 0.681 reproduces on ALL CODE-15 records (train + test),
    # not on the test-only view B (DECISIONS.md D-2026-10-01-4); both are reported.
    split = pd.read_csv(SPLIT, low_memory=False, dtype={"stem": str}).merge(durations, on="stem")
    c15 = split[split["source"].astype(str).str.startswith("CODE")]
    dur_all = compute_auc(c15["label"].to_numpy(dtype=float),
                          (c15["duration_s"].to_numpy(dtype=float) < 7.5).astype(float))[0]
    return [
        ("duration-only AUROC, view B (test)", 0.681, float(dur_b["auroc"]), 3),
        ("duration-only AUROC, all CODE-15 records", 0.681, float(dur_all), 3),
        ("finalA_negcap S7 AUROC", 0.791, fa_s7[1], 3),
        ("finalA_negcap S7 TPR@5%", 0.365, fa_s7[0], 3),
        ("finalA_negcap S10 AUROC", 0.708, fa_s10[1], 3),
        ("B0 3-seed ensemble, view A TPR@5%", 0.4525, ens_a[0], 4),
        ("B0 3-seed ensemble, S7 TPR@5%", 0.404, ens_s7[0], 3),
        ("B0 3-seed ensemble, S7 AUROC", 0.801, ens_s7[1], 3),
        ("CODE-15 test positives, short (< 7.5 s)", 1259, pos_short, 0),
        ("CODE-15 test positives, long (>= 9.9 s)", 77, pos_long, 0),
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    failed = 0
    print(f"{'check':<44} {'target':>8} {'measured':>10}  ok")
    for name, target, measured, decimals in checks():
        ok = abs(measured - target) <= 0.5 * 10 ** (-decimals) + 1e-12
        failed += not ok
        print(f"{name:<44} {target:>8} {measured:>10.{max(decimals, 4)}f}  {'OK' if ok else 'MISMATCH'}")
    print("ALL ACCEPTANCE CHECKS PASS" if not failed else f"{failed} CHECK(S) DIFFER")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
