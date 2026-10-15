#!/usr/bin/env python
"""Audit the held-out split: leakage checks and counts by source x split x length group x label.

The held-out split (``fixbranch_test/split/split.csv``, built by ``build_split.py``) is what
every test view is computed on, so its integrity is checked from the file itself:

* no patient group (``group``: CODE-15 ``patient_id``, PTB-XL ``patient_id``, SaMi-Trop
  record) appears in both splits;
* no record stem appears twice;
* every record has a duration in ``experiments/data/record_durations.csv``.

and its composition is tabulated by source x split x length group (:mod:`chagas.lengths`:
short < 7.5 s, mid, long >= 9.9 s, unknown): records, negatives, positives, prevalence
and unique groups, plus per-(source, split) and per-split totals (``length_group = all``).
CODE-15's record length predicts its label (DECISIONS.md D-2026-09-30-2), which is why
the length group is part of every cell.

Writes ``<out>.md`` and ``<out>.csv``; exits 1 if a check fails (the files are written
either way).  CPU-only, pandas, well under a minute; run it through
``experiments/batch_split_audit.sh``, not on the login node.

    python experiments/check_split.py --out experiments/results/SPLIT_AUDIT
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chagas.lengths import LENGTH_GROUPS, length_group  # noqa: E402

DEFAULT_SPLIT = "/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/split/split.csv"
DEFAULT_DURATIONS = REPO_ROOT / "experiments" / "data" / "record_durations.csv"
DEFAULT_OUT = REPO_ROOT / "experiments" / "results" / "SPLIT_AUDIT"
#: How many offending stems / groups a failed check lists.
N_EXAMPLES = 10

__all__ = ["load_split", "audit", "to_markdown", "main"]


def load_split(path: str) -> pd.DataFrame:
    """``split.csv`` with ``stem`` and ``group`` read as strings (stems have leading zeros)."""
    return pd.read_csv(path, dtype={"stem": str, "group": str, "source": str, "split": str})


def _cell(df: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(df))
    pos = int((df["label"] == 1).sum())
    return {
        "n": n,
        "n_neg": n - pos,
        "n_pos": pos,
        "prevalence": pos / n if n else float("nan"),
        "groups": int(df["group"].nunique()),
    }


def audit(split: pd.DataFrame, durations: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """``(counts table, checks)`` for a split frame and a ``stem, duration_s`` frame."""
    df = split.copy()
    df["stem"] = df["stem"].astype(str)
    df["label"] = df["label"].astype(int)
    dur = durations[["stem", "duration_s"]].copy()
    dur["stem"] = dur["stem"].astype(str)
    dur = dur.drop_duplicates("stem")
    df = df.merge(dur, on="stem", how="left")
    df["length_group"] = [length_group(d) for d in df["duration_s"].to_numpy(dtype=float)]

    splits_per_group = df.groupby("group")["split"].nunique()
    leaked = sorted(splits_per_group[splits_per_group > 1].index)
    dup = df["stem"][df["stem"].duplicated(keep=False)]
    duplicated = sorted(dup.unique())
    no_duration = sorted(df.loc[df["duration_s"].isna(), "stem"])
    checks: Dict[str, Any] = {
        "records": int(len(df)),
        "groups": int(df["group"].nunique()),
        "splits": sorted(df["split"].unique()),
        "groups_in_both_splits": len(leaked),
        "groups_in_both_splits_examples": leaked[:N_EXAMPLES],
        "duplicate_stems": len(duplicated),
        "duplicate_stems_examples": duplicated[:N_EXAMPLES],
        "records_without_duration": len(no_duration),
        "records_without_duration_examples": no_duration[:N_EXAMPLES],
    }
    checks["ok"] = not leaked and not duplicated and not no_duration

    order = {g: i for i, g in enumerate(LENGTH_GROUPS + ("all",))}
    rows: List[Dict[str, Any]] = []
    for (source, split_name), sub in df.groupby(["source", "split"], sort=True):
        for group_name, cell in sub.groupby("length_group", sort=False):
            rows.append({"source": source, "split": split_name, "length_group": group_name, **_cell(cell)})
        rows.append({"source": source, "split": split_name, "length_group": "all", **_cell(sub)})
    for split_name, sub in df.groupby("split", sort=True):
        rows.append({"source": "all", "split": split_name, "length_group": "all", **_cell(sub)})
    table = pd.DataFrame(rows)
    table["_order"] = table["length_group"].map(order)
    table["_all"] = table["source"] == "all"
    table = (table.sort_values(["_all", "source", "split", "_order"], kind="mergesort")
             .drop(columns=["_order", "_all"]).reset_index(drop=True))
    return table, checks


def to_markdown(table: pd.DataFrame, checks: Dict[str, Any], split_path: str, durations_path: str) -> str:
    verdict = "PASS" if checks["ok"] else "FAIL"
    lines = [
        f"# Split audit: {verdict}",
        "",
        f"Split `{split_path}` ({checks['records']:,} records, {checks['groups']:,} patient groups, "
        f"splits {', '.join(checks['splits'])}); durations `{durations_path}`.  "
        "Generated by `experiments/check_split.py`.",
        "",
        "| check | result |",
        "|---|---|",
        f"| patient groups in both splits | {checks['groups_in_both_splits']} |",
        f"| duplicate record stems | {checks['duplicate_stems']} |",
        f"| records without a duration | {checks['records_without_duration']} |",
    ]
    for key in ("groups_in_both_splits", "duplicate_stems", "records_without_duration"):
        if checks[key]:
            lines.append(f"\n{key} (first {N_EXAMPLES}): {', '.join(map(str, checks[key + '_examples']))}")
    lines += [
        "",
        "## Counts by source x split x length group (short < 7.5 s, long >= 9.9 s)",
        "",
        "| source | split | length group | records | negatives | positives | prevalence | groups |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        prev = f"{r['prevalence']:.4f}" if np.isfinite(r["prevalence"]) else "n/a"
        lines.append(
            f"| {r['source']} | {r['split']} | {r['length_group']} | {int(r['n']):,} | {int(r['n_neg']):,} | "
            f"{int(r['n_pos']):,} | {prev} | {int(r['groups']):,} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--durations", default=str(DEFAULT_DURATIONS))
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output prefix: writes <out>.md and <out>.csv")
    args = ap.parse_args(argv)

    table, checks = audit(load_split(args.split), pd.read_csv(args.durations, dtype={"stem": str}))
    md = to_markdown(table, checks, args.split, args.durations)
    print(md)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text(md)
    table.to_csv(out.with_suffix(".csv"), index=False)
    print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.csv')}")
    return 0 if checks["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
