#!/usr/bin/env python
"""Re-score every saved run with the round-2 views into ``experiments/results/RESCORED.md``.

PLAN_R2.md section 2: all scoring is from saved predictions
(``results/<run>/predictions_{A,C}.csv``), so re-scoring costs no GPU time.  For every
run: views A, B, S7, S10, S10+ on the Test-A predictions and the same views on Test C,
each with a bootstrap 95% CI, and the length-cue gap.  Then the multi-seed
configurations are aggregated with ``aggregate.py`` (mean +/- sd, seed ensemble, paired
differences against the shipped config B0).

    python experiments/rescore_all.py            # writes RESCORED.md / RESCORED_runs.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
for _p in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import aggregate as AG  # noqa: E402
import scoring as S  # noqa: E402

RESULTS = EXPERIMENTS_DIR / "results"

#: Multi-seed configurations among the round-1 runs (DECISIONS.md D-2026-09-30-1).
ROUND1_GROUPS: Dict[str, List[str]] = {
    "B0 shipped (lr 2e-5)": ["finalA_negcap", "lrsweep_lr2e5_s43", "lrsweep_lr2e5_s44"],
    "lr 5e-6": ["lrsweep_lr5e6_s42", "lrsweep_lr5e6_s43", "lrsweep_lr5e6_s44"],
    "lr 1e-6": ["lrsweep_lr1e6_s42", "lrsweep_lr1e6_s43", "lrsweep_lr1e6_s44"],
    "full-data final": ["final_full_seed42", "final_full_seed43"],
}
BASELINE = "B0 shipped (lr 2e-5)"


def discover_runs(results_dir: Path) -> List[str]:
    """Every run directory holding a ``predictions_A.csv``, sorted by name."""
    return sorted(p.parent.name for p in results_dir.glob("*/predictions_A.csv"))


def score_run(results_dir: Path, run: str, durations: pd.DataFrame, n_boot: int) -> List[Dict[str, object]]:
    """All views on A (and C when present) for one run, plus its length-cue gap."""
    rows: List[Dict[str, object]] = []
    for which in ("A", "C"):
        path = results_dir / run / f"predictions_{which}.csv"
        if not path.is_file():
            continue
        df = AG.load_run(results_dir, run, which, durations)
        gap = S.length_cue_gap(df)["gap"]
        for r in S.score_all_views(df, suffix="" if which == "A" else "·C", n_boot=n_boot):
            r.update({"run": run, "predictions": which, "length_cue_gap": gap})
            rows.append(r)
    return rows


def _cell(row: pd.Series, ci: bool) -> str:
    text = f"{row['challenge']:.3f} / {row['auroc']:.3f}"
    if ci and "challenge_lo" in row and pd.notna(row.get("challenge_lo")):
        text += f" [{row['challenge_lo']:.3f}–{row['challenge_hi']:.3f}]"
    return text


def runs_markdown(table: pd.DataFrame) -> str:
    views = ["A", "B", "S7", "S10", "S10+", "S7·C", "S10+·C"]
    lines = ["| run | " + " | ".join(views) + " | length-cue gap |",
             "|---|" + "---|" * (len(views) + 1)]
    for run, sub in table.groupby("run", sort=True):
        by = sub.set_index("view")
        cells = [_cell(by.loc[v], ci=v in ("S7", "S10", "S10+")) if v in by.index else "—" for v in views]
        gap = float(sub[sub["predictions"] == "A"]["length_cue_gap"].iloc[0])
        lines.append(f"| {run} | " + " | ".join(cells) + f" | {gap:+.3f} |")
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(RESULTS))
    ap.add_argument("--durations", default=str(S.DEFAULT_DURATIONS))
    ap.add_argument("--n-boot", type=int, default=S.DEFAULT_N_BOOT)
    args = ap.parse_args(argv)
    results_dir = Path(args.results_dir)
    durations = S.load_durations(args.durations)

    runs = discover_runs(results_dir)
    print(f"[rescore] {len(runs)} runs: {runs}", flush=True)
    rows: List[Dict[str, object]] = []
    for run in runs:
        rows.extend(score_run(results_dir, run, durations, args.n_boot))
        print(f"[rescore] scored {run}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(results_dir / "RESCORED_runs.csv", index=False)

    # Aggregate: the multi-seed configurations plus every single run as its own group,
    # all paired against B0 on the same bootstrap resamples.
    grouped = {name: [r for r in members if r in runs] for name, members in ROUND1_GROUPS.items()}
    in_groups = {r for members in grouped.values() for r in members}
    specs = [f"{name}={','.join(members)}" for name, members in grouped.items() if members]
    specs += [f"{r}={r}" for r in runs if r not in in_groups]
    agg_md = {}
    for which in ("A", "C"):
        agg, gaps, rule, md = AG.run(specs, BASELINE, which=which, results_dir=results_dir,
                                     durations_path=args.durations, n_boot=args.n_boot)
        agg.to_csv(results_dir / f"RESCORED_aggregate_{which}.csv", index=False)
        gaps.to_csv(results_dir / f"RESCORED_gap_{which}.csv", index=False)
        agg_md[which] = md

    dur_rows = []
    ref = AG.load_run(results_dir, runs[0], "A", durations)
    for v in ("A", "B"):
        dur_rows.append(S.duration_only(ref, v))
    counts = {v: (int(S.view_mask(ref, v).sum()), int(ref.loc[S.view_mask(ref, v), "label"].sum()))
              for v in S.VIEW_ORDER}

    lines = [
        "# Every saved run, re-scored with the round-2 views (PLAN_R2.md section 2)",
        "",
        "Generated by `experiments/rescore_all.py` from the saved `predictions_{A,C}.csv`; no model was re-run.",
        "Cells are **TPR@5% / AUROC**; S-views also give the 95% bootstrap CI of TPR@5% "
        f"({args.n_boot} resamples).  `·C` = the same view on Test C.",
        "",
        "View sizes (records / positives): " + "; ".join(f"{v} {n:,} / {p:,}" for v, (n, p) in counts.items()) + ".",
        "",
        "Duration-only baseline (score = 1 for records < 7.5 s): "
        + "; ".join(f"view {d['view']}: TPR@5% {d['challenge']:.3f}, AUROC {d['auroc']:.3f}" for d in dur_rows) + ".",
        "",
        "Length-cue gap = mean predicted probability of CODE-15 negatives, short (< 7.5 s) minus long (≥ 9.9 s).",
        "",
        "## Per run",
        "",
        runs_markdown(table),
        "",
        "## Configurations (Test A predictions)",
        "",
        agg_md["A"].split("\n", 1)[1],
        "",
        "## Configurations (Test C predictions)",
        "",
        agg_md["C"].split("\n", 1)[1],
    ]
    out = results_dir / "RESCORED.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"[rescore] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
