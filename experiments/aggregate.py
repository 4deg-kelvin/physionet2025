#!/usr/bin/env python
"""Aggregate runs into configurations: mean +/- sd over seeds, seed ensemble, paired bootstrap.

PLAN_R2.md sections 2 and 5.  Given named groups of runs (one run per seed), for every
view (``S7``, ``S10``, ``S10+``, ``A``, ``B``; see ``experiments/scoring.py``):

* per-configuration **mean +/- sd** of TPR@5% (the challenge score) and AUROC over seeds;
* the **seed-ensemble** score (mean of the members' probabilities; deep ensembles,
  Lakshminarayanan et al., NeurIPS 2017) with a bootstrap 95% CI;
* **paired-bootstrap differences** against a named baseline: test records are resampled
  (1,000 times by default) and every model is scored on the *same* resamples, so the SE
  of a difference reflects only model-to-model variation.  Reported for the seed means
  and for the ensembles;
* the **length-cue gap** (CODE-15 negatives, short minus long mean probability), mean +/- sd;
* a pre-registered decision rule, when ``S7`` and ``S10+`` are among the views:
  ``--rule r2`` (default) is PLAN_R2.md section 5, ``--rule r3`` the round-3 rule
  (:func:`decision_rule_r3`).

Runs are directories under ``--results-dir`` holding ``predictions_A.csv`` /
``predictions_C.csv`` as written by ``eval_report.py``; a WiSE-FT mixture is a
sub-directory, e.g. ``r2_MW_s42/alpha0.50``.  ``run:suffix`` reads
``predictions_<A|C>_<suffix>.csv`` from the run instead, e.g. ``r3_W1_s42:tta`` for the
overlapping-window re-scoring that ``run_r3.sh`` writes.

    python experiments/aggregate.py \\
        --group B0=finalA_negcap,lrsweep_lr2e5_s43,lrsweep_lr2e5_s44 \\
        --group M1=r2_M1_s42,r2_M1_s43,r2_M1_s44 \\
        --baseline B0 --which A --out experiments/results/r2_table_A
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

EXPERIMENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENTS_DIR.parent
for _p in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import scoring as S  # noqa: E402
from chagas.metrics import challenge_score, compute_auc  # noqa: E402

DEFAULT_RESULTS = EXPERIMENTS_DIR / "results"
DEFAULT_VIEWS = ("S7", "S10", "S10+", "A", "B")


@dataclass
class Group:
    """A configuration: a name and one prediction frame per seed."""

    name: str
    runs: List[str]
    frames: List[pd.DataFrame] = field(default_factory=list)


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def parse_group(spec: str) -> Tuple[str, List[str]]:
    """``"NAME=run1,run2"`` -> ``("NAME", ["run1", "run2"])``."""
    if "=" not in spec:
        raise ValueError(f"group spec must be NAME=run1,run2,...: {spec!r}")
    name, runs = spec.split("=", 1)
    names = [r.strip() for r in runs.split(",") if r.strip()]
    if not name.strip() or not names:
        raise ValueError(f"empty group spec {spec!r}")
    return name.strip(), names


def predictions_file(results_dir: Path, run: str, which: str) -> Path:
    """``run`` -> its predictions CSV; ``run:suffix`` -> ``predictions_<which>_<suffix>.csv``."""
    name, _, suffix = run.partition(":")
    path = Path(name) if Path(name).is_absolute() else results_dir / name
    return path / (f"predictions_{which}_{suffix}.csv" if suffix else f"predictions_{which}.csv")


def load_run(results_dir: Path, run: str, which: str, durations: Optional[pd.DataFrame]) -> pd.DataFrame:
    """One run's saved predictions (:func:`predictions_file`), with durations attached."""
    df = pd.read_csv(predictions_file(results_dir, run, which), dtype={"stem": str}, low_memory=False)
    if "prob_filled" not in df.columns:
        df["prob_filled"] = df["prob"].fillna(-1.0)
    if durations is not None:
        df = S.attach_durations(df, durations)
    return df.sort_values("stem", kind="mergesort").reset_index(drop=True)


def check_aligned(frames: Sequence[pd.DataFrame]) -> None:
    """Every run must be scored on the same records, in the same order, with the same labels."""
    ref = frames[0]
    for f in frames[1:]:
        if len(f) != len(ref) or not np.array_equal(f["stem"].to_numpy(), ref["stem"].to_numpy()):
            raise ValueError("runs were scored on different record sets; cannot pair them")
        if not np.array_equal(f["label"].to_numpy(dtype=float), ref["label"].to_numpy(dtype=float)):
            raise ValueError("runs disagree on labels")


def mean_ensemble(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Mean-of-probabilities ensemble of aligned frames; exact for identical members.

    Computed as ``p0 + sum(p_i - p0) / k`` so an ensemble of identical models returns
    the member's probabilities bit for bit (tested).  Failed records (NaN) are averaged
    over the members that produced them and rank last if none did.
    """
    out = frames[0].copy()
    stacked = np.vstack([f["prob"].to_numpy(dtype=float) for f in frames])
    finite = np.isfinite(stacked)
    first = np.where(finite.any(axis=0), stacked[np.argmax(finite, axis=0), np.arange(stacked.shape[1])], np.nan)
    diffs = np.where(finite, stacked - first[None, :], 0.0)
    counts = finite.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(counts > 0, first + diffs.sum(axis=0) / np.maximum(counts, 1), np.nan)
    out["prob"] = mean
    out["prob_filled"] = np.where(np.isfinite(mean), mean, -1.0)
    return out


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def _point(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    return float(challenge_score(y, p)), float(compute_auc(y, p)[0])


def _sd(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def aggregate_view(
    groups: Sequence[Group],
    baseline: str,
    view: str,
    *,
    n_boot: int = S.DEFAULT_N_BOOT,
    seed: int = S.DEFAULT_SEED,
) -> List[Dict[str, object]]:
    """One row per group for ``view``: seed mean/sd, ensemble, paired differences."""
    ref = groups[0].frames[0]
    mask = S.view_mask(ref, view)
    y = ref.loc[mask, "label"].to_numpy(dtype=float)
    if y.size == 0 or np.unique(y).size < 2:
        return [{"group": g.name, "view": view, "note": "not scorable"} for g in groups]

    probs: List[np.ndarray] = []
    index: Dict[str, Tuple[List[int], int]] = {}
    for g in groups:
        member_rows = []
        for f in g.frames:
            member_rows.append(len(probs))
            probs.append(f.loc[mask, "prob_filled"].to_numpy(dtype=float))
        ens = mean_ensemble(g.frames)
        index[g.name] = (member_rows, len(probs))
        probs.append(ens.loc[mask, "prob_filled"].to_numpy(dtype=float))

    boot_ch, boot_au = S.bootstrap_metrics(y, probs, n_boot, S.view_seed(view, seed))
    if baseline not in index:
        raise KeyError(f"baseline {baseline!r} is not one of the groups")
    base_members, base_ens = index[baseline]
    base_mean_ch = boot_ch[base_members].mean(axis=0)
    base_mean_au = boot_au[base_members].mean(axis=0)

    rows: List[Dict[str, object]] = []
    for g in groups:
        members, ens_row = index[g.name]
        points = [_point(y, probs[i]) for i in members]
        ens_ch, ens_au = _point(y, probs[ens_row])
        mean_ch = boot_ch[members].mean(axis=0)
        mean_au = boot_au[members].mean(axis=0)
        row: Dict[str, object] = {
            "group": g.name,
            "view": view,
            "n_runs": len(members),
            "n": int(mask.sum()),
            "positives": int(y.sum()),
            "challenge_mean": float(np.mean([p[0] for p in points])),
            "challenge_sd": _sd([p[0] for p in points]),
            "auroc_mean": float(np.mean([p[1] for p in points])),
            "auroc_sd": _sd([p[1] for p in points]),
            "ens_challenge": ens_ch,
            "ens_auroc": ens_au,
        }
        row["ens_challenge_lo"], row["ens_challenge_hi"], _ = S.ci(boot_ch[ens_row])
        row["ens_auroc_lo"], row["ens_auroc_hi"], _ = S.ci(boot_au[ens_row])
        for key, diff, point in (
            ("d_challenge", mean_ch - base_mean_ch, None),
            ("d_auroc", mean_au - base_mean_au, None),
            ("d_ens_challenge", boot_ch[ens_row] - boot_ch[base_ens], None),
            ("d_ens_auroc", boot_au[ens_row] - boot_au[base_ens], None),
        ):
            lo, hi, se = S.ci(diff)
            row[key + "_se"], row[key + "_lo"], row[key + "_hi"] = se, lo, hi
        base_points = [_point(y, probs[i]) for i in base_members]
        base_ens_ch, base_ens_au = _point(y, probs[base_ens])
        row["d_challenge"] = row["challenge_mean"] - float(np.mean([p[0] for p in base_points]))
        row["d_auroc"] = row["auroc_mean"] - float(np.mean([p[1] for p in base_points]))
        row["d_ens_challenge"] = ens_ch - base_ens_ch
        row["d_ens_auroc"] = ens_au - base_ens_au
        rows.append(row)
    return rows


def length_gap_rows(groups: Sequence[Group]) -> List[Dict[str, object]]:
    """Length-cue gap per group (view-independent), mean +/- sd over seeds."""
    rows = []
    for g in groups:
        gaps = [S.length_cue_gap(f)["gap"] for f in g.frames]
        ens_gap = S.length_cue_gap(mean_ensemble(g.frames))["gap"]
        rows.append({"group": g.name, "gap_mean": float(np.mean(gaps)), "gap_sd": _sd(gaps),
                     "ens_gap": ens_gap})
    return rows


def decision_rule(rows: pd.DataFrame, gaps: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """PLAN_R2.md section 5, applied mechanically to seed means.

    Qualifies if (S7 TPR@5% gain > its paired SE, or S7 AUROC gain > 0.010 with the S7
    TPR@5% not lower by more than one SE) and S10+ AUROC not lower by more than its
    paired SE.  The length-cue gap must also shrink against the baseline.
    """
    out = []
    by = rows.set_index(["group", "view"])
    base_gap = float(gaps.set_index("group").loc[baseline, "gap_mean"])
    for name in rows["group"].unique():
        if name == baseline or (name, "S7") not in by.index or (name, "S10+") not in by.index:
            continue
        s7, s10 = by.loc[(name, "S7")], by.loc[(name, "S10+")]
        a = s7["d_challenge"] > s7["d_challenge_se"]
        b = s7["d_auroc"] > 0.010 and s7["d_challenge"] >= -s7["d_challenge_se"]
        c = s10["d_auroc"] >= -s10["d_auroc_se"]
        gap = float(gaps.set_index("group").loc[name, "gap_mean"])
        out.append({
            "group": name,
            "S7_tpr_gain_gt_se": bool(a),
            "S7_auroc_gain_gt_0.010_tpr_not_lower": bool(b),
            "S10+_auroc_not_lower": bool(c),
            "gap_shrinks": bool(gap < base_gap),
            "qualifies": bool((a or b) and c),
            "S7_d_challenge": float(s7["d_challenge"]),
        })
    return pd.DataFrame(out)


def decision_rule_r3(rows: pd.DataFrame, gaps: pd.DataFrame, baseline: str) -> pd.DataFrame:
    """The round-3 rule, applied mechanically to seed means.

    Round 2 made the 10 s format a guard only, and every remedy that removed the length
    cue moved S10+ AUROC by 4-5.5 SE while S7 stayed within noise (DECISIONS.md
    D-2026-10-01-5).  The hidden cohorts are ~10 s recordings, so round 3 makes S10+ the
    primary criterion:

    * **primary**: S10+ AUROC gain > its paired SE, and S7 TPR@5% not lower by more than
      its SE;
    * **secondary**: S7 TPR@5% gain > its SE, and S10+ AUROC not lower by more than its SE.

    ``qualifies`` = primary or secondary.  The length-cue gap is reported, not required.
    """
    out = []
    by = rows.set_index(["group", "view"])
    gap_by = gaps.set_index("group")
    base_gap = float(gap_by.loc[baseline, "gap_mean"])
    for name in rows["group"].unique():
        if name == baseline or (name, "S7") not in by.index or (name, "S10+") not in by.index:
            continue
        s7, s10 = by.loc[(name, "S7")], by.loc[(name, "S10+")]
        s10_gain = s10["d_auroc"] > s10["d_auroc_se"]
        s10_not_lower = s10["d_auroc"] >= -s10["d_auroc_se"]
        s7_gain = s7["d_challenge"] > s7["d_challenge_se"]
        s7_not_lower = s7["d_challenge"] >= -s7["d_challenge_se"]
        primary = bool(s10_gain and s7_not_lower)
        secondary = bool(s7_gain and s10_not_lower)
        out.append({
            "group": name,
            "S10+_auroc_gain_gt_se": bool(s10_gain),
            "S7_tpr_not_lower": bool(s7_not_lower),
            "S7_tpr_gain_gt_se": bool(s7_gain),
            "S10+_auroc_not_lower": bool(s10_not_lower),
            "gap_shrinks": bool(float(gap_by.loc[name, "gap_mean"]) < base_gap),
            "primary": primary,
            "secondary": secondary,
            "qualifies": primary or secondary,
            "S10+_d_auroc": float(s10["d_auroc"]),
            "S7_d_challenge": float(s7["d_challenge"]),
        })
    return pd.DataFrame(out)


#: ``--rule`` -> (rule function, markdown heading).
RULES = {
    "r2": (decision_rule, "PLAN_R2.md §5 decision rule (mechanical; seed means)"),
    "r3": (decision_rule_r3, "Round-3 decision rule (mechanical; seed means): primary = S10+ AUROC, "
                             "secondary = S7 TPR@5%"),
}


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
def _pm(mean: float, sd: float) -> str:
    return f"{mean:.4f}" if not np.isfinite(sd) else f"{mean:.4f} ± {sd:.4f}"


def _ci(point: float, lo: float, hi: float) -> str:
    return f"{point:.4f} [{lo:.4f}, {hi:.4f}]"


def _d(point: float, se: float) -> str:
    return f"{point:+.4f} ± {se:.4f}"


def to_markdown(rows: pd.DataFrame, gaps: pd.DataFrame, rule: pd.DataFrame, baseline: str,
                which: str, n_boot: int, rule_name: str = "r2") -> str:
    lines = [f"# Aggregate ({which} predictions; baseline `{baseline}`; {n_boot} paired resamples)", ""]
    lines.append("TPR@5% = challenge score.  `mean ± sd` over seeds; ensemble = mean of probabilities "
                 "with a 95% bootstrap CI; Δ = difference vs the baseline ± paired-bootstrap SE.")
    for view in rows["view"].unique():
        sub = rows[rows["view"] == view]
        if "note" in sub.columns and sub["note"].notna().all():
            continue
        first = sub.iloc[0]
        lines += ["", f"## {view} — {S.VIEW_DESCRIPTIONS.get(view, '')} "
                  f"(n = {int(first['n'])}, positives = {int(first['positives'])})", "",
                  "| config | seeds | TPR@5% | AUROC | ens TPR@5% | ens AUROC | Δ TPR@5% | Δ AUROC | Δ ens TPR@5% | Δ ens AUROC |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['group']} | {int(r['n_runs'])} | {_pm(r['challenge_mean'], r['challenge_sd'])} | "
                f"{_pm(r['auroc_mean'], r['auroc_sd'])} | "
                f"{_ci(r['ens_challenge'], r['ens_challenge_lo'], r['ens_challenge_hi'])} | "
                f"{_ci(r['ens_auroc'], r['ens_auroc_lo'], r['ens_auroc_hi'])} | "
                f"{_d(r['d_challenge'], r['d_challenge_se'])} | {_d(r['d_auroc'], r['d_auroc_se'])} | "
                f"{_d(r['d_ens_challenge'], r['d_ens_challenge_se'])} | {_d(r['d_ens_auroc'], r['d_ens_auroc_se'])} |"
            )
    lines += ["", "## Length-cue gap (CODE-15 negatives: mean prob short − long; ~0 = no length cue)", "",
              "| config | gap (mean ± sd over seeds) | ensemble gap |", "|---|---|---|"]
    for _, r in gaps.iterrows():
        lines.append(f"| {r['group']} | {_pm(r['gap_mean'], r['gap_sd'])} | {r['ens_gap']:.4f} |")
    if not rule.empty:
        lines += ["", f"## {RULES[rule_name][1]}", "",
                  "```", rule.to_string(index=False), "```"]
    return "\n".join(lines) + "\n"


def run(
    specs: Sequence[str],
    baseline: str,
    *,
    which: str = "A",
    views: Sequence[str] = DEFAULT_VIEWS,
    results_dir: Path = DEFAULT_RESULTS,
    durations_path: Optional[str] = None,
    n_boot: int = S.DEFAULT_N_BOOT,
    seed: int = S.DEFAULT_SEED,
    rule: str = "r2",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Load, aggregate and format; returns ``(rows, gaps, rule, markdown)``."""
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; choose from {sorted(RULES)}")
    durations = S.load_durations(durations_path)
    groups = []
    for spec in specs:
        name, runs = parse_group(spec)
        groups.append(Group(name, runs, [load_run(results_dir, r, which, durations) for r in runs]))
    check_aligned([f for g in groups for f in g.frames])
    rows: List[Dict[str, object]] = []
    for view in views:
        rows.extend(aggregate_view(groups, baseline, view, n_boot=n_boot, seed=seed))
    table = pd.DataFrame(rows)
    gaps = pd.DataFrame(length_gap_rows(groups))
    decided = RULES[rule][0](table, gaps, baseline) if {"S7", "S10+"} <= set(views) else pd.DataFrame()
    return table, gaps, decided, to_markdown(table, gaps, decided, baseline, which, n_boot, rule)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="append", required=True,
                    help="NAME=run1,run2[:suffix],... (repeatable)")
    ap.add_argument("--baseline", required=True, help="name of the baseline group")
    ap.add_argument("--which", default="A", choices=("A", "C"), help="which prediction set")
    ap.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    ap.add_argument("--results-dir", default=str(DEFAULT_RESULTS))
    ap.add_argument("--durations", default=str(S.DEFAULT_DURATIONS))
    ap.add_argument("--n-boot", type=int, default=S.DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=S.DEFAULT_SEED)
    ap.add_argument("--rule", default="r2", choices=sorted(RULES), help="decision rule (default r2)")
    ap.add_argument("--out", default="", help="output prefix: writes <out>.md, <out>.csv, <out>_gap.csv")
    args = ap.parse_args(argv)

    views = [v.strip() for v in args.views.split(",") if v.strip()]
    table, gaps, rule, md = run(
        args.group, args.baseline, which=args.which, views=views,
        results_dir=Path(args.results_dir), durations_path=args.durations,
        n_boot=args.n_boot, seed=args.seed, rule=args.rule,
    )
    print(md)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.with_suffix(".md").write_text(md)
        table.to_csv(out.with_suffix(".csv"), index=False)
        gaps.to_csv(out.parent / f"{out.name}_gap.csv", index=False)
        if not rule.empty:
            rule.to_csv(out.parent / f"{out.name}_rule.csv", index=False)
        print(f"wrote {out.with_suffix('.md')} and {out.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
