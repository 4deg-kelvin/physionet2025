#!/usr/bin/env python
"""Score one inference run and print the diagnostics PLAN.md section 2 and PLAN_R2.md section 2 ask for.

Round 2 (PLAN_R2.md section 2).  CODE-15 record length predicts the label (7.34 s vs
10.24 s records, 12.7x prevalence ratio; DECISIONS.md D-2026-09-30-2), so views A, B and
C below are all inflated by a length cue.  The decision views are now:

    S7    CODE-15, duration < 7.5 s   (PRIMARY: no source or length cue, 1,259 positives)
    S10   CODE-15, duration >= 9.9 s  (closest to the hidden cohorts' format, 77 positives)
    S10+  CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)

each with a bootstrap 95% CI (1,000 resamples of patients; ``--bootstrap-unit record``
resamples records instead), plus the duration-only baseline (a score
that is 1 for short records) and the **length-cue gap** (mean predicted probability of
CODE-15 negatives, short minus long).  Definitions live in ``experiments/scoring.py``.

The round-1 description follows.

    Test A  full held-out set, natural composition -- the in-distribution number,
            comparable to the top teams' CV numbers.
    Test B  the CODE-15-only subset of the SAME predictions.  SaMi-Trop is 100%
            positive and PTB-XL 100% negative, so any set preserving that composition
            rewards a model for recognising the *dataset*; B removes that shortcut.
    Test C  Test A perturbed the way the organisers perturbed the hidden cohorts
            (see experiments/build_test_C.py).  The C-A gap is the local estimate of
            the out-of-distribution collapse every team suffered: median hidden
            validation 0.279 -> ELSA-Brasil 0.100.

Also reported: per-source mean predicted probability (source-shortcut diagnostic) and
the score by age decile (age-shortcut diagnostic -- ELSA-Brasil is a broader,
asymptomatic population, so a model that is really reading age will not transfer).

Inputs are ``run_model.py``-style output trees, or the CSV that
``experiments/predict_batched.py`` writes (``--preds-a/--preds-c``); a record with no
parsable probability is
scored at the worst possible rank rather than dropped, because dropping it would
silently flatter the model by removing the records it could not handle.

Usage
-----
    python experiments/eval_report.py \
        --name row5_two_stage_ft \
        --outputs-a  .../outputs_A \
        --outputs-c  .../outputs_C \
        --data       .../data_test \
        --split      .../split/split.csv \
        --out-dir    experiments/results/row5_two_stage_ft

    # or, from predict_batched.py output:
    python experiments/eval_report.py --name r2_W1_s42 \
        --preds-a experiments/results/r2_W1_s42/raw_predictions_A.csv \
        --preds-c experiments/results/r2_W1_s42/raw_predictions_C.csv \
        --out-dir experiments/results/r2_W1_s42

    # re-score an existing results folder from its saved predictions_{A,C}.csv, in place
    # (report.md and the tables; the predictions files are not rewritten):
    python experiments/eval_report.py --from-saved experiments/results/r2_W1_s42
    python experiments/eval_report.py --from-saved experiments/results/r3_W1_s42 --saved-suffix tta
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, EXPERIMENTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from chagas.metrics import challenge_score, compute_auc  # noqa: E402

import scoring as S  # noqa: E402  (experiments/scoring.py)

_PROB_RE = re.compile(r"probability:\s*(\S+)", re.I)
DEFAULT_SPLIT = "/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/split/split.csv"
DEFAULT_DATA = "/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/data_test"


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def read_outputs(output_folder: str) -> pd.DataFrame:
    """``stem -> probability`` from a ``run_model.py`` output tree."""
    rows: List[Tuple[str, float]] = []
    for path in glob.glob(os.path.join(output_folder, "**", "*.txt"), recursive=True):
        stem = os.path.basename(path)[: -len(".txt")]
        try:
            text = open(path).read()
        except OSError:
            continue
        match = _PROB_RE.search(text)
        value = np.nan
        if match:
            try:
                value = float(match.group(1))
            except ValueError:
                value = np.nan
        rows.append((stem, value))
    return pd.DataFrame(rows, columns=["stem", "prob"])


def _read_age(stem_path: str) -> Tuple[str, float]:
    try:
        with open(stem_path + ".hea") as fh:
            for line in fh:
                if line.startswith("# Age:"):
                    return os.path.basename(stem_path), float(line[len("# Age:"):].strip())
    except (OSError, ValueError):
        pass
    return os.path.basename(stem_path), float("nan")


def read_ages(data_folder: str, workers: int = 16) -> pd.DataFrame:
    stems = []
    for root, _dirs, files in os.walk(data_folder):
        for name in files:
            if name.endswith(".hea"):
                stems.append(os.path.join(root, name)[: -len(".hea")])
    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(_read_age, stems))
    return pd.DataFrame(rows, columns=["stem", "age"])


def load_truth(split_csv: str) -> pd.DataFrame:
    df = pd.read_csv(split_csv, low_memory=False)
    df = df[df["split"] == "test"][["stem", "label", "source"]].copy()
    df["stem"] = df["stem"].astype(str)
    df["label"] = df["label"].astype(float)
    return df


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------
def score(sub: pd.DataFrame, name: str) -> Dict[str, object]:
    """Challenge score, AUROC, AUPRC and the prevalence ceiling for one view."""
    if len(sub) == 0 or sub["label"].nunique() < 2:
        return {"view": name, "n": len(sub), "note": "not scorable"}
    y = sub["label"].to_numpy(dtype=float)
    p = sub["prob_filled"].to_numpy(dtype=float)
    prevalence = float(y.mean())
    # The challenge score is recall within the top 5%, so it is capped at
    # 0.05 / prevalence.  Without that ceiling the raw number is not interpretable.
    ceiling = min(1.0, 0.05 / prevalence) if prevalence > 0 else float("nan")
    cs = challenge_score(y, p)
    auroc, auprc = compute_auc(y, p)
    return {
        "view": name,
        "n": int(len(sub)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "challenge": float(cs),
        "ceiling": float(ceiling),
        "frac_of_max": float(cs / ceiling) if ceiling and ceiling == ceiling else float("nan"),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_failed": int(sub["prob"].isna().sum()),
    }


def by_age_decile(sub: pd.DataFrame) -> pd.DataFrame:
    """Challenge score inside each age decile -- the age-shortcut diagnostic."""
    df = sub.dropna(subset=["age"]).copy()
    if df.empty:
        return pd.DataFrame()
    try:
        df["decile"] = pd.qcut(df["age"], 10, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows = []
    for decile, group in df.groupby("decile", observed=True):
        y = group["label"].to_numpy(dtype=float)
        p = group["prob_filled"].to_numpy(dtype=float)
        rows.append(
            {
                "age_decile": str(decile),
                "n": int(len(group)),
                "positives": int(y.sum()),
                "mean_prob": float(p.mean()),
                "challenge": float(challenge_score(y, p)) if y.sum() > 0 else float("nan"),
                "auroc": float(compute_auc(y, p)[0]),
            }
        )
    return pd.DataFrame(rows)


def by_source(sub: pd.DataFrame) -> pd.DataFrame:
    return (
        sub.groupby("source")
        .agg(n=("prob_filled", "size"), positives=("label", "sum"),
             mean_prob=("prob_filled", "mean"))
        .reset_index()
    )


def read_predictions_csv(path: str) -> pd.DataFrame:
    """``stem -> probability`` from ``predict_batched.py``'s CSV (``stem, binary, prob``)."""
    df = pd.read_csv(path, dtype={"stem": str})
    return df[["stem", "prob"]]


def read_predictions(source: str) -> pd.DataFrame:
    """A ``run_model.py`` output tree (directory) or a ``predict_batched.py`` CSV (file)."""
    if os.path.isdir(source):
        return read_outputs(source)
    return read_predictions_csv(source)


def build_view(
    source: str,
    truth: pd.DataFrame,
    ages: pd.DataFrame,
    durations: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    preds = read_predictions(source)
    df = truth.merge(preds, on="stem", how="left").merge(ages, on="stem", how="left")
    # Records that produced no output, or a nan probability, rank last.
    df["prob_filled"] = df["prob"].fillna(-1.0)
    if durations is not None:
        df = S.attach_durations(df, durations)
    return df


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
_VIEW_COLS = [
    "view", "n", "positives", "prevalence", "challenge", "challenge_lo", "challenge_hi",
    "auroc", "auroc_lo", "auroc_hi", "auprc", "ceiling", "n_failed",
]


def _fmt(rows: Sequence[Dict[str, object]], columns: Optional[Sequence[str]] = None) -> str:
    df = pd.DataFrame(rows)
    if columns is not None:
        df = df[[c for c in columns if c in df.columns]]
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def _row(rows: Sequence[Dict[str, object]], view: str) -> Dict[str, object]:
    for r in rows:
        if r.get("view") == view:
            return r
    return {}


def build_report(
    name: str,
    view_a: pd.DataFrame,
    view_c: Optional[pd.DataFrame] = None,
    *,
    n_boot: int = S.DEFAULT_N_BOOT,
    seed: int = S.DEFAULT_SEED,
    unit: str = S.DEFAULT_BOOTSTRAP_UNIT,
    groups: Optional[Mapping[str, str]] = None,
) -> Tuple[str, Dict[str, pd.DataFrame]]:
    """Markdown report and its tables for one run (A predictions, optional C predictions).

    Both frames carry ``stem, label, source, prob, prob_filled`` and, for the round-2
    views, ``duration_s`` / ``length_group`` (:func:`scoring.attach_durations`).
    ``unit`` / ``groups``: bootstrap resampling unit and ``stem -> patient group`` map.
    """
    has_dur = "duration_s" in view_a.columns
    views = S.VIEW_ORDER if has_dur else ("A", "B")
    kw = dict(n_boot=n_boot, seed=seed, unit=unit, groups=groups)
    summary = S.score_all_views(view_a, views=views, **kw)
    if view_c is not None:
        summary += S.score_all_views(view_c, suffix="·C", views=views, **kw)

    diagnostics: List[Dict[str, object]] = []
    if has_dur:
        for v in ("A", "B"):
            row = S.duration_only(view_a, v)
            row["diagnostic"] = "duration-only baseline (1 = short record)"
            diagnostics.append(row)
        for tag, frame in (("A", view_a), ("C", view_c)):
            if frame is None:
                continue
            gap = S.length_cue_gap(frame)
            gap.update({"diagnostic": "length-cue gap (CODE-15 negatives)", "view": tag})
            diagnostics.append(gap)

    lines: List[str] = [f"# Evaluation report -- {name}", ""]
    lines += ["## Views (TPR@5% = `challenge`; 95% bootstrap CIs, "
              f"{n_boot} {unit}-level resamples)", ""]
    for v in views:
        lines.append(f"* `{v}`: {S.VIEW_DESCRIPTIONS[v]}")
    lines += ["* `·C`: the same view on Test C (organiser-style perturbations)", ""]
    lines += ["```", _fmt(summary, _VIEW_COLS), "```"]

    a_score = _row(summary, "A").get("challenge")
    c_score = _row(summary, "A·C").get("challenge")
    if isinstance(a_score, float) and isinstance(c_score, float):
        gap = c_score - a_score
        lines += ["", f"**C - A gap: {gap:+.4f}** "
                  f"({'robust' if gap > -0.02 else 'degrades out of distribution'}); "
                  f"A = {a_score:.4f}, C = {c_score:.4f}."]
    s7, s7c = _row(summary, "S7").get("challenge"), _row(summary, "S7·C").get("challenge")
    if isinstance(s7, float) and isinstance(s7c, float):
        lines.append(f"**S7·C - S7 gap: {s7c - s7:+.4f}** (the C-A gap with the length cue removed).")

    if diagnostics:
        lines += ["", "## Length-shortcut diagnostics", ""]
        base = [d for d in diagnostics if d["diagnostic"].startswith("duration")]
        gaps = [d for d in diagnostics if d["diagnostic"].startswith("length-cue")]
        lines += ["Duration-only baseline (what record length alone scores):", "```",
                  _fmt(base, ["view", "n", "challenge", "auroc"]), "```", ""]
        lines += ["Length-cue gap: mean predicted probability of CODE-15 negatives by length "
                  "group; `gap = short - long` should be ~0 for a model that ignores length:",
                  "```", _fmt(gaps, ["view", "neg_mean_short", "neg_mean_mid", "neg_mean_long",
                                     "gap", "neg_n_short", "neg_n_long"]), "```"]

    lines += ["", "## Per-source mean predicted probability (source-shortcut diagnostic)", "```",
              by_source(view_a).to_string(index=False, float_format=lambda v: f"{v:.4f}"), "```"]
    decile = by_age_decile(view_a) if "age" in view_a.columns else pd.DataFrame()
    if not decile.empty:
        lines += ["", "## Score by age decile (age-shortcut diagnostic)", "```",
                  decile.to_string(index=False, float_format=lambda v: f"{v:.4f}"), "```"]

    tables = {
        "summary": pd.DataFrame(summary),
        "diagnostics": pd.DataFrame(diagnostics),
        "per_source": by_source(view_a),
        "per_age_decile": decile,
    }
    return "\n".join(lines), tables


def _saved_name(folder: Path, report: str) -> str:
    """The run name from the header of an existing report (``# Evaluation report -- NAME``)."""
    path = folder / report
    if path.is_file():
        first = path.read_text().split("\n", 1)[0]
        if first.startswith("# Evaluation report -- "):
            return first[len("# Evaluation report -- "):].strip()
    return folder.name


def load_saved(folder: Path, which: str, suffix: str, durations: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """``predictions_<which>[_<suffix>].csv`` that :func:`main` wrote earlier, in its row order."""
    path = folder / (f"predictions_{which}_{suffix}.csv" if suffix else f"predictions_{which}.csv")
    if not path.is_file():
        return None
    df = pd.read_csv(path, dtype={"stem": str}, low_memory=False)
    if "prob_filled" not in df.columns:
        df["prob_filled"] = df["prob"].fillna(-1.0)
    if durations is not None:
        df = S.attach_durations(df, durations)
    return df


def rescore_saved(args: argparse.Namespace) -> int:
    """``--from-saved``: rebuild report and tables from saved predictions, in place.

    Writes ``report.md`` and ``<table>.csv`` (or ``report_<suffix>.md`` and
    ``report_<suffix>_<table>.csv``, the names ``run_r3.sh`` gives the test-time-window
    pass); never rewrites the predictions files.
    """
    folder = Path(args.from_saved)
    suffix = args.saved_suffix
    durations = S.load_durations(args.durations) if args.durations else None
    view_a = load_saved(folder, "A", suffix, durations)
    if view_a is None:
        raise FileNotFoundError(f"no saved Test-A predictions in {folder} (suffix {suffix!r})")
    view_c = load_saved(folder, "C", suffix, durations)
    report_name = f"report_{suffix}.md" if suffix else "report.md"
    name = args.name or _saved_name(folder, report_name)
    groups = S.load_groups(args.split) if args.bootstrap_unit == "patient" else None
    report, tables = build_report(name, view_a, view_c, n_boot=args.n_boot, seed=args.seed,
                                  unit=args.bootstrap_unit, groups=groups)
    print(report)
    (folder / report_name).write_text(report + "\n")
    for key, table in tables.items():
        if not table.empty:
            table.to_csv(folder / (f"report_{suffix}_{key}.csv" if suffix else f"{key}.csv"), index=False)
    print(f"\nrewrote {folder / report_name} and the accompanying CSVs")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", default="", help="run name, used in the report header")
    ap.add_argument("--outputs-a", default="", help="run_model.py outputs on data_test")
    ap.add_argument("--outputs-c", default="", help="run_model.py outputs on data_test_C")
    ap.add_argument("--preds-a", default="", help="predict_batched.py CSV on data_test")
    ap.add_argument("--preds-c", default="", help="predict_batched.py CSV on data_test_C")
    ap.add_argument("--data", default=DEFAULT_DATA, help="data_test, for the age column")
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--durations", default=str(S.DEFAULT_DURATIONS),
                    help="experiments/data/record_durations.csv ('' to skip the S-views)")
    ap.add_argument("--n-boot", type=int, default=S.DEFAULT_N_BOOT)
    ap.add_argument("--seed", type=int, default=S.DEFAULT_SEED)
    ap.add_argument("--bootstrap-unit", default=S.DEFAULT_BOOTSTRAP_UNIT, choices=S.BOOTSTRAP_UNITS,
                    help="resample patients (default; cluster bootstrap) or records")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--from-saved", default="",
                    help="results folder: re-score its saved predictions in place (no --preds/--outputs)")
    ap.add_argument("--saved-suffix", default="", help="with --from-saved: e.g. 'tta' -> predictions_A_tta.csv")
    args = ap.parse_args(argv)
    if args.from_saved:
        return rescore_saved(args)
    if not args.name:
        ap.error("--name is required")

    source_a = args.preds_a or args.outputs_a
    if not source_a:
        ap.error("one of --outputs-a / --preds-a is required")
    source_c = args.preds_c or args.outputs_c

    truth = load_truth(args.split)
    ages = read_ages(args.data)
    durations = S.load_durations(args.durations) if args.durations else None

    view_a = build_view(source_a, truth, ages, durations)
    view_c = None
    if source_c and os.path.exists(source_c):
        view_c = build_view(source_c, truth, ages, durations)

    groups = S.load_groups(args.split) if args.bootstrap_unit == "patient" else None
    report, tables = build_report(args.name, view_a, view_c, n_boot=args.n_boot, seed=args.seed,
                                  unit=args.bootstrap_unit, groups=groups)
    print(report)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(report + "\n")
        for key, table in tables.items():
            if not table.empty:
                table.to_csv(out / f"{key}.csv", index=False)
        view_a.to_csv(out / "predictions_A.csv", index=False)
        if view_c is not None:
            view_c.to_csv(out / "predictions_C.csv", index=False)
        print(f"\nwrote {out}/report.md and the accompanying CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
