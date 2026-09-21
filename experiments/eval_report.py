#!/usr/bin/env python
"""Score one inference run three ways and print the diagnostics PLAN.md section 2 asks for.

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

Inputs are ``run_model.py``-style output trees; a record with no parsable probability is
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
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chagas.metrics import challenge_score, compute_auc  # noqa: E402

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


def build_view(outputs_dir: str, truth: pd.DataFrame, ages: pd.DataFrame) -> pd.DataFrame:
    preds = read_outputs(outputs_dir)
    df = truth.merge(preds, on="stem", how="left").merge(ages, on="stem", how="left")
    # Records that produced no output, or a nan probability, rank last.
    df["prob_filled"] = df["prob"].fillna(-1.0)
    return df


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def _fmt(rows: Sequence[Dict[str, object]]) -> str:
    df = pd.DataFrame(rows)
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="ladder row name, used in the report header")
    ap.add_argument("--outputs-a", required=True, help="run_model.py outputs on data_test")
    ap.add_argument("--outputs-c", default="", help="run_model.py outputs on data_test_C")
    ap.add_argument("--data", default=DEFAULT_DATA, help="data_test, for the age column")
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args(argv)

    truth = load_truth(args.split)
    ages = read_ages(args.data)

    view_a = build_view(args.outputs_a, truth, ages)
    summary = [
        score(view_a, "A  full held-out (source shortcut AVAILABLE)"),
        score(view_a[view_a["source"].str.startswith("CODE")],
              "B  CODE-15 only (source shortcut REMOVED)"),
    ]

    view_c = None
    if args.outputs_c and os.path.isdir(args.outputs_c):
        view_c = build_view(args.outputs_c, truth, ages)
        summary.append(score(view_c, "C  perturbed as the organisers perturbed the hidden sets"))

    lines: List[str] = []
    lines.append(f"# Evaluation report -- {args.name}")
    lines.append("")
    lines.append("## Views")
    lines.append("```")
    lines.append(_fmt(summary))
    lines.append("```")

    a_score = summary[0].get("challenge")
    if view_c is not None and isinstance(a_score, float):
        c_score = summary[-1].get("challenge")
        if isinstance(c_score, float):
            gap = c_score - a_score
            lines.append("")
            lines.append(
                f"**C - A gap: {gap:+.4f}** "
                f"({'robust' if gap > -0.02 else 'degrades out of distribution'}); "
                f"A = {a_score:.4f}, C = {c_score:.4f}."
            )

    lines.append("")
    lines.append("## Per-source mean predicted probability (source-shortcut diagnostic)")
    lines.append("```")
    lines.append(by_source(view_a).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    lines.append("```")

    decile = by_age_decile(view_a)
    if not decile.empty:
        lines.append("")
        lines.append("## Score by age decile (age-shortcut diagnostic)")
        lines.append("```")
        lines.append(decile.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        lines.append("```")

    report = "\n".join(lines)
    print(report)

    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.md").write_text(report + "\n")
        pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)
        by_source(view_a).to_csv(out / "per_source.csv", index=False)
        if not decile.empty:
            decile.to_csv(out / "per_age_decile.csv", index=False)
        view_a.to_csv(out / "predictions_A.csv", index=False)
        if view_c is not None:
            view_c.to_csv(out / "predictions_C.csv", index=False)
        print(f"\nwrote {out}/report.md and the accompanying CSVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
