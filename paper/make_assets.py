#!/usr/bin/env python
"""Generate every figure, table and in-text number of the manuscript from saved results.

Nothing in ``main.tex`` is typed by hand: numbers come from ``numbers.tex`` (``\\newcommand``
macros), tables from ``tables/*.tex`` and figures from ``figures/*.pdf``, all written here
from the CSVs and manifests under ``experiments/results``.  No model is run.

    python paper/make_assets.py            # then: bash paper/build.sh

Inputs (all produced by committed scripts; see experiments/REPRODUCE.md):
  experiments/data/record_durations.csv      build_durations.py
  <split.csv>                                fixbranch_test/build_split.py
  experiments/results/RESCORED_*.csv         rescore_all.py   (round 1, re-scored)
  experiments/results/R2_TABLE_{A,C}.csv     aggregate.py     (round 2)
  experiments/results/<run>/...              per-run predictions, summaries, manifests
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PAPER = Path(__file__).resolve().parent
WT = PAPER.parent
RES = WT / "experiments" / "results"
sys.path.insert(0, str(WT / "experiments"))
sys.path.insert(0, str(WT))
import scoring  # noqa: E402  (experiments/scoring.py)

DEFAULT_SPLIT = "/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/split/split.csv"

# -- palette: the validated reference palette, first three categorical slots only (the set
# that passes the colour-vision-deficiency checks for all pairs); marker shape is always a
# second encoding so no figure depends on colour alone (IOP figure guidance).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_DARK, BLUE_LIGHT = "#1c5cab", "#86b6ef"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

B0_RUNS = ["finalA_negcap", "lrsweep_lr2e5_s43", "lrsweep_lr2e5_s44"]
SEEDS = (42, 43, 44)
R2_RUNS: Dict[str, List[str]] = {
    "B0": B0_RUNS,
    "M1": [f"r2_M1_s{s}" for s in SEEDS],
    "W1": [f"r2_W1_s{s}" for s in SEEDS],
    "MW_a1.00": [f"r2_MW_s{s}/alpha1.00" for s in SEEDS],
    "MW_a0.50": [f"r2_MW_s{s}/alpha0.50" for s in SEEDS],
    "MW_a0.00": [f"r2_MW_s{s}/alpha0.00" for s in SEEDS],
}
R2_LABEL = {
    "B0": "Baseline (B0)",
    "M1": "+ matched negatives (M1)",
    "W1": "+ 5 s windows (W1)",
    "MW_a1.00": "+ both, fine-tuned ($\\alpha=1$)",
    "MW_a0.50": "+ both, $\\alpha=0.5$",
    "MW_a0.00": "+ both, frozen ($\\alpha=0$)",
}
R2_TEX = {
    "B0": "Baseline (B0)",
    "M1": "+ matched negatives (M1)",
    "W1": "+ 5 s windows (W1)",
    "MW_a1.00": "+ both, fine-tuned ($\\alpha=1$)",
    "MW_a0.50": "+ both, $\\alpha=0.5$",
    "MW_a0.00": "+ both, frozen ($\\alpha=0$)",
}
LADDER = [  # (group in RESCORED_aggregate, runs, figure label, table label)
    ("row0_baseline", ["row0_baseline"], "0  Original entry (frozen encoder)", "0\\quad Original entry (frozen encoder)"),
    ("row1_defect_fixes", ["row1_defect_fixes"], "1  + pipeline fixes, validation split", "1\\quad + pipeline fixes, validation split"),
    ("row2_soft_labels", ["row2_soft_labels"], "2  + soft labels, class weighting", "2\\quad + soft labels, class weighting"),
    ("row3_augmentation", ["row3_augmentation"], "3  + augmentation", "3\\quad + augmentation"),
    ("row4_film", ["row4_film"], "4  + FiLM demographics", "4\\quad + FiLM demographics"),
    ("row5_two_stage_ft", ["row5_two_stage_ft"], "5  + two-stage fine-tuning", "5\\quad + two-stage fine-tuning"),
    ("row5b_no_demographics", ["row5b_no_demographics"], "5b  demographics removed", "5b\\quad demographics removed"),
    ("row5c_bandpass45", ["row5c_bandpass45"], "5c  0.5-45 Hz band-pass", "5c\\quad 0.5--45 Hz band-pass"),
    ("B0 shipped (lr 2e-5)", B0_RUNS, "B0  5c, negatives capped in training (3 seeds)", "B0\\quad 5c, negatives capped in training (3 seeds)"),
    ("full-data final", ["final_full_seed42", "final_full_seed43"], "All negatives, no cap (2 seeds)", "\\phantom{B0}\\quad all negatives, no cap (2 seeds)"),
]
LR_GROUPS = [("2e-5", "B0 shipped (lr 2e-5)", B0_RUNS, "$2\\times10^{-5}$"),
             ("5e-6", "lr 5e-6", [f"lrsweep_lr5e6_s{s}" for s in SEEDS], "$5\\times10^{-6}$"),
             ("1e-6", "lr 1e-6", [f"lrsweep_lr1e6_s{s}" for s in SEEDS], "$1\\times10^{-6}$")]


def style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 8,
        "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2, "text.color": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
        "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03, "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.titlelocation": "left", "axes.titlepad": 6,
    })


# ---------------------------------------------------------------------------
# data access
# ---------------------------------------------------------------------------
def load_predictions(run: str, durations: pd.DataFrame, which: str = "A") -> pd.DataFrame:
    df = pd.read_csv(RES / run / f"predictions_{which}.csv", dtype={"stem": str})
    return scoring.attach_durations(df, durations)


def gap_of(runs: Sequence[str], durations: pd.DataFrame) -> Dict[str, float]:
    rows = [scoring.length_cue_gap(load_predictions(r, durations)) for r in runs]
    out = {k: float(np.mean([r[k] for r in rows])) for k in ("neg_mean_short", "neg_mean_long", "gap")}
    out["gap_sd"] = float(np.std([r["gap"] for r in rows], ddof=1)) if len(rows) > 1 else float("nan")
    return out


def run_views(run: str, rescored: pd.DataFrame) -> pd.DataFrame:
    """Per-view scores of one run on Test A: new-format summary.csv, else the re-scored table."""
    path = RES / run / "summary.csv"
    if path.is_file():
        s = pd.read_csv(path)
        if "S7" in set(s["view"]):
            return s.set_index("view")
    r = rescored[(rescored["run"] == run) & (rescored["predictions"] == "A")]
    if r.empty:
        raise FileNotFoundError(f"no within-length scores for run {run}")
    return r.set_index("view")


def stage2_history(run: str) -> pd.DataFrame:
    m = json.load(open(RES / run / "model" / "chagas_manifest.json"))["training"][0]
    h = [x for x in m["history"] if x["stage"] == 2]
    # the last stage-2 record is the re-validation of the restored best checkpoint
    return pd.DataFrame(h[:-1]).assign(k=lambda d: np.arange(1, len(d) + 1)), m["best_by_stage"]


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def fig_shortcut(split: pd.DataFrame, gaps: Dict[str, Dict[str, float]], out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.1, 2.35), gridspec_kw={"width_ratios": [1.0, 1.25], "wspace": 0.62})
    c15 = split[split["source"].astype(str).str.startswith("CODE")]
    g = c15.groupby("length_group")["label"].agg(["size", "sum"])
    names = [("short", "< 7.5 s\n(7.34 s)"), ("mid", "7.5-9.9 s"), ("long", "≥ 9.9 s\n(10.24 s)")]
    ys = np.arange(len(names))[::-1]
    prev = [100.0 * g.loc[k, "sum"] / g.loc[k, "size"] for k, _ in names]
    ax1.barh(ys, prev, height=0.42, color=BLUE, zorder=3)
    for y, p, (k, _) in zip(ys, prev, names):
        ax1.text(p + 0.08, y, f"{p:.2f}%\n{int(g.loc[k, 'sum']):,} of {int(g.loc[k, 'size']):,}", va="center", ha="left", fontsize=7, color=INK, linespacing=1.25)
    ax1.set_yticks(ys, [n for _, n in names])
    ax1.set_xlim(0, max(prev) * 1.75)
    ax1.set_xlabel("Chagas-positive records (%)")
    ax1.set_title("(a) CODE-15% prevalence by record length")
    ax1.grid(axis="y", visible=False)
    ax1.tick_params(axis="y", length=0)

    order = [("row0", "Original entry"), ("B0", "Round-1 baseline (B0)"), ("M1", "+ matched negatives"), ("W1", "+ 5 s windows"), ("MW_a1.00", "+ both")]
    ys = np.arange(len(order))[::-1]
    for y, (key, _) in zip(ys, order):
        d = gaps[key]
        ax2.plot([d["neg_mean_long"], d["neg_mean_short"]], [y, y], color=AXIS, lw=1.4, zorder=2, solid_capstyle="round")
        ax2.scatter(d["neg_mean_long"], y, s=74, color=BLUE_LIGHT, marker="s", zorder=3, edgecolor="white", linewidth=1.2)
        ax2.scatter(d["neg_mean_short"], y, s=26, color=BLUE_DARK, marker="o", zorder=4, edgecolor="white", linewidth=0.9)
        ax2.text(0.99, y, f"gap {d['gap']:+.2f}", va="center", ha="right", fontsize=7, color=INK2, transform=ax2.get_yaxis_transform())
    ax2.set_yticks(ys, [n for _, n in order])
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(-0.6, len(order) - 0.4)
    ax2.set_xlabel("Mean predicted probability, CODE-15% negatives")
    ax2.set_title("(b) Models score negatives by their length")
    ax2.grid(axis="y", visible=False)
    ax2.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=BLUE_DARK, markeredgecolor="white", markersize=7, label="7.34 s records"),
               Line2D([], [], marker="s", color="none", markerfacecolor=BLUE_LIGHT, markeredgecolor="white", markersize=7, label="10.24 s records")]
    ax2.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.45, -0.3), ncol=2, handletextpad=0.2, columnspacing=1.4)
    fig.savefig(out)
    plt.close(fig)


def fig_inflation(agg: pd.DataFrame, rescored: pd.DataFrame, dur_only_b: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    ys = np.arange(len(LADDER))[::-1]
    series = [("B", "o", BLUE, "Pooled CODE-15% (length cue available)"),
              ("S7", "s", ORANGE, "Within 7.34 s records"),
              ("S10", "^", AQUA, "Within 10.24 s records (95% CI)")]
    for y, (group, runs, _, _) in zip(ys, LADDER):
        a = agg[agg["group"] == group].set_index("view")
        per = pd.concat([rescored[(rescored["run"] == r) & (rescored["predictions"] == "A")] for r in runs])
        lo = per[per["view"] == "S10"]["auroc_lo"].mean()
        hi = per[per["view"] == "S10"]["auroc_hi"].mean()
        ax.plot([lo, hi], [y, y], color=AQUA, lw=1.1, zorder=2, alpha=0.75, solid_capstyle="round")
        for view, marker, color, _ in series:
            ax.scatter(a.loc[view, "auroc_mean"], y, s=40, marker=marker, color=color, zorder=3, edgecolor="white", linewidth=1.1)
    ax.axvline(dur_only_b, color=MUTED, lw=0.9, zorder=1)
    ax.text(dur_only_b + 0.003, len(LADDER) - 0.42, "record length alone", fontsize=7, color=INK2, va="center", ha="left")
    ax.set_yticks(ys, [lab for _, _, lab, _ in LADDER])
    ax.set_ylim(-0.6, len(LADDER) - 0.05)
    ax.set_xlabel("AUROC on held-out CODE-15% records")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    handles = [Line2D([], [], marker=m, color="none", markerfacecolor=c, markeredgecolor="white", markersize=7, label=lab) for _, m, c, lab in series]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.36, -0.14), ncol=3, handletextpad=0.2, columnspacing=1.2)
    fig.savefig(out)
    plt.close(fig)


def fig_stage2(out: Path) -> Dict[str, float]:
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    styles = {"2e-5": (BLUE, "o"), "5e-6": (ORANGE, "s"), "1e-6": (AQUA, "^")}
    best_epoch: Dict[str, float] = {}
    for key, _, runs, _ in LR_GROUPS:
        color, marker = styles[key]
        hs, bests = [], []
        for r in runs:
            h, best = stage2_history(r)
            hs.append(h)
            bests.append(best["stage2"]["epoch"] + 1)
            ax.plot(h["k"], h["val_auroc"], color=color, lw=0.8, alpha=0.45, zorder=2)
        best_epoch[key] = float(np.mean(bests))
        n_common = min(len(h) for h in hs)
        mean = np.mean([h["val_auroc"].to_numpy()[:n_common] for h in hs], axis=0)
        mantissa, exponent = key.split("e-")
        label = "learning rate $" + mantissa + r"\times10^{-" + exponent + "}$"
        ax.plot(np.arange(1, n_common + 1), mean, color=color, lw=1.8, marker=marker, markersize=5.5, markeredgecolor="white", markeredgewidth=1.0, zorder=3,
                label=label)
    ax.set_xlabel("Stage-2 epoch (about 1,400 optimiser steps each)")
    ax.set_ylabel("Validation AUROC")
    ax.set_xticks(range(1, 12))
    ax.legend(loc="lower right")
    fig.savefig(out)
    plt.close(fig)
    return best_epoch


def fig_round2(r2: pd.DataFrame, rescored: pd.DataFrame, out: Path) -> None:
    panels = [("S7", "challenge", "(a) TPR@5%\n7.34 s records"), ("S7", "auroc", "(b) AUROC\n7.34 s records"), ("S10+", "auroc", "(c) AUROC\n10 s format")]
    keys = list(R2_RUNS)
    ys = np.arange(len(keys))[::-1]
    fig, axes = plt.subplots(1, 3, figsize=(6.1, 2.7), sharey=True, gridspec_kw={"wspace": 0.12})
    for ax, (view, metric, title) in zip(axes, panels):
        base = r2[(r2["group"] == "B0") & (r2["view"] == view)].iloc[0]
        ax.axvline(base[f"{metric}_mean"], color=AXIS, lw=0.9, zorder=1)
        for y, key in zip(ys, keys):
            row = r2[(r2["group"] == key) & (r2["view"] == view)].iloc[0]
            seeds = [run_views(r, rescored).loc[view, metric] for r in R2_RUNS[key]]
            ax.scatter(seeds, [y + 0.17] * len(seeds), s=16, color=MUTED, marker="o", zorder=2, linewidth=0)
            ax.scatter(row[f"{metric}_mean"], y + 0.17, s=40, color=BLUE, marker="o", zorder=3, edgecolor="white", linewidth=1.1)
            ax.plot([row[f"ens_{metric}_lo"], row[f"ens_{metric}_hi"]], [y - 0.17, y - 0.17], color=INK2, lw=1.0, zorder=2, solid_capstyle="round")
            ax.scatter(row[f"ens_{metric}"], y - 0.17, s=38, facecolor="white", edgecolor=INK, marker="D", zorder=3, linewidth=1.1)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(-0.6, len(keys) - 0.4)
    axes[0].set_yticks(ys, [R2_LABEL[k] for k in keys])
    handles = [Line2D([], [], marker="o", color="none", markerfacecolor=MUTED, markersize=4, label="single seed"),
               Line2D([], [], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="white", markersize=7, label="mean of 3 seeds"),
               Line2D([], [], marker="D", color=INK2, lw=1.0, markerfacecolor="white", markeredgecolor=INK, markersize=6, label="3-seed ensemble, 95% CI")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.6, -0.1), ncol=3, handletextpad=0.3, columnspacing=1.6)
    fig.savefig(out)
    plt.close(fig)


# ---------------------------------------------------------------------------
# tables and numbers
# ---------------------------------------------------------------------------
def f3(x: float) -> str:
    return f"{x:.3f}"


def pm(d: float, se: float) -> str:
    return f"${d:+.3f} \\pm {se:.3f}$"


def tab_data(split: pd.DataFrame, out: Path) -> None:
    rows = []
    src_names = [("CODE", "CODE-15\\%"), ("SaMi", "SaMi-Trop"), ("PTB", "PTB-XL")]
    grp_names = [("short", "$<7.5$ s"), ("mid", "7.5--9.9 s"), ("long", "$\\geq 9.9$ s")]
    for sk, sn in src_names:
        s = split[split["source"].astype(str).str.startswith(sk)]
        first = True
        for gk, gn in grp_names:
            g = s[s["length_group"] == gk]
            if g.empty:
                continue
            cells = []
            for sp in ("train", "test"):
                h = g[g["split"] == sp]
                cells += [f"{len(h):,}", f"{int(h['label'].sum()):,}"]
            prev = 100.0 * g["label"].mean()
            rows.append(f"{sn if first else ''} & {gn} & {' & '.join(cells)} & {prev:.2f} \\\\")
            first = False
        rows.append("\\hline")
    body = "\n".join(rows)
    out.write_text(
        "\\begin{tabular}{l l r r r r r}\n\\hline\n"
        " & & \\multicolumn{2}{c}{Training split} & \\multicolumn{2}{c}{Held-out split} & Prevalence \\\\\n"
        "Source & Record length & Records & Positive & Records & Positive & (\\%) \\\\\n\\hline\n" + body + "\n\\end{tabular}\n")


def tab_ladder(agg: pd.DataFrame, gaps: Dict[str, Dict[str, float]], out: Path) -> None:
    rows = []
    for group, runs, _, tex in LADDER:
        a = agg[agg["group"] == group].set_index("view")
        g = gaps["ladder:" + group]
        rows.append(f"{tex} & {f3(a.loc['A','challenge_mean'])} & {f3(a.loc['A','auroc_mean'])} & {f3(a.loc['B','auroc_mean'])} & "
                    f"{f3(a.loc['S7','challenge_mean'])} & {f3(a.loc['S7','auroc_mean'])} & {f3(a.loc['S10','auroc_mean'])} & ${g['gap']:+.2f}$ \\\\")
    out.write_text(
        "\\begin{tabular}{l c c c c c c c}\n\\hline\n"
        " & \\multicolumn{2}{c}{Pooled (A)} & Pooled (B) & \\multicolumn{2}{c}{Within 7.34 s (S7)} & 10.24 s (S10) & Length-cue \\\\\n"
        "Configuration & TPR@5\\% & AUROC & AUROC & TPR@5\\% & AUROC & AUROC & gap \\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_lr(agg: pd.DataFrame, best_epoch: Dict[str, float], out: Path) -> None:
    rows = []
    for key, group, _, tex in LR_GROUPS:
        a = agg[agg["group"] == group].set_index("view")
        rows.append(f"{tex} & ${a.loc['A','challenge_mean']:.3f} \\pm {a.loc['A','challenge_sd']:.3f}$ & {f3(a.loc['A','auroc_mean'])} & "
                    f"${a.loc['S7','challenge_mean']:.3f} \\pm {a.loc['S7','challenge_sd']:.3f}$ & {f3(a.loc['S7','auroc_mean'])} & {f3(a.loc['S7','ens_challenge'])} & {best_epoch[key]:.1f} \\\\")
    out.write_text(
        "\\begin{tabular}{l c c c c c c}\n\\hline\n"
        "Stage-2 & \\multicolumn{2}{c}{Pooled (A)} & \\multicolumn{3}{c}{Within 7.34 s (S7)} & Best stage-2 \\\\\n"
        "learning rate & TPR@5\\% & AUROC & TPR@5\\% & AUROC & Ensemble TPR@5\\% & epoch \\\\\n\\hline\n" + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def tab_round2(r2: pd.DataFrame, gaps: Dict[str, Dict[str, float]], out: Path) -> None:
    rows = []
    for key in R2_RUNS:
        v = r2[r2["group"] == key].set_index("view")
        s7, sp, a = v.loc["S7"], v.loc["S10+"], v.loc["A"]
        if key == "B0":
            d1, d2, d3 = "--", "--", "--"
        else:
            d1, d2, d3 = pm(s7["d_challenge"], s7["d_challenge_se"]), pm(s7["d_auroc"], s7["d_auroc_se"]), pm(sp["d_auroc"], sp["d_auroc_se"])
        rows.append(f"{R2_TEX[key]} & {f3(a['challenge_mean'])} & {f3(s7['challenge_mean'])} & {d1} & {f3(s7['auroc_mean'])} & {d2} & {f3(sp['auroc_mean'])} & {d3} & ${gaps[key]['gap']:+.2f}$ \\\\")
    out.write_text(
        "\\begin{tabular}{l c c c c c c c c}\n\\hline\n"
        " & Pooled & \\multicolumn{4}{c}{Within 7.34 s records (S7)} & \\multicolumn{2}{c}{10 s format (S10+)} & Length- \\\\\n"
        "Configuration & TPR@5\\% & TPR@5\\% & $\\Delta$ & AUROC & $\\Delta$ & AUROC & $\\Delta$ & cue gap \\\\\n\\hline\n"
        + "\n".join(rows) + "\n\\hline\n\\end{tabular}\n")


def ptb_mean(run: str) -> float:
    """Mean predicted probability of held-out PTB-XL records (all negative): the source cue."""
    d = pd.read_csv(RES / run / "per_source.csv")
    return float(d[d["source"].astype(str).str.startswith("PTB")]["mean_prob"].iloc[0])


def macro(name: str, value: str) -> str:
    assert name.isalpha(), name
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def numbers(split, dur, agg, r2, r2c, gaps, best_epoch, dur_only, out: Path) -> None:
    c15 = split[split["source"].astype(str).str.startswith("CODE")]
    g = c15.groupby("length_group")["label"].agg(["size", "sum"])
    ps, pl = g.loc["short", "sum"] / g.loc["short", "size"], g.loc["long", "sum"] / g.loc["long", "size"]
    test = split[split["split"] == "test"]
    tc = test[test["source"].astype(str).str.startswith("CODE")]
    A = lambda grp, view, col, t=agg: float(t[(t["group"] == grp) & (t["view"] == view)].iloc[0][col])  # noqa: E731
    R = lambda grp, view, col: A(grp, view, col, r2)  # noqa: E731
    B0g = "B0 shipped (lr 2e-5)"
    m = [
        macro("nAll", f"{len(split):,}"), macro("nTrain", f"{(split['split']=='train').sum():,}"), macro("nTest", f"{len(test):,}"),
        macro("nTestPos", f"{int(test['label'].sum()):,}"), macro("testPrev", f"{100*test['label'].mean():.2f}"),
        macro("nCodeShort", f"{int(g.loc['short','size']):,}"), macro("nCodeLong", f"{int(g.loc['long','size']):,}"),
        macro("nCodeShortPos", f"{int(g.loc['short','sum']):,}"), macro("nCodeLongPos", f"{int(g.loc['long','sum']):,}"),
        macro("nCodePos", f"{int(c15['label'].sum()):,}"),
        macro("prevShort", f"{100*ps:.2f}"), macro("prevLong", f"{100*pl:.2f}"), macro("prevRatio", f"{ps/pl:.1f}"),
        macro("shareCodePosShort", f"{100*g.loc['short','sum']/c15['label'].sum():.1f}"),
        macro("shareAllShort", f"{100*(split['length_group']=='short').mean():.0f}"),
        macro("nSsevenPos", f"{int(tc[tc['length_group']=='short']['label'].sum()):,}"),
        macro("nStenPos", f"{int(tc[tc['length_group']=='long']['label'].sum()):,}"),
        macro("nSseven", f"{int((tc['length_group']=='short').sum()):,}"), macro("nSten", f"{int((tc['length_group']=='long').sum()):,}"),
        macro("nStenPlusPos", f"{int(R('B0','S10+','positives')):,}"),
        macro("durOnlyB", f3(dur_only["B"])), macro("durOnlyA", f3(dur_only["A"])),
        # round-1 ladder
        macro("rowZeroA", f3(A("row0_baseline", "A", "challenge_mean"))), macro("rowZeroAauc", f3(A("row0_baseline", "A", "auroc_mean"))),
        macro("rowZeroSseven", f3(A("row0_baseline", "S7", "challenge_mean"))), macro("rowZeroSsevenAuc", f3(A("row0_baseline", "S7", "auroc_mean"))),
        macro("rowZeroGap", f"${gaps['row0']['gap']:+.2f}$"),
        macro("bzA", f3(A(B0g, "A", "challenge_mean"))), macro("bzAsd", f3(A(B0g, "A", "challenge_sd"))), macro("bzAauc", f3(A(B0g, "A", "auroc_mean"))),
        macro("bzBauc", f3(A(B0g, "B", "auroc_mean"))),
        macro("bzSseven", f3(A(B0g, "S7", "challenge_mean"))), macro("bzSsevenSd", f3(A(B0g, "S7", "challenge_sd"))), macro("bzSsevenAuc", f3(A(B0g, "S7", "auroc_mean"))),
        macro("bzStenAuc", f3(A(B0g, "S10", "auroc_mean"))), macro("bzStenPlusAuc", f3(R("B0", "S10+", "auroc_mean"))),
        macro("bzEnsA", f3(A(B0g, "A", "ens_challenge"))), macro("bzEnsSseven", f3(A(B0g, "S7", "ens_challenge"))), macro("bzEnsSsevenAuc", f3(A(B0g, "S7", "ens_auroc"))),
        macro("bzGap", f"${gaps['B0']['gap']:+.2f}$"), macro("bzNegShort", f"{gaps['B0']['neg_mean_short']:.2f}"), macro("bzNegLong", f"{gaps['B0']['neg_mean_long']:.2f}"),
        macro("fullAuc", f3(A("full-data final", "A", "auroc_mean"))), macro("fullA", f3(A("full-data final", "A", "challenge_mean"))),
        macro("fullSsevenAuc", f3(A("full-data final", "S7", "auroc_mean"))),
        macro("aucInflation", f"{A(B0g,'B','auroc_mean') - A(B0g,'S7','auroc_mean'):.3f}"),
        macro("ladderGainA", f"{A(B0g,'A','challenge_mean') - A('row0_baseline','A','challenge_mean'):.3f}"),
        macro("ptbRowFour", f"{ptb_mean('row4_film'):.2f}"), macro("ptbRowFive", f"{ptb_mean('row5_two_stage_ft'):.2f}"),
        macro("ptbRowFiveB", f"{ptb_mean('row5b_no_demographics'):.2f}"),
        # learning-rate sweep
        macro("lrMidSseven", f3(A("lr 5e-6", "S7", "challenge_mean"))), macro("lrLowSseven", f3(A("lr 1e-6", "S7", "challenge_mean"))),
        macro("lrMidA", f3(A("lr 5e-6", "A", "challenge_mean"))), macro("lrLowA", f3(A("lr 1e-6", "A", "challenge_mean"))),
        macro("bestEpHigh", f"{best_epoch['2e-5']:.1f}"), macro("bestEpMid", f"{best_epoch['5e-6']:.1f}"), macro("bestEpLow", f"{best_epoch['1e-6']:.1f}"),
        # round 2
        macro("mOneA", f3(R("M1", "A", "challenge_mean"))), macro("mOneSseven", f3(R("M1", "S7", "challenge_mean"))), macro("mOneGap", f"${gaps['M1']['gap']:+.2f}$"),
        macro("mOneStenPlusAuc", f3(R("M1", "S10+", "auroc_mean"))),
        macro("wOneA", f3(R("W1", "A", "challenge_mean"))), macro("wOneSseven", f3(R("W1", "S7", "challenge_mean"))), macro("wOneGap", f"${gaps['W1']['gap']:+.2f}$"),
        macro("wOneSsevenAuc", f3(R("W1", "S7", "auroc_mean"))), macro("wOneStenPlusAuc", f3(R("W1", "S10+", "auroc_mean"))),
        macro("mwA", f3(R("MW_a1.00", "A", "challenge_mean"))), macro("mwSseven", f3(R("MW_a1.00", "S7", "challenge_mean"))),
        macro("mwSsevenSd", f3(R("MW_a1.00", "S7", "challenge_sd"))), macro("mwSsevenAuc", f3(R("MW_a1.00", "S7", "auroc_mean"))),
        macro("mwStenAuc", f3(R("MW_a1.00", "S10", "auroc_mean"))), macro("mwStenPlusAuc", f3(R("MW_a1.00", "S10+", "auroc_mean"))),
        macro("mwStenPlus", f3(R("MW_a1.00", "S10+", "challenge_mean"))), macro("bzStenPlus", f3(R("B0", "S10+", "challenge_mean"))),
        macro("mwGap", f"${gaps['MW_a1.00']['gap']:+.2f}$"),
        macro("mOneSsevenSd", f3(R("M1", "S7", "challenge_sd"))), macro("wOneSsevenSd", f3(R("W1", "S7", "challenge_sd"))),
        macro("ensGainSseven", f"{A(B0g,'S7','ens_challenge') - A(B0g,'S7','challenge_mean'):.3f}"),
        macro("mOneDSseven", f"${R('M1','S7','d_challenge'):+.3f}$"), macro("mOneDSsevenSe", f3(R("M1", "S7", "d_challenge_se"))),
        macro("wOneDSseven", f"${R('W1','S7','d_challenge'):+.3f}$"), macro("wOneDSsevenSe", f3(R("W1", "S7", "d_challenge_se"))),
        macro("wOneDSsevenAuc", f"${R('W1','S7','d_auroc'):+.3f}$"), macro("wOneDSsevenAucSe", f3(R("W1", "S7", "d_auroc_se"))),
        macro("frozenDSseven", f"${R('MW_a0.00','S7','d_challenge'):+.3f}$"),
        macro("mwDSseven", f"${R('MW_a1.00','S7','d_challenge'):+.3f}$"), macro("mwDSsevenSe", f3(R("MW_a1.00", "S7", "d_challenge_se"))),
        macro("mwDSsevenAuc", f"${R('MW_a1.00','S7','d_auroc'):+.3f}$"), macro("mwDSsevenAucSe", f3(R("MW_a1.00", "S7", "d_auroc_se"))),
        macro("mwDStenPlusAuc", f"${R('MW_a1.00','S10+','d_auroc'):+.3f}$"), macro("mwDStenPlusAucSe", f3(R("MW_a1.00", "S10+", "d_auroc_se"))),
        macro("wOneDStenPlusAuc", f"${R('W1','S10+','d_auroc'):+.3f}$"), macro("wOneDStenPlusAucSe", f3(R("W1", "S10+", "d_auroc_se"))),
        macro("mOneDStenPlusAuc", f"${R('M1','S10+','d_auroc'):+.3f}$"), macro("mOneDStenPlusAucSe", f3(R("M1", "S10+", "d_auroc_se"))),
        macro("mwEnsSseven", f3(R("MW_a1.00", "S7", "ens_challenge"))), macro("mwEnsStenPlusAuc", f3(R("MW_a1.00", "S10+", "ens_auroc"))),
        macro("bzEnsStenPlusAuc", f3(R("B0", "S10+", "ens_auroc"))),
        macro("frozenSseven", f3(R("MW_a0.00", "S7", "challenge_mean"))), macro("frozenSsevenAuc", f3(R("MW_a0.00", "S7", "auroc_mean"))),
        macro("frozenStenPlusAuc", f3(R("MW_a0.00", "S10+", "auroc_mean"))),
        macro("halfSseven", f3(R("MW_a0.50", "S7", "challenge_mean"))), macro("halfSsevenAuc", f3(R("MW_a0.50", "S7", "auroc_mean"))),
        macro("ftGainSseven", f"{R('MW_a1.00','S7','challenge_mean') - R('MW_a0.00','S7','challenge_mean'):.3f}"),
        # perturbed copy (Test C)
        macro("bzSsevenC", f3(A(B0g, "S7", "challenge_mean", pd.read_csv(RES / "RESCORED_aggregate_C.csv")))),
        macro("mwSsevenC", f3(float(r2c[(r2c["group"] == "MW_a1.00") & (r2c["view"] == "S7")].iloc[0]["challenge_mean"]))),
    ]
    out.write_text("% generated by paper/make_assets.py -- do not edit\n" + "\n".join(m) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    args = ap.parse_args()
    style()
    (PAPER / "figures").mkdir(exist_ok=True)
    (PAPER / "tables").mkdir(exist_ok=True)

    durations = scoring.load_durations()
    split = scoring.attach_durations(pd.read_csv(args.split, dtype={"stem": str}), durations)
    rescored = pd.read_csv(RES / "RESCORED_runs.csv")
    agg = pd.read_csv(RES / "RESCORED_aggregate_A.csv")
    for name in ("R2_TABLE_A.csv", "R2_TABLE_C.csv"):
        if not (RES / name).is_file():
            raise SystemExit(f"missing {RES / name}: run `sbatch experiments/batch_r2_table.sh` first")
    r2, r2c = pd.read_csv(RES / "R2_TABLE_A.csv"), pd.read_csv(RES / "R2_TABLE_C.csv")

    gaps: Dict[str, Dict[str, float]] = {key: gap_of(runs, durations) for key, runs in R2_RUNS.items()}
    gaps["row0"] = gap_of(["row0_baseline"], durations)
    for group, runs, _, _ in LADDER:
        gaps["ladder:" + group] = gap_of(runs, durations)
    ref = load_predictions(B0_RUNS[0], durations)
    dur_only = {v: float(scoring.duration_only(ref, v)["auroc"]) for v in ("A", "B")}

    fig_shortcut(split, gaps, PAPER / "figures" / "fig_shortcut.pdf")
    fig_inflation(agg, rescored, dur_only["B"], PAPER / "figures" / "fig_inflation.pdf")
    best_epoch = fig_stage2(PAPER / "figures" / "fig_stage2.pdf")
    fig_round2(r2, rescored, PAPER / "figures" / "fig_round2.pdf")

    tab_data(split, PAPER / "tables" / "tab_data.tex")
    tab_ladder(agg, gaps, PAPER / "tables" / "tab_ladder.tex")
    tab_lr(agg, best_epoch, PAPER / "tables" / "tab_lr.tex")
    tab_round2(r2, gaps, PAPER / "tables" / "tab_round2.tex")
    numbers(split, durations, agg, r2, r2c, gaps, best_epoch, dur_only, PAPER / "numbers.tex")
    print("wrote figures/, tables/, numbers.tex")


if __name__ == "__main__":
    main()
