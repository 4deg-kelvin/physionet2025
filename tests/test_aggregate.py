"""PLAN_R2.md section 7 item 7: ``experiments/aggregate.py``."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import aggregate as AG


def _write_run(results: Path, name: str, probs: np.ndarray, frame: pd.DataFrame) -> None:
    run = results / name
    run.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out["prob"] = probs
    out["prob_filled"] = probs
    out.to_csv(run / "predictions_A.csv", index=False)


@pytest.fixture()
def toy_results(tmp_path):
    rng = np.random.default_rng(0)
    n = 600
    durations = np.where(rng.random(n) < 0.6, 7.335, 10.24)
    labels = (rng.random(n) < np.where(durations < 7.5, 0.08, 0.03)).astype(int)
    frame = pd.DataFrame({
        "stem": [f"r{i:04d}" for i in range(n)],
        "label": labels,
        "source": np.where(rng.random(n) < 0.9, "CODE-15%", "PTB-XL"),
    })
    frame.loc[frame["source"] == "PTB-XL", "label"] = 0
    results = tmp_path / "results"
    signal = labels * 0.8 + rng.normal(0, 0.6, n)
    _write_run(results, "a", 1 / (1 + np.exp(-signal)), frame)
    _write_run(results, "a_copy", 1 / (1 + np.exp(-signal)), frame)
    _write_run(results, "b", 1 / (1 + np.exp(-(signal + rng.normal(0, 0.5, n)))), frame)
    dur = tmp_path / "durations.csv"
    pd.DataFrame({"stem": frame["stem"], "n_samples": 0, "fs": 400.0, "duration_s": durations}).to_csv(dur, index=False)
    return results, dur


def test_ensemble_of_identical_models_equals_the_single_model(toy_results):
    results, dur = toy_results
    frames = [AG.load_run(results, r, "A", None) for r in ("a", "a_copy", "a")]
    ens = AG.mean_ensemble(frames)
    assert np.array_equal(ens["prob"].to_numpy(), frames[0]["prob"].to_numpy())
    table, _gaps, _rule, _md = AG.run(["single=a", "triple=a,a_copy,a"], "single",
                                      results_dir=results, durations_path=str(dur), n_boot=100)
    for view in ("A", "B", "S7"):
        rows = table[table["view"] == view].set_index("group")
        assert rows.loc["triple", "ens_challenge"] == rows.loc["single", "ens_challenge"]
        assert rows.loc["triple", "ens_auroc"] == rows.loc["single", "ens_auroc"]
        assert rows.loc["triple", "challenge_sd"] == pytest.approx(0.0, abs=1e-12)  # np.std rounding


def test_paired_difference_of_a_model_with_itself_is_exactly_zero(toy_results):
    results, dur = toy_results
    table, gaps, _rule, md = AG.run(["base=a", "same=a_copy", "other=b"], "base",
                                    results_dir=results, durations_path=str(dur), n_boot=200)
    same = table[table["group"] == "same"]
    for col in ("d_challenge", "d_auroc", "d_ens_challenge", "d_ens_auroc",
                "d_challenge_se", "d_auroc_se", "d_challenge_lo", "d_challenge_hi"):
        assert (same[col] == 0.0).all(), col
    other = table[(table["group"] == "other") & (table["view"] == "A")].iloc[0]
    assert other["d_auroc_se"] > 0
    assert "## S7" in md and "Length-cue gap" in md
    assert set(gaps["group"]) == {"base", "same", "other"}


def test_unaligned_runs_are_refused(toy_results, tmp_path):
    results, dur = toy_results
    frame = pd.read_csv(results / "a" / "predictions_A.csv").iloc[:-1]
    (results / "short").mkdir()
    frame.to_csv(results / "short" / "predictions_A.csv", index=False)
    with pytest.raises(ValueError):
        AG.run(["base=a", "x=short"], "base", results_dir=results, durations_path=str(dur), n_boot=10)


def test_group_spec_parsing():
    assert AG.parse_group("M1=r2_M1_s42, r2_M1_s43") == ("M1", ["r2_M1_s42", "r2_M1_s43"])
    with pytest.raises(ValueError):
        AG.parse_group("no_equals_sign")


def test_run_suffix_reads_the_tta_predictions(toy_results):
    results, dur = toy_results
    tta = pd.read_csv(results / "b" / "predictions_A.csv")
    tta.to_csv(results / "a" / "predictions_A_tta.csv", index=False)
    assert AG.predictions_file(results, "a:tta", "C") == results / "a" / "predictions_C_tta.csv"
    assert AG.predictions_file(results, "a", "A") == results / "a" / "predictions_A.csv"
    table, _gaps, _rule, _md = AG.run(["b=b", "a_tta=a:tta", "a=a"], "b",
                                      results_dir=results, durations_path=str(dur), n_boot=50)
    rows = table[table["view"] == "A"].set_index("group")
    assert rows.loc["a_tta", "d_challenge"] == 0.0 and rows.loc["a_tta", "d_auroc"] == 0.0
    assert rows.loc["a", "auroc_mean"] != rows.loc["a_tta", "auroc_mean"]


def _rule_rows(s7_d, s7_se, s10_d, s10_se):
    rows = []
    for group, (d7, d10) in {"base": (0.0, 0.0), "x": (s7_d, s10_d)}.items():
        rows.append({"group": group, "view": "S7", "d_challenge": d7, "d_challenge_se": s7_se,
                     "d_auroc": 0.0, "d_auroc_se": 0.01})
        rows.append({"group": group, "view": "S10+", "d_challenge": 0.0, "d_challenge_se": 0.01,
                     "d_auroc": d10, "d_auroc_se": s10_se})
    gaps = pd.DataFrame({"group": ["base", "x"], "gap_mean": [0.17, 0.05]})
    return pd.DataFrame(rows), gaps


@pytest.mark.parametrize(
    "s7_d, s10_d, primary, secondary",
    [
        (0.000, 0.050, True, False),    # S10+ up 5 SE, S7 flat
        (-0.006, 0.050, False, False),  # S10+ up but S7 down by more than 1 SE
        (0.010, 0.000, False, True),    # S7 up 2 SE, S10+ flat
        (0.010, -0.020, False, False),  # S7 up but S10+ down by more than 1 SE
        (0.010, 0.050, True, True),
    ],
)
def test_round_three_rule(s7_d, s10_d, primary, secondary):
    rows, gaps = _rule_rows(s7_d, 0.005, s10_d, 0.010)
    out = AG.decision_rule_r3(rows, gaps, "base").set_index("group")
    assert list(out.index) == ["x"]
    assert out.loc["x", "primary"] == primary and out.loc["x", "secondary"] == secondary
    assert out.loc["x", "qualifies"] == (primary or secondary)
    assert out.loc["x", "gap_shrinks"]


def test_rule_option_defaults_to_round_two(toy_results):
    results, dur = toy_results
    _t, _g, r2, md2 = AG.run(["a=a", "b=b"], "a", results_dir=results, durations_path=str(dur), n_boot=20)
    _t, _g, r3, md3 = AG.run(["a=a", "b=b"], "a", results_dir=results, durations_path=str(dur), n_boot=20,
                             rule="r3")
    assert "S7_auroc_gain_gt_0.010_tpr_not_lower" in r2.columns and "primary" in r3.columns
    assert "PLAN_R2.md §5" in md2 and "Round-3 decision rule" in md3
    with pytest.raises(ValueError):
        AG.run(["a=a"], "a", results_dir=results, durations_path=str(dur), n_boot=5, rule="r9")
