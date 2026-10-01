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
