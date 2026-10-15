"""``experiments/check_split.py``: leakage checks and composition of the held-out split."""

from __future__ import annotations

import pandas as pd
import pytest

import check_split as CS


def _frames():
    split = pd.DataFrame({
        "path": [f"/d/{s}" for s in ("001", "002", "003", "004", "00005_hr", "00006_hr", "900")],
        "stem": ["001", "002", "003", "004", "00005_hr", "00006_hr", "900"],
        "label": [1, 0, 0, 0, 0, 0, 1],
        "source": ["CODE-15%", "CODE-15%", "CODE-15%", "CODE-15%", "PTB-XL", "PTB-XL", "SaMi-Trop"],
        "group": ["c15_1", "c15_1", "c15_2", "c15_3", "ptb_7", "ptb_8", "smt_900"],
        "split": ["train", "train", "test", "test", "train", "test", "test"],
    })
    durations = pd.DataFrame({
        "stem": ["001", "002", "003", "004", "00005_hr", "00006_hr", "900"],
        "n_samples": [2934, 4096, 2934, 4096, 5000, 5000, 4000],
        "fs": [400.0, 400.0, 400.0, 400.0, 500.0, 500.0, 400.0],
        "duration_s": [7.335, 10.24, 7.335, 10.24, 10.0, 10.0, 10.0],
    })
    return split, durations


def test_clean_split_passes_and_is_tabulated():
    split, durations = _frames()
    table, checks = CS.audit(split, durations)
    assert checks["ok"] and checks["groups_in_both_splits"] == 0 and checks["duplicate_stems"] == 0
    rows = table.set_index(["source", "split", "length_group"])
    assert rows.loc[("CODE-15%", "train", "short"), "n_pos"] == 1
    assert rows.loc[("CODE-15%", "train", "all"), "n"] == 2
    assert rows.loc[("CODE-15%", "train", "all"), "groups"] == 1          # one patient, two exams
    assert rows.loc[("CODE-15%", "test", "long"), "prevalence"] == 0.0
    assert rows.loc[("SaMi-Trop", "test", "long"), "prevalence"] == 1.0
    assert rows.loc[("all", "test", "all"), "n"] == 4
    assert rows.loc[("all", "test", "all"), "prevalence"] == pytest.approx(0.25)
    md = CS.to_markdown(table, checks, "split.csv", "durations.csv")
    assert md.startswith("# Split audit: PASS")


def test_leaked_group_duplicate_stem_and_missing_duration_fail(tmp_path):
    split, durations = _frames()
    split.loc[2, "group"] = "c15_1"                                         # patient 1 now in both splits
    split = pd.concat([split, split.iloc[[3]]], ignore_index=True)         # stem 004 twice
    durations = durations[durations["stem"] != "900"]                       # no duration for 900
    table, checks = CS.audit(split, durations)
    assert not checks["ok"]
    assert checks["groups_in_both_splits_examples"] == ["c15_1"]
    assert checks["duplicate_stems_examples"] == ["004"]
    assert checks["records_without_duration_examples"] == ["900"]
    assert table.set_index(["source", "split", "length_group"]).loc[("SaMi-Trop", "test", "unknown"), "n"] == 1

    split_csv, dur_csv = tmp_path / "split.csv", tmp_path / "durations.csv"
    split.to_csv(split_csv, index=False)
    durations.to_csv(dur_csv, index=False)
    out = tmp_path / "audit" / "SPLIT_AUDIT"
    assert CS.main(["--split", str(split_csv), "--durations", str(dur_csv), "--out", str(out)]) == 1
    assert out.with_suffix(".md").read_text().startswith("# Split audit: FAIL")
    assert pd.read_csv(out.with_suffix(".csv"))["n"].sum() > 0
    # stems keep their leading zeros through the CSV round trip
    assert "00005_hr" in set(CS.load_split(str(split_csv))["stem"])
