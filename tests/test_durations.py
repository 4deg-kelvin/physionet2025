"""PLAN_R2.md section 7 item 1: record durations come from the headers and match signal_lengths.csv."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chagas.lengths import header_duration, length_group, parse_header_length

SIGNAL_LENGTHS = Path("/juice2/scr2/kelvinkn/other_work/edwards/physionet2025/signal_lengths.csv")
SPLIT = Path("/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/split/split.csv")


def test_parse_header_length_formats():
    assert parse_header_length("1000010 12 400 4096\n1000010.mat 16 ...") == (4096, 400.0)
    assert parse_header_length("00001_hr 12 500 5000 09:17:34 09/11/1984\n") == (5000, 500.0)
    # counter frequency and base counter value are allowed after the sampling rate
    assert parse_header_length("# comment first\nrec 12 500/2(0) 2500\n") == (2500, 500.0)
    with pytest.raises(ValueError):
        parse_header_length("rec 12 500\n")  # no sample count: refuse to guess
    assert np.isnan(header_duration("garbage"))
    assert header_duration("r 12 400 2934") == pytest.approx(7.335)


def test_length_group_thresholds():
    assert length_group(7.335) == "short"
    assert length_group(10.24) == "long" and length_group(10.0) == "long" and length_group(9.9) == "long"
    assert length_group(8.0) == "mid" and length_group(7.5) == "mid"
    assert length_group(float("nan")) == "unknown"


def test_build_durations_equals_wfdb_headers_and_signal_lengths(real_records):
    """Our parser vs wfdb's own header reader on 100 smoke records, and vs signal_lengths.csv."""
    import wfdb

    import build_durations as B

    sample = real_records[::4]
    df = B.build_durations(sample, workers=4).set_index("stem")
    assert len(df) == len(sample) and df["n_samples"].notna().all()
    for path in sample:
        header = wfdb.rdheader(path)
        row = df.loc[Path(path).name]
        assert int(row["n_samples"]) == int(header.sig_len), path
        assert float(row["fs"]) == pytest.approx(float(header.fs)), path
        assert float(row["duration_s"]) == pytest.approx(header.sig_len / header.fs), path
    if not SIGNAL_LENGTHS.is_file():
        pytest.skip("signal_lengths.csv not available")
    report = B.cross_check(df.reset_index(), pd.read_csv(SIGNAL_LENGTHS, dtype={"record_name": str}))
    assert report["ok"], report


def test_committed_durations_file_matches_headers():
    """The committed experiments/data/record_durations.csv, spot-checked on 300 records."""
    import build_durations as B

    if not B.DEFAULT_OUT.is_file() or not SPLIT.is_file():
        pytest.skip("record_durations.csv or split.csv not available")
    durations = pd.read_csv(B.DEFAULT_OUT, dtype={"stem": str}).set_index("stem")
    assert len(durations) == 366_185
    assert durations["duration_s"].notna().all()
    split = pd.read_csv(SPLIT, low_memory=False, dtype={"stem": str}).sample(300, random_state=0)
    fresh = B.build_durations(split["path"].tolist(), workers=8).set_index("stem")
    committed = durations.loc[fresh.index]
    assert (committed["n_samples"].astype(int) == fresh["n_samples"].astype(int)).all()
    assert np.allclose(committed["fs"].astype(float), fresh["fs"].astype(float))
