#!/usr/bin/env python
"""Build ``experiments/data/record_durations.csv`` from the WFDB headers (PLAN_R2.md section 2).

One row per record in ``split.csv``: ``stem, n_samples, fs, duration_s``, read from the
record line of each record's own ``.hea`` file (:func:`chagas.lengths.parse_header_length`),
then cross-checked **in full** against the team's earlier
``physionet2025/signal_lengths.csv`` (``record_name, signal_length, original_fs``, which
was produced by loading every signal with ``helper_code.load_signals``).  Any mismatch
or missing record is printed and makes the script exit non-zero, so the CSV is only
written when the two independent sources agree on every record.

Why: CODE-15 record length predicts the label (DECISIONS.md D-2026-09-30-2), so every
evaluation view in round 2 is defined by duration.  Reading ~366k headers over NFS takes
minutes, so this runs as a CPU-only SLURM job (``experiments/batch_build_durations.sh``).

    python experiments/build_durations.py \
        --split  .../fixbranch_test/split/split.csv \
        --signal-lengths .../physionet2025/signal_lengths.csv \
        --out    experiments/data/record_durations.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chagas.lengths import parse_header_length  # noqa: E402

DEFAULT_SPLIT = "/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/split/split.csv"
DEFAULT_SIGNAL_LENGTHS = "/juice2/scr2/kelvinkn/other_work/edwards/physionet2025/signal_lengths.csv"
DEFAULT_OUT = REPO_ROOT / "experiments" / "data" / "record_durations.csv"
COLUMNS = ["stem", "n_samples", "fs", "duration_s"]


def read_header_row(record_path: str) -> Tuple[str, Optional[int], Optional[float]]:
    """``(stem, n_samples, fs)`` for one record path (no extension); ``None`` on failure."""
    stem = Path(record_path).name
    try:
        with open(record_path + ".hea", "r") as fh:
            n_samples, fs = parse_header_length(fh.read())
    except (OSError, ValueError):
        return stem, None, None
    return stem, int(n_samples), float(fs)


def build_durations(record_paths: Sequence[str], workers: int = 32) -> pd.DataFrame:
    """Read every header; return the durations table sorted by stem.

    Records whose header cannot be read keep NaN in ``n_samples``/``fs``/``duration_s``
    so they are visible to :func:`cross_check` rather than silently dropped.
    """
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        rows = list(pool.map(read_header_row, record_paths, chunksize=256))
    df = pd.DataFrame(rows, columns=["stem", "n_samples", "fs"])
    df["n_samples"] = df["n_samples"].astype("Int64")
    df["duration_s"] = df["n_samples"].astype(float) / df["fs"].astype(float)
    return df.sort_values("stem", kind="mergesort").reset_index(drop=True)[COLUMNS]


def cross_check(durations: pd.DataFrame, signal_lengths: pd.DataFrame) -> Dict[str, object]:
    """Compare header-derived lengths with ``signal_lengths.csv`` record by record."""
    ref = signal_lengths.rename(
        columns={"record_name": "stem", "signal_length": "ref_n", "original_fs": "ref_fs"}
    )
    ref["stem"] = ref["stem"].astype(str)
    merged = durations.merge(ref, on="stem", how="left", indicator=True)
    missing_ref = merged["_merge"] == "left_only"
    unreadable = merged["n_samples"].isna()
    both = ~missing_ref & ~unreadable
    n_mismatch = int(
        (merged.loc[both, "n_samples"].astype(int) != merged.loc[both, "ref_n"].astype(int)).sum()
    )
    fs_mismatch = int(
        (~np.isclose(merged.loc[both, "fs"].astype(float), merged.loc[both, "ref_fs"].astype(float))).sum()
    )
    bad = merged[both & (
        (merged["n_samples"].astype(float) != merged["ref_n"].astype(float))
        | (~np.isclose(merged["fs"].astype(float), merged["ref_fs"].astype(float)))
    )]
    return {
        "n_records": int(len(durations)),
        "unreadable_headers": int(unreadable.sum()),
        "missing_from_signal_lengths": int(missing_ref.sum()),
        "n_samples_mismatches": n_mismatch,
        "fs_mismatches": fs_mismatch,
        "examples": bad.head(10)[["stem", "n_samples", "fs", "ref_n", "ref_fs"]].to_dict("records"),
        "ok": bool(not unreadable.any() and not missing_ref.any() and n_mismatch == 0 and fs_mismatch == 0),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default=DEFAULT_SPLIT, help="split.csv with a 'path' column")
    ap.add_argument("--signal-lengths", default=DEFAULT_SIGNAL_LENGTHS)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args(argv)

    started = time.time()
    split = pd.read_csv(args.split, low_memory=False, dtype={"stem": str})
    paths: List[str] = split["path"].astype(str).tolist()
    print(f"[durations] reading {len(paths)} headers with {args.workers} threads", flush=True)
    durations = build_durations(paths, args.workers)
    print(f"[durations] read in {time.time() - started:.0f} s", flush=True)

    reference = pd.read_csv(args.signal_lengths, dtype={"record_name": str})
    report = cross_check(durations, reference)
    for key, value in report.items():
        print(f"[durations] {key}: {value}", flush=True)
    if not report["ok"]:
        print("[durations] FAIL: headers and signal_lengths.csv disagree; nothing written", flush=True)
        return 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    durations.to_csv(out, index=False)
    print(f"[durations] wrote {out} ({len(durations)} rows)", flush=True)
    print(durations["duration_s"].describe().to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
