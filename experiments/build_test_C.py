#!/usr/bin/env python
"""Build **Test C**: the held-out set perturbed the way the organisers perturbed the
hidden cohorts (PLAN.md section 2).

    "we added small amounts of various forms of noise, applied filters that are
    representative of different ECG devices, and resampled the data to different
    sampling frequencies"
    -- Reyna et al., *Detection of Chagas Disease from the ECG: the George B. Moody
       PhysioNet Challenge 2025*, CinC 2025 / arXiv:2510.02202, section 2.2

Every team's in-distribution score collapsed out-of-distribution (median hidden
validation 0.279 -> ELSA-Brasil 0.100).  Test C is a direct, local estimate of that
collapse: it is Test A, record for record, with the documented perturbations applied
under a fixed seed, written back out as WFDB so ``run_model.py`` and
``evaluate_model.py`` consume it unchanged.

Per record (seeded from the record's own stem, so the perturbation is deterministic and
independent of the processing order):

* resample to a random fs in {300, 500, 600, 1000} Hz and back (``resample_poly``);
* a random "device" filter: Butterworth high-pass 0.05-0.67 Hz and/or low-pass
  40-150 Hz, order 1-2;
* with p=0.5 each: Gaussian noise at SNR 20-35 dB, baseline wander at 0.1-0.5 Hz with
  amplitude <= 0.1 of the lead standard deviation, and 50/60 Hz powerline at SNR
  15-30 dB.

Usage
-----
    python experiments/build_test_C.py \
        --source /juice2/.../fixbranch_test/data_test \
        --dest   /juice2/.../fixbranch_test/data_test_C \
        --workers 16
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SOURCE = "/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/data_test"
DEFAULT_DEST = "/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/data_test_C"
SEED = 20250921


@dataclass(frozen=True)
class PerturbCfg:
    """Exactly the perturbations PLAN.md section 2 specifies for Test C."""

    resample_rates: Tuple[int, ...] = (300, 500, 600, 1000)
    hp_hz: Tuple[float, float] = (0.05, 0.67)
    lp_hz: Tuple[float, float] = (40.0, 150.0)
    order: Tuple[int, int] = (1, 2)
    p_gaussian: float = 0.5
    gaussian_snr_db: Tuple[float, float] = (20.0, 35.0)
    p_wander: float = 0.5
    wander_hz: Tuple[float, float] = (0.1, 0.5)
    wander_amp: float = 0.1
    p_powerline: float = 0.5
    powerline_freqs: Tuple[float, ...] = (50.0, 60.0)
    powerline_snr_db: Tuple[float, float] = (15.0, 30.0)


def record_seed(stem: str, base_seed: int = SEED) -> int:
    """Deterministic per-record seed derived from the record name.

    Derived from the stem rather than from a counter so the perturbation of a record
    does not depend on how many workers ran or in what order.
    """
    h = 1469598103934665603
    for byte in stem.encode("utf-8"):
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return (h ^ (base_seed & 0xFFFFFFFFFFFFFFFF)) % (2 ** 63 - 1)


def _resample(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if fs_in == fs_out:
        return x
    ratio = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)


def _filtfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    padlen = 3 * (2 * sos.shape[0] + 1)
    if x.shape[-1] <= padlen + 1:
        return x
    return np.ascontiguousarray(sosfiltfilt(sos, x, axis=-1))


def perturb(
    signal: np.ndarray, fs: float, rng: np.random.Generator, cfg: PerturbCfg = PerturbCfg()
) -> np.ndarray:
    """Apply the Test-C perturbation to a ``(leads, samples)`` array at ``fs``.

    The returned array has the same shape and sampling rate as the input: the resample
    round trip models the organisers' rate changes without changing the file's declared
    ``fs``, which keeps the header valid and keeps Test C aligned record-for-record with
    Test A.
    """
    x = np.ascontiguousarray(np.asarray(signal, dtype=np.float64))
    n = x.shape[-1]

    # 1. rate round trip
    target = float(cfg.resample_rates[int(rng.integers(len(cfg.resample_rates)))])
    if target != fs:
        y = _resample(x, fs, target)
        if y.shape[-1] >= 2:
            x = _resample(y, target, fs)
            if x.shape[-1] >= n:
                x = x[:, :n]
            else:
                x = np.pad(x, ((0, 0), (0, n - x.shape[-1])), mode="edge")

    # 2. device filter
    nyq = fs / 2.0
    n_ord = int(rng.integers(cfg.order[0], cfg.order[1] + 1))
    f_hp = float(rng.uniform(*cfg.hp_hz))
    f_lp = min(float(rng.uniform(*cfg.lp_hz)), 0.95 * nyq)
    if 0 < f_hp < nyq:
        x = _filtfilt(butter(n_ord, f_hp / nyq, btype="highpass", output="sos"), x)
    if 0 < f_lp < nyq:
        x = _filtfilt(butter(n_ord, f_lp / nyq, btype="lowpass", output="sos"), x)

    power = float(np.mean(x ** 2))
    t = np.arange(n, dtype=np.float64) / float(fs)

    # 3. Gaussian noise
    if rng.random() < cfg.p_gaussian and power > 0:
        snr = float(rng.uniform(*cfg.gaussian_snr_db))
        x = x + rng.normal(0.0, np.sqrt(power / (10.0 ** (snr / 10.0))), size=x.shape)

    # 4. baseline wander, per-lead amplitude relative to that lead's std
    if rng.random() < cfg.p_wander:
        f = float(rng.uniform(*cfg.wander_hz))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        lead_std = np.std(x, axis=1, keepdims=True)
        x = x + cfg.wander_amp * lead_std * np.sin(2.0 * np.pi * f * t + phase)[None, :]

    # 5. powerline
    if rng.random() < cfg.p_powerline and power > 0:
        f0 = float(cfg.powerline_freqs[int(rng.integers(len(cfg.powerline_freqs)))])
        snr = float(rng.uniform(*cfg.powerline_snr_db))
        amp = float(np.sqrt(power / (10.0 ** (snr / 10.0))))
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        x = x + amp * np.sin(2.0 * np.pi * f0 * t + phase)[None, :]

    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# WFDB I/O
# ---------------------------------------------------------------------------
def _header_comment_lines(header_text: str) -> List[str]:
    return [ln for ln in header_text.split("\n") if ln.startswith("#")]


def write_record(
    src_stem: str, dst_dir: str, cfg: PerturbCfg = PerturbCfg(), base_seed: int = SEED
) -> Optional[str]:
    """Perturb one record and write it into ``dst_dir`` as WFDB.  Returns the new stem."""
    import wfdb

    stem = os.path.basename(src_stem)
    with open(src_stem + ".hea", "r") as fh:
        header_text = fh.read()
    signal, fields = wfdb.rdsamp(src_stem)
    fs = float(fields["fs"])
    x = np.asarray(signal, dtype=np.float64).T  # (leads, samples)
    x = perturb(x, fs, np.random.default_rng(record_seed(stem, base_seed)), cfg)

    os.makedirs(dst_dir, exist_ok=True)
    n_sig = x.shape[0]
    wfdb.wrsamp(
        record_name=stem,
        fs=fs,
        units=list(fields.get("units") or ["mV"] * n_sig),
        sig_name=list(fields.get("sig_name") or [f"L{i}" for i in range(n_sig)]),
        p_signal=x.T,
        fmt=["16"] * n_sig,
        write_dir=dst_dir,
    )
    # wfdb's comment handling differs between versions; rewrite the '#' block verbatim
    # from the source so '# Chagas label:' and '# Source:' survive byte-for-byte and
    # evaluate_model.py keeps working.
    out_header = os.path.join(dst_dir, stem + ".hea")
    with open(out_header, "r") as fh:
        lines = [ln for ln in fh.read().split("\n") if not ln.startswith("#")]
    while lines and lines[-1] == "":
        lines.pop()
    lines.extend(_header_comment_lines(header_text))
    with open(out_header, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return os.path.join(dst_dir, stem)


def _worker(args: Tuple[str, str]) -> Optional[str]:
    src_stem, dst_dir = args
    try:
        return write_record(src_stem, dst_dir)
    except Exception as exc:  # pragma: no cover - a corrupt source record
        print(f"[build_test_C] FAILED {src_stem}: {exc}", flush=True)
        return None


def find_records(folder: str) -> List[str]:
    out = set()
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if name.endswith(".hea"):
                out.add(os.path.join(root, name)[: -len(".hea")])
    return sorted(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default=DEFAULT_SOURCE)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="0 = all records")
    args = ap.parse_args(argv)

    records = find_records(args.source)
    if args.limit:
        records = records[: args.limit]
    print(f"[build_test_C] {len(records)} records from {args.source}", flush=True)

    jobs = []
    for stem in records:
        rel = os.path.relpath(os.path.dirname(stem), args.source)
        jobs.append((stem, os.path.join(args.dest, rel) if rel != "." else args.dest))

    written = 0
    if args.workers > 1:
        from multiprocessing import Pool

        with Pool(args.workers) as pool:
            for i, out in enumerate(pool.imap_unordered(_worker, jobs, chunksize=64)):
                written += out is not None
                if (i + 1) % 5000 == 0:
                    print(f"[build_test_C] {i + 1}/{len(jobs)}", flush=True)
    else:
        for job in jobs:
            written += _worker(job) is not None

    print(f"[build_test_C] wrote {written}/{len(jobs)} records to {args.dest}", flush=True)
    return 0 if written == len(jobs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
