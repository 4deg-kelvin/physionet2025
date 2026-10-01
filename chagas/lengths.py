"""Record length: parse it from a WFDB header and bin it into length groups.

Why this module exists (PLAN_R2.md section 1, DECISIONS.md D-2026-09-30-2).  In CODE-15
the recording length predicts the label: records are either 7.34 s (2,934 samples at
400 Hz) or 10.24 s (4,096 samples), and Chagas prevalence is 3.05% in the short group
against 0.24% in the long one.  The hidden cohorts are all ~10 s (Reyna et al., CinC
2025 / arXiv:2510.02202), so the cue is absent at test time.  The discovery and the
within-length evaluation built on it are this project's own (analysis of
``physionet2025/signal_lengths.csv`` and saved predictions, 2026-09-30); that CODE
records are 7-10 s long is documented by Ribeiro et al., Nat. Commun. 2020, 11:1760.

Everything that needs a length group -- length-matched negative sampling
(:func:`chagas.training._subsample`), the within-length validation metrics
(:mod:`chagas.model`) and the evaluation views (``experiments/scoring.py``) -- uses the
thresholds defined here, so the three can never disagree.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

__all__ = [
    "SHORT_MAX_SECONDS",
    "LONG_MIN_SECONDS",
    "LENGTH_GROUPS",
    "LENGTH_GROUP_IDS",
    "length_group",
    "parse_header_length",
    "header_duration",
]

#: A record shorter than this is "short" (CODE-15's 7.34 s records).
SHORT_MAX_SECONDS: float = 7.5
#: A record at least this long is "long" (CODE-15's 10.24 s, PTB-XL's 10 s).
LONG_MIN_SECONDS: float = 9.9

#: Group names, in a fixed order.  ``unknown`` holds records whose header could not be
#: parsed (or synthetic entries built without one).
LENGTH_GROUPS: Tuple[str, ...] = ("short", "mid", "long", "unknown")
#: Stable integer ids, carried through the DataLoader as ``length_group``.
LENGTH_GROUP_IDS: Dict[str, int] = {name: i for i, name in enumerate(LENGTH_GROUPS)}


def length_group(duration_s: float) -> str:
    """``"short"`` (< 7.5 s), ``"long"`` (>= 9.9 s), ``"mid"`` otherwise, ``"unknown"`` if NaN."""
    if duration_s is None or not math.isfinite(float(duration_s)):
        return "unknown"
    if duration_s < SHORT_MAX_SECONDS:
        return "short"
    if duration_s >= LONG_MIN_SECONDS:
        return "long"
    return "mid"


def parse_header_length(header_text: str) -> Tuple[int, float]:
    """``(n_samples, fs)`` from the record line of a WFDB header.

    The record line is the first non-comment line:
    ``name[/nseg] nsig [fs[/counterfreq[(base)]] [nsamp [basetime [basedate]]]]``
    (PhysioNet WFDB header format).  Raises :class:`ValueError` when the sampling rate
    or the sample count is missing, because a guessed length would silently put a record
    into the wrong group.
    """
    for line in header_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = stripped.split()
        if len(tokens) < 4:
            raise ValueError(f"WFDB record line has no sample count: {stripped!r}")
        fs_token = tokens[2].split("/")[0].split("(")[0]
        fs = float(fs_token)
        n_samples = int(tokens[3])
        if fs <= 0 or n_samples < 0:
            raise ValueError(f"implausible fs/nsamp in WFDB record line: {stripped!r}")
        return n_samples, fs
    raise ValueError("WFDB header has no record line")


def header_duration(header_text: str) -> float:
    """Duration in seconds from a WFDB header, or NaN when it cannot be parsed."""
    try:
        n_samples, fs = parse_header_length(header_text)
    except (ValueError, IndexError):
        return float("nan")
    return float(n_samples) / float(fs)
