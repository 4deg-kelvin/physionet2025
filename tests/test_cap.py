"""Round 3: the prevalence-based negative cap (``data.cap_mode: prevalence``).

All positives are kept and ``round(n_pos * (1 - p) / p)`` negatives are drawn, uniformly
or length-matched as ``match_negatives_on`` says; ``cap_mode: count`` is the round-1/2
code path, unchanged.
"""

from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np
import pytest

from chagas.config import ExperimentCfg
from chagas.dataset import RecordEntry
from chagas.training import _subsample, negative_budget

DUR = {"short": 7.335, "long": 10.24}


def _entries(n_pos: int, n_neg: int, pos_short: float = 0.9, neg_short: float = 0.5) -> List[RecordEntry]:
    out = []
    for label, n, frac in ((1, n_pos, pos_short), (0, n_neg, neg_short)):
        for i in range(n):
            group = "short" if i < round(frac * n) else "long"
            out.append(RecordEntry(f"/x/{label}_{i}", f"{label}_{i}", "code15", label, duration_s=DUR[group]))
    return out


def _round_two_subsample(entries, max_records, seed):
    """Verbatim round-1/2 ``_subsample`` body for ``match_on="none"`` (the count reference)."""
    entries = list(entries)
    if max_records <= 0 or len(entries) <= max_records:
        return entries
    pos = [e for e in entries if e.label_hard == 1]
    neg = [e for e in entries if e.label_hard != 1]
    budget = max(0, max_records - len(pos))
    if budget < len(neg):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(neg), size=budget, replace=False))
        neg = [neg[i] for i in keep]
    return pos + neg


def test_prevalence_cap_gives_the_expected_negative_count():
    entries = _entries(100, 5000)
    assert negative_budget(100, 0.065) == 1438  # 100 * 0.935 / 0.065 = 1438.46
    for match in ("none", "duration"):
        out = _subsample(entries, 0, seed=3, match_on=match, cap_mode="prevalence", cap_prevalence=0.065)
        assert sum(e.label_hard for e in out) == 100, "every positive is kept"
        assert len(out) - 100 == 1438, match
        assert sum(e.label_hard for e in out) / len(out) == pytest.approx(0.065, abs=1e-3)
        assert len({e.record_id for e in out}) == len(out)
        again = _subsample(entries, 0, seed=3, match_on=match, cap_mode="prevalence", cap_prevalence=0.065)
        assert [e.record_id for e in again] == [e.record_id for e in out]
    matched = _subsample(entries, 0, seed=3, match_on="duration", cap_mode="prevalence", cap_prevalence=0.065)
    neg_groups = Counter(e.length_group for e in matched if e.label_hard == 0)
    assert neg_groups["short"] / 1438 == pytest.approx(0.9, abs=1e-3), "negatives follow the positives' lengths"


def test_prevalence_cap_ignores_max_records_and_keeps_everything_when_short_of_negatives():
    entries = _entries(100, 5000)
    a = _subsample(entries, 200, seed=1, cap_mode="prevalence", cap_prevalence=0.5)
    b = _subsample(entries, 0, seed=1, cap_mode="prevalence", cap_prevalence=0.5)
    assert [e.record_id for e in a] == [e.record_id for e in b] and len(a) == 200
    few = _entries(100, 300)
    assert _subsample(few, 0, seed=1, cap_mode="prevalence", cap_prevalence=0.065) == few


def test_count_mode_is_unchanged():
    entries = _entries(30, 1000)
    for max_records in (0, 130, 500, 5000):
        expected = _round_two_subsample(entries, max_records, 7)
        assert _subsample(entries, max_records, 7) == expected
        assert _subsample(entries, max_records, 7, cap_mode="count") == expected


def test_prevalence_must_lie_strictly_between_zero_and_one():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            ExperimentCfg.from_dict({"data": {"cap_mode": "prevalence", "cap_prevalence": bad}})
        with pytest.raises(ValueError):
            negative_budget(10, bad)
    with pytest.raises(ValueError):
        ExperimentCfg.from_dict({"data": {"cap_mode": "ratio"}})
    with pytest.raises(ValueError):
        _subsample(_entries(2, 10), 5, seed=0, cap_mode="ratio")
