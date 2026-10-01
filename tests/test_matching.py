"""PLAN_R2.md section 7 item 2: length-matched negative sampling (``data.match_negatives_on``)."""

from __future__ import annotations

from collections import Counter
from typing import List

import numpy as np
import pytest

from chagas.dataset import RecordEntry
from chagas.lengths import length_group
from chagas.training import _subsample, matched_negative_quota

DUR = {"short": 7.335, "mid": 8.5, "long": 10.24}


def _entries(pos: dict, neg: dict) -> List[RecordEntry]:
    out: List[RecordEntry] = []
    for label, counts in ((1, pos), (0, neg)):
        for group, n in counts.items():
            for i in range(n):
                out.append(RecordEntry(path=f"/x/{label}{group}{i}", record_id=f"{label}{group}{i}",
                                       label_hard=label, duration_s=DUR[group]))
    return out


def _groups(entries, label):
    return Counter(length_group(e.duration_s) for e in entries if e.label_hard == label)


def _round1_subsample(entries, max_records, seed):
    """Verbatim copy of the round-1 ``_subsample`` body, the reference for ``"none"``."""
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


def test_negatives_follow_the_positive_length_distribution():
    entries = _entries({"short": 80, "mid": 5, "long": 15}, {"short": 600, "mid": 50, "long": 400})
    out = _subsample(entries, 500, seed=3, match_on="duration")
    assert len(out) == 500
    pos, neg = _groups(out, 1), _groups(out, 0)
    assert pos == Counter({"short": 80, "mid": 5, "long": 15}), "every positive must be kept"
    n_neg = sum(neg.values())
    assert n_neg == 400
    for g, n in pos.items():
        assert abs(neg[g] - n_neg * n / 100) <= 1, (g, neg[g])  # within rounding


def test_matching_is_seeded_and_deterministic():
    entries = _entries({"short": 30, "long": 10}, {"short": 300, "long": 300})
    a = [e.record_id for e in _subsample(entries, 200, seed=1, match_on="duration")]
    b = [e.record_id for e in _subsample(entries, 200, seed=1, match_on="duration")]
    c = [e.record_id for e in _subsample(entries, 200, seed=2, match_on="duration")]
    assert a == b
    assert a != c


def test_none_is_exactly_round_one():
    entries = _entries({"short": 30, "long": 10}, {"short": 300, "mid": 20, "long": 300})
    for cap, seed in ((200, 0), (100, 7), (0, 1), (10_000, 2)):
        ref = [e.record_id for e in _round1_subsample(entries, cap, seed)]
        assert [e.record_id for e in _subsample(entries, cap, seed)] == ref
        assert [e.record_id for e in _subsample(entries, cap, seed, match_on="none")] == ref


def test_group_with_too_few_negatives_is_taken_whole_and_the_rest_redistributed():
    # positives 80% short, but only 10 short negatives exist
    entries = _entries({"short": 80, "mid": 10, "long": 10}, {"short": 10, "mid": 200, "long": 200})
    out = _subsample(entries, 300, seed=0, match_on="duration")
    neg = _groups(out, 0)
    assert len(out) == 300 and sum(_groups(out, 1).values()) == 100
    assert neg["short"] == 10
    # the 160 short negatives that do not exist are split mid:long in the positives' 1:1 ratio
    assert neg["mid"] + neg["long"] == 190 and abs(neg["mid"] - neg["long"]) <= 1


def test_quota_is_exact_and_bounded():
    q = matched_negative_quota({"short": 5541, "mid": 80, "long": 907},
                               {"short": 158678, "mid": 5220, "long": 122459}, 93472)
    assert sum(q.values()) == 93472
    assert q["short"] == pytest.approx(93472 * 5541 / 6528, abs=1)
    # budget above what exists: take everything
    q = matched_negative_quota({"short": 1}, {"short": 3, "long": 4}, 100)
    assert q == {"short": 3, "long": 4}
    # a group without positives only receives negatives once the others are exhausted
    q = matched_negative_quota({"short": 1}, {"short": 50, "long": 50}, 20)
    assert q == {"short": 20, "long": 0}


def test_unknown_duration_is_its_own_group():
    entries = _entries({"short": 10}, {"short": 100})
    entries += [RecordEntry(path="/x/u", record_id="u", label_hard=0)]  # duration NaN
    out = _subsample(entries, 50, seed=0, match_on="duration")
    assert "u" not in {e.record_id for e in out}  # no positive is 'unknown'
    assert len(out) == 50
