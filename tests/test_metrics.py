"""PLAN.md section 5 item 7: our challenge score against the organisers' and the legacy copy.

``chagas.metrics.challenge_score`` computes in closed form the expectation that
``helper_code.compute_challenge_score`` estimates with 10,000 permutations.  With no
ties the two are identical; with ties the reference is a Monte-Carlo estimate of ours,
so they agree only up to MC noise.  Both statements are asserted separately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import helper_code  # the organisers' frozen file
import utils as legacy_utils  # the team's legacy copy
from chagas.metrics import challenge_score, challenge_score_permutation, compute_auc


def _reference(labels, outputs):
    return helper_code.compute_challenge_score(np.asarray(labels), np.asarray(outputs))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_the_organisers_exactly_without_ties(seed):
    rng = np.random.default_rng(seed)
    n = 200
    labels = (rng.random(n) < 0.15).astype(float)
    outputs = rng.permutation(n).astype(float) / n  # all distinct
    ours = challenge_score(labels, outputs)
    theirs = _reference(labels, outputs)
    assert ours == pytest.approx(theirs, abs=1e-12), (ours, theirs)
    assert ours == pytest.approx(legacy_utils.compute_challenge_score(labels, outputs), abs=1e-12)


def test_matches_the_organisers_within_mc_noise_with_ties():
    rng = np.random.default_rng(7)
    n = 400
    labels = (rng.random(n) < 0.1).astype(float)
    # Heavy ties: only 5 distinct scores.
    outputs = rng.integers(0, 5, size=n).astype(float) / 4.0
    ours = challenge_score(labels, outputs)
    theirs = _reference(labels, outputs)
    print(f"\n[item 7] ties: exact = {ours:.6f}, organisers' 10k-permutation MC = {theirs:.6f}")
    assert ours == pytest.approx(theirs, abs=0.01)
    # Our own Monte-Carlo replica of their algorithm must reproduce their number bit for bit.
    assert challenge_score_permutation(labels, outputs) == pytest.approx(theirs, abs=1e-12)


def test_all_tied_is_the_analytic_answer():
    """Every score equal -> the top 5% is a uniform random sample, so TPR = capacity/n."""
    labels = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 1] * 2, dtype=float)
    outputs = np.full(20, 0.3)
    assert challenge_score(labels, outputs) == pytest.approx(int(0.05 * 20) / 20)


def test_perfect_and_worst_rankings():
    labels = np.array([1] * 5 + [0] * 95, dtype=float)
    perfect = np.concatenate([np.linspace(1.0, 0.9, 5), np.linspace(0.5, 0.0, 95)])
    assert challenge_score(labels, perfect) == pytest.approx(1.0)
    assert challenge_score(labels, -perfect) == pytest.approx(0.0)
    assert challenge_score(labels, perfect) == pytest.approx(_reference(labels, perfect))


def test_no_positives_gives_nan():
    assert np.isnan(challenge_score(np.zeros(50), np.random.rand(50)))


def test_non_finite_predictions_rank_last():
    """A record we failed on must never be promoted into the top 5%."""
    labels = np.array([1.0] + [0.0] * 39)
    outputs = np.array([np.nan] + list(np.linspace(0.0, 1.0, 39)))
    assert challenge_score(labels, outputs) == pytest.approx(0.0)


def test_compute_auc_uses_hard_labels():
    rng = np.random.default_rng(0)
    labels = (rng.random(300) < 0.2).astype(float)
    probs = np.clip(labels * 0.25 + rng.random(300) * 0.75, 0, 1)
    auroc, auprc = compute_auc(labels, probs)
    assert 0.5 < auroc < 1.0 and 0.0 < auprc < 1.0
    ref_auroc, ref_auprc = helper_code.compute_auc(labels, probs)
    assert auroc == pytest.approx(ref_auroc, abs=1e-6)
    assert auprc == pytest.approx(ref_auprc, abs=0.02)
    assert np.isnan(compute_auc(np.zeros(10), np.random.rand(10))[0])
