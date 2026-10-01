"""PLAN_R2.md section 7 item 6: the round-2 evaluation views, diagnostics and bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import scoring as S
from chagas.metrics import challenge_score, compute_auc


def _toy() -> pd.DataFrame:
    """Hand-built predictions with known within-group behaviour.

    * CODE-15 short (7.335 s): 4 positives scored 0.9..0.6 above 36 negatives at 0.5..0.1
      -> within-group AUROC 1.0 and TPR@5% = 2/4 (capacity int(0.05 * 40) = 2).
    * CODE-15 long (10.24 s): 2 positives at 0.05 below 38 negatives at 0.2
      -> within-group AUROC 0.0.
    * PTB-XL (10 s, all negative) and SaMi-Trop (one 10.24 s positive, one 7.335 s).
    """
    rows = []
    for i, p in enumerate([0.9, 0.8, 0.7, 0.6]):
        rows.append((f"s_pos{i}", 1, "CODE-15%", p, 7.335))
    for i in range(36):
        rows.append((f"s_neg{i}", 0, "CODE-15%", 0.1 + 0.4 * i / 35, 7.335))
    for i in range(2):
        rows.append((f"l_pos{i}", 1, "CODE-15%", 0.05, 10.24))
    for i in range(38):
        rows.append((f"l_neg{i}", 0, "CODE-15%", 0.2, 10.24))
    for i in range(10):
        rows.append((f"ptb{i}", 0, "PTB-XL", 0.01, 10.0))
    rows.append(("sami_long", 1, "SaMi-Trop", 0.99, 10.24))
    rows.append(("sami_short", 1, "SaMi-Trop", 0.98, 7.335))
    df = pd.DataFrame(rows, columns=["stem", "label", "source", "prob", "duration_s"])
    df["prob_filled"] = df["prob"]
    durations = df[["stem", "duration_s"]]
    return S.attach_durations(df.drop(columns=["duration_s"]), durations)


def test_view_membership():
    df = _toy()
    sizes = {v: int(S.view_mask(df, v).sum()) for v in S.VIEW_ORDER}
    assert sizes == {"A": 92, "B": 80, "S7": 40, "S10": 40, "S10+": 41}
    s10p = df[S.view_mask(df, "S10+")]
    assert set(s10p["source"]) == {"CODE-15%", "SaMi-Trop"}  # PTB-XL negatives are not CODE-15
    assert "sami_short" not in set(s10p["stem"])


def test_known_within_group_scores_and_cis_contain_the_point():
    df = _toy()
    s7 = S.score_view(df, "S7", n_boot=200)
    assert s7["auroc"] == pytest.approx(1.0) and s7["challenge"] == pytest.approx(0.5)
    s10 = S.score_view(df, "S10", n_boot=200)
    assert s10["auroc"] == pytest.approx(0.0) and s10["challenge"] == pytest.approx(0.0)
    for row in (s7, s10, S.score_view(df, "B", n_boot=200), S.score_view(df, "S10+", n_boot=200)):
        assert row["challenge_lo"] <= row["challenge"] <= row["challenge_hi"], row
        assert row["auroc_lo"] <= row["auroc"] <= row["auroc_hi"], row


def test_duration_only_baseline():
    df = _toy()
    b = S.duration_only(df, "B")
    # 4 of 6 CODE-15 positives are short; 36 of 74 negatives are short
    y = np.array([1] * 4 + [0] * 36 + [1] * 2 + [0] * 38)
    score = np.array([1.0] * 40 + [0.0] * 40)
    assert b["auroc"] == pytest.approx(compute_auc(y, score)[0])
    assert b["auroc"] == pytest.approx((4 / 6) * (38 / 74) + 0.5 * ((4 / 6) * (36 / 74) + (2 / 6) * (38 / 74)))
    assert b["challenge"] == pytest.approx(challenge_score(y, score))


def test_length_cue_gap():
    gap = S.length_cue_gap(_toy())
    assert gap["neg_mean_short"] == pytest.approx(0.3)
    assert gap["neg_mean_long"] == pytest.approx(0.2)
    assert gap["gap"] == pytest.approx(0.1)


def test_weighted_bootstrap_equals_explicit_resampling():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.1).astype(float)
    p = np.round(rng.random(n), 2)  # plenty of ties
    p[:5] = np.nan                 # failures rank last
    ranked = S.RankedScores.build(y, p)
    weights = next(S.bootstrap_weights(n, 20, seed=1, chunk=20))
    ch, au = ranked.metrics(weights)
    for b in range(weights.shape[0]):
        idx = np.repeat(np.arange(n), weights[b].astype(int))
        assert ch[b] == pytest.approx(challenge_score(y[idx], p[idx]), abs=1e-12)
        assert au[b] == pytest.approx(compute_auc(y[idx], p[idx])[0], abs=1e-12)
    # unit weights reproduce the plain metrics
    ch1, au1 = ranked.metrics(np.ones((1, n)))
    assert ch1[0] == pytest.approx(challenge_score(y, p)) and au1[0] == pytest.approx(compute_auc(y, p)[0])


def test_bootstrap_weights_are_reproducible_and_sum_to_n():
    a = np.vstack(list(S.bootstrap_weights(100, 120, seed=5, chunk=50)))
    b = np.vstack(list(S.bootstrap_weights(100, 120, seed=5, chunk=50)))
    assert a.shape == (120, 100) and np.array_equal(a, b)
    assert (a.sum(axis=1) == 100).all()


def test_eval_report_builds_every_view(tmp_path):
    import eval_report as ER

    df = _toy()
    report, tables = ER.build_report("toy", df, df, n_boot=50)
    views = tables["summary"]["view"].tolist()
    assert views == ["A", "B", "S7", "S10", "S10+", "A·C", "B·C", "S7·C", "S10·C", "S10+·C"]
    assert "Length-cue gap" in report and "Duration-only" in report
    assert set(tables["diagnostics"]["diagnostic"]) == {
        "duration-only baseline (1 = short record)", "length-cue gap (CODE-15 negatives)"
    }
