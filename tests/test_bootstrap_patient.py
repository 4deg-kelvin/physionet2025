"""Patient-level (cluster) bootstrap in ``experiments/scoring.py``.

CODE-15% has ~1.47 examinations per patient, so the held-out records are not
independent; ``bootstrap_unit="patient"`` resamples patient groups and gives each record
its group's count.  Checks: (a) with singleton groups it equals the record bootstrap bit
for bit; (b) grouping correlated records widens the SE; (c) the weights stay paired
across models; (d) a stem without a group is an error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import aggregate as AG
import eval_report as ER
import scoring as S
from chagas.metrics import challenge_score, compute_auc


def _frame(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """CODE-15-like predictions with both length groups, durations attached."""
    rng = np.random.default_rng(seed)
    dur = np.where(rng.random(n) < 0.6, 7.335, 10.24)
    y = (rng.random(n) < 0.15).astype(int)
    df = pd.DataFrame({
        "stem": [f"{i:05d}" for i in range(n)],  # leading zeros: stems are strings
        "label": y,
        "source": "CODE-15%",
        "prob": 1 / (1 + np.exp(-(1.2 * y + rng.normal(0, 0.8, n)))),
    })
    df["prob_filled"] = df["prob"]
    return S.attach_durations(df, pd.DataFrame({"stem": df["stem"], "duration_s": dur}))


def _singletons(df: pd.DataFrame) -> dict:
    return {s: f"c15_{s}" for s in df["stem"]}


# ---------------------------------------------------------------------------
# (a) singleton groups == record level, bit for bit
# ---------------------------------------------------------------------------
def test_singleton_groups_draw_the_record_level_weights():
    n = 257
    codes = S.group_codes([f"r{i}" for i in range(n)], "patient", {f"r{i}": f"g{i}" for i in range(n)})
    assert np.array_equal(codes, np.arange(n))
    rec = np.vstack(list(S.bootstrap_weights(n, 120, seed=11, chunk=50)))
    pat = np.vstack(list(S.bootstrap_weights(n, 120, seed=11, chunk=50, groups=codes)))
    assert np.array_equal(rec, pat)


def test_singleton_groups_reproduce_record_level_scores_exactly():
    df = _frame()
    groups = _singletons(df)
    for view in ("A", "S7", "S10"):
        rec = S.score_view(df, view, n_boot=200, unit="record")
        pat = S.score_view(df, view, n_boot=200, unit="patient", groups=groups)
        assert rec == pat, view  # every CI / SE column identical, not just close


def test_singleton_groups_reproduce_record_level_aggregate_exactly(tmp_path):
    df = _frame(400, seed=1)
    results = tmp_path / "results"
    rng = np.random.default_rng(2)
    for name, noise in (("a", 0.0), ("b", 0.4)):
        out = df.drop(columns=["duration_s", "length_group"]).copy()
        out["prob"] = out["prob_filled"] = np.clip(out["prob"] + rng.normal(0, noise, len(out)), 0, 1)
        (results / name).mkdir(parents=True)
        out.to_csv(results / name / "predictions_A.csv", index=False)
    dur = tmp_path / "durations.csv"
    df[["stem", "duration_s"]].to_csv(dur, index=False)
    kw = dict(results_dir=results, durations_path=str(dur), n_boot=100)
    rec, *_ = AG.run(["a=a", "b=b"], "a", unit="record", **kw)
    pat, *_ = AG.run(["a=a", "b=b"], "a", unit="patient", stem_groups=_singletons(df), **kw)
    pd.testing.assert_frame_equal(rec, pat, check_exact=True)


# ---------------------------------------------------------------------------
# (b) clustering widens the SE
# ---------------------------------------------------------------------------
def test_duplicating_a_record_into_a_two_record_group_changes_the_se():
    df = _frame(120, seed=4)
    pos = df.index[df["label"] == 1][0]
    dup = df.loc[[pos]].copy()
    dup["stem"] = dup["stem"] + "_b"
    two = pd.concat([df, dup], ignore_index=True)
    separate = {s: f"g_{s}" for s in two["stem"]}
    together = dict(separate)
    together[dup["stem"].iloc[0]] = separate[df.loc[pos, "stem"]]
    a = S.score_view(two, "A", n_boot=300, unit="patient", groups=separate)
    b = S.score_view(two, "A", n_boot=300, unit="patient", groups=together)
    assert a["auroc"] == b["auroc"] and a["challenge"] == b["challenge"]  # points unchanged
    assert a["auroc_se"] != b["auroc_se"]


def test_record_level_se_is_smaller_on_correlated_examinations():
    """Each patient has 3 examinations with the same label and near-identical scores."""
    rng = np.random.default_rng(7)
    n_pat, k = 300, 3
    y_pat = (rng.random(n_pat) < 0.2).astype(int)
    s_pat = 1.0 * y_pat + rng.normal(0, 1.0, n_pat)
    rows = []
    for i in range(n_pat):
        for j in range(k):
            rows.append((f"p{i}_e{j}", y_pat[i], "CODE-15%", s_pat[i] + rng.normal(0, 0.05), 7.335, f"c15_{i}"))
    df = pd.DataFrame(rows, columns=["stem", "label", "source", "prob", "duration_s", "group"])
    df["prob_filled"] = df["prob"]
    df = S.attach_durations(df.drop(columns=["duration_s"]), df[["stem", "duration_s"]])
    groups = dict(zip(df["stem"], df["group"]))
    rec = S.score_view(df, "S7", n_boot=500, unit="record")
    pat = S.score_view(df, "S7", n_boot=500, unit="patient", groups=groups)
    # k exact duplicates would inflate the variance ~k-fold, i.e. the SE by ~sqrt(3)
    assert pat["auroc_se"] > 1.4 * rec["auroc_se"]
    assert pat["challenge_se"] > 1.2 * rec["challenge_se"]


def test_patient_weights_are_constant_within_a_group_and_draw_g_groups():
    groups = {"a": "g1", "b": "g1", "c": "g2", "d": "g3", "e": "g3", "f": "g3"}
    codes = S.group_codes(list("abcdef"), "patient", groups)
    assert codes.tolist() == [0, 0, 1, 2, 2, 2]  # order of first appearance
    w = np.vstack(list(S.bootstrap_weights(6, 200, seed=3, groups=codes)))
    assert (w[:, 0] == w[:, 1]).all() and (w[:, 3] == w[:, 4]).all() and (w[:, 4] == w[:, 5]).all()
    counts = w[:, [0, 2, 3]]
    assert (counts.sum(axis=1) == 3).all()  # G = 3 group draws per resample
    # the weighted metric still equals the metric on the explicitly resampled records
    y = np.array([1, 0, 1, 0, 0, 1], dtype=float)
    p = np.array([0.9, 0.2, 0.6, 0.6, 0.1, 0.4])
    ch, au = S.RankedScores.build(y, p).metrics(w)
    for r in range(20):
        idx = np.repeat(np.arange(6), w[r].astype(int))
        if 0 < y[idx].sum() < idx.size:
            assert au[r] == pytest.approx(compute_auc(y[idx], p[idx])[0], abs=1e-12)
            assert ch[r] == pytest.approx(challenge_score(y[idx], p[idx]), abs=1e-12)


# ---------------------------------------------------------------------------
# (c) paired across models
# ---------------------------------------------------------------------------
def test_weights_are_paired_across_models():
    df = _frame(200, seed=5)
    groups = {s: f"c15_{int(s) // 2}" for s in df["stem"]}  # pairs of records per patient
    codes = S.group_codes(df["stem"].tolist(), "patient", groups)
    seed = S.view_seed("A")
    first = np.vstack(list(S.bootstrap_weights(len(df), 100, seed, groups=codes)))
    again = np.vstack(list(S.bootstrap_weights(len(df), 100, seed, groups=codes)))
    assert np.array_equal(first, again)
    y = df["label"].to_numpy(dtype=float)
    p1 = df["prob_filled"].to_numpy()
    p2 = np.clip(p1 + np.random.default_rng(0).normal(0, 0.2, len(df)), 0, 1)
    ch, au = S.bootstrap_metrics(y, [p1, p2], 100, seed, groups=codes)
    ch1, au1 = S.bootstrap_metrics(y, [p1], 100, seed, groups=codes)
    ch2, au2 = S.bootstrap_metrics(y, [p2], 100, seed, groups=codes)
    # scoring the models together or separately sees the same resamples
    assert np.array_equal(ch[0], ch1[0]) and np.array_equal(au[1], au2[0])
    assert np.array_equal(au[0], au1[0]) and np.array_equal(ch[1], ch2[0])


def test_paired_self_difference_is_zero_at_patient_level(tmp_path):
    df = _frame(300, seed=6)
    results = tmp_path / "results"
    for name in ("a", "a_copy"):
        (results / name).mkdir(parents=True)
        df.drop(columns=["duration_s", "length_group"]).to_csv(results / name / "predictions_A.csv", index=False)
    dur = tmp_path / "durations.csv"
    df[["stem", "duration_s"]].to_csv(dur, index=False)
    groups = {s: f"c15_{int(s) // 3}" for s in df["stem"]}
    table, *_ = AG.run(["base=a", "same=a_copy"], "base", results_dir=results, durations_path=str(dur),
                       n_boot=100, unit="patient", stem_groups=groups)
    same = table[table["group"] == "same"]
    for col in ("d_challenge_se", "d_auroc_se", "d_ens_challenge_se", "d_ens_auroc_se"):
        assert (same[col] == 0.0).all(), col


# ---------------------------------------------------------------------------
# (d) errors and the loader
# ---------------------------------------------------------------------------
def test_missing_stem_raises_a_clear_error():
    df = _frame(50)
    groups = _singletons(df)
    del groups[df["stem"].iloc[3]]
    with pytest.raises(KeyError, match="no patient group"):
        S.score_view(df, "A", n_boot=10, unit="patient", groups=groups)
    with pytest.raises(ValueError, match="unknown bootstrap unit"):
        S.group_codes(["x"], "visit", {"x": "g"})


def test_load_groups_reads_stems_as_strings(tmp_path):
    path = tmp_path / "split.csv"
    pd.DataFrame({
        "path": ["/a/00012", "/b/12", "/c/x"],
        "stem": ["00012", "12", "x"],
        "label": [0, 1, 0],
        "source": ["CODE-15%", "SaMi-Trop", "PTB-XL"],
        "group": ["c15_7", "smt_12", "ptb_3"],
        "split": ["test", "test", "train"],
    }).to_csv(path, index=False)
    groups = S.load_groups(str(path))
    assert groups == {"00012": "c15_7", "12": "smt_12", "x": "ptb_3"}
    assert S.load_groups(str(path)) is groups  # cached


def test_eval_report_from_saved_rewrites_tables_not_predictions(tmp_path):
    df = _frame(200, seed=8)
    run = tmp_path / "run"
    run.mkdir()
    for which in ("A", "C"):
        df.to_csv(run / f"predictions_{which}.csv", index=False)
    (run / "report.md").write_text("# Evaluation report -- toy run\n")
    before = (run / "predictions_A.csv").read_bytes()
    dur = tmp_path / "durations.csv"
    df[["stem", "duration_s"]].to_csv(dur, index=False)
    split = tmp_path / "split.csv"
    pd.DataFrame({"stem": df["stem"], "group": [f"c15_{int(s) // 2}" for s in df["stem"]]}).to_csv(split, index=False)
    assert ER.main(["--from-saved", str(run), "--durations", str(dur), "--split", str(split), "--n-boot", "30"]) == 0
    summary = pd.read_csv(run / "summary.csv")
    assert {"A", "S7", "S10", "A·C"} <= set(summary["view"]) and summary["auroc_se"].notna().any()
    assert (run / "report.md").read_text().startswith("# Evaluation report -- toy run")
    assert "patient-level resamples" in (run / "report.md").read_text()
    assert (run / "predictions_A.csv").read_bytes() == before
