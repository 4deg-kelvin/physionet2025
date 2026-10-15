"""Round 3: the fixed validation split, PTB-XL patient groups, the per-view monitor.

* ``train.val_protocol: fixed`` carves validation from every scanned record before the
  negative cap, stratified by source x label x length group with its own seed: it is the
  same for every training seed, cap and ``match_negatives_on``, keeps every stratum and
  the natural prevalence.  ``legacy`` is unchanged (pinned partition).
* PTB-XL records are grouped by ``patient_id`` from ``ptbxl_database.csv``.
* ``data.exclude_sources`` drops records at scan time.
* The S7 / S10+ validation metrics use the same view definitions as
  ``experiments/scoring.view_mask``; a view metric with an empty view falls back to
  ``val_challenge``; ``best_by_stage`` follows the monitor.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
import pytest

from chagas import labels as L
from chagas.config import ExperimentCfg, default_config
from chagas.dataset import RecordEntry, scan_records
from chagas.lengths import LENGTH_GROUP_IDS, length_group
from chagas.training import resolve_monitor, split_entries


def _entry(rid: str, source: str, label: int, group: str, duration: float) -> RecordEntry:
    return RecordEntry(f"/x/{rid}", rid, source, int(label), float(label), group=group,
                       source_id=L.SOURCE_IDS[source], duration_s=duration)


def _synthetic(seed: int = 0) -> List[RecordEntry]:
    """CODE-15 with the length shortcut (3% short / 1% long prevalence, 1-2 exams per
    patient), SaMi-Trop positives of both lengths, PTB-XL negatives."""
    rng = np.random.default_rng(seed)
    out: List[RecordEntry] = []
    for pid in range(1500):
        short = rng.random() < 0.6
        label = int(rng.random() < (0.03 if short else 0.01))
        for k in range(1 + int(rng.random() < 0.4)):
            out.append(_entry(f"c{pid}_{k}", "code15", label, f"c15_{pid}", 7.34 if short else 10.24))
    for i in range(80):
        out.append(_entry(f"s{i}", "samitrop", 1, f"rec_samitrop_s{i}", 7.34 if i % 2 else 10.0))
    for i in range(200):
        out.append(_entry(f"{i + 1:05d}_hr", "ptbxl", 0, f"ptb_{i // 2}", 10.0))
    return out


# ---------------------------------------------------------------------------
# the split itself
# ---------------------------------------------------------------------------
def _legacy_set() -> List[RecordEntry]:
    out = []
    for i in range(24):  # 12 CODE-15 patients x 2 exams, patients 0-2 positive
        pid = i // 2
        out.append(_entry(f"c{i}", "code15", int(pid < 3), f"c15_{pid}", 7.34 if i % 3 else 10.24))
    for i in range(8):
        out.append(_entry(f"s{i}", "samitrop", 1, f"rec_samitrop_s{i}", 10.0))
    for i in range(10):
        out.append(_entry(f"p{i}", "ptbxl", 0, f"rec_ptbxl_p{i}", 10.0))
    return out


def test_legacy_split_is_the_round_two_partition():
    # Pinned from the pre-round-3 split_entries (commit 484feae) on this set.
    expected = {
        42: ["c20", "c21", "c4", "c5", "c8", "c9", "p4", "p8", "s0", "s7"],
        7: ["c18", "c19", "c2", "c20", "c21", "c3", "p3", "p8", "s4", "s7"],
    }
    for seed, val_ids in expected.items():
        _train, val = split_entries(_legacy_set(), 0.25, seed)
        assert sorted(e.record_id for e in val) == val_ids
        assert split_entries(_legacy_set(), 0.25, seed, protocol="legacy")[1] == val


def test_fixed_split_keeps_every_stratum_patients_and_natural_prevalence():
    entries = _synthetic()
    train, val = split_entries(entries, 0.1, 12345, protocol="fixed")
    assert len(train) + len(val) == len(entries)
    assert not {e.group for e in train} & {e.group for e in val}, "a patient was split"

    def strata(records):
        return {(e.source, e.label_hard, e.length_group) for e in records}
    assert strata(val) == strata(entries), "a source x label x length stratum is missing from val"
    natural = np.mean([e.label_hard for e in entries])
    assert np.mean([e.label_hard for e in val]) == pytest.approx(natural, abs=0.01)
    for source in ("code15", "samitrop", "ptbxl"):
        for group in ("short", "long"):
            n_all = sum(e.source == source and e.length_group == group for e in entries)
            n_val = sum(e.source == source and e.length_group == group for e in val)
            if n_all > 50:
                assert n_val / n_all == pytest.approx(0.1, abs=0.03), (source, group)


def test_fixed_split_strata_are_independent_of_each_other():
    entries = _synthetic()
    _t, val = split_entries(entries, 0.1, 12345, protocol="fixed")
    _t, val_no_ptb = split_entries([e for e in entries if e.source != "ptbxl"], 0.1, 12345, protocol="fixed")
    assert [e.record_id for e in val if e.source != "ptbxl"] == [e.record_id for e in val_no_ptb]
    _t, other_seed = split_entries(entries, 0.1, 999, protocol="fixed")
    assert {e.record_id for e in other_seed} != {e.record_id for e in val}
    with pytest.raises(ValueError):
        split_entries(entries, 0.1, 1, protocol="sometimes")


# ---------------------------------------------------------------------------
# train_model wiring: validation before the cap, independent of seed / cap / matching
# ---------------------------------------------------------------------------
def _run_train_model(monkeypatch, tmp_path, cfg, entries):
    import chagas.training as T

    captured = {}

    def fake_datamodule(cfg_, train, val):
        captured["train"], captured["val"] = list(train), list(val)
        return None

    def fake_member(cfg_, datamodule, work_dir, *, seed, max_hours, deadline, verbose, log_dir=None,
                    wise_dir=None, monitor=None):
        captured["monitor"] = monitor
        return SimpleNamespace(val_history=[])

    monkeypatch.setattr(T, "find_records", lambda folder: ["/x/0"])
    monkeypatch.setattr(T.L.Code15Table, "from_csv", classmethod(lambda cls, *a, **k: SimpleNamespace(__len__=lambda: 0)))
    monkeypatch.setattr(T, "scan_records", lambda paths, cfg, table=None, ptbxl_patients=None, workers=8: list(entries))
    monkeypatch.setattr(T, "ChagasDataModule", fake_datamodule)
    monkeypatch.setattr(T, "_train_member", fake_member)
    monkeypatch.setattr(T, "_save_checkpoint", lambda module, path, cfg, extra_meta=None: Path(path).write_text("x"))
    out = tmp_path / f"model_{len(list(tmp_path.iterdir()))}"
    T.train_model(str(tmp_path / "data_without_ptbxl_csv"), str(out), verbose=False, cfg=cfg)
    captured["manifest"] = json.loads((out / "chagas_manifest.json").read_text())
    return captured


def test_fixed_validation_ignores_training_seed_cap_and_matching(monkeypatch, tmp_path):
    entries = _synthetic()
    variants = [
        dict(seed=42, cap_mode="count", max_records=400, match="none"),
        dict(seed=43, cap_mode="prevalence", max_records=400, match="duration"),
        dict(seed=44, cap_mode="count", max_records=0, match="none"),
    ]
    vals = []
    for v in variants:
        cfg = default_config()
        cfg.train.val_protocol = "fixed"
        cfg.train.seed = v["seed"]
        cfg.data.cap_mode = v["cap_mode"]
        cfg.data.cap_prevalence = 0.2
        cfg.data.max_records = v["max_records"]
        cfg.data.match_negatives_on = v["match"]
        got = _run_train_model(monkeypatch, tmp_path, cfg, entries)
        vals.append([e.record_id for e in got["val"]])
        assert not {e.group for e in got["train"]} & {e.group for e in got["val"]}
        assert got["manifest"]["split"]["val_protocol"] == "fixed"
        assert got["manifest"]["split"]["val_seed"] == 12345
        assert got["manifest"]["config"]["train"]["val_protocol"] == "fixed"
        n_pos = sum(e.label_hard for e in got["train"])
        if v["cap_mode"] == "prevalence":
            assert len(got["train"]) - n_pos == round(n_pos * 0.8 / 0.2)
        elif v["max_records"]:
            assert len(got["train"]) == v["max_records"]
    assert vals[0] == vals[1] == vals[2], "the fixed validation split moved"
    natural = np.mean([e.label_hard for e in entries])
    assert np.mean([e.label_hard for e in got["val"]]) == pytest.approx(natural, abs=0.01)


def test_legacy_train_model_splits_after_the_cap(monkeypatch, tmp_path):
    from chagas.training import _subsample

    entries = _synthetic()
    cfg = default_config()
    cfg.data.max_records = 400
    got = _run_train_model(monkeypatch, tmp_path, cfg, entries)
    _t, expected = split_entries(_subsample(entries, 400, cfg.train.seed), cfg.train.val_fraction, cfg.train.seed)
    assert [e.record_id for e in got["val"]] == [e.record_id for e in expected]
    assert got["monitor"] == "val_challenge"
    assert got["manifest"]["split"]["val_protocol"] == "legacy"


# ---------------------------------------------------------------------------
# PTB-XL patients and excluded sources
# ---------------------------------------------------------------------------
def _write_ptbxl_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ecg_id,patient_id,age\n1,15709.0,56\n2,13243.0,19\n17,15709.0,57\n100,999.0,40\n")


def test_ptbxl_records_are_grouped_by_patient(tmp_path):
    data = tmp_path / "data"
    (data / "ptb").mkdir(parents=True)
    assert L.find_ptbxl_database(str(data)) is None
    _write_ptbxl_csv(data / "ptb" / "ptbxl_database.csv")  # one directory down
    path = L.find_ptbxl_database(str(data))
    assert path == data / "ptb" / "ptbxl_database.csv"
    _write_ptbxl_csv(data / "ptbxl_database.csv")         # the top level wins
    assert L.find_ptbxl_database(str(data)) == data / "ptbxl_database.csv"

    patients = L.load_ptbxl_patients(path)
    assert patients == {"1": "15709.0", "2": "13243.0", "17": "15709.0", "100": "999.0"}
    assert L.ptbxl_record_key("00017_hr") == "17" and L.ptbxl_record_key("00100_hr") == "100"
    group = lambda rid: L.patient_group(rid, "ptbxl", None, patients)  # noqa: E731
    assert group("00001_hr") == group("00017_hr") == "ptb_15709.0"
    assert group("00002_hr") == "ptb_13243.0"
    assert group("00005_hr") == "rec_ptbxl_00005_hr"           # not in the table: its own group
    assert L.patient_group("00001_hr", "ptbxl", None) == "rec_ptbxl_00001_hr"  # rounds 1-2
    assert L.patient_group("00001_hr", "samitrop", None, patients) == "rec_samitrop_00001_hr"


def _write_header(folder: Path, stem: str, source: str, label: bool, n: int = 5000, fs: int = 500) -> str:
    lines = [f"{stem} 12 {fs} {n}"] + [f"{stem}.mat 16 1000/mV 16 0 0 0 0 L{i}" for i in range(12)]
    lines += ["# Age: 50", "# Sex: Male", f"# Chagas label: {label}", f"# Source: {source}"]
    (folder / f"{stem}.hea").write_text("\n".join(lines) + "\n")
    return str(folder / stem)


def test_scan_records_groups_ptbxl_patients_and_drops_excluded_sources(tmp_path):
    paths = [
        _write_header(tmp_path, "00001_hr", "PTB-XL", False),
        _write_header(tmp_path, "00017_hr", "PTB-XL", False),
        _write_header(tmp_path, "00002_hr", "PTB-XL", False),
        _write_header(tmp_path, "555", "SaMi-Trop", True, n=2934, fs=400),
        _write_header(tmp_path, "777", "CODE-15%", False, n=4096, fs=400),
    ]
    _write_ptbxl_csv(tmp_path / "ptbxl_database.csv")
    patients = L.load_ptbxl_patients(tmp_path / "ptbxl_database.csv")
    cfg = default_config()
    table = L.Code15Table.empty(cfg.labels.aux_columns)
    entries = {e.record_id: e for e in scan_records(paths, cfg, table=table, ptbxl_patients=patients, workers=1)}
    assert entries["00001_hr"].group == entries["00017_hr"].group == "ptb_15709.0"
    assert entries["00002_hr"].group == "ptb_13243.0"
    assert entries["555"].length_group == "short" and entries["777"].length_group == "long"
    _t, val = split_entries(list(entries.values()), 0.5, 1, protocol="fixed")
    assert ("00001_hr" in {e.record_id for e in val}) == ("00017_hr" in {e.record_id for e in val})

    cfg.data.exclude_sources = ("ptbxl",)
    kept = scan_records(paths, cfg, table=table, ptbxl_patients=patients, workers=1)
    assert sorted(e.record_id for e in kept) == ["555", "777"]
    with pytest.raises(ValueError):
        ExperimentCfg.from_dict({"data": {"exclude_sources": ["ptb"]}})


def test_ptbxl_table_is_only_for_the_fixed_protocol():
    with pytest.raises(ValueError):
        ExperimentCfg.from_dict({"data": {"ptbxl_database_csv": "x.csv"}})
    cfg = ExperimentCfg.from_dict({"data": {"ptbxl_database_csv": "x.csv"}, "train": {"val_protocol": "fixed"}})
    assert cfg.data.ptbxl_database_csv == "x.csv"
    with pytest.raises(ValueError):
        ExperimentCfg.from_dict({"train": {"val_protocol": "random"}})


def test_round_three_options_default_to_round_two_behaviour():
    cfg = default_config()
    assert cfg.train.val_protocol == "legacy" and cfg.train.monitor == "val_challenge"
    assert cfg.data.cap_mode == "count" and cfg.data.exclude_sources == ()
    assert cfg.data.ptbxl_database_csv is None
    assert cfg.preprocess.window_stride_seconds is None


# ---------------------------------------------------------------------------
# per-view validation metrics and the monitor
# ---------------------------------------------------------------------------
def test_validation_view_masks_equal_scoring_view_mask():
    import scoring as S
    from chagas.model import view_mask

    rng = np.random.default_rng(3)
    n = 500
    sources = rng.choice(["CODE-15%", "SaMi-Trop", "PTB-XL"], size=n, p=[0.7, 0.1, 0.2])
    durations = rng.choice([7.34, 8.0, 10.0, 10.24, np.nan], size=n)
    frame = pd.DataFrame({"source": sources, "duration_s": durations,
                          "label": (rng.random(n) < 0.2).astype(int)})
    source_ids = np.array([L.SOURCE_IDS[L.normalise_source(s)] for s in sources])
    groups = np.array([LENGTH_GROUP_IDS[length_group(d)] for d in durations])
    for view in ("S7", "S10", "S10+"):
        expected = S.view_mask(frame, view)
        assert expected.any(), view
        assert np.array_equal(view_mask(view, frame["label"].to_numpy(), groups, source_ids), expected), view


def test_view_metrics_are_nan_when_a_view_cannot_be_scored():
    from chagas.model import format_val_line, view_metrics

    short, long_ = LENGTH_GROUP_IDS["short"], LENGTH_GROUP_IDS["long"]
    labels = np.array([1, 0, 0, 1, 0, 0], dtype=float)
    probs = np.array([0.9, 0.2, 0.1, 0.8, 0.3, 0.7])
    groups = np.array([short, short, short, long_, long_, long_])
    sources = np.array([0, 0, 0, 1, 0, 2])  # long positive is SaMi-Trop, long PTB-XL is out of S10+
    out = view_metrics(labels, probs, groups, sources)
    assert out["val_auroc_s7"] == pytest.approx(1.0)
    assert out["val_auroc_s10p"] == pytest.approx(1.0)        # {SaMi-Trop pos 0.8, CODE-15 neg 0.3}
    assert np.isfinite(out["val_challenge_s7"])
    none_pos = view_metrics(np.zeros(6), probs, groups, sources)
    assert all(np.isnan(v) for v in none_pos.values())
    assert all(np.isnan(v) for v in view_metrics(labels, probs, None, None).values())
    line = format_val_line({"stage": 1, "epoch": 0, "val_challenge": 0.1, "val_auroc": 0.5, "val_auprc": 0.1,
                            "n": 6, "val_challenge_s7": None, "val_auroc_s7": 0.75, "val_auroc_s10p": None})
    assert "auroc_s7=0.7500" in line and "challenge_s7" not in line


def test_monitor_must_be_a_logged_metric_and_falls_back_when_its_view_is_empty():
    for bad in ("val_loss", "val_challenge_short", "s7"):
        with pytest.raises(ValueError):
            ExperimentCfg.from_dict({"train": {"monitor": bad}})
    entries = _synthetic()
    assert resolve_monitor("val_challenge_s7", entries) == "val_challenge_s7"
    assert resolve_monitor("val_auroc_s10p", entries) == "val_auroc_s10p"
    no_short_pos = [e for e in entries if not (e.source == "code15" and e.length_group == "short" and e.label_hard)]
    with pytest.warns(RuntimeWarning, match="S7"):
        assert resolve_monitor("val_challenge_s7", no_short_pos) == "val_challenge"
    assert resolve_monitor("val_auroc", no_short_pos) == "val_auroc"


def test_best_by_stage_follows_the_monitor():
    from chagas.training import _summarise_val_history

    history = [
        {"stage": 2, "epoch": 0, "val_challenge": 0.50, "val_auroc": 0.8, "val_auprc": 0.2,
         "val_challenge_s7": 0.30, "val_auroc_s7": 0.7, "val_auroc_s10p": None},
        {"stage": 2, "epoch": 1, "val_challenge": 0.40, "val_auroc": 0.8, "val_auprc": 0.2,
         "val_challenge_s7": 0.35, "val_auroc_s7": 0.7, "val_auroc_s10p": 0.6},
    ]
    module = SimpleNamespace(val_history=history)
    assert _summarise_val_history(0, module)["best_by_stage"]["stage2"]["epoch"] == 0
    best = _summarise_val_history(0, module, "val_challenge_s7")["best_by_stage"]["stage2"]
    assert best["epoch"] == 1 and best["monitor"] == "val_challenge_s7"
    assert _summarise_val_history(0, module, "val_auroc_s10p")["best_by_stage"]["stage2"]["epoch"] == 1
