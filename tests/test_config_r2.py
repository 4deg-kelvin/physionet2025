"""PLAN_R2.md section 7 item 8 plus sections 3.4 and 6.

* The shipped default is unchanged: ``default_config()`` equals
  ``experiments/configs/finalA_negcap.yaml`` field for field (name/description aside),
  and every round-2 flag defaults to round-1 behaviour.
* Every committed config is exactly what ``make_configs.py`` generates.
* Logging helpers (validation line with the global step, within-length metrics, data
  summary) and run_info.json.
* The seed ensemble averages member probabilities and splits the time budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chagas.config import ExperimentCfg, default_config

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "experiments" / "configs"


def _strip(d):
    d = dict(d)
    d.pop("name")
    d.pop("description")
    return d


def test_shipped_default_equals_finalA_negcap():
    shipped = default_config().to_dict()
    final_a = ExperimentCfg.from_yaml(CONFIGS / "finalA_negcap.yaml").to_dict()
    assert _strip(shipped) == _strip(final_a)


def test_round_two_flags_default_to_round_one_behaviour():
    cfg = default_config()
    assert cfg.data.match_negatives_on == "none"
    assert cfg.preprocess.window_mode == "single"
    assert cfg.preprocess.window_seconds == 10.0
    assert cfg.train.wise_ft == "off"
    assert cfg.train.wise_ft_save_all is False
    assert cfg.train.ensemble_members == 1
    for bad in ({"data": {"match_negatives_on": "length"}},
                {"preprocess": {"window_mode": "many"}},
                {"train": {"wise_ft": "on"}},
                {"train": {"wise_ft_alphas": [1.5]}}):
        with pytest.raises(ValueError):
            ExperimentCfg.from_dict(bad)


def test_committed_configs_are_exactly_what_make_configs_generates(tmp_path):
    import make_configs

    make_configs.main(tmp_path)
    generated = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*.yaml"))
    assert generated, "make_configs wrote nothing"
    for rel in generated:
        committed = CONFIGS / rel
        assert committed.is_file(), f"{rel} is generated but not committed"
        assert committed.read_text() == (tmp_path / rel).read_text(), f"{rel} is stale"
    assert len([g for g in generated if str(g).startswith("r2/")]) == 9


def test_round_two_configs_change_only_their_flags():
    base = _strip(ExperimentCfg.from_yaml(CONFIGS / "finalA_negcap.yaml").to_dict())
    allowed = {
        "M1": {("data", "match_negatives_on")},
        "W1": {("preprocess", "window_seconds"), ("preprocess", "window_mode")},
        "MW": {("data", "match_negatives_on"), ("preprocess", "window_seconds"),
               ("preprocess", "window_mode"), ("train", "wise_ft"), ("train", "wise_ft_save_all")},
    }
    for variant, fields in allowed.items():
        for seed in (42, 43, 44):
            cfg = _strip(ExperimentCfg.from_yaml(CONFIGS / "r2" / f"r2_{variant}_s{seed}.yaml").to_dict())
            changed = {(sec, key) for sec in base for key in base[sec] if base[sec][key] != cfg[sec][key]}
            expected = set(fields) | ({("train", "seed")} if seed != 42 else set())
            assert changed == expected, (variant, seed, changed)


# ---------------------------------------------------------------------------
# section 6 -- logging
# ---------------------------------------------------------------------------
def test_validation_line_carries_the_global_step_and_within_length_scores():
    from chagas.model import format_val_line

    line = format_val_line({"stage": 2, "epoch": 0, "global_step": 1234, "val_challenge": 0.4,
                            "val_auroc": 0.8, "val_auprc": 0.2, "n": 100,
                            "val_challenge_short": 0.3, "val_auroc_short": 0.7,
                            "windows_per_record": 2.1, "wise_ft_alpha": 0.5})
    assert "step 1234" in line and "short: challenge=0.3000" in line
    assert "windows/record=2.10" in line and "wise_ft_alpha=0.50" in line


def test_within_length_metrics_use_code15_records_of_one_group():
    from chagas.lengths import LENGTH_GROUP_IDS
    from chagas.model import within_length_metrics

    short, long_ = LENGTH_GROUP_IDS["short"], LENGTH_GROUP_IDS["long"]
    labels = np.array([1, 0, 0, 0, 1, 0, 1, 0], dtype=float)
    probs = np.array([0.9, 0.1, 0.2, 0.3, 0.1, 0.9, 0.99, 0.0])
    groups = np.array([short, short, short, short, long_, long_, long_, long_])
    sources = np.array([0, 0, 0, 0, 0, 0, 2, 2])  # last two are PTB-XL-coded: excluded
    out = within_length_metrics(labels, probs, groups, sources)
    assert out["val_auroc_short"] == pytest.approx(1.0)
    assert "val_auroc_long" not in out or out["val_auroc_long"] == pytest.approx(0.0)
    assert within_length_metrics(labels, probs, None, None) == {}


def test_data_summary_counts_split_source_label_length():
    from chagas.dataset import RecordEntry
    from chagas.training import data_summary

    train = [RecordEntry("p", "a", "code15", 1, duration_s=7.3),
             RecordEntry("p", "b", "code15", 0, duration_s=10.24),
             RecordEntry("p", "c", "code15", 0, duration_s=10.24)]
    val = [RecordEntry("p", "d", "ptbxl", 0, duration_s=10.0)]
    rows = {(r["split"], r["source"], r["label"], r["length_group"]): r["n"] for r in data_summary(train, val)}
    assert rows == {("train", "code15", 0, "long"): 2, ("train", "code15", 1, "short"): 1,
                    ("val", "ptbxl", 0, "long"): 1}


def test_run_info_records_provenance(tmp_path):
    import run_info

    out = tmp_path / "run_info.json"
    assert run_info.main(["--out", str(out), "--phase", "start",
                          "--config", str(CONFIGS / "finalA_negcap.yaml")]) == 0
    model = tmp_path / "model"
    model.mkdir()
    (model / "chagas_manifest.json").write_text(json.dumps(
        {"members": ["chagas_model.ckpt"], "training": [{"best_by_stage": {}}],
         "data_summary": [{"split": "train", "n": 3}]}))
    assert run_info.main(["--out", str(out), "--phase", "end", "--model-folder", str(model)]) == 0
    info = json.loads(out.read_text())
    for key in ("config_sha256", "seed", "git", "environment", "start_time", "end_time", "data_summary"):
        assert key in info, key
    assert len(info["config_sha256"]) == 64 and info["seed"] == 42
    assert {"commit", "dirty"} <= set(info["git"])
    assert {"hostname", "torch", "numpy", "pip_freeze_sha256"} <= set(info["environment"])


# ---------------------------------------------------------------------------
# section 3.4 -- the seed ensemble
# ---------------------------------------------------------------------------
class _ConstantLogit(torch.nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def predict_proba(self, signal, valid_len=None, demographics=None, n_windows=None):
        n = signal.shape[0] if n_windows is None else n_windows.shape[0]
        return torch.sigmoid(torch.full((n,), self.logit))


def test_ensemble_averages_member_probabilities():
    from chagas.inference import ChagasPredictor, _Member
    from chagas.preprocessing import Sample

    cfg = default_config()
    members = [_Member(_ConstantLogit(0.0), 0.4), _Member(_ConstantLogit(2.0), 0.6)]
    predictor = ChagasPredictor(members, cfg, torch.device("cpu"))
    sample = Sample(signal=np.zeros((12, 5000), np.float32), valid_len=5000,
                    demographics=np.zeros(5, np.float32))
    prob = predictor.predict_samples([sample, sample])
    expected = (0.5 + 1 / (1 + np.exp(-2.0))) / 2
    assert np.allclose(prob, expected)
    assert predictor.threshold == pytest.approx(0.5)


def test_ensemble_members_split_the_time_budget(monkeypatch, tmp_path):
    import chagas.training as T
    from chagas.dataset import RecordEntry

    cfg = default_config()
    cfg.train.ensemble_members = 3
    cfg.train.max_hours = 6.0
    entries = [RecordEntry(f"/x/{i}", str(i), "code15", int(i % 10 == 0), group=str(i), duration_s=7.3)
               for i in range(60)]
    seen = []

    def fake_member(cfg_, datamodule, work_dir, *, seed, max_hours, deadline, verbose, log_dir=None, wise_dir=None):
        seen.append((seed, max_hours))
        return SimpleNamespace(val_history=[])

    monkeypatch.setattr(T, "find_records", lambda folder: ["/x/0"])
    monkeypatch.setattr(T.L.Code15Table, "from_csv", classmethod(lambda cls, *a, **k: SimpleNamespace(__len__=lambda: 0)))
    monkeypatch.setattr(T, "scan_records", lambda paths, cfg, table=None, workers=8: entries)
    monkeypatch.setattr(T, "ChagasDataModule", lambda cfg, tr, va: None)
    monkeypatch.setattr(T, "_train_member", fake_member)
    monkeypatch.setattr(T, "_save_checkpoint", lambda module, path, cfg, extra_meta=None: Path(path).write_text("x"))
    T.train_model("/unused", str(tmp_path), verbose=False, cfg=cfg)
    assert seen == [(42, 2.0), (43, 2.0), (44, 2.0)]
    manifest = json.loads((tmp_path / "chagas_manifest.json").read_text())
    assert manifest["members"] == ["chagas_model.ckpt", "chagas_model_1.ckpt", "chagas_model_2.ckpt"]
    assert manifest["data_summary"], "train/val counts by source x label x length must be recorded"
