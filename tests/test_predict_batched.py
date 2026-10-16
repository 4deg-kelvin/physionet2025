"""PLAN_R2.md section 7 item 5: batched inference equals the ``run_model.py`` path.

``run_model.py`` calls ``team_code.run_model(record, model)`` once per record, which is
:func:`chagas.inference.run_model`.  ``experiments/predict_batched.py`` must produce the
same probabilities (max |delta| < 1e-4) from batches built by worker processes.  Checked
here on CPU with a seeded random head on 32 smoke records, in both window modes; the
SLURM smoke (``experiments/batch_r2_smoke.sh``) repeats it on all 400 records against
the files the unmodified ``run_model.py`` wrote with a trained model.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from chagas.config import default_config
from _legacy import legacy_config

pytestmark = pytest.mark.backbone

N_RECORDS = 32


def _model_folder(tmp_path, cfg):
    from chagas.model import ChagasModule
    from chagas.training import _save_checkpoint

    torch.manual_seed(0)
    module = ChagasModule(cfg=cfg.to_dict(), stage=1)
    with torch.no_grad():
        # a head with some spread, so probabilities are not all ~0.5
        for p in module.head.parameters():
            p.mul_(20.0)
    module.threshold.fill_(0.5)
    _save_checkpoint(module, str(tmp_path / "chagas_model.ckpt"), cfg)
    (tmp_path / "chagas_manifest.json").write_text(json.dumps({"members": ["chagas_model.ckpt"]}))
    return tmp_path


def _spread(records):
    """Every 12th smoke record: a mix of CODE-15 short/long, PTB-XL and SaMi-Trop."""
    return records[::12][:N_RECORDS]


@pytest.mark.parametrize("mode", ["single", "multi"])
def test_batched_predictions_equal_the_run_model_path(tmp_path, checkpoint_path, real_records, mode):
    import predict_batched as PB

    from chagas.inference import load_model, run_model

    cfg = default_config()
    if mode == "multi":
        cfg.preprocess.window_seconds = 5.0
        cfg.preprocess.window_mode = "multi"
    folder = _model_folder(tmp_path, cfg)
    predictor = load_model(str(folder), device="cpu")
    records = _spread(real_records)

    per_record = np.array([run_model(r, predictor)[1] for r in records], dtype=float)
    batched = PB.predict_records(predictor, records, batch_size=8, num_workers=2)

    assert batched["stem"].tolist() == [r.rsplit("/", 1)[-1] for r in records]
    probs = batched["prob"].to_numpy(dtype=float)
    assert np.isfinite(per_record).all() and np.isfinite(probs).all()
    assert per_record.std() > 1e-3, "degenerate test: all probabilities equal"
    delta = np.abs(probs - per_record).max()
    assert delta < 1e-4, delta
    assert np.array_equal(batched["binary"].to_numpy(), (probs > predictor.threshold).astype(float))


def test_window_stride_override_changes_only_the_preprocessing_config():
    """``predict_batched.py --window-stride`` (round 3 test-time windows), no backbone needed."""
    from chagas.inference import ChagasPredictor
    from predict_batched import with_window_stride

    cfg = legacy_config()
    cfg.preprocess.window_seconds = 5.0
    cfg.preprocess.window_mode = "multi"
    predictor = ChagasPredictor([SimpleNamespace(threshold=0.5)], cfg, torch.device("cpu"))  # members untouched
    with_window_stride(predictor, 2.5)
    assert predictor.cfg.preprocess.window_stride_seconds == 2.5
    assert predictor.cfg.preprocess.window_seconds == 5.0 and predictor.window_mode == "multi"
    assert cfg.preprocess.window_stride_seconds is None, "the loaded config object is not mutated"
    with pytest.raises(ValueError):
        with_window_stride(ChagasPredictor([SimpleNamespace(threshold=0.5)], legacy_config(), torch.device("cpu")), 2.5)  # single mode has no windows to stride
