"""PLAN.md section 5 items 8 (LLRD / two-stage), 9 (serialisation) and 10 (time budget)."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
import torch
from torch import nn

from chagas.config import default_config
from chagas.training import TimeLimitCallback, _save_checkpoint


# ---------------------------------------------------------------------------
# item 8 -- two-stage schedule and layer-wise LR decay
# ---------------------------------------------------------------------------
@pytest.mark.backbone
def test_two_stage_freezing_llrd_and_dropout_modes(checkpoint_path):
    """One module, walked through the whole stage-1 -> stage-2 transition.

    Asserted:
      * stage 1 trains only the aggregation weights, FiLM and the head;
      * stage 2 makes the 12 transformer layers trainable and leaves the conv frontend
        out of the graph (the checkpoint's ``feature_grad_mult = 0.0``);
      * the param-group learning rates follow ``base * decay ** (12 - k)``;
      * ``train()`` keeps a frozen encoder in eval mode but lets an unfrozen one train,
        so ``activation_dropout = 0.1`` actually regularises in stage 2;
      * ``encoder_layerdrop`` is 0, so all 12 hook outputs are captured in both modes.
    """
    from chagas.model import ChagasModule

    cfg = default_config()
    # This test exercises the FiLM path explicitly. The shipped default became
    # demographics="none" after ladder row 5b (DECISIONS.md D-2026-09-26-3 / D-2026-09-28-1);
    # the freezing / LLRD / dropout assertions below must not depend on that default.
    cfg.model.demographics = "film"
    module = ChagasModule(cfg=cfg.to_dict(), stage=1)
    extractor = module.feature_extractor
    encoder = extractor.ecg_fm_model

    # -- stage 1 ------------------------------------------------------------
    assert extractor.freeze_encoder is True
    assert not any(p.requires_grad for p in encoder.parameters())
    trainable = {n for n, p in module.named_parameters() if p.requires_grad}
    assert any(n.endswith("layer_weights") for n in trainable)
    assert any(n.startswith("film.") for n in trainable)
    assert any(n.startswith("head.") for n in trainable)
    assert not any(n.startswith("feature_extractor.ecg_fm_model.") for n in trainable)

    module.train()
    assert encoder.training is False, "a frozen encoder must stay in eval mode"

    groups = module.param_groups()
    assert {round(g["lr"], 12) for g in groups} == {cfg.train.stage1_lr}

    # -- stage 2 ------------------------------------------------------------
    module.enter_stage_two()
    assert extractor.freeze_encoder is False
    layers = extractor.transformer_layers
    assert len(layers) == 12
    assert all(p.requires_grad for layer in layers for p in layer.parameters())
    conv = encoder.encoder.feature_extractor
    assert not any(p.requires_grad for p in conv.parameters()), (
        "the conv frontend must stay frozen (feature_grad_mult = 0.0)"
    )
    assert float(encoder.encoder.feature_grad_mult) == 0.0

    module.train()
    assert encoder.training is True, "an unfrozen encoder must follow train()"
    module.eval()
    assert encoder.training is False
    assert float(extractor.transformer_encoder.layerdrop) == 0.0

    # -- LLRD ---------------------------------------------------------------
    groups = module.param_groups()
    layer_param_ids = {
        id(p): k for k, layer in enumerate(layers, start=1) for p in layer.parameters()
    }
    seen = {}
    for group in groups:
        ks = {layer_param_ids[id(p)] for p in group["params"] if id(p) in layer_param_ids}
        if len(ks) == 1:
            seen.setdefault(ks.pop(), group["lr"])
    assert set(seen) == set(range(1, 13))
    for k, lr in seen.items():
        expected = cfg.train.stage2_lr * (cfg.train.llrd ** (12 - k))
        assert lr == pytest.approx(expected, rel=1e-9), (k, lr, expected)
    assert seen[12] == pytest.approx(cfg.train.stage2_lr)

    head_lrs = {
        round(g["lr"], 12)
        for g in groups
        if all(id(p) not in layer_param_ids for p in g["params"])
        and any(p.requires_grad for p in g["params"])
    }
    assert cfg.train.stage2_head_lr in head_lrs

    # -- the aggregation still sees all 12 layers in train mode -------------
    module.train()
    x = torch.randn(1, 12, 2000)
    out = module(x, torch.tensor([2000]), torch.zeros(1, 5))
    assert torch.isfinite(out["logits"]).all()


# ---------------------------------------------------------------------------
# item 9 -- threshold / layer weights / config round trip
# ---------------------------------------------------------------------------
@pytest.mark.backbone
def test_checkpoint_round_trips_threshold_layer_weights_and_config(tmp_path, checkpoint_path):
    from chagas.inference import load_model
    from chagas.model import ChagasModule

    cfg = default_config()
    cfg.name = "roundtrip-test"
    cfg.train.threshold_quantile = 0.95
    module = ChagasModule(cfg=cfg.to_dict(), stage=1)

    with torch.no_grad():
        module.feature_extractor.layer_weights.copy_(torch.linspace(-1.0, 1.0, 12))
    probs = np.linspace(0.0, 1.0, 1001)
    threshold = module.update_threshold(probs)
    assert threshold == pytest.approx(0.95, abs=1e-6)
    expected_weights = module.feature_extractor.layer_weight_values()

    out = tmp_path / "chagas_model.ckpt"
    _save_checkpoint(module, str(out), cfg)
    with open(tmp_path / "chagas_manifest.json", "w") as fh:
        json.dump({"members": ["chagas_model.ckpt"], "config": cfg.to_dict()}, fh)

    # The metadata must be readable without instantiating the backbone.
    raw = torch.load(str(out), map_location="cpu", weights_only=False)
    assert raw["chagas"]["threshold"] == pytest.approx(threshold)
    assert raw["chagas"]["config"]["name"] == "roundtrip-test"
    assert np.allclose(raw["chagas"]["layer_weights"], expected_weights)

    predictor = load_model(str(tmp_path), verbose=False, device="cpu")
    assert predictor.threshold == pytest.approx(threshold, abs=1e-6)
    assert predictor.cfg.name == "roundtrip-test"
    reloaded = predictor.members[0].module
    assert np.allclose(reloaded.feature_extractor.layer_weight_values(), expected_weights)
    assert float(reloaded.threshold) == pytest.approx(threshold, abs=1e-6)
    for (n1, p1), (n2, p2) in zip(
        module.state_dict().items(), reloaded.state_dict().items()
    ):
        assert n1 == n2
        assert torch.equal(p1, p2.cpu()), n1


def test_threshold_is_the_requested_quantile_of_validation_scores():
    """Audit defect D9: a hard 0.5 on a 50%-prior-calibrated model floods the binary column."""
    from chagas.model import ChagasModule

    # No backbone needed: exercise the pure method on a stand-in object.
    cfg = default_config()
    probs = np.random.default_rng(0).random(10_000)
    fake = object.__new__(ChagasModule)
    fake.cfg = cfg
    fake.threshold = torch.tensor(0.5)
    value = ChagasModule.update_threshold(fake, probs)
    assert value == pytest.approx(float(np.quantile(probs, 0.95)), abs=1e-9)
    assert float(fake.threshold) == pytest.approx(value)
    # ~5% of the cohort is flagged, which is the metric's own operating point.
    assert 0.045 < float((probs > value).mean()) < 0.055


# ---------------------------------------------------------------------------
# item 10 -- time budget
# ---------------------------------------------------------------------------
class _TinyModule(__import__("pytorch_lightning").LightningModule):
    """A deliberately trivial module: this test is about the callback, not the backbone."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(4, 1)
        self.seen = 0

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        self.seen += 1
        import time as _t

        _t.sleep(0.02)
        return torch.nn.functional.mse_loss(self(x).squeeze(-1), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.log("val_challenge", float(1.0 / (1.0 + batch_idx)), batch_size=x.shape[0])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


def test_time_limit_callback_stops_training_and_leaves_a_checkpoint(tmp_path):
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint
    from torch.utils.data import DataLoader, TensorDataset

    data = TensorDataset(torch.randn(256, 4), torch.randn(256))
    loader = DataLoader(data, batch_size=4)

    limit = TimeLimitCallback(max_hours=0.002)  # 7.2 seconds
    ckpt = ModelCheckpoint(
        dirpath=str(tmp_path), filename="best", monitor="val_challenge", mode="max"
    )
    module = _TinyModule()
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        default_root_dir=str(tmp_path),
        callbacks=[limit, ckpt],
    )
    trainer.fit(module, loader, loader)

    assert limit.stopped, "the callback did not fire"
    assert trainer.current_epoch < 50, "training ran to completion instead of stopping"
    assert ckpt.best_model_path and os.path.isfile(ckpt.best_model_path)
    state = torch.load(ckpt.best_model_path, map_location="cpu", weights_only=False)
    assert "state_dict" in state and state["state_dict"]
    # And the saved weights actually load back.
    reloaded = _TinyModule()
    reloaded.load_state_dict(state["state_dict"], strict=True)


def test_time_limit_callback_respects_an_absolute_deadline():
    import time as _t

    limit = TimeLimitCallback(max_hours=100.0, deadline=_t.time() - 1.0)
    limit.start_time = _t.time()
    assert limit._expired() is True

    limit = TimeLimitCallback(max_hours=100.0, deadline=_t.time() + 3600.0)
    limit.start_time = _t.time()
    assert limit._expired() is False


# ---------------------------------------------------------------------------
# stage-2 step cadence and negative capping (DECISIONS.md D-2026-09-29-2)
# ---------------------------------------------------------------------------
def test_stage2_step_validation_wiring(tmp_path):
    from chagas.training import _make_trainer

    cfg = default_config()
    cfg.train.accumulate_grad_batches = 4
    # default: legacy per-epoch validation in both stages, no stage-2 step cap
    t1, _, _ = _make_trainer(cfg, tmp_path, stage=1, max_hours=1.0, deadline=None, has_val=True)
    t2, _, _ = _make_trainer(cfg, tmp_path, stage=2, max_hours=1.0, deadline=None, has_val=True)
    assert t1.max_steps == cfg.train.stage1_max_steps
    assert t2.max_steps == -1
    assert t2.check_val_every_n_epoch == 1
    # step cadence: 500 optimiser steps x 4 accumulated batches = every 2000 training batches
    cfg.train.stage2_val_every_steps = 500
    cfg.train.stage2_max_steps = 6000
    t2, _, _ = _make_trainer(cfg, tmp_path, stage=2, max_hours=1.0, deadline=None, has_val=True)
    assert t2.val_check_interval == 2000
    assert t2.check_val_every_n_epoch is None
    assert t2.max_steps == 6000
    # stage 1 is unaffected by the stage-2 settings
    t1, _, _ = _make_trainer(cfg, tmp_path, stage=1, max_hours=1.0, deadline=None, has_val=True)
    assert t1.check_val_every_n_epoch == 1 and t1.max_steps == cfg.train.stage1_max_steps


def test_subsample_keeps_every_positive_and_is_deterministic():
    from types import SimpleNamespace
    from chagas.training import _subsample

    entries = [SimpleNamespace(label_hard=1, i=i) for i in range(30)] + \
              [SimpleNamespace(label_hard=0, i=i) for i in range(1000)]
    out = _subsample(entries, max_records=130, seed=7)
    assert len(out) == 130
    assert sum(e.label_hard for e in out) == 30
    assert [e.i for e in out] == [e.i for e in _subsample(entries, 130, seed=7)]
    assert _subsample(entries, max_records=0, seed=7) == entries          # 0 = no cap
    assert len(_subsample(entries, max_records=5000, seed=7)) == 1030     # cap above size = no-op
