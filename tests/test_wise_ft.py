"""PLAN_R2.md section 7 item 4: WiSE-FT weight interpolation (Wortsman et al., CVPR 2022)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from chagas.config import default_config
from chagas.wise_ft import interpolate_state_dicts, select_alpha, snapshot_state


def _pair():
    g = torch.Generator().manual_seed(0)
    a = {
        "w": torch.randn(4, 3, generator=g),
        "h": torch.randn(5, generator=g).half(),
        "count": torch.tensor(3, dtype=torch.long),
        "mask": torch.tensor([True, False]),
    }
    b = {
        "w": torch.randn(4, 3, generator=g),
        "h": torch.randn(5, generator=g).half(),
        "count": torch.tensor(9, dtype=torch.long),
        "mask": torch.tensor([False, True]),
    }
    return a, b


def test_endpoints_reproduce_the_stage_state_dicts_exactly():
    a, b = _pair()
    zero = interpolate_state_dicts(a, b, 0.0)
    one = interpolate_state_dicts(a, b, 1.0)
    for k in a:
        assert torch.equal(zero[k], a[k]) and zero[k].dtype == a[k].dtype
        assert torch.equal(one[k], b[k]) and one[k].dtype == b[k].dtype
    zero["w"].add_(1.0)
    assert not torch.equal(zero["w"], a["w"]), "results must be copies, not views"


def test_half_is_the_elementwise_mean_and_non_float_entries_are_handled():
    a, b = _pair()
    mid = interpolate_state_dicts(a, b, 0.5)
    assert torch.allclose(mid["w"], (a["w"] + b["w"]) / 2, atol=1e-7)
    assert mid["h"].dtype == torch.float16
    assert torch.allclose(mid["h"].float(), (a["h"].float() + b["h"].float()) / 2, atol=1e-3)
    # integers and booleans cannot be averaged: nearest endpoint (alpha >= 0.5 -> stage 2)
    assert torch.equal(mid["count"], b["count"]) and torch.equal(mid["mask"], b["mask"])
    low = interpolate_state_dicts(a, b, 0.25)
    assert torch.equal(low["count"], a["count"])
    with pytest.raises(ValueError):
        interpolate_state_dicts(a, b, 1.5)
    with pytest.raises(KeyError):
        interpolate_state_dicts(a, {k: v for k, v in b.items() if k != "w"}, 0.5)


def test_select_alpha_takes_the_best_and_breaks_ties_towards_the_frozen_end():
    res = [{"alpha": 0.0, "val_challenge": 0.40}, {"alpha": 0.5, "val_challenge": 0.45},
           {"alpha": 1.0, "val_challenge": 0.45}]
    assert select_alpha(res)["alpha"] == 0.5
    assert select_alpha(res[:1])["alpha"] == 0.0
    with pytest.raises(ValueError):
        select_alpha([])


class _ToyModule(nn.Module):
    """Stand-in exposing exactly what ``_wise_ft_select`` touches."""

    def __init__(self, value: float) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.tensor([value]))
        self.register_buffer("threshold", torch.tensor(0.5))
        self.val_history: list = []
        self.last_val_probs = None
        self.wise_ft_alpha = None

    def update_threshold(self, probs):
        self.threshold.fill_(float(np.quantile(probs, 0.95)))
        return float(self.threshold)


class _ToyTrainer:
    """``validate`` scores the current weight: best when w == 0.3."""

    def validate(self, module, datamodule=None, verbose=False):
        w = float(module.w)
        module.last_val_probs = np.linspace(0, w, 101)
        module.val_history.append({"val_challenge": -abs(w - 0.3), "wise_ft_alpha": module.wise_ft_alpha})


def test_wise_ft_select_scores_every_alpha_and_keeps_the_best():
    from chagas.training import _summarise_val_history, _wise_ft_select

    cfg = default_config()
    cfg.train.wise_ft = "select"
    cfg.train.wise_ft_alphas = (0.0, 0.25, 0.5, 1.0)
    module = _ToyModule(1.0)                       # stage-2 weights: w = 1
    theta1 = snapshot_state(_ToyModule(0.0))      # stage-1 weights: w = 0
    out = _wise_ft_select(module, theta1, _ToyTrainer(), None, cfg, wise_dir=None, verbose=False)
    assert out.wise_ft_selected_alpha == 0.25     # w(0.25) = 0.25 is closest to 0.3
    assert float(out.w) == pytest.approx(0.25)
    assert [r["alpha"] for r in out.wise_ft_results] == [0.0, 0.25, 0.5, 1.0]
    assert float(out.threshold) == pytest.approx(out.wise_ft_results[1]["threshold"])
    assert out.wise_ft_alpha is None, "later validation passes must not be tagged"
    summary = _summarise_val_history(0, out)
    assert summary["wise_ft"]["selected_alpha"] == 0.25
    assert summary["best_by_stage"] == {}, "mixture passes must not count as stage results"


@pytest.mark.backbone
def test_selected_alpha_is_stored_in_the_checkpoint_and_reloaded(tmp_path, checkpoint_path):
    from chagas.inference import load_model
    from chagas.model import ChagasModule
    from chagas.training import _save_checkpoint

    cfg = default_config()
    module = ChagasModule(cfg=cfg.to_dict(), stage=2)
    _save_checkpoint(module, str(tmp_path / "chagas_model.ckpt"), cfg, extra_meta={"wise_ft_alpha": 0.5})
    (tmp_path / "chagas_manifest.json").write_text(json.dumps({"members": ["chagas_model.ckpt"]}))
    raw = torch.load(str(tmp_path / "chagas_model.ckpt"), map_location="cpu", weights_only=False)
    assert raw["chagas"]["wise_ft_alpha"] == 0.5
    predictor = load_model(str(tmp_path), device="cpu")
    assert predictor.members[0].wise_ft_alpha == 0.5
    assert predictor.wise_ft_alphas == [0.5]
    # without WiSE-FT the checkpoint keeps the round-1 layout
    _save_checkpoint(module, str(tmp_path / "plain.ckpt"), cfg)
    raw = torch.load(str(tmp_path / "plain.ckpt"), map_location="cpu", weights_only=False)
    assert "wise_ft_alpha" not in raw["chagas"]
