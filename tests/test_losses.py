"""PLAN.md section 5 item 6: losses and the gradient-reversal layer."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from chagas.losses import (
    grad_reverse,
    hard_negative_ranking_loss,
    masked_aux_bce,
    weighted_soft_bce,
)


def test_weighted_soft_bce_matches_hand_computation():
    logits = torch.tensor([0.5, -1.2, 2.0, -0.3])
    targets = torch.tensor([0.8, 0.2, 1.0, 0.0])
    w = 5.0

    expected = []
    for x, y in zip(logits.tolist(), targets.tolist()):
        p = 1.0 / (1.0 + math.exp(-x))
        expected.append(-(w * y * math.log(p) + (1.0 - y) * math.log(1.0 - p)))
    got = weighted_soft_bce(logits, targets, w)
    assert got.item() == pytest.approx(float(np.mean(expected)), rel=1e-6)

    # pos_weight = 1 must reduce to plain BCE.
    plain = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    assert weighted_soft_bce(logits, targets, 1.0).item() == pytest.approx(plain.item())


def test_weighted_soft_bce_shapes_and_gradient():
    logits = torch.randn(8, 1, requires_grad=True)
    targets = torch.rand(8)
    loss = weighted_soft_bce(logits, targets, 5.0)
    loss.backward()
    assert loss.ndim == 0 and torch.isfinite(loss)
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_ranking_loss_is_zero_when_the_margin_is_satisfied():
    # Positives score far above the negatives -> the hinge is inactive.
    logits = torch.tensor([5.0, 4.5, -3.0, -2.5, -4.0])
    soft = torch.tensor([0.8, 1.0, 0.2, 0.2, 0.0])
    hard = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    assert hard_negative_ranking_loss(logits, soft, hard).item() == pytest.approx(0.0)


def test_ranking_loss_is_positive_when_a_hard_negative_outranks_a_positive():
    logits = torch.tensor([0.0, 0.1, 1.0, -3.0, -4.0])
    soft = torch.tensor([0.8, 1.0, 0.2, 0.2, 0.0])
    hard = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0])
    loss = hard_negative_ranking_loss(logits, soft, hard, margin=0.5, top_frac=0.5)
    assert loss.item() > 0.0


def test_ranking_loss_on_a_single_class_batch_is_finite_and_differentiable():
    """The legacy ``TopKRankingLoss`` returned a bare Python int here and crashed
    ``.backward()``."""
    for hard in (torch.zeros(6), torch.ones(6)):
        logits = torch.randn(6, requires_grad=True)
        soft = hard * 0.8 + 0.1
        loss = hard_negative_ranking_loss(logits, soft, hard)
        assert torch.isfinite(loss) and loss.item() == 0.0
        loss.backward()
        assert logits.grad is not None
        assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_masked_aux_bce_ignores_unlabelled_entries():
    logits = torch.tensor([[2.0, -2.0, 3.0]])
    targets = torch.tensor([[1.0, 0.0, 0.0]])
    full = torch.ones(1, 3)
    partial = torch.tensor([[1.0, 1.0, 0.0]])

    ref = torch.nn.functional.binary_cross_entropy_with_logits(
        logits[:, :2], targets[:, :2]
    )
    assert masked_aux_bce(logits, targets, partial).item() == pytest.approx(ref.item(), rel=1e-6)
    assert masked_aux_bce(logits, targets, full).item() != pytest.approx(ref.item())

    # An all-masked batch (every SaMi-Trop / PTB-XL record) contributes nothing but is
    # still differentiable.
    logits = logits.clone().requires_grad_(True)
    loss = masked_aux_bce(logits, targets, torch.zeros(1, 3))
    assert loss.item() == 0.0
    loss.backward()
    assert torch.allclose(logits.grad, torch.zeros_like(logits.grad))


def test_grad_reverse_flips_the_gradient_sign():
    x = torch.tensor([1.0, -2.0], requires_grad=True)
    y = grad_reverse(x, 0.3)
    assert torch.equal(y.detach(), x.detach())  # identity forwards
    y.sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-0.3, -0.3]))

    x = torch.tensor([1.0], requires_grad=True)
    grad_reverse(x, 0.0).sum().backward()
    assert torch.allclose(x.grad, torch.zeros(1))
