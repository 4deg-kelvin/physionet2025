"""Losses (PLAN.md section 3.2) and the gradient-reversal layer (section 3.6).

The repo's previous ranking losses (``fm.TopKRankingLoss``, ``fm.ChallengeScoreLoss``,
``fm.DAM_Momentum``) were unreachable from the submission path and numerically unsafe --
``TopKRankingLoss`` returns a bare Python ``int`` when a batch has no positive/negative
pair, which crashes ``.backward()``, and its "missed positives" term has zero gradient.
They are replaced here, not extended.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.autograd import Function

__all__ = [
    "weighted_soft_bce",
    "WeightedSoftBCE",
    "hard_negative_ranking_loss",
    "masked_aux_bce",
    "GradientReversal",
    "grad_reverse",
]


# ---------------------------------------------------------------------------
# primary loss
# ---------------------------------------------------------------------------
def weighted_soft_bce(
    logits: Tensor, targets: Tensor, pos_weight: float = 5.0
) -> Tensor:
    """``BCEWithLogitsLoss(pos_weight)`` on soft targets, mean-reduced.

    ``pos_weight = 5`` is the winner's setting (Van Santvliet et al., CinC 2025) and is
    also what the reliability-aware model of Physiol. Meas. 2026 used.  It is far milder
    than the legacy ~50/50 inverse-frequency ``WeightedRandomSampler``, which at 2%
    prevalence effectively re-weights positives 43x (audit defect D5).

    With a soft target ``y``, the per-element loss is
    ``-[w*y*log sigmoid(x) + (1-y)*log(1 - sigmoid(x))]``, i.e. the positive term alone
    is up-weighted -- which is exactly what ``torch.nn.BCEWithLogitsLoss`` does.
    """
    logits = logits.reshape(-1)
    targets = targets.reshape(-1).to(logits.dtype)
    weight = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=weight)


class WeightedSoftBCE(nn.Module):
    """Module wrapper around :func:`weighted_soft_bce`."""

    def __init__(self, pos_weight: float = 5.0) -> None:
        super().__init__()
        self.pos_weight = float(pos_weight)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        return weighted_soft_bce(logits, targets, self.pos_weight)


# ---------------------------------------------------------------------------
# ranking loss
# ---------------------------------------------------------------------------
def hard_negative_ranking_loss(
    logits: Tensor,
    soft_targets: Tensor,
    hard_labels: Tensor,
    margin: float = 0.5,
    top_frac: float = 0.2,
) -> Tensor:
    """Pairwise hinge between positives and the batch's highest-scoring negatives.

    Hong, Chou & Gonzalez, CinC 2025, eq. 2-3.  The challenge metric is a ranking
    metric (recall in the top 5%), so a loss that explicitly pushes true positives above
    the negatives the model currently likes best is better matched to it than BCE alone.
    The margin is scaled by the soft-label gap, so a pair whose labels are nearly equally
    uncertain is asked for a proportionally smaller separation.

    Returns a zero tensor that is still attached to the graph when the batch has no
    positive/negative pair -- a bare ``0`` there is what made the legacy
    ``TopKRankingLoss`` crash ``.backward()``.
    """
    scores = logits.reshape(-1)
    soft = soft_targets.reshape(-1).to(scores.dtype)
    hard = hard_labels.reshape(-1)

    pos_mask = hard > 0.5
    neg_mask = ~pos_mask
    zero = scores.sum() * 0.0
    if not bool(pos_mask.any()) or not bool(neg_mask.any()):
        return zero

    pos_scores = scores[pos_mask]
    neg_scores = scores[neg_mask]
    pos_soft = soft[pos_mask]
    neg_soft = soft[neg_mask]

    # Hard negatives: the top `top_frac` highest-scoring negatives in the batch.
    k = max(1, int(round(float(top_frac) * neg_scores.numel())))
    k = min(k, neg_scores.numel())
    top_idx = torch.topk(neg_scores, k=k, largest=True).indices
    neg_scores = neg_scores[top_idx]
    neg_soft = neg_soft[top_idx]

    gap = (pos_soft.unsqueeze(1) - neg_soft.unsqueeze(0)).clamp(min=0.0)
    required = float(margin) * gap
    diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
    return F.relu(required - diff).mean()


# ---------------------------------------------------------------------------
# auxiliary heads
# ---------------------------------------------------------------------------
def masked_aux_bce(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Binary cross-entropy over auxiliary targets, applied only where they exist.

    Team idea (the dev tree's ``train_multitask_fm.py``); motivated by Jidling et al.,
    PLoS NTD 2023 -- the network's evidence for Chagas is RBBB/LAFB-type conduction
    disease -- and by Caruana, *Multitask Learning*, Mach. Learn. 1997.

    ``mask`` is 1 where the label is known.  Every SaMi-Trop, PTB-XL and hidden-cohort
    record has an all-zero mask, so they contribute nothing rather than being trained
    towards a fabricated negative.
    """
    mask = mask.to(logits.dtype)
    denom = mask.sum()
    if float(denom) <= 0.0:
        return logits.sum() * 0.0
    per_element = F.binary_cross_entropy_with_logits(
        logits, targets.to(logits.dtype), reduction="none"
    )
    return (per_element * mask).sum() / denom


# ---------------------------------------------------------------------------
# gradient reversal
# ---------------------------------------------------------------------------
class _GradReverse(Function):
    @staticmethod
    def forward(ctx, x: Tensor, lambd: float) -> Tensor:  # type: ignore[override]
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: Tensor, lambd: float = 1.0) -> Tensor:
    """Identity forwards, ``-lambda * grad`` backwards.

    Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 2016.  Applied
    to Chagas by "CinCo Amigos" (CinC 2025), the only top-15 team using it -- and the
    one with the best ELSA-Brasil score of the top 15.
    """
    return _GradReverse.apply(x, lambd)


class GradientReversal(nn.Module):
    """Module form of :func:`grad_reverse` with a mutable ``lambd`` for ramping 0 -> 0.1."""

    def __init__(self, lambd: float = 0.0) -> None:
        super().__init__()
        self.lambd = float(lambd)

    def forward(self, x: Tensor) -> Tensor:
        return grad_reverse(x, self.lambd)

    def extra_repr(self) -> str:
        return f"lambd={self.lambd}"
