"""WiSE-FT: interpolate the stage-1 and stage-2 weights and pick the mixture on validation.

Wortsman, Ilharco, Kim, Li, Kornblith, Roelofs, Gontijo-Lopes, Hajishirzi, Farhadi,
Namkoong & Schmidt, *Robust fine-tuning of zero-shot models*, CVPR 2022:

    theta(alpha) = (1 - alpha) * theta_stage1 + alpha * theta_stage2

is the standard remedy for fine-tuning that gains in distribution but loses
out-of-distribution robustness (see also Kumar, Raghunathan, Jones, Ma & Liang,
*Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*,
ICLR 2022).  Here alpha = 0 is the frozen-encoder model (stage 1: aggregation weights and
head trained on a frozen ECG-FM) and alpha = 1 the fully fine-tuned one, so one training
run measures the whole path between them.  On the hidden Chagas cohorts a frozen encoder
beat a fine-tuned one for CinCo Amigos (Wang et al., CinC 2025, DOI
10.22489/CinC.2025.158) and for the team's own entry (Nguyen et al., CinC 2025, DOI
10.22489/CinC.2025.416).  PLAN_R2.md section 3.3.

Both endpoints are the *same* :class:`chagas.model.ChagasModule` (stage switching only
changes ``requires_grad``), so their state dicts share keys and shapes exactly.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch
from torch import Tensor

__all__ = ["snapshot_state", "interpolate_state_dicts", "select_alpha"]


def snapshot_state(module: torch.nn.Module) -> Dict[str, Tensor]:
    """Detached CPU copy of ``module.state_dict()`` (survives later in-place updates)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in module.state_dict().items()}


def interpolate_state_dicts(
    theta0: Mapping[str, Tensor], theta1: Mapping[str, Tensor], alpha: float
) -> Dict[str, Tensor]:
    """``(1 - alpha) * theta0 + alpha * theta1`` for every floating-point entry.

    * ``alpha == 0`` and ``alpha == 1`` return exact copies of ``theta0`` / ``theta1``
      (no arithmetic, so no rounding).
    * Non-floating entries (integer counters, boolean masks) cannot be interpolated;
      they are taken from the nearer endpoint (``theta0`` for ``alpha < 0.5``, else
      ``theta1``).
    * The arithmetic runs in float32 or wider and is cast back to the stored dtype.
    """
    alpha = float(alpha)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1], got {alpha}")
    if set(theta0) != set(theta1):
        missing = sorted(set(theta0) ^ set(theta1))
        raise KeyError(f"state dicts differ in keys: {missing[:5]}")
    out: Dict[str, Tensor] = {}
    for key, a in theta0.items():
        b = theta1[key]
        if a.shape != b.shape:
            raise ValueError(f"{key}: shape {tuple(a.shape)} vs {tuple(b.shape)}")
        if alpha == 0.0:
            out[key] = a.clone()
        elif alpha == 1.0:
            out[key] = b.clone()
        elif not torch.is_floating_point(a):
            out[key] = (a if alpha < 0.5 else b).clone()
        else:
            work = torch.promote_types(a.dtype, torch.float32)
            mixed = (1.0 - alpha) * a.to(work) + alpha * b.to(work)
            out[key] = mixed.to(a.dtype)
    return out


def select_alpha(results: Sequence[Mapping[str, Any]], key: str = "val_challenge") -> Mapping[str, Any]:
    """The result with the highest ``key``; ties go to the **smaller** alpha.

    Ties go towards the frozen end because, at equal validation score, the less
    fine-tuned model is the one with the better out-of-distribution record (see module
    docstring).  ``results`` are dicts with at least ``alpha`` and ``key``.
    """
    if not results:
        raise ValueError("no WiSE-FT results to select from")
    return max(results, key=lambda r: (float(r[key]), -float(r["alpha"])))
