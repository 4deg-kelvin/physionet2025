"""Challenge metric, vectorised, plus AUROC/AUPRC.

The Challenge 2025 score is **TPR@5%**: the recall among the top 5% of records ranked
by predicted probability, which models limited serological testing capacity (Reyna
et al., CinC 2025).  It is not AUROC, and optimising ranking at the head of the
distribution matters more than global calibration.

The organisers' reference implementation (``helper_code.compute_challenge_score``)
estimates the expectation over tie orderings with 10,000 random permutations, each
requiring a full argsort.  At 73,300 held-out records that is ~15 minutes per
evaluation, which makes it unusable as a per-epoch validation metric.
:func:`challenge_score` computes the **same expectation in closed form**; for a record
with probability ``p``,

    P(record lands in the top ``c``) = clip(c - #{q > p}, 0, #{q == p}) / #{q == p}

so ``E[TP] = sum over positives of that probability`` exactly.  With no ties this is
identical to the reference implementation; with ties it is the exact value the
reference implementation is a Monte-Carlo estimate of.  ``tests/test_metrics.py``
checks both statements against ``helper_code`` and against the legacy ``utils`` copy.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = ["challenge_score", "challenge_score_permutation", "compute_auc", "top_k_capacity"]


def _clean(outputs: Sequence[float]) -> np.ndarray:
    """Non-finite predictions rank last: a record we failed on must not be promoted."""
    arr = np.asarray(outputs, dtype=np.float64)
    return np.where(np.isfinite(arr), arr, -np.inf)


def top_k_capacity(n: int, fraction_capacity: float = 0.05) -> int:
    """Number of records the screening budget covers -- ``int(fraction * n)``, as the
    organisers compute it."""
    return int(fraction_capacity * n)


def challenge_score(
    labels: Sequence[float],
    outputs: Sequence[float],
    fraction_capacity: float = 0.05,
) -> float:
    """Expected TPR among the top ``fraction_capacity`` of records, ties handled exactly.

    Returns ``nan`` when there are no positives, matching the reference implementation.
    """
    labels_arr = np.asarray(labels, dtype=np.float64)
    outputs_arr = _clean(outputs)
    if labels_arr.shape[0] != outputs_arr.shape[0]:
        raise ValueError("labels and outputs must have the same length")
    n = labels_arr.shape[0]
    if n == 0:
        return float("nan")
    capacity = top_k_capacity(n, fraction_capacity)

    # Group records by predicted value.  np.unique returns ascending values.
    values, inverse, counts = np.unique(outputs_arr, return_inverse=True, return_counts=True)
    n_ge = np.cumsum(counts[::-1])[::-1]          # #{q >= value}
    n_greater = n_ge - counts                      # #{q >  value}
    slots = np.clip(capacity - n_greater, 0, counts).astype(np.float64)
    prob_selected = np.divide(
        slots, counts.astype(np.float64), out=np.zeros_like(slots), where=counts > 0
    )

    expected_tp = float(np.sum(labels_arr * prob_selected[inverse]))
    n_pos = float(np.sum(labels_arr == 1))
    if n_pos <= 0:
        return float("nan")
    return expected_tp / n_pos


def challenge_score_permutation(
    labels: Sequence[float],
    outputs: Sequence[float],
    fraction_capacity: float = 0.05,
    num_permutations: int = 10 ** 4,
    seed: Optional[int] = 12345,
) -> float:
    """The organisers' Monte-Carlo estimator, kept for cross-checking only.

    Byte-for-byte the algorithm in ``helper_code.compute_challenge_score``; do not use
    it in the training loop -- it is O(num_permutations * n log n).
    """
    labels_arr = np.asarray(labels, dtype=np.float64)
    outputs_arr = np.asarray(outputs, dtype=np.float64)
    n = labels_arr.shape[0]
    capacity = top_k_capacity(n, fraction_capacity)
    tp = np.zeros(num_permutations)
    fn = np.zeros(num_permutations)
    if seed is not None:
        np.random.seed(seed)
    for i in range(num_permutations):
        permuted_idx = np.random.permutation(np.arange(n))
        permuted_labels = labels_arr[permuted_idx]
        permuted_outputs = outputs_arr[permuted_idx]
        ordered_idx = np.argsort(permuted_outputs, kind="stable")[::-1]
        ordered_labels = permuted_labels[ordered_idx]
        tp[i] = np.sum(ordered_labels[:capacity] == 1)
        fn[i] = np.sum(ordered_labels[capacity:] == 1)
    mean_tp, mean_fn = float(np.mean(tp)), float(np.mean(fn))
    if mean_tp + mean_fn > 0:
        return mean_tp / (mean_tp + mean_fn)
    return float("nan")


def compute_auc(
    labels: Sequence[float], outputs: Sequence[float]
) -> Tuple[float, float]:
    """``(AUROC, AUPRC)`` on **hard** labels; ``(nan, nan)`` if only one class is present.

    Audit defect D10: the legacy validation path fed raw logits to a binary accuracy
    metric and soft float targets to AUROC.  Both metrics here take probabilities and
    0/1 labels.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(labels, dtype=np.float64)
    p = _clean(outputs)
    if np.unique(y).size < 2:
        return float("nan"), float("nan")
    finite = np.isfinite(p)
    if not finite.all():
        p = np.where(finite, p, np.nanmin(p[finite]) - 1.0 if finite.any() else 0.0)
    return float(roc_auc_score(y, p)), float(average_precision_score(y, p))
