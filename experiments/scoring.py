"""Scoring library for round 2: length-aware views, diagnostics and a fast paired bootstrap.

Used by ``eval_report.py`` (one run), ``aggregate.py`` (seeds, ensembles, paired
differences) and ``rescore_all.py`` (every saved run).  PLAN_R2.md section 2.

Views (all on one set of saved predictions; the S-views are CODE-15 only, so the source
cue is absent too):

========  ==========================================================  ==================
``A``     full held-out set                                            continuity only
``B``     CODE-15 only                                                 continuity only
``S7``    CODE-15, duration < 7.5 s                                    **primary**
``S10``   CODE-15, duration >= 9.9 s                                   closest to hidden
``S10+``  CODE-15 >= 9.9 s negatives + every >= 9.9 s positive         more power, secondary
========  ==========================================================  ==================

The length-shortcut discovery behind these views and the within-length evaluation are
this project's own (DECISIONS.md D-2026-09-30-2).

The bootstrap
-------------
Resampling ``n`` records with replacement is equivalent to giving record ``i`` an integer
weight ``w_i`` (how often it was drawn).  Duplicated records are ties, and both metrics
here treat ties exactly (the challenge score through the closed form in
:mod:`chagas.metrics`, AUROC as Mann-Whitney with 1/2 per tie), so a metric on the
resampled array equals the same metric computed from the *original* sorted scores with
weights ``w``.  That needs one sort per model instead of one per resample.
``tests/test_eval_r2.py`` checks the weighted path against explicit resampling.  Because
the weights depend only on ``(seed, n)`` (and the patient groups, below), every model
scored on the same records with the same seed sees the same resamples -- the paired
bootstrap of ``aggregate.py``.

Patient-level (cluster) resampling
----------------------------------
CODE-15% has about 1.47 examinations per patient, so held-out records are not
independent.  By default (``bootstrap_unit="patient"``) the bootstrap resamples
**patients**: each resample draws ``G`` patient groups with replacement (``G`` = the
number of distinct groups among the view's records) and every record inherits the count
of its group -- the cluster bootstrap (Davison & Hinkley 1997, *Bootstrap Methods and
their Application*, section 3.8; Field & Welsh 2007, JRSS-B 69:369).  Groups are the
``group`` column of the split CSV (``load_groups``): ``c15_<patient>`` for CODE-15%,
``ptb_<patient>`` for PTB-XL, ``smt_<stem>`` for SaMi-Trop.  ``"record"`` reproduces the
earlier record-level numbers; with every group a single record the two are bit-for-bit
equal (tested), since group codes follow the records' order of first appearance.
"""

from __future__ import annotations

import sys
import zlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chagas.lengths import LONG_MIN_SECONDS, SHORT_MAX_SECONDS, length_group  # noqa: E402
from chagas.metrics import challenge_score, compute_auc  # noqa: E402

DEFAULT_DURATIONS = REPO_ROOT / "experiments" / "data" / "record_durations.csv"
DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 20260930
#: Patient groups of every record (``stem -> group``), from the held-out split.
DEFAULT_SPLIT = Path("/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/split/split.csv")
#: ``"patient"`` (cluster bootstrap over patient groups) or ``"record"`` (the original).
BOOTSTRAP_UNITS: Tuple[str, ...] = ("patient", "record")
DEFAULT_BOOTSTRAP_UNIT = "patient"

__all__ = [
    "VIEW_ORDER",
    "VIEW_DESCRIPTIONS",
    "load_durations",
    "attach_durations",
    "view_mask",
    "RankedScores",
    "load_groups",
    "group_codes",
    "bootstrap_weights",
    "bootstrap_metrics",
    "view_seed",
    "score_view",
    "duration_only",
    "length_cue_gap",
    "score_all_views",
    "ci",
]

VIEW_ORDER: Tuple[str, ...] = ("A", "B", "S7", "S10", "S10+")
VIEW_DESCRIPTIONS: Dict[str, str] = {
    "A": "full held-out (source and length cues available)",
    "B": "CODE-15 only (source cue removed; length cue still available)",
    "S7": "CODE-15, duration < 7.5 s (no source or length cue) -- PRIMARY",
    "S10": "CODE-15, duration >= 9.9 s (closest format to the hidden cohorts)",
    "S10+": "CODE-15 >= 9.9 s negatives + all >= 9.9 s positives (secondary)",
}


# ---------------------------------------------------------------------------
# durations and views
# ---------------------------------------------------------------------------
def load_durations(path: Optional[str] = None) -> pd.DataFrame:
    """``stem, duration_s`` from ``experiments/data/record_durations.csv``."""
    df = pd.read_csv(path or DEFAULT_DURATIONS, dtype={"stem": str})
    return df[["stem", "duration_s"]]


def attach_durations(df: pd.DataFrame, durations: pd.DataFrame) -> pd.DataFrame:
    """Add ``duration_s`` and ``length_group`` to a predictions frame (by ``stem``)."""
    out = df.copy()
    out["stem"] = out["stem"].astype(str)
    if "duration_s" in out.columns:
        out = out.drop(columns=["duration_s"])
    out = out.merge(durations, on="stem", how="left")
    out["length_group"] = [length_group(d) for d in out["duration_s"].to_numpy(dtype=float)]
    return out


def _is_code15(df: pd.DataFrame) -> np.ndarray:
    return df["source"].astype(str).str.upper().str.startswith("CODE").to_numpy()


def view_mask(df: pd.DataFrame, view: str) -> np.ndarray:
    """Boolean mask selecting the records of ``view`` (see module docstring)."""
    dur = df["duration_s"].to_numpy(dtype=float) if "duration_s" in df.columns else None
    if view == "A":
        return np.ones(len(df), dtype=bool)
    code15 = _is_code15(df)
    if view == "B":
        return code15
    if dur is None:
        raise KeyError("duration_s is needed for the S-views; call attach_durations first")
    with np.errstate(invalid="ignore"):
        short = dur < SHORT_MAX_SECONDS
        long_ = dur >= LONG_MIN_SECONDS
    if view == "S7":
        return code15 & short
    if view == "S10":
        return code15 & long_
    if view == "S10+":
        positive = df["label"].to_numpy(dtype=float) == 1
        return long_ & (code15 | positive)
    raise KeyError(f"unknown view {view!r}")


# ---------------------------------------------------------------------------
# weighted metrics on pre-sorted scores
# ---------------------------------------------------------------------------
@dataclass
class RankedScores:
    """One model's scores, grouped into tie groups sorted by **descending** score.

    ``group`` maps each record to its tie group; group 0 holds the highest score.
    """

    y: np.ndarray
    group: np.ndarray
    n_groups: int

    @classmethod
    def build(cls, y: np.ndarray, p: np.ndarray) -> "RankedScores":
        p = np.asarray(p, dtype=np.float64)
        p = np.where(np.isfinite(p), p, -np.inf)  # failures rank last, as chagas.metrics
        values, inverse = np.unique(-p, return_inverse=True)  # ascending -p == descending p
        return cls(np.asarray(y, dtype=np.float64), inverse.astype(np.int64), int(values.size))

    def metrics(self, weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """``(challenge, auroc)`` for each row of ``weights`` (shape ``(B, n)``)."""
        w = np.atleast_2d(np.asarray(weights, dtype=np.float64))
        b = w.shape[0]
        offsets = (np.arange(b, dtype=np.int64) * self.n_groups)[:, None]
        flat = (self.group[None, :] + offsets).ravel()
        size = b * self.n_groups
        wg = np.bincount(flat, weights=w.ravel(), minlength=size).reshape(b, self.n_groups)
        pg = np.bincount(flat, weights=(w * self.y[None, :]).ravel(), minlength=size).reshape(b, self.n_groups)
        ng = wg - pg
        total = w.sum(axis=1)
        n_pos = pg.sum(axis=1)
        n_neg = ng.sum(axis=1)

        # challenge score: expected recall in the top int(0.05 * n), ties shared evenly
        capacity = np.floor(0.05 * total)  # == int(0.05 * n), as chagas.metrics.top_k_capacity
        n_greater = np.cumsum(wg, axis=1) - wg
        slots = np.clip(capacity[:, None] - n_greater, 0.0, wg)
        frac = np.divide(slots, wg, out=np.zeros_like(slots), where=wg > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            challenge = (pg * frac).sum(axis=1) / n_pos
            # AUROC: a positive beats every negative in a lower group, half a tie
            neg_below = n_neg[:, None] - np.cumsum(ng, axis=1)
            auroc = (pg * (neg_below + 0.5 * ng)).sum(axis=1) / (n_pos * n_neg)
        challenge = np.where(n_pos > 0, challenge, np.nan)
        auroc = np.where((n_pos > 0) & (n_neg > 0), auroc, np.nan)
        return challenge, auroc


def view_seed(view: str, base_seed: int = DEFAULT_SEED) -> int:
    """Deterministic per-view seed, so every view has its own (reproducible) resamples."""
    return (int(base_seed) + zlib.crc32(view.encode())) % (2 ** 32)


# ---------------------------------------------------------------------------
# patient groups
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def _load_groups_cached(path: str) -> Dict[str, str]:
    df = pd.read_csv(path, usecols=["stem", "group"], dtype={"stem": str, "group": str})
    if df["stem"].duplicated().any():
        raise ValueError(f"{path}: duplicated stems; cannot map records to patient groups")
    return dict(zip(df["stem"], df["group"]))


def load_groups(path: Optional[str] = None) -> Dict[str, str]:
    """``stem -> patient group`` from the split CSV (cached per path).

    The ``group`` column is ``c15_<patient>`` (CODE-15%), ``ptb_<patient>`` (PTB-XL) or
    ``smt_<stem>`` (SaMi-Trop, one record per patient).
    """
    return _load_groups_cached(str(path or DEFAULT_SPLIT))


def group_codes(
    stems: Sequence[str],
    unit: str = DEFAULT_BOOTSTRAP_UNIT,
    groups: Optional[Mapping[str, str]] = None,
) -> Optional[np.ndarray]:
    """Integer resampling-unit code of each record, or ``None`` for record-level resampling.

    Codes number the groups in order of first appearance among ``stems`` (not sorted),
    so that when every group holds one record the codes are ``0..n-1`` and the patient
    bootstrap draws exactly the record bootstrap's weights.  ``groups`` defaults to
    :func:`load_groups`; a stem missing from it raises ``KeyError``.
    """
    if unit not in BOOTSTRAP_UNITS:
        raise ValueError(f"unknown bootstrap unit {unit!r}; choose from {BOOTSTRAP_UNITS}")
    if unit == "record":
        return None
    mapping = load_groups() if groups is None else groups
    stems = [str(s) for s in stems]
    missing = [s for s in stems if s not in mapping]
    if missing:
        raise KeyError(
            f"{len(missing)} of {len(stems)} records have no patient group in the split map "
            f"(first: {missing[:5]}); pass the split CSV that produced these predictions, or "
            f"use bootstrap_unit='record'"
        )
    codes, _ = pd.factorize(pd.Series([mapping[s] for s in stems], dtype=object), sort=False)
    return codes.astype(np.int64)


def bootstrap_weights(
    n: int, n_boot: int, seed: int, chunk: int = 50, groups: Optional[np.ndarray] = None
) -> Iterator[np.ndarray]:
    """Yield ``(<= chunk, n)`` integer weight matrices; ``n_boot`` rows in total.

    Record level (``groups=None``): row ``r`` is ``bincount`` of ``n`` indices drawn with
    replacement -- one ordinary bootstrap resample of the ``n`` records.

    Patient level (``groups`` = :func:`group_codes`, values ``0..G-1``): row ``r`` is the
    ``bincount`` of ``G`` group indices drawn with replacement, expanded so that each
    record carries its group's count -- one cluster-bootstrap resample.

    Depends only on ``(n, n_boot, seed, chunk, groups)``, which is what makes the
    bootstrap *paired* across models.
    """
    if groups is not None:
        groups = np.asarray(groups, dtype=np.int64)
        if groups.shape != (n,):
            raise ValueError(f"groups has shape {groups.shape}, expected ({n},)")
    n_units = n if groups is None else (int(groups.max()) + 1 if n else 0)
    rng = np.random.default_rng(seed)
    done = 0
    while done < n_boot:
        b = min(chunk, n_boot - done)
        idx = rng.integers(0, n_units, size=(b, n_units))
        flat = (idx + (np.arange(b, dtype=np.int64) * n_units)[:, None]).ravel()
        counts = np.bincount(flat, minlength=b * n_units).reshape(b, n_units).astype(np.float64)
        yield counts if groups is None else counts[:, groups]
        done += b


def bootstrap_metrics(
    y: np.ndarray,
    probs: Sequence[np.ndarray],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    groups: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Paired bootstrap of ``(challenge, auroc)`` for several models on the same records.

    ``groups`` (:func:`group_codes`) switches to patient-level resampling.  Returns two
    arrays of shape ``(n_models, n_boot)``.
    """
    ranked = [RankedScores.build(y, p) for p in probs]
    ch = np.empty((len(ranked), n_boot))
    au = np.empty((len(ranked), n_boot))
    start = 0
    for w in bootstrap_weights(len(y), n_boot, seed, groups=groups):
        stop = start + w.shape[0]
        for i, r in enumerate(ranked):
            ch[i, start:stop], au[i, start:stop] = r.metrics(w)
        start = stop
    return ch, au


def ci(samples: np.ndarray, level: float = 0.95) -> Tuple[float, float, float]:
    """``(low, high, se)`` percentile interval and standard error of bootstrap samples."""
    s = np.asarray(samples, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return float("nan"), float("nan"), float("nan")
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(s, [tail, 100.0 - tail])
    return float(lo), float(hi), float(s.std(ddof=1)) if s.size > 1 else float("nan")


# ---------------------------------------------------------------------------
# one view, the diagnostics
# ---------------------------------------------------------------------------
def score_view(
    df: pd.DataFrame,
    view: str,
    *,
    label: Optional[str] = None,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    prob_col: str = "prob_filled",
    unit: str = DEFAULT_BOOTSTRAP_UNIT,
    groups: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    """Point estimates and bootstrap 95% CIs for one view of one prediction set.

    ``unit`` / ``groups``: resampling unit and ``stem -> group`` map (:func:`group_codes`).
    """
    sub = df[view_mask(df, view)]
    name = label or view
    if len(sub) == 0 or sub["label"].nunique() < 2:
        return {"view": name, "n": int(len(sub)), "note": "not scorable"}
    y = sub["label"].to_numpy(dtype=float)
    p = sub[prob_col].to_numpy(dtype=float)
    prevalence = float(y.mean())
    cs = challenge_score(y, p)
    auroc, auprc = compute_auc(y, p)
    row: Dict[str, object] = {
        "view": name,
        "n": int(len(sub)),
        "positives": int(y.sum()),
        "prevalence": prevalence,
        "challenge": float(cs),
        "ceiling": float(min(1.0, 0.05 / prevalence)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "n_failed": int(sub["prob"].isna().sum()) if "prob" in sub.columns else 0,
    }
    if n_boot > 0:
        codes = group_codes(sub["stem"].astype(str).tolist(), unit, groups)
        ch, au = bootstrap_metrics(y, [p], n_boot, view_seed(view, seed), groups=codes)
        row["challenge_lo"], row["challenge_hi"], row["challenge_se"] = ci(ch[0])
        row["auroc_lo"], row["auroc_hi"], row["auroc_se"] = ci(au[0])
    return row


def duration_only(df: pd.DataFrame, view: str) -> Dict[str, object]:
    """Score of a model that knows only the length: 1 for short (< 7.5 s) records, else 0."""
    sub = df[view_mask(df, view)]
    if len(sub) == 0 or sub["label"].nunique() < 2:
        return {"view": view, "n": int(len(sub)), "note": "not scorable"}
    y = sub["label"].to_numpy(dtype=float)
    score = (sub["duration_s"].to_numpy(dtype=float) < SHORT_MAX_SECONDS).astype(float)
    return {
        "view": view,
        "n": int(len(sub)),
        "challenge": float(challenge_score(y, score)),
        "auroc": float(compute_auc(y, score)[0]),
    }


def length_cue_gap(df: pd.DataFrame, prob_col: str = "prob_filled") -> Dict[str, float]:
    """Mean predicted probability of CODE-15 **negatives** per length group, and the gap.

    ``gap = mean(short) - mean(long)``.  A model that does not use record length has a
    gap of ~0; the shipped round-1 model's is ~0.23 (0.62 vs 0.39).
    """
    neg = df[_is_code15(df) & (df["label"].to_numpy(dtype=float) == 0)]
    out: Dict[str, float] = {}
    for group in ("short", "mid", "long"):
        vals = neg.loc[neg["length_group"] == group, prob_col].to_numpy(dtype=float)
        out[f"neg_mean_{group}"] = float(vals.mean()) if vals.size else float("nan")
        out[f"neg_n_{group}"] = float(vals.size)
    out["gap"] = out["neg_mean_short"] - out["neg_mean_long"]
    return out


def score_all_views(
    df: pd.DataFrame,
    *,
    suffix: str = "",
    views: Sequence[str] = VIEW_ORDER,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    unit: str = DEFAULT_BOOTSTRAP_UNIT,
    groups: Optional[Mapping[str, str]] = None,
) -> List[Dict[str, object]]:
    """:func:`score_view` for every view; ``suffix`` tags the rows (e.g. ``"·C"``)."""
    return [score_view(df, v, label=f"{v}{suffix}", n_boot=n_boot, seed=seed, unit=unit, groups=groups)
            for v in views]

