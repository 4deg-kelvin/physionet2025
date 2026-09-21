"""Header parsing, source-aware soft labels, auxiliary labels and patient groups.

PLAN.md section 3.2.  Nothing here touches the signal.

Soft labels exist because the three training cohorts do not carry the same kind of
label: SaMi-Trop records are serologically confirmed positives, PTB-XL is a European
cohort assumed negative, and CODE-15's labels are **self-reported**.  Training a hard
0/1 target on all three teaches the network that a self-report is as certain as a
serology result.  Source-aware label smoothing was used by every published top-15 team:
Van Santvliet et al. (CinC 2025), Hong et al. (CinC 2025) and the reliability-aware
hierarchical model of Physiol. Meas. 2026 (doi:10.1088/1361-6579/ae98e0).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .config import LabelCfg

__all__ = [
    "PatientInfo",
    "parse_patient_info",
    "normalise_source",
    "Code15Table",
    "MultiReportStats",
    "soft_label",
    "aux_targets",
    "patient_group",
    "SOURCE_IDS",
]

_AGE_KEY = "# Age:"
_SEX_KEY = "# Sex:"
_SOURCE_KEY = "# Source:"
_LABEL_KEY = "# Chagas label:"

#: Stable integer ids for the domain-adversarial head (Ganin et al., JMLR 2016).
SOURCE_IDS: Dict[str, int] = {"code15": 0, "samitrop": 1, "ptbxl": 2, "unknown": 0}

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f", ""}


@dataclass
class PatientInfo:
    """What a WFDB header tells us about the patient and the cohort."""

    age: Optional[float] = None
    sex: Optional[str] = None
    label: Optional[int] = None
    source: str = "unknown"


def _to_bool(value: object) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        if np.isnan(value):
            return None
        return bool(int(value))
    text = str(value).strip().casefold()
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return None


def normalise_source(raw: Optional[str]) -> str:
    """Map a ``# Source:`` value onto ``code15`` / ``samitrop`` / ``ptbxl`` / ``unknown``.

    An unrecognised source is deliberately **not** an error: the hidden cohorts
    (REDS-II, SaMi-Trop 3, ELSA-Brasil) will not be any of the three training names, and
    inference must still work.  Soft labelling treats an unknown source as CODE-15-like,
    which is the conservative choice (PLAN.md section 3.2).
    """
    if not raw:
        return "unknown"
    text = str(raw).strip().casefold()
    if "sami" in text:
        return "samitrop"
    if "ptb" in text:
        return "ptbxl"
    if "code" in text:
        return "code15"
    return "unknown"


def parse_patient_info(header: str, want_label: bool = False) -> PatientInfo:
    """Single-pass parse of age, sex, source and (optionally) the Chagas label.

    ``want_label=True`` on a header that has no label raises
    :class:`NotImplementedError` -- the codebase's signal for "you are trying to read
    held-out labels".  ``team_code.run_model`` deliberately re-raises that exception
    rather than degrading to ``(None, None)``.
    """
    info = PatientInfo()
    has_label = False
    for line in header.split("\n"):
        if line.startswith(_AGE_KEY):
            text = line[len(_AGE_KEY):].strip()
            try:
                info.age = float(text)
            except ValueError:
                info.age = None
        elif line.startswith(_SEX_KEY):
            info.sex = line[len(_SEX_KEY):].strip() or None
        elif line.startswith(_SOURCE_KEY):
            info.source = normalise_source(line[len(_SOURCE_KEY):].strip())
        elif line.startswith(_LABEL_KEY):
            has_label = True
            flag = _to_bool(line[len(_LABEL_KEY):].strip())
            info.label = None if flag is None else int(flag)
    if want_label and not has_label:
        raise NotImplementedError(
            "No label is available: are you trying to load the labels from the held-out data?"
        )
    return info


# ---------------------------------------------------------------------------
# code15_exams.csv
# ---------------------------------------------------------------------------
class Code15Table:
    """Lookup over the shipped ``code15_exams.csv``: patient ids and comorbidity flags.

    ``exam_id`` is the record stem for CODE-15 records.  Missing rows are normal (the
    hidden cohorts are not in this CSV), so every accessor returns a "no information"
    answer rather than raising.
    """

    def __init__(
        self,
        patient_of: Optional[Mapping[str, str]] = None,
        flags: Optional[Mapping[str, Dict[str, Optional[bool]]]] = None,
        columns: Sequence[str] = (),
    ) -> None:
        self._patient_of: Dict[str, str] = dict(patient_of or {})
        self._flags: Dict[str, Dict[str, Optional[bool]]] = dict(flags or {})
        self.columns: Tuple[str, ...] = tuple(columns)

    # -- construction ------------------------------------------------------
    @classmethod
    def empty(cls, columns: Sequence[str] = ()) -> "Code15Table":
        return cls({}, {}, columns)

    @classmethod
    def from_csv(
        cls, path: Optional[Path], columns: Sequence[str] = ()
    ) -> "Code15Table":
        """Load the CSV.  A missing file yields an empty table, not an exception."""
        import pandas as pd  # local import: pandas is heavy

        if path is None or not os.path.isfile(str(path)):
            return cls.empty(columns)
        usecols = ["exam_id", "patient_id"] + [c for c in columns]
        df = pd.read_csv(str(path), usecols=lambda c: c in set(usecols), low_memory=False)
        if "exam_id" not in df.columns:
            return cls.empty(columns)
        df["exam_id"] = df["exam_id"].astype(str)
        patient_of: Dict[str, str] = {}
        if "patient_id" in df.columns:
            patient_of = dict(zip(df["exam_id"], df["patient_id"].astype(str)))
        present = [c for c in columns if c in df.columns]
        flags: Dict[str, Dict[str, Optional[bool]]] = {}
        if present:
            for exam_id, row in zip(df["exam_id"], df[present].to_dict("records")):
                flags[exam_id] = {k: _to_bool(v) for k, v in row.items()}
        return cls(patient_of, flags, tuple(present))

    # -- accessors ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self._patient_of) or len(self._flags)

    def patient_id(self, exam_id: str) -> Optional[str]:
        return self._patient_of.get(str(exam_id))

    def flag(self, exam_id: str, column: str) -> Optional[bool]:
        row = self._flags.get(str(exam_id))
        if row is None:
            return None
        return row.get(column)

    def has(self, exam_id: str) -> bool:
        return str(exam_id) in self._flags or str(exam_id) in self._patient_of


class MultiReportStats:
    """Per-patient positive counts, for Hong et al. 2025's multi-report soft label.

    Built from the *training* records only (their labels are ours to read), keyed by
    exam id.  A patient with a single exam is not affected.
    """

    def __init__(self, counts: Optional[Mapping[str, Tuple[int, int]]] = None) -> None:
        self._by_exam: Dict[str, Tuple[int, int]] = dict(counts or {})

    @classmethod
    def build(
        cls,
        records: Iterable[Tuple[str, int, str]],
        table: Code15Table,
    ) -> "MultiReportStats":
        """``records`` is an iterable of ``(exam_id, hard_label, normalised_source)``."""
        per_patient: Dict[str, List[int]] = {}
        exam_patient: Dict[str, str] = {}
        for exam_id, label, source in records:
            if source != "code15":
                continue
            pid = table.patient_id(exam_id)
            if pid is None:
                continue
            exam_patient[str(exam_id)] = pid
            per_patient.setdefault(pid, []).append(int(label))
        counts: Dict[str, Tuple[int, int]] = {}
        for exam_id, pid in exam_patient.items():
            labels = per_patient[pid]
            counts[exam_id] = (int(sum(labels)), int(len(labels)))
        return cls(counts)

    def __len__(self) -> int:
        return len(self._by_exam)

    def get(self, exam_id: str) -> Optional[Tuple[int, int]]:
        return self._by_exam.get(str(exam_id))


# ---------------------------------------------------------------------------
# soft labels
# ---------------------------------------------------------------------------
def _team_rbbb_soft_label(
    label: int, rbbb: bool, avb1: bool, af: bool, sb: bool
) -> float:
    """The team's own ``dataloader.ECGDataset._compute_soft_label``, verbatim in spirit.

    Kept as ``soft_label_scheme="team_rbbb"`` because it is a stronger version of Hong
    et al. 2025's RBBB bump and it is the team's idea, not a paper's.
    """
    if label == 1:
        if rbbb and avb1:
            base = 0.95
        elif rbbb or avb1:
            base = 0.85
        else:
            base = 0.60
        return float(min(base + 0.05 * (int(af) + int(sb)), 0.98))
    if not rbbb and not avb1:
        base = 0.05
    elif rbbb and avb1:
        base = 0.40
    else:
        base = 0.15
    return float(base + 0.05 * (int(af) + int(sb)))


def _hong_rbbb_bump(value: float, rbbb: bool, avb1: bool, af: bool) -> float:
    """Hong et al. 2025: a CODE-15 *negative* showing RBBB is more likely a false negative.

    +0.15 for RBBB alone, +0.25 for RBBB together with first-degree AV block or atrial
    fibrillation, capped at 0.49 so the record can never outrank a real positive.
    """
    if not rbbb:
        return value
    bump = 0.25 if (avb1 or af) else 0.15
    return float(min(value + bump, 0.49))


def soft_label(
    label: Optional[int],
    source: str,
    exam_id: str,
    cfg: LabelCfg,
    table: Optional[Code15Table] = None,
    stats: Optional[MultiReportStats] = None,
) -> Optional[float]:
    """Target the loss sees.  ``None`` in, ``None`` out.

    * SaMi-Trop -> ``samitrop_value`` (1.0): serologically confirmed.
    * PTB-XL -> ``ptbxl_value`` (0.0): European cohort, assumed negative.
    * CODE-15 (and any unknown source) -> 0.8 / 0.2, self-reported and therefore noisy
      (Van Santvliet et al. 2025).
    * With ``multi_report``, a CODE-15 patient with several exams gets
      ``alpha*y + (1-alpha)*n_pos/n_total`` instead (Hong et al. 2025 eq. 1): a patient
      with 3 of 5 exams positive gets 0.804 for their positive exams and 0.294 for their
      negative ones.
    * ``rbbb_scheme`` optionally folds conduction-disease flags in -- see
      :func:`_team_rbbb_soft_label` and :func:`_hong_rbbb_bump`.

    Metrics never use this value; they use the hard label (audit defect D10).
    """
    if label is None:
        return None
    y = int(label)
    if not cfg.soft_labels:
        return float(y)
    if source == "samitrop":
        return float(cfg.samitrop_value)
    if source == "ptbxl":
        return float(cfg.ptbxl_value)

    value = float(cfg.code15_pos_value if y == 1 else cfg.code15_neg_value)

    if cfg.multi_report and stats is not None:
        counts = stats.get(exam_id)
        if counts is not None and counts[1] > 1:
            n_pos, n_total = counts
            a = float(cfg.multi_report_alpha)
            value = a * y + (1.0 - a) * (float(n_pos) / float(n_total))

    if cfg.rbbb_scheme != "none" and table is not None and table.has(exam_id):
        rbbb = bool(table.flag(exam_id, "RBBB"))
        avb1 = bool(table.flag(exam_id, "1dAVb"))
        af = bool(table.flag(exam_id, "AF"))
        sb = bool(table.flag(exam_id, "SB"))
        if cfg.rbbb_scheme == "team_rbbb":
            value = _team_rbbb_soft_label(y, rbbb, avb1, af, sb)
        elif cfg.rbbb_scheme == "hong_rbbb" and y == 0:
            value = _hong_rbbb_bump(value, rbbb, avb1, af)
    return float(value)


def aux_targets(
    exam_id: str, table: Optional[Code15Table], columns: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Auxiliary multitask targets and their availability mask.

    Returns ``(values, mask)``, both ``(len(columns),)`` float32.  ``mask[i] == 0``
    means "this label does not exist for this record" and the auxiliary loss ignores it
    -- which is the case for every SaMi-Trop, PTB-XL and hidden-cohort record.
    """
    values = np.zeros(len(columns), dtype=np.float32)
    mask = np.zeros(len(columns), dtype=np.float32)
    if table is None:
        return values, mask
    for i, column in enumerate(columns):
        flag = table.flag(exam_id, column)
        if flag is None:
            continue
        values[i] = 1.0 if flag else 0.0
        mask[i] = 1.0
    return values, mask


def patient_group(record_id: str, source: str, table: Optional[Code15Table]) -> str:
    """Grouping key for the patient-grouped split (PLAN.md section 3.7).

    CODE-15 has ~1.48 exams per patient, so an exam-level split leaks the same patient
    into both halves; the repo's own ``train_val_test_sets.csv`` leaks 45.5% of test
    exams' patients into train.  Records whose patient is unknown are their own group,
    which is conservative (it can only reduce leakage, never create it).
    """
    if source == "code15" and table is not None:
        pid = table.patient_id(record_id)
        if pid is not None:
            return f"c15_{pid}"
    return f"rec_{source}_{record_id}"
