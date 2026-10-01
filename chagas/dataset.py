"""Dataset, collate and DataModule over :mod:`chagas.preprocessing` + :mod:`chagas.augment`.

The dataset is a thin wrapper: it does no preprocessing of its own, it calls
:func:`chagas.preprocessing.preprocess_record`.  That is the fix for audit defect D11 --
the legacy tree reimplemented the pipeline inside ``ECGDataset.__getitem__`` and the two
copies had already diverged once (training z-scored with global per-lead statistics
while inference min-maxed per record, and nothing crashed to say so).
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import labels as L
from . import preprocessing as P
from .lengths import LENGTH_GROUP_IDS, header_duration, length_group
from .config import ExperimentCfg

__all__ = [
    "RecordEntry",
    "find_records",
    "scan_records",
    "ChagasDataset",
    "collate",
    "ChagasDataModule",
]


@dataclass
class RecordEntry:
    """One training record: where it is, what it is, and what it is grouped with."""

    path: str
    record_id: str
    source: str = "unknown"
    label_hard: int = 0
    label_soft: float = 0.0
    group: str = ""
    source_id: int = 0
    aux: Optional[np.ndarray] = None
    aux_mask: Optional[np.ndarray] = None
    #: Recording length from the record's own header (``n_samples / fs``); NaN when
    #: unknown.  Drives length-matched negative sampling and the within-length
    #: validation metrics (PLAN_R2.md sections 3.1 and 6).
    duration_s: float = float("nan")

    @property
    def length_group(self) -> str:
        """``short`` / ``mid`` / ``long`` / ``unknown`` (:func:`chagas.lengths.length_group`)."""
        return length_group(self.duration_s)


def find_records(folder: str, file_extension: str = ".hea") -> List[str]:
    """Absolute, extension-stripped record paths under ``folder``, sorted.

    Mirrors ``helper_code.find_records`` but returns absolute paths, so nothing depends
    on the process CWD.
    """
    out = set()
    for root, _dirs, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1] == file_extension:
                out.add(os.path.abspath(os.path.join(root, name))[: -len(file_extension)])
    return sorted(out)


def _read_header_meta(path: str) -> Optional[Tuple[str, str, int, str, float]]:
    """``(path, record_id, hard_label, source, duration_s)`` for one record, or ``None``.

    The duration is read from the same header (record line), so no extra I/O.
    """
    try:
        with open(path + ".hea", "r") as fh:
            header = fh.read()
        info = L.parse_patient_info(header, want_label=True)
    except NotImplementedError:
        raise
    except Exception:
        return None
    if info.label is None:
        return None
    return path, os.path.basename(path), int(info.label), info.source, header_duration(header)


def scan_records(
    paths: Sequence[str],
    cfg: ExperimentCfg,
    *,
    table: Optional[L.Code15Table] = None,
    workers: int = 8,
) -> List[RecordEntry]:
    """Read every header once and build the label/group metadata for the whole split.

    Done up front rather than per ``__getitem__`` because the soft label of a CODE-15
    record depends on its patient's *other* exams (Hong et al. 2025 eq. 1), which is not
    knowable from one record in isolation.
    """
    if table is None:
        table = L.Code15Table.from_csv(cfg.code15_exams_path, cfg.labels.aux_columns)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        raw = [r for r in pool.map(_read_header_meta, paths) if r is not None]

    stats = None
    if cfg.labels.soft_labels and cfg.labels.multi_report:
        stats = L.MultiReportStats.build(
            ((rid, lab, src) for _p, rid, lab, src, _d in raw), table
        )

    entries: List[RecordEntry] = []
    want_aux = cfg.model.aux_heads
    for path, record_id, label, source, duration in raw:
        soft = L.soft_label(label, source, record_id, cfg.labels, table, stats)
        aux, aux_mask = (
            L.aux_targets(record_id, table, cfg.labels.aux_columns)
            if want_aux
            else (None, None)
        )
        entries.append(
            RecordEntry(
                path=path,
                record_id=record_id,
                source=source,
                label_hard=int(label),
                label_soft=float(label if soft is None else soft),
                group=L.patient_group(record_id, source, table),
                source_id=L.SOURCE_IDS.get(source, 0),
                aux=aux,
                aux_mask=aux_mask,
                duration_s=float(duration),
            )
        )
    return entries


class ChagasDataset(Dataset):
    """Records -> model-ready tensors.  Returns ``None`` for records that cannot be used.

    Augmentation is applied only when ``train=True`` and ``cfg.aug.enabled``; the RNG is
    seeded from ``torch.initial_seed()`` (which PyTorch advances per epoch and per
    worker) combined with the record index, so a run is reproducible from the global
    seed while a record still sees a different augmentation each epoch.
    """

    def __init__(
        self,
        entries: Sequence[RecordEntry],
        cfg: ExperimentCfg,
        *,
        train: bool = False,
        want_label: bool = True,
    ) -> None:
        self.entries = list(entries)
        self.cfg = cfg
        self.train = bool(train)
        self.want_label = bool(want_label)

    def __len__(self) -> int:
        return len(self.entries)

    def _rng(self, idx: int) -> np.random.Generator:
        seed = (int(torch.initial_seed()) + 1_000_003 * int(idx)) % (2 ** 63 - 1)
        return np.random.default_rng(seed)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        entry = self.entries[idx]
        aug = self.cfg.aug if self.train else None
        try:
            sample = P.preprocess_record(
                entry.path,
                self.cfg.preprocess,
                train=self.train,
                aug=aug,
                rng=self._rng(idx) if self.train else None,
                want_label=False,
            )
        except NotImplementedError:
            raise
        except Exception:
            return None
        if sample is None:
            return None

        item: Dict[str, Any] = {}
        if self.cfg.preprocess.window_mode == "multi":
            # (K, 12, W) windows; ``collate`` concatenates them across the batch and the
            # model averages the K logits back into one per record (PLAN_R2.md 3.2).
            windows, lengths = sample.windows()
            item["signal"] = torch.from_numpy(np.ascontiguousarray(windows))
            item["valid_len"] = torch.from_numpy(lengths)
            item["n_windows"] = torch.tensor(int(lengths.shape[0]), dtype=torch.long)
        else:
            item["signal"] = torch.from_numpy(np.ascontiguousarray(sample.signal))
            item["valid_len"] = torch.tensor(int(sample.valid_len), dtype=torch.long)
        item.update({
            "demographics": torch.from_numpy(sample.demographics),
            "label_hard": torch.tensor(float(entry.label_hard), dtype=torch.float32),
            "label_soft": torch.tensor(float(entry.label_soft), dtype=torch.float32),
            "source_id": torch.tensor(int(entry.source_id), dtype=torch.long),
            "length_group": torch.tensor(
                LENGTH_GROUP_IDS[entry.length_group], dtype=torch.long
            ),
            "record_id": entry.record_id,
        })
        if entry.aux is not None and entry.aux_mask is not None:
            item["aux"] = torch.from_numpy(entry.aux)
            item["aux_mask"] = torch.from_numpy(entry.aux_mask)
        return item


#: Per-window keys: concatenated (not stacked) when items carry ``n_windows``.
WINDOW_KEYS = ("signal", "valid_len")


def collate(batch: Sequence[Optional[Dict[str, Any]]]) -> Optional[Dict[str, Any]]:
    """Stack a batch, dropping unusable records; ``None`` if every record failed.

    Every consumer must check for ``None`` before unpacking -- that is what
    :meth:`chagas.model.ChagasModule._unpack` does.

    Multi-window items (``n_windows`` present) have ``signal`` ``(K_i, 12, W)`` and
    ``valid_len`` ``(K_i,)``; those two are concatenated into ``(sum K_i, 12, W)`` and
    ``n_windows`` says which windows belong to which record.  Everything else is
    per-record and stacked as before.
    """
    items = [b for b in batch if b is not None]
    if not items:
        return None
    windowed = "n_windows" in items[0]
    out: Dict[str, Any] = {}
    for key in items[0]:
        if key == "record_id":
            out[key] = [item[key] for item in items]
        elif windowed and key in WINDOW_KEYS:
            out[key] = torch.cat([item[key] for item in items], dim=0)
        else:
            out[key] = torch.stack([item[key] for item in items])
    return out


class ChagasDataModule(pl.LightningDataModule):
    """Train/val dataloaders with the positive-oversampling sampler.

    ``WeightedRandomSampler`` with positives weighted ``oversample`` (default 5x, Van
    Santvliet et al. 2025) rather than inverse-frequency.  Some sampler is not optional:
    records arrive grouped by source, so without one the model sees a long run of
    all-negative CODE-15 records and then a run of SaMi-Trop positives, in the same order
    every epoch.  But inverse-frequency (~50/50, the legacy behaviour) is far more
    aggressive than any top team and is harmful under label noise (Physiol. Meas. 2026);
    ``oversample: 1.0`` is the ablation.
    """

    def __init__(
        self,
        cfg: ExperimentCfg,
        train_entries: Sequence[RecordEntry],
        val_entries: Sequence[RecordEntry] = (),
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_entries = list(train_entries)
        self.val_entries = list(val_entries)
        self.train_dataset = ChagasDataset(self.train_entries, cfg, train=True)
        self.val_dataset = (
            ChagasDataset(self.val_entries, cfg, train=False) if self.val_entries else None
        )

    def _loader_kwargs(self) -> Dict[str, Any]:
        workers = max(0, int(self.cfg.train.num_workers))
        kwargs: Dict[str, Any] = {
            "batch_size": self.cfg.train.batch_size,
            "num_workers": workers,
            "collate_fn": collate,
            "pin_memory": torch.cuda.is_available(),
        }
        if workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self) -> DataLoader:
        weights = None
        if self.cfg.train.oversample != 1.0:
            labels = np.array([e.label_hard for e in self.train_entries], dtype=np.float64)
            weights = np.where(labels > 0.5, float(self.cfg.train.oversample), 1.0)
        if weights is None:
            return DataLoader(self.train_dataset, shuffle=True, **self._loader_kwargs())
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(self.train_dataset),
            replacement=True,
        )
        return DataLoader(self.train_dataset, sampler=sampler, **self._loader_kwargs())

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val_dataset is None:
            return None
        return DataLoader(self.val_dataset, shuffle=False, **self._loader_kwargs())
