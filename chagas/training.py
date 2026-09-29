"""Training: patient-grouped split, two-stage schedule, budget, threshold, ensemble.

PLAN.md sections 3.1 and 3.7.  This module owns everything the judged
``train_model.py`` reaches.

What changed relative to commit a72c51c (audit defect D1): the submission path had
**no validation split, no model selection, no early stopping, no seed and no gradient
clipping** -- it trained on every record and shipped whatever epoch 16 produced.  Here
there is a patient-grouped 90/10 internal holdout, the checkpoint is selected on the
validation *challenge score* (the competition metric, not the loss), training stops
early on patience 3 (Physiol. Meas. 2026), and the decision threshold is calibrated on
that same holdout and stored in the checkpoint.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from . import labels as L
from .config import ExperimentCfg, load_config
from .dataset import ChagasDataModule, RecordEntry, find_records, scan_records
from .model import ChagasModule

__all__ = [
    "TimeLimitCallback",
    "split_entries",
    "train_model",
    "MODEL_FILENAME",
    "member_filename",
]

#: Name of the (single-member) checkpoint written into ``model_folder``.
MODEL_FILENAME = "chagas_model.ckpt"


def member_filename(index: int) -> str:
    return MODEL_FILENAME if index == 0 else f"chagas_model_{index}.ckpt"


# ---------------------------------------------------------------------------
# budget
# ---------------------------------------------------------------------------
class TimeLimitCallback(pl.Callback):
    """Stop training when wall-clock time approaches a limit.

    (team; kept from ``fm.TimeLimitCallback`` and made budget-aware.)  The grader gives
    72 h on a T4, which is roughly 2x slower than the TitanRTX we measure on, so the
    stage budgets are fractions of ``max_hours`` rather than epoch counts.  Setting
    ``trainer.should_stop`` rather than raising lets Lightning finish the current epoch,
    run validation and let ``ModelCheckpoint`` save -- so a budget-limited run still
    ships its best weights.
    """

    def __init__(self, max_hours: float, deadline: Optional[float] = None) -> None:
        super().__init__()
        self.max_seconds = float(max_hours) * 3600.0
        self.deadline = deadline
        self.start_time: Optional[float] = None
        self.stopped = False

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.start_time = time.time()

    def _expired(self) -> bool:
        now = time.time()
        if self.start_time is not None and (now - self.start_time) >= self.max_seconds:
            return True
        return self.deadline is not None and now >= self.deadline

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if not self.stopped and self._expired():
            self.stopped = True
            trainer.should_stop = True
            print(
                f"[TimeLimitCallback] budget of {self.max_seconds / 3600:.3f} h reached; "
                f"stopping after step {trainer.global_step}.",
                flush=True,
            )

    def on_validation_end(self, trainer, pl_module) -> None:
        if not self.stopped and self._expired():
            self.stopped = True
            trainer.should_stop = True


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------
def split_entries(
    entries: Sequence[RecordEntry], val_fraction: float, seed: int
) -> Tuple[List[RecordEntry], List[RecordEntry]]:
    """Patient-grouped, source x label stratified 90/10 split (PLAN.md section 3.7).

    Grouping is by :func:`chagas.labels.patient_group`, i.e. CODE-15 ``patient_id`` from
    the shipped ``code15_exams.csv`` and the record itself for every other source.  A
    patient is never split: CODE-15 has ~1.48 exams per patient, and the repo's own
    ``train_val_test_sets.csv`` leaks 45.5% of its test exams' patients into train, which
    inflates every number computed on it.

    Stratification is on ``(source, whether the group contains a positive)`` so the small
    SaMi-Trop cohort and the CODE-15 positives are both represented in the holdout.
    """
    by_group: Dict[str, List[RecordEntry]] = defaultdict(list)
    for e in entries:
        by_group[e.group].append(e)

    strata: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    for group, members in by_group.items():
        source = members[0].source
        positive = int(any(m.label_hard == 1 for m in members))
        strata[(source, positive)].append(group)

    rng = np.random.default_rng(seed)
    train: List[RecordEntry] = []
    val: List[RecordEntry] = []
    for key in sorted(strata):
        groups = sorted(strata[key])
        rng.shuffle(groups)
        n_val = int(round(val_fraction * len(groups)))
        # Never take the whole stratum, and take at least one group when the stratum is
        # big enough to spare it -- a holdout with no SaMi-Trop positives cannot select
        # a model on the challenge score.
        n_val = min(max(n_val, 1 if len(groups) > 1 else 0), max(len(groups) - 1, 0))
        for group in groups[:n_val]:
            val.extend(by_group[group])
        for group in groups[n_val:]:
            train.extend(by_group[group])
    return train, val


def _subsample(
    entries: Sequence[RecordEntry], max_records: int, seed: int
) -> List[RecordEntry]:
    """Thin the negatives to ``max_records``, keeping every positive.

    The positive count is what binds learning at 2% prevalence, so budgeted runs drop
    negatives rather than sampling uniformly.
    """
    entries = list(entries)
    if max_records <= 0 or len(entries) <= max_records:
        return entries
    pos = [e for e in entries if e.label_hard == 1]
    neg = [e for e in entries if e.label_hard != 1]
    budget = max(0, max_records - len(pos))
    if budget < len(neg):
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(len(neg), size=budget, replace=False))
        neg = [neg[i] for i in keep]
    return pos + neg


# ---------------------------------------------------------------------------
# one member
# ---------------------------------------------------------------------------
def _assert_no_bf16(precision: str) -> None:
    """bf16 has no native support on SM 7.5 (T4 / TitanRTX); fp16 tensor cores do."""
    if "bf16" in str(precision):
        raise ValueError(f"precision {precision!r} uses bf16, unsupported on SM 7.5")
    if torch.cuda.is_available():
        major = torch.cuda.get_device_capability(0)[0]
        if major < 8 and "bf16" in str(precision):
            raise ValueError("bf16 requested on a pre-Ampere GPU")


def _make_trainer(
    cfg: ExperimentCfg,
    work_dir: Path,
    *,
    stage: int,
    max_hours: float,
    deadline: Optional[float],
    has_val: bool,
    log_dir: Optional[Path] = None,
) -> Tuple[pl.Trainer, ModelCheckpoint, TimeLimitCallback]:
    ckpt_dir = work_dir / f"stage{stage}"
    checkpoint = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=f"best-stage{stage}",
        monitor=cfg.train.monitor if has_val else None,
        mode="max",
        save_top_k=1,
        # A budget-limited or step-limited stage may stop before any validation has run,
        # leaving `best_model_path` empty; `last.ckpt` is the fallback so a run always
        # has weights to carry into the next stage.  It lives in model_folder/_work,
        # which is deleted once the final checkpoint is written.
        save_last=True,
        save_weights_only=False,
    )
    time_limit = TimeLimitCallback(max_hours=max_hours, deadline=deadline)
    callbacks: List[pl.Callback] = [checkpoint, time_limit]
    # Persist loss / LR / validation metrics per stage.  Outside _work on purpose: _work is
    # deleted once the final checkpoint is written, the metrics are meant to survive.
    logger: Any = False
    if cfg.train.csv_log and log_dir is not None:
        logger = CSVLogger(save_dir=str(log_dir), name="", version=f"stage{stage}")
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    if has_val and cfg.train.early_stop_patience > 0:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.train.monitor,
                mode="max",
                patience=cfg.train.early_stop_patience,
                check_on_train_epoch_end=False,
            )
        )
    max_epochs = cfg.train.max_epochs if stage == 1 else cfg.train.stage2_max_epochs
    if stage == 1:
        max_steps = cfg.train.stage1_max_steps if cfg.train.stage1_max_steps > 0 else -1
    else:
        max_steps = cfg.train.stage2_max_steps if cfg.train.stage2_max_steps > 0 else -1
    # Lightning counts val_check_interval in TRAINING BATCHES, so convert optimiser steps.
    step_val = stage == 2 and cfg.train.stage2_val_every_steps > 0
    val_every = (
        cfg.train.stage2_val_every_steps * cfg.train.accumulate_grad_batches if step_val else 1.0
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_steps=max_steps,
        accelerator="auto",
        devices=1,
        precision=cfg.train.precision if torch.cuda.is_available() else 32,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        gradient_clip_val=cfg.train.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=cfg.train.log_every_n_steps,
        # Everything Lightning writes goes inside model_folder; without this an implicit
        # ModelCheckpoint lands in the CWD (i.e. /challenge in the judged container).
        default_root_dir=str(work_dir),
        enable_checkpointing=True,
        enable_progress_bar=False,
        num_sanity_val_steps=0,
        limit_val_batches=1.0 if has_val else 0,
        limit_train_batches=cfg.train.limit_train_batches,
        val_check_interval=val_every,
        # An integer interval may span epochs; Lightning requires this to be None then.
        check_val_every_n_epoch=None if step_val else 1,
    )
    return trainer, checkpoint, time_limit


def _load_best(module: ChagasModule, checkpoint: ModelCheckpoint) -> ChagasModule:
    """Restore the best weights in place, if the callback managed to save any."""
    for path in (checkpoint.best_model_path, checkpoint.last_model_path):
        if path and os.path.isfile(path):
            state = torch.load(path, map_location="cpu", weights_only=False)
            module.load_state_dict(state["state_dict"], strict=True)
            return module
    return module


def _train_member(
    cfg: ExperimentCfg,
    datamodule: ChagasDataModule,
    work_dir: Path,
    *,
    seed: int,
    max_hours: float,
    deadline: Optional[float],
    verbose: bool,
    log_dir: Optional[Path] = None,
) -> ChagasModule:
    pl.seed_everything(seed, workers=True)
    has_val = datamodule.val_dataset is not None and len(datamodule.val_dataset) > 0

    module = ChagasModule(cfg=cfg.to_dict(), stage=1)

    stage1_hours = max_hours * cfg.train.stage1_budget_frac if cfg.train.two_stage else max_hours
    trainer, checkpoint, _ = _make_trainer(
        cfg, work_dir, stage=1, max_hours=stage1_hours, deadline=deadline, has_val=has_val, log_dir=log_dir
    )
    if verbose:
        print(f"[chagas] stage 1 (frozen encoder), budget {stage1_hours:.3f} h", flush=True)
    trainer.fit(module, datamodule=datamodule)
    module = _load_best(module, checkpoint)

    if cfg.train.two_stage:
        module.enter_stage_two()
        stage2_hours = max_hours * cfg.train.stage2_budget_frac
        trainer2, checkpoint2, _ = _make_trainer(
            cfg, work_dir, stage=2, max_hours=stage2_hours, deadline=deadline, has_val=has_val, log_dir=log_dir
        )
        if verbose:
            print(
                f"[chagas] stage 2 (12 transformer layers unfrozen, LLRD "
                f"{cfg.train.llrd}), budget {stage2_hours:.3f} h",
                flush=True,
            )
        trainer2.fit(module, datamodule=datamodule)
        module = _load_best(module, checkpoint2)
        trainer = trainer2

    # Calibrate the decision threshold on the validation split, with the *selected*
    # weights (audit defect D9: a hard 0.5 on a 50%-prior-calibrated model flags a wild
    # over-share of a 2%-prevalence cohort).
    if has_val:
        trainer.validate(module, datamodule=datamodule, verbose=False)
        if module.last_val_probs is not None and module.last_val_probs.size:
            value = module.update_threshold(module.last_val_probs)
            if verbose:
                print(f"[chagas] threshold = q{cfg.train.threshold_quantile:.2f} = {value:.6f}", flush=True)
    if verbose:
        print(f"[chagas] layer weights = {np.round(module.feature_extractor.layer_weight_values(), 4)}", flush=True)
    return module


def _summarise_val_history(member: int, module: ChagasModule) -> dict:
    """Best validation score and epoch per stage, plus the full per-epoch history.

    Written into chagas_manifest.json so a finished run records its own model
    selection; previously the only trace was the checkpoint filename
    (DECISIONS.md, 2026-09-22).
    """
    history = list(getattr(module, "val_history", []) or [])
    best_by_stage: dict = {}
    for rec in history:
        key = f"stage{rec['stage']}"
        if key not in best_by_stage or rec["val_challenge"] > best_by_stage[key]["val_challenge"]:
            best_by_stage[key] = {k: rec[k] for k in ("epoch", "val_challenge", "val_auroc", "val_auprc")}
    return {"member": member, "best_by_stage": best_by_stage, "history": history}


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------
def train_model(
    data_folder: str,
    model_folder: str,
    verbose: bool = False,
    cfg: Optional[ExperimentCfg] = None,
) -> None:
    """Train and save the Challenge model.  Called by ``team_code.train_model``.

    Nothing is written outside ``model_folder``; the per-stage Lightning checkpoints go
    into ``model_folder/_work`` and that directory is removed once the final
    checkpoint(s) are saved.
    """
    cfg = cfg or load_config()
    _assert_no_bf16(cfg.train.precision)
    os.makedirs(model_folder, exist_ok=True)
    work_dir = Path(model_folder) / "_work"

    if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8:
        torch.set_float32_matmul_precision("high")

    paths = find_records(data_folder)
    if not paths:
        raise FileNotFoundError(f"No WFDB records found under {data_folder!r}")
    if verbose:
        print(f"[chagas] {len(paths)} records under {data_folder}", flush=True)

    table = L.Code15Table.from_csv(cfg.code15_exams_path, cfg.labels.aux_columns)
    if verbose:
        print(f"[chagas] code15_exams.csv: {len(table)} exams", flush=True)
    entries = scan_records(paths, cfg, table=table, workers=cfg.data.scan_workers)
    if not entries:
        raise RuntimeError("no records carried a readable Chagas label")
    entries = _subsample(entries, cfg.data.max_records, cfg.train.seed)

    train_entries, val_entries = split_entries(
        entries, cfg.train.val_fraction, cfg.train.seed
    )
    if verbose:
        n_pos = sum(e.label_hard for e in train_entries)
        n_vpos = sum(e.label_hard for e in val_entries)
        print(
            f"[chagas] train {len(train_entries)} ({n_pos} pos) / "
            f"val {len(val_entries)} ({n_vpos} pos); "
            f"groups disjoint: {not (set(e.group for e in train_entries) & set(e.group for e in val_entries))}",
            flush=True,
        )

    datamodule = ChagasDataModule(cfg, train_entries, val_entries)

    n_members = max(1, int(cfg.train.ensemble_members))
    deadline = time.time() + cfg.train.max_hours * 3600.0
    per_member_hours = cfg.train.max_hours / n_members

    saved: List[str] = []
    summaries: List[dict] = []
    for member in range(n_members):
        module = _train_member(
            cfg,
            datamodule,
            work_dir / f"member{member}",
            seed=cfg.train.seed + member,
            max_hours=per_member_hours,
            deadline=deadline,
            verbose=verbose,
            log_dir=Path(model_folder) / "metrics" / f"member{member}",
        )
        out_path = os.path.join(model_folder, member_filename(member))
        _save_checkpoint(module, out_path, cfg)
        saved.append(out_path)
        summaries.append(_summarise_val_history(member, module))
        if verbose:
            print(f"[chagas] saved {out_path}", flush=True)

    shutil.rmtree(work_dir, ignore_errors=True)
    with open(os.path.join(model_folder, "chagas_manifest.json"), "w") as fh:
        json.dump(
            {
                "members": [os.path.basename(p) for p in saved],
                "training": summaries,
                "config": cfg.to_dict(),
            },
            fh,
            indent=2,
        )


def _save_checkpoint(module: ChagasModule, path: str, cfg: ExperimentCfg) -> None:
    """Write a self-contained checkpoint: weights, config, threshold, layer weights."""
    state = {
        "state_dict": module.state_dict(),
        "hyper_parameters": dict(module.hparams),
        "chagas": {
            "version": 2,
            "config": cfg.to_dict(),
            "threshold": float(module.threshold),
            "layer_weights": module.feature_extractor.layer_weight_values().tolist(),
            "stage": int(module.stage),
        },
    }
    torch.save(state, path)
