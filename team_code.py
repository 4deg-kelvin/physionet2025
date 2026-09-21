#!/usr/bin/env python
"""Team entry points for the PhysioNet/CinC Challenge 2025 (Chagas detection).

The organisers replace and run ``train_model.py``, ``run_model.py`` and
``evaluate_model.py``; those three wrappers reach the team's code through exactly the
three functions below.  Everything else lives in the :mod:`chagas` package.

This file is deliberately thin.  The previous version imported ``transformer``,
``pretrain_mae_vit_ecg``, ``finetuning_mae_vit_ecg`` and ``fm`` (and, through it,
``wandb``, ``matplotlib`` and ``neurokit2``) without using most of them -- every one of
those is an import-time failure surface inside the judged container (audit defect D12).

Note that the organisers call all three functions positionally, and that
``run_model``'s argument order is **record first, model second**.
"""

from __future__ import annotations

from typing import Optional, Tuple

from chagas.inference import load_model as _load_model
from chagas.inference import run_model as _run_model
from chagas.training import train_model as _train_model


def train_model(data_folder: str, model_folder: str, verbose: bool = False) -> None:
    """Train the Challenge model and write it into ``model_folder``.

    Configuration comes from ``$CHAGAS_CONFIG`` when that is set (the ablation ladder
    uses it) and from :func:`chagas.config.default_config` otherwise.
    """
    _train_model(data_folder, model_folder, verbose)


def load_model(model_folder: str, verbose: bool = False):
    """Load the trained model once, before the per-record loop in ``run_model.py``."""
    return _load_model(model_folder, verbose)


def run_model(
    record: str, model, verbose: bool = False
) -> Tuple[Optional[int], Optional[float]]:
    """Return ``(binary_label, probability)`` for one record.

    Per-record failures degrade to ``(None, None)``; a :class:`NotImplementedError` is
    re-raised, because it means preprocessing is wrong and no prediction should be
    emitted at all.
    """
    return _run_model(record, model, verbose)
