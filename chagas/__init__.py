"""Chagas detection from the 12-lead ECG -- PhysioNet/CinC Challenge 2025.

This package is the whole submission path.  ``team_code.py`` is a thin adapter over
:func:`chagas.training.train_model`, :func:`chagas.inference.load_model` and
:func:`chagas.inference.run_model`; nothing on the judged path imports the legacy
``fm.py`` / ``dataloader.py`` / ``utils.py`` modules, which stay in the tree only for
history.

The core modelling idea is the team's and is preserved unchanged: the ECG-FM backbone
(McKeen et al., *ECG-FM: an open electrocardiogram foundation model*, JAMIA Open 2025,
arXiv:2408.05178) with a **learnable softmax-weighted aggregation over all 12
transformer layers**, captured through forward hooks, masked-mean-pooled to a 768-d
feature and classified by a small MLP with age/sex side information.

See ``PROVENANCE.md`` for the origin of every non-team idea in this package and
``METHODS.md`` for what changed relative to commit a72c51c.
"""

from __future__ import annotations

__all__ = [
    "config",
    "preprocessing",
    "augment",
    "labels",
    "dataset",
    "model",
    "losses",
    "training",
    "inference",
    "metrics",
]

__version__ = "2.0.0"
