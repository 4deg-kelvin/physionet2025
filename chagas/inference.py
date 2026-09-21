"""Inference (PLAN.md section 3.8).

``team_code.load_model`` / ``team_code.run_model`` are thin adapters over
:func:`load_model` and :func:`run_model` here.  The same
:func:`chagas.preprocessing.preprocess_record` runs as in training, with augmentation
off -- there is exactly one pipeline, so train/inference skew cannot appear.

``run_model`` returns ``(None, None)`` on a per-record failure but deliberately
**re-raises** :class:`NotImplementedError`, which is this codebase's signal for
"preprocessing is wrong, do not emit predictions": ``run_model.py`` without
``--allow_failures`` then aborts the pass instead of silently scoring every record 0.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import torch

from .config import ExperimentCfg
from .model import ChagasModule
from .preprocessing import Sample, preprocess_record
from .training import MODEL_FILENAME

__all__ = ["ChagasPredictor", "load_model", "run_model", "predict"]


@dataclass
class _Member:
    module: ChagasModule
    threshold: float


class ChagasPredictor:
    """One or more trained members, averaged on the probability scale.

    Averaging sigmoid outputs (rather than logits) is what the ensemble papers do
    (Jidling et al., PLoS NTD 2023 averaged 15 seeds; Ahus AIM used 5 folds).  The
    decision threshold is the mean of the members' calibrated thresholds.
    """

    def __init__(self, members: Sequence[_Member], cfg: ExperimentCfg, device: torch.device):
        if not members:
            raise ValueError("no members to predict with")
        self.members = list(members)
        self.cfg = cfg
        self.device = device
        self.threshold = float(np.mean([m.threshold for m in self.members]))

    # -- torch plumbing ----------------------------------------------------
    def parameters(self):  # pragma: no cover - convenience for `next(model.parameters())`
        return self.members[0].module.parameters()

    def to(self, device: Any) -> "ChagasPredictor":
        self.device = torch.device(device)
        for m in self.members:
            m.module.to(self.device)
        return self

    def eval(self) -> "ChagasPredictor":
        for m in self.members:
            m.module.eval()
        return self

    # -- prediction --------------------------------------------------------
    @torch.no_grad()
    def predict_samples(self, samples: Sequence[Sample]) -> np.ndarray:
        """Mean probability per sample, shape ``(n,)``."""
        if not samples:
            return np.zeros(0, dtype=np.float64)
        signal = torch.from_numpy(
            np.stack([s.signal for s in samples]).astype(np.float32)
        ).to(self.device)
        valid_len = torch.tensor(
            [s.valid_len for s in samples], dtype=torch.long, device=self.device
        )
        demographics = torch.from_numpy(
            np.stack([s.demographics for s in samples]).astype(np.float32)
        ).to(self.device)
        probs = []
        for m in self.members:
            out = m.module.predict_proba(signal, valid_len, demographics)
            probs.append(out.float().reshape(-1).cpu().numpy())
        return np.mean(np.stack(probs, axis=0), axis=0).astype(np.float64)

    def predict_record(self, record: str) -> Tuple[Optional[int], Optional[float]]:
        sample = preprocess_record(
            record, self.cfg.preprocess, train=False, aug=None, want_label=False
        )
        if sample is None:
            return None, None
        prob = float(self.predict_samples([sample])[0])
        return int(prob > self.threshold), prob

    def predict_records(self, records: Sequence[str], batch_size: int = 32):
        """Batched path for our own offline evaluation.

        The organisers' ``run_model.py`` is per-record and must keep working, so this is
        an addition, not a replacement.
        """
        results: List[Tuple[str, Optional[int], Optional[float]]] = []
        buffer: List[Tuple[str, Sample]] = []

        def flush() -> None:
            if not buffer:
                return
            probs = self.predict_samples([s for _r, s in buffer])
            for (rec, _s), p in zip(buffer, probs):
                results.append((rec, int(p > self.threshold), float(p)))
            buffer.clear()

        for record in records:
            sample = preprocess_record(
                record, self.cfg.preprocess, train=False, aug=None, want_label=False
            )
            if sample is None:
                results.append((record, None, None))
                continue
            buffer.append((record, sample))
            if len(buffer) >= batch_size:
                flush()
        flush()
        return results


def _member_paths(model_folder: str) -> List[str]:
    manifest = os.path.join(model_folder, "chagas_manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, "r") as fh:
            names = json.load(fh).get("members", [])
        paths = [os.path.join(model_folder, n) for n in names]
        paths = [p for p in paths if os.path.isfile(p)]
        if paths:
            return paths
    default = os.path.join(model_folder, MODEL_FILENAME)
    if os.path.isfile(default):
        extra = sorted(glob.glob(os.path.join(model_folder, "chagas_model_*.ckpt")))
        return [default] + extra
    found = sorted(glob.glob(os.path.join(model_folder, "*.ckpt")))
    if not found:
        raise FileNotFoundError(f"No Chagas checkpoint found in {model_folder!r}")
    return found


def load_model(
    model_folder: str,
    verbose: bool = False,
    device: Optional[str] = None,
    checkpoint_override: Optional[str] = None,
) -> ChagasPredictor:
    """Load every member in ``model_folder`` and return a :class:`ChagasPredictor`.

    State dicts load with ``strict=True``.  The legacy loader accepted ``strict=False``,
    which meant a checkpoint missing the trained head loaded "successfully" and the
    model emitted confident noise from a randomly initialised head; a scarce submission
    is too expensive for that failure mode.
    """
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    members: List[_Member] = []
    cfg: Optional[ExperimentCfg] = None
    for path in _member_paths(model_folder):
        state = torch.load(path, map_location="cpu", weights_only=False)
        meta = state.get("chagas", {})
        hparams = dict(state.get("hyper_parameters", {}))
        cfg_dict = meta.get("config") or hparams.get("cfg") or {}
        cfg = ExperimentCfg.from_dict(cfg_dict)
        backbone = checkpoint_override or str(cfg.checkpoint_path)
        module = ChagasModule(
            cfg=cfg.to_dict(),
            checkpoint_override=backbone,
            stage=int(meta.get("stage", hparams.get("stage", 1))),
        )
        module.load_state_dict(state["state_dict"], strict=True)
        module.eval().to(dev)
        threshold = float(meta.get("threshold", float(module.threshold)))
        members.append(_Member(module=module, threshold=threshold))
        if verbose:
            print(f"[chagas] loaded {os.path.basename(path)} (threshold {threshold:.6f})", flush=True)
    assert cfg is not None
    predictor = ChagasPredictor(members, cfg, dev)
    if verbose:
        print(
            f"[chagas] {len(members)} member(s) on {dev}; "
            f"decision threshold {predictor.threshold:.6f}",
            flush=True,
        )
    return predictor


def predict(record: str, model: ChagasPredictor) -> Tuple[Optional[int], Optional[float]]:
    """Single-record prediction without the error handling of :func:`run_model`."""
    return model.predict_record(record)


def run_model(
    record: str, model: Optional[ChagasPredictor], verbose: bool = False
) -> Tuple[Optional[int], Optional[float]]:
    """``(binary_label, probability)`` for one record.

    Note the organisers' argument order: **record first, model second**.
    """
    if model is None:
        raise ValueError("Model is None; cannot run inference.")
    try:
        pred, prob = model.predict_record(record)
        if prob is None:
            return None, None
        return pred, prob
    except NotImplementedError as exc:
        # The tripwire: something systematic is wrong (e.g. we are reading held-out
        # labels, or layer aggregation lost a layer).  Do not emit predictions.
        raise NotImplementedError(f"run_model error: {exc}") from exc
    except Exception as exc:  # pragma: no cover - per-record robustness
        if verbose:
            print(f"run_model error on {record}: {exc}", flush=True)
        return None, None
