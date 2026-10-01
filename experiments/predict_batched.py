#!/usr/bin/env python
"""Batched inference for experiments: the ``run_model.py`` result, minutes instead of hours.

PLAN_R2.md section 2.  The organisers' ``run_model.py`` loads one record at a time and
runs the model on it (batch 1, no workers), which is the judged contract and stays the
test for the final candidate.  For experiments this script produces the same
probabilities with a ``DataLoader`` (worker processes doing the preprocessing) and
batches of 64 records:

* the model is loaded by :func:`chagas.inference.load_model` (same members, same
  thresholds, same averaging as ``team_code.run_model``);
* each record goes through :func:`chagas.preprocessing.preprocess_record` with
  augmentation off -- the one pipeline training and ``run_model.py`` also use;
* batches are assembled by :func:`chagas.inference.samples_to_arrays` and scored by
  :meth:`chagas.inference.ChagasPredictor.predict_arrays`, the code ``run_model.py``
  reaches through ``predict_samples``.

Output: a CSV ``stem, record, binary, prob`` (``binary = prob > threshold``, as
``run_model.py`` writes it; NaN for a record that could not be preprocessed, which
``run_model.py`` writes as ``None``).  ``eval_report.py --preds-a`` reads it.

Equivalence with the ``run_model.py`` path (max |delta prob| < 1e-4 on the 400-record
smoke set) is asserted by ``tests/test_predict_batched.py`` and by
``experiments/batch_r2_smoke.sh`` (against real ``run_model.py`` output files).

    python experiments/predict_batched.py -m MODEL_FOLDER -d DATA_FOLDER -o preds.csv
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chagas.config import PreprocessCfg  # noqa: E402
from chagas.dataset import find_records  # noqa: E402
from chagas.inference import ChagasPredictor, load_model, samples_to_arrays  # noqa: E402
from chagas.preprocessing import Sample, preprocess_record  # noqa: E402

__all__ = ["RecordDataset", "predict_folder", "predict_records"]


class RecordDataset(Dataset):
    """Records -> :class:`chagas.preprocessing.Sample` (``None`` when unusable)."""

    def __init__(self, records: Sequence[str], cfg: PreprocessCfg) -> None:
        self.records = list(records)
        self.cfg = cfg

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        try:
            sample = preprocess_record(self.records[idx], self.cfg, train=False, aug=None, want_label=False)
        except NotImplementedError:
            raise
        except Exception:  # pragma: no cover - mirrors run_model's per-record robustness
            sample = None
        return {"index": idx, "sample": sample}


def _collate(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep the samples as objects; batching into arrays happens in the main process."""
    return list(batch)


def predict_records(
    predictor: ChagasPredictor,
    records: Sequence[str],
    *,
    batch_size: int = 64,
    num_workers: int = 10,
) -> pd.DataFrame:
    """Probabilities for ``records`` (absolute, extension-less paths), in input order."""
    dataset = RecordDataset(records, predictor.cfg.preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        collate_fn=_collate,
    )
    probs = np.full(len(records), np.nan, dtype=np.float64)
    for batch in loader:
        good = [(b["index"], b["sample"]) for b in batch if b["sample"] is not None]
        if not good:
            continue
        samples: List[Sample] = [s for _i, s in good]
        out = predictor.predict_arrays(*samples_to_arrays(samples, predictor.window_mode))
        probs[[i for i, _s in good]] = out
    binary = np.where(np.isfinite(probs), (probs > predictor.threshold).astype(float), np.nan)
    return pd.DataFrame({
        "stem": [os.path.basename(r) for r in records],
        "record": list(records),
        "binary": binary,
        "prob": probs,
    })


def predict_folder(
    model_folder: str,
    data_folder: str,
    *,
    batch_size: int = 64,
    num_workers: int = 10,
    device: Optional[str] = None,
    limit: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Load the model in ``model_folder`` and score every record under ``data_folder``."""
    predictor = load_model(model_folder, verbose=verbose, device=device)
    records = find_records(data_folder)
    if limit > 0:
        records = records[:limit]
    if verbose:
        print(f"[predict] {len(records)} records, batch {batch_size}, {num_workers} workers, "
              f"window_mode={predictor.window_mode}, wise_ft_alpha={predictor.wise_ft_alphas}", flush=True)
    return predict_records(predictor, records, batch_size=batch_size, num_workers=num_workers)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-m", "--model-folder", required=True)
    ap.add_argument("-d", "--data-folder", required=True)
    ap.add_argument("-o", "--out", required=True, help="output CSV")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--device", default=None)
    ap.add_argument("--cap-gib", type=float, default=14.5, help="CUDA allocator cap (T4 emulation)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N records (debug)")
    args = ap.parse_args(argv)

    if torch.cuda.is_available() and args.cap_gib > 0:
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(args.cap_gib / total, 1.0), 0)
        torch.cuda.reset_peak_memory_stats(0)

    started = time.time()
    df = predict_folder(
        args.model_folder, args.data_folder, batch_size=args.batch_size,
        num_workers=args.workers, device=args.device, limit=args.limit,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    elapsed = time.time() - started
    n_failed = int(df["prob"].isna().sum())
    print(f"[predict] wrote {out}: {len(df)} records, {n_failed} failed, {elapsed / 60:.1f} min "
          f"({len(df) / max(elapsed, 1e-9):.0f} records/s)", flush=True)
    if torch.cuda.is_available():
        print(f"[predict] PEAK ALLOCATED: {torch.cuda.max_memory_allocated(0) / 2 ** 30:.2f} GiB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
