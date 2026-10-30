"""Extract the learned layer-aggregation weights from saved Lightning checkpoints.

The model combines the outputs of ECG-FM's 12 transformer layers with a static
softmax over ``ECGFMFeatureExtractor.layer_weights`` (``chagas/model.py``, the
``nn.Parameter`` zero-initialised to a uniform mixture; ``_aggregate`` applies
``torch.softmax`` to it).  In a ``ChagasModule`` checkpoint the raw logits live in
``state_dict["feature_extractor.layer_weights"]``; ``on_save_checkpoint`` also stores
the softmaxed values under ``checkpoint["chagas"]["layer_weights"]``.  This script
reads the logits, applies the softmax, cross-checks against the stored block when it
is present, and writes one row per (run, member, layer).

CPU only; nothing is instantiated, so the 85 M-parameter backbone is never built.
Run through ``experiments/batch_layer_weights.sh``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

KEY = "feature_extractor.layer_weights"
SKIP = {"e2e_smoke", "r2_smoke", "r3_smoke", "smoke_r2"}


def member_index(path: Path) -> int:
    m = re.fullmatch(r"chagas_model(?:_(\d+))?\.ckpt", path.name)
    return int(m.group(1)) if m and m.group(1) else 0


def extract(path: Path) -> dict:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    logits = sd[KEY].detach().float()
    weights = torch.softmax(logits, dim=0).numpy()
    block = ckpt.get("chagas", {})
    stored = block.get("layer_weights")
    max_diff = float("nan")
    if stored is not None:
        max_diff = float(np.max(np.abs(np.asarray(stored, dtype=np.float64) - weights)))
    cfg = block.get("config", {}) or {}
    pre = cfg.get("preprocess", {}) or {}
    train = cfg.get("train", {}) or {}
    model = cfg.get("model", {}) or {}
    return {
        "logits": logits.numpy(),
        "weights": weights,
        "stored_max_abs_diff": max_diff,
        "stage": block.get("stage"),
        "window_seconds": pre.get("window_seconds"),
        "window_mode": pre.get("window_mode"),
        "two_stage": train.get("two_stage"),
        "ranking_loss": train.get("ranking_loss"),
        "demographics": model.get("demographics"),
        "seed": train.get("seed"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/results")
    ap.add_argument("--out", default="experiments/results/entry_final_r3/layer_weights.csv")
    ap.add_argument(
        "--first",
        nargs="*",
        default=["entry_final_r3", "r3_B0_s42", "r3_B0_s43", "r3_B0_s44"],
        help="runs to process first, so a timeout still leaves the important rows",
    )
    args = ap.parse_args()

    results = Path(args.results)
    runs = sorted(p.name for p in results.iterdir() if (p / "model").is_dir() and p.name not in SKIP)
    ordered = [r for r in args.first if r in runs] + [r for r in runs if r not in args.first]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run", "member", "layer", "weight", "logit", "stage", "window_seconds", "window_mode",
        "two_stage", "ranking_loss", "demographics", "seed", "stored_max_abs_diff",
    ]
    n_done = 0
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for run in ordered:
            for ckpt in sorted((results / run / "model").glob("chagas_model*.ckpt"), key=member_index):
                t0 = time.time()
                try:
                    info = extract(ckpt)
                except Exception as exc:  # report and move on; one bad file must not stop the rest
                    print(f"[layer_weights] FAIL {run}/{ckpt.name}: {exc!r}", flush=True)
                    continue
                member = member_index(ckpt)
                for layer, (w, z) in enumerate(zip(info["weights"], info["logits"]), start=1):
                    writer.writerow({
                        "run": run, "member": member, "layer": layer,
                        "weight": f"{w:.6f}", "logit": f"{z:.6f}",
                        "stage": info["stage"], "window_seconds": info["window_seconds"],
                        "window_mode": info["window_mode"], "two_stage": info["two_stage"],
                        "ranking_loss": info["ranking_loss"], "demographics": info["demographics"],
                        "seed": info["seed"], "stored_max_abs_diff": f"{info['stored_max_abs_diff']:.2e}",
                    })
                fh.flush()
                n_done += 1
                ws = " ".join(f"{w:.3f}" for w in info["weights"])
                print(
                    f"[layer_weights] {run}/{ckpt.name} ({time.time() - t0:.1f}s) "
                    f"win={info['window_seconds']} stage={info['stage']} "
                    f"stored-diff={info['stored_max_abs_diff']:.1e}: {ws}",
                    flush=True,
                )
    print(f"[layer_weights] wrote {out} ({n_done} checkpoints)", flush=True)
    return 0 if n_done else 1


if __name__ == "__main__":
    sys.exit(main())
