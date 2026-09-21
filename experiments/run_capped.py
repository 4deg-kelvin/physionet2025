#!/usr/bin/env python
"""Run one of the organisers' unmodified wrapper scripts under a T4-sized memory cap.

The grader runs on a ``g4dn.4xlarge`` with one NVIDIA **T4 (16 GB)**.  Our TitanRTX is
the same Turing architecture (SM 7.5) but has 24 GB, so a run that fits here can still
OOM there.  Capping the CUDA allocator to 14.5 GiB makes the failure mode
representative, and ``torch.cuda.max_memory_allocated()`` afterwards is the number to
compare against the budget.

The wrapper script is executed with ``runpy`` under ``__main__`` so it runs exactly as
``python train_model.py ...`` would -- nothing in ``train_model.py`` / ``run_model.py``
is modified or imported around.

    python experiments/run_capped.py --cap 14.5 --script train_model.py -- -d D -m M -v
"""

from __future__ import annotations

import argparse
import os
import runpy
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cap", type=float, default=14.5, help="GiB of CUDA allocator budget")
    ap.add_argument("--script", required=True, help="path to the wrapper script to run")
    ap.add_argument("--label", default="", help="prefix for the reported timings")
    ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then the script's own argv")
    args = ap.parse_args(argv)

    import torch

    label = args.label or Path(args.script).stem
    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(args.cap / total, 1.0), 0)
        torch.cuda.reset_peak_memory_stats(0)
        print(
            f"[{label}] capped at {args.cap} GiB of {total:.1f} GiB "
            f"({torch.cuda.get_device_name(0)}, sm {torch.cuda.get_device_capability(0)})",
            flush=True,
        )
    else:
        print(f"[{label}] no CUDA device; running on CPU", flush=True)

    script = str(REPO_ROOT / args.script) if not os.path.isabs(args.script) else args.script
    rest: List[str] = [a for a in args.rest if a != "--"]
    sys.argv = [os.path.basename(script)] + rest
    os.chdir(str(REPO_ROOT))
    # runpy.run_path does not put the script's directory on sys.path the way `python
    # train_model.py` does, and train_model.py does `from helper_code import *`.
    if str(REPO_ROOT) in sys.path:
        sys.path.remove(str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT))

    started = time.time()
    status = 0
    try:
        runpy.run_path(script, run_name="__main__")
    except SystemExit as exc:  # the wrapper scripts may sys.exit()
        status = int(exc.code or 0)
    finally:
        elapsed = (time.time() - started) / 60.0
        if torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated(0) / 2 ** 30
            reserved = torch.cuda.max_memory_reserved(0) / 2 ** 30
            print(f"[{label}] PEAK ALLOCATED: {peak:.2f} GiB", flush=True)
            print(f"[{label}] PEAK RESERVED : {reserved:.2f} GiB", flush=True)
        print(f"[{label}] WALL TIME     : {elapsed:.1f} min", flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
