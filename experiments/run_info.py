#!/usr/bin/env python
"""Write ``results/<run>/run_info.json``: everything needed to say how a run was produced.

PLAN_R2.md section 6.  Called twice by ``run_ladder.sh`` / ``run_r2.sh``:

    python experiments/run_info.py --out R/run_info.json --phase start --config CFG
    ... train, infer, score ...
    python experiments/run_info.py --out R/run_info.json --phase end --model-folder R/model

``start`` records the git commit and dirty flag, the config path and its SHA-256, the
seed, hostname, GPU, SLURM job id, torch / lightning / numpy versions, the SHA-256 of
``pip freeze`` and the start time.  ``end`` adds the end time and the train/val record
counts by source x label x length group that ``chagas.training`` wrote into
``chagas_manifest.json`` (plus the WiSE-FT summary when present).

It lives on the experiment side, not in ``chagas.training``, on purpose: the judged
container must not shell out to ``git`` or ``pip`` (DECISIONS.md D-2026-10-01-3).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: Paths whose changes do not make a run "dirty": logs and results are outputs.
DIRTY_EXCLUDES = ("experiments/logs", "experiments/results")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd: Sequence[str], cwd: Optional[Path] = None) -> str:
    try:
        return subprocess.run(list(cmd), cwd=cwd, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def git_state(repo: Path = REPO_ROOT) -> Dict[str, Any]:
    """Commit, branch and a dirty flag over tracked files, ignoring logs and results."""
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo)
    dirty: List[str] = []
    for line in status.splitlines():
        path = line[3:].strip()
        if not any(path.startswith(ex) for ex in DIRTY_EXCLUDES):
            dirty.append(path)
    return {
        "commit": _run(["git", "rev-parse", "HEAD"], cwd=repo).strip(),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=repo).strip(),
        "dirty": bool(dirty),
        "dirty_paths": dirty[:50],
    }


def environment() -> Dict[str, Any]:
    """Host, GPU and library versions, and a fingerprint of the whole environment."""
    import numpy
    import torch

    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", ""),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
    }
    for module in ("lightning", "pytorch_lightning"):
        try:
            info[module] = __import__(module).__version__
        except ImportError:
            info[module] = ""
    freeze = _run([sys.executable, "-m", "pip", "freeze"])
    info["pip_freeze_sha256"] = hashlib.sha256(freeze.encode()).hexdigest()
    info["pip_freeze_lines"] = len(freeze.splitlines())
    return info


def start_info(config_path: str, name: str = "") -> Dict[str, Any]:
    from chagas.config import ExperimentCfg

    cfg = ExperimentCfg.from_yaml(config_path)
    return {
        "name": name or cfg.name,
        "config_path": str(Path(config_path).resolve()),
        "config_sha256": sha256_file(config_path),
        "seed": int(cfg.train.seed),
        "git": git_state(),
        "environment": environment(),
        "start_time": _dt.datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def end_info(model_folder: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {"end_time": _dt.datetime.now().astimezone().isoformat(timespec="seconds")}
    manifest = Path(model_folder) / "chagas_manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text())
        out["data_summary"] = data.get("data_summary", [])
        out["members"] = data.get("members", [])
        out["wise_ft"] = [t.get("wise_ft") for t in data.get("training", []) if t.get("wise_ft")]
        out["best_by_stage"] = [t.get("best_by_stage") for t in data.get("training", [])]
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase", choices=("start", "end"), required=True)
    ap.add_argument("--config", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--model-folder", default="")
    args = ap.parse_args(argv)

    path = Path(args.out)
    current: Dict[str, Any] = json.loads(path.read_text()) if path.is_file() and args.phase == "end" else {}
    if args.phase == "start":
        if not args.config:
            ap.error("--config is required for --phase start")
        current = start_info(args.config, args.name)
    else:
        current.update(end_info(args.model_folder))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(current, indent=2) + "\n")
    print(f"[run_info] {args.phase}: wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
