"""Shared pytest fixtures.

Everything here runs on CPU.  The two tests that need the real 85M-parameter ECG-FM
backbone (PLAN.md section 5 items 2 and 3) share **one** session-scoped, read-only
module so the checkpoint is loaded once; they run a single forward over two short
records, which stays well under 2 GB of RAM.  Tests that mutate the model (stage
switching, serialisation) build their own.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

#: 400-record smoke set shipped with the evaluation harness (flat WFDB, labels in the
#: headers).
TESTDATA = Path("/juice2/scr2/kelvinkn/other_work/edwards/fixbranch_test/testdata")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "backbone: needs the real ECG-FM checkpoint (slow, ~1 GB load)"
    )


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def testdata_dir() -> Path:
    if not TESTDATA.is_dir():
        pytest.skip(f"smoke data not available at {TESTDATA}")
    return TESTDATA


@pytest.fixture(scope="session")
def real_records(testdata_dir: Path) -> List[str]:
    """Absolute, extension-stripped paths of the smoke records, sorted."""
    stems = sorted(
        str(testdata_dir / p.stem) for p in testdata_dir.glob("*.hea")
    )
    if not stems:
        pytest.skip("no .hea files in the smoke data folder")
    return stems


@pytest.fixture(scope="session")
def checkpoint_path(repo_root: Path) -> Path:
    path = repo_root / "mimic_iv_ecg_finetuned.pt"
    if not path.exists():
        pytest.skip(f"ECG-FM checkpoint not found at {path}")
    return path


@pytest.fixture(scope="session")
def shared_module(checkpoint_path: Path):
    """One read-only :class:`chagas.model.ChagasModule` in eval mode, built once."""
    import torch

    from chagas.config import default_config
    from chagas.model import ChagasModule

    torch.manual_seed(0)
    cfg = default_config()
    module = ChagasModule(cfg=cfg.to_dict(), stage=1)
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
    return module
