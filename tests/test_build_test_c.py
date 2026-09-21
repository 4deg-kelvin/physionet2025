"""PLAN.md section 5 item 13: the Test-C builder is deterministic and its output loads.

Test C is Test A perturbed the way the organisers perturbed the hidden cohorts (Reyna
et al. 2025 section 2.2).  It is generated once and cached, so it must be reproducible
from the seed alone -- otherwise two ladder rows are not comparable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location(
    "build_test_C", REPO / "experiments" / "build_test_C.py"
)
assert _spec and _spec.loader
build_test_C = importlib.util.module_from_spec(_spec)
sys.modules["build_test_C"] = build_test_C
_spec.loader.exec_module(build_test_C)

from chagas.config import default_config
from chagas.preprocessing import load_record, preprocess_record


def test_record_seed_is_stable_and_stem_dependent():
    a = build_test_C.record_seed("1000010")
    assert a == build_test_C.record_seed("1000010")
    assert a != build_test_C.record_seed("1000011")
    assert a != build_test_C.record_seed("1000010", base_seed=1)


def test_perturbation_is_deterministic_and_shape_preserving():
    rng_seed = 123
    x = np.random.default_rng(0).normal(size=(12, 4000))
    a = build_test_C.perturb(x, 400.0, np.random.default_rng(rng_seed))
    b = build_test_C.perturb(x, 400.0, np.random.default_rng(rng_seed))
    c = build_test_C.perturb(x, 400.0, np.random.default_rng(rng_seed + 1))
    assert a.shape == x.shape
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert np.isfinite(a).all()


def test_perturbation_actually_perturbs_but_keeps_the_waveform():
    x = np.random.default_rng(1).normal(size=(12, 4000))
    t = np.arange(4000) / 400.0
    x = x * 0.02 + np.sin(2 * np.pi * 1.1 * t)[None, :]
    y = build_test_C.perturb(x, 400.0, np.random.default_rng(5))
    assert not np.array_equal(x, y)
    corr = float(np.corrcoef(x.reshape(-1), y.reshape(-1))[0, 1])
    assert corr > 0.9, f"the perturbation destroyed the signal (corr {corr:.3f})"


def test_written_records_load_through_the_pipeline(tmp_path, real_records):
    """Round trip: write a few perturbed records and read them back with our own loader."""
    cfg = default_config()
    cfg.aug.enabled = False
    dest = tmp_path / "data_test_C"

    for src in real_records[:3]:
        stem = build_test_C.write_record(src, str(dest))
        assert stem is not None

        _sig, fs, header = load_record(stem)
        src_header = open(src + ".hea").read()
        # The '#' block must survive byte for byte, or evaluate_model.py loses the labels.
        src_comments = [l for l in src_header.split("\n") if l.startswith("#")]
        new_comments = [l for l in header.split("\n") if l.startswith("#")]
        assert new_comments == src_comments, stem

        sample = preprocess_record(stem, cfg.preprocess, train=False)
        assert sample is not None
        assert sample.signal.shape == (12, cfg.preprocess.window_size)
        assert np.isfinite(sample.signal).all()

        original = preprocess_record(src, cfg.preprocess, train=False)
        assert original is not None
        assert not np.array_equal(sample.signal, original.signal)


def test_writing_is_reproducible(tmp_path, real_records):
    a = tmp_path / "a"
    b = tmp_path / "b"
    src = real_records[0]
    build_test_C.write_record(src, str(a))
    build_test_C.write_record(src, str(b))
    sig_a, _fs, _h = load_record(str(a / Path(src).name))
    sig_b, _fs, _h = load_record(str(b / Path(src).name))
    assert np.array_equal(sig_a, sig_b)
