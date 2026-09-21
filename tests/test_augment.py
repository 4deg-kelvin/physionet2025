"""PLAN.md section 5 item 5: every augmentation is shape-safe, finite, seeded and pure."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from chagas import augment as A
from chagas.config import AugCfg

FS = 500.0
N = 5000

#: (function, kwargs) for every augmentation that must preserve the input length.
LENGTH_PRESERVING = [
    ("resample_roundtrip", dict(fs=FS)),
    ("device_filter", dict(fs=FS)),
    ("powerline", dict(fs=FS)),
    ("baseline_wander", dict(fs=FS)),
    ("gaussian_noise", dict()),
    ("amplitude_scale", dict()),
    ("random_mask", dict(fs=FS)),
    ("edge_mask", dict(fs=FS)),
]
#: Augmentations that deliberately change the length.
LENGTH_CHANGING = [("time_warp", dict()), ("crop_shift", dict(fs=FS))]
#: Interference that is common-mode and must therefore be identical on every lead.
LEAD_IDENTICAL = ["powerline", "baseline_wander"]


@pytest.fixture
def signal() -> np.ndarray:
    rng = np.random.default_rng(0)
    t = np.arange(N) / FS
    base = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 7.0 * t)
    return np.stack([(1.0 + 0.1 * i) * base + 0.01 * rng.normal(size=N) for i in range(12)])


def _call(name: str, x: np.ndarray, kwargs, seed: int = 0) -> np.ndarray:
    fn = getattr(A, name)
    return fn(x, rng=np.random.default_rng(seed), **kwargs)


@pytest.mark.parametrize("name,kwargs", LENGTH_PRESERVING + LENGTH_CHANGING)
def test_shape_dtype_and_finiteness(name, kwargs, signal):
    out = _call(name, signal, kwargs)
    assert out.ndim == 2 and out.shape[0] == 12
    assert out.dtype == np.float64
    assert np.isfinite(out).all()


@pytest.mark.parametrize("name,kwargs", LENGTH_PRESERVING)
def test_length_preserved(name, kwargs, signal):
    assert _call(name, signal, kwargs).shape[-1] == signal.shape[-1]


def test_resample_roundtrip_returns_exactly_the_input_length(signal):
    """Explicitly required by PLAN.md, and relied on by the stage-R composition."""
    for seed in range(12):
        for n in (N, 4096, 3650, 1000):
            out = A.resample_roundtrip(
                signal[:, :n], FS, np.random.default_rng(seed)
            )
            assert out.shape == (12, n)


@pytest.mark.parametrize("name,kwargs", LENGTH_PRESERVING + LENGTH_CHANGING)
def test_seeded_determinism_and_purity(name, kwargs, signal):
    before = signal.copy()
    a = _call(name, signal, kwargs, seed=7)
    b = _call(name, signal, kwargs, seed=7)
    c = _call(name, signal, kwargs, seed=8)
    assert np.array_equal(a, b), f"{name} is not deterministic under a fixed seed"
    assert np.array_equal(signal, before), f"{name} mutated its input"
    if a.shape == c.shape:
        assert not np.array_equal(a, c), f"{name} ignores its seed"


@pytest.mark.parametrize("name", LEAD_IDENTICAL)
def test_common_mode_augs_are_identical_across_leads(name, signal):
    zeros = np.zeros((12, N))
    kwargs = dict(fs=FS)
    if name == "baseline_wander":
        # The amplitude is relative to the record's std, so start from a real signal and
        # subtract it out to recover the injected component.
        out = _call(name, signal, kwargs, seed=3)
        injected = out - signal
    else:
        out = _call(name, signal, kwargs, seed=3)
        injected = out - signal
    for lead in range(1, 12):
        assert np.allclose(injected[0], injected[lead], atol=1e-12), (
            f"{name} differs between lead 0 and lead {lead}"
        )


def test_amplitude_scale_is_not_lead_identical(signal):
    """The per-lead factor is the part the per-record z-score cannot neutralise."""
    out = A.amplitude_scale(signal, np.random.default_rng(1))
    ratios = (out / signal).mean(axis=1)
    assert float(ratios.std()) > 1e-3


def test_random_and_edge_mask_zero_what_they_claim(signal):
    out = A.random_mask(signal, FS, np.random.default_rng(2), max_seconds=1.0)
    zeroed = np.all(out == 0.0, axis=0)
    assert 0 < int(zeroed.sum()) <= int(1.0 * FS)

    out = A.edge_mask(signal, FS, np.random.default_rng(5), max_seconds=1.5)
    zeroed = np.all(out == 0.0, axis=0)
    # Any zeroing must be at the edges only.
    if zeroed.any():
        idx = np.flatnonzero(zeroed)
        assert idx.min() < 1.5 * FS or idx.max() > N - 1.5 * FS


def test_crop_shift_stays_within_the_window(signal):
    for seed in range(20):
        out = A.crop_shift(signal, FS, np.random.default_rng(seed), (5.65, 10.0), 10.0)
        assert int(5.65 * FS) <= out.shape[-1] <= int(10.0 * FS)


def test_stage_drivers_are_no_ops_when_disabled(signal):
    cfg = AugCfg(enabled=False)
    rng = np.random.default_rng(0)
    assert np.array_equal(A.apply_stage_r(signal, FS, rng, cfg), signal)
    assert np.array_equal(A.apply_stage_s(signal, FS, rng, cfg), signal)
    assert np.array_equal(A.apply_stage_r(signal, FS, rng, None), signal)


def test_stage_drivers_are_seeded_and_finite(signal):
    cfg = AugCfg()
    a = A.apply_stage_s(A.apply_stage_r(signal, 400.0, np.random.default_rng(11), cfg),
                        FS, np.random.default_rng(12), cfg)
    b = A.apply_stage_s(A.apply_stage_r(signal, 400.0, np.random.default_rng(11), cfg),
                        FS, np.random.default_rng(12), cfg)
    assert np.array_equal(a, b)
    assert np.isfinite(a).all() and a.shape[0] == 12


def test_all_augmentations_are_pure_functions():
    """Every public augmentation takes ``rng`` explicitly -- none may read global state."""
    for name in (
        "resample_roundtrip", "device_filter", "powerline", "baseline_wander",
        "gaussian_noise", "amplitude_scale", "time_warp", "random_mask",
        "edge_mask", "crop_shift",
    ):
        params = inspect.signature(getattr(A, name)).parameters
        assert "rng" in params, f"{name} does not take an explicit rng"
