"""PLAN_R2.md section 7 item 3: fixed 5 s windows (``preprocess.window_mode: "multi"``)."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest
import torch
from scipy.signal import resample_poly

from chagas import augment as A
from chagas.config import default_config
from chagas.dataset import ChagasDataset, RecordEntry, collate
from chagas.model import segment_mean
from chagas.preprocessing import (
    clean_signal,
    extract_windows,
    load_record,
    preprocess_array,
    random_window,
    window_and_pad,
    window_starts,
    zscore_per_record,
)

FS = 500
W = 5 * FS  # 2500 samples


def _multi_cfg():
    cfg = default_config()
    cfg.preprocess.window_seconds = 5.0
    cfg.preprocess.window_mode = "multi"
    return cfg


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (7.34, [0, 1170]),            # CODE-15 short: 3670 samples -> 2 windows
        (10.0, [0, 2500]),            # PTB-XL: exactly 2 windows, no overlap
        (4.0, [0]),                   # shorter than a window: one padded window
        (23.0, [0, 3000, 6000, 9000]),  # ceil(23/5) = 5, capped at 4
    ],
)
def test_window_count_and_starts(seconds, expected):
    n = int(round(seconds * FS))
    starts = window_starts(n, W, max_windows=4)
    assert starts.tolist() == expected
    assert starts[-1] + W == max(n, W) or n < W, "the last window must end at the last sample"


def test_no_padding_for_records_at_least_one_window_long():
    rng = np.random.default_rng(0)
    for seconds in (5.0, 7.34, 10.0, 10.24, 23.0):
        x = rng.normal(size=(12, int(round(seconds * FS))))
        windows, lengths = extract_windows(x, W, 4)
        assert windows.shape[1:] == (12, W)
        assert (lengths == W).all(), seconds
        for w, s in zip(windows, window_starts(x.shape[1], W, 4)):
            assert np.array_equal(w, x[:, s:s + W])
    x = rng.normal(size=(12, 4 * FS))
    windows, lengths = extract_windows(x, W, 4)
    assert windows.shape == (1, 12, W) and lengths.tolist() == [4 * FS]
    assert np.all(windows[0, :, 4 * FS:] == 0.0), "a short record keeps its padding (and mask)"


def test_training_crop_lies_inside_the_valid_region_and_is_seeded():
    n = 3670
    x = np.tile(np.arange(n, dtype=np.float64), (12, 1))  # value == sample index
    starts = set()
    for seed in range(50):
        crop, valid = random_window(x, W, np.random.default_rng(seed))
        assert valid == W and crop.shape == (12, W)
        start = int(crop[0, 0])
        assert 0 <= start <= n - W
        assert np.array_equal(crop, x[:, start:start + W])
        again, _ = random_window(x, W, np.random.default_rng(seed))
        assert np.array_equal(crop, again)
        starts.add(start)
    assert len(starts) > 10, "crops should move around"
    short, valid = random_window(x[:, :2000], W, np.random.default_rng(0))
    assert valid == 2000 and np.all(short[:, 2000:] == 0)


def test_preprocess_array_multi_shapes():
    cfg = _multi_cfg()
    x = np.random.default_rng(1).normal(size=(12, 2934))  # 7.335 s at 400 Hz
    windows, lengths = preprocess_array(x, 400.0, cfg.preprocess, train=False)
    assert windows.shape == (2, 12, W) and windows.dtype == np.float32
    assert lengths.tolist() == [W, W]
    one, valid = preprocess_array(x, 400.0, cfg.preprocess, train=True, aug=cfg.aug,
                                  rng=np.random.default_rng(0))
    assert one.shape == (12, W) and valid == W


def _round1_preprocess_array(x, fs, cfg, *, train, aug, rng):
    """Verbatim round-1 signal pipeline, the reference for ``window_mode: "single"``."""
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    use_aug = bool(train and aug is not None and aug.enabled)
    if use_aug:
        x = A.apply_stage_r(x, fs, rng, aug)
    if float(fs) != float(cfg.sampling_rate):
        ratio = Fraction(float(cfg.sampling_rate) / float(fs)).limit_denominator(1000)
        x = np.ascontiguousarray(resample_poly(x, ratio.numerator, ratio.denominator, axis=-1))
    x = clean_signal(x, float(cfg.sampling_rate), cfg)
    if use_aug:
        x = A.apply_stage_s(x, float(cfg.sampling_rate), rng, aug, cfg.window_seconds)
    x = zscore_per_record(x)
    signal, valid_len = window_and_pad(x, cfg.window_size)
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal.astype(np.float32, copy=False), valid_len


def test_single_mode_is_bit_exact_round_one(real_records):
    cfg = default_config()
    assert cfg.preprocess.window_mode == "single"
    for path in real_records[:12]:
        x, fs, _ = load_record(path)
        for train, seed in ((False, None), (True, 0), (True, 123)):
            got = preprocess_array(x, fs, cfg.preprocess, train=train, aug=cfg.aug,
                                   rng=None if seed is None else np.random.default_rng(seed))
            ref = _round1_preprocess_array(x, fs, cfg.preprocess, train=train, aug=cfg.aug,
                                           rng=None if seed is None else np.random.default_rng(seed))
            assert got[1] == ref[1], (path, train, seed)
            assert np.array_equal(got[0], ref[0]), (path, train, seed)


def test_dataset_and_collate_concatenate_windows(real_records):
    cfg = _multi_cfg()
    entries = [RecordEntry(path=p, record_id=p.rsplit("/", 1)[-1], source="code15") for p in real_records[:6]]
    dataset = ChagasDataset(entries, cfg, train=False)
    items = [dataset[i] for i in range(len(entries))]
    items = [it for it in items if it is not None]
    batch = collate(items)
    counts = batch["n_windows"]
    assert counts.shape == (len(items),)
    assert batch["signal"].shape == (int(counts.sum()), 12, W)
    assert batch["valid_len"].shape == (int(counts.sum()),)
    assert batch["label_hard"].shape == (len(items),)
    train_item = ChagasDataset(entries, cfg, train=True)[0]
    assert train_item["signal"].shape == (1, 12, W) and int(train_item["n_windows"]) == 1


def test_segment_mean_is_exact():
    x = torch.tensor([[1.0], [2.0], [4.0], [8.0], [16.0], [32.0]])
    out = segment_mean(x, torch.tensor([1, 2, 3]))
    assert out.reshape(-1).tolist() == [1.0, 3.0, 56.0 / 3.0]
    with pytest.raises(ValueError):
        segment_mean(x, torch.tensor([1, 2]))


@pytest.mark.backbone
def test_record_logit_is_the_mean_of_its_window_logits(shared_module, real_records):
    cfg = _multi_cfg()
    x, fs, _ = load_record(real_records[0])
    windows, lengths = preprocess_array(x, fs, cfg.preprocess, train=False)
    assert windows.shape[0] >= 2
    signal = torch.from_numpy(windows)
    valid = torch.from_numpy(lengths)
    demo = torch.zeros(1, 5)
    with torch.no_grad():
        joint = shared_module(signal, valid, demo, torch.tensor([windows.shape[0]]))["logits"]
        per_window = torch.stack([
            shared_module(signal[i:i + 1], valid[i:i + 1], demo)["logits"][0]
            for i in range(windows.shape[0])
        ])
    assert joint.shape == (1,)
    assert torch.allclose(joint[0], per_window.mean(), atol=1e-5), (joint, per_window)
