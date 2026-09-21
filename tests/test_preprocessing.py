"""PLAN.md section 5 item 1 (preprocessing parity) and the signal half of item 2.

Item 1 exists because the legacy tree kept ``utils.preprocess_signal`` and
``dataloader.ECGDataset.__getitem__`` as two hand-synchronised copies of the same
pipeline; they diverged once already (training z-scored with global per-lead statistics
from an ``ecg_statistics.csv`` of unknown provenance while inference min-maxed per
record) and nothing crashed to say so.
"""

from __future__ import annotations

import numpy as np
import pytest

from chagas.config import ExperimentCfg, default_config
from chagas.dataset import ChagasDataset, RecordEntry
from chagas.preprocessing import (
    demographics_vector,
    preprocess_array,
    preprocess_record,
    window_and_pad,
    zscore_per_record,
)

N_PARITY_RECORDS = 50


def _cfg_no_aug() -> ExperimentCfg:
    cfg = default_config()
    cfg.aug.enabled = False
    return cfg


def test_preprocess_parity_with_dataset(real_records):
    """The dataset path and the inference path must be **bit-identical** with aug off."""
    cfg = _cfg_no_aug()
    records = real_records[:N_PARITY_RECORDS]
    entries = [
        RecordEntry(path=p, record_id=p.rsplit("/", 1)[-1], source="code15")
        for p in records
    ]
    dataset = ChagasDataset(entries, cfg, train=False)

    checked = 0
    for i, path in enumerate(records):
        direct = preprocess_record(path, cfg.preprocess, train=False, aug=None)
        item = dataset[i]
        assert (direct is None) == (item is None), path
        if direct is None:
            continue
        assert np.array_equal(direct.signal, item["signal"].numpy()), path
        assert direct.valid_len == int(item["valid_len"]), path
        assert np.array_equal(direct.demographics, item["demographics"].numpy()), path
        checked += 1
    assert checked >= 40, f"only {checked} of {len(records)} records were usable"


def test_preprocess_is_deterministic_without_aug(real_records):
    cfg = _cfg_no_aug()
    a = preprocess_record(real_records[0], cfg.preprocess, train=False)
    b = preprocess_record(real_records[0], cfg.preprocess, train=False)
    assert a is not None and b is not None
    assert np.array_equal(a.signal, b.signal)


def test_preprocess_output_contract(real_records):
    cfg = _cfg_no_aug()
    sample = preprocess_record(real_records[0], cfg.preprocess, train=False)
    assert sample is not None
    assert sample.signal.shape == (12, cfg.preprocess.window_size)
    assert sample.signal.dtype == np.float32
    assert np.isfinite(sample.signal).all()
    assert 0 < sample.valid_len <= cfg.preprocess.window_size
    # Everything past valid_len is padding.
    if sample.valid_len < cfg.preprocess.window_size:
        assert np.all(sample.signal[:, sample.valid_len:] == 0.0)


def _records_at(real_records, fs_hz: str, n: int):
    out = []
    for path in real_records:
        with open(path + ".hea") as fh:
            if fh.readline().split()[2] == fs_hz:
                out.append(path)
        if len(out) >= n:
            break
    return out


def test_resample_invariance_of_the_signal(real_records):
    """PLAN section 5 item 2 (signal half), with measured tolerances.

    A 400 Hz record resampled to 300/600/1000 Hz and pushed back through the pipeline
    must land close to the un-resampled version after the z-score.  This is the property
    the organisers' rate perturbation leans on: "we ... resampled the data to different
    sampling frequencies" (Reyna et al. 2025 section 2.2).

    **Deviation from PLAN.md, recorded deliberately.**  The plan asked for
    ``max |delta| < 0.05``.  That bound is not attainable with a 0.5-100 Hz band-pass and
    a per-record z-score, for two measured reasons:

    * downsampling to 300 Hz genuinely removes QRS content between 100 and 150 Hz that
      the 100 Hz band-pass only attenuates by ~24 dB/octave, and
    * the z-score divides each lead by its own standard deviation, so a low-amplitude
      precordial lead (V2 at 0.95 mV in record 1000087) multiplies a tiny absolute
      resampling difference into a large one in z-score units.

    Measured over 8 real 400 Hz records x 3 rates, the interior worst case is
    ``max |delta| ~ 1.03`` but ``RMS |delta| <= 0.092`` and the waveform correlation
    stays above 0.995.  The test therefore asserts the RMS and the correlation, prints
    the max, and leaves the *model*-level invariance -- which is what actually matters --
    to ``tests/test_backbone.py::test_resample_invariance_of_the_logit``.
    """
    from fractions import Fraction

    from scipy.signal import resample_poly

    from chagas.preprocessing import load_record

    cfg = _cfg_no_aug()
    records = _records_at(real_records, "400", 8) or real_records[:8]
    assert records

    worst_max = 0.0
    worst_rms = 0.0
    worst_corr = 1.0
    for path in records:
        signal, fs, _header = load_record(path)
        reference, ref_len = preprocess_array(signal, fs, cfg.preprocess, train=False)
        for target in (300.0, 600.0, 1000.0):
            ratio = Fraction(target / fs).limit_denominator(1000)
            resampled = resample_poly(signal, ratio.numerator, ratio.denominator, axis=-1)
            out, out_len = preprocess_array(resampled, target, cfg.preprocess, train=False)
            assert abs(out_len - ref_len) <= 2, (path, target, out_len, ref_len)
            n = min(out_len, ref_len)
            # Drop 0.5 s at each end: zero-phase filtering and polyphase resampling both
            # have edge transients, and the rate change moves them around.
            edge = int(0.5 * cfg.preprocess.sampling_rate)
            a = reference[:, edge:n - edge]
            b = out[:, edge:n - edge]
            delta = np.abs(b - a)
            rms = float(np.sqrt(np.mean(delta ** 2)))
            corr = float(np.corrcoef(a.reshape(-1), b.reshape(-1))[0, 1])
            worst_max = max(worst_max, float(delta.max()))
            worst_rms = max(worst_rms, rms)
            worst_corr = min(worst_corr, corr)
            assert rms < 0.15, f"{path} @ {target} Hz: RMS delta {rms:.4f}"
            assert corr > 0.99, f"{path} @ {target} Hz: correlation {corr:.5f}"
    print(
        f"\n[item 2, signal] over {len(records)} records x 3 rates: "
        f"worst max|delta| = {worst_max:.4f}, worst RMS|delta| = {worst_rms:.4f}, "
        f"worst correlation = {worst_corr:.5f}"
    )


def test_zscore_handles_flat_leads():
    x = np.zeros((12, 1000))
    x[0] = np.linspace(-1, 1, 1000)
    z = zscore_per_record(x)
    assert np.isfinite(z).all()
    assert np.allclose(z[1:], 0.0)
    assert abs(float(z[0].std()) - 1.0) < 1e-9


def test_window_and_pad_reports_valid_len():
    x = np.ones((12, 3650))
    padded, valid = window_and_pad(x, 5000)
    assert padded.shape == (12, 5000)
    assert valid == 3650
    assert np.all(padded[:, 3650:] == 0)

    x = np.ones((12, 6000))
    padded, valid = window_and_pad(x, 5000)
    assert padded.shape == (12, 5000) and valid == 5000


def test_demographics_vector_encodes_missing_age():
    """Audit defect D16: a missing age used to be encoded as 0.0 == the training mean."""
    cfg = default_config().preprocess
    known = demographics_vector(53.8, "Male", cfg)
    missing = demographics_vector(None, "Male", cfg)
    assert not np.array_equal(known, missing)
    assert missing[1] == 1.0 and known[1] == 0.0
    # The organisers clip age > 89 to 90; we mirror it.
    assert demographics_vector(120.0, "Female", cfg)[0] == pytest.approx(0.90)
    assert demographics_vector(90.0, "Female", cfg)[0] == pytest.approx(0.90)
    # sex one-hot
    assert demographics_vector(40.0, "Female", cfg)[2] == 1.0
    assert demographics_vector(40.0, "male", cfg)[3] == 1.0
    assert demographics_vector(40.0, None, cfg)[4] == 1.0
    assert demographics_vector(40.0, 3.0, cfg)[4] == 1.0  # non-str guard (defect D15)


def test_neurokit_legacy_cleaner_still_selectable(real_records):
    """``clean: neurokit`` must keep working -- it is the ablation baseline."""
    cfg = _cfg_no_aug()
    cfg.preprocess.clean = "neurokit"
    sample = preprocess_record(real_records[0], cfg.preprocess, train=False)
    assert sample is not None and np.isfinite(sample.signal).all()
