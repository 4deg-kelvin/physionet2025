"""The ONE preprocessing pipeline (PLAN.md section 3.5).

Both the training path (:mod:`chagas.dataset`) and the inference path
(:mod:`chagas.inference`) call :func:`preprocess_record`.  The legacy tree kept three
hand-synchronised copies of this logic (``utils.preprocess_signal``,
``dataloader.ECGDataset.__getitem__`` and ``transformer.py``), so train/inference skew
was always one edit away (audit defect D11); ``tests/test_preprocessing.py`` asserts
the two paths are bit-identical.

Pipeline
--------
1. load, reorder leads to ECG-FM's order, transpose to ``(12, T)``, reject non-12-lead;
2. **stage-R augmentation** at the record's native rate (training only);
3. ``resample_poly`` to 500 Hz -- anti-aliased polyphase, replacing the legacy FFT
   ``scipy.signal.resample`` (audit defect D7);
4. cleaning: one vectorised ``sosfiltfilt`` Butterworth band-pass over all 12 leads at
   once plus 50 **and** 60 Hz notches (audit defect D8: 12 sequential ``nk.ecg_clean``
   calls were the inference bottleneck, and their notch is 50 Hz only while Brazil's
   mains is 60 Hz).  ``clean="neurokit"`` keeps the legacy behaviour for the ablation;
5. **stage-S augmentation** (training only);
6. per-record per-lead z-score -- what ECG-FM was pretrained on (McKeen et al. 2025;
   ``fairseq-signals/scripts/preprocess/ecg/preprocess.py``);
7. window to 10 s = 5000 samples, zero-pad at the end, and **return ``valid_len``** so
   the model can build a real padding mask (audit defect D3).

A note on filtering and the backbone: ECG-FM was pretrained on z-scored, essentially
unfiltered signal, so the band-pass is kept deliberately wide (0.5-100 Hz) -- minimal
filtering stays closest to pretraining while still removing the mains and the drift the
organisers' device filters introduce.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, tf2sos

from . import augment as A
from .config import ECG_FM_LEAD_ORDER, AugCfg, PreprocessCfg

__all__ = [
    "Sample",
    "load_record",
    "reorder_leads",
    "clean_signal",
    "zscore_per_record",
    "window_and_pad",
    "demographics_vector",
    "preprocess_array",
    "preprocess_record",
    "preprocess",
]

#: Number of demographic features: ``[age/100, age_missing, sex_f, sex_m, sex_unknown]``.
N_DEMOGRAPHIC_FEATURES = 5


@dataclass
class Sample:
    """One model-ready record.

    Attributes
    ----------
    signal:
        ``(12, window_size)`` float32, z-scored and zero-padded at the end.
    valid_len:
        Number of leading samples of ``signal`` that are real signal.  Everything from
        ``valid_len`` on is zero padding and is masked out of attention and pooling.
    demographics:
        ``(5,)`` float32 ``[age/100, age_missing, sex_female, sex_male, sex_unknown]``.
        The explicit missing indicator is audit defect D16: encoding a missing age as
        0.0 made it indistinguishable from a 53.8-year-old.
    label_hard / label_soft:
        ``None`` at inference time.  ``label_hard`` is always the organisers' 0/1 label
        (metrics must use it -- audit defect D10); ``label_soft`` is what the loss sees.
    source:
        Normalised cohort name: ``"samitrop" | "ptbxl" | "code15" | "unknown"``.
    aux / aux_mask:
        Optional auxiliary targets and their availability mask (loss is applied only
        where the label exists).
    """

    signal: np.ndarray
    valid_len: int
    demographics: np.ndarray
    label_hard: Optional[int] = None
    label_soft: Optional[float] = None
    source: str = "unknown"
    record_id: str = ""
    aux: Optional[np.ndarray] = None
    aux_mask: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_record(record_path: str) -> Tuple[np.ndarray, float, str]:
    """Read a WFDB record as ``(signal[12, T] float64, fs, header_text)``.

    ``wfdb`` is imported lazily so importing this module stays cheap.
    Raises :class:`ValueError` when the record does not carry 12 usable leads.
    """
    import wfdb  # local import: keeps module import cheap and side-effect free

    header_path = record_path if record_path.endswith(".hea") else record_path + ".hea"
    stem = header_path[: -len(".hea")]
    with open(header_path, "r") as fh:
        header_text = fh.read()
    signal, fields = wfdb.rdsamp(stem)
    sig_names = list(fields.get("sig_name") or [])
    signal = reorder_leads(np.asarray(signal, dtype=np.float64), sig_names)
    signal = signal.T  # (leads, samples)
    if signal.shape[0] != len(ECG_FM_LEAD_ORDER):
        raise ValueError(
            f"{record_path}: got {signal.shape[0]} leads, expected {len(ECG_FM_LEAD_ORDER)}"
        )
    # wfdb emits NaN for gaps in some sources.  A single NaN propagates through the
    # z-score to the whole lead and then to a NaN probability, which the organisers'
    # evaluator reads as 0.  Replace before anything else.
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal, float(fields["fs"]), header_text


def reorder_leads(signal: np.ndarray, sig_names: Sequence[str]) -> np.ndarray:
    """Reorder ``(samples, leads)`` into ECG-FM's lead order, case-insensitively.

    All three training sources already emit this order, so this is a no-op today; it is
    enforced so a differently-ordered source in the hidden test set cannot silently feed
    the backbone scrambled channels.  (team)
    """
    if not sig_names:
        return signal
    wanted = list(ECG_FM_LEAD_ORDER)
    lower = {str(n).casefold(): i for i, n in enumerate(sig_names)}
    if [str(n) for n in sig_names] == wanted:
        return signal
    if not all(w.casefold() in lower for w in wanted):
        return signal  # not 12-lead; the caller's shape guard rejects it
    idx = [lower[w.casefold()] for w in wanted]
    return signal[:, idx]


# ---------------------------------------------------------------------------
# DSP
# ---------------------------------------------------------------------------
_SOS_CACHE: Dict[Any, np.ndarray] = {}


def _cleaning_sos(cfg: PreprocessCfg, fs: float) -> np.ndarray:
    """Cascade of second-order sections: band-pass, then one notch per mains frequency.

    Cached per (fs, parameters); designing Butterworth sections for every record was
    measurable overhead at 200k records.
    """
    key = (
        float(fs), cfg.bandpass_low_hz, cfg.bandpass_high_hz, cfg.bandpass_order,
        tuple(cfg.notch_hz), cfg.notch_q,
    )
    cached = _SOS_CACHE.get(key)
    if cached is not None:
        return cached
    nyq = fs / 2.0
    low = max(cfg.bandpass_low_hz / nyq, 1e-6)
    high = min(cfg.bandpass_high_hz / nyq, 0.99)
    sections = [butter(cfg.bandpass_order, [low, high], btype="bandpass", output="sos")]
    for f0 in cfg.notch_hz:
        if 0.0 < f0 < nyq:
            b, a = iirnotch(f0 / nyq, cfg.notch_q)
            sections.append(tf2sos(b, a))
    sos = np.concatenate(sections, axis=0)
    _SOS_CACHE[key] = sos
    return sos


def clean_signal(x: np.ndarray, fs: float, cfg: PreprocessCfg) -> np.ndarray:
    """Remove drift and mains from all 12 leads at once.

    ``"butterworth"`` -- one zero-phase ``sosfiltfilt`` pass over the whole ``(12, T)``
    array (band-pass 0.5-100 Hz order 2 plus 50 and 60 Hz notches, Q=30).  Hong et al.
    2025 use a Butterworth band-pass; the vectorisation and the second notch are the fix
    for audit defect D8.

    ``"neurokit"`` -- the team's legacy per-lead ``nk.ecg_clean`` loop, kept so the
    ablation can isolate the effect of the change.
    """
    if cfg.clean == "none":
        return np.ascontiguousarray(x, dtype=np.float64)
    if cfg.clean == "neurokit":
        import neurokit2 as nk  # local import: heavy, and only the legacy path needs it

        return np.stack([nk.ecg_clean(lead, sampling_rate=int(fs)) for lead in x]).astype(
            np.float64
        )
    sos = _cleaning_sos(cfg, fs)
    n = x.shape[-1]
    padlen = 3 * (2 * sos.shape[0] + 1)
    if n <= padlen + 1:
        # Too short to filter without padding artefacts dominating; a 2 s record is
        # already at the edge of usefulness, so pass it through rather than fail.
        return np.ascontiguousarray(x, dtype=np.float64)
    return np.ascontiguousarray(sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1))


def zscore_per_record(signal: np.ndarray) -> np.ndarray:
    """Per-lead z-score over the whole record (team; ECG-FM's pretraining recipe).

    Constant (flat) leads map to 0 rather than NaN, matching fairseq-signals'
    ``constant_lead_strategy='zero'``: ECG-FM's lead-masking objective masks with zeros,
    so a dead lead reads as "masked".  Applied **before** windowing so the statistics
    come from the whole recording, exactly as in pretraining.
    """
    sig = signal - signal.mean(axis=1, keepdims=True)
    std = sig.std(axis=1, keepdims=True)
    return np.divide(sig, std, out=np.zeros_like(sig), where=std != 0)


def window_and_pad(x: np.ndarray, window_size: int) -> Tuple[np.ndarray, int]:
    """Truncate to ``window_size`` or zero-pad at the end; return ``(signal, valid_len)``.

    The valid region is always a **prefix**, which is what fairseq-signals'
    ``_get_feat_extract_output_lengths`` assumes when it turns a sample-level mask into
    a token-level one.
    """
    n = x.shape[-1]
    if n >= window_size:
        return np.ascontiguousarray(x[:, :window_size]), window_size
    out = np.zeros((x.shape[0], window_size), dtype=x.dtype)
    out[:, :n] = x
    return out, int(n)


# ---------------------------------------------------------------------------
# demographics
# ---------------------------------------------------------------------------
def demographics_vector(
    age: Optional[float], sex: Optional[str], cfg: PreprocessCfg
) -> np.ndarray:
    """``[age/100, age_missing, sex_female, sex_male, sex_unknown]`` as float32.

    The organisers clip age above 89 to 90 (Reyna et al. 2025); mirror that so a
    hidden-set 95-year-old and a training 90-year-old land on the same value.  The
    explicit ``age_missing`` indicator is audit defect D16.
    """
    vec = np.zeros(N_DEMOGRAPHIC_FEATURES, dtype=np.float32)
    has_age = age is not None and isinstance(age, (int, float)) and not (
        isinstance(age, float) and math.isnan(age)
    )
    if has_age:
        clipped = float(np.clip(float(age), 0.0, cfg.age_clip_max))
        vec[0] = clipped / cfg.age_scale
        vec[1] = 0.0
    else:
        vec[0] = 0.0
        vec[1] = 1.0
    sex_str = sex.strip().casefold() if isinstance(sex, str) else ""
    if sex_str == "female":
        vec[2] = 1.0
    elif sex_str == "male":
        vec[3] = 1.0
    else:
        vec[4] = 1.0
    return vec


# ---------------------------------------------------------------------------
# the pipeline
# ---------------------------------------------------------------------------
def preprocess_array(
    x: np.ndarray,
    fs: float,
    cfg: PreprocessCfg,
    *,
    train: bool = False,
    aug: Optional[AugCfg] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, int]:
    """Signal half of the pipeline: ``(12, T)`` at ``fs`` -> ``(12, window_size)``, ``valid_len``.

    ``aug`` is applied only when ``train`` is True and ``aug.enabled``; inference is
    always deterministic.
    """
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float64))
    if x.ndim != 2 or x.shape[0] != len(ECG_FM_LEAD_ORDER):
        raise ValueError(f"expected a (12, T) array, got shape {x.shape}")
    use_aug = bool(train and aug is not None and aug.enabled)
    if use_aug and rng is None:
        rng = np.random.default_rng()

    # 2. acquisition-level augmentation at the native rate
    if use_aug:
        x = A.apply_stage_r(x, fs, rng, aug)

    # 3. rate conversion
    if float(fs) != float(cfg.sampling_rate):
        ratio = Fraction(float(cfg.sampling_rate) / float(fs)).limit_denominator(1000)
        x = np.ascontiguousarray(
            resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)
        )

    # 4. cleaning
    x = clean_signal(x, float(cfg.sampling_rate), cfg)

    # 5. signal-level augmentation
    if use_aug:
        x = A.apply_stage_s(x, float(cfg.sampling_rate), rng, aug, cfg.window_seconds)

    # 6. per-record z-score, before windowing
    x = zscore_per_record(x)

    # 7. window + pad, remembering how much is real
    signal, valid_len = window_and_pad(x, cfg.window_size)
    if not np.isfinite(signal).all():
        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return signal.astype(np.float32, copy=False), valid_len


def preprocess_record(
    record_path: str,
    cfg: PreprocessCfg,
    *,
    train: bool = False,
    aug: Optional[AugCfg] = None,
    rng: Optional[np.random.Generator] = None,
    want_label: bool = False,
) -> Optional[Sample]:
    """Full pipeline for one WFDB record.  Returns ``None`` for unusable records.

    ``want_label`` gates reading ``# Chagas label:`` from the header.  It must stay
    False at inference: ``chagas.labels.parse_patient_info`` raises
    :class:`NotImplementedError` when a label is asked for and missing, which is the
    codebase's "you are reading held-out labels" tripwire.
    """
    from . import labels as L  # local import avoids a module cycle

    try:
        signal, fs, header = load_record(record_path)
    except Exception:
        return None
    if train and signal.shape[-1] < cfg.min_seconds * fs:
        return None

    info = L.parse_patient_info(header, want_label=want_label)
    demographics = demographics_vector(info.age, info.sex, cfg)
    try:
        windowed, valid_len = preprocess_array(
            signal, fs, cfg, train=train, aug=aug, rng=rng
        )
    except ValueError:
        return None
    if valid_len <= 0:
        return None
    return Sample(
        signal=windowed,
        valid_len=valid_len,
        demographics=demographics,
        label_hard=info.label,
        label_soft=None if info.label is None else float(info.label),
        source=info.source,
        record_id=os.path.splitext(os.path.basename(record_path))[0],
    )


def preprocess(
    record_or_array: Any,
    fs: Optional[float] = None,
    cfg: Optional[PreprocessCfg] = None,
    *,
    train: bool = False,
    aug: Optional[AugCfg] = None,
    rng: Optional[np.random.Generator] = None,
    want_label: bool = False,
):
    """Dispatch helper: a record path goes to :func:`preprocess_record`, an array to
    :func:`preprocess_array`."""
    cfg = cfg or PreprocessCfg()
    if isinstance(record_or_array, (str, os.PathLike)):
        return preprocess_record(
            str(record_or_array), cfg, train=train, aug=aug, rng=rng, want_label=want_label
        )
    if fs is None:
        raise ValueError("fs is required when preprocessing a raw array")
    return preprocess_array(record_or_array, fs, cfg, train=train, aug=aug, rng=rng)
