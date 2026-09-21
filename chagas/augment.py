"""ECG augmentation suite (PLAN.md section 3.4).

Every augmentation is a **pure function** ``(x, fs, rng, ...) -> x'`` over a
``float64`` array of shape ``(12, T)``.  Nothing is stateful, nothing reads the global
NumPy RNG, and every call is reproducible given a :class:`numpy.random.Generator`.

Two stages:

* **Stage R** runs on the raw signal at its *native* sampling rate, before rate
  conversion, and simulates acquisition.  ``resample_roundtrip`` and ``device_filter``
  reproduce exactly what the organisers did to the hidden test records: "we added small
  amounts of various forms of noise, applied filters that are representative of
  different ECG devices, and resampled the data to different sampling frequencies"
  (Reyna et al., *Detection of Chagas Disease from the ECG: the George B. Moody
  PhysioNet Challenge 2025*, CinC 2025 / arXiv:2510.02202, section 2.2).
* **Stage S** runs after cleaning and before the per-record z-score, and adds
  signal-level nuisance.

Provenance
----------
``powerline`` is the team's own implementation (``augmentations.py``, Andy Smithwick,
September 2025) restated as a pure NumPy function; its parameter ranges coincide with
Van Santvliet, Nguyen, Vandenberk & De Vos, *Detecting Chagas Disease Using a Vision
Transformer-based ECG Foundation Model*, CinC 2025, Table 1 -- both are credited.
``crop_shift`` likewise mirrors the team's ``_random_crop_and_shift``.
``baseline_wander``, ``gaussian_noise``, ``amplitude_scale``, ``time_warp``,
``random_mask`` and ``edge_mask`` follow Hong, Chou & Gonzalez, *Managing Label
Uncertainty in the Detection of Chagas Disease from the ECG*, CinC 2025.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Optional, Sequence, Tuple

import numpy as np
from scipy.signal import butter, resample_poly, sosfiltfilt

from .config import AugCfg

__all__ = [
    "resample_roundtrip",
    "device_filter",
    "powerline",
    "baseline_wander",
    "gaussian_noise",
    "amplitude_scale",
    "time_warp",
    "random_mask",
    "edge_mask",
    "crop_shift",
    "apply_stage_r",
    "apply_stage_s",
]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_2d(x: np.ndarray) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError(f"expected a (leads, samples) array, got shape {x.shape}")
    return np.ascontiguousarray(x, dtype=np.float64)


def _resample_to(x: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Anti-aliased polyphase rate conversion (``scipy.signal.resample_poly``).

    ``resample_poly`` replaces the legacy FFT ``scipy.signal.resample`` (audit defect
    D7), which rings at the edges of non-periodic signals and is slow for awkward ratios.
    """
    if fs_in == fs_out:
        return x
    ratio = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    up, down = ratio.numerator, ratio.denominator
    if up <= 0 or down <= 0:
        return x
    return resample_poly(x, up, down, axis=-1)


def _fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """Truncate or edge-pad ``x`` along the last axis to exactly ``n`` samples."""
    t = x.shape[-1]
    if t == n:
        return x
    if t > n:
        return x[..., :n]
    pad = np.repeat(x[..., -1:], n - t, axis=-1)
    return np.concatenate([x, pad], axis=-1)


def _safe_sosfiltfilt(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``sosfiltfilt`` that degrades to a no-op on signals too short to pad.

    ``sosfiltfilt`` raises when ``padlen >= n``; a 2 s crop through a low-order filter is
    fine, but an aggressive crop of an already-short record is not.  Leaving the signal
    unfiltered is the right failure mode for an augmentation.
    """
    n = x.shape[-1]
    padlen = 3 * (2 * sos.shape[0] + 1)
    if n <= padlen + 1:
        return x
    return np.ascontiguousarray(sosfiltfilt(sos, x, axis=-1))


# ---------------------------------------------------------------------------
# Stage R -- acquisition
# ---------------------------------------------------------------------------
def resample_roundtrip(
    x: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    rates: Sequence[int] = (300, 400, 500, 600, 1000),
) -> np.ndarray:
    """Resample to a random rate and back, preserving the input length exactly.

    This is the organisers' own test-time perturbation (Reyna et al. 2025 section 2.2):
    the hidden cohorts were recorded at 300/500/600/1000 Hz and the records were
    additionally resampled.  Returning the *same* number of samples keeps this
    composable with the rest of the pipeline.
    """
    x = _as_2d(x)
    choices = [r for r in rates if float(r) != float(fs)]
    if not choices:
        return x
    target = float(choices[int(rng.integers(len(choices)))])
    n = x.shape[-1]
    y = _resample_to(x, fs, target)
    if y.shape[-1] < 2:
        return x
    y = _resample_to(y, target, fs)
    return _fit_length(y, n)


def device_filter(
    x: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    hp_hz: Tuple[float, float] = (0.05, 0.67),
    lp_hz: Tuple[float, float] = (40.0, 150.0),
    order: Tuple[int, int] = (1, 2),
) -> np.ndarray:
    """Random Butterworth high-pass and/or low-pass simulating an ECG device's filters.

    "filters that are representative of different ECG devices" (Reyna et al. 2025
    section 2.2).  The high-pass range spans the AHA diagnostic 0.05 Hz and the
    monitoring 0.67 Hz limits; the low-pass range spans 40-150 Hz.
    """
    x = _as_2d(x)
    nyq = fs / 2.0
    n_ord = int(rng.integers(order[0], order[1] + 1))
    do_hp = bool(rng.random() < 0.5)
    do_lp = bool(rng.random() < 0.5)
    if not do_hp and not do_lp:
        do_lp = True
    if do_hp:
        f_hp = float(rng.uniform(*hp_hz))
        if 0 < f_hp < nyq:
            sos = butter(n_ord, f_hp / nyq, btype="highpass", output="sos")
            x = _safe_sosfiltfilt(sos, x)
    if do_lp:
        f_lp = float(rng.uniform(*lp_hz))
        f_lp = min(f_lp, 0.95 * nyq)
        if 0 < f_lp < nyq:
            sos = butter(n_ord, f_lp / nyq, btype="lowpass", output="sos")
            x = _safe_sosfiltfilt(sos, x)
    return x


# ---------------------------------------------------------------------------
# Stage S -- signal-level nuisance
# ---------------------------------------------------------------------------
def powerline(
    x: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    freqs: Sequence[float] = (50.0, 60.0),
    freq_jitter_hz: float = 0.2,
    snr_db: Tuple[float, float] = (15.0, 30.0),
    harmonics: bool = True,
) -> np.ndarray:
    """Add mains interference at 50 or 60 Hz, identical across all 12 leads.

    Team implementation (``augmentations.ECGAugmentations._add_powerline_interference``)
    restated as a pure function; parameters coincide with Van Santvliet et al. 2025
    Table 1.  Harmonics are added at 1/2 and 1/3 of the fundamental's amplitude.
    Interference is common-mode, hence identical across leads.
    """
    x = _as_2d(x)
    n = x.shape[-1]
    power = float(np.mean(x ** 2))
    if not np.isfinite(power) or power <= 0.0:
        return x
    f0 = float(freqs[int(rng.integers(len(freqs)))]) + float(
        rng.uniform(-freq_jitter_hz, freq_jitter_hz)
    )
    snr = float(rng.uniform(*snr_db))
    amp = float(np.sqrt(power / (10.0 ** (snr / 10.0))))
    t = np.arange(n, dtype=np.float64) / float(fs)
    noise = amp * np.sin(2.0 * np.pi * f0 * t + rng.uniform(0.0, 2.0 * np.pi))
    if harmonics:
        for k, scale in ((2.0, 0.5), (3.0, 1.0 / 3.0)):
            if k * f0 < fs / 2.0:
                noise = noise + amp * scale * np.sin(
                    2.0 * np.pi * k * f0 * t + rng.uniform(0.0, 2.0 * np.pi)
                )
    return x + noise[None, :]


def baseline_wander(
    x: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    freq_hz: Tuple[float, float] = (0.1, 0.5),
    amplitude: float = 0.1,
) -> np.ndarray:
    """Add ``a * sin(2*pi*f*t)`` baseline wander, identical across leads.

    Hong et al. 2025.  ``a = amplitude * std(x)``: this runs before the per-record
    z-score, so an absolute amplitude would mean something different for every source.
    """
    x = _as_2d(x)
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd <= 0.0:
        return x
    f = float(rng.uniform(*freq_hz))
    t = np.arange(x.shape[-1], dtype=np.float64) / float(fs)
    wander = amplitude * sd * np.sin(2.0 * np.pi * f * t + rng.uniform(0.0, 2.0 * np.pi))
    return x + wander[None, :]


def gaussian_noise(
    x: np.ndarray, rng: np.random.Generator, sigma: float = 0.01
) -> np.ndarray:
    """Add white Gaussian noise with ``sigma`` relative to the record's std (Hong et al. 2025)."""
    x = _as_2d(x)
    sd = float(np.std(x))
    if not np.isfinite(sd) or sd <= 0.0:
        return x
    return x + rng.normal(0.0, sigma * sd, size=x.shape)


def amplitude_scale(
    x: np.ndarray,
    rng: np.random.Generator,
    global_range: Tuple[float, float] = (0.8, 1.2),
    per_lead_range: Tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    """Global and per-lead amplitude scaling (Hong et al. 2025).

    The global factor is neutralised by the downstream per-record z-score; the per-lead
    factor is not, and is the one that actually regularises.  Both are applied so that
    the augmentation matches the cited recipe.
    """
    x = _as_2d(x)
    g = float(rng.uniform(*global_range))
    per_lead = rng.uniform(per_lead_range[0], per_lead_range[1], size=(x.shape[0], 1))
    return x * g * per_lead


def time_warp(
    x: np.ndarray,
    rng: np.random.Generator,
    factor_range: Tuple[float, float] = (0.9, 1.1),
) -> np.ndarray:
    """Linear time warp: resample the record by a random factor (Hong et al. 2025).

    Changes the length; ``crop_shift`` and the windowing step restore the model's fixed
    input size afterwards.
    """
    x = _as_2d(x)
    factor = float(rng.uniform(*factor_range))
    n_out = max(2, int(round(x.shape[-1] * factor)))
    ratio = Fraction(n_out, x.shape[-1]).limit_denominator(1000)
    if ratio.numerator == ratio.denominator:
        return x
    return np.ascontiguousarray(
        resample_poly(x, ratio.numerator, ratio.denominator, axis=-1)
    )


def random_mask(
    x: np.ndarray, fs: float, rng: np.random.Generator, max_seconds: float = 1.0
) -> np.ndarray:
    """Zero a random interior segment of at most ``max_seconds`` (Hong et al. 2025)."""
    x = _as_2d(x).copy()
    n = x.shape[-1]
    width = int(rng.integers(1, max(2, int(max_seconds * fs)) + 1))
    width = min(width, n)
    start = int(rng.integers(0, n - width + 1))
    x[:, start:start + width] = 0.0
    return x


def edge_mask(
    x: np.ndarray, fs: float, rng: np.random.Generator, max_seconds: float = 1.5
) -> np.ndarray:
    """Zero 0-``max_seconds`` at the start and, independently, at the end (Hong et al. 2025).

    Real records often begin or end mid-calibration; this also teaches the model not to
    read meaning into the flat region that zero-padding creates.
    """
    x = _as_2d(x).copy()
    n = x.shape[-1]
    cap = max(0, min(n // 3, int(max_seconds * fs)))
    if cap <= 0:
        return x
    head = int(rng.integers(0, cap + 1))
    tail = int(rng.integers(0, cap + 1))
    if head:
        x[:, :head] = 0.0
    if tail:
        x[:, n - tail:] = 0.0
    return x


def crop_shift(
    x: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    crop_seconds: Tuple[float, float] = (5.65, 10.0),
    window_seconds: float = 10.0,
) -> np.ndarray:
    """Crop a random contiguous segment of 5.65-10 s (Van Santvliet et al. 2025; team).

    The crop's random *start* position is the temporal shift.  The team's
    ``augmentations._random_crop_and_shift`` instead pads zeros at the front, which
    would put valid samples behind padding; the token-level padding mask built in
    :mod:`chagas.preprocessing` assumes the valid region is a **prefix** of the padded
    window, and fairseq-signals' own ``_get_feat_extract_output_lengths`` mask is
    prefix-shaped too.  Randomising the start achieves the same augmentation while
    keeping that invariant.  (Trailing/leading zeroing inside the valid region is still
    available through :func:`edge_mask`.)
    """
    x = _as_2d(x)
    n = x.shape[-1]
    lo = max(1, int(round(crop_seconds[0] * fs)))
    hi = max(lo, int(round(min(crop_seconds[1], window_seconds) * fs)))
    width = int(rng.integers(lo, hi + 1))
    width = min(width, n)
    start = int(rng.integers(0, n - width + 1))
    return x[:, start:start + width]


# ---------------------------------------------------------------------------
# stage drivers
# ---------------------------------------------------------------------------
def apply_stage_r(
    x: np.ndarray, fs: float, rng: np.random.Generator, cfg: Optional[AugCfg]
) -> np.ndarray:
    """Acquisition-level augmentations, applied at the record's native rate."""
    if cfg is None or not cfg.enabled:
        return _as_2d(x)
    x = _as_2d(x)
    if rng.random() < cfg.p_resample_roundtrip:
        x = resample_roundtrip(x, fs, rng, cfg.resample_rates)
    if rng.random() < cfg.p_device_filter:
        x = device_filter(x, fs, rng, cfg.device_hp_hz, cfg.device_lp_hz, cfg.device_order)
    return x


def apply_stage_s(
    x: np.ndarray,
    fs: float,
    rng: np.random.Generator,
    cfg: Optional[AugCfg],
    window_seconds: float = 10.0,
) -> np.ndarray:
    """Signal-level augmentations, applied after cleaning and before the z-score.

    ``crop_shift`` runs last so the returned array is at most ``window_seconds`` long;
    the caller z-scores it and zero-pads it to the model's fixed input size.
    """
    if cfg is None or not cfg.enabled:
        return _as_2d(x)
    x = _as_2d(x)
    if rng.random() < cfg.p_powerline:
        x = powerline(
            x, fs, rng, cfg.powerline_freqs, cfg.powerline_freq_jitter_hz,
            cfg.powerline_snr_db, cfg.powerline_harmonics,
        )
    if rng.random() < cfg.p_baseline_wander:
        x = baseline_wander(x, fs, rng, cfg.baseline_wander_hz, cfg.baseline_wander_amp)
    if rng.random() < cfg.p_gaussian_noise:
        x = gaussian_noise(x, rng, cfg.gaussian_sigma)
    if rng.random() < cfg.p_amplitude_scale:
        x = amplitude_scale(x, rng, cfg.amplitude_global, cfg.amplitude_per_lead)
    if rng.random() < cfg.p_time_warp:
        x = time_warp(x, rng, cfg.time_warp)
    if rng.random() < cfg.p_random_mask:
        x = random_mask(x, fs, rng, cfg.random_mask_max_s)
    if rng.random() < cfg.p_edge_mask:
        x = edge_mask(x, fs, rng, cfg.edge_mask_max_s)
    if rng.random() < cfg.p_crop_shift:
        x = crop_shift(x, fs, rng, cfg.crop_seconds, window_seconds)
    return x
