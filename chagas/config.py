"""Typed configuration for the Chagas pipeline, with YAML load/dump.

Every default carries a comment naming where it comes from.  Defaults that are the
team's own carry "(team)"; everything else names a paper -- see ``PROVENANCE.md``.

The whole experiment is described by a single :class:`ExperimentCfg`.  ``experiments/
configs/*.yaml`` holds one file per rung of the ablation ladder (PLAN.md section 2); a
config is loaded with :meth:`ExperimentCfg.from_yaml` and round-trips through
:meth:`ExperimentCfg.to_yaml`.

No module-level side effects: paths are resolved lazily against :data:`REPO_ROOT`,
which is derived from ``__file__`` rather than from the process CWD.
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Type, TypeVar

import yaml

#: Repository root (the directory that holds ``team_code.py``).  Resolved from this
#: file rather than the CWD so that nothing depends on where python was started.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

#: Lead order the ECG-FM backbone was pretrained on -- see
#: ``fairseq-signals/scripts/preprocess/ecg/preprocess.py``.  (team)
ECG_FM_LEAD_ORDER: Tuple[str, ...] = (
    "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6",
)

#: Total downsampling factor of the ECG-FM conv frontend, ``[(256, 2, 2)] * 4``:
#: four kernel-2 stride-2 convolutions, i.e. ``floor(L/2)`` four times == ``L // 16``.
#: Verified against ``ECGTransformerModel._get_feat_extract_output_lengths`` by
#: ``tests/test_backbone.py``.
CONV_DOWNSAMPLE: int = 16


def resolve_path(p: Optional[str], base: Optional[Path] = None) -> Optional[Path]:
    """Resolve ``p`` against ``base`` (default :data:`REPO_ROOT`) unless it is absolute.

    Bare relative paths in the legacy code resolved against the process CWD, which
    happened to work only because Docker's ``WORKDIR`` is ``/challenge``.  Every path in
    this package goes through here instead.
    """
    if p is None:
        return None
    path = Path(os.path.expanduser(str(p)))
    if path.is_absolute():
        return path
    return (Path(base) if base is not None else REPO_ROOT) / path


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@dataclass
class PreprocessCfg:
    """The single preprocessing pipeline (PLAN.md section 3.5), used by train AND inference."""

    #: ECG-FM's pretraining rate (McKeen et al. 2025).  (team default, kept)
    sampling_rate: int = 500
    #: Model input length in seconds.  10 s == every PTB-XL record.  (team)
    window_seconds: float = 10.0
    #: Training-time minimum usable duration.  (team)
    min_seconds: float = 2.0
    #: ``"butterworth"`` -- vectorised ``sosfiltfilt`` band-pass + 50/60 Hz notches
    #: (Hong, Chou & Gonzalez, CinC 2025).  ``"neurokit"`` is the team's legacy
    #: per-lead ``nk.ecg_clean`` (kept for the ablation).  ``"none"`` disables cleaning.
    clean: str = "butterworth"
    bandpass_low_hz: float = 0.5      # Hong et al. 2025
    bandpass_high_hz: float = 45.0    # ladder row 5c (DECISIONS.md D-2026-09-28-1); Physiol. Meas. 2026 used 45, Hong et al. 2025 60
    bandpass_order: int = 2           # Hong et al. 2025
    #: Brazil's mains is 60 Hz; the legacy neurokit cleaner notches 50 Hz only.
    notch_hz: Tuple[float, ...] = (50.0, 60.0)
    notch_q: float = 30.0
    #: The organisers clip age > 89 to 90; mirror that (Reyna et al. 2025).
    age_clip_max: float = 90.0
    #: Age is fed as ``age / 100`` ("centuries") plus a missing indicator, so a missing
    #: age is no longer indistinguishable from the training mean (audit defect D16).
    age_scale: float = 100.0

    @property
    def window_size(self) -> int:
        """Model input length in samples."""
        return int(round(self.window_seconds * self.sampling_rate))

    def __post_init__(self) -> None:
        if self.clean not in ("butterworth", "neurokit", "none"):
            raise ValueError(f"clean must be butterworth|neurokit|none, got {self.clean!r}")
        if self.bandpass_high_hz >= self.sampling_rate / 2:
            raise ValueError("bandpass_high_hz must be below Nyquist")


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
@dataclass
class AugCfg:
    """Augmentation suite (PLAN.md section 3.4).

    Stage **R** runs on the raw signal at its native rate (it simulates acquisition);
    stage **S** runs after cleaning and before the per-record z-score.

    Evidence: the winner's augmentations gained +5% on the organisers' hidden
    validation set but only +2.7% on their own CV (Van Santvliet et al., CinC 2025
    section 3) -- i.e. this is an out-of-distribution measure, not an in-distribution one.
    """

    enabled: bool = True

    # -- stage R: acquisition-level, before rate conversion -------------------
    #: Resample to another rate and back.  This is exactly what the organisers did to
    #: the hidden test data (Reyna et al. 2025 section 2.2).
    p_resample_roundtrip: float = 0.5
    resample_rates: Tuple[int, ...] = (300, 400, 500, 600, 1000)
    #: "filters that are representative of different ECG devices" (ibid.).
    p_device_filter: float = 0.5
    device_hp_hz: Tuple[float, float] = (0.05, 0.67)
    device_lp_hz: Tuple[float, float] = (40.0, 150.0)
    device_order: Tuple[int, int] = (1, 2)

    # -- stage S: signal-level nuisance, after cleaning ------------------------
    #: Powerline interference, identical across leads.  Implementation is the team's
    #: ``augmentations.ECGAugmentations._add_powerline_interference``; the parameter
    #: ranges coincide with Van Santvliet et al. 2025 Table 1.
    p_powerline: float = 0.5
    powerline_freqs: Tuple[float, ...] = (50.0, 60.0)
    powerline_freq_jitter_hz: float = 0.2
    powerline_snr_db: Tuple[float, float] = (15.0, 30.0)
    powerline_harmonics: bool = True
    #: Baseline wander ``a * sin(2*pi*f*t)`` (Hong et al. 2025).  ``a`` is relative to
    #: the record's standard deviation, because this runs before the z-score.
    p_baseline_wander: float = 0.5
    baseline_wander_hz: Tuple[float, float] = (0.1, 0.5)
    baseline_wander_amp: float = 0.1
    #: Additive Gaussian noise, sigma relative to the record's standard deviation
    #: (Hong et al. 2025 use sigma = 0.01 on normalised signal).
    p_gaussian_noise: float = 0.5
    gaussian_sigma: float = 0.01
    #: Amplitude scaling.  The global factor is neutralised by the per-record z-score;
    #: the per-lead factor is not, which is why both are applied (Hong et al. 2025).
    p_amplitude_scale: float = 0.5
    amplitude_global: Tuple[float, float] = (0.8, 1.2)
    amplitude_per_lead: Tuple[float, float] = (0.9, 1.1)
    #: Linear time warp (Hong et al. 2025).
    p_time_warp: float = 0.5
    time_warp: Tuple[float, float] = (0.9, 1.1)
    #: Zero a random interior segment (Hong et al. 2025).
    p_random_mask: float = 0.5
    random_mask_max_s: float = 1.0
    #: Zero up to N seconds at each edge (Hong et al. 2025).
    p_edge_mask: float = 0.5
    edge_mask_max_s: float = 1.5
    #: Random crop.  The crop's random start position IS the temporal shift, so the
    #: valid region stays a prefix of the padded window and the padding mask stays
    #: prefix-shaped (Van Santvliet et al. 2025; team ``augmentations.py``).
    p_crop_shift: float = 1.0
    crop_seconds: Tuple[float, float] = (5.65, 10.0)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------
@dataclass
class LabelCfg:
    """Source-aware soft labels and auxiliary labels (PLAN.md section 3.2)."""

    #: Source-aware label smoothing (Van Santvliet et al. 2025; Physiol. Meas. 2026
    #: doi:10.1088/1361-6579/ae98e0; Hong et al. 2025).
    soft_labels: bool = True
    samitrop_value: float = 1.0       # serologically confirmed positives
    ptbxl_value: float = 0.0          # European cohort, assumed negative
    code15_pos_value: float = 0.8     # self-reported, noisy
    code15_neg_value: float = 0.2
    #: Multi-report consistency (Hong et al. 2025 eq. 1): for a CODE-15 patient with
    #: several exams, ``c = alpha * y + (1 - alpha) * n_pos / n_total``.
    multi_report: bool = True
    multi_report_alpha: float = 0.51
    #: ``"none"`` | ``"team_rbbb"`` (the team's ``dataloader._compute_soft_label``) |
    #: ``"hong_rbbb"`` (Hong et al. 2025's +0.15/+0.25 bump, capped at 0.49).
    rbbb_scheme: str = "none"
    #: Auxiliary multitask targets read from ``code15_exams.csv``.  (team idea; see
    #: the dev tree's ``train_multitask_fm.py``.)
    aux_columns: Tuple[str, ...] = ("RBBB", "1dAVb", "LBBB", "AF", "SB")
    #: Path to ``code15_exams.csv``; relative paths resolve against the repo root.
    code15_exams_csv: str = "code15_exams.csv"

    def __post_init__(self) -> None:
        if self.rbbb_scheme not in ("none", "team_rbbb", "hong_rbbb"):
            raise ValueError(f"unknown rbbb_scheme {self.rbbb_scheme!r}")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
@dataclass
class ModelCfg:
    """The classifier (PLAN.md section 3.6).  The team's core idea is unchanged."""

    #: ECG-FM backbone checkpoint; relative paths resolve against the repo root.
    #: The Dockerfile symlinks it there from ``/opt/ecgfm``.
    checkpoint: str = "mimic_iv_ecg_finetuned.pt"
    embed_dim: int = 768
    #: ``"static_softmax"`` is the team's learnable softmax over the 12 layers (default).
    #: ``"pma"`` is the input-dependent gate of Nguyen et al., arXiv:2509.00102.
    aggregation: str = "static_softmax"
    #: ``"film"`` (Perez et al., AAAI 2018) | ``"concat"`` (team, legacy) | ``"none"``
    #: (the age-shortcut ablation).
    demographics: str = "none"  # ladder row 5b (D-2026-09-26-3): no gain with a fine-tuned encoder, 9x less source cue
    film_hidden: int = 32
    hidden_dim: int = 256
    dropout: float = 0.1
    #: Build a real token-level padding mask from ``valid_len`` (audit defect D3).
    use_padding_mask: bool = True
    #: The checkpoint ships ``feature_grad_mult = 0.0``, so the conv frontend runs under
    #: ``no_grad`` even when the transformer is unfrozen.  Set True to override.
    train_conv_frontend: bool = False
    #: Optional auxiliary heads off the 256-d hidden layer (team; Caruana 1997;
    #: Jidling et al., PLoS NTD 2023).
    aux_heads: bool = False
    aux_weight: float = 0.25
    #: Optional domain-adversarial source head (Ganin et al., JMLR 2016).
    grl: bool = False
    grl_weight: float = 1.0
    grl_lambda_max: float = 0.1
    #: Activation checkpointing over the 12 transformer layers; the automatic fallback
    #: if the stage-2 memory smoke test exceeds the T4 budget.
    gradient_checkpointing: bool = False

    def __post_init__(self) -> None:
        if self.aggregation not in ("static_softmax", "pma"):
            raise ValueError(f"unknown aggregation {self.aggregation!r}")
        if self.demographics not in ("film", "concat", "none"):
            raise ValueError(f"unknown demographics {self.demographics!r}")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
@dataclass
class TrainCfg:
    """Two-stage fine-tuning, loss, sampling and budget (PLAN.md sections 3.1/3.2/3.7)."""

    seed: int = 42
    batch_size: int = 16                       # Van Santvliet et al. 2025
    accumulate_grad_batches: int = 4           # effective batch 64 (ibid.)
    num_workers: int = 10
    #: fp16 AMP.  The grader's T4 and our TitanRTX are both SM 7.5, which has fp16
    #: tensor cores but NO native bf16 -- asserted in :mod:`chagas.training`.
    precision: str = "16-mixed"

    # -- two-stage schedule ----------------------------------------------------
    two_stage: bool = True                     # Van Santvliet et al. 2025; McKeen et al. 2025
    stage1_lr: float = 2e-4                    # Van Santvliet et al. 2025
    stage1_max_steps: int = 2000               # Van Santvliet et al. 2025
    stage2_lr: float = 2e-5                    # Van Santvliet et al. 2025
    stage2_head_lr: float = 1e-4               # Van Santvliet et al. 2025
    #: Layer-wise LR decay: layer k gets ``base * llrd ** (n_layers - k)``
    #: (Howard & Ruder 2018 ULMFiT; Clark et al. 2020 ELECTRA).
    llrd: float = 0.85
    warmup_frac: float = 0.05
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    max_epochs: int = 16
    stage2_max_epochs: int = 16
    #: Stage-2 validation cadence in OPTIMISER steps; 0 = once per epoch (legacy).
    #: Every two-stage ladder run peaked at its first stage-2 validation and then
    #: drifted, so per-epoch checkpoints land past the peak once an epoch is long
    #: (4.6k steps on the full 292k split vs 1.4k on the 100k ladder subsample).
    #: A step cadence makes checkpoint timing independent of dataset size
    #: (DECISIONS.md D-2026-09-29-1/2).
    stage2_val_every_steps: int = 0
    #: Hard cap on stage-2 optimiser steps; 0 = bounded by epochs and the time budget.
    stage2_max_steps: int = 0

    # -- time budget -----------------------------------------------------------
    #: The grader allows 72 h on a T4; stop just under it (team ``TimeLimitCallback``).
    max_hours: float = 71.0
    stage1_budget_frac: float = 0.05
    stage2_budget_frac: float = 0.80

    # -- loss and sampling -----------------------------------------------------
    pos_weight: float = 5.0                    # Van Santvliet et al. 2025; Physiol. Meas. 2026
    #: Positive oversampling factor for the ``WeightedRandomSampler``.  5.0 is the
    #: winner's; 1.0 is the Physiol. Meas. 2026 ablation ("upsampling is harmful under
    #: label noise").  The legacy code used inverse-frequency (~50/50), far more
    #: aggressive than any top team.
    oversample: float = 5.0
    #: Margin-ranking loss on hard negatives (Hong et al. 2025 eq. 2-3).
    ranking_loss: bool = False
    ranking_weight: float = 2.5
    ranking_margin: float = 0.5
    ranking_top_frac: float = 0.2

    # -- validation, selection, threshold --------------------------------------
    val_fraction: float = 0.1
    #: Early stopping on the challenge score (Physiol. Meas. 2026 used patience 3).
    early_stop_patience: int = 3
    monitor: str = "val_challenge"
    #: Decision threshold = this quantile of the validation probabilities; the metric's
    #: own operating point is the top 5%, so 0.95 (audit defect D9).
    threshold_quantile: float = 0.95
    val_check_interval: float = 1.0
    limit_train_batches: float = 1.0

    #: Seed ensemble; each member gets an equal share of the remaining budget
    #: (Jidling et al. 2023 averaged 15 seeds; inference cost scales linearly, so <= 3).
    ensemble_members: int = 1

    def __post_init__(self) -> None:
        if "bf16" in str(self.precision):
            raise ValueError(
                "bf16 is not natively supported on SM 7.5 (T4 / TitanRTX); use '16-mixed'."
            )


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class DataCfg:
    """Where the records are and how they are grouped."""

    #: Training data folder.  ``train_model.py`` passes this in; the YAML value is only
    #: a default for offline experiments.
    data_folder: Optional[str] = None
    #: Cap on the number of training records (0 = all).  Used by the smoke configs.
    #: Keep every positive, thin negatives to this total (``training._subsample``).
    #: 100k reproduces the validated ladder regime; training on all ~290k negatives
    #: cost ~0.05 AUROC in two runs (DECISIONS.md D-2026-09-29-1/3). 0 = no cap.
    max_records: int = 100_000
    #: Drop CODE-15 records listed in ``code15_label_issues.csv`` (team artefact).
    exclude_label_issues: bool = False
    label_issues_csv: str = "code15_label_issues.csv"
    #: Threads used to read headers when building the split.
    scan_workers: int = 8


# ---------------------------------------------------------------------------
# Top level
# ---------------------------------------------------------------------------
@dataclass
class ExperimentCfg:
    """Everything needed to reproduce one rung of the ablation ladder."""

    name: str = "default"
    description: str = ""
    data: DataCfg = field(default_factory=DataCfg)
    preprocess: PreprocessCfg = field(default_factory=PreprocessCfg)
    aug: AugCfg = field(default_factory=AugCfg)
    labels: LabelCfg = field(default_factory=LabelCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)

    # -- serialisation ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Plain nested dict, tuples flattened to lists so YAML stays readable."""
        return _to_plain(self)

    def to_yaml(self, path: Optional[Any] = None) -> str:
        text = yaml.safe_dump(self.to_dict(), sort_keys=False, default_flow_style=False)
        if path is not None:
            Path(path).write_text(text)
        return text

    @classmethod
    def from_dict(cls, d: Optional[Mapping[str, Any]]) -> "ExperimentCfg":
        return _from_plain(cls, d or {})

    @classmethod
    def from_yaml(cls, path: Any) -> "ExperimentCfg":
        """Load a config file.  Unknown keys raise rather than being silently ignored."""
        with open(path, "r") as fh:
            payload = yaml.safe_load(fh)
        return cls.from_dict(payload)

    # -- resolved paths --------------------------------------------------------
    @property
    def checkpoint_path(self) -> Path:
        p = resolve_path(self.model.checkpoint)
        assert p is not None
        return p

    @property
    def code15_exams_path(self) -> Optional[Path]:
        return resolve_path(self.labels.code15_exams_csv)


# ---------------------------------------------------------------------------
# dataclass <-> plain-python helpers
# ---------------------------------------------------------------------------
_T = TypeVar("_T")


def _to_plain(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _from_plain(cls: Type[_T], payload: Mapping[str, Any]) -> _T:
    """Build ``cls`` from a nested mapping, recursing into dataclass fields.

    Unknown keys are an error: a typo in an ablation YAML must not silently leave the
    default in place and produce a run labelled as something it is not.
    """
    if not isinstance(payload, Mapping):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(payload).__name__}")
    known = {f.name: f for f in fields(cls)}  # type: ignore[arg-type]
    unknown = sorted(set(payload) - set(known))
    if unknown:
        raise ValueError(f"unknown config keys for {cls.__name__}: {unknown}")
    kwargs: Dict[str, Any] = {}
    for name, f in known.items():
        if name not in payload:
            continue
        value = payload[name]
        ftype = f.type
        # dataclasses store the *declared* type; resolve the nested dataclass by
        # looking at the field's default_factory instead of parsing the annotation.
        nested = _nested_dataclass(f)
        if nested is not None:
            kwargs[name] = _from_plain(nested, value)
        elif isinstance(value, list) and _is_tuple_field(ftype):
            kwargs[name] = tuple(value)
        else:
            kwargs[name] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def _nested_dataclass(f: "dataclasses.Field[Any]") -> Optional[type]:
    factory = f.default_factory  # type: ignore[attr-defined]
    if factory is not dataclasses.MISSING and is_dataclass(factory):
        return factory  # type: ignore[return-value]
    return None


def _is_tuple_field(ftype: Any) -> bool:
    return "Tuple" in str(ftype) or "tuple" in str(ftype)


def default_config() -> ExperimentCfg:
    """The shipped default: the final rung of the ladder (PLAN.md section 2, row 5).

    FiLM demographics, source-aware soft labels, the augmentation suite and two-stage
    fine-tuning are ON; auxiliary heads, ranking loss, the domain-adversarial head, the
    PMA gate and the ensemble are OFF until an ablation earns them.
    """
    return ExperimentCfg(
        name="default",
        description="Ladder row 5: defect fixes + soft labels + augmentation + FiLM + two-stage FT.",
    )


def load_config(path: Optional[Any] = None) -> ExperimentCfg:
    """Load a config from ``path``, or from ``$CHAGAS_CONFIG``, or fall back to defaults.

    ``train_model.py`` takes no config argument (it is a do-not-edit organiser script),
    so the environment variable is the only way to point the judged entry point at a
    different rung of the ladder.  Unset, it trains the shipped default.
    """
    if path is None:
        path = os.environ.get("CHAGAS_CONFIG")
    if not path:
        return default_config()
    return ExperimentCfg.from_yaml(path)
