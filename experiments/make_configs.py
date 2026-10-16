#!/usr/bin/env python
"""Generate ``experiments/configs/row*.yaml`` -- one file per rung of PLAN.md section 2.

Each row adds to the previous one, like Hong et al. 2025 Table 2.  Regenerating is
idempotent: run ``python experiments/make_configs.py`` after changing a default and the
YAMLs are rewritten from :class:`chagas.config.ExperimentCfg`, so they can never drift
out of sync with the dataclasses.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chagas.config import ExperimentCfg  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parent / "configs"

#: Inverse-frequency positive weighting on the 100k subsample (6,528 positives /
#: 93,472 negatives) -- what the legacy ``fm.train_model`` computed, i.e. ~50/50 batches.
LEGACY_OVERSAMPLE = 93472.0 / 6528.0


def _base() -> ExperimentCfg:
    """Everything OFF: the starting point that row 0 turns into the a72c51c baseline."""
    cfg = ExperimentCfg()
    cfg.train.num_workers = 10
    cfg.train.max_epochs = 16
    cfg.train.stage2_max_epochs = 16
    cfg.train.max_hours = 20.0
    # The shipped defaults moved to the round-3 entry (final_r3, D-2026-10-16-1); the
    # ladder and rounds 1-3 are defined against the round-1 values, pinned here so every
    # committed config stays byte-identical.
    cfg.preprocess.window_seconds = 10.0
    cfg.preprocess.window_mode = "single"
    cfg.train.ranking_loss = False
    cfg.train.val_protocol = "legacy"
    cfg.train.monitor = "val_challenge"
    cfg.train.ensemble_members = 1
    cfg.data.cap_mode = "count"
    return cfg


def row0(cfg: ExperimentCfg) -> None:
    """Baseline-equivalent to commit a72c51c, rerun under the new eval harness.

    Faithful in: per-lead ``nk.ecg_clean``, no padding mask, hard labels, plain BCE with
    inverse-frequency (~50/50) sampling, demographics concatenated, frozen encoder for
    16 epochs, no augmentation.

    Two things cannot be reproduced through the config and are noted in
    ``experiments/README.md``: the legacy FFT ``scipy.signal.resample`` (replaced
    unconditionally by ``resample_poly`` -- defect D7 is not toggleable), and "train on
    every record and ship epoch 16" (this harness always holds out a patient-grouped
    10% and selects on it).  The authoritative a72c51c numbers remain the recorded
    held-out run 16878337: Test A 0.434 / Test B 0.440 / AUROC 0.858.
    """
    cfg.name = "row0_baseline"
    cfg.description = "PLAN row 0: a72c51c-equivalent baseline under the new eval harness."
    # Pinned: the ladder ran with these; shipped defaults may move (D-2026-09-28-1).
    cfg.preprocess.bandpass_high_hz = 100.0
    cfg.data.max_records = 0  # pinned: ladder rows ran uncapped (their folders were pre-thinned)
    cfg.preprocess.clean = "neurokit"
    cfg.model.use_padding_mask = False
    cfg.model.demographics = "concat"
    cfg.labels.soft_labels = False
    cfg.labels.multi_report = False
    cfg.aug.enabled = False
    cfg.train.two_stage = False
    cfg.train.stage1_max_steps = 0
    cfg.train.stage1_lr = 1e-4
    cfg.train.pos_weight = 1.0
    cfg.train.oversample = LEGACY_OVERSAMPLE
    cfg.train.accumulate_grad_batches = 1
    cfg.train.early_stop_patience = 0


def row1(cfg: ExperimentCfg) -> None:
    """+ the defect fixes: D1 (split/selection/seed), D3 (padding mask), D7/D8
    (resample_poly and the vectorised 0.5-100 Hz band-pass with 50 **and** 60 Hz
    notches), D9 (threshold from the 95th percentile), D10 (metrics on probabilities and
    hard labels)."""
    row0(cfg)
    cfg.name = "row1_defect_fixes"
    cfg.description = "PLAN row 1: + D1/D3/D7/D8/D9/D10 defect fixes, val split, selection, seed."
    cfg.preprocess.clean = "butterworth"
    cfg.model.use_padding_mask = True
    cfg.train.early_stop_patience = 3


def row1b(cfg: ExperimentCfg) -> None:
    """Row 1 with the legacy neurokit cleaner (DECISIONS.md D-2026-09-26-1).

    Isolates the cleaner from the other two row-1 changes (padding mask, early stopping).
    """
    row1(cfg)
    cfg.preprocess.clean = "neurokit"
    cfg.name = "row1b_neurokit"
    cfg.description = "Row 1 variant: padding mask + early stop, but the legacy neurokit cleaner."


def row2(cfg: ExperimentCfg) -> None:
    """+ source-aware soft labels and ``pos_weight = 5`` with 5x oversampling."""
    row1(cfg)
    cfg.name = "row2_soft_labels"
    cfg.description = "PLAN row 2: + source-aware soft labels, w_pos=5, 5x oversampling."
    cfg.labels.soft_labels = True
    cfg.labels.multi_report = True
    cfg.train.pos_weight = 5.0
    cfg.train.oversample = 5.0
    cfg.train.accumulate_grad_batches = 4


def row2b(cfg: ExperimentCfg) -> None:
    """The disagreement between the two top papers: no oversampling at all.

    Physiol. Meas. 2026 found upsampling harmful under label noise; Van Santvliet et al.
    2025 used 5x.  Run both.
    """
    row2(cfg)
    cfg.name = "row2b_no_oversample"
    cfg.description = "PLAN row 2 variant: soft labels + w_pos=5 but NO oversampling."
    cfg.train.oversample = 1.0


def row3(cfg: ExperimentCfg) -> None:
    row2(cfg)
    cfg.name = "row3_augmentation"
    cfg.description = "PLAN row 3: + the stage-R/stage-S augmentation suite."
    cfg.aug.enabled = True


def row4(cfg: ExperimentCfg) -> None:
    row3(cfg)
    cfg.name = "row4_film"
    cfg.description = "PLAN row 4: + FiLM demographics with the age-missing indicator."
    cfg.model.demographics = "film"


def row5(cfg: ExperimentCfg) -> None:
    """The shipped default: two-stage fine-tuning with layer-wise LR decay."""
    row4(cfg)
    cfg.name = "row5_two_stage_ft"
    cfg.description = "PLAN row 5: + two-stage fine-tuning with LLRD 0.85 (the default config)."
    cfg.train.two_stage = True
    cfg.train.stage1_max_steps = 2000
    cfg.train.stage1_lr = 2e-4
    cfg.train.stage2_lr = 2e-5
    cfg.train.stage2_head_lr = 1e-4
    cfg.train.llrd = 0.85
    cfg.train.max_hours = 40.0


def row5b(cfg: ExperimentCfg) -> None:
    """Row 5 with demographics removed (DECISIONS.md D-2026-09-25-3).

    Row 3 had erased the PTB-XL fingerprint (PTB-XL mean prob 0.078 -> 0.316); row 5 brought
    back a stronger one (0.008). PTB-XL is European with a different age/sex mix, so FiLM on
    age/sex alone could identify it. This variant asks whether the source cue survives
    without any demographic input.
    """
    row5(cfg)
    cfg.model.demographics = "none"
    cfg.name = "row5b_no_demographics"
    cfg.description = "Row 5 variant: identical, but demographics=none (age-shortcut / source-cue ablation)."


def row5c(cfg: ExperimentCfg) -> None:
    """Row 5 with the band-pass upper cutoff lowered to 45 Hz (DECISIONS.md D-2026-09-26-1).

    Same fast vectorised cleaner; the 45 Hz cutoff is the one used by the
    reliability-aware Chagas paper (Physiol. Meas. 2026; Hong et al. 2025 used 60 Hz).
    """
    row5b(cfg)  # demographics=none, D-2026-09-26-3
    cfg.preprocess.bandpass_high_hz = 45.0
    cfg.name = "row5c_bandpass45"
    cfg.description = "Row 5 variant: butterworth 0.5-45 Hz instead of 0.5-100 Hz."


def row5d(cfg: ExperimentCfg) -> None:
    """Row 5 with the legacy neurokit cleaner (DECISIONS.md D-2026-09-26-1)."""
    row5b(cfg)  # demographics=none, D-2026-09-26-3
    cfg.preprocess.clean = "neurokit"
    cfg.name = "row5d_neurokit"
    cfg.description = "Row 5 variant: legacy per-lead neurokit cleaner (12x slower at inference)."


def row6(cfg: ExperimentCfg) -> None:
    row5(cfg)
    cfg.name = "row6_aux_heads"
    cfg.description = "PLAN row 6: + auxiliary RBBB/1dAVb/LBBB/AF/SB heads (optional)."
    cfg.model.aux_heads = True


def row7(cfg: ExperimentCfg) -> None:
    row5(cfg)
    cfg.name = "row7_ranking_loss"
    cfg.description = "PLAN row 7: + margin-ranking loss on hard negatives (optional)."
    cfg.train.ranking_loss = True


def row8(cfg: ExperimentCfg) -> None:
    row5(cfg)
    cfg.name = "row8_domain_adversarial"
    cfg.description = "PLAN row 8: + domain-adversarial source head (judged on C and B)."
    cfg.model.grl = True


def row9(cfg: ExperimentCfg) -> None:
    row5(cfg)
    cfg.name = "row9_ensemble"
    cfg.description = "PLAN row 9: + a 2-seed ensemble within the time budget."
    cfg.train.ensemble_members = 2


def row_final(cfg: ExperimentCfg) -> None:
    """The final configuration, for the full 292,885-record training split."""
    row5c(cfg)  # winner of the cleaner ablation, D-2026-09-28-1
    cfg.name = "final_full"
    cfg.description = "Final config on the full training split; 71 h budget, 2 seeds via two runs."
    cfg.train.max_hours = 71.0


def final_a(cfg: ExperimentCfg) -> None:
    """Final candidate A: the 5c config, trained on all positives + ~93k negatives.

    Reproduces the validated 100k ladder regime inside ``train_model`` so the grader's
    full-data retrain lands where the ladder measured (DECISIONS.md D-2026-09-29-1/2).
    """
    row_final(cfg)
    cfg.name = "finalA_negcap"
    cfg.description = "Final A: 5c config with data.max_records=100000 (all positives, capped negatives)."
    cfg.data.max_records = 100_000


def final_r3(cfg: ExperimentCfg) -> None:
    """The round-3 entry (D-2026-10-16-1): final A + the fixed validation protocol and
    prevalence cap (r3_base without the cluster path to ptbxl_database.csv, which the
    entry looks up in the data folder) + 5 s windows + the hard-negative ranking loss,
    trained as a 3-seed ensemble inside one ``train_model.py`` call.  Equals the
    ``chagas.config`` defaults (tests/test_config_r2.py)."""
    final_a(cfg)
    r3_base(cfg)
    cfg.data.ptbxl_database_csv = None
    r2_windows(cfg)
    r3_ranking(cfg)
    cfg.train.ensemble_members = 3
    cfg.name = "final_r3"
    cfg.description = ("Round-3 entry: final A + fixed val split, prevalence cap 0.065, S7 monitor, "
                       "5 s multi-window, hard-negative ranking loss, 3-seed ensemble.")


def final_b(cfg: ExperimentCfg) -> None:
    """Final candidate B: the 5c config on all data, stage 2 validated every 500 optimiser steps.

    Catches the early stage-2 peak regardless of epoch length (DECISIONS.md D-2026-09-29-2).
    """
    row_final(cfg)
    cfg.name = "finalB_stepval"
    cfg.description = "Final B: 5c config on all data; stage-2 validation every 500 steps, capped at 6000."
    cfg.train.stage2_val_every_steps = 500
    cfg.train.stage2_max_steps = 6000


ROWS: List[Tuple[str, Callable[[ExperimentCfg], None]]] = [
    ("row0_baseline", row0),
    ("row1_defect_fixes", row1),
    ("row1b_neurokit", row1b),
    ("row2_soft_labels", row2),
    ("row2b_no_oversample", row2b),
    ("row3_augmentation", row3),
    ("row4_film", row4),
    ("row5_two_stage_ft", row5),
    ("row5b_no_demographics", row5b),
    ("row5c_bandpass45", row5c),
    ("row5d_neurokit", row5d),
    ("row6_aux_heads", row6),
    ("row7_ranking_loss", row7),
    ("row8_domain_adversarial", row8),
    ("row9_ensemble", row9),
    ("final_full", row_final),
    ("finalA_negcap", final_a),
    ("final_r3", final_r3),
    ("finalB_stepval", final_b),
]


# ---------------------------------------------------------------------------
# Round 2 (PLAN_R2.md section 4): M1 / W1 / MW x seeds 42, 43, 44 on top of the shipped
# config (finalA_negcap == B0).  Every change is a config flag; the shipped default is
# untouched (DECISIONS.md D-2026-10-01-1).
# ---------------------------------------------------------------------------
R2_SEEDS: Tuple[int, ...] = (42, 43, 44)


def r2_matched(cfg: ExperimentCfg) -> None:
    """M1: + length-matched negative sampling (PLAN_R2.md 3.1; Idrissi et al. 2022)."""
    cfg.data.match_negatives_on = "duration"


def r2_windows(cfg: ExperimentCfg) -> None:
    """W1: + fixed 5 s windows, random crop in training, mean of window logits at
    inference (PLAN_R2.md 3.2; McKeen et al. 2025; FjordNet, CinC 2025)."""
    cfg.preprocess.window_seconds = 5.0
    cfg.preprocess.window_mode = "multi"
    cfg.preprocess.max_windows = 4


def r2_wise(cfg: ExperimentCfg) -> None:
    """WiSE-FT over alpha in {0, 0.5, 1}, one model folder per alpha (PLAN_R2.md 3.3;
    Wortsman et al., CVPR 2022)."""
    cfg.train.wise_ft = "select"
    cfg.train.wise_ft_alphas = (0.0, 0.5, 1.0)
    cfg.train.wise_ft_save_all = True


R2_VARIANTS: List[Tuple[str, str, Sequence[Callable[[ExperimentCfg], None]]]] = [
    ("M1", "length-matched negatives", (r2_matched,)),
    ("W1", "5 s windows (random crop / multi-window logit mean)", (r2_windows,)),
    ("MW", "matched negatives + 5 s windows + WiSE-FT alpha in {0, 0.5, 1}",
     (r2_matched, r2_windows, r2_wise)),
]


def r2_configs() -> List[Tuple[str, ExperimentCfg]]:
    """The 9 round-2 runs: ``r2_<variant>_s<seed>``, each finalA_negcap + the variant."""
    out: List[Tuple[str, ExperimentCfg]] = []
    for variant, what, steps in R2_VARIANTS:
        for seed in R2_SEEDS:
            cfg = _base()
            final_a(cfg)
            for step in steps:
                step(cfg)
            cfg.train.seed = seed
            cfg.name = f"r2_{variant}_s{seed}"
            cfg.description = f"Round 2 {variant} (PLAN_R2.md section 4): shipped config + {what}; seed {seed}."
            out.append((cfg.name, cfg))
    return out


# ---------------------------------------------------------------------------
# Round 3: a fixed, natural-prevalence validation split (carved before the cap, PTB-XL
# grouped by patient), a prevalence-based negative cap, and checkpoint selection on the
# shortcut-blind S7 view, under which B0 / W1 / W1R / W1S are re-measured.  Every change
# is a config flag; the shipped default is untouched.
# ---------------------------------------------------------------------------
R3_SEEDS: Tuple[int, ...] = (42, 43, 44)
#: W1 gets two more seeds, so its seed variance is estimated from 5 runs.
R3_W1_EXTRA_SEEDS: Tuple[int, ...] = (45, 46)
#: PTB-XL's ``ecg_id -> patient_id`` table.  ``data_train_full`` holds only the WFDB
#: records, so the fixed split is pointed at the copy the training folder was built from.
PTBXL_DATABASE = (
    "/juice2/scr2/kelvinkn/other_work/edwards/physionet2025/training_data/"
    "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/ptbxl_database.csv"
)


def r3_base(cfg: ExperimentCfg) -> None:
    """Round-3 protocol on top of finalA_negcap: fixed validation split (seed 12345,
    stratified by source x label x length group, natural prevalence), all positives +
    negatives to 6.5% prevalence, checkpoints selected on validation S7 TPR@5%."""
    cfg.train.val_protocol = "fixed"
    cfg.train.monitor = "val_challenge_s7"
    cfg.data.cap_mode = "prevalence"
    cfg.data.cap_prevalence = 0.065
    cfg.data.ptbxl_database_csv = PTBXL_DATABASE


def r3_ranking(cfg: ExperimentCfg) -> None:
    """+ margin-ranking loss on hard negatives with the existing defaults (Hong, Chou &
    Gonzalez, CinC 2025, eq. 2-3; ladder row 7)."""
    cfg.train.ranking_loss = True
    cfg.train.ranking_weight = 2.5
    cfg.train.ranking_margin = 0.5
    cfg.train.ranking_top_frac = 0.2


def r3_no_ptbxl(cfg: ExperimentCfg) -> None:
    """Drop PTB-XL (all negative, all 10 s, European) from training and validation, so
    no all-negative source can be told apart by its recording format or population."""
    cfg.data.exclude_sources = ("ptbxl",)


R3_VARIANTS: List[Tuple[str, str, Sequence[Callable[[ExperimentCfg], None]], Tuple[int, ...]]] = [
    ("B0", "round-3 protocol only", (), R3_SEEDS),
    ("W1", "5 s windows", (r2_windows,), R3_SEEDS + R3_W1_EXTRA_SEEDS),
    ("W1R", "5 s windows + hard-negative ranking loss", (r2_windows, r3_ranking), R3_SEEDS),
    ("W1S", "5 s windows, PTB-XL excluded", (r2_windows, r3_no_ptbxl), R3_SEEDS),
]


def r3_configs() -> List[Tuple[str, ExperimentCfg]]:
    """The 14 round-3 runs: ``r3_<variant>_s<seed>``, each finalA_negcap + r3_base + the variant."""
    out: List[Tuple[str, ExperimentCfg]] = []
    for variant, what, steps, seeds in R3_VARIANTS:
        for seed in seeds:
            cfg = _base()
            final_a(cfg)
            r3_base(cfg)
            for step in steps:
                step(cfg)
            cfg.train.seed = seed
            cfg.name = f"r3_{variant}_s{seed}"
            cfg.description = (
                f"Round 3 {variant}: shipped config + fixed val split, prevalence cap, S7 monitor; "
                f"{what}; seed {seed}."
            )
            out.append((cfg.name, cfg))
    return out


def smoke(cfg: ExperimentCfg) -> None:
    """Two very short epochs on the 400-record smoke set (PLAN section 5 item 12)."""
    row5(cfg)
    cfg.name = "smoke"
    cfg.description = "End-to-end smoke: 2 short epochs, tiny budget, everything wired up."
    cfg.train.max_epochs = 2
    cfg.train.stage2_max_epochs = 1
    cfg.train.stage1_max_steps = 20
    cfg.train.max_hours = 0.5
    cfg.train.batch_size = 8
    cfg.train.accumulate_grad_batches = 1
    cfg.train.num_workers = 4
    cfg.train.val_fraction = 0.2
    cfg.train.early_stop_patience = 0
    cfg.train.limit_train_batches = 20


def smoke_r2(cfg: ExperimentCfg) -> None:
    """The round-2 contract smoke: every new flag on, tiny budget, 400-record set.

    ``max_records = 300`` (106 positives in the smoke set) so the length-matched
    negative cap actually runs; ``wise_ft_save_all`` so the per-alpha folders exist.
    """
    smoke(cfg)
    r2_matched(cfg)
    r2_windows(cfg)
    r2_wise(cfg)
    cfg.data.max_records = 300
    cfg.name = "smoke_r2"
    cfg.description = "Round-2 smoke: matched negatives + 5 s multi-window + WiSE-FT select, tiny budget."


def r3_smoke(cfg: ExperimentCfg) -> None:
    """The round-3 contract smoke: r3_base + 5 s windows, tiny budget, 400-record set.

    ``cap_prevalence = 0.4`` so the prevalence cap actually thins the smoke set's
    negatives (at 0.065 it would keep all of them).
    """
    smoke(cfg)
    r3_base(cfg)
    r2_windows(cfg)
    cfg.data.cap_prevalence = 0.4
    cfg.name = "r3_smoke"
    cfg.description = "Round-3 smoke: fixed val split + prevalence cap + S7 monitor + 5 s windows, tiny budget."


def all_configs() -> List[Tuple[str, ExperimentCfg]]:
    """``(relative path without .yaml, config)`` for every generated file."""
    out: List[Tuple[str, ExperimentCfg]] = []
    for name, fn in ROWS + [("smoke", smoke), ("smoke_r2", smoke_r2), ("r3_smoke", r3_smoke)]:
        cfg = _base()
        fn(cfg)
        out.append((name, cfg))
    out.extend((f"r2/{name}", cfg) for name, cfg in r2_configs())
    out.extend((f"r3/{name}", cfg) for name, cfg in r3_configs())
    return out


def render(cfg: ExperimentCfg) -> str:
    """The exact text written for ``cfg``: a two-line header plus the YAML."""
    header = f"# {cfg.description}\n# Generated by experiments/make_configs.py -- edit that, not this.\n"
    return header + cfg.to_yaml()


def main(out_dir: Optional[Path] = None) -> int:
    root = Path(out_dir) if out_dir is not None else CONFIG_DIR
    for name, cfg in all_configs():
        path = root / f"{name}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render(cfg))
        # Round-trip check: a YAML that cannot be loaded back is worse than useless.
        assert ExperimentCfg.from_yaml(path).to_dict() == cfg.to_dict(), name
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
