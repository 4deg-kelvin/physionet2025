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
from typing import Callable, List, Tuple

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
    ("finalB_stepval", final_b),
]


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


def main() -> int:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in ROWS + [("smoke", smoke)]:
        cfg = _base()
        fn(cfg)
        path = CONFIG_DIR / f"{name}.yaml"
        header = f"# {cfg.description}\n# Generated by experiments/make_configs.py -- edit that, not this.\n"
        path.write_text(header + cfg.to_yaml())
        # Round-trip check: a YAML that cannot be loaded back is worse than useless.
        assert ExperimentCfg.from_yaml(path).to_dict() == cfg.to_dict(), name
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
