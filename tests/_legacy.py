"""The round-1/2 shipped configuration, for tests written against it.

The ``chagas.config`` defaults became the round-3 entry (final_r3, D-2026-10-16-1):
5 s multi-window input, ranking loss, fixed validation protocol, S7 monitor, three
ensemble members, prevalence cap.  Tests that exercise single-window preprocessing, the
legacy split order or the old monitor build their config here instead; the pins are the
same seven fields ``experiments/make_configs._base`` pins for the historical configs.
"""

from chagas.config import ExperimentCfg, default_config


def legacy_config() -> ExperimentCfg:
    cfg = default_config()
    cfg.preprocess.window_seconds = 10.0
    cfg.preprocess.window_mode = "single"
    cfg.train.ranking_loss = False
    cfg.train.val_protocol = "legacy"
    cfg.train.monitor = "val_challenge"
    cfg.train.ensemble_members = 1
    cfg.data.cap_mode = "count"
    return cfg
