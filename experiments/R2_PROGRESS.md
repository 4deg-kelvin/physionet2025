# Round 2 progress (PLAN_R2.md) — running checklist

Updated at every milestone so the work can be resumed by reading this file.

## Done
- (none committed yet)

## In progress
- Code: config flags (`data.match_negatives_on`, `preprocess.window_mode/max_windows`, `train.wise_ft*`),
  `chagas/lengths.py`, augment `crop` flag.

## Todo
1. §2 eval: `experiments/build_durations.py` + SLURM job -> `experiments/data/record_durations.csv`;
   `experiments/scoring.py`, `eval_report.py` views S7/S10/S10+, duration-only, length-cue gap, CIs;
   `experiments/aggregate.py`; `experiments/predict_batched.py`; `RESCORED.md`; acceptance numbers.
2. §3.1 matched negatives; §3.2 5 s windows; §3.3 WiSE-FT; §3.4 ensemble checks.
3. §6 logging (global step, within-length val metrics, per-alpha), `run_info.json`, `REPRODUCE.md`.
4. §7 tests; full suite on SLURM.
5. Configs `experiments/configs/r2/` + `experiments/batch_r2.sh`.
6. Smoke on SLURM (train -> run_model -> evaluate on testdata; predict_batched equivalence; 5 s memory).
7. Launch §4 array.
8. METHODS.md, PROVENANCE.md, README round-2 sections.

## Job ids
- (none yet)
