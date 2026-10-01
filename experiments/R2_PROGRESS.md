# Round 2 progress (PLAN_R2.md) — running checklist

Updated at every milestone so the work can be resumed by reading this file.

## Done
- RESCORED.md (job 17682392, COMPLETED) force-added with its CSVs.
- Code for all §3 flags, eval tools (scoring/eval_report/aggregate/predict_batched/run_info/rescore_all),
  r2 configs + smoke_r2 config (commits 0461809, 4076061, e9fd2b1).
- Durations: SLURM 17682272 read all 366,185 headers; 0 mismatches vs signal_lengths.csv (committed CSV).
- Tests: 49 new (138 total). CPU suite 17682391: 137 passed, 1 failed (float32-vs-float64 comparison in my own
  segment_mean test; fixed in e9fd2b1). Full suite must be re-run on SLURM before launching the array.
- Acceptance (17682392): all targets reproduce except duration-only on view B (0.677; 0.681 is the all-CODE-15 value,
  D-2026-10-01-4).
- Docs: DECISIONS D-2026-10-01-1..4, PROVENANCE/METHODS/README round-2 sections, REPRODUCE.md.

## In progress
- Round-2 array **17682658** (9 tasks, %4) launched at commit 65fb3bf after: CPU suite 17682446 = 138 passed;
  smoke 17682393 passed; run_r2.sh end-to-end on testdata 17682505 passed. DO NOT edit chagas/ or experiments/*.py|sh
  while it runs (tasks import the worktree when they start).

## Todo
- When the array finishes: run the aggregate commands in experiments/REPRODUCE.md section 5 (A and C).

## Job ids
- 17682272 durations (COMPLETED); 17682391 pytest CPU (1 fail, fixed); 17682392 rescore; 17682393 r2 smoke (PASS); 17682446 pytest CPU (138 passed); 17682505 run_r2.sh runner smoke (PASS); 17682658 r2 array
