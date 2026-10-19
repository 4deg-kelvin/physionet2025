# Round 3 progress (PLAN_R3.md) — running checklist

Updated at every milestone so the work can be resumed by reading this file.

## Done
- PLAN_R3.md written and the decision rule fixed before any run (D-2026-10-15-1..4).

## Done (cont.)
- Code for §2 (fixed validation protocol, per-view validation metrics, prevalence cap, window stride), r3 configs,
  run_r3.sh, batch_r3*.sh, aggregate `--rule r3`, tests, split audit — implemented by an Opus agent (reviewed). CPU tests 17885765: 173 passed. Split audit 17885742: PASS.

## In progress
- Nothing. The entry is frozen (tag `r3-submission` == `main`); the organisers' own `docker build` is the only unexercised step.

## Done (round 3 → submission)
- Array 17885830 (14/14) and tables 17885831 / 17899069: rule adopted W1R (D-2026-10-16-1).
- Default config = final_r3; tests 174→175 passed; entry time-budget check 17902758 (4.1 h for 3 members); container
  build + in-container run (17902785/6, 17967832/3, 17969157/8) all passed; audits in SUBMISSION_AUDIT.md;
  organisers' instructions checked (D-2026-10-19-2); README, SUBMISSION.md.

## Job ids
- Website check + unlabelled-header tolerance (2026-10-19): 17969156 CPU tests; 17969157 container rebuild; 17969158 in-container entry run
- Submission hardening (2026-10-19): 17967831 CPU tests (174 passed); 17967832 container rebuild (BUILD OK); 17967833 in-container entry run (RUN OK) -> tag r3-submission
- 17902785 container rebuild from commit 3103ff9: BUILD OK (Dockerfile-equivalent, 22 min); 17902786 entry inside the SIF on the 400-record set: RUN OK (3 members, 400/400 outputs, 7m45); 17902758 entry time-budget check: COMPLETED 5h01 (train 4.1 h for 3 members, inference 52 min; T4 projection ~10 h), ENTRY CHECK OK
- 17902698 CPU tests after the default flip: 11 failed (tests assumed the old defaults; now use tests/_legacy.legacy_config); 17902744 tests: collection error (import placed inside a parenthesised import); 17902757 tests re-run; 17902758 entry time-budget check (afterok)
- 17885659 container build FAILED (buildkit needs docker.sock); 17885678 def build FAILED (host fakeroot lib needs glibc 2.38); 17885786 def build: SIF BUILT (11 GB, Dockerfile-equivalent: apt+pip in a root-mapped namespace); its verify step failed only because the exec mounted $HOME and picked up a broken user-site numpy (fixed: --containall --cleanenv PYTHONNOUSERSITE=1); 17886048 container run
- 17885828 r3 smoke (PASS); 17885829 runner smoke (PASS); 17885830 r3 array (14/14 COMPLETED, 1.6-2.8 h each, jagupard27); 17885831 r3 table (COMPLETED); 17899069 table vs W1; 17886048 container run FAILED (symlinked smoke set invisible under --containall); 17899070 container run FAILED (ENOSPC: --containall tmpfs /tmp); 17902656 container run FAILED (.nfs busy: /tmp on NFS); 17902709 container run PASSED (train_model.py -> run_model.py -> evaluate_model.py inside the SIF on the 400-record set; image built from the pre-flip tree, so it needs one rebuild with the final code)
