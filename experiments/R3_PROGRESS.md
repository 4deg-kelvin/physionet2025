# Round 3 progress (PLAN_R3.md) — running checklist

Updated at every milestone so the work can be resumed by reading this file.

## Done
- PLAN_R3.md written and the decision rule fixed before any run (D-2026-10-15-1..4).

## Done (cont.)
- Code for §2 (fixed validation protocol, per-view validation metrics, prevalence cap, window stride), r3 configs,
  run_r3.sh, batch_r3*.sh, aggregate `--rule r3`, tests, split audit — implemented by an Opus agent (reviewed). CPU tests 17885765: 173 passed. Split audit 17885742: PASS.

## In progress
- Chain submitted (code committed as 12896b6 before the array started): r3 smoke 17885828 -> runner smoke 17885829 (afterok) -> array 17885830 (afterok, 14 tasks %4) -> table 17885831 (afterany). DO NOT edit chagas/ or experiments/*.py|sh while it runs.

## Todo
- Apply the rule; set the default config; T4 time-budget check; container check (docker daemon is not reachable
  from the login node — permission denied on docker.sock; apptainer 1.5.4 is available).
- README round-3 section, REPRODUCE §7, METHODS/PROVENANCE, paper (Methods 2.1/2.2, Results 3.6).

## Job ids
- 17902698 CPU tests after the default flip; 17902710 entry time-budget check (afterok, final_r3: 3 members on data_train_full)
- 17885659 container build FAILED (buildkit needs docker.sock); 17885678 def build FAILED (host fakeroot lib needs glibc 2.38); 17885786 def build: SIF BUILT (11 GB, Dockerfile-equivalent: apt+pip in a root-mapped namespace); its verify step failed only because the exec mounted $HOME and picked up a broken user-site numpy (fixed: --containall --cleanenv PYTHONNOUSERSITE=1); 17886048 container run
- 17885828 r3 smoke (PASS); 17885829 runner smoke (PASS); 17885830 r3 array (14/14 COMPLETED, 1.6-2.8 h each, jagupard27); 17885831 r3 table (COMPLETED); 17899069 table vs W1; 17886048 container run FAILED (symlinked smoke set invisible under --containall); 17899070 container run FAILED (ENOSPC: --containall tmpfs /tmp); 17902656 container run FAILED (.nfs busy: /tmp on NFS); 17902709 container run with node-local /tmp
