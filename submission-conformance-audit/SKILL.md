---
name: submission-conformance-audit
description: Audit the runtime code that team_code.py executes for PhysioNet Challenge submission-readiness on the organizers' isolated EC2 (g4dn.4xlarge, T4 GPU, 72h training limit, no internet, limited submissions) — checks the 4 protected files are byte-for-byte unchanged, the train/load/run_model signatures still match the organizer wrappers, and every user-authored module in team_code.py's import path is free of absolute paths, hard internet dependencies, stray file writes, T4-memory risks, and fatal-on-one-record exception gaps. Use when finalizing code for submission, after touching team_code.py or the modules it calls, or whenever asked "is this submission ready" / "will this run on the challenge server".
---

# Submission conformance audit

This is a read-only audit. Do not "fix" anything automatically — report findings first so the user
decides. Run every check below, in order, then emit the consolidated report described at the end.
Skip nothing: an absent risk (e.g. no network calls) is reported as PASS/N/A, not silently omitted,
so its absence is visible.

Context for why this exists: the organizers run this repo on an **isolated `g4dn.4xlarge` EC2 (16 vCPU,
60 GB RAM, one 16 GB NVIDIA T4) with no internet access**, under a **72-hour training limit** and a
24-hour validation limit, and the number of submissions is **strictly limited**. A crash that could have
been caught locally — an absolute path that doesn't exist on their box, a stray `http` call that
throws when offline, one malformed record that aborts the entire test run — wastes a scarce
submission. This checklist catches that class of failure before it ships. The **Dockerfile is out
of scope here** — it gets internet at build time and is covered by the separate
`docker-submission-audit` skill; defer all Dockerfile/dependency questions there.

## Scope — what is and isn't audited

The organizers invoke exactly three entry points — `train_model.py`, `run_model.py`,
`evaluate_model.py` — which are thin wrappers that call functions in `team_code.py`. `team_code.py`
itself is mostly a dispatcher; the substantive logic lives in the modules it **imports**. So the
conformance surface is:

- **Protected / frozen files** (immutability check only, Check 1): `train_model.py`, `run_model.py`,
  `evaluate_model.py`, `helper_code.py`. These must never change. `helper_code.py` is in the execution
  path but is do-not-edit, so it is checked for drift, not audited for conformance. Note this repo also
  has `custom_helper_code.py` — a team-owned superset of `helper_code.py`. It is **not** protected; it
  is part of the audited surface below.
- **User-authored call-path modules** (full conformance audit, Checks 4–9): `team_code.py` plus every
  repo-local module it imports transitively that is **not** one of the protected files. This is a large
  set in this repo — `team_code.py` is a thin dispatcher and the substance lives several imports deep.
  Check 3 discovers it dynamically each run; do not assume the closure from a previous audit.
- **Out of scope**: repo files not reachable from `team_code.py`'s import graph are exempt — they never
  run on the challenge server. At time of writing that exempts the alternative model branches
  (`se_model.py`, `wavelnet.py`, `mae.py`, `finetuning.py`, `ensemble.py`, `train_se.py`) — but these are
  live alternatives that get swapped into `team_code.py`, so re-run Check 3 after any such swap rather
  than reusing this list. The Dockerfile is out of scope (see `docker-submission-audit`).

## Before you start

Read `team_code.py` in full, then read every user-authored module Check 3 identifies in its import
path. You need their exact current contents for every check below — do not rely on memory or on
summaries from earlier in the conversation; the files may have changed.

## Check 1 — The 4 protected files are byte-for-byte unchanged

These files must match their canonical baseline exactly. `train_model.py`, `run_model.py`, and
`evaluate_model.py` are replaced by the organizers with their own copies at judging time, so any
local edit is both pointless and a sign of drift; `helper_code.py` ships as-is and must stay pristine.

Compute the SHA-256 of each file and compare against the baseline below.

PowerShell:
```powershell
Get-FileHash -Algorithm SHA256 train_model.py,run_model.py,evaluate_model.py,helper_code.py | Format-List Hash,Path
```
Bash:
```bash
sha256sum train_model.py run_model.py evaluate_model.py helper_code.py
```

Baseline (captured 2026-08-11 from this repo's clean working tree at commit `9aeb53d`; `git status`
reported no modifications to any of these four files, so this is the as-committed template state):

| File | SHA-256 |
|------|---------|
| `train_model.py`    | `C10BAC1C96390672B182E39A5D7FBCA9E83CE06F7F90D71698AA0A44A7A81216` |
| `run_model.py`      | `A32837B6AF881E351AFB1411B8FC733D4F34FA178C2E03D960829946C6F591A5` |
| `evaluate_model.py` | `216B62EC767B673A8CA8334C9B1EFF8E2EBAF5166EBC722587DD5DC662F1E673` |
| `helper_code.py`    | `63C2E2D0161E12049303F4B25F0528AC930BD75B122302B4DC7BA11DB669DEEE` |

- Every hash matches → **PASS**.
- Any mismatch → **FAIL**. Name the file, and diff it (`git diff -- <file>` if the change is
  uncommitted, or show the offending lines) so the user can revert or, if the change was intentional
  and the file is genuinely final, re-snapshot the baseline in this table. Hashes are case-insensitive;
  compare uppercase-to-uppercase.
- If comparing against `git` instead is preferred, note that these files may already show as modified
  vs the last commit — the SHA-256 table above, not `git HEAD`, is the source of truth for "unchanged".

## Check 2 — Required function signatures still match the organizer wrappers

The organizers call the `team_code.py` functions with a fixed calling convention. Read the wrapper
call sites (`train_model.py`, `run_model.py`) to confirm the expected shape, then verify
`team_code.py` still satisfies it:

- `train_model(data_folder, model_folder, verbose)` — extra trailing params are allowed **only if they
  have defaults**, because the wrapper passes exactly 3 positional args. A new required 4th param → **FAIL**.
- `load_model(model_folder, verbose)` — must return something `run_model` accepts (currently an
  `fm.FMChagasClassifier`, i.e. a live `nn.Module` already moved to the target device).
- `run_model(record, model, verbose)` — note the argument order: `run_model.py` calls
  `run_model(os.path.join(data_folder, record), model, verbose)`, so **record comes first and model
  second**. A signature that swaps them will not raise `TypeError`; it will silently pass a path where
  a model is expected. Check the order explicitly, don't just count parameters. Must return the
  `(binary_output, probability_output)` tuple the wrapper unpacks.
- Any signature that would raise `TypeError` when called the way the wrapper calls it → **FAIL**, quote
  both the wrapper call site and the `def` line. Otherwise → **PASS**.

## Check 3 — Trace the execution path and classify modules

Establish exactly which files Checks 4–9 apply to. Starting from `team_code.py`, resolve its
**repo-local** imports transitively (follow `import x` / `from x import ...` where `x.py` exists in the
repo; ignore stdlib and third-party packages). `from helper_code import *` counts as importing
`helper_code`.

```bash
grep -nE '^[[:space:]]*(import|from)[[:space:]]' team_code.py
```
Then repeat for each local module discovered, until the set stops growing.

Partition the resulting local modules:
- **Protected** (matches a Check-1 file, e.g. `helper_code`) → frozen; not audited for conformance here.
- **User-authored** (everything else) → the audit surface for Checks 4–9.

Report the discovered surface as a list. At time of writing the closure is `team_code.py` → `fm.py`,
`utils.py`, `dataloader.py`, `custom_helper_code.py`, `transformer.py`, `pretrain_mae_vit_ecg.py`,
`finetuning_mae_vit_ecg.py`, `mae_vit_ecg.py`, `augmentations.py` (lazily imported inside
`ECGDataset.__init__`), plus the protected `helper_code.py`. Re-derive it rather than copying that
list — `team_code.py` imports several modules it does not currently call (`transformer`,
`pretrain_mae_vit_ecg`, `finetuning_mae_vit_ecg`), and those still execute at **import** time, so
their module-level code and their own imports are in scope even though their functions are never
invoked. Flag unused-but-imported modules as a **WARN** under Check 9 — each one is import-time
risk taken for no benefit.

This check is **informational** (no PASS/FAIL), but if an import cannot be resolved to a repo file or
an installed package, flag it — that is a likely `ModuleNotFoundError` on the challenge server →
**FAIL**. Also flag an attribute referenced on the wrong module (e.g. calling a function on
`helper_code` that only exists in `custom_helper_code`): that is an `AttributeError` at runtime and
static import resolution alone will not catch it.

## Check 4 — No absolute or hardcoded paths in the audited surface

All filesystem access must derive from the arguments passed in (`data_folder`, `model_folder`, and the
organizer wrapper's `output_folder`) or from script-relative bundled data that ships inside the image.
On the challenge EC2, absolute paths from the dev machine simply don't exist.

```bash
grep -nE 'C:\\|/home/|/Users/|/mnt/|/tmp/|/data/|[A-Za-z]:\\\\' team_code.py <user-authored modules>
grep -nE 'os\.path\.abspath|os\.getcwd|Path\(.*\)\.absolute|expanduser|/root/' team_code.py <user-authored modules>
```

- Any absolute OS path, home-directory reference, or drive letter → **FAIL**, cite `file:line`. Two
  known instances to expect: `fm.py`'s `--grad-cam` branch hardcodes a `/sailhome/...` checkpoint path.
  Both live inside `if __name__ == '__main__'`, so they never execute on the submission path — report
  them as **WARN** rather than FAIL, but confirm the guard is really there before downgrading.
- Script-relative bundled data is **allowed**: the pattern `SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))`
  then joining a repo file is fine **provided that file is committed and ships in the image** — verify
  the referenced file exists in the repo. `abspath(__file__)` is the one sanctioned use of `abspath`.
- **Bare relative filenames are the sharp edge in this repo** and deserve their own pass. They are not
  absolute paths, so the greps above miss them, but they resolve against the *process CWD* rather than
  the script directory. They happen to work only because the Docker `WORKDIR` is `/challenge` and the
  files sit at the repo root. Check each one resolves to a committed file:
  - `mimic_iv_ecg_finetuned.pt` — `fm.train_model` and `fm.load_finetuned_model`. **Not committed**; it
    exists only because the Dockerfile `gdown`s it into `/challenge`. Verify the Dockerfile still
    fetches it under exactly that filename — a rename on either side is a silent `FileNotFoundError`
    on the organizers' box.
  - `ecg_statistics.csv` and `code15_exams.csv` — `ECGDataset` defaults. Both committed; note that a
    miss here does **not** raise, it logs a warning and silently disables normalization / multi-task
    labels, which is worse than a crash. → **WARN**.
- All paths derived from passed-in args, `SCRIPT_DIR`+committed-file, or a CWD-relative committed file
  that the image guarantees → **PASS**.

## Check 5 — No hard internet dependency at train/infer time

The challenge server has no internet. Code must not *require* a network call to succeed. A call that is
absent is best; a call that is present but wrapped so an offline failure is caught and the code
continues is tolerable; a call whose exception would propagate and crash the run is a blocker.

```bash
grep -nE 'requests|urllib|urlopen|http[s]?://|socket|boto3|\bs3\b|gdown|wget|curl|download|from_pretrained|hf_hub|torch\.hub' team_code.py <user-authored modules>
```

- No network usage anywhere → **PASS**.
- Network call present but inside a `try/except` (or otherwise guarded) so that an offline failure
  degrades gracefully rather than raising → **WARN**, cite `file:line` and confirm the fallback path.
- Network call that would raise on failure with no guard (e.g. downloading weights at load time) →
  **FAIL** — this will crash on the isolated server. Any model weights must be committed/shipped, not
  fetched at runtime.
- Reminder: build-time downloads in the **Dockerfile** are fine and out of scope — do not flag those
  here; defer to `docker-submission-audit`.

## Check 6 — File writes confined to the passed-in output folders

At training time the only writable destination is `model_folder`; at inference the organizer wrapper
writes predictions into `output_folder` (team_code's `run_model` should write nothing itself).

```bash
grep -nE "open\(.*['\"][wa]|to_csv|to_pickle|np\.save|savez|joblib\.dump|savefig|makedirs|mkdir|os\.remove|shutil\." team_code.py <user-authored modules>
```

- Every write target is built from `model_folder` (or is inside the organizer's `output_folder` path)
  → **PASS**.
- A write to any path not derived from those args (cwd, a hardcoded dir, a sibling of the repo) →
  **FAIL**, cite `file:line`.
- Note as **WARN** any large intermediate the code leaves inside `model_folder` and ships with the
  model; not fatal, but flag the bloat.
- **`wandb` is a live risk in this check and in Check 5.** `dataloader.py`, `transformer.py`,
  `pretrain_mae_vit_ecg.py`, and `finetuning_mae_vit_ecg.py` all import it, and `fm.py` imports
  `WandbLogger`. A `wandb` run both writes a `wandb/` directory into the CWD *and* attempts to reach
  the network. Importing it is harmless; **initializing** a logger on the submission path is not.
  Confirm the code path the organizers actually execute passes `logger=False` to `pl.Trainer` (as
  `fm.train_model` currently does) and never calls `wandb.init`. If a logger is constructed on that
  path → **FAIL** under Check 5, and flag the `wandb/` directory write here.

## Check 7 — Fits on a T4 GPU (16 GB) / bounded memory

The runtime box is a single T4 (16 GB VRAM). Anything CPU-only trivially satisfies this; the risk is a
GPU model or an unbounded in-memory load.

- No `torch` / `tensorflow` / `cupy` / CUDA / `.cuda()` / `device='cuda'` anywhere in the audited
  surface → GPU concern is **N/A** (the T4 simply sits idle, which is allowed).
- This repo **does** use the GPU: it fine-tunes a 12-layer ECG-FM transformer under PyTorch Lightning.
  Confirm each of the following:
  - Device selection is graceful. `team_code.load_model` uses `'cuda:0' if torch.cuda.is_available()
    else 'cpu'` and `fm.train_model` uses `accelerator="auto"` — both fine. A hardcoded `.cuda()` with
    no availability guard → **FAIL**.
  - Model + batch fit in 16 GB. The current configuration (`BATCH_SIZE=16`, 12 leads × 5000 samples,
    768-d embeddings, encoder **frozen** via `freeze_encoder=True`) fits comfortably. The number to
    watch is `unfreeze_after_epoch` / `unfreeze()`: unfreezing the encoder makes every layer's
    activations and optimizer state trainable and is the one change most likely to OOM a T4. If the
    submission path unfreezes, say so explicitly and re-estimate.
  - `run_model` processes **one record at a time** with `torch.no_grad()` — verify the no-grad guard is
    still present; losing it leaks activation memory across records.
- Independently, flag any code path that loads **all** records into memory at once rather than
  streaming. `ECGDataset` streams per-record, which is correct. Unbounded accumulation across records →
  **WARN** (or FAIL if it plausibly exhausts the 60 GB host RAM).
- **Wall-clock is a hard constraint too**: training must finish inside 72 hours. `fm.py` provides
  `TimeLimitCallback(max_hours=71.0)` for exactly this. Verify it is still in the `pl.Trainer`
  `callbacks` list on the submission path — a training path that can exceed the limit without a
  self-imposed stop is a **WARN** (it wastes a scarce submission on a timeout, not a crash).

## Check 8 — Exception-handling gaps (flag only, do not prescribe)

The goal at runtime: a single bad record should be **skipped** so the submission continues, while a
**critical/setup failure** (missing model file, unreadable config, out-of-memory) should stop early
rather than silently produce garbage across a limited submission. This check **flags gaps** — it does
not mandate a specific structure.

Read each per-record processing loop and each entry function, and report:
- **Per-record loops with no surrounding `try/except`** whose failure would abort the whole run. The
  organizer wrapper only tolerates per-record failures when invoked with `-f/--allow_failures`, so do
  not rely on that flag being passed.
- The current `team_code.run_model` **does** guard itself: it catches broad `Exception` and returns
  `(None, None)`, while deliberately re-raising `NotImplementedError` as a "preprocessing is wrong, do
  not emit predictions" signal. Verify both halves survive edits — losing the broad catch reintroduces
  the one-bad-record-aborts-everything failure; losing the `NotImplementedError` re-raise means a
  systematically broken preprocessing path silently returns `None` for every record and scores zero.
- Note the downstream consequence of returning `(None, None)`: `evaluate_model.py` scores a missing
  output as **0**, not as an abstention. Silent per-record failure is therefore invisible in the logs
  but costly in the score — flag any handler that swallows without printing under `verbose`.
- **Except handlers that can themselves raise** — e.g. a handler that interpolates a variable assigned
  inside the `try`; if the exception fires before that assignment the handler raises `NameError` and
  crashes. Flag any instance.
- Broad `except: pass` that would **swallow a critical setup error** (missing checkpoint, corrupt
  config) and march on producing meaningless output → flag as risky. Contrast with
  `team_code.load_model`, which correctly raises `FileNotFoundError` when the checkpoint is absent —
  setup failures should stop early, per-record failures should not.
- Report each as **WARN** with `file:line` and a one-line description of the failure scenario. Do not
  rewrite the error-handling strategy; surface the gaps and let the user decide.

## Check 9 — Hygiene (WARN)

Minor, non-blocking, but worth listing:
- Dead / unused imports in the audited surface. Distinguish two severities: an unused *third-party*
  import (e.g. `joblib`, `argparse`, `pathlib` in `team_code.py`) is cosmetic, while an unused
  *repo-local module* import (`transformer`, `pretrain_mae_vit_ecg`, `finetuning_mae_vit_ecg`) drags
  that module's entire import-time surface onto the submission path for no benefit — call the latter
  out specifically.
- Large committed artifacts that ship inside the image without being needed at runtime (this repo
  commits `code15_exams.csv` at ~36 MB, `train_val_test_sets.csv` at ~5 MB, and
  `code15_label_issues.csv` at ~2 MB at the repo root).
- Anything that adds needless size or noise to the shipped model.
All **WARN**.

## Report format

Emit one consolidated report, one line per check (1–9), each tagged **PASS / FAIL / WARN / N/A**, ranked
most-severe first (all FAILs, then WARNs, then PASS/N/A), with a one-sentence reason and a `file:line`
reference where applicable. Check 3 additionally lists the discovered audit surface. End with a single
overall verdict line: **"Submission-ready: YES"** or **"Submission-ready: NO — see FAIL items above."**
The verdict is driven purely by whether any check produced a FAIL; WARN items don't block but must be
listed.

After the report, you may **offer** to apply fixes — but only in **user-authored files in the
`team_code.py` call path** (Check 3's surface). Never edit the 4 protected files (revert them instead),
and never edit the Dockerfile (defer to `docker-submission-audit`). Wait for the user to confirm which
specific findings to act on before changing anything.
