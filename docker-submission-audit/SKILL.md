---
name: docker-submission-audit
description: Audit this repo's Dockerfile and dependencies for PhysioNet Challenge submission-readiness — checks the Dockerfile exists, the base image matches what team_code.py actually needs, the 3 "DO NOT EDIT" lines are untouched, every requirements file referenced actually exists, every gdown download is commented and reachable, every third-party import has an install path, and layers are ordered for Docker caching. Use before submitting, after editing the Dockerfile/requirements.txt, or whenever asked "is this submission ready" / "check the Dockerfile".
---

# Docker submission audit

This is a read-only audit. Do not "fix" anything automatically — report findings so the user (or
their PR) can decide. Run every check below, in order, then emit the consolidated report described
at the end. Skip nothing: an absent feature (e.g. no `gdown` usage) is reported as N/A, not silently
omitted, so its absence is visible.

Context for why this exists: a submission was shipped broken at least once because a Google Drive
file ID in the Dockerfile had a typo — the download silently failed only when the organizers built
the image. This repo is especially exposed to that failure mode: the ECG-FM backbone weights
(`mimic_iv_ecg_finetuned.pt`) are **not committed** and exist only because the Dockerfile `gdown`s
them. This checklist exists to catch that class of failure (dead links, missing deps, edited
template lines, mismatched base image) before submission, not after.

## Before you start

Read the repo's `Dockerfile`, its requirements file(s) (here `requirements_linux.txt`), `team_code.py`,
and `helper_code.py` in full. You need their exact contents for every check below — do not rely on
memory or summaries from earlier in the conversation, the files may have changed since.

## Check 1 — Dockerfile exists

The repo root must contain a file named exactly `Dockerfile`. If it's missing, this is a hard
FAIL and nothing else can be meaningfully checked — report that immediately and stop, don't
guess at what a hypothetical Dockerfile would contain.

## Check 2 — Base image matches actual runtime needs

Parse the `FROM` line (the first non-comment line of the Dockerfile). Build the dependency set
first (see Check 6 — do that resolution now if you haven't) and cross-reference:

- If the dependency set includes anything that requires a GPU/CUDA runtime (`torch` used with
  `.cuda()`/`device='cuda'`, `tensorflow-gpu`, `cupy`, etc.) but `FROM` is a plain image with no
  `nvidia/cuda` (or similar GPU-enabled) ancestry → **FAIL**. A CPU-only image cannot run CUDA code.
- If `FROM` is an `nvidia/cuda` (or similar GPU) image but nothing in the dependency set actually
  needs a GPU (e.g. a pure `numpy`/`pandas`/`scikit-learn`/`xgboost` stack) → **WARN** (works, but
  needlessly bloats image size and build time — CUDA base images are multiple GB larger).
- If any dependency is installed via `pip install -e <path>/` (an editable/source install of a
  package with native extensions) or otherwise needs a compiler toolchain, `FROM` must be a
  `-devel` (or equivalent full) variant, not a `-slim`/`-runtime`/`-alpine` image → **FAIL** if
  mismatched, since the build step itself will fail without a compiler.
- Otherwise → **PASS**.

This repo trains a torch model on the organizers' T4 and does an editable install of a vendored
source package (`pip install -e fairseq-signals/`), so it needs **both** GPU ancestry and a
compiler toolchain. Base image `nvidia/cuda:12.1.0-devel-ubuntu22.04` satisfies both and should
PASS. Two regressions this check is watching for: a switch to a `-runtime` or `-slim` variant
(breaks the `fairseq-signals` build), or a CUDA/torch version drift between the base image and the
`--index-url` used for the torch wheel — note the Dockerfile currently pairs a **cu121** base image
with a **cu118** torch wheel. That mismatch is tolerated by the driver but flag it as a **WARN**
so it stays a deliberate choice.

## Check 3 — The 3 "DO NOT EDIT" lines are unmodified

The official template marks 3 lines as fixed, usually directly below a comment like
`## DO NOT EDIT these 3 lines.` (wording may vary slightly — search for "DO NOT EDIT" case
insensitively if the exact phrasing differs). The canonical content, in this exact order, is:

```
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge
```

- Locate these 3 lines and diff them character-for-character against the canonical text above.
  Any deviation — a different path, added flags, reordering, a line inserted between them, a
  changed source/destination in the `COPY` — is a **FAIL**. These lines exist so the organizers'
  harness can rely on a fixed container layout; editing them breaks that contract even if the
  Docker build still succeeds locally.
- If the 3 lines are present verbatim but the marker comment above them has been deleted or
  reworded, still PASS the content check but separately note a WARN for the missing marker (a
  future editor without this skill loses the warning not to touch them).
- If the lines are missing entirely (not just reworded), that's a FAIL — the container will not
  have the expected `/challenge` working directory.

## Check 4 — Every referenced requirements file actually exists

Extract every `-r <path>` argument from any `RUN pip install ...` line in the Dockerfile (there
may be more than one, e.g. `requirements.txt` plus a platform-specific
`requirements_linux.txt`). For each extracted path, verify the file exists in the repo at that
relative path (relative to the Docker build context, i.e. the repo root). Any dangling reference
is a **FAIL** — `docker build` will fail outright on this line.

## Check 5 — Every `gdown` invocation is commented and reachable

- Find every line that invokes `gdown` in the Dockerfile: `RUN gdown ...`, or `gdown` chained with
  `&&` inside a larger `RUN`. Also check whether `gdown` itself is actually installed (via
  requirements file or a direct `pip install gdown`) before it's invoked — if not, that's a
  separate FAIL under Check 6 as well as this one, since the download step can't run at all.
- **Comment requirement**: each gdown line must have an adjacent comment — either directly above
  it or on the same line — stating what file/model the ID or URL points to. Follow the convention
  seen in prior submissions:
  ```
  # 1uI2J_gMk0eh0vu3MbKPBEoupgakZ06j2 is for FAIRSEQ FOUNDATION MODEL
  RUN gdown 1uI2J_gMk0eh0vu3MbKPBEoupgakZ06j2
  ```
  A gdown line with no such comment is a **FAIL** — a bare file ID is meaningless to a future
  reader and impossible to sanity-check without re-downloading it.
- **Reachability test**: for each gdown line, actually attempt the download to confirm the link
  still works (Google Drive links rot — files get deleted, made private, or hit quota). Use the
  repo's scratch/temp directory, never the repo itself, and always clean up afterward:
  ```bash
  gdown --id <FILE_ID> -O <scratch_dir>/gdown_test_<FILE_ID> --fuzzy
  echo "exit code: $?"
  ls -la <scratch_dir>/gdown_test_<FILE_ID>
  rm -f <scratch_dir>/gdown_test_<FILE_ID>
  ```
  - Exit code 0 and a non-empty file → **PASS**.
  - Non-zero exit, an HTML error/quota page instead of the real file, "Permission denied", or
    "Cannot retrieve the public link" → **FAIL**, quote the actual error text in the report.
  - For large files, don't block indefinitely: run the download with a bounded timeout (e.g.
    `timeout 60 gdown ...` or equivalent). If it's still transferring real bytes when the timeout
    hits, treat that as evidence the link is reachable (report as PASS with a note that full
    integrity wasn't confirmed within the time budget) rather than blocking the whole audit on one
    multi-GB checkpoint.
  - If `gdown` isn't installed in the current environment, install it in the scratch venv/context
    only for the purpose of this test (`pip install gdown` in a way that doesn't touch the repo's
    own requirements file), or note the limitation clearly in the report rather than skipping
    silently.
- If there are no `gdown` lines in the Dockerfile at all, report this check as **N/A — no gdown
  usage found**, not simply omitted.

## Check 6 — Every third-party import used by team_code.py has an install path

1. Read `team_code.py` fully and enumerate every third-party import: top-level imports, and any
   lazy/inline imports inside function bodies (e.g. `dataloader.py` imports `augmentations` inside
   `ECGDataset.__init__`, and `fm.py` re-imports `matplotlib` inside function bodies — don't miss
   imports like these just because they're not at the top of the file).
2. `team_code.py` is a thin dispatcher; nearly all third-party dependencies arrive **transitively**
   through the repo-local modules it imports. Resolve those imports recursively and enumerate the
   third-party imports of each. At time of writing the local closure is `team_code.py` → `fm.py`,
   `utils.py`, `dataloader.py`, `custom_helper_code.py`, `helper_code.py`, `transformer.py`,
   `pretrain_mae_vit_ecg.py`, `finetuning_mae_vit_ecg.py`, `mae_vit_ecg.py`, `augmentations.py`.
   Re-derive it rather than trusting that list. Auditing only `team_code.py`'s own import block
   would miss almost everything (`neurokit2`, `wfdb`, `biosppy`, `timm`, `wandb`, `pandas`,
   `pytorch_lightning`, `torchmetrics`, `fairseq_signals`, …).
3. Build the combined set of third-party (non-stdlib) package names actually used.
4. Cross-reference against everything the Dockerfile makes available:
   - Every package listed in any `requirements*.txt` file installed via `-r`.
   - Any package installed via a standalone `RUN pip install <pkg>` line directly in the
     Dockerfile (used when a package needs special handling, e.g. a specific `--index-url` for a
     CUDA build of `torch`, or a pinned version installed separately from the rest).
   - Any package satisfied by an `apt-get install` system package (rare for Python imports, but
     relevant for e.g. `libsndfile` needed by `soundfile`).
5. Apply import-name → distribution-name mapping rather than naive string matching. Use your
   knowledge of common PyPI naming mismatches, and note the ones specific to this repo:
   - `import sklearn` ← `scikit-learn`
   - `import pywt` ← `pywavelets`
   - `import pytorch_lightning` ← the `lightning` distribution (which ships the
     `pytorch_lightning` top-level module as well); every model module here imports the
     `pytorch_lightning` name, not `lightning`
   - `import fairseq_signals` ← `RUN python -m pip install -e fairseq-signals/`, i.e. the vendored
     source tree, not PyPI
   - `import timm`, `import wfdb`, `import neurokit2`, `from biosppy...` ← same-named distributions
6. Any third-party import with no install path anywhere → **FAIL**, name the specific package and
   where it's imported. Report unused-but-listed requirements (declared in the requirements file but
   never imported anywhere) as a WARN, not a FAIL — harmless but worth flagging as dead weight.
7. Separately, flag as **WARN** any import satisfied only **transitively** — used directly in the
   code but absent from the requirements file, resolved by chance because some other dependency
   pulls it in. Two live examples here: `matplotlib` (arrives via `biosppy`) and `tqdm` (via
   `gdown`). These work today and break silently if the intermediate dependency drops them; name
   them and recommend an explicit pin.

## Check 7 — Layers are ordered for Docker build-cache efficiency

The 3 mandated "DO NOT EDIT" lines already copy the *entire* repo into the image before anything
else can run, which limits how much caching benefit is achievable — but there's still real room
for improvement around that fixed block:

- System-level `apt-get`/OS package installs that don't depend on repo contents (compilers,
  shared libraries, etc.) should appear *before* the "DO NOT EDIT" block. They should never be
  invalidated just because a `.py` file changed.
- `RUN pip install -r requirements.txt` should immediately follow `WORKDIR /challenge` — this
  matches the template's own default placement. Flag (as **WARN**) any unrelated `RUN` steps
  wedged between `WORKDIR` and the requirements install with no dependency reason, since that adds
  needless cache-invalidation risk for no benefit.
- Version- or index-url-sensitive installs (e.g. a CUDA-specific `torch` wheel, an editable
  install of a vendored source package, a `gdown`-fetched model checkpoint) should each be their
  own `RUN` layer, placed after the requirements install — not bundled into one giant `RUN` with
  unrelated steps. This mirrors the working pattern from a prior year's submission:
  ```dockerfile
  RUN pip install -r requirements.txt
  RUN pip install torch --index-url https://download.pytorch.org/whl/cu118
  RUN gdown <id>
  ```
- Combine multi-line `apt-get` installs into a single `RUN` ending in
  `&& rm -rf /var/lib/apt/lists/*` to avoid leaving package-manager cache in an intermediate layer.
- These are all **WARN**, not FAIL — they affect build speed/cache-hit rate, not correctness. Cite
  specific line numbers and a concrete reordering suggestion for each one raised.

## Check 8 — (Optional) Attempt a real `docker build`

If `docker` is available (`docker --version` succeeds), run the strongest possible check:

```bash
docker build -t physionet-submission-check .
```

from the repo root. Report the actual failure output verbatim if it fails — this catches anything
the static checks above might miss (e.g. a subtle syntax error, a network hiccup during the real
build). If Docker isn't available in the current environment, skip this step and say so in the
report; don't fail the whole audit just because the tool is absent.

## Report format

Emit one consolidated report, one line per check (1–8), each tagged PASS / FAIL / WARN / N/A, with
a one-sentence reason and a `file:line` reference where applicable. End with a single overall
verdict line: **"Submission-ready: YES"** or **"Submission-ready: NO — see FAIL items above."**
The verdict is driven purely by whether any check produced a FAIL; WARN items don't block
submission but should still be listed.
