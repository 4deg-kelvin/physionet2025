# Submission audits (2026-10-19, commit after f62daef)

Two read-only audits were run by independent agents following the checklists in `submission-conformance-audit/SKILL.md`
and `docker-submission-audit/SKILL.md`. **Neither found a FAIL.** Their WARN items and what was done about them:

## Runtime code (team_code.py → chagas/)
| Check | Result | Note |
|---|---|---|
| Protected files unchanged (`train_model.py`, `run_model.py`, `evaluate_model.py`, `helper_code.py`) | PASS | identical to the baseline commit; SKILL hash table was captured with CRLF endings |
| Wrapper signatures | PASS | `train_model(data, model, verbose)`, `load_model(model, verbose)`, `run_model(record, model, verbose)` |
| Absolute paths at run time | PASS | none; bundled files resolve against the repo root, `CHAGAS_CONFIG` optional, missing `ptbxl_database.csv` handled |
| Network at run time | PASS (static) | no HTTP/hub/wandb on the judged path |
| Writes outside model/output folders | PASS | all under `model_folder`; inference writes nothing |
| T4 memory / 72 h | PASS | peak 5.0 GiB reserved (train), 2.7 GiB (inference); members get 71/3 h each, measured 4.1 h total on a Titan RTX |
| Per-record exception handling | WARN → fixed | records that cannot be preprocessed returned `None` silently; `run_model` now logs them under `-v` |
| Hygiene | WARN, left | unused `exclude_label_issues` option; legacy modules and two unused CSVs ship but are not imported |

## Dockerfile and image
| Check | Result | Note |
|---|---|---|
| "DO NOT EDIT" lines | PASS | byte-identical |
| Base image / CUDA / T4 | WARN, by design | cu121 base with the cu118 torch wheel (documented in the Dockerfile); sm_75 covered; needs driver ≥ 530 or the Tesla forward-compat path |
| Checkpoint download | WARN → fixed | Drive link verified working on 2026-10-19 (1,081,641,499 bytes); added a Hugging Face mirror fallback and a sha256 check so a Drive refusal or truncated file fails loudly at build time |
| Imports all installable | PASS | `pytorch-lightning` was floating (image resolved 2.6.6, results used 2.5.2) → pinned 2.5.2 |
| Layer order / cache | WARN → fixed | `PIP_NO_CACHE_DIR=1` (≈3 GB smaller image); `.dockerignore` patterns now `**/`-prefixed so nested checkpoints are excluded |
| Run-time network insurance | added | `HF_HUB_OFFLINE=1 WANDB_MODE=disabled` |
| Real `docker build` | N/A | no daemon here; the Apptainer rendering (`experiments/container/chagas.def`) builds and runs the entry |

Re-verification after these edits: CPU test suite, container rebuild and in-container entry run (job ids in
`experiments/R3_PROGRESS.md`).
