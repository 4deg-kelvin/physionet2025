# Manuscript

Draft for the *Physiological Measurement* focus collection **"Detection of Neglected Tropical Diseases from
the Cardiac Signals"** (https://iopscience.iop.org/collections/pmea-250826-920; guest editors Clifford, Gaye,
Reyna, A H Ribeiro, T Ribeiro, Sameni; deadline 31 December 2026).

Journal requirements followed: research paper (not normally more than 8,000 words), structured abstract with
the headings *Objective, Approach, Main results, Significance* (at most 250 words), Harvard (author-year,
alphabetical) references with article titles, IOP `iopjournal` LaTeX class.

## Build

    bash paper/build.sh          # writes paper/main.pdf

`build.sh` runs `make_assets.py` and then compiles `main.tex`.  No model is run.

| File | Role |
|---|---|
| `main.tex` | the manuscript; contains no hand-typed result numbers |
| `make_assets.py` | writes `numbers.tex` (macros), `tables/*.tex`, `figures/*.pdf` from `experiments/results` |
| `refs.bib` | references |
| `iop/` | IOP Publishing's author template: class file, ORCID logo, template source (the guidelines PDF and the zip are not committed; download link in `build.sh`) |

Inputs of `make_assets.py`: `experiments/data/record_durations.csv`, the split CSV, `RESCORED_*.csv`
(round 1), `R2_TABLE_{A,C}.csv` (round 2; `sbatch experiments/batch_r2_table.sh`), per-run predictions and
manifests.  See `experiments/REPRODUCE.md` for how each of those is produced.

LaTeX engine: [tectonic](https://tectonic-typesetting.github.io), a single self-contained binary (the
cluster has no TeX installation).  Set `TECTONIC` to its path; packages are fetched on first use.

## Before submission (authors)

Search `main.tex` for `TODO(authors)`:
1. Author list, order, ORCIDs, affiliations, corresponding address (currently the Challenge-entry team).
2. Funding statement and CRediT author contributions.
3. The generative-AI disclosure in the acknowledgements (IOP Publishing requires one; edit to reflect
   what the authors verified).
4. Data/code availability: the repository URL.
5. Conflict-of-interest and ethics statements requested by the submission form.

Claims that rest on sources outside this repository (leaderboard figures, other teams' results) cite
Reyna et al 2025 and the teams' Computing in Cardiology papers; check them against the published versions.
