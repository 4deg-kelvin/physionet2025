# Chagas disease from the 12-lead ECG — PhysioNet Challenge 2025 entry

Fine-tuned ECG-FM foundation model (learnable aggregation over its 12 transformer layers → MLP head), 5 s input
windows, hard-negative ranking loss, three-seed ensemble trained in one call. Layout follows the organisers'
[python-example-2025](https://github.com/physionetchallenges/python-example-2025): `train_model.py`,
`run_model.py` are the organisers' scripts (byte-identical to their example), `helper_code.py` and
`evaluate_model.py` are their earlier versions (see SUBMISSION.md); the team's code is `team_code.py` → the
`chagas/` package.

```bash
docker build -t chagas .
python train_model.py -d training_data -m model -v
python run_model.py -d holdout_data -m model -o holdout_outputs -v
python evaluate_model.py -d holdout_data -o holdout_outputs -s scores.csv
```

No configuration is needed: the package defaults (`chagas/config.py`) are the entry. Training needs
`code15_exams.csv` (committed) and the ECG-FM checkpoint, which the Dockerfile downloads; nothing is fetched at run time.

| Document | Contents |
|---|---|
| `SUBMISSION.md` | what is submitted, measured resources, checks performed |
| `SUBMISSION_AUDIT.md` | submission-readiness audits |
| `PLAN.md`, `PLAN_R2.md`, `PLAN_R3.md`, `DECISIONS.md` | the dated record of experiments and decision rules |
| `METHODS.md`, `PROVENANCE.md` | what the code does and which ideas are borrowed from whom |
| `experiments/README.md`, `experiments/REPRODUCE.md` | every result table and how to regenerate it |
| `paper/` | manuscript (`bash paper/build.sh`) |

Legacy modules at the root (`fm.py`, `mae*.py`, `se_model.py`, `transformer.py`, `wavelnet.py`, …) are earlier
approaches kept for reference; nothing on the judged path imports them.
