"""PLAN.md section 5 item 4: source parsing, soft labels, multi-report, aux, grouping."""

from __future__ import annotations

import pytest

from chagas.config import LabelCfg
from chagas.dataset import RecordEntry
from chagas.labels import (
    Code15Table,
    MultiReportStats,
    aux_targets,
    normalise_source,
    parse_patient_info,
    patient_group,
    soft_label,
)
from chagas.training import split_entries

HEADER = (
    "12345 12 400 4096\n"
    "12345.dat 16 1000/mV 16 0 0 0 0 I\n"
    "# Age: 54\n"
    "# Sex: Female\n"
    "# Chagas label: True\n"
    "# Source: CODE-15%\n"
)


def test_source_parsing():
    assert normalise_source("CODE-15%") == "code15"
    assert normalise_source("SaMi-Trop") == "samitrop"
    assert normalise_source("PTB-XL") == "ptbxl"
    assert normalise_source("ELSA-Brasil") == "unknown"
    assert normalise_source(None) == "unknown"


def test_parse_patient_info():
    info = parse_patient_info(HEADER, want_label=True)
    assert info.age == 54.0 and info.sex == "Female"
    assert info.label == 1 and info.source == "code15"

    # A header without a label: silent when we do not ask, tripwire when we do.
    no_label = HEADER.replace("# Chagas label: True\n", "")
    assert parse_patient_info(no_label, want_label=False).label is None
    with pytest.raises(NotImplementedError):
        parse_patient_info(no_label, want_label=True)


def test_soft_labels_per_source():
    cfg = LabelCfg(multi_report=False)
    assert soft_label(1, "samitrop", "x", cfg) == pytest.approx(1.0)
    assert soft_label(0, "ptbxl", "x", cfg) == pytest.approx(0.0)
    assert soft_label(1, "code15", "x", cfg) == pytest.approx(0.8)
    assert soft_label(0, "code15", "x", cfg) == pytest.approx(0.2)
    # Unknown source (the hidden cohorts) is treated as CODE-15-like.
    assert soft_label(1, "unknown", "x", cfg) == pytest.approx(0.8)
    # Disabled -> the hard label straight through.
    hard = LabelCfg(soft_labels=False)
    assert soft_label(1, "code15", "x", hard) == 1.0
    assert soft_label(None, "code15", "x", cfg) is None


def test_multi_report_three_of_five():
    """Hong et al. 2025 eq. 1: a patient with 3 of 5 exams positive -> 0.804 / 0.294."""
    table = Code15Table(patient_of={f"e{i}": "p1" for i in range(5)})
    records = [(f"e{i}", 1 if i < 3 else 0, "code15") for i in range(5)]
    stats = MultiReportStats.build(records, table)
    cfg = LabelCfg(multi_report=True, multi_report_alpha=0.51)
    assert soft_label(1, "code15", "e0", cfg, table, stats) == pytest.approx(0.804)
    assert soft_label(0, "code15", "e4", cfg, table, stats) == pytest.approx(0.294)
    # A patient with a single exam keeps the flat 0.8 / 0.2.
    solo_table = Code15Table(patient_of={"s1": "p9"})
    solo_stats = MultiReportStats.build([("s1", 1, "code15")], solo_table)
    assert soft_label(1, "code15", "s1", cfg, solo_table, solo_stats) == pytest.approx(0.8)


def test_rbbb_schemes():
    table = Code15Table(
        patient_of={"e1": "p1"},
        flags={"e1": {"RBBB": True, "1dAVb": True, "AF": False, "SB": False}},
        columns=("RBBB", "1dAVb", "AF", "SB"),
    )
    hong = LabelCfg(multi_report=False, rbbb_scheme="hong_rbbb")
    # negative with RBBB + AVb -> 0.2 + 0.25, capped at 0.49
    assert soft_label(0, "code15", "e1", hong, table) == pytest.approx(0.45)
    # positives are untouched by Hong's bump
    assert soft_label(1, "code15", "e1", hong, table) == pytest.approx(0.8)
    # the team's own scheme replaces the value entirely
    team = LabelCfg(multi_report=False, rbbb_scheme="team_rbbb")
    assert soft_label(1, "code15", "e1", team, table) == pytest.approx(0.95)
    assert soft_label(0, "code15", "e1", team, table) == pytest.approx(0.40)
    # unknown exam -> the scheme cannot fire
    assert soft_label(0, "code15", "nope", team, table) == pytest.approx(0.2)


def test_aux_labels_masked_when_exam_unknown():
    columns = ("RBBB", "1dAVb", "LBBB")
    table = Code15Table(
        flags={"e1": {"RBBB": True, "1dAVb": False, "LBBB": None}}, columns=columns
    )
    values, mask = aux_targets("e1", table, columns)
    assert values.tolist() == [1.0, 0.0, 0.0]
    assert mask.tolist() == [1.0, 1.0, 0.0], "an absent flag must be masked out"
    values, mask = aux_targets("unknown-exam", table, columns)
    assert mask.tolist() == [0.0, 0.0, 0.0]
    values, mask = aux_targets("e1", None, columns)
    assert mask.tolist() == [0.0, 0.0, 0.0]


def test_patient_grouping_and_split_never_split_a_patient():
    table = Code15Table(patient_of={f"e{i}": f"p{i // 3}" for i in range(30)})
    assert patient_group("e0", "code15", table) == patient_group("e2", "code15", table)
    assert patient_group("e0", "code15", table) != patient_group("e3", "code15", table)
    # A non-CODE-15 record, or an exam with no patient, is its own group.
    assert patient_group("x", "ptbxl", table).startswith("rec_")
    assert patient_group("nope", "code15", table).startswith("rec_")

    entries = [
        RecordEntry(
            path=f"/tmp/e{i}",
            record_id=f"e{i}",
            source="code15",
            label_hard=i % 2,
            group=patient_group(f"e{i}", "code15", table),
        )
        for i in range(30)
    ]
    train, val = split_entries(entries, 0.2, seed=0)
    assert len(train) + len(val) == len(entries)
    assert not (set(e.group for e in train) & set(e.group for e in val))
    assert val, "the holdout must not be empty"


def test_split_keeps_every_source_and_both_classes():
    entries = []
    for i in range(200):
        source = ["code15", "ptbxl", "samitrop"][i % 3]
        label = 1 if source == "samitrop" else (i % 7 == 0)
        entries.append(
            RecordEntry(
                path=f"/tmp/{source}{i}",
                record_id=f"{source}{i}",
                source=source,
                label_hard=int(label),
                group=f"g{i}",
            )
        )
    train, val = split_entries(entries, 0.1, seed=1)
    assert {e.source for e in val} == {"code15", "ptbxl", "samitrop"}
    assert any(e.label_hard == 1 for e in val)
    assert any(e.label_hard == 0 for e in val)



def test_training_scan_skips_unlabelled_headers(tmp_path):
    """A training header without a label is skipped and counted, never fatal (the
    organisers' robustness tests alter labels); a header with a label is kept."""
    from chagas.config import default_config
    from chagas.dataset import scan_records
    from chagas.labels import Code15Table

    def write(stem: str, header: str) -> str:
        (tmp_path / f"{stem}.hea").write_text(header)
        return str(tmp_path / stem)

    labelled = write("a", "a 12 400 2934\n" + "\n".join(f"a.mat 16 1000 0 0 0 0 0 l{i}" for i in range(12)) + "\n# Age: 50\n# Sex: Male\n# Source: CODE-15%\n# Chagas label: True\n")
    unlabelled = write("b", "b 12 400 2934\n" + "\n".join(f"b.mat 16 1000 0 0 0 0 0 l{i}" for i in range(12)) + "\n# Age: 50\n# Sex: Male\n# Source: CODE-15%\n")
    cfg = default_config()
    cfg.labels.multi_report = False
    table = Code15Table.empty(cfg.labels.aux_columns)
    entries = scan_records([labelled, unlabelled], cfg, table=table, workers=1)
    assert [e.record_id for e in entries] == ["a"]
