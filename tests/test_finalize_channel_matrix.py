from __future__ import annotations

import json
from pathlib import Path

from experiments.paper.export_channel_matrix_raw import (
    _artifact_reference,
    _loss_shake_profile_valid,
)
from experiments.paper.finalize_channel_matrix import (
    _build_plan,
    _load_yaml,
    _materialize_final_latex,
    _prediction_parents,
    _prediction_roster,
    _protection_selections,
    _remove_obsolete_latex,
)
from experiments.paper.publish_evidence import merge_ledgers, publish
from rsus.evidence.raw import raw_plan_from_mapping


ROOT = Path(__file__).resolve().parents[1]


def test_artifact_reference_accepts_external_campaign_root(tmp_path):
    campaign_root = tmp_path / "group-volume" / "fdmu" / "runs" / "campaign"
    damage = campaign_root / "audit" / "qwen25_7b" / "seed-0" / "damage.json"

    assert (
        _artifact_reference(damage, campaign_root)
        == "audit/qwen25_7b/seed-0/damage.json"
    )


def test_7b_backfill_plan_keeps_all_parents_and_marks_fallbacks():
    config_path = ROOT / "configs/channel_matrix/7b_tofu.yaml"
    cfg = _load_yaml(config_path)
    freeze = _load_yaml(
        ROOT / "configs/channel_matrix/prediction_alpha_freeze_7b.yaml"
    )
    frozen_parents, _ = _prediction_parents(cfg, freeze)
    parents, selections = _prediction_roster(cfg, freeze)
    protection, frozen, _ = _protection_selections(
        cfg, config_path, "qwen25_7b", parents
    )
    mapping = _build_plan(
        cfg,
        setting_id="tofu_qwen25_7b",
        parents=parents,
        prediction_selections=selections,
        protection=protection,
        control="knn_lexical",
        bootstrap_replicates=7,
    )
    plan = raw_plan_from_mapping(mapping)

    assert frozen_parents == ["graddiff", "rmu"]
    assert parents == [
        "graddiff",
        "rmu",
        "npo",
        "simnpo",
        "gru",
        "repnoise",
        "circuit_breakers",
    ]
    assert len(plan.units) == 42
    assert frozen is False
    by_parent = {
        parent: next(
            unit.prediction_selection
            for unit in plan.units.values()
            if unit.key[1] == parent
        )
        for parent in parents
    }
    assert by_parent["graddiff"].valid
    assert by_parent["graddiff"].alpha == 1.0
    assert by_parent["rmu"].valid
    assert by_parent["rmu"].alpha == 0.75
    for parent in ("npo", "simnpo", "gru", "repnoise", "circuit_breakers"):
        assert not by_parent[parent].valid
        assert by_parent[parent].fallback
        assert by_parent[parent].alpha == 0.5
    assert all(unit.protection_selection.fallback for unit in plan.units.values())


def test_profile_integrity_does_not_require_external_certificate():
    responses = [
        {"d0": 1.0, "d1": 2.0, "sealed-audit": 99.0},
        {"d0": 2.0, "d1": 4.0, "sealed-audit": 98.0},
        {"d0": 3.0, "d1": 6.0, "sealed-audit": 97.0},
        {"d0": 4.0, "d1": 8.0, "sealed-audit": 96.0},
    ]
    dimension = 10
    scores = {
        candidate: dimension
        * sum(row[candidate] ** 2 for row in responses)
        / len(responses)
        for candidate in ("d0", "d1")
    }
    profile = {
        "probe": {
            "norm_eta": 0.003,
            "direction_count": 4,
            "direction_seed": 0,
        },
        "artifacts": {
            "schema": "loss-shake-responses-v1",
            "direction_responses": responses,
            "direction_count": 4,
            "direction_seed": 0,
            "eta": 0.003,
            "block_dimension": dimension,
        },
    }

    assert _loss_shake_profile_valid(profile, None, scores)


def test_shared_ledger_preserves_other_settings_and_replaces_same_key():
    existing = {
        "schema_version": 2,
        "rows": [
            {"setting": "tofu_qwen25_1p5b", "parent": "graddiff"},
            {
                "setting": "tofu_qwen25_7b",
                "parent": "graddiff",
                "prediction_selection": {
                    "valid": False,
                    "fallback": False,
                    "alpha": 0.25,
                },
            },
        ],
        "artifacts": {},
    }
    incoming = {
        "schema_version": 2,
        "rows": [
            {
                "setting": "tofu_qwen25_7b",
                "parent": "graddiff",
                "prediction_selection": {
                    "valid": False,
                    "fallback": False,
                    "alpha": 0.75,
                },
            },
            {"setting": "tofu_qwen25_7b", "parent": "rmu"},
        ],
        "artifacts": {},
    }

    merged = merge_ledgers(existing, incoming)
    rows = {
        (row["setting"], row["parent"]): row
        for row in merged["rows"]
    }

    assert set(rows) == {
        ("tofu_qwen25_1p5b", "graddiff"),
        ("tofu_qwen25_7b", "graddiff"),
        ("tofu_qwen25_7b", "rmu"),
    }
    assert (
        rows[("tofu_qwen25_7b", "graddiff")]["prediction_selection"]["alpha"]
        == 0.75
    )


def test_publish_writes_only_complete_roster_table1_and_table2(tmp_path):
    ledger_path = tmp_path / "incoming.json"
    ledger_path.write_text(
        json.dumps({"schema_version": 2, "rows": [], "artifacts": {}}),
        encoding="utf-8",
    )
    output = tmp_path / "paper_v4"
    output.mkdir()
    (output / "table1.tex").write_text("obsolete", encoding="utf-8")
    (output / "table2.tex").write_text("obsolete", encoding="utf-8")

    paths = publish(
        ledger_path=ledger_path,
        combined_root=output,
        evidence_config=ROOT / "configs/paper/evidence.yaml",
        print_tables=False,
    )

    table1 = Path(paths["table1"]).read_text(encoding="utf-8")
    table2 = Path(paths["table2"]).read_text(encoding="utf-8")
    assert sorted(path.name for path in output.glob("*.tex")) == [
        "table_core_evidence.tex",
        "table_robustness.tex",
    ]
    for parent in ("GradDiff", "NPO", "SimNPO", "GRU", "RMU", "RepNoise", "CB"):
        assert parent in table1
    for setting in (
        "held-out TOFU requests",
        "WMDP-bio/MMLU",
        "MUSE-News",
        "RWKU",
        "MUSE-Books (stress)",
        "PISTOL (stress)",
        "Qwen2.5-1.5B (boundary)",
        "Qwen2.5-14B",
        "Llama-3.1-8B",
    ):
        assert setting in table2
    assert r"\label{tab:pred-value}" in table1
    assert r"\label{tab:prot-contract}" in table1
    assert r"\label{tab:robustness}" in table2
    assert r"\label{tab:robustness-funnel}" in table2


def test_finalizer_removes_pre_unified_latex_duplicates(tmp_path):
    names = (
        "table1.tex",
        "table2.tex",
        "table1_core_evidence_qwen25_7b.tex",
        "table2_robustness_qwen25_7b.tex",
    )
    for name in names:
        (tmp_path / name).write_text("obsolete", encoding="utf-8")
    keep = tmp_path / "evidence_ledger.json"
    keep.write_text("{}", encoding="utf-8")

    removed = _remove_obsolete_latex(tmp_path, "qwen25_7b")

    assert {Path(path).name for path in removed} == set(names)
    assert all(not (tmp_path / name).exists() for name in names)
    assert keep.is_file()


def test_finalizer_materializes_shared_tables_in_per_run_directory(tmp_path):
    shared = tmp_path / "shared"
    per_run = tmp_path / "run" / "aggregate" / "paper_v4"
    shared.mkdir()
    per_run.mkdir(parents=True)
    core = shared / "table_core_evidence.tex"
    robustness = shared / "table_robustness.tex"
    core.write_text("core", encoding="utf-8")
    robustness.write_text("robustness", encoding="utf-8")

    outputs = _materialize_final_latex(
        per_run,
        {"table1": str(core), "table2": str(robustness)},
    )

    assert Path(outputs["table1"]).is_file()
    assert Path(outputs["table2"]).is_file()
    assert not Path(outputs["table1"]).is_symlink()
    assert not Path(outputs["table2"]).is_symlink()
    assert Path(outputs["table1"]).read_text(encoding="utf-8") == "core"
    assert Path(outputs["table2"]).read_text(encoding="utf-8") == "robustness"
