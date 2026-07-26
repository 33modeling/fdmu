from __future__ import annotations

from pathlib import Path

from experiments.paper.export_channel_matrix_raw import (
    _artifact_reference,
    _loss_shake_profile_valid,
)
from experiments.paper.finalize_channel_matrix import (
    _build_plan,
    _load_yaml,
    _prediction_parents,
    _protection_selections,
)
from experiments.paper.publish_evidence import merge_ledgers
from rsus.evidence.raw import raw_plan_from_mapping


ROOT = Path(__file__).resolve().parents[1]


def test_artifact_reference_accepts_external_campaign_root(tmp_path):
    campaign_root = tmp_path / "group-volume" / "fdmu" / "runs" / "campaign"
    damage = campaign_root / "audit" / "qwen25_7b" / "seed-0" / "damage.json"

    assert (
        _artifact_reference(damage, campaign_root)
        == "audit/qwen25_7b/seed-0/damage.json"
    )


def test_7b_backfill_plan_uses_only_frozen_prediction_parents():
    config_path = ROOT / "configs/channel_matrix/7b_tofu.yaml"
    cfg = _load_yaml(config_path)
    freeze = _load_yaml(
        ROOT / "configs/channel_matrix/prediction_alpha_freeze_7b.yaml"
    )
    parents, alphas = _prediction_parents(cfg, freeze)
    protection, frozen, _ = _protection_selections(
        cfg, config_path, "qwen25_7b", parents
    )
    mapping = _build_plan(
        cfg,
        setting_id="tofu_qwen25_7b",
        parents=parents,
        prediction_alphas=alphas,
        protection=protection,
        control="knn_lexical",
        bootstrap_replicates=7,
    )
    plan = raw_plan_from_mapping(mapping)

    assert parents == ["graddiff", "rmu"]
    assert len(plan.units) == 12
    assert frozen is False
    assert all(unit.prediction_selection.valid for unit in plan.units.values())
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
