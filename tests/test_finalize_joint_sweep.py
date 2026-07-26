from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from experiments.paper.finalize_joint_sweep import (
    _validate_existing_freeze,
    _write_final_campaign,
    resolve_joint_winner,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_resolve_joint_winner_uses_sealed_trial_artifacts(tmp_path: Path) -> None:
    joint = tmp_path / "joint_sweep"
    trial = joint / "trials" / "winner"
    campaign = trial / "config" / "campaign.local.yaml"
    runtime = trial / "config" / "tofu_v4.local.yaml"
    evidence = tmp_path / "evidence.yaml"
    comparison = trial / "joint_comparison.json"
    protection = trial / "stage" / "selection_inputs.jsonl"
    for path in (campaign, runtime, evidence, protection):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("value: true\n", encoding="utf-8")
    _write_json(comparison, {"passed": True})
    _write_json(
        trial / "trial.json",
        {
            "resolved_configs": {
                "campaign": str(campaign),
                "runtime": str(runtime),
            },
            "canonical_sources": {"evidence": str(evidence)},
        },
    )
    _write_json(
        joint / "BEST.json",
        {
            "status": "draft",
            "human_review_required": True,
            "target_used": False,
            "trial_dir": str(trial),
            "recommended_runtime": str(runtime),
            "joint_comparison": str(comparison),
        },
    )
    _write_json(
        joint / "SWEEP_STATUS.json",
        {
            "status": "joint_best",
            "terminal": True,
            "target_used": False,
            "trial_dir": str(trial),
        },
    )

    resolved = resolve_joint_winner(joint)

    assert resolved == {
        "trial_dir": trial.resolve(),
        "campaign": campaign.resolve(),
        "evidence": evidence.resolve(),
        "runtime": runtime.resolve(),
        "protection_input": protection.resolve(),
    }


def test_final_campaign_points_to_external_selection_freeze(
    tmp_path: Path,
) -> None:
    source = tmp_path / "campaign.yaml"
    destination = tmp_path / "final" / "campaign.final.yaml"
    freeze = tmp_path / "final" / "selection_freeze.yaml"
    source.write_text(
        yaml.safe_dump(
            {
                "campaign_id": "campaign-1",
                "execution": {"selection_freeze": "old.yaml"},
            }
        ),
        encoding="utf-8",
    )

    _write_final_campaign(source, destination, freeze)
    _write_final_campaign(source, destination, freeze)

    final = yaml.safe_load(destination.read_text(encoding="utf-8"))
    assert final["execution"]["selection_freeze"] == str(freeze.resolve())


def test_existing_freeze_must_hash_both_development_inputs(
    tmp_path: Path,
) -> None:
    prediction = tmp_path / "prediction.jsonl"
    protection = tmp_path / "protection.jsonl"
    prediction.write_text('{"kind": "prediction"}\n', encoding="utf-8")
    protection.write_text('{"kind": "protection"}\n', encoding="utf-8")
    freeze = tmp_path / "selection_freeze.yaml"
    freeze.write_text(
        yaml.safe_dump(
            {
                "status": "frozen",
                "frozen_before_target": True,
                "development_artifacts": [
                    {
                        "path": str(path.resolve()),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in (prediction, protection)
                ],
            }
        ),
        encoding="utf-8",
    )

    assert _validate_existing_freeze(freeze, prediction, protection) is True
