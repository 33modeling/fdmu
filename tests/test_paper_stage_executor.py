"""CPU-only contract tests for the exact PDF-v4 paper stage executor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
import yaml

from experiments.paper.run_v4_stage import StageContractError, execute


def _write_yaml(path: Path, value: dict) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def _identity() -> dict:
    return {
        "setting": "primary",
        "parent": "npo",
        "request": "tofu-a180",
        "seed": "2025",
    }


def _contracts(tmp_path: Path) -> tuple[Path, Path]:
    campaign = {
        "campaign_id": "test-v4",
        "execution": {
            "seeds": [2025],
            "repeated_random_draws": ["rand-000"],
        },
        "stages": {
            "target_evaluation": {
                "roster": "target",
                "adapter_capability": "target_evaluation",
            }
        },
        "datasets": {
            "TOFU": {
                "adapter": "tofu",
                "rosters": {"target": ["tofu-a180"]},
            }
        },
    }
    evidence = {
        "settings": [
            {
                "id": "primary",
                "dataset": "TOFU",
                "model": "M0",
                "parents": ["npo"],
            }
        ]
    }
    campaign_path = tmp_path / "campaign.yaml"
    evidence_path = tmp_path / "evidence.yaml"
    _write_yaml(campaign_path, campaign)
    _write_yaml(evidence_path, evidence)
    return campaign_path, evidence_path


def _raw_shards(tmp_path: Path) -> dict[str, str]:
    prediction = []
    for index in range(2):
        prediction.append(
            {
                **_identity(),
                "candidate_id": f"c{index}",
                "group": f"g{index}",
                "s0": float(index),
                "s1": float(1 - index),
                "joint": float(index),
                "simple_control": float(1 - index),
                "damage": float(index),
                "profile_valid": True,
                "reached": True,
                "trajectory_completed": True,
                "parent_checkpoint_id": "first-reaching-10",
                "parent_checkpoint_first_reaching": True,
                "prediction_selection": {
                    "valid": True,
                    "fallback": False,
                    "alpha": 0.4,
                },
            }
        )
    fidelity = [
        {
            **_identity(),
            "f_rho": 0.9,
            "f_k": 0.8,
            "perturbations_valid": True,
            "exact_reference_valid": True,
            "common_control_support": True,
        }
    ]
    protection = []
    for candidate in ("c0", "c1"):
        for arm in ("joint", "no_repair", "s0", "s1", "repeated_random"):
            row = {
                **_identity(),
                "candidate_id": candidate,
                "group": candidate,
                "arm": arm,
                "damage": 0.1,
                "native_retention": 0.9,
                "feasible": True,
                "direct_forget_margin": 0.1,
                "paraphrase_forget_margin": 0.1,
                "extraction_generation_margin": 0.1,
                "utility_margin": 0.1,
                "parent_checkpoint_id": "first-reaching-10",
                "parent_checkpoint_first_reaching": True,
                "protection_selection": {
                    "valid": True,
                    "fallback": False,
                    "alpha": 0.6,
                },
            }
            if arm == "repeated_random":
                row.update({"draw_id": "rand-000", "draw_complete": True})
            protection.append(row)
    paths = {
        "prediction_raw": tmp_path / "prediction.jsonl",
        "fidelity_raw": tmp_path / "fidelity.jsonl",
        "protection_raw": tmp_path / "protection.jsonl",
    }
    _write_jsonl(paths["prediction_raw"], prediction)
    _write_jsonl(paths["fidelity_raw"], fidelity)
    _write_jsonl(paths["protection_raw"], protection)
    return {key: path.name for key, path in paths.items()}


def _args(
    campaign: Path, evidence: Path, manifest: Path, output: Path
) -> argparse.Namespace:
    return argparse.Namespace(
        campaign=str(campaign),
        evidence=str(evidence),
        manifest=str(manifest),
        output_dir=str(output),
        action="verify",
    )


def test_target_stage_seals_exact_candidate_level_raw(tmp_path):
    campaign, evidence = _contracts(tmp_path)
    outputs = _raw_shards(tmp_path)
    manifest = tmp_path / "manifest.yaml"
    _write_yaml(
        manifest,
        {
            "schema_version": 1,
            "contract": "kdd-unlearning-fail-pdf-v4-stage",
            "status": "frozen",
            "campaign_id": "test-v4",
            "stage": "target_evaluation",
            "setting": "primary",
            "units": [{**{key: _identity()[key] for key in ("parent", "request", "seed")}, "outputs": outputs}],
        },
    )
    summary_path = execute(
        _args(campaign, evidence, manifest, tmp_path / "sealed")
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["units"] == 1
    assert set(summary["artifacts"]) == {
        "prediction_raw",
        "fidelity_raw",
        "protection_raw",
    }
    assert all(len(item["sha256"]) == 64 for item in summary["artifacts"].values())


def test_target_stage_rejects_incomplete_repeated_random_roster(tmp_path):
    campaign, evidence = _contracts(tmp_path)
    outputs = _raw_shards(tmp_path)
    protection_path = tmp_path / outputs["protection_raw"]
    rows = [
        json.loads(line)
        for line in protection_path.read_text(encoding="utf-8").splitlines()
        if '"repeated_random"' not in line
    ]
    _write_jsonl(protection_path, rows)
    manifest = tmp_path / "manifest.yaml"
    _write_yaml(
        manifest,
        {
            "schema_version": 1,
            "contract": "kdd-unlearning-fail-pdf-v4-stage",
            "status": "frozen",
            "campaign_id": "test-v4",
            "stage": "target_evaluation",
            "setting": "primary",
            "units": [{**{key: _identity()[key] for key in ("parent", "request", "seed")}, "outputs": outputs}],
        },
    )
    with pytest.raises(StageContractError, match="five-arm/repeated-draw"):
        execute(_args(campaign, evidence, manifest, tmp_path / "sealed"))
