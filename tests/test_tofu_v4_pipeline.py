from __future__ import annotations

import ast
from pathlib import Path

import yaml

from experiments.paper.init_v4_stage import build_manifest


ROOT = Path(__file__).resolve().parents[1]


def test_tofu_unit_declares_the_complete_literal_contract():
    source = ROOT / "experiments/paper/tofu_v4_unit.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    marker = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id == "PAPER_UNIT_CONTRACT"
            for target in node.targets
        ):
            marker = ast.literal_eval(node.value)
            break
    assert marker is not None
    assert marker["adapters"] == ["tofu"]
    assert marker["runs_pdf_v4_repair"]
    assert marker["runs_all_comparator_arms"]
    assert marker["reuses_request_level_sft_cache"]
    assert marker["computes_exact_gradient_reference"]
    assert marker["emits_parent_selection_inputs"]
    assert marker["emits_dataset_native_retention"]


def test_stage_manifest_expands_the_exact_primary_cartesian_roster(tmp_path):
    campaign_path = ROOT / "configs/paper/campaign.yaml"
    evidence_path = ROOT / "configs/paper/evidence.yaml"
    runtime_path = ROOT / "configs/paper/tofu_v4.yaml"
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
    manifest = build_manifest(
        campaign,
        evidence,
        stage="target_evaluation",
        setting_id="tofu_qwen25_1p5b",
        campaign_path=campaign_path,
        evidence_path=evidence_path,
        runtime_path=runtime_path,
        unit_root=tmp_path / "units",
        python="python",
    )
    assert len(manifest["units"]) == 7 * 10 * 2
    keys = {
        (unit["parent"], unit["request"], unit["seed"])
        for unit in manifest["units"]
    }
    assert len(keys) == len(manifest["units"])
    assert all(
        set(unit["outputs"])
        == {"prediction_raw", "fidelity_raw", "protection_raw"}
        for unit in manifest["units"]
    )
    assert all("--selection-freeze" in unit["command"] for unit in manifest["units"])

    calibration = build_manifest(
        campaign,
        evidence,
        stage="calibration",
        setting_id="tofu_qwen25_1p5b",
        campaign_path=campaign_path,
        evidence_path=evidence_path,
        runtime_path=runtime_path,
        unit_root=tmp_path / "calibration-units",
        python="python",
    )
    assert len(calibration["units"]) == 7 * 2 * 2
    assert all(
        set(unit["outputs"])
        == {"fidelity_raw", "parent_selection_inputs"}
        for unit in calibration["units"]
    )


def test_stage_manifest_preserves_virtualenv_python_symlink(tmp_path):
    campaign_path = ROOT / "configs/paper/campaign.yaml"
    evidence_path = ROOT / "configs/paper/evidence.yaml"
    runtime_path = ROOT / "configs/paper/tofu_v4.yaml"
    campaign = yaml.safe_load(campaign_path.read_text(encoding="utf-8"))
    evidence = yaml.safe_load(evidence_path.read_text(encoding="utf-8"))
    base_python = tmp_path / "base-python"
    base_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python = tmp_path / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    manifest = build_manifest(
        campaign,
        evidence,
        stage="calibration",
        setting_id="tofu_qwen25_1p5b",
        campaign_path=campaign_path,
        evidence_path=evidence_path,
        runtime_path=runtime_path,
        unit_root=tmp_path / "units",
        python=str(venv_python),
    )

    assert {unit["command"][0] for unit in manifest["units"]} == {
        str(venv_python)
    }
