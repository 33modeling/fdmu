from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from experiments.paper.approve_parent_freeze import (
    ApprovalError,
    validate_draft,
)
from experiments.paper.run_joint_dev_sweep import (
    SweepError,
    _absolute_executable,
    _best_parent_freeze,
    _unit_complete,
    _with_parent_freeze,
    build_exhaustion_report,
    candidate_score,
    evaluate_cell,
    evaluate_trial,
    validate_spec,
)


def _artifact(path: Path) -> dict[str, str]:
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


DRAWS = ["rand-000", "rand-001"]


def test_executable_path_keeps_virtualenv_symlink(tmp_path):
    base_python = tmp_path / "base-python"
    base_python.write_text("#!/bin/sh\n", encoding="utf-8")
    venv_python = tmp_path / ".venv/bin/python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(base_python)

    resolved = _absolute_executable(".venv/bin/python", base=tmp_path)

    assert resolved == venv_python
    assert resolved.is_symlink()
    assert resolved != base_python.resolve()


def _arm(name, mean, cvar, *, feasible=True, draw=None):
    return {
        "arm": name,
        "draw_id": draw,
        "metrics": {"feasible": feasible},
        "mean_damage": mean,
        "cvar95_damage": cvar,
    }


def _diagnostic(parent="graddiff", request="tofu-a184", seed=2025):
    return {
        "parent": parent,
        "request": request,
        "seed": seed,
        "arms": [
            _arm("joint", 1.0, 2.0),
            _arm("no_repair", 1.3, 2.3),
            _arm("s0", 1.4, 2.4),
            _arm("s1", 1.5, 2.5),
            _arm("repeated_random", 1.6, 2.6, draw="rand-000"),
            _arm("repeated_random", 1.7, 2.7, draw="rand-001"),
        ],
    }


def _spec():
    return {
        "schema_version": 1,
        "contract": "tofu-pdf-v4-joint-dev-sweep",
        "development_only": True,
        "setting": "tofu_qwen25_1p5b",
        "paths": {
            key: key
            for key in (
                "campaign",
                "evidence",
                "runtime",
                "python",
                "model_source",
                "sft_cache_root",
                "output_root",
            )
        },
        "gpus": [0, 1],
        "stop": {
            "require_all_development_cells": True,
            "require_joint_feasible": True,
            "repeated_random_rule": "beat_every_draw",
            "mean_damage_margin": 0.0,
            "cvar95_damage_margin": 0.0,
            "parent_groups": {
                "output": {
                    "members": ["graddiff"],
                    "minimum_passing": 1,
                }
            },
        },
        "budget": {
            "maximum_trials": 1,
            "run_all_declared_trials": True,
        },
        "trials": [
            {
                "id": "baseline",
                "alpha": 0.5,
                "Kp": 40,
                "repair": {"step_size": 3e-5},
            }
        ],
    }


def test_spec_is_development_only_and_trial_ids_are_unique():
    assert validate_spec(_spec())["development_only"] is True
    bad = _spec()
    bad["development_only"] = False
    with pytest.raises(SweepError, match="development_only"):
        validate_spec(bad)

    duplicate = _spec()
    duplicate["trials"].append(deepcopy(duplicate["trials"][0]))
    with pytest.raises(SweepError, match="duplicate trial"):
        validate_spec(duplicate)


def test_spec_rejects_unreviewed_repair_keys():
    bad = _spec()
    bad["trials"][0]["repair"]["target_score"] = 0.1
    with pytest.raises(SweepError, match="unsupported keys"):
        validate_spec(bad)


def test_spec_requires_an_explicit_complete_trial_budget():
    bad = _spec()
    bad["budget"]["maximum_trials"] = 2
    with pytest.raises(SweepError, match="maximum_trials"):
        validate_spec(bad)


def test_cell_requires_feasible_joint_and_both_damage_wins():
    result = evaluate_cell(
        _diagnostic(),
        expected_draws=DRAWS,
        mean_margin=0.0,
        cvar_margin=0.0,
    )
    assert result["passed"]
    assert result["comparisons"][0]["mean_advantage"] > 0.0
    assert result["comparisons"][0]["cvar95_advantage"] > 0.0

    infeasible = _diagnostic()
    infeasible["arms"][0]["metrics"]["feasible"] = False
    assert not evaluate_cell(
        infeasible,
        expected_draws=DRAWS,
        mean_margin=0.0,
        cvar_margin=0.0,
    )["passed"]

    loses_cvar = _diagnostic()
    loses_cvar["arms"][2]["cvar95_damage"] = 1.9
    assert not evaluate_cell(
        loses_cvar,
        expected_draws=DRAWS,
        mean_margin=0.0,
        cvar_margin=0.0,
    )["passed"]

    infeasible_competitor = _diagnostic()
    infeasible_competitor["arms"][1]["metrics"]["feasible"] = False
    infeasible_competitor["arms"][1]["mean_damage"] = 0.1
    infeasible_competitor["arms"][1]["cvar95_damage"] = 0.1
    constrained = evaluate_cell(
        infeasible_competitor,
        expected_draws=DRAWS,
        mean_margin=0.0,
        cvar_margin=0.0,
    )
    assert constrained["passed"]
    assert constrained["comparisons"][0]["competitor_feasible"] is False
    assert constrained["comparisons"][0]["joint_wins_constrained"] is True


def test_cell_rejects_missing_random_draw():
    diagnostic = _diagnostic()
    diagnostic["arms"].pop()
    with pytest.raises(SweepError, match="roster mismatch"):
        evaluate_cell(
            diagnostic,
            expected_draws=DRAWS,
            mean_margin=0.0,
            cvar_margin=0.0,
        )


def test_trial_requires_every_cell_and_each_parent_group():
    stop = validate_spec(_spec())["stop"]
    diagnostics = [
        _diagnostic(parent=parent, request=request, seed=seed)
        for parent in ("graddiff", "rmu")
        for request in ("tofu-a184", "tofu-a185")
        for seed in (2025, 2026)
    ]
    stop["parent_groups"]["representation"] = {
        "members": ["rmu"],
        "minimum_passing": 1,
    }
    result = evaluate_trial(
        diagnostics,
        parents=["graddiff", "rmu"],
        requests=["tofu-a184", "tofu-a185"],
        seeds=[2025, 2026],
        expected_draws=DRAWS,
        stop=stop,
    )
    assert result["passed"]
    assert result["groups"]["output"]["passing_parents"] == ["graddiff"]
    assert result["groups"]["representation"]["passing_parents"] == ["rmu"]

    diagnostics[-1]["arms"][0]["mean_damage"] = 9.0
    failed = evaluate_trial(
        diagnostics,
        parents=["graddiff", "rmu"],
        requests=["tofu-a184", "tofu-a185"],
        seeds=[2025, 2026],
        expected_draws=DRAWS,
        stop=stop,
    )
    assert not failed["passed"]
    assert not failed["parents"]["rmu"]["passed"]

    failed.update({"trial_id": "failed", "trial_dir": "/tmp/failed"})
    score = candidate_score(failed)
    assert score["cells_passed"] == 7
    assert score["blocking_count"] >= 1


def test_exhaustion_report_selects_the_closest_failed_trial():
    stop = validate_spec(_spec())["stop"]
    near = evaluate_trial(
        [_diagnostic()],
        parents=["graddiff"],
        requests=["tofu-a184"],
        seeds=[2025],
        expected_draws=DRAWS,
        stop=stop,
    )
    near["cells"]["graddiff/tofu-a184/seed-2025"]["joint_feasible"] = False
    near["passed"] = False
    near.update({"trial_id": "near", "trial_dir": "/tmp/near"})

    far_diagnostic = _diagnostic()
    far_diagnostic["arms"][0]["mean_damage"] = 10.0
    far_diagnostic["arms"][0]["cvar95_damage"] = 10.0
    far = evaluate_trial(
        [far_diagnostic],
        parents=["graddiff"],
        requests=["tofu-a184"],
        seeds=[2025],
        expected_draws=DRAWS,
        stop=stop,
    )
    far.update({"trial_id": "far", "trial_dir": "/tmp/far"})
    report = build_exhaustion_report(
        [far, near],
        setting="tofu_qwen25_1p5b",
        spec_sha256="a" * 64,
    )
    assert report["status"] == "no_joint_dominance"
    assert report["exit_code"] == 3
    assert report["closest_candidate"]["trial_id"] == "near"
    assert report["target_used"] is False


def test_paper_runtime_registers_the_14b_scale_setting():
    root = Path(__file__).resolve().parents[1]
    runtime = yaml.safe_load(
        (root / "configs/paper/tofu_v4.yaml").read_text(encoding="utf-8")
    )
    setting = runtime["settings"]["tofu_qwen25_14b"]
    assert setting["model"] == "Qwen2.5-14B"
    assert setting["channel_model_id"] == "qwen25_14b"
    assert setting["parent_freeze"].endswith("objective_freeze_14b.yaml")
    assert setting["sft"]["steps"] == 800


def test_parent_freeze_override_is_absolute_and_does_not_mutate_source(tmp_path):
    freeze = tmp_path / "parent_freeze.yaml"
    freeze.write_text("status: frozen\n", encoding="utf-8")
    runtime = {
        "settings": {
            "tofu_qwen25_1p5b": {
                "parent_freeze": "configs/paper/old.yaml",
            }
        }
    }
    resolved = _with_parent_freeze(
        runtime, "tofu_qwen25_1p5b", freeze
    )
    assert (
        resolved["settings"]["tofu_qwen25_1p5b"]["parent_freeze"]
        == str(freeze.resolve())
    )
    assert (
        runtime["settings"]["tofu_qwen25_1p5b"]["parent_freeze"]
        == "configs/paper/old.yaml"
    )


def test_existing_best_exposes_its_resolved_parent_freeze(tmp_path):
    freeze = tmp_path / "parent_freeze.yaml"
    freeze.write_text("status: frozen\n", encoding="utf-8")
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        yaml.safe_dump(
            {
                "settings": {
                    "tofu_qwen25_1p5b": {
                        "parent_freeze": str(freeze),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    assert _best_parent_freeze(
        {"recommended_runtime": str(runtime)},
        "tofu_qwen25_1p5b",
    ) == freeze.resolve()


def test_calibration_resume_requires_hashed_outputs_and_fidelity(tmp_path):
    fidelity = tmp_path / "fidelity_raw.jsonl"
    selection = tmp_path / "parent_selection_inputs.jsonl"
    diagnostics = tmp_path / "fidelity_diagnostics.json"
    profile = tmp_path / "profiles.json"
    score_manifest = tmp_path / "score_independent_manifest.json"
    for path in (fidelity, selection, diagnostics, profile, score_manifest):
        path.write_text("{}\n", encoding="utf-8")
    hashes = {
        "campaign_config_sha256": "a" * 64,
        "evidence_config_sha256": "b" * 64,
        "runtime_config_sha256": "c" * 64,
    }
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "contract": "tofu-pdf-v4-unit-output",
                "stage": "calibration",
                "setting": "tofu_qwen25_1p5b",
                "parent": "graddiff",
                "request": "tofu-a198",
                "seed": 2025,
                **hashes,
                "outputs": {
                    "fidelity_raw": _artifact(fidelity),
                    "parent_selection_inputs": _artifact(selection),
                },
                "profile_artifact": _artifact(profile),
                "score_independent_manifest": _artifact(score_manifest),
                "fidelity_diagnostics": _artifact(diagnostics),
            }
        ),
        encoding="utf-8",
    )
    unit = {
        "setting": "tofu_qwen25_1p5b",
        "parent": "graddiff",
        "request": "tofu-a198",
        "seed": "2025",
        "run_manifest": str(run_manifest),
        "outputs": {
            "fidelity_raw": str(fidelity),
            "parent_selection_inputs": str(selection),
        },
    }
    assert _unit_complete(
        unit,
        campaign_hash=hashes["campaign_config_sha256"],
        evidence_hash=hashes["evidence_config_sha256"],
        runtime_hash=hashes["runtime_config_sha256"],
        stage="calibration",
    )
    diagnostics.write_text('{"changed": true}\n', encoding="utf-8")
    assert not _unit_complete(
        unit,
        campaign_hash=hashes["campaign_config_sha256"],
        evidence_hash=hashes["evidence_config_sha256"],
        runtime_hash=hashes["runtime_config_sha256"],
        stage="calibration",
    )


@pytest.mark.parametrize(
    ("stage", "corrupt_field"),
    (
        ("prediction", "profile_artifact"),
        ("prediction", "score_independent_manifest"),
        ("protection", "protection_diagnostics"),
        ("protection", "parent_freeze"),
        ("target_evaluation", "fidelity_diagnostics"),
        ("target_evaluation", "protection_diagnostics"),
        ("target_evaluation", "selection_freeze"),
    ),
)
def test_resume_rejects_every_stage_contract_artifact(
    tmp_path, stage, corrupt_field
):
    output_names = {
        "prediction": ("prediction_raw", "selection_inputs"),
        "protection": ("protection_raw", "selection_inputs"),
        "target_evaluation": (
            "prediction_raw",
            "fidelity_raw",
            "protection_raw",
        ),
    }[stage]
    files = {}
    for name in (
        *output_names,
        "profile_artifact",
        "score_independent_manifest",
        "fidelity_diagnostics",
        "protection_diagnostics",
        "selection_freeze",
        "parent_freeze",
    ):
        path = tmp_path / f"{name}.json"
        path.write_text("{}\n", encoding="utf-8")
        files[name] = path
    hashes = {
        "campaign_config_sha256": "a" * 64,
        "evidence_config_sha256": "b" * 64,
        "runtime_config_sha256": "c" * 64,
    }
    payload = {
        "schema_version": 1,
        "contract": "tofu-pdf-v4-unit-output",
        "stage": stage,
        "setting": "tofu_qwen25_1p5b",
        "parent": "graddiff",
        "request": "tofu-a188",
        "seed": 2025,
        **hashes,
        "outputs": {name: _artifact(files[name]) for name in output_names},
        "profile_artifact": _artifact(files["profile_artifact"]),
        "score_independent_manifest": _artifact(
            files["score_independent_manifest"]
        ),
        "fidelity_diagnostics": (
            _artifact(files["fidelity_diagnostics"])
            if stage == "target_evaluation"
            else None
        ),
        "protection_diagnostics": (
            _artifact(files["protection_diagnostics"])
            if stage in {"protection", "target_evaluation"}
            else None
        ),
        "selection_freeze": (
            _artifact(files["selection_freeze"])
            if stage == "target_evaluation"
            else None
        ),
        "parent_freeze": str(files["parent_freeze"]),
        "parent_freeze_sha256": hashlib.sha256(
            files["parent_freeze"].read_bytes()
        ).hexdigest(),
    }
    run_manifest = tmp_path / "run_manifest.json"
    run_manifest.write_text(json.dumps(payload), encoding="utf-8")
    unit = {
        "setting": "tofu_qwen25_1p5b",
        "parent": "graddiff",
        "request": "tofu-a188",
        "seed": "2025",
        "run_manifest": str(run_manifest),
        "outputs": {name: str(files[name]) for name in output_names},
    }
    assert _unit_complete(
        unit,
        campaign_hash=hashes["campaign_config_sha256"],
        evidence_hash=hashes["evidence_config_sha256"],
        runtime_hash=hashes["runtime_config_sha256"],
        stage=stage,
    )
    files[corrupt_field].write_text('{"corrupt": true}\n', encoding="utf-8")
    assert not _unit_complete(
        unit,
        campaign_hash=hashes["campaign_config_sha256"],
        evidence_hash=hashes["evidence_config_sha256"],
        runtime_hash=hashes["runtime_config_sha256"],
        stage=stage,
    )


def test_parent_freeze_approval_rejects_unresolved_or_changed_draft(tmp_path):
    selection = tmp_path / "parent_selection_inputs.jsonl"
    selection.write_text("{}\n", encoding="utf-8")
    source = {
        "path": str(selection.resolve()),
        "sha256": hashlib.sha256(selection.read_bytes()).hexdigest(),
    }
    proposal = {
        "schema_version": 1,
        "contract": "tofu-pdf-v4-parent-freeze",
        "status": "draft",
        "freeze_id": "PENDING",
        "frozen_at_utc": None,
        "frozen_before_prediction": False,
        "parents": {"graddiff": {"lr": 1e-6, "steps": 120}},
        "unresolved": [],
        "development_artifacts": [source],
    }
    validate_draft(proposal, deepcopy(proposal), input_path=selection)

    unresolved = deepcopy(proposal)
    unresolved["unresolved"] = ["graddiff"]
    with pytest.raises(ApprovalError, match="unresolved"):
        validate_draft(unresolved, unresolved, input_path=selection)

    changed = deepcopy(proposal)
    changed["parents"]["graddiff"]["steps"] = 240
    with pytest.raises(ApprovalError, match="recomputation"):
        validate_draft(proposal, changed, input_path=selection)
