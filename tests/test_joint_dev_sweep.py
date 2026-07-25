from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from experiments.paper.run_joint_dev_sweep import (
    SweepError,
    evaluate_cell,
    evaluate_trial,
    validate_spec,
)


DRAWS = ["rand-000", "rand-001"]


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
                "sentence_encoder",
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


def test_cell_requires_feasible_joint_and_both_damage_wins():
    result = evaluate_cell(
        _diagnostic(),
        expected_draws=DRAWS,
        mean_margin=0.0,
        cvar_margin=0.0,
    )
    assert result["passed"]

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
