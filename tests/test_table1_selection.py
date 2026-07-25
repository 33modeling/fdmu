from __future__ import annotations

import pytest

from rsus.analysis.table1_selection import (
    select_protection_pair,
    select_simple_control,
)


def test_simple_control_requires_exact_development_cells():
    rows = [
        {
            "campaign_phase": "development",
            "request": request,
            "seed": seed,
            "control_spearman": {
                "initial_nll": 0.2,
                "answer_length": 0.4 if request == "r1" else 0.3,
            },
        }
        for request in ("r1", "r2")
        for seed in (1, 2)
    ]
    result = select_simple_control(
        rows,
        expected_run_keys={
            (request, seed)
            for request in ("r1", "r2")
            for seed in (1, 2)
        },
        controls=("initial_nll", "answer_length"),
    )
    assert result["resolved"]
    assert result["control"] == "answer_length"

    incomplete = select_simple_control(
        rows[:-1],
        expected_run_keys={
            (request, seed)
            for request in ("r1", "r2")
            for seed in (1, 2)
        },
        controls=("initial_nll", "answer_length"),
    )
    assert not incomplete["resolved"]


def test_protection_pair_is_all_cell_feasible_and_minimax():
    expected = {("r1", 1), ("r2", 1)}
    rows = []
    values = {
        (0.0, 20): (True, [0.9, 0.8]),
        (0.5, 20): (True, [0.7, 0.6]),
        (0.5, 40): (False, [0.1, 0.2]),
        (1.0, 20): (True, [0.8, 0.4]),
        (0.0, 40): (True, [1.0, 1.0]),
        (1.0, 40): (True, [0.9, 0.9]),
    }
    for (alpha, kp), (feasible, cvars) in values.items():
        for index, (request, seed) in enumerate(sorted(expected)):
            rows.append(
                {
                    "campaign_phase": "development",
                    "request": request,
                    "seed": seed,
                    "alpha": alpha,
                    "Kp": kp,
                    "feasible": feasible,
                    "cvar95_damage": cvars[index],
                    "mean_damage": cvars[index] / 2,
                }
            )
    result = select_protection_pair(
        rows,
        alpha_grid=(0.0, 0.5, 1.0),
        kp_grid=(20, 40),
        expected_run_keys=expected,
    )
    assert result["resolved"]
    assert result["alpha"] == pytest.approx(0.5)
    assert result["Kp"] == 20


def test_protection_selector_rejects_target_rows():
    with pytest.raises(ValueError, match="development"):
        select_protection_pair(
            [
                {
                    "campaign_phase": "target",
                    "request": "r1",
                    "seed": 1,
                    "alpha": 0.5,
                    "Kp": 20,
                    "feasible": True,
                    "cvar95_damage": 0.1,
                    "mean_damage": 0.1,
                }
            ],
            alpha_grid=(0.5,),
            kp_grid=(20,),
            expected_run_keys={("r1", 1)},
        )
