"""Target-free selectors used by the claim-bearing Table 1 pipeline."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping, Sequence


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def select_simple_control(
    rows: Sequence[Mapping[str, object]],
    *,
    expected_run_keys: Iterable[tuple[str, int]],
    controls: Sequence[str],
) -> dict[str, object]:
    """Select the strongest predeclared control on prediction development data."""
    expected = {(str(request), int(seed)) for request, seed in expected_run_keys}
    names = tuple(str(control) for control in controls)
    if not expected or not names or len(set(names)) != len(names):
        raise ValueError("expected cells and unique controls are required")

    by_control: dict[str, dict[tuple[str, int], float]] = defaultdict(dict)
    duplicate: set[tuple[str, tuple[str, int]]] = set()
    for index, row in enumerate(rows):
        if row.get("campaign_phase") != "development":
            raise ValueError("simple-control selection accepts development rows only")
        key = (str(row.get("request")), int(row.get("seed")))
        correlations = row.get("control_spearman")
        if not isinstance(correlations, Mapping):
            raise ValueError(f"rows[{index}].control_spearman must be a mapping")
        for control in names:
            value = _finite_number(
                correlations.get(control),
                f"rows[{index}].control_spearman.{control}",
            )
            if key in by_control[control]:
                duplicate.add((control, key))
            by_control[control][key] = value

    diagnostics = []
    eligible = []
    for control in names:
        observed = set(by_control[control])
        complete = observed == expected and not any(
            name == control for name, _key in duplicate
        )
        values = list(by_control[control].values())
        item = {
            "control": control,
            "complete": complete,
            "n_expected": len(expected),
            "n_observed": len(observed),
            "missing_runs": sorted(expected - observed),
            "extra_runs": sorted(observed - expected),
            "mean_spearman": sum(values) / len(values) if values else None,
        }
        diagnostics.append(item)
        if complete:
            eligible.append(item)
    if not eligible:
        return {
            "resolved": False,
            "control": None,
            "selection_rule": "mean_spearman_then_declared_order",
            "diagnostics": diagnostics,
        }
    winner = min(
        eligible,
        key=lambda item: (
            -float(item["mean_spearman"]),
            names.index(str(item["control"])),
        ),
    )
    return {
        "resolved": True,
        "control": winner["control"],
        "selection_rule": "mean_spearman_then_declared_order",
        "winner": winner,
        "diagnostics": diagnostics,
    }


def select_protection_pair(
    rows: Sequence[Mapping[str, object]],
    *,
    alpha_grid: Sequence[float],
    kp_grid: Sequence[int],
    expected_run_keys: Iterable[tuple[str, int]],
) -> dict[str, object]:
    """Select ``(alpha_prot, Kp)`` by complete-cell minimax CVaR.

    Every development request/seed must be present and satisfy all four
    feasibility constraints. No target row or best-effort incomplete pair can
    enter the selector.
    """
    alphas = tuple(_finite_number(value, "alpha_grid[]") for value in alpha_grid)
    kps = tuple(int(value) for value in kp_grid)
    expected = {(str(request), int(seed)) for request, seed in expected_run_keys}
    if (
        not alphas
        or len(set(alphas)) != len(alphas)
        or any(not 0.0 <= value <= 1.0 for value in alphas)
    ):
        raise ValueError("alpha_grid must contain unique values in [0, 1]")
    if (
        not kps
        or len(set(kps)) != len(kps)
        or any(value < 1 for value in kps)
    ):
        raise ValueError("kp_grid must contain unique positive integers")
    if not expected:
        raise ValueError("expected_run_keys must be non-empty")

    keyed: dict[tuple[float, int], dict[tuple[str, int], Mapping[str, object]]] = (
        defaultdict(dict)
    )
    duplicates: set[tuple[float, int, str, int]] = set()
    for index, row in enumerate(rows):
        if row.get("campaign_phase") != "development":
            raise ValueError("protection selection accepts development rows only")
        alpha = _finite_number(row.get("alpha"), f"rows[{index}].alpha")
        kp = int(row.get("Kp"))
        if alpha not in alphas or kp not in kps:
            raise ValueError(
                f"rows[{index}] contains an undeclared protection pair {(alpha, kp)}"
            )
        run_key = (str(row.get("request")), int(row.get("seed")))
        pair = (alpha, kp)
        if run_key in keyed[pair]:
            duplicates.add((alpha, kp, *run_key))
        keyed[pair][run_key] = row

    diagnostics = []
    eligible = []
    for alpha in alphas:
        for kp in kps:
            pair = (alpha, kp)
            cells = keyed.get(pair, {})
            observed = set(cells)
            complete = observed == expected and not any(
                item[:2] == pair for item in duplicates
            )
            feasible = complete and all(row.get("feasible") is True for row in cells.values())
            cvars = [
                _finite_number(row.get("cvar95_damage"), "cvar95_damage")
                for row in cells.values()
            ]
            means = [
                _finite_number(row.get("mean_damage"), "mean_damage")
                for row in cells.values()
            ]
            item = {
                "alpha": alpha,
                "Kp": kp,
                "complete": complete,
                "feasible": feasible,
                "n_expected": len(expected),
                "n_observed": len(observed),
                "missing_runs": sorted(expected - observed),
                "extra_runs": sorted(observed - expected),
                "worst_cvar95_damage": max(cvars) if cvars else None,
                "mean_cvar95_damage": sum(cvars) / len(cvars) if cvars else None,
                "mean_damage": sum(means) / len(means) if means else None,
            }
            diagnostics.append(item)
            if feasible:
                eligible.append(item)

    if not eligible:
        return {
            "resolved": False,
            "alpha": None,
            "Kp": None,
            "selection_rule": (
                "minimax_cvar95_then_mean_cvar_mean_damage_midpoint_smaller_kp"
            ),
            "diagnostics": diagnostics,
        }
    winner = min(
        eligible,
        key=lambda item: (
            float(item["worst_cvar95_damage"]),
            float(item["mean_cvar95_damage"]),
            float(item["mean_damage"]),
            abs(float(item["alpha"]) - 0.5),
            float(item["alpha"]),
            int(item["Kp"]),
        ),
    )
    return {
        "resolved": True,
        "alpha": float(winner["alpha"]),
        "Kp": int(winner["Kp"]),
        "selection_rule": (
            "minimax_cvar95_then_mean_cvar_mean_damage_midpoint_smaller_kp"
        ),
        "winner": winner,
        "diagnostics": diagnostics,
    }
