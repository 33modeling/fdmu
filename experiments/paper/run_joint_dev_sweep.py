#!/usr/bin/env python3
"""Run an append-only, development-only sweep until the joint arm is best."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paper.init_v4_stage import build_manifest  # noqa: E402


CONTRACT = "tofu-pdf-v4-joint-dev-sweep"
ARMS = ("joint", "no_repair", "s0", "s1")
REPAIR_KEYS = {
    "step_size",
    "beta",
    "momentum",
    "max_steps",
    "batch_size",
    "m_ref",
    "ridge_lambda",
    "kappa_tok",
    "kappa_ex",
    "epsilon_tok",
    "epsilon_ex",
    "max_retries",
    "retry_shrink",
    "token_budget",
    "save_every",
    "constraint_reduction",
}
TRIAL_ID_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class SweepError(ValueError):
    """The sweep contract or one of its artifacts is invalid."""


class HumanFreezeRequired(SweepError):
    """A validated parent freeze is required before further execution."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _status(message: str) -> None:
    print(f"[{_utc_now()}] {message}", flush=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SweepError(f"cannot read YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise SweepError(f"{path} must contain one mapping")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SweepError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise SweepError(f"{path} must contain one JSON object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise SweepError(f"append-only artifact changed: {path}")
        return
    _atomic_text(path, text)


def _write_or_rebind_manifest(
    path: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Refresh operational unit commands without weakening the frozen contract."""
    expected_copy = json.loads(json.dumps(expected))
    if not path.exists():
        _atomic_text(path, yaml.safe_dump(expected_copy, sort_keys=False))
        return expected_copy

    existing = _load_yaml(path)
    existing_units = existing.get("units")
    expected_units = expected_copy.get("units")
    if not isinstance(existing_units, list) or not isinstance(expected_units, list):
        raise SweepError(f"manifest has no unit list: {path}")

    def identity(unit: Mapping[str, Any]) -> tuple[str, str, str]:
        return (
            str(unit.get("parent")),
            str(unit.get("request")),
            str(unit.get("seed")),
        )

    expected_by_id = {
        identity(unit): unit
        for unit in expected_units
        if isinstance(unit, Mapping)
    }
    existing_ids = {
        identity(unit)
        for unit in existing_units
        if isinstance(unit, Mapping)
    }
    if len(expected_by_id) != len(expected_units) or existing_ids != set(expected_by_id):
        raise SweepError(f"existing manifest unit roster changed: {path}")

    rebound = json.loads(json.dumps(existing))
    old_interpreters = set()
    for unit in rebound["units"]:
        command = unit.get("command")
        if isinstance(command, list) and command:
            old_interpreters.add(str(command[0]))
        unit["command"] = list(expected_by_id[identity(unit)]["command"])

    if rebound != expected_copy:
        raise SweepError(f"append-only manifest contract changed beyond commands: {path}")
    if existing != expected_copy:
        _atomic_text(path, yaml.safe_dump(expected_copy, sort_keys=False))
        new_interpreters = sorted({
            str(unit["command"][0]) for unit in expected_units
        })
        _status(
            f"MANIFEST_COMMAND_REBOUND path={path} "
            f"old_python={sorted(old_interpreters)} new_python={new_interpreters}"
        )
    return expected_copy


def _recover_git_snapshot(source: Path, expected_sha256: str, destination: Path) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    if source.is_file() and _sha256(source) == expected_sha256:
        return source
    if destination.is_file() and _sha256(destination) == expected_sha256:
        return destination
    try:
        relative = source.relative_to(ROOT).as_posix()
    except ValueError as error:
        raise SweepError(f"changed config has no frozen snapshot: {source}") from error
    history = subprocess.run(
        ["git", "log", "--format=%H", "--", relative],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    for commit in history:
        blob = subprocess.run(
            ["git", "show", f"{commit}:{relative}"],
            cwd=ROOT,
            check=False,
            capture_output=True,
        )
        if blob.returncode or hashlib.sha256(blob.stdout).hexdigest() != expected_sha256:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(blob.stdout)
        os.replace(temporary, destination)
        _status(
            f"CONFIG_SNAPSHOT_RECOVERED commit={commit[:12]} "
            f"snapshot={destination} sha256={expected_sha256}"
        )
        return destination
    raise SweepError(
        f"cannot recover config sha256={expected_sha256} from Git history: {source}"
    )


def _snapshot_current_config(source: Path, destination: Path) -> Path:
    source = source.resolve()
    destination = destination.resolve()
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_bytes(source.read_bytes())
        os.replace(temporary, destination)
    return destination


def _resolve(value: str | Path, *, base: Path = ROOT) -> Path:
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    path = Path(expanded)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _absolute_executable(value: str | Path, *, base: Path = ROOT) -> Path:
    """Make an executable path absolute without dereferencing a venv symlink."""
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    path = Path(expanded)
    if not path.is_absolute():
        path = base / path
    return Path(os.path.abspath(path))


def _finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SweepError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SweepError(f"{name} must be finite")
    return number


def validate_spec(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize the target-free sweep specification."""
    if raw.get("schema_version") != 1 or raw.get("contract") != CONTRACT:
        raise SweepError(f"sweep spec must use schema_version 1 and contract {CONTRACT}")
    if raw.get("development_only") is not True:
        raise SweepError("development_only must be true")
    setting = raw.get("setting")
    if not isinstance(setting, str) or not setting:
        raise SweepError("setting must be a non-empty string")

    paths = raw.get("paths")
    required_paths = (
        "campaign",
        "evidence",
        "runtime",
        "python",
        "model_source",
        "sft_cache_root",
        "output_root",
    )
    if not isinstance(paths, Mapping) or any(
        not isinstance(paths.get(name), str) or not paths.get(name)
        for name in required_paths
    ):
        raise SweepError(f"paths must define {', '.join(required_paths)}")

    raw_gpus = raw.get("gpus")
    if (
        not isinstance(raw_gpus, list)
        or not raw_gpus
        or any(isinstance(gpu, bool) or not isinstance(gpu, int) or gpu < 0 for gpu in raw_gpus)
        or len(set(raw_gpus)) != len(raw_gpus)
    ):
        raise SweepError("gpus must contain unique non-negative integers")

    stop = raw.get("stop")
    if not isinstance(stop, Mapping) or stop.get("require_all_development_cells") is not True:
        raise SweepError("stop.require_all_development_cells must be true")
    if stop.get("require_joint_feasible") is not True:
        raise SweepError("stop.require_joint_feasible must be true")
    random_rule = stop.get("repeated_random_rule")
    if random_rule != "beat_every_draw":
        raise SweepError("stop.repeated_random_rule must be beat_every_draw")
    mean_margin = _finite(stop.get("mean_damage_margin", 0.0), "mean_damage_margin")
    cvar_margin = _finite(stop.get("cvar95_damage_margin", 0.0), "cvar95_damage_margin")
    if mean_margin < 0.0 or cvar_margin < 0.0:
        raise SweepError("damage margins must be non-negative")

    groups = stop.get("parent_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise SweepError("stop.parent_groups must be a non-empty mapping")
    normalized_groups: dict[str, dict[str, Any]] = {}
    for name, group in groups.items():
        if not isinstance(name, str) or not isinstance(group, Mapping):
            raise SweepError("each parent group must be a named mapping")
        members = group.get("members")
        minimum = group.get("minimum_passing")
        if (
            not isinstance(members, list)
            or not members
            or any(not isinstance(parent, str) or not parent for parent in members)
            or len(set(members)) != len(members)
            or isinstance(minimum, bool)
            or not isinstance(minimum, int)
            or not 1 <= minimum <= len(members)
        ):
            raise SweepError(f"invalid parent group {name!r}")
        normalized_groups[name] = {
            "members": list(members),
            "minimum_passing": minimum,
        }

    raw_trials = raw.get("trials")
    if not isinstance(raw_trials, list) or not raw_trials:
        raise SweepError("trials must be a non-empty ordered list")
    trials: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, trial in enumerate(raw_trials):
        if not isinstance(trial, Mapping):
            raise SweepError(f"trials[{index}] must be a mapping")
        trial_id = trial.get("id")
        if not isinstance(trial_id, str) or TRIAL_ID_RE.fullmatch(trial_id) is None:
            raise SweepError(f"trials[{index}].id is invalid")
        if trial_id in seen:
            raise SweepError(f"duplicate trial id {trial_id!r}")
        seen.add(trial_id)
        alpha = _finite(trial.get("alpha"), f"trials[{index}].alpha")
        kp = trial.get("Kp")
        repair = trial.get("repair")
        if not 0.0 <= alpha <= 1.0:
            raise SweepError(f"trials[{index}].alpha must be in [0, 1]")
        if isinstance(kp, bool) or not isinstance(kp, int) or kp < 1:
            raise SweepError(f"trials[{index}].Kp must be a positive integer")
        if not isinstance(repair, Mapping) or not repair:
            raise SweepError(f"trials[{index}].repair must be a non-empty mapping")
        unknown = set(repair) - REPAIR_KEYS
        if unknown:
            raise SweepError(f"trials[{index}].repair has unsupported keys {sorted(unknown)}")
        trials.append(
            {"id": trial_id, "alpha": alpha, "Kp": kp, "repair": dict(repair)}
        )

    budget = raw.get("budget")
    if (
        not isinstance(budget, Mapping)
        or budget.get("run_all_declared_trials") is not True
        or isinstance(budget.get("maximum_trials"), bool)
        or not isinstance(budget.get("maximum_trials"), int)
        or int(budget["maximum_trials"]) != len(trials)
    ):
        raise SweepError(
            "budget must require all declared trials and maximum_trials must "
            "equal the ordered trial count"
        )

    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "development_only": True,
        "setting": setting,
        "paths": dict(paths),
        "gpus": list(raw_gpus),
        "stop": {
            "require_all_development_cells": True,
            "require_joint_feasible": True,
            "repeated_random_rule": "beat_every_draw",
            "mean_damage_margin": mean_margin,
            "cvar95_damage_margin": cvar_margin,
            "parent_groups": normalized_groups,
        },
        "budget": {
            "maximum_trials": len(trials),
            "run_all_declared_trials": True,
            "success_exit_code": 0,
            "exhaustion_exit_code": 3,
        },
        "trials": trials,
    }


def evaluate_cell(
    diagnostic: Mapping[str, Any],
    *,
    expected_draws: Sequence[str],
    mean_margin: float,
    cvar_margin: float,
) -> dict[str, Any]:
    """Return the strict constrained-damage comparison for one D_prot cell."""
    raw_arms = diagnostic.get("arms")
    if not isinstance(raw_arms, list):
        raise SweepError("protection diagnostic arms must be a list")
    fixed: dict[str, Mapping[str, Any]] = {}
    random: dict[str, Mapping[str, Any]] = {}
    for index, arm in enumerate(raw_arms):
        if not isinstance(arm, Mapping):
            raise SweepError(f"arms[{index}] must be a mapping")
        name = arm.get("arm")
        draw = arm.get("draw_id")
        if name == "repeated_random":
            if not isinstance(draw, str) or draw in random:
                raise SweepError("repeated_random draw ids must be unique strings")
            random[draw] = arm
        elif name in ARMS:
            if draw is not None or name in fixed:
                raise SweepError(f"arm {name!r} is duplicated or has a draw id")
            fixed[str(name)] = arm
        else:
            raise SweepError(f"unexpected protection arm {name!r}")
    if set(fixed) != set(ARMS):
        raise SweepError(f"fixed arm roster mismatch: observed={sorted(fixed)}")
    if set(random) != set(expected_draws):
        raise SweepError(
            "repeated_random roster mismatch: "
            f"expected={sorted(expected_draws)}, observed={sorted(random)}"
        )

    joint = fixed["joint"]
    metrics = joint.get("metrics")
    if not isinstance(metrics, Mapping) or type(metrics.get("feasible")) is not bool:
        raise SweepError("joint.metrics.feasible must be boolean")
    joint_mean = _finite(joint.get("mean_damage"), "joint.mean_damage")
    joint_cvar = _finite(joint.get("cvar95_damage"), "joint.cvar95_damage")
    competitors = [
        (name, fixed[name]) for name in ("no_repair", "s0", "s1")
    ] + [(f"repeated_random/{draw}", random[draw]) for draw in expected_draws]
    comparisons = []
    for name, arm in competitors:
        other_metrics = arm.get("metrics")
        if (
            not isinstance(other_metrics, Mapping)
            or type(other_metrics.get("feasible")) is not bool
        ):
            raise SweepError(f"{name}.metrics.feasible must be boolean")
        other_feasible = bool(other_metrics["feasible"])
        other_mean = _finite(arm.get("mean_damage"), f"{name}.mean_damage")
        other_cvar = _finite(arm.get("cvar95_damage"), f"{name}.cvar95_damage")
        better_mean = joint_mean + mean_margin < other_mean
        better_cvar = joint_cvar + cvar_margin < other_cvar
        comparisons.append(
            {
                "competitor": name,
                "competitor_feasible": other_feasible,
                "mean_damage": other_mean,
                "cvar95_damage": other_cvar,
                "mean_advantage": other_mean - joint_mean - mean_margin,
                "cvar95_advantage": other_cvar - joint_cvar - cvar_margin,
                "joint_better_mean": better_mean,
                "joint_better_cvar95": better_cvar,
                "joint_wins_constrained": (
                    not other_feasible or (better_mean and better_cvar)
                ),
            }
        )
    passed = bool(metrics["feasible"]) and all(
        item["joint_wins_constrained"] for item in comparisons
    )
    return {
        "passed": passed,
        "joint_feasible": bool(metrics["feasible"]),
        "joint_mean_damage": joint_mean,
        "joint_cvar95_damage": joint_cvar,
        "comparisons": comparisons,
    }


def evaluate_trial(
    diagnostics: Iterable[Mapping[str, Any]],
    *,
    parents: Sequence[str],
    requests: Sequence[str],
    seeds: Sequence[int],
    expected_draws: Sequence[str],
    stop: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate all exact D_prot cells and the configured parent-group rule."""
    expected = {
        (str(parent), str(request), int(seed))
        for parent in parents
        for request in requests
        for seed in seeds
    }
    keyed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    duplicates = []
    for diagnostic in diagnostics:
        key = (
            str(diagnostic.get("parent")),
            str(diagnostic.get("request")),
            int(diagnostic.get("seed")),
        )
        if key in keyed:
            duplicates.append(key)
        keyed[key] = diagnostic
    observed = set(keyed)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)

    cells: dict[str, Any] = {}
    parent_results: dict[str, Any] = {}
    for parent in parents:
        parent_keys = sorted(key for key in expected if key[0] == parent)
        passed = not duplicates and not missing and not extra
        for key in parent_keys:
            label = f"{key[0]}/{key[1]}/seed-{key[2]}"
            if key not in keyed:
                cells[label] = {"passed": False, "error": "missing diagnostic"}
                passed = False
                continue
            result = evaluate_cell(
                keyed[key],
                expected_draws=expected_draws,
                mean_margin=float(stop["mean_damage_margin"]),
                cvar_margin=float(stop["cvar95_damage_margin"]),
            )
            cells[label] = result
            passed = passed and bool(result["passed"])
        parent_results[parent] = {
            "passed": passed,
            "passing_cells": sum(
                bool(cells[f"{key[0]}/{key[1]}/seed-{key[2]}"]["passed"])
                for key in parent_keys
            ),
            "required_cells": len(parent_keys),
        }

    group_results = {}
    for name, group in stop["parent_groups"].items():
        passing = [
            parent
            for parent in group["members"]
            if parent_results.get(parent, {}).get("passed") is True
        ]
        group_results[name] = {
            "passed": len(passing) >= int(group["minimum_passing"]),
            "passing_parents": passing,
            "minimum_passing": int(group["minimum_passing"]),
            "members": list(group["members"]),
        }
    return {
        "schema_version": 1,
        "development_only": True,
        "target_used": False,
        "passed": (
            not duplicates
            and not missing
            and not extra
            and all(group["passed"] for group in group_results.values())
        ),
        "missing_cells": [list(key) for key in missing],
        "extra_cells": [list(key) for key in extra],
        "duplicate_cells": [list(key) for key in duplicates],
        "parents": parent_results,
        "groups": group_results,
        "cells": cells,
    }


def candidate_score(result: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize how close one unsuccessful trial came to the stop rule."""
    groups = result.get("groups", {})
    parents = result.get("parents", {})
    cells = result.get("cells", {})
    if not all(isinstance(value, Mapping) for value in (groups, parents, cells)):
        raise SweepError("trial comparison lacks group, parent, or cell mappings")
    feasible_cells = 0
    passing_cells = 0
    metric_wins = 0
    metric_tests = 0
    total_shortfall = 0.0
    blocking: list[dict[str, Any]] = []
    for label, cell in cells.items():
        if not isinstance(cell, Mapping):
            raise SweepError(f"cell {label!r} must be a mapping")
        passing_cells += int(cell.get("passed") is True)
        feasible_cells += int(cell.get("joint_feasible") is True)
        comparisons = cell.get("comparisons", [])
        if not isinstance(comparisons, list):
            raise SweepError(f"cell {label!r}.comparisons must be a list")
        if cell.get("joint_feasible") is not True:
            blocking.append(
                {
                    "cell": str(label),
                    "reason": "joint_infeasible",
                    "competitor": None,
                }
            )
        for comparison in comparisons:
            if not isinstance(comparison, Mapping):
                raise SweepError(f"cell {label!r} has an invalid comparison")
            mean_win = comparison.get("joint_better_mean") is True
            cvar_win = comparison.get("joint_better_cvar95") is True
            competitor_feasible = comparison.get("competitor_feasible") is True
            metric_wins += (
                int(mean_win) + int(cvar_win) if competitor_feasible else 2
            )
            metric_tests += 2
            mean_advantage = _finite(
                comparison.get("mean_advantage"),
                f"{label}.mean_advantage",
            )
            cvar_advantage = _finite(
                comparison.get("cvar95_advantage"),
                f"{label}.cvar95_advantage",
            )
            mean_shortfall = (
                max(0.0, -mean_advantage) if competitor_feasible else 0.0
            )
            cvar_shortfall = (
                max(0.0, -cvar_advantage) if competitor_feasible else 0.0
            )
            total_shortfall += mean_shortfall + cvar_shortfall
            if comparison.get("joint_wins_constrained") is not True:
                blocking.append(
                    {
                        "cell": str(label),
                        "reason": "damage_not_strictly_better",
                        "competitor": comparison.get("competitor"),
                        "mean_shortfall": mean_shortfall,
                        "cvar95_shortfall": cvar_shortfall,
                    }
                )
    return {
        "trial_id": result.get("trial_id"),
        "trial_dir": result.get("trial_dir"),
        "groups_passed": sum(
            item.get("passed") is True for item in groups.values()
        ),
        "groups_required": len(groups),
        "parents_passed": sum(
            item.get("passed") is True for item in parents.values()
        ),
        "parents_required": len(parents),
        "cells_passed": passing_cells,
        "cells_required": len(cells),
        "joint_feasible_cells": feasible_cells,
        "metric_wins": metric_wins,
        "metric_tests": metric_tests,
        "total_shortfall": total_shortfall,
        "blocking_count": len(blocking),
        "blocking": blocking,
    }


def build_exhaustion_report(
    results: Sequence[Mapping[str, Any]],
    *,
    setting: str,
    spec_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic failure record after the declared grid is exhausted."""
    if not results:
        raise SweepError("cannot exhaust a sweep without evaluated trials")
    if any(result.get("passed") is True for result in results):
        raise SweepError("an exhaustion report cannot contain a passing trial")
    scored = [candidate_score(result) for result in results]
    ranked = sorted(
        scored,
        key=lambda item: (
            -int(item["groups_passed"]),
            -int(item["parents_passed"]),
            -int(item["cells_passed"]),
            -int(item["joint_feasible_cells"]),
            -int(item["metric_wins"]),
            float(item["total_shortfall"]),
            str(item["trial_id"]),
        ),
    )
    return {
        "schema_version": 1,
        "contract": CONTRACT,
        "status": "no_joint_dominance",
        "terminal": True,
        "exit_code": 3,
        "development_only": True,
        "target_used": False,
        "setting": setting,
        "spec_sha256": spec_sha256,
        "evaluated_trials": len(results),
        "declared_trial_budget": len(results),
        "reason": (
            "all declared development trials were evaluated without satisfying "
            "the strict joint-best stop rule"
        ),
        "closest_candidate": ranked[0],
        "ranked_candidates": ranked,
        "next_action": (
            "report the negative result, or prospectively append reviewed "
            "D_prot-only trials; do not inspect target outcomes"
        ),
    }


def _setting_contract(
    campaign: Mapping[str, Any],
    evidence: Mapping[str, Any],
    runtime: Mapping[str, Any],
    setting: str,
) -> tuple[str, list[str], list[str], list[int], list[str]]:
    settings = {
        item.get("id"): item
        for item in evidence.get("settings", [])
        if isinstance(item, Mapping)
    }
    evidence_setting = settings.get(setting)
    runtime_setting = runtime.get("settings", {}).get(setting)
    if not isinstance(evidence_setting, Mapping) or not isinstance(runtime_setting, Mapping):
        raise SweepError(f"setting {setting!r} is missing from evidence or runtime")
    model = evidence_setting.get("model")
    parents = evidence_setting.get("parents")
    tofu = campaign.get("datasets", {}).get("TOFU", {})
    requests = tofu.get("rosters", {}).get("D_prot")
    seeds = campaign.get("execution", {}).get("seeds")
    draws = campaign.get("execution", {}).get("repeated_random_draws")
    if not isinstance(model, str):
        raise SweepError(f"setting {setting!r} has no model")
    if not all(isinstance(value, list) and value for value in (parents, requests, seeds, draws)):
        raise SweepError("campaign/evidence lacks exact parents, D_prot, seeds, or draws")
    return (
        model,
        [str(value) for value in parents],
        [str(value) for value in requests],
        [int(value) for value in seeds],
        [str(value) for value in draws],
    )


def _require_parent_freeze(runtime: Mapping[str, Any], setting: str) -> None:
    setting_runtime = runtime.get("settings", {}).get(setting)
    if not isinstance(setting_runtime, Mapping):
        raise SweepError(f"runtime has no setting {setting!r}")
    path = _resolve(str(setting_runtime.get("parent_freeze", "")))
    freeze = _load_yaml(path)
    ready = freeze.get("status") == "frozen"
    if freeze.get("contract") == "tofu-pdf-v4-parent-freeze":
        ready = (
            ready
            and freeze.get("schema_version") == 1
            and freeze.get("setting") == setting
            and freeze.get("frozen_before_prediction") is True
            and not freeze.get("unresolved")
        )
    else:
        ready = ready and freeze.get("frozen_before_audit") is True and not freeze.get(
            "unresolved"
        )
    if not ready:
        raise HumanFreezeRequired(
            "PARENT_FREEZE_REQUIRED: parent calibration must be validated and frozen "
            f"before the joint sweep: {path}"
        )


def _with_parent_freeze(
    runtime: Mapping[str, Any], setting: str, parent_freeze: Path
) -> dict[str, Any]:
    path = parent_freeze.resolve()
    if not path.is_file():
        raise SweepError(f"parent_freeze is missing: {path}")
    resolved = json.loads(json.dumps(runtime))
    setting_runtime = resolved.get("settings", {}).get(setting)
    if not isinstance(setting_runtime, dict):
        raise SweepError(f"runtime has no setting {setting!r}")
    setting_runtime["parent_freeze"] = str(path)
    return resolved


def _best_parent_freeze(
    best: Mapping[str, Any], setting: str
) -> Path:
    value = best.get("recommended_runtime")
    if not isinstance(value, str) or not value:
        raise SweepError("existing BEST.json has no recommended_runtime")
    best_runtime = _load_yaml(_resolve(value))
    setting_runtime = best_runtime.get("settings", {}).get(setting)
    if not isinstance(setting_runtime, Mapping):
        raise SweepError(f"existing BEST runtime has no setting {setting!r}")
    parent_freeze = setting_runtime.get("parent_freeze")
    if not isinstance(parent_freeze, str) or not parent_freeze:
        raise SweepError("existing BEST runtime has no parent_freeze")
    return _resolve(parent_freeze)


def joint_sweep_completion(
    output_root: Path,
    *,
    setting: str,
    parent_freeze: Path,
) -> tuple[bool, str]:
    """Validate a terminal joint winner without preparing any GPU work."""
    output_root = output_root.resolve()
    expected_freeze = parent_freeze.resolve()
    best_path = output_root / "BEST.json"
    status_path = output_root / "SWEEP_STATUS.json"
    if not best_path.is_file():
        return False, f"missing {best_path}"
    if not status_path.is_file():
        return False, f"missing {status_path}"
    try:
        best = _load_json(best_path)
        status = _load_json(status_path)
        if (
            best.get("status") not in {"draft", "selected"}
            or best.get("human_review_required") not in {True, False}
            or best.get("development_only") is not True
            or best.get("target_used") is not False
        ):
            return False, "BEST.json is not a terminal target-free selection"
        if (
            status.get("status") != "joint_best"
            or status.get("terminal") is not True
            or status.get("exit_code") != 0
            or status.get("target_used") is not False
        ):
            return False, "SWEEP_STATUS.json is not a terminal joint winner"
        trial_dir = Path(str(best.get("trial_dir", ""))).resolve()
        if (
            not trial_dir.is_dir()
            or trial_dir != Path(str(status.get("trial_dir", ""))).resolve()
        ):
            return False, "BEST.json and SWEEP_STATUS.json trial directories differ"
        comparison = Path(str(best.get("joint_comparison", ""))).resolve()
        if not comparison.is_file() or _load_json(comparison).get("passed") is not True:
            return False, "winning joint comparison is missing or not passing"
        runtime = Path(str(best.get("recommended_runtime", ""))).resolve()
        if not runtime.is_file():
            return False, f"winning runtime is missing at {runtime}"
        if _best_parent_freeze(best, setting) != expected_freeze:
            return False, (
                "winning parent freeze differs "
                f"expected={expected_freeze} "
                f"actual={_best_parent_freeze(best, setting)}"
            )
        if not expected_freeze.is_file():
            return False, f"winning parent freeze is missing at {expected_freeze}"
        metadata = _load_json(trial_dir / "trial.json")
        resolved_runtime = (
            metadata.get("resolved_configs", {}).get("runtime")
            if isinstance(metadata.get("resolved_configs"), Mapping)
            else None
        )
        if Path(str(resolved_runtime or "")).resolve() != runtime:
            return False, "winning trial metadata points to a different runtime"
    except (OSError, SweepError, TypeError, ValueError) as error:
        return False, f"{type(error).__name__}: {error}"
    return True, f"validated terminal joint winner {best_path}"


def _valid_artifact(entry: object, expected: Path | None = None) -> bool:
    if not isinstance(entry, Mapping):
        return False
    path = Path(str(entry.get("path", ""))).resolve()
    digest = entry.get("sha256")
    return (
        isinstance(digest, str)
        and len(digest) == 64
        and path.is_file()
        and (expected is None or path == expected.resolve())
        and _sha256(path) == digest
    )


def _artifact_completion_status(
    entry: object,
    *,
    label: str,
    expected: Path | None = None,
) -> tuple[bool, str]:
    if not isinstance(entry, Mapping):
        return False, f"{label}: manifest entry is missing"
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        return False, f"{label}: artifact path is missing"
    path = Path(raw_path).resolve()
    if expected is not None and path != expected.resolve():
        return False, (
            f"{label}: path mismatch expected={expected.resolve()} actual={path}"
        )
    if not path.is_file():
        return False, f"{label}: artifact is missing at {path}"
    digest = entry.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return False, f"{label}: SHA-256 is missing or malformed"
    actual = _sha256(path)
    if actual != digest:
        return False, (
            f"{label}: SHA-256 mismatch expected={digest} actual={actual}"
        )
    return True, "complete"


def _unit_completion_status(
    unit: Mapping[str, Any],
    *,
    campaign_hash: str,
    evidence_hash: str,
    runtime_hash: str,
    stage: str = "protection",
) -> tuple[bool, str]:
    try:
        run_manifest = Path(str(unit["run_manifest"])).resolve()
        if not run_manifest.is_file():
            return False, f"run manifest is missing at {run_manifest}"
        raw = _load_json(run_manifest)
        if raw.get("schema_version") != 1:
            return False, (
                "run manifest schema mismatch: "
                f"expected=1 actual={raw.get('schema_version')}"
            )
        if raw.get("contract") != "tofu-pdf-v4-unit-output":
            return False, (
                "run manifest contract mismatch: "
                f"actual={raw.get('contract')}"
            )
        if raw.get("stage") != stage:
            return False, (
                f"stage mismatch expected={stage} actual={raw.get('stage')}"
            )
        expected_setting = unit.get("setting", raw.get("setting"))
        if raw.get("setting") != expected_setting:
            return False, (
                "setting mismatch "
                f"expected={expected_setting} actual={raw.get('setting')}"
            )
        expected_hashes = {
            "campaign_config_sha256": campaign_hash,
            "evidence_config_sha256": evidence_hash,
            "runtime_config_sha256": runtime_hash,
        }
        for name, digest in expected_hashes.items():
            if raw.get(name) != digest:
                return False, (
                    f"{name} mismatch expected={digest} actual={raw.get(name)}"
                )
        for name in ("parent", "request"):
            if raw.get(name) != unit.get(name):
                return False, (
                    f"{name} mismatch expected={unit.get(name)} "
                    f"actual={raw.get(name)}"
                )
        if str(raw.get("seed")) != str(unit.get("seed")):
            return False, (
                f"seed mismatch expected={unit.get('seed')} actual={raw.get('seed')}"
            )
        outputs = unit.get("outputs")
        raw_outputs = raw.get("outputs")
        if not isinstance(outputs, Mapping) or not isinstance(raw_outputs, Mapping):
            return False, "output artifact roster is missing"
        if set(outputs) != set(raw_outputs):
            return False, (
                "output artifact roster mismatch "
                f"expected={sorted(outputs)} actual={sorted(raw_outputs)}"
            )
        for name, path in outputs.items():
            valid, reason = _artifact_completion_status(
                raw_outputs[name],
                label=f"output.{name}",
                expected=Path(str(path)),
            )
            if not valid:
                return False, reason
        required_artifacts = [
            ("profile_artifact", raw.get("profile_artifact")),
            (
                "score_independent_manifest",
                raw.get("score_independent_manifest"),
            ),
        ]
        if stage in {"calibration", "target_evaluation"}:
            required_artifacts.append(
                ("fidelity_diagnostics", raw.get("fidelity_diagnostics"))
            )
        if stage in {"protection", "target_evaluation"}:
            required_artifacts.append(
                ("protection_diagnostics", raw.get("protection_diagnostics"))
            )
        if stage == "target_evaluation":
            required_artifacts.append(
                ("selection_freeze", raw.get("selection_freeze"))
            )
        for label, artifact in required_artifacts:
            valid, reason = _artifact_completion_status(
                artifact,
                label=label,
            )
            if not valid:
                return False, reason
        if stage != "calibration":
            valid, reason = _artifact_completion_status(
                {
                    "path": raw.get("parent_freeze"),
                    "sha256": raw.get("parent_freeze_sha256"),
                },
                label="parent_freeze",
            )
            if not valid:
                return False, reason
        return True, "all declared artifacts and hashes are valid"
    except (KeyError, OSError, SweepError, TypeError, ValueError) as error:
        return False, f"{type(error).__name__}: {error}"


def _unit_complete(
    unit: Mapping[str, Any],
    *,
    campaign_hash: str,
    evidence_hash: str,
    runtime_hash: str,
    stage: str = "protection",
) -> bool:
    complete, _reason = _unit_completion_status(
        unit,
        campaign_hash=campaign_hash,
        evidence_hash=evidence_hash,
        runtime_hash=runtime_hash,
        stage=stage,
    )
    return complete


class _EventLog:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def write(self, event: Mapping[str, Any]) -> None:
        row = {"at_utc": _utc_now(), **dict(event)}
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")


def _run_unit(
    unit: Mapping[str, Any],
    *,
    gpu: int,
    trial_dir: Path,
    events: _EventLog,
    active_processes: dict[int, subprocess.Popen[str]] | None = None,
    process_lock: threading.Lock | None = None,
) -> tuple[bool, str]:
    identity = (
        f"{unit['parent']}__{unit['request']}__seed-{unit['seed']}"
    )
    log_dir = trial_dir / "logs" / "units" / identity
    log_dir.mkdir(parents=True, exist_ok=True)
    attempt = len(list(log_dir.glob("attempt-*.log"))) + 1
    log_path = log_dir / f"attempt-{attempt:03d}.log"
    status_path = log_dir / f"attempt-{attempt:03d}.json"
    command = unit.get("command")
    if not isinstance(command, list) or any(not isinstance(value, str) for value in command):
        raise SweepError(f"invalid command for unit {identity}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["TOKENIZERS_PARALLELISM"] = "false"
    started = _utc_now()
    started_monotonic = time.monotonic()
    events.write({"event": "unit_started", "unit": identity, "gpu": gpu, "attempt": attempt})
    _status(
        f"UNIT_START gpu={gpu} unit={identity} attempt={attempt} log={log_path}"
    )
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "started_at_utc": started,
                    "gpu": gpu,
                    "CUDA_VISIBLE_DEVICES": str(gpu),
                    "command": command,
                },
                sort_keys=True,
            )
            + "\n"
        )
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if active_processes is not None and process_lock is not None:
            with process_lock:
                active_processes[gpu] = process
        try:
            if process.stdout is None:
                raise SweepError(f"cannot capture output for unit {identity}")
            prefix = f"[GPU{gpu} {identity}]"
            for line in process.stdout:
                handle.write(line)
                handle.flush()
                print(f"{prefix} {line}", end="", flush=True)
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if active_processes is not None and process_lock is not None:
                with process_lock:
                    if active_processes.get(gpu) is process:
                        active_processes.pop(gpu, None)
    duration = time.monotonic() - started_monotonic
    status = {
        "unit": identity,
        "attempt": attempt,
        "gpu": gpu,
        "started_at_utc": started,
        "finished_at_utc": _utc_now(),
        "duration_seconds": duration,
        "returncode": returncode,
        "log": str(log_path),
        "command": command,
    }
    _atomic_json(status_path, status)
    events.write(
        {
            "event": "unit_finished",
            "unit": identity,
            "gpu": gpu,
            "attempt": attempt,
            "duration_seconds": duration,
            "returncode": returncode,
        }
    )
    _status(
        f"UNIT_END gpu={gpu} unit={identity} returncode={returncode} "
        f"elapsed={duration:.1f}s"
    )
    return returncode == 0, identity


def _run_lanes(
    units: Sequence[Mapping[str, Any]],
    *,
    gpus: Sequence[int],
    trial_dir: Path,
    events: _EventLog,
    progress_interval: float,
    progress_label: str = "",
    completed_offset: int = 0,
    total_units: int | None = None,
) -> None:
    if not units:
        _status("UNITS all manifest units already validated; nothing to execute")
        return
    lanes = [list(units[index::len(gpus)]) for index in range(len(gpus))]
    stopped = threading.Event()
    monitor_stopped = threading.Event()
    state_lock = threading.Lock()
    running: dict[int, str] = {}
    completed = 0
    failures: list[str] = []
    process_lock = threading.Lock()
    active_processes: dict[int, subprocess.Popen[str]] = {}
    started = time.monotonic()
    total = total_units if total_units is not None else len(units)
    cache_locks = {
        (str(unit["request"]), str(unit["seed"])): threading.Lock()
        for unit in units
    }

    def terminate_active(reason: str) -> None:
        with process_lock:
            processes = list(active_processes.items())
        if not processes:
            return
        _status(
            f"UNIT_CLEANUP reason={reason} active_gpus="
            f"{[gpu for gpu, _process in processes]} signal=TERM"
        )
        for _gpu, process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 5.0
        while (
            any(process.poll() is None for _gpu, process in processes)
            and time.monotonic() < deadline
        ):
            time.sleep(0.1)
        survivors = [
            (gpu, process)
            for gpu, process in processes
            if process.poll() is None
        ]
        if survivors:
            _status(
                f"UNIT_CLEANUP reason={reason} active_gpus="
                f"{[gpu for gpu, _process in survivors]} signal=KILL"
            )
            for _gpu, process in survivors:
                process.kill()

    def monitor() -> None:
        while not monitor_stopped.wait(progress_interval):
            with state_lock:
                active = ", ".join(
                    f"gpu{gpu}={identity}"
                    for gpu, identity in sorted(running.items())
                ) or "none"
                done = completed
                failed = len(failures)
            elapsed = time.monotonic() - started
            remaining = len(units) - done
            eta = (elapsed / done * remaining) if done else None
            prefix = f"{progress_label} " if progress_label else ""
            _status(
                f"PROGRESS {prefix}completed={completed_offset + done}/{total} "
                f"failed={failed} elapsed={elapsed:.1f}s "
                f"eta_seconds={eta:.0f} running=[{active}]"
                if eta is not None
                else
                f"PROGRESS {prefix}completed={completed_offset + done}/{total} "
                f"failed={failed} elapsed={elapsed:.1f}s "
                f"eta_seconds=pending running=[{active}]"
            )

    def worker(gpu: int, lane: Sequence[Mapping[str, Any]]) -> None:
        nonlocal completed
        for unit in lane:
            if stopped.is_set():
                break
            identity = (
                f"{unit['parent']}__{unit['request']}__seed-{unit['seed']}"
            )
            with state_lock:
                running[gpu] = identity
            cache_key = (str(unit["request"]), str(unit["seed"]))
            if cache_locks[cache_key].locked():
                _status(
                    f"CACHE_WAIT gpu={gpu} unit={identity} "
                    f"request={cache_key[0]} seed={cache_key[1]}"
                )
            with cache_locks[cache_key]:
                ok, identity = _run_unit(
                    unit,
                    gpu=gpu,
                    trial_dir=trial_dir,
                    events=events,
                    active_processes=active_processes,
                    process_lock=process_lock,
                )
            with state_lock:
                running.pop(gpu, None)
                completed += 1
            if not ok:
                with state_lock:
                    failures.append(identity)
                stopped.set()
                terminate_active(f"unit-failed:{identity}")
                break

    monitor_thread = threading.Thread(
        target=monitor,
        name="joint-sweep-progress",
        daemon=True,
    )
    monitor_thread.start()
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, lane) for gpu, lane in zip(gpus, lanes)]
        try:
            for future in futures:
                future.result()
        except BaseException:
            stopped.set()
            terminate_active("worker-exception")
            raise
        finally:
            monitor_stopped.set()
            monitor_thread.join()
    with state_lock:
        active = ", ".join(
            f"gpu{gpu}={identity}" for gpu, identity in sorted(running.items())
        ) or "none"
        elapsed = time.monotonic() - started
        prefix = f"{progress_label} " if progress_label else ""
        _status(
            f"PROGRESS {prefix}completed={completed_offset + completed}/{total} "
            f"failed={len(failures)} elapsed={elapsed:.1f}s "
            f"eta_seconds=0 running=[{active}]"
        )
    if failures:
        raise SweepError(f"unit execution failed; inspect logs for {failures}")


def _prepare_trial(
    *,
    spec: Mapping[str, Any],
    trial: Mapping[str, Any],
    campaign: Mapping[str, Any],
    evidence: Mapping[str, Any],
    runtime: Mapping[str, Any],
    campaign_source: Path,
    evidence_source: Path,
    python: Path,
    output_root: Path,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    setting = str(spec["setting"])
    model, _parents, _requests, _seeds, _draws = _setting_contract(
        campaign, evidence, runtime, setting
    )
    campaign_local = json.loads(json.dumps(campaign))
    runtime_local = json.loads(json.dumps(runtime))
    campaign_local["models"][model]["source"] = str(
        _resolve(spec["paths"]["model_source"])
    )
    campaign_local["models"][model]["source_kind"] = "local_path"
    runtime_local["runtime"]["sft_cache_root"] = str(
        _resolve(spec["paths"]["sft_cache_root"])
    )
    runtime_local["probe"]["alpha_grid"] = [float(trial["alpha"])]
    runtime_local["protection"]["Kp_grid"] = [int(trial["Kp"])]
    runtime_local["protection"]["repair"].update(dict(trial["repair"]))

    fingerprint = _json_sha(
        {
            "setting": setting,
            "trial": trial,
            "campaign_sha256": _sha256(campaign_source),
            "evidence_sha256": _sha256(evidence_source),
            "runtime_sha256": _sha256(_resolve(spec["paths"]["runtime"])),
            "resolved_runtime_sha256": _json_sha(runtime),
            "model_source": str(_resolve(spec["paths"]["model_source"])),
            "sft_cache_root": str(_resolve(spec["paths"]["sft_cache_root"])),
            "parent_freeze": str(
                _resolve(runtime["settings"][setting]["parent_freeze"])
            ),
            "parent_freeze_sha256": _sha256(
                _resolve(runtime["settings"][setting]["parent_freeze"])
            ),
        }
    )
    trial_dir = output_root / "trials" / f"{trial['id']}--{fingerprint[:12]}"
    config_dir = trial_dir / "config"
    campaign_path = config_dir / "campaign.local.yaml"
    runtime_path = config_dir / "tofu_v4.local.yaml"
    _write_once(
        campaign_path,
        yaml.safe_dump(campaign_local, sort_keys=False),
    )
    _write_once(runtime_path, yaml.safe_dump(runtime_local, sort_keys=False))
    metadata = {
        "schema_version": 1,
        "contract": CONTRACT,
        "development_only": True,
        "target_used": False,
        "setting": setting,
        "trial": dict(trial),
        "fingerprint": fingerprint,
        "canonical_sources": {
            "campaign": str(campaign_source),
            "campaign_sha256": _sha256(campaign_source),
            "evidence": str(evidence_source),
            "evidence_sha256": _sha256(evidence_source),
            "runtime": str(_resolve(spec["paths"]["runtime"])),
            "runtime_sha256": _sha256(_resolve(spec["paths"]["runtime"])),
        },
        "resolved_configs": {
            "campaign": str(campaign_path),
            "runtime": str(runtime_path),
        },
    }
    _write_once(
        trial_dir / "trial.json",
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
    )

    manifest = build_manifest(
        campaign_local,
        evidence,
        stage="protection",
        setting_id=setting,
        campaign_path=campaign_path,
        evidence_path=evidence_source,
        runtime_path=runtime_path,
        unit_root=trial_dir / "units",
        python=str(python),
    )
    for unit in manifest["units"]:
        unit["setting"] = setting
    manifest_path = trial_dir / "manifest.yaml"
    manifest = _write_or_rebind_manifest(manifest_path, manifest)
    return trial_dir, campaign_path, runtime_path, manifest


def _collect_diagnostics(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    diagnostics = []
    for unit in manifest["units"]:
        path = Path(str(unit["run_manifest"])).resolve().parent / "protection_diagnostics.json"
        diagnostics.append(_load_json(path))
    return diagnostics


def _seal_trial(
    *,
    python: Path,
    campaign_path: Path,
    evidence_path: Path,
    manifest_path: Path,
    trial_dir: Path,
) -> None:
    log_path = trial_dir / "logs" / "verify.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        "-u",
        str(ROOT / "experiments/paper/run_v4_stage.py"),
        "--campaign",
        str(campaign_path),
        "--evidence",
        str(evidence_path),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(trial_dir / "stage"),
        "--action",
        "verify",
    ]
    _status(f"VERIFY_START manifest={manifest_path} log={log_path}")
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout is None:
            raise SweepError("cannot capture stage verification output")
        for line in process.stdout:
            handle.write(line)
            handle.flush()
            print(f"[VERIFY] {line}", end="", flush=True)
        returncode = process.wait()
    _status(
        f"VERIFY_END returncode={returncode} "
        f"elapsed={time.monotonic() - started:.1f}s"
    )
    if returncode != 0:
        raise SweepError(f"stage verification failed; inspect {log_path}")


def _write_summary(output_root: Path) -> None:
    rows = []
    for path in sorted((output_root / "trials").glob("*/joint_comparison.json")):
        value = _load_json(path)
        rows.append(
            {
                "trial": path.parent.name,
                "passed": str(bool(value.get("passed"))).lower(),
                "passing_parents": ",".join(
                    parent
                    for parent, result in value.get("parents", {}).items()
                    if result.get("passed") is True
                ),
                "comparison": str(path),
            }
        )
    destination = output_root / "summary.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("trial", "passed", "passing_parents", "comparison"),
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, destination)


def _environment_snapshot(output_root: Path, python: Path, gpus: Sequence[int]) -> None:
    path = output_root / "environment.json"
    if path.is_file():
        return

    def output(command: list[str]) -> str:
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        return (result.stdout + result.stderr).strip()

    payload = {
        "created_at_utc": _utc_now(),
        "python": str(python),
        "python_version": output([str(python), "--version"]),
        "gpus": list(gpus),
        "nvidia_smi": output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader",
            ]
        ),
        "pip_freeze": output([str(python), "-m", "pip", "freeze"]).splitlines(),
    }
    _write_once(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run(args: argparse.Namespace) -> int:
    spec_source = args.spec.resolve()
    _status(f"LOAD_SPEC path={spec_source}")
    spec = validate_spec(_load_yaml(spec_source))
    if args.gpus:
        try:
            supplied_gpus = [int(value) for value in args.gpus.split(",")]
        except ValueError as error:
            raise SweepError("--gpus must be a comma-separated integer list") from error
        if (
            not supplied_gpus
            or any(gpu < 0 for gpu in supplied_gpus)
            or len(set(supplied_gpus)) != len(supplied_gpus)
        ):
            raise SweepError("--gpus must contain unique non-negative integers")
        spec["gpus"] = supplied_gpus
    if args.max_trials is not None and args.max_trials < 1:
        raise SweepError("--max-trials must be positive")
    if args.progress_interval < 1.0:
        raise SweepError("--progress-interval must be at least 1 second")
    for name, value in (
        ("python", args.python),
        ("model_source", args.model_source),
        ("sft_cache_root", args.sft_cache_root),
        ("parent_freeze", args.parent_freeze),
        ("output_root", args.output_root),
    ):
        if value is not None:
            spec["paths"][name] = str(value)

    campaign_path = _resolve(spec["paths"]["campaign"])
    evidence_path = _resolve(spec["paths"]["evidence"])
    runtime_source = _resolve(spec["paths"]["runtime"])
    python = _absolute_executable(spec["paths"]["python"])
    output_root = _resolve(spec["paths"]["output_root"])
    _status(
        f"CONFIG setting={spec['setting']} trials={len(spec['trials'])} "
        f"gpus={spec['gpus']} progress_interval={args.progress_interval}s"
    )
    _status(
        f"PATHS output={output_root} model={_resolve(spec['paths']['model_source'])}"
    )
    campaign = _load_yaml(campaign_path)
    evidence = _load_yaml(evidence_path)
    runtime = _load_yaml(runtime_source)
    if "parent_freeze" in spec["paths"]:
        runtime = _with_parent_freeze(
            runtime,
            str(spec["setting"]),
            _resolve(spec["paths"]["parent_freeze"]),
        )
    _require_parent_freeze(runtime, str(spec["setting"]))
    model, parents, requests, seeds, draws = _setting_contract(
        campaign, evidence, runtime, str(spec["setting"])
    )
    del model

    if not python.is_file():
        raise SweepError(f"Python executable is missing: {python}")
    for name in ("model_source",):
        path = _resolve(spec["paths"][name])
        if not path.exists():
            raise SweepError(f"{name} is missing: {path}")
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_spec = {**spec, "paths": dict(spec["paths"])}
    resolved_spec["paths"]["output_root"] = str(output_root)
    spec_digest = _json_sha(resolved_spec)
    _write_once(
        output_root / "specs" / f"{spec_digest}.yaml",
        yaml.safe_dump(resolved_spec, sort_keys=False),
    )
    sweep_manifest_path = output_root / "sweep_manifest.json"
    trial_ids = [trial["id"] for trial in spec["trials"]]
    revision = {
        "spec_sha256": spec_digest,
        "snapshot": str(output_root / "specs" / f"{spec_digest}.yaml"),
        "source": str(spec_source),
        "source_sha256": _sha256(spec_source),
        "ordered_trial_ids": trial_ids,
        "recorded_at_utc": _utc_now(),
    }
    if sweep_manifest_path.is_file():
        sweep_manifest = _load_json(sweep_manifest_path)
        previous_ids = sweep_manifest.get("ordered_trial_ids")
        if (
            sweep_manifest.get("setting") != spec["setting"]
            or sweep_manifest.get("stop") != spec["stop"]
            or not isinstance(previous_ids, list)
            or trial_ids[: len(previous_ids)] != previous_ids
        ):
            raise SweepError(
                "an existing sweep may only append trials; setting, stop rule, "
                "and prior trial order are immutable"
            )
        revisions = sweep_manifest.get("spec_revisions")
        if not isinstance(revisions, list):
            raise SweepError("existing sweep manifest has no spec revision list")
        if not any(item.get("spec_sha256") == spec_digest for item in revisions):
            revisions.append(revision)
        sweep_manifest["ordered_trial_ids"] = trial_ids
        sweep_manifest["budget"] = spec["budget"]
    else:
        sweep_manifest = {
            "schema_version": 1,
            "contract": CONTRACT,
            "development_only": True,
            "target_used": False,
            "setting": spec["setting"],
            "ordered_trial_ids": trial_ids,
            "stop": spec["stop"],
            "budget": spec["budget"],
            "spec_revisions": [revision],
        }
    _atomic_json(sweep_manifest_path, sweep_manifest)
    _environment_snapshot(output_root, python, spec["gpus"])
    events = _EventLog(output_root / "events.jsonl")
    events.write(
        {
            "event": "sweep_invoked",
            "setting": spec["setting"],
            "trials": len(spec["trials"]),
            "gpus": spec["gpus"],
            "progress_interval_seconds": args.progress_interval,
        }
    )
    _status(
        "STOP_RULE joint feasible; infeasible comparators lose; feasible "
        "comparators require strict mean and CVaR wins"
    )

    best_path = output_root / "BEST.json"
    if best_path.is_file():
        best = _load_json(best_path)
        expected_parent_freeze = _resolve(
            runtime["settings"][str(spec["setting"])]["parent_freeze"]
        )
        observed_parent_freeze = _best_parent_freeze(
            best, str(spec["setting"])
        )
        if observed_parent_freeze != expected_parent_freeze:
            raise SweepError(
                "existing BEST uses a different parent freeze; refusing stale "
                f"winner reuse (expected {expected_parent_freeze}, "
                f"found {observed_parent_freeze}). Use a new RESULTS_ROOT to "
                "run the sweep under the approved freeze"
            )
        print(f"joint sweep already satisfied by {best['trial_dir']}")
        return 0

    evaluated_results: list[dict[str, Any]] = []
    for index, trial in enumerate(spec["trials"], start=1):
        if args.max_trials is not None and index > args.max_trials:
            break
        trial_dir, local_campaign, local_runtime, manifest = _prepare_trial(
            spec=spec,
            trial=trial,
            campaign=campaign,
            evidence=evidence,
            runtime=runtime,
            campaign_source=campaign_path,
            evidence_source=evidence_path,
            python=python,
            output_root=output_root,
        )
        events.write(
            {
                "event": "trial_started",
                "trial_id": trial["id"],
                "trial_dir": str(trial_dir),
            }
        )
        campaign_hash = _sha256(local_campaign)
        evidence_hash = _sha256(evidence_path)
        runtime_hash = _sha256(local_runtime)
        completion = [
            (
                unit,
                *_unit_completion_status(
                    unit,
                    campaign_hash=campaign_hash,
                    evidence_hash=evidence_hash,
                    runtime_hash=runtime_hash,
                ),
            )
            for unit in manifest["units"]
        ]
        pending = [
            unit
            for unit, complete, _reason in completion
            if not complete
        ]
        for unit, complete, reason in completion:
            identity = (
                f"{unit.get('parent')}__{unit.get('request')}"
                f"__seed-{unit.get('seed')}"
            )
            if complete:
                _status(
                    f"TRIAL_REUSE trial={trial['id']} unit={identity} "
                    "retraining=0"
                )
            else:
                _status(
                    f"TRIAL_PENDING trial={trial['id']} unit={identity} "
                    f"reason={reason}"
                )
        _status(
            f"[{index}/{len(spec['trials'])}] {trial['id']}: "
            f"alpha={trial['alpha']} Kp={trial['Kp']} repair={trial['repair']} "
            f"valid={len(manifest['units']) - len(pending)} "
            f"pending={len(pending)}"
        )
        _status(
            f"CURRENT_BEST status=pending trial={index}/{len(spec['trials'])} "
            f"id={trial['id']} completed={len(manifest['units']) - len(pending)}/"
            f"{len(manifest['units'])} decision_at={len(manifest['units'])}/"
            f"{len(manifest['units'])}"
        )
        if args.dry_run:
            continue
        _run_lanes(
            pending,
            gpus=spec["gpus"],
            trial_dir=trial_dir,
            events=events,
            progress_interval=args.progress_interval,
            progress_label=(
                f"trial={index}/{len(spec['trials'])} id={trial['id']} BEST=pending"
            ),
            completed_offset=len(manifest["units"]) - len(pending),
            total_units=len(manifest["units"]),
        )
        manifest_path = trial_dir / "manifest.yaml"
        _seal_trial(
            python=python,
            campaign_path=local_campaign,
            evidence_path=evidence_path,
            manifest_path=manifest_path,
            trial_dir=trial_dir,
        )
        result = evaluate_trial(
            _collect_diagnostics(manifest),
            parents=parents,
            requests=requests,
            seeds=seeds,
            expected_draws=draws,
            stop=spec["stop"],
        )
        result.update(
            {
                "trial_id": trial["id"],
                "trial_dir": str(trial_dir),
                "trial": trial,
                "evaluated_at_utc": _utc_now(),
            }
        )
        evaluated_results.append(result)
        _atomic_json(trial_dir / "joint_comparison.json", result)
        _write_summary(output_root)
        score = candidate_score(result)
        passing_parents = [
            parent
            for parent, parent_result in result["parents"].items()
            if parent_result["passed"] is True
        ]
        _status(
            f"TRIAL_RESULT id={trial['id']} passed={result['passed']} "
            f"groups={score['groups_passed']}/{score['groups_required']} "
            f"parents={score['parents_passed']}/{score['parents_required']} "
            f"cells={score['cells_passed']}/{score['cells_required']} "
            f"feasible_cells={score['joint_feasible_cells']}/"
            f"{score['cells_required']} blocking={score['blocking_count']} "
            f"shortfall={score['total_shortfall']:.8g} "
            f"passing_parents={passing_parents}"
        )
        for blocker in score["blocking"][:10]:
            _status(
                f"BLOCKER trial={trial['id']} cell={blocker['cell']} "
                f"reason={blocker['reason']} "
                f"competitor={blocker.get('competitor')} "
                f"mean_shortfall={blocker.get('mean_shortfall', 0.0):.8g} "
                f"cvar_shortfall={blocker.get('cvar95_shortfall', 0.0):.8g}"
            )
        if len(score["blocking"]) > 10:
            _status(
                f"BLOCKER remaining={len(score['blocking']) - 10} "
                f"details={trial_dir / 'joint_comparison.json'}"
            )
        events.write(
            {
                "event": "trial_evaluated",
                "trial_id": trial["id"],
                "passed": result["passed"],
            }
        )
        if result["passed"]:
            _status(
                f"CURRENT_BEST status=selected trial={index}/"
                f"{len(spec['trials'])} id={trial['id']} "
                f"comparison={trial_dir / 'joint_comparison.json'}"
            )
            recommendation = {
                "schema_version": 1,
                "contract": CONTRACT,
                "status": "selected",
                "human_review_required": False,
                "automatic_freeze_ready": True,
                "development_only": True,
                "target_used": False,
                "trial_id": trial["id"],
                "trial_dir": str(trial_dir),
                "recommended_runtime": str(local_runtime),
                "joint_comparison": str(trial_dir / "joint_comparison.json"),
                "next_action": (
                    "validate and record the target-free freeze before target evaluation"
                ),
            }
            _write_once(
                best_path,
                json.dumps(recommendation, indent=2, sort_keys=True) + "\n",
            )
            _write_once(
                output_root / "recommendation.yaml",
                yaml.safe_dump(recommendation, sort_keys=False),
            )
            _atomic_json(
                output_root / "SWEEP_STATUS.json",
                {
                    "status": "joint_best",
                    "terminal": True,
                    "exit_code": 0,
                    "trial_id": trial["id"],
                    "trial_dir": str(trial_dir),
                    "development_only": True,
                    "target_used": False,
                    "updated_at_utc": _utc_now(),
                },
            )
            print(f"JOINT_BEST_DEVELOPMENT: {trial_dir}")
            print("target evaluation was not run; automatic freeze validation is next")
            return 0
        closest = min(
            (candidate_score(candidate) for candidate in evaluated_results),
            key=lambda item: (
                -int(item["groups_passed"]),
                -int(item["parents_passed"]),
                -int(item["cells_passed"]),
                -int(item["joint_feasible_cells"]),
                -int(item["metric_wins"]),
                float(item["total_shortfall"]),
                str(item["trial_id"]),
            ),
        )
        _status(
            f"CURRENT_BEST status=not_found closest={closest['trial_id']} "
            f"evaluated={len(evaluated_results)}/{len(spec['trials'])} "
            f"groups={closest['groups_passed']}/{closest['groups_required']} "
            f"parents={closest['parents_passed']}/{closest['parents_required']} "
            f"cells={closest['cells_passed']}/{closest['cells_required']} "
            f"shortfall={closest['total_shortfall']:.8g}"
        )

    if args.dry_run:
        print(f"dry run complete: {output_root}")
        return 0
    if args.max_trials is not None and args.max_trials < len(spec["trials"]):
        pause = {
            "status": "paused_budget_limit",
            "terminal": False,
            "exit_code": 5,
            "completed_trial_limit": args.max_trials,
            "declared_trials": len(spec["trials"]),
            "development_only": True,
            "target_used": False,
            "updated_at_utc": _utc_now(),
        }
        _atomic_json(output_root / "SWEEP_STATUS.json", pause)
        events.write({"event": "sweep_paused_budget_limit", **pause})
        print(
            "PAUSED_BUDGET_LIMIT: not all declared trials were evaluated",
            file=sys.stderr,
        )
        return 5

    report = build_exhaustion_report(
        evaluated_results,
        setting=str(spec["setting"]),
        spec_sha256=spec_digest,
    )
    report["recorded_at_utc"] = _utc_now()
    exhaustion_path = output_root / "exhaustions" / f"{spec_digest}.json"
    _write_once(
        exhaustion_path,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    _write_once(
        exhaustion_path.with_suffix(".yaml"),
        yaml.safe_dump(report, sort_keys=False),
    )
    _atomic_json(
        output_root / "SWEEP_STATUS.json",
        {
            "status": "no_joint_dominance",
            "terminal": True,
            "exit_code": 3,
            "report": str(exhaustion_path),
            "closest_trial_id": report["closest_candidate"]["trial_id"],
            "development_only": True,
            "target_used": False,
            "updated_at_utc": _utc_now(),
        },
    )
    events.write(
        {
            "event": "sweep_exhausted_no_joint_dominance",
            "report": str(exhaustion_path),
            "closest_trial_id": report["closest_candidate"]["trial_id"],
            "exit_code": 3,
        }
    )
    print(
        "NO_JOINT_DOMINANCE: all declared development trials were exhausted; "
        f"failure report: {exhaustion_path}",
        file=sys.stderr,
    )
    return 3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--spec",
        type=Path,
        default=ROOT / "configs/local/joint_sweep_1p5b_4090x2.yaml",
    )
    parser.add_argument("--gpus", default=None, help="physical GPU ids, e.g. 0,1")
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--model-source", type=Path, default=None)
    parser.add_argument("--sft-cache-root", type=Path, default=None)
    parser.add_argument("--parent-freeze", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-trials", type=int, default=None)
    parser.add_argument("--progress-interval", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-complete",
        action="store_true",
        help="validate an existing joint winner without starting GPU work",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.check_complete:
            if args.output_root is None or args.parent_freeze is None:
                raise SweepError(
                    "--check-complete requires --output-root and --parent-freeze"
                )
            complete, reason = joint_sweep_completion(
                args.output_root,
                setting="tofu_qwen25_1p5b",
                parent_freeze=args.parent_freeze,
            )
            _status(
                f"JOINT_SWEEP_STATUS complete={str(complete).lower()} "
                f"reason={reason}"
            )
            return 0 if complete else 1
        return run(args)
    except HumanFreezeRequired as error:
        print(str(error), file=sys.stderr)
        return 4
    except (SweepError, OSError, subprocess.SubprocessError) as error:
        print(f"joint development sweep failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
