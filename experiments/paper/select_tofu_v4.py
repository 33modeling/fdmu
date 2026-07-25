#!/usr/bin/env python3
"""Create TOFU parent or prediction/protection freezes from target-free stages."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.analysis.mixture import select_prediction_alpha  # noqa: E402
from rsus.analysis.table1_selection import (  # noqa: E402
    select_protection_pair,
    select_simple_control,
)


class SelectionError(ValueError):
    """Development artifacts do not satisfy the frozen selection design."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise SelectionError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SelectionError(f"{path} must contain one mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SelectionError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise SelectionError(f"{name} must be finite")
    return number


def _records(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    rows: list[dict[str, Any]] = []
    sources = []
    seen_paths: set[Path] = set()
    for supplied in paths:
        candidates = (
            sorted(supplied.glob("**/*.jsonl"))
            if supplied.is_dir()
            else [supplied]
        )
        for path in candidates:
            path = path.resolve()
            if path in seen_paths:
                continue
            seen_paths.add(path)
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                raise SelectionError(f"cannot read {path}: {error}") from error
            parsed = []
            for number, line in enumerate(lines, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise SelectionError(f"invalid JSON at {path}:{number}") from error
                if not isinstance(value, dict):
                    raise SelectionError(f"{path}:{number} must be a mapping")
                parsed.append(value)
            if parsed:
                rows.extend(parsed)
                sources.append({"path": str(path), "sha256": _sha256(path)})
    if not rows:
        raise SelectionError("no selection JSONL records were found")
    return rows, sources


def _expected(
    campaign: Mapping[str, Any],
    *,
    roster: str,
) -> set[tuple[str, int]]:
    requests = campaign.get("datasets", {}).get("TOFU", {}).get("rosters", {}).get(roster)
    seeds = campaign.get("execution", {}).get("seeds")
    if not isinstance(requests, list) or not isinstance(seeds, list):
        raise SelectionError(f"campaign lacks TOFU {roster} or seeds")
    return {(str(request), int(seed)) for request in requests for seed in seeds}


def _allowed_parent_grid(
    runtime: Mapping[str, Any], *, setting: str, parent: str
) -> dict[str, dict[str, Any]]:
    setting_runtime = runtime.get("settings", {}).get(setting)
    if not isinstance(setting_runtime, Mapping):
        raise SelectionError(f"runtime has no setting {setting!r}")
    path = Path(str(setting_runtime.get("channel_config", "")))
    if not path.is_absolute():
        path = ROOT / path
    channel = _load(path.resolve())
    raw = channel.get("calibration", {}).get("objective_grid", {}).get(parent)
    if not isinstance(raw, list) or not raw:
        raise SelectionError(f"channel config lacks objective grid for {parent}")
    allowed = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise SelectionError(f"objective grid for {parent} contains a non-mapping")
        value = dict(item)
        key = json.dumps(value, sort_keys=True, separators=(",", ":"))
        if key in allowed:
            raise SelectionError(f"objective grid for {parent} contains duplicates")
        allowed[key] = value
    return allowed


def _freeze_metadata(
    payload: Mapping[str, Any],
    *,
    campaign_id: str,
    frozen: bool,
) -> dict[str, Any]:
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "freeze_id": (
            f"FREEZE-{campaign_id}-{digest[:12]}"
            if frozen
            else "PENDING-REVIEW-AND-FREEZE"
        ),
        "status": "frozen" if frozen else "draft",
        "frozen_at_utc": (
            datetime.now(timezone.utc).isoformat() if frozen else None
        ),
    }


def select_parents(
    rows: list[dict[str, Any]],
    *,
    campaign: Mapping[str, Any],
    runtime: Mapping[str, Any],
    setting: str,
    sources: list[dict[str, str]],
    frozen: bool,
) -> dict[str, Any]:
    expected = _expected(campaign, roster="D_cal")
    parent_cfg = runtime["parent"]
    parent_roster = tuple(
        runtime.get("parent_roster", ())
        or (
            "graddiff",
            "npo",
            "simnpo",
            "gru",
            "rmu",
            "repnoise",
            "circuit_breakers",
        )
    )
    foreign = [
        index
        for index, row in enumerate(rows)
        if row.get("campaign_phase") != "calibration"
        or row.get("setting") != setting
        or row.get("parent") not in parent_roster
    ]
    if foreign:
        raise SelectionError(
            "parent selection input contains rows outside the exact "
            f"calibration contract at indices {foreign[:10]}"
        )
    parents = {}
    diagnostics = {}
    unresolved = []
    for parent in parent_roster:
        members = [
            row
            for row in rows
            if row.get("campaign_phase") == "calibration"
            and row.get("setting") == setting
            and row.get("parent") == parent
        ]
        allowed = _allowed_parent_grid(runtime, setting=setting, parent=parent)
        by_candidate: dict[str, dict[tuple[str, int], Mapping[str, Any]]] = {}
        duplicates = []
        for row in members:
            raw_candidate = row.get("candidate_setting")
            if not isinstance(raw_candidate, Mapping):
                raise SelectionError(f"{setting}/{parent} row lacks candidate_setting")
            key_text = json.dumps(raw_candidate, sort_keys=True, separators=(",", ":"))
            if key_text not in allowed:
                raise SelectionError(
                    f"{setting}/{parent} contains a candidate outside the frozen grid"
                )
            if type(row.get("reached")) is not bool:
                raise SelectionError(
                    f"{setting}/{parent} calibration row lacks reached boolean"
                )
            _finite_number(
                row.get("mean_damage"),
                f"{setting}/{parent}.mean_damage",
            )
            _finite_number(
                row.get("cvar95_damage"),
                f"{setting}/{parent}.cvar95_damage",
            )
            step = row.get("step")
            if isinstance(step, bool) or not isinstance(step, int) or step < 1:
                raise SelectionError(
                    f"{setting}/{parent} calibration step must be positive"
                )
            run_key = (str(row.get("request")), int(row.get("seed")))
            if run_key in by_candidate.setdefault(key_text, {}):
                duplicates.append((key_text, run_key))
            by_candidate[key_text][run_key] = row
        candidate_diagnostics = []
        eligible = []
        for key_text, candidate in allowed.items():
            cells = by_candidate.get(key_text, {})
            observed = set(cells)
            complete = observed == expected and not any(
                duplicate[0] == key_text for duplicate in duplicates
            )
            feasible = complete and all(
                row.get("reached") is True
                and _finite_number(row["mean_damage"], "mean_damage")
                <= float(parent_cfg["calibration_mean_damage_max"])
                and _finite_number(row["cvar95_damage"], "cvar95_damage")
                <= float(parent_cfg["calibration_cvar95_damage_max"])
                for row in cells.values()
            )
            cvars = [
                _finite_number(row["cvar95_damage"], "cvar95_damage")
                for row in cells.values()
            ]
            means = [
                _finite_number(row["mean_damage"], "mean_damage")
                for row in cells.values()
            ]
            item = {
                "candidate_setting": candidate,
                "complete": complete,
                "feasible": feasible,
                "missing_runs": sorted(expected - observed),
                "extra_runs": sorted(observed - expected),
                "worst_cvar95_damage": max(cvars) if cvars else None,
                "mean_damage": sum(means) / len(means) if means else None,
                "max_step": max(
                    (int(row["step"]) for row in cells.values()), default=None
                ),
            }
            candidate_diagnostics.append(item)
            if feasible:
                eligible.append(item)
        if eligible:
            winner = min(
                eligible,
                key=lambda item: (
                    float(item["worst_cvar95_damage"]),
                    float(item["mean_damage"]),
                    int(item["max_step"]),
                    float(item["candidate_setting"]["lr"]),
                    json.dumps(item["candidate_setting"], sort_keys=True),
                ),
            )
            parents[parent] = winner["candidate_setting"]
        else:
            parents[parent] = None
            unresolved.append(parent)
        diagnostics[parent] = candidate_diagnostics
    if frozen and unresolved:
        raise SelectionError(
            "cannot freeze unresolved parent calibration: " + ", ".join(unresolved)
        )
    body = {
        "schema_version": 1,
        "contract": "tofu-pdf-v4-parent-freeze",
        "source_campaign": campaign["campaign_id"],
        "setting": setting,
        "selection_rule": "all-cell-reach-then-minimax-cvar-mean-damage-steps-lr",
        "frozen_before_prediction": frozen,
        "parents": parents,
        "unresolved": unresolved,
        "development_artifacts": sources,
        "diagnostics": diagnostics,
    }
    return {
        **body,
        **_freeze_metadata(
            body,
            campaign_id=str(campaign["campaign_id"]),
            frozen=frozen,
        ),
    }


def select_claim_parameters(
    rows: list[dict[str, Any]],
    *,
    campaign: Mapping[str, Any],
    evidence: Mapping[str, Any],
    runtime: Mapping[str, Any],
    setting: str,
    sources: list[dict[str, str]],
    frozen: bool,
) -> dict[str, Any]:
    pred_expected = _expected(campaign, roster="D_pred")
    prot_expected = _expected(campaign, roster="D_prot")
    settings = {
        item["id"]: item
        for item in evidence.get("settings", [])
        if isinstance(item, Mapping)
    }
    primary = settings.get(setting)
    if not isinstance(primary, Mapping):
        raise SelectionError(f"unknown setting {setting!r}")
    foreign = [
        index
        for index, row in enumerate(rows)
        if row.get("setting") != setting
        or row.get("campaign_phase") != "development"
        or row.get("selection_kind") not in {"prediction", "protection"}
        or row.get("parent") not in primary["parents"]
    ]
    if foreign:
        raise SelectionError(
            "claim selection input contains rows outside the exact "
            f"development contract at indices {foreign[:10]}"
        )
    fallback_alpha = float(runtime["selection"]["fallback_alpha"])
    fallback_kp = int(runtime["selection"]["fallback_Kp"])
    fallback_control = str(runtime["selection"]["fallback_control"])
    if fallback_control not in runtime["probe"]["simple_controls"]:
        raise SelectionError("selection.fallback_control is not a declared control")
    prediction_rows = [
        row
        for row in rows
        if row.get("setting") == setting
        and row.get("selection_kind") == "prediction"
    ]
    protection_rows = [
        row
        for row in rows
        if row.get("setting") == setting
        and row.get("selection_kind") == "protection"
    ]

    selected_primary = {}
    diagnostics = {}
    unresolved = []
    for parent in primary["parents"]:
        parent_prediction = [
            row for row in prediction_rows if row.get("parent") == parent
        ]
        flat_prediction = []
        control_rows = []
        for row in parent_prediction:
            grid = row.get("alpha_grid")
            if not isinstance(grid, list):
                raise SelectionError(f"{setting}/{parent} prediction grid is missing")
            observed_alphas = {
                float(item.get("alpha"))
                for item in grid
                if isinstance(item, Mapping)
            }
            expected_alphas = {
                float(value) for value in runtime["probe"]["alpha_grid"]
            }
            if observed_alphas != expected_alphas or len(grid) != len(expected_alphas):
                raise SelectionError(
                    f"{setting}/{parent} prediction alpha grid is not exact"
                )
            if row.get("simple_controls") != runtime["probe"]["simple_controls"]:
                raise SelectionError(
                    f"{setting}/{parent} simple-control roster is not exact"
                )
            for item in grid:
                if not isinstance(item, Mapping):
                    raise SelectionError("prediction alpha grid contains a non-mapping")
                flat_prediction.append(
                    {
                        **item,
                        "campaign_phase": "development",
                        "selector_type": "mixture",
                        "request": row["request"],
                        "seed": int(row["seed"]),
                        "reached": row.get("reached") is True,
                    }
                )
            control_rows.append(
                {
                    "campaign_phase": "development",
                    "request": row["request"],
                    "seed": int(row["seed"]),
                    "control_spearman": row.get("control_spearman"),
                }
            )
        prediction = select_prediction_alpha(
            flat_prediction,
            alpha_grid=runtime["probe"]["alpha_grid"],
            expected_run_keys=pred_expected,
            min_reached_requests=int(
                runtime["selection"]["prediction_min_reached_requests"]
            ),
            fallback_alpha=fallback_alpha,
        )
        control = select_simple_control(
            control_rows,
            expected_run_keys=pred_expected,
            controls=runtime["probe"]["simple_controls"],
        )

        parent_protection = [
            row for row in protection_rows if row.get("parent") == parent
        ]
        flat_protection = []
        for row in parent_protection:
            grid = row.get("grid")
            if not isinstance(grid, list):
                raise SelectionError(f"{setting}/{parent} protection grid is missing")
            expected_pairs = {
                (float(alpha), int(kp))
                for alpha in runtime["probe"]["alpha_grid"]
                for kp in runtime["protection"]["Kp_grid"]
            }
            observed_pairs = {
                (float(item.get("alpha")), int(item.get("Kp")))
                for item in grid
                if isinstance(item, Mapping)
            }
            if observed_pairs != expected_pairs or len(grid) != len(expected_pairs):
                raise SelectionError(
                    f"{setting}/{parent} protection grid is not exact"
                )
            for item in grid:
                if not isinstance(item, Mapping):
                    raise SelectionError("protection grid contains a non-mapping")
                flat_protection.append(
                    {
                        **item,
                        "campaign_phase": "development",
                        "request": row["request"],
                        "seed": int(row["seed"]),
                    }
                )
        protection = select_protection_pair(
            flat_protection,
            alpha_grid=runtime["probe"]["alpha_grid"],
            kp_grid=runtime["protection"]["Kp_grid"],
            expected_run_keys=prot_expected,
        )
        prediction_resolved = bool(prediction["resolved"] and control["resolved"])
        protection_resolved = bool(protection["resolved"])
        if not prediction_resolved:
            unresolved.append(f"{setting}/{parent}/prediction")
        if not protection_resolved:
            unresolved.append(f"{setting}/{parent}/protection")
        selected_primary[parent] = {
            "prediction": {
                "valid": prediction_resolved,
                "fallback": not prediction_resolved,
                "alpha": float(prediction["alpha"]),
                "simple_control": (
                    control["control"]
                    if control["resolved"]
                    else fallback_control
                ),
            },
            "protection": {
                "valid": protection_resolved,
                "fallback": not protection_resolved,
                "alpha": (
                    float(protection["alpha"])
                    if protection_resolved
                    else fallback_alpha
                ),
                "Kp": (
                    int(protection["Kp"])
                    if protection_resolved
                    else fallback_kp
                ),
            },
        }
        diagnostics[parent] = {
            "prediction": prediction,
            "simple_control": control,
            "protection": protection,
        }

    selections = {}
    for setting_id, setting_config in settings.items():
        if setting_id == setting:
            selections[setting_id] = selected_primary
            continue
        selections[setting_id] = {
            parent: {
                "prediction": {
                    "valid": False,
                    "fallback": True,
                    "alpha": fallback_alpha,
                    "simple_control": fallback_control,
                },
                "protection": {
                    "valid": False,
                    "fallback": True,
                    "alpha": fallback_alpha,
                    "Kp": fallback_kp,
                },
            }
            for parent in setting_config["parents"]
        }
    body = {
        "schema_version": 1,
        "source_campaign": campaign["campaign_id"],
        "frozen_before_target": frozen,
        "selection_rule": {
            "prediction": "equal-request-spearman-top-q-midpoint-smaller",
            "simple_control": "mean-spearman-then-declared-order",
            "protection": "all-cell-feasible-minimax-cvar95",
        },
        "selections": selections,
        "unresolved_primary": unresolved,
        "development_artifacts": sources,
        "development_diagnostics": {setting: diagnostics},
    }
    return {
        **body,
        **_freeze_metadata(
            body,
            campaign_id=str(campaign["campaign_id"]),
            frozen=frozen,
        ),
    }


def _atomic_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(payload), sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("parent", "claims"), required=True)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--setting", default="tofu_qwen25_1p5b")
    parser.add_argument(
        "--campaign", type=Path, default=ROOT / "configs/paper/campaign.yaml"
    )
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "configs/paper/evidence.yaml"
    )
    parser.add_argument(
        "--runtime", type=Path, default=ROOT / "configs/paper/tofu_v4.yaml"
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        campaign = _load(args.campaign.resolve())
        evidence = _load(args.evidence.resolve())
        runtime = _load(args.runtime.resolve())
        rows, sources = _records(path.resolve() for path in args.input)
        if args.kind == "parent":
            payload = select_parents(
                rows,
                campaign=campaign,
                runtime=runtime,
                setting=args.setting,
                sources=sources,
                frozen=args.freeze,
            )
        else:
            payload = select_claim_parameters(
                rows,
                campaign=campaign,
                evidence=evidence,
                runtime=runtime,
                setting=args.setting,
                sources=sources,
                frozen=args.freeze,
            )
        destination = args.out.resolve()
        _atomic_yaml(destination, payload)
        print(f"wrote {args.kind} selection: {destination}")
        print(f"status: {payload['status']}")
        return 0
    except (SelectionError, ValueError, KeyError, TypeError) as error:
        print(f"TOFU selection failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
