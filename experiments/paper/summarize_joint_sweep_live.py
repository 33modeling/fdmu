#!/usr/bin/env python3
"""Build a read-only live snapshot of an in-progress joint development sweep."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paper.run_joint_dev_sweep import candidate_score, evaluate_cell


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return value


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _current_trial(
    joint_root: Path,
    events: list[dict[str, Any]],
) -> tuple[str | None, Path | None, list[dict[str, Any]]]:
    for index in range(len(events) - 1, -1, -1):
        row = events[index]
        if row.get("event") == "trial_started":
            trial_id = str(row.get("trial_id", ""))
            raw = row.get("trial_dir")
            if trial_id and isinstance(raw, str) and raw:
                return trial_id, Path(raw), events[index:]
    trial_dirs = sorted(
        (joint_root / "trials").glob("*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not trial_dirs:
        return None, None, []
    trial_dir = trial_dirs[0]
    trial_json = trial_dir / "trial.json"
    trial_id = trial_dir.name.split("--", 1)[0]
    if trial_json.is_file():
        try:
            trial_id = str(_load_json(trial_json).get("trial", {}).get("id", trial_id))
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    return trial_id, trial_dir, []


def _latest_attempts(trial_dir: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, tuple[int, dict[str, Any]]] = {}
    for path in trial_dir.glob("logs/units/*/attempt-*.json"):
        try:
            payload = _load_json(path)
            attempt = int(payload.get("attempt", 0))
            identity = str(payload.get("unit") or path.parent.name)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        previous = latest.get(identity)
        if previous is None or attempt > previous[0]:
            latest[identity] = (attempt, payload)
    return {identity: value[1] for identity, value in latest.items()}


def _manifest_units(trial_dir: Path) -> list[dict[str, Any]]:
    path = trial_dir / "manifest.yaml"
    if not path.is_file():
        return []
    value = _load_yaml(path).get("units")
    return [dict(item) for item in value or [] if isinstance(item, Mapping)]


def _partial_cells(
    trial_dir: Path,
    *,
    expected_draws: list[str],
    mean_margin: float,
    cvar_margin: float,
) -> dict[str, Any]:
    rows = []
    invalid = []
    for path in sorted(trial_dir.glob("units/*/protection_diagnostics.json")):
        try:
            diagnostic = _load_json(path)
            result = evaluate_cell(
                diagnostic,
                expected_draws=expected_draws,
                mean_margin=mean_margin,
                cvar_margin=cvar_margin,
            )
        except Exception as error:
            invalid.append(
                {
                    "path": str(path),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            continue
        rows.append(
            {
                "cell": (
                    f"{diagnostic.get('parent')}/{diagnostic.get('request')}/"
                    f"seed-{diagnostic.get('seed')}"
                ),
                "passed": result["passed"],
                "joint_feasible": result["joint_feasible"],
                "joint_mean_damage": result["joint_mean_damage"],
                "joint_cvar95_damage": result["joint_cvar95_damage"],
                "comparison_wins": sum(
                    item.get("joint_wins_constrained") is True
                    for item in result["comparisons"]
                ),
                "comparison_total": len(result["comparisons"]),
            }
        )
    return {
        "descriptive_only": True,
        "incomplete_roster_must_not_drive_selection": True,
        "evaluated_cells": len(rows),
        "passing_cells": sum(row["passed"] is True for row in rows),
        "joint_feasible_cells": sum(row["joint_feasible"] is True for row in rows),
        "comparison_wins": sum(int(row["comparison_wins"]) for row in rows),
        "comparison_total": sum(int(row["comparison_total"]) for row in rows),
        "mean_joint_damage": (
            statistics.fmean(float(row["joint_mean_damage"]) for row in rows)
            if rows
            else None
        ),
        "mean_joint_cvar95": (
            statistics.fmean(float(row["joint_cvar95_damage"]) for row in rows)
            if rows
            else None
        ),
        "cells": rows,
        "invalid_artifacts": invalid,
    }


def _completed_trials(joint_root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted((joint_root / "trials").glob("*/joint_comparison.json")):
        try:
            result = _load_json(path)
            score = candidate_score(result)
        except Exception as error:
            rows.append(
                {
                    "trial": path.parent.name,
                    "invalid": f"{type(error).__name__}: {error}",
                }
            )
            continue
        rows.append(
            {
                "trial": result.get("trial_id", path.parent.name),
                "passed": result.get("passed") is True,
                "groups": f"{score['groups_passed']}/{score['groups_required']}",
                "parents": f"{score['parents_passed']}/{score['parents_required']}",
                "cells": f"{score['cells_passed']}/{score['cells_required']}",
                "feasible_cells": score["joint_feasible_cells"],
                "blocking": score["blocking_count"],
                "shortfall": score["total_shortfall"],
                "path": str(path),
            }
        )
    return rows


def _duration_summary(
    attempts: Mapping[str, Mapping[str, Any]],
    *,
    total: int,
    gpu_count: int,
) -> dict[str, Any]:
    successful = [
        float(value["duration_seconds"])
        for value in attempts.values()
        if value.get("returncode") == 0
        and isinstance(value.get("duration_seconds"), (int, float))
    ]
    completed = len(successful)
    remaining = max(0, total - completed)
    mean = statistics.fmean(successful) if successful else None
    median = statistics.median(successful) if successful else None
    eta = (
        math.ceil(remaining / max(1, gpu_count)) * mean
        if mean is not None
        else None
    )
    return {
        "successful_units": completed,
        "failed_units": sum(value.get("returncode") not in (None, 0) for value in attempts.values()),
        "mean_unit_seconds": mean,
        "median_unit_seconds": median,
        "trial_eta_seconds_at_observed_rate": eta,
        "eta_scope": "current_trial_only",
    }


def build_snapshot(joint_root: Path, spec_path: Path) -> dict[str, Any]:
    spec = _load_yaml(spec_path)
    campaign_path = Path(str(spec["paths"]["campaign"]))
    if not campaign_path.is_absolute():
        campaign_path = (Path.cwd() / campaign_path).resolve()
    campaign = _load_yaml(campaign_path)
    event_rows = _events(joint_root / "events.jsonl")
    trial_id, trial_dir, current_events = _current_trial(joint_root, event_rows)
    trials = [str(item["id"]) for item in spec["trials"]]
    completed = _completed_trials(joint_root)
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "contract": "tofu-joint-sweep-live-status",
        "generated_at_utc": _utc_now(),
        "development_only": True,
        "target_used": False,
        "authoritative_for_selection": False,
        "warning": (
            "Partial cells are descriptive only. BEST.json or a complete "
            "joint_comparison.json is required for a selection decision."
        ),
        "paths": {
            "joint_root": str(joint_root),
            "events": str(joint_root / "events.jsonl"),
            "summary": str(joint_root / "summary.csv"),
            "best": str(joint_root / "BEST.json"),
            "sweep_status": str(joint_root / "SWEEP_STATUS.json"),
        },
        "sweep": {
            "declared_trials": len(trials),
            "completed_trials": len(
                [row for row in completed if "invalid" not in row]
            ),
            "best_exists": (joint_root / "BEST.json").is_file(),
            "terminal_status_exists": (joint_root / "SWEEP_STATUS.json").is_file(),
            "completed": completed,
        },
        "current_trial": None,
    }
    if trial_id is None or trial_dir is None:
        return snapshot

    manifest_units = _manifest_units(trial_dir)
    attempts = _latest_attempts(trial_dir)
    successful_ids = {
        identity for identity, value in attempts.items() if value.get("returncode") == 0
    }
    started = {
        str(row.get("unit"))
        for row in current_events
        if row.get("event") == "unit_started"
    }
    finished = {
        str(row.get("unit"))
        for row in current_events
        if row.get("event") == "unit_finished"
    }
    running = sorted(started - finished)
    stop = spec["stop"]
    partial = _partial_cells(
        trial_dir,
        expected_draws=[
            str(item) for item in campaign["execution"]["repeated_random_draws"]
        ],
        mean_margin=float(stop["mean_damage_margin"]),
        cvar_margin=float(stop["cvar95_damage_margin"]),
    )
    duration = _duration_summary(
        attempts,
        total=len(manifest_units),
        gpu_count=len(spec["gpus"]),
    )
    total = len(manifest_units)
    snapshot["current_trial"] = {
        "id": trial_id,
        "index": trials.index(trial_id) + 1 if trial_id in trials else None,
        "directory": str(trial_dir),
        "total_units": total,
        "completed_units": len(successful_ids),
        "running_units": running,
        "pending_units": max(0, total - len(successful_ids) - len(running)),
        "progress_fraction": len(successful_ids) / total if total else 0.0,
        **duration,
        "partial_result": partial,
    }
    return snapshot


def _fmt_duration(value: object) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "unknown"
    seconds = int(value)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {seconds}s"


def render_markdown(snapshot: Mapping[str, Any]) -> str:
    sweep = snapshot["sweep"]
    current = snapshot.get("current_trial")
    lines = [
        "# Live Joint Sweep Status",
        "",
        f"- Updated: `{snapshot['generated_at_utc']}`",
        "- Scope: development-only, partial and non-authoritative",
        f"- Completed trials: `{sweep['completed_trials']}/{sweep['declared_trials']}`",
        f"- BEST exists: `{str(sweep['best_exists']).lower()}`",
        "",
    ]
    if isinstance(current, Mapping):
        partial = current["partial_result"]
        lines.extend(
            [
                "## Current Trial",
                "",
                f"- Trial: `{current['index']}/{sweep['declared_trials']} {current['id']}`",
                f"- Units: `{current['completed_units']}/{current['total_units']}` complete, "
                f"`{len(current['running_units'])}` running, "
                f"`{current['pending_units']}` pending, `{current['failed_units']}` failed",
                f"- Mean unit time: `{_fmt_duration(current['mean_unit_seconds'])}`",
                f"- Current-trial ETA: "
                f"`{_fmt_duration(current['trial_eta_seconds_at_observed_rate'])}`",
                "",
                "## Partial Cell Results",
                "",
                "> Descriptive only. The roster is incomplete and these values must not "
                "select a trial.",
                "",
                f"- Evaluated cells: `{partial['evaluated_cells']}/{current['total_units']}`",
                f"- Joint feasible: `{partial['joint_feasible_cells']}`",
                f"- Passing cells: `{partial['passing_cells']}`",
                f"- Constrained comparison wins: "
                f"`{partial['comparison_wins']}/{partial['comparison_total']}`",
                f"- Mean joint damage: `{partial['mean_joint_damage']}`",
                f"- Mean joint CVaR95: `{partial['mean_joint_cvar95']}`",
                "",
            ]
        )
    completed = sweep["completed"]
    if completed:
        lines.extend(
            [
                "## Completed Trials",
                "",
                "| Trial | Passed | Groups | Parents | Cells | Blocking | Shortfall |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in completed:
            if "invalid" in row:
                lines.append(
                    f"| {row['trial']} | invalid | - | - | - | - | - |"
                )
            else:
                lines.append(
                    f"| {row['trial']} | {row['passed']} | {row['groups']} | "
                    f"{row['parents']} | {row['cells']} | {row['blocking']} | "
                    f"{row['shortfall']:.6g} |"
                )
        lines.append("")
    lines.extend(
        [
            "## Source Files",
            "",
            f"- Events: `{snapshot['paths']['events']}`",
            f"- Completed-trial summary: `{snapshot['paths']['summary']}`",
            f"- Winner: `{snapshot['paths']['best']}`",
            "",
        ]
    )
    return "\n".join(lines)


def _history_key(snapshot: Mapping[str, Any]) -> str:
    reduced = json.loads(json.dumps(snapshot))
    reduced.pop("generated_at_utc", None)
    return hashlib.sha256(
        json.dumps(reduced, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def write_snapshot(joint_root: Path, snapshot: dict[str, Any]) -> None:
    live = joint_root / "live"
    json_path = live / "LIVE_STATUS.json"
    markdown_path = live / "LIVE_STATUS.md"
    history_path = live / "history.jsonl"
    previous_key = None
    if json_path.is_file():
        try:
            previous_key = _history_key(_load_json(json_path))
        except (OSError, ValueError, json.JSONDecodeError):
            previous_key = None
    _atomic_text(
        json_path,
        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
    )
    _atomic_text(markdown_path, render_markdown(snapshot))
    key = _history_key(snapshot)
    if key != previous_key:
        live.mkdir(parents=True, exist_ok=True)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(snapshot, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = build_snapshot(args.joint_root.resolve(), args.spec.resolve())
    write_snapshot(args.joint_root.resolve(), snapshot)
    current = snapshot.get("current_trial")
    if isinstance(current, Mapping):
        print(
            f"LIVE_STATUS trial={current['index']} "
            f"units={current['completed_units']}/{current['total_units']} "
            f"running={len(current['running_units'])} "
            f"eta={_fmt_duration(current['trial_eta_seconds_at_observed_rate'])}"
        )
    else:
        print("LIVE_STATUS waiting for the first trial")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
