#!/usr/bin/env python3
"""Seal a development-only objective recommendation without operator input."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import yaml


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one mapping")
    return value


def _winner_key(row: dict, recall_max: float) -> tuple:
    setting = row["setting"]
    return (
        float(row["forget_recall_max"]) > recall_max,
        float(row["forget_recall_max"]),
        float(row["mean_dnll"]),
        float(row["cvar05_dnll"]),
        int(setting["steps"]),
        float(setting["lr"]),
    )


def build_freeze(config: dict, recommendation: dict) -> dict:
    campaign = str(config["campaign_id"])
    if recommendation.get("source_campaign") != campaign:
        raise ValueError(
            f"recommendation belongs to {recommendation.get('source_campaign')!r}, "
            f"not {campaign!r}"
        )
    expected_models = [
        str(model["id"])
        for model in config["models"]
        if model.get("enabled", True)
    ]
    expected_objectives = list(config["audit"]["objectives"])
    expected_runs = (
        len(config["calibration"]["authors"])
        * len(config["calibration"]["seeds"])
    )
    diagnostics = recommendation.get("development_diagnostics")
    if not isinstance(diagnostics, list):
        raise ValueError("recommendation lacks development_diagnostics")
    recall_max = float(config["calibration"]["selection"]["forget_recall_max"])

    models: dict[str, dict] = {}
    statuses: dict[str, dict] = {}
    selected_rows: dict[str, dict] = {}
    for model in expected_models:
        models[model] = {}
        statuses[model] = {}
        for objective in expected_objectives:
            complete = [
                row
                for row in diagnostics
                if row.get("model") == model
                and row.get("objective") == objective
                and int(row.get("n_runs", -1)) == expected_runs
            ]
            if not complete:
                raise ValueError(
                    f"no complete calibration setting for {model}/{objective}"
                )
            strict = [row for row in complete if row.get("eligible") is True]
            winner = min(
                strict,
                key=lambda row: (
                    float(row["mean_dnll"]),
                    float(row["cvar05_dnll"]),
                    int(row["setting"]["steps"]),
                    float(row["setting"]["lr"]),
                ),
            ) if strict else min(
                complete,
                key=lambda row: _winner_key(row, recall_max),
            )
            models[model][objective] = dict(winner["setting"])
            statuses[model][objective] = (
                "strict_eligible" if strict else "best_observed_ineligible"
            )
            selected_rows[f"{model}/{objective}"] = {
                key: winner[key]
                for key in (
                    "setting_id",
                    "eligible",
                    "forget_recall_max",
                    "mean_dnll",
                    "cvar05_dnll",
                    "n_runs",
                )
            }

    identity = {
        "source_campaign": campaign,
        "models": models,
        "selection_status": statuses,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "freeze_id": f"AUTO-{campaign}-{digest[:12]}",
        "status": "frozen",
        "frozen_before_audit": True,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_campaign": campaign,
        "selection_rule": config["calibration"]["selection"],
        "selection_policy": (
            "Choose the lowest-damage strict-eligible development setting. "
            "If none is eligible, freeze the best complete observed setting "
            "by reach, damage, steps, and learning rate and label it ineligible."
        ),
        "models": models,
        "selection_status": statuses,
        "selected_development_rows": selected_rows,
        "unresolved": [],
        "selection_sha256": digest,
    }


def _write_once(path: Path, payload: dict) -> str:
    if path.exists():
        existing = _load(path)
        comparable = (
            "freeze_id",
            "source_campaign",
            "models",
            "selection_status",
            "selection_sha256",
        )
        if any(existing.get(key) != payload.get(key) for key in comparable):
            raise RuntimeError(
                f"existing freeze differs from current development selection: {path}; "
                "use a new RUN_ROOT"
            )
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return "created"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--recommendation", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        payload = build_freeze(
            _load(args.config.resolve()),
            _load(args.recommendation.resolve()),
        )
        state = _write_once(args.out.resolve(), payload)
        fallback = sum(
            status == "best_observed_ineligible"
            for values in payload["selection_status"].values()
            for status in values.values()
        )
        print(
            f"[FREEZE] {state} id={payload['freeze_id']} "
            f"fallback_objectives={fallback} path={args.out.resolve()}"
        )
        return 0
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        print(
            f"automatic freeze failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
