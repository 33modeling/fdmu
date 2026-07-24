#!/usr/bin/env python3
"""Execute and seal one exact PDF-v4 paper stage.

The model-specific unit command remains explicit in a frozen stage manifest.
This orchestrator owns the paper-level denominator: it checks the campaign
roster against the adapter registry, executes every planned unit without a
shell, validates candidate-level outputs, and writes consolidated JSONL plus
content hashes. Missing or extra units and partial comparator arms fail closed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable, Mapping

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.data.registry import AdapterNotFoundError, get_adapter  # noqa: E402


PAPER_STAGE_CONTRACT = {
    "schema_version": 1,
    "stages": ["calibration", "prediction", "protection", "target_evaluation"],
    "consumes_campaign_config": True,
    "uses_adapter_registry": True,
    "consumes_exact_roster": True,
    "executes_unit_commands": True,
    "validates_selection_inputs": True,
    "validates_candidate_level_prediction_raw": True,
    "validates_fidelity_raw": True,
    "validates_candidate_level_protection_raw": True,
}

STAGE_OUTPUTS = {
    "calibration": ("fidelity_raw",),
    "prediction": ("prediction_raw", "selection_inputs"),
    "protection": ("protection_raw", "selection_inputs"),
    "target_evaluation": ("prediction_raw", "fidelity_raw", "protection_raw"),
}
CLAIM_ARMS = {"joint", "no_repair", "repeated_random", "s0", "s1"}


class StageContractError(ValueError):
    """A frozen stage manifest or produced shard violates the paper contract."""


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise StageContractError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise StageContractError(f"{path} must contain one mapping")
    return value


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise StageContractError(f"{name} must be a non-empty string")
    return value.strip()


def _resolve(value: str, *, base: Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(value)))
    return path if path.is_absolute() else (base / path).resolve()


def _records(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise StageContractError(f"cannot read output shard {path}: {error}") from error
    result: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise StageContractError(f"invalid JSON at {path}:{line_number}: {error}") from error
        if not isinstance(value, dict):
            raise StageContractError(f"{path}:{line_number} must be a JSON object")
        result.append(value)
    if not result:
        raise StageContractError(f"output shard {path} is empty")
    return result


def _unit_key(raw: Mapping[str, Any], where: str) -> tuple[str, str, str]:
    return (
        _text(raw.get("parent"), f"{where}.parent"),
        _text(raw.get("request"), f"{where}.request"),
        _text(str(raw.get("seed", "")), f"{where}.seed"),
    )


def _number(raw: Mapping[str, Any], field: str, where: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StageContractError(f"{where}.{field} must be numeric")
    return float(value)


def _validate_selection_mapping(
    raw: Mapping[str, Any], field: str, where: str
) -> None:
    value = raw.get(field)
    if not isinstance(value, dict):
        raise StageContractError(f"{where}.{field} must be a mapping")
    valid = value.get("valid")
    fallback = value.get("fallback")
    alpha = value.get("alpha")
    if type(valid) is not bool or type(fallback) is not bool or valid == fallback:
        raise StageContractError(
            f"{where}.{field} must resolve exactly one of valid/fallback"
        )
    if valid and (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 <= float(alpha) <= 1.0
    ):
        raise StageContractError(f"{where}.{field}.alpha must be in [0, 1]")
    if fallback and alpha is not None and (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not 0.0 <= float(alpha) <= 1.0
    ):
        raise StageContractError(
            f"{where}.{field}.alpha must be null or in [0, 1] for fallback"
        )


def _validate_identity(
    row: Mapping[str, Any],
    *,
    setting: str,
    key: tuple[str, str, str],
    where: str,
) -> None:
    expected = (setting, *key)
    observed = (
        _text(row.get("setting"), f"{where}.setting"),
        _text(row.get("parent"), f"{where}.parent"),
        _text(row.get("request"), f"{where}.request"),
        _text(str(row.get("seed", "")), f"{where}.seed"),
    )
    if observed != expected:
        raise StageContractError(
            f"{where} identity {observed!r} does not match planned {expected!r}"
        )


def _validate_prediction(
    rows: list[dict[str, Any]], setting: str, key: tuple[str, str, str]
) -> None:
    seen: set[str] = set()
    statuses: set[tuple[bool, bool, bool]] = set()
    for index, row in enumerate(rows):
        where = f"prediction_raw[{index}]"
        _validate_identity(row, setting=setting, key=key, where=where)
        candidate = _text(row.get("candidate_id"), f"{where}.candidate_id")
        if candidate in seen:
            raise StageContractError(f"{where} duplicates candidate {candidate!r}")
        seen.add(candidate)
        _text(row.get("group"), f"{where}.group")
        for field in ("s0", "s1", "joint", "simple_control", "damage"):
            _number(row, field, where)
        _validate_selection_mapping(row, "prediction_selection", where)
        status = tuple(
            row.get(field)
            for field in ("profile_valid", "reached", "trajectory_completed")
        )
        if any(type(value) is not bool for value in status):
            raise StageContractError(f"{where} status fields must be booleans")
        statuses.add(status)
        checkpoint = row.get("parent_checkpoint_first_reaching")
        if row["reached"] and checkpoint is not True:
            raise StageContractError(
                f"{where} reached damage must use the first reaching checkpoint"
            )
        _text(row.get("parent_checkpoint_id"), f"{where}.parent_checkpoint_id")
    if len(seen) < 2:
        raise StageContractError("prediction_raw requires at least two candidates")
    if len(statuses) != 1:
        raise StageContractError("prediction_raw has inconsistent unit status")


def _validate_fidelity(
    rows: list[dict[str, Any]], setting: str, key: tuple[str, str, str]
) -> None:
    if len(rows) != 1:
        raise StageContractError("fidelity_raw requires exactly one row per unit")
    row = rows[0]
    _validate_identity(row, setting=setting, key=key, where="fidelity_raw[0]")
    for field in ("f_rho", "f_k"):
        _number(row, field, "fidelity_raw[0]")
    for field in (
        "perturbations_valid",
        "exact_reference_valid",
        "common_control_support",
    ):
        if type(row.get(field)) is not bool:
            raise StageContractError(f"fidelity_raw[0].{field} must be boolean")


def _validate_protection(
    rows: list[dict[str, Any]],
    setting: str,
    key: tuple[str, str, str],
    repeated_draws: tuple[str, ...],
) -> None:
    by_candidate: dict[str, set[tuple[str, str | None]]] = {}
    for index, row in enumerate(rows):
        where = f"protection_raw[{index}]"
        _validate_identity(row, setting=setting, key=key, where=where)
        candidate = _text(row.get("candidate_id"), f"{where}.candidate_id")
        _text(row.get("group"), f"{where}.group")
        arm = _text(row.get("arm"), f"{where}.arm")
        if arm not in CLAIM_ARMS:
            raise StageContractError(f"{where}.arm is not a PDF-v4 claim arm")
        draw: str | None = None
        if arm == "repeated_random":
            draw = _text(row.get("draw_id"), f"{where}.draw_id")
            if draw not in repeated_draws:
                raise StageContractError(f"{where}.draw_id is not frozen in campaign")
            if row.get("draw_complete") is not True:
                raise StageContractError(f"{where} repeated-random draw is incomplete")
        elif row.get("draw_id") is not None:
            raise StageContractError(f"{where}.draw_id is only valid for repeated_random")
        identity = arm, draw
        if identity in by_candidate.setdefault(candidate, set()):
            raise StageContractError(f"{where} duplicates arm/draw for {candidate!r}")
        by_candidate[candidate].add(identity)
        for field in (
            "damage",
            "native_retention",
            "direct_forget_margin",
            "paraphrase_forget_margin",
            "extraction_generation_margin",
            "utility_margin",
        ):
            _number(row, field, where)
        if type(row.get("feasible")) is not bool:
            raise StageContractError(f"{where}.feasible must be boolean")
        if row.get("parent_checkpoint_first_reaching") is not True:
            raise StageContractError(
                f"{where} must start from the first reaching parent checkpoint"
            )
        _text(row.get("parent_checkpoint_id"), f"{where}.parent_checkpoint_id")
        _validate_selection_mapping(row, "protection_selection", where)
    expected = {
        *((arm, None) for arm in CLAIM_ARMS - {"repeated_random"}),
        *(("repeated_random", draw) for draw in repeated_draws),
    }
    incomplete = sorted(
        candidate for candidate, observed in by_candidate.items() if observed != expected
    )
    if not by_candidate or incomplete:
        raise StageContractError(
            "protection_raw lacks an exact five-arm/repeated-draw roster for "
            f"candidates: {incomplete or '<all>'}"
        )


def _validate_selection(
    rows: list[dict[str, Any]],
    setting: str,
    key: tuple[str, str, str],
    stage: str,
) -> None:
    if len(rows) != 1:
        raise StageContractError("selection_inputs requires exactly one row per unit")
    row = rows[0]
    _validate_identity(row, setting=setting, key=key, where="selection_inputs[0]")
    if row.get("target_free") is not True:
        raise StageContractError("selection_inputs must certify target_free: true")
    if stage == "prediction":
        if row.get("selection_kind") != "prediction":
            raise StageContractError(
                "prediction selection_inputs must use selection_kind: prediction"
            )
        value = _number(row, "alpha_pred", "selection_inputs[0]")
        if not 0.0 <= value <= 1.0:
            raise StageContractError(
                "selection_inputs[0].alpha_pred must be in [0, 1]"
            )
    elif stage == "protection":
        if row.get("selection_kind") != "protection":
            raise StageContractError(
                "protection selection_inputs must use selection_kind: protection"
            )
        value = _number(row, "alpha_prot", "selection_inputs[0]")
        if not 0.0 <= value <= 1.0:
            raise StageContractError(
                "selection_inputs[0].alpha_prot must be in [0, 1]"
            )
        kp = row.get("Kp")
        if isinstance(kp, bool) or not isinstance(kp, int) or kp < 1:
            raise StageContractError(
                "selection_inputs[0].Kp must be a positive integer"
            )
    else:
        raise StageContractError(
            f"selection_inputs are not valid for stage {stage!r}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(path)


def _contracts(
    campaign: Mapping[str, Any],
    evidence: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...], tuple[str, ...], list[dict[str, Any]]]:
    if manifest.get("schema_version") != 1:
        raise StageContractError("stage manifest schema_version must be 1")
    if manifest.get("contract") != "kdd-unlearning-fail-pdf-v4-stage":
        raise StageContractError("stage manifest contract is not PDF v4")
    if manifest.get("status") != "frozen":
        raise StageContractError("stage manifest status must be frozen")
    if manifest.get("campaign_id") != campaign.get("campaign_id"):
        raise StageContractError("stage manifest campaign_id mismatch")
    stage = _text(manifest.get("stage"), "manifest.stage")
    if stage not in STAGE_OUTPUTS:
        raise StageContractError(f"unknown stage {stage!r}")
    setting_id = _text(manifest.get("setting"), "manifest.setting")
    settings = {
        item.get("id"): item
        for item in evidence.get("settings", [])
        if isinstance(item, dict)
    }
    setting = settings.get(setting_id)
    if not isinstance(setting, dict):
        raise StageContractError(f"unknown evidence setting {setting_id!r}")
    stage_cfg = campaign.get("stages", {}).get(stage)
    dataset_cfg = campaign.get("datasets", {}).get(setting.get("dataset"))
    execution = campaign.get("execution")
    if not isinstance(stage_cfg, dict) or not isinstance(dataset_cfg, dict):
        raise StageContractError("campaign lacks stage or dataset contract")
    if not isinstance(execution, dict):
        raise StageContractError("campaign.execution is missing")
    adapter_name = _text(dataset_cfg.get("adapter"), "dataset.adapter")
    try:
        adapter = get_adapter(adapter_name)
    except AdapterNotFoundError as error:
        raise StageContractError(str(error)) from error
    capability = _text(stage_cfg.get("adapter_capability"), "stage.adapter_capability")
    if not adapter.capabilities.supports(capability):
        raise StageContractError(
            f"adapter {adapter_name!r} does not support {capability!r}"
        )
    roster_name = _text(stage_cfg.get("roster"), "stage.roster")
    roster = dataset_cfg.get("rosters", {}).get(roster_name)
    if not isinstance(roster, list) or not roster:
        raise StageContractError(f"{roster_name} must be a non-empty exact roster")
    requests = tuple(_text(value, f"{roster_name}[]") for value in roster)
    if len(set(requests)) != len(requests) or any(
        not adapter.accepts_roster_id(value) for value in requests
    ):
        raise StageContractError(f"{roster_name} contains duplicate or invalid request ids")
    seeds = tuple(str(value) for value in execution.get("seeds", []))
    draws = tuple(str(value) for value in execution.get("repeated_random_draws", []))
    parents = tuple(str(value) for value in setting.get("parents", []))
    if not seeds or not draws or not parents:
        raise StageContractError("campaign seeds, random draws, and parents must be non-empty")
    units = manifest.get("units")
    if not isinstance(units, list):
        raise StageContractError("stage manifest units must be a list")
    expected = {(parent, request, seed) for parent in parents for request in requests for seed in seeds}
    observed: dict[tuple[str, str, str], dict[str, Any]] = {}
    for index, unit in enumerate(units):
        if not isinstance(unit, dict):
            raise StageContractError(f"manifest.units[{index}] must be a mapping")
        key = _unit_key(unit, f"manifest.units[{index}]")
        if key in observed:
            raise StageContractError(f"duplicate stage unit {key!r}")
        observed[key] = unit
    if set(observed) != expected:
        missing = sorted(expected - set(observed))
        extra = sorted(set(observed) - expected)
        raise StageContractError(f"stage unit roster mismatch; missing={missing}, extra={extra}")
    return stage, setting_id, draws, STAGE_OUTPUTS[stage], [observed[key] for key in sorted(expected)]


def execute(args: argparse.Namespace) -> Path:
    campaign_path = Path(args.campaign).resolve()
    evidence_path = Path(args.evidence).resolve()
    manifest_path = Path(args.manifest).resolve()
    campaign = _load_yaml(campaign_path)
    evidence = _load_yaml(evidence_path)
    manifest = _load_yaml(manifest_path)
    stage, setting, draws, output_kinds, units = _contracts(campaign, evidence, manifest)
    if args.action == "validate":
        print(f"valid PDF-v4 stage manifest: {stage}/{setting}, units={len(units)}")
        return Path(args.output_dir).resolve()

    collected = {kind: [] for kind in output_kinds}
    for index, unit in enumerate(units):
        key = _unit_key(unit, f"manifest.units[{index}]")
        command = unit.get("command")
        if args.action == "run":
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(value, str) or not value for value in command)
            ):
                raise StageContractError(f"unit {key!r} command must be a non-empty argv list")
            subprocess.run(command, cwd=ROOT, check=True)
        raw_outputs = unit.get("outputs")
        if not isinstance(raw_outputs, dict) or set(raw_outputs) != set(output_kinds):
            raise StageContractError(
                f"unit {key!r} outputs must be exactly {list(output_kinds)}"
            )
        for kind in output_kinds:
            path = _resolve(_text(raw_outputs[kind], f"unit {key!r}.{kind}"), base=manifest_path.parent)
            rows = _records(path)
            if kind == "prediction_raw":
                _validate_prediction(rows, setting, key)
            elif kind == "fidelity_raw":
                _validate_fidelity(rows, setting, key)
            elif kind == "protection_raw":
                _validate_protection(rows, setting, key, draws)
            else:
                _validate_selection(rows, setting, key, stage)
            collected[kind].extend(rows)

    output_dir = Path(args.output_dir).resolve()
    artifacts: dict[str, dict[str, Any]] = {}
    for kind, rows in collected.items():
        path = output_dir / f"{kind}.jsonl"
        _write_jsonl(path, rows)
        artifacts[kind] = {
            "path": str(path),
            "records": len(rows),
            "sha256": _sha256(path),
        }
    summary = {
        "schema_version": 1,
        "contract": "kdd-unlearning-fail-pdf-v4-stage-output",
        "campaign_id": campaign.get("campaign_id"),
        "stage": stage,
        "setting": setting,
        "units": len(units),
        "source_manifest_sha256": _sha256(manifest_path),
        "artifacts": artifacts,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "stage_manifest.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"sealed PDF-v4 stage output: {summary_path}")
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", default=str(ROOT / "configs/paper/campaign.yaml"))
    parser.add_argument("--evidence", default=str(ROOT / "configs/paper/evidence.yaml"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--action", choices=("validate", "verify", "run"), default="run")
    return parser.parse_args()


def main() -> int:
    try:
        execute(parse_args())
    except (StageContractError, subprocess.CalledProcessError) as error:
        print(f"paper stage failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
