#!/usr/bin/env python3
"""Continue a frozen joint development winner through Table 1 LaTeX."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paper.run_joint_dev_sweep import (  # noqa: E402
    SweepError,
    _EventLog,
    _absolute_executable,
    _run_lanes,
    _unit_completion_status,
)


class FinalizationError(ValueError):
    """A joint winner cannot be safely continued to target evaluation."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalizationError(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise FinalizationError(f"{path} must contain one mapping")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise FinalizationError(f"cannot read YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise FinalizationError(f"{path} must contain one mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise FinalizationError(f"cannot read JSONL {path}: {error}") from error
    if any(not isinstance(row, dict) for row in rows):
        raise FinalizationError(f"{path} must contain only JSON objects")
    return rows


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise FinalizationError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise FinalizationError(f"{name} must be finite")
    return number


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2.0
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise FinalizationError("split-half fidelity requires paired non-trivial scores")
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_mean = sum(left_rank) / len(left_rank)
    right_mean = sum(right_rank) / len(right_rank)
    numerator = sum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_rank, right_rank)
    )
    left_var = sum((value - left_mean) ** 2 for value in left_rank)
    right_var = sum((value - right_mean) ** 2 for value in right_rank)
    if left_var == 0.0 or right_var == 0.0:
        raise FinalizationError("split-half fidelity scores have zero rank variance")
    return numerator / math.sqrt(left_var * right_var)


def _split_half_rho(profile_path: Path) -> tuple[float, dict[str, object]]:
    profile = _load_json(profile_path)
    probe = profile.get("probe")
    gradient = profile.get("gradient")
    if not isinstance(probe, Mapping) or not isinstance(gradient, Mapping):
        raise FinalizationError(f"profile lacks probe/gradient mappings: {profile_path}")
    artifacts = gradient.get("artifacts")
    scores = gradient.get("scores")
    if not isinstance(artifacts, Mapping) or not isinstance(scores, Mapping):
        raise FinalizationError(f"profile lacks loss-shake artifacts: {profile_path}")
    responses = artifacts.get("direction_responses")
    dimension = artifacts.get("block_dimension")
    if (
        artifacts.get("schema") != "loss-shake-responses-v1"
        or not isinstance(responses, list)
        or len(responses) < 2
        or len(responses) % 2
        or isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 1
    ):
        raise FinalizationError(f"profile has invalid loss-shake response bank: {profile_path}")
    candidate_ids = sorted(str(value) for value in scores)
    parsed: list[dict[str, float]] = []
    expected = set(candidate_ids)
    for index, response in enumerate(responses):
        if not isinstance(response, Mapping) or set(response) != expected:
            raise FinalizationError(
                f"profile response {index} has inconsistent candidates: {profile_path}"
            )
        parsed.append(
            {
                candidate: _finite_number(
                    response[candidate],
                    f"{profile_path}.direction_responses[{index}].{candidate}",
                )
                for candidate in candidate_ids
            }
        )

    def energy(bank: list[dict[str, float]]) -> list[float]:
        return [
            dimension
            * sum(row[candidate] ** 2 for row in bank)
            / len(bank)
            for candidate in candidate_ids
        ]

    midpoint = len(parsed) // 2
    full = energy(parsed)
    observed = [
        _finite_number(scores[candidate], f"{profile_path}.scores.{candidate}")
        for candidate in candidate_ids
    ]
    if any(
        not math.isclose(left, right, rel_tol=1e-6, abs_tol=1e-10)
        for left, right in zip(full, observed)
    ):
        raise FinalizationError(
            f"profile scores do not match the persisted response bank: {profile_path}"
        )
    return (
        _spearman(energy(parsed[:midpoint]), energy(parsed[midpoint:])),
        {
            "R": len(parsed),
            "eta": probe.get("eta"),
            "norm_eta": probe.get("norm_eta"),
            "seed": probe.get("seed"),
            "block": probe.get("block"),
        },
    )


def _percentile(values: list[float], probability: float) -> float:
    if not values:
        raise FinalizationError("cannot take a percentile of an empty sample")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def build_setting_fidelity_summary(
    *,
    manifest_path: Path,
    fidelity_raw_path: Path,
    setting: str,
    output_path: Path,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    alpha: float,
) -> Path:
    """Reject the superseded attempt to certify RQ2 on target support."""
    raise FinalizationError(
        "target-support diagnostics cannot produce the setting-level RQ2 "
        "certificate; run the declared fidelity-support stage instead"
    )

    # Unreachable for one compatibility cycle so stale callers get the
    # protocol error above instead of silently changing evidence.
    if bootstrap_replicates < 1:
        raise FinalizationError("fidelity bootstrap_replicates must be positive")
    if not 0.0 < alpha < 1.0:
        raise FinalizationError("fidelity alpha must be in (0, 1)")
    manifest = _load_yaml(manifest_path)
    if manifest.get("stage") != "target_evaluation" or manifest.get("setting") != setting:
        raise FinalizationError("fidelity summary requires the exact target manifest")
    raw_units = manifest.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise FinalizationError("target manifest has no units")

    units: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for index, unit in enumerate(raw_units):
        if not isinstance(unit, Mapping):
            raise FinalizationError(f"target manifest unit {index} is invalid")
        key = (str(unit.get("parent")), str(unit.get("request")), int(unit.get("seed")))
        if key in units:
            raise FinalizationError(f"duplicate target manifest unit: {key}")
        units[key] = unit

    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    for index, row in enumerate(_read_jsonl(fidelity_raw_path)):
        if row.get("setting") != setting:
            raise FinalizationError(
                f"fidelity row {index} has setting {row.get('setting')!r}"
            )
        key = (str(row.get("parent")), str(row.get("request")), int(row.get("seed")))
        if key in rows:
            raise FinalizationError(f"duplicate fidelity row: {key}")
        f_rho = _finite_number(row.get("f_rho"), f"fidelity row {key}.f_rho")
        f_k = _finite_number(row.get("f_k"), f"fidelity row {key}.f_k")
        if not -1.0 <= f_rho <= 1.0 or not 0.0 <= f_k <= 1.0:
            raise FinalizationError(f"fidelity row {key} is outside metric bounds")
        for field in (
            "perturbations_valid",
            "exact_reference_valid",
            "common_control_support",
        ):
            if type(row.get(field)) is not bool:
                raise FinalizationError(f"fidelity row {key}.{field} must be boolean")
        rows[key] = row
    if set(rows) != set(units):
        missing = sorted(set(units) - set(rows))
        extra = sorted(set(rows) - set(units))
        raise FinalizationError(
            f"fidelity roster differs from target manifest; missing={missing}, extra={extra}"
        )

    cell_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for (_parent, request, seed), row in rows.items():
        cell_rows.setdefault((request, seed), []).append(row)
    cells: dict[tuple[str, int], tuple[float, float]] = {}
    split_half: dict[tuple[str, int], float] = {}
    protocol: dict[str, object] | None = None
    for cell, duplicates in sorted(cell_rows.items()):
        reference = duplicates[0]
        rho = float(reference["f_rho"])
        top_k = float(reference["f_k"])
        for row in duplicates[1:]:
            if not (
                math.isclose(float(row["f_rho"]), rho, rel_tol=0.0, abs_tol=1e-12)
                and math.isclose(float(row["f_k"]), top_k, rel_tol=0.0, abs_tol=1e-12)
            ):
                raise FinalizationError(
                    f"parent-replicated fidelity values disagree for {cell}"
                )
        cells[cell] = (rho, top_k)

        representative = min(
            key for key in units if (key[1], key[2]) == cell
        )
        run_manifest = _load_json(Path(str(units[representative]["run_manifest"])))
        artifact = run_manifest.get("profile_artifact")
        if not isinstance(artifact, Mapping):
            raise FinalizationError(f"unit {representative} lacks profile artifact")
        profile_path = Path(str(artifact.get("path", ""))).resolve()
        if (
            not profile_path.is_file()
            or artifact.get("sha256") != _sha256(profile_path)
        ):
            raise FinalizationError(f"unit {representative} profile hash is invalid")
        split_half[cell], cell_protocol = _split_half_rho(profile_path)
        if protocol is None:
            protocol = cell_protocol
        elif protocol != cell_protocol:
            raise FinalizationError("target fidelity cells use different probe protocols")

    requests = sorted({request for request, _seed in cells})
    seeds_by_request = {
        request: sorted(seed for candidate, seed in cells if candidate == request)
        for request in requests
    }
    observed_seed_rosters = {tuple(seeds) for seeds in seeds_by_request.values()}
    if len(observed_seed_rosters) != 1:
        raise FinalizationError("target requests have inconsistent fidelity seed rosters")

    rng = random.Random(bootstrap_seed)
    rho_draws: list[float] = []
    k_draws: list[float] = []
    for _ in range(bootstrap_replicates):
        sampled: list[tuple[float, float]] = []
        for request in (rng.choice(requests) for _ in requests):
            seeds = seeds_by_request[request]
            for _seed in seeds:
                sampled.append(cells[(request, rng.choice(seeds))])
        rho_draws.append(sum(value[0] for value in sampled) / len(sampled))
        k_draws.append(sum(value[1] for value in sampled) / len(sampled))

    tau_rho = 0.80
    tau_k = 0.70
    all_integrity_valid = all(
        row["perturbations_valid"]
        and row["exact_reference_valid"]
        and row["common_control_support"]
        for row in rows.values()
    )
    split_half_threshold = 0.70
    certificate_passed = bool(
        all_integrity_valid
        and split_half
        and min(split_half.values()) >= split_half_threshold
    )
    values = list(cells.values())
    summary = {
        "schema_version": 1,
        "setting": setting,
        "source": {
            "target_manifest": str(manifest_path.resolve()),
            "target_manifest_sha256": _sha256(manifest_path),
            "fidelity_raw": str(fidelity_raw_path.resolve()),
            "fidelity_raw_sha256": _sha256(fidelity_raw_path),
        },
        "certificate_passed": certificate_passed,
        "f_rho": sum(value[0] for value in values) / len(values),
        "f_k": sum(value[1] for value in values) / len(values),
        "f_rho_lb": _percentile(rho_draws, alpha),
        "f_k_lb": _percentile(k_draws, alpha),
        "f_rho_p_one_sided": (
            1 + sum(value <= tau_rho for value in rho_draws)
        )
        / (len(rho_draws) + 1),
        "f_k_p_one_sided": (
            1 + sum(value <= tau_k for value in k_draws)
        )
        / (len(k_draws) + 1),
        "tau_rho": tau_rho,
        "tau_k": tau_k,
        "split_half_rho": min(split_half.values()),
        "split_half_threshold": split_half_threshold,
        "integrity_valid_n": sum(
            row["perturbations_valid"]
            and row["exact_reference_valid"]
            and row["common_control_support"]
            for row in rows.values()
        ),
        "integrity_total_n": len(rows),
        "request_seed_cells": len(cells),
        "parent_replicates_per_cell": sorted(
            {len(duplicates) for duplicates in cell_rows.values()}
        ),
        "bootstrap": {
            "unit": "request_then_seed",
            "replicates": bootstrap_replicates,
            "seed": bootstrap_seed,
            "one_sided_alpha": alpha,
        },
        "protocol": protocol,
        "target_outcomes_used": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    return output_path


def _required_file(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FinalizationError(f"{name} must be a non-empty path")
    path = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if not path.is_file():
        raise FinalizationError(f"{name} is missing: {path}")
    return path


def finalization_completion(
    *,
    joint_root: Path,
    output_root: Path,
    table_out: Path,
    fidelity_input: Path,
    setting: str,
) -> tuple[bool, str]:
    status_path = output_root.resolve() / "FINALIZATION_STATUS.json"
    if not status_path.is_file():
        return False, f"missing {status_path}"
    try:
        status = _load_json(status_path)
        if (
            status.get("schema_version") != 1
            or status.get("status") != "complete"
            or status.get("setting") != setting
        ):
            return False, "finalization status contract mismatch"
        expected = {
            "joint_best": joint_root.resolve() / "BEST.json",
            "joint_status": joint_root.resolve() / "SWEEP_STATUS.json",
            "fidelity_summary": fidelity_input.resolve(),
            "table1": table_out.resolve(),
            "readiness": (
                output_root.resolve() / setting / "evidence_readiness.json"
            ),
            "ledger": output_root.resolve() / setting / "evidence_ledger.json",
            "selection_freeze": (
                output_root.resolve() / setting / "selection_freeze.yaml"
            ),
            "conclusion": output_root.resolve() / "RESULT_CONCLUSION.json",
        }
        artifacts = status.get("artifacts")
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(expected):
            return False, "finalization artifact roster mismatch"
        for name, path in expected.items():
            entry = artifacts.get(name)
            if (
                not isinstance(entry, Mapping)
                or Path(str(entry.get("path", ""))).resolve() != path
                or not path.is_file()
                or entry.get("sha256") != _sha256(path)
            ):
                return False, f"finalization artifact mismatch: {name}"
        if table_out.stat().st_size == 0:
            return False, f"Table 1 output is empty: {table_out}"
    except (FinalizationError, OSError, TypeError, ValueError) as error:
        return False, f"{type(error).__name__}: {error}"
    return True, f"validated {status_path}"


def _final_stage(
    output_root: Path,
    index: int,
    total: int,
    name: str,
) -> None:
    path = output_root / "FINAL_CURRENT_STAGE.json"
    _atomic_json(
        path,
        {
            "state": "running",
            "stage": name,
            "stage_index": index,
            "stage_total": total,
        },
    )
    print(
        f"\n========== [FINALIZE STAGE {index}/{total}] {name} ==========",
        flush=True,
    )


def resolve_joint_winner(joint_root: Path) -> dict[str, Path]:
    joint_root = joint_root.resolve()
    best = _load_json(joint_root / "BEST.json")
    status = _load_json(joint_root / "SWEEP_STATUS.json")
    if (
        status.get("status") not in {"joint_best", "best_available"}
        or status.get("terminal") is not True
        or status.get("target_used") is not False
    ):
        raise FinalizationError(
            f"joint sweep is not a target-free terminal winner: {joint_root}"
        )
    if (
        best.get("status") not in {"draft", "selected", "best_available"}
        or best.get("human_review_required") not in {True, False}
        or best.get("development_only") is not True
        or best.get("target_used") is not False
    ):
        raise FinalizationError("BEST.json does not describe a target-free joint winner")

    trial_dir = Path(str(best.get("trial_dir", ""))).resolve()
    if (
        not trial_dir.is_dir()
        or trial_dir != Path(str(status.get("trial_dir", ""))).resolve()
    ):
        raise FinalizationError("BEST.json and SWEEP_STATUS.json disagree on trial_dir")
    runtime = _required_file(best.get("recommended_runtime"), "recommended_runtime")
    comparison = _required_file(best.get("joint_comparison"), "joint_comparison")
    comparison_payload = _load_json(comparison)
    strict_joint_dominance = best.get("strict_joint_dominance", True)
    if type(strict_joint_dominance) is not bool:
        raise FinalizationError("BEST.json strict_joint_dominance must be boolean")
    comparison_passed = comparison_payload.get("passed") is True
    if comparison_passed != strict_joint_dominance:
        raise FinalizationError(
            "selected joint comparison disagrees with strict-dominance status"
        )

    metadata = _load_json(trial_dir / "trial.json")
    resolved = metadata.get("resolved_configs")
    canonical = metadata.get("canonical_sources")
    if not isinstance(resolved, Mapping) or not isinstance(canonical, Mapping):
        raise FinalizationError("winning trial metadata lacks resolved/canonical configs")
    campaign = _required_file(resolved.get("campaign"), "winning campaign")
    evidence = _required_file(canonical.get("evidence"), "canonical evidence")
    if runtime != _required_file(resolved.get("runtime"), "winning runtime"):
        raise FinalizationError("BEST.json runtime disagrees with trial metadata")
    protection_input = _required_file(
        str(trial_dir / "stage" / "selection_inputs.jsonl"),
        "winning protection selection input",
    )
    return {
        "trial_dir": trial_dir,
        "campaign": campaign,
        "evidence": evidence,
        "runtime": runtime,
        "protection_input": protection_input,
    }


def _record_joint_best_freeze(
    joint_root: Path,
    winner: Mapping[str, Path],
) -> Path:
    joint_root = joint_root.resolve()
    best_path = joint_root / "BEST.json"
    status_path = joint_root / "SWEEP_STATUS.json"
    freeze_record = joint_root / "JOINT_BEST_FREEZE.json"
    record = {
        "schema_version": 1,
        "contract": "tofu-pdf-v4-joint-best-freeze",
        "status": "frozen",
        "automatic": True,
        "development_only": True,
        "target_used": False,
        "strict_joint_dominance": _load_json(best_path).get(
            "strict_joint_dominance",
            True,
        ),
        "best": str(best_path),
        "best_sha256": _sha256(best_path),
        "sweep_status": str(status_path),
        "sweep_status_sha256": _sha256(status_path),
        "trial_dir": str(winner["trial_dir"]),
    }
    if freeze_record.is_file():
        current_record = _load_json(freeze_record)
        if current_record != record:
            raise FinalizationError(
                f"existing joint BEST freeze differs from validated winner: {freeze_record}"
            )
        print(f"[RESUME] reusing joint BEST freeze: {freeze_record}", flush=True)
        return freeze_record
    _atomic_json(freeze_record, record)
    print(f"[FREEZE] automatic joint BEST freeze: {freeze_record}", flush=True)
    return freeze_record


def _write_final_campaign(
    source: Path,
    destination: Path,
    selection_freeze: Path,
) -> None:
    campaign = _load_yaml(source)
    execution = campaign.get("execution")
    if not isinstance(execution, dict):
        raise FinalizationError("winning campaign has no execution mapping")
    execution["selection_freeze"] = str(selection_freeze.resolve())
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(campaign, sort_keys=False)
    if destination.exists():
        if _load_yaml(destination) != campaign:
            raise FinalizationError(
                f"final campaign already exists with different content: {destination}"
            )
        return
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, destination)


def _run(command: list[str]) -> None:
    print("[COMMAND] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _stage_paths(
    output_root: Path, setting: str, stage: str
) -> tuple[Path, Path, Path]:
    return (
        output_root / "manifests" / f"{setting}__{stage}.yaml",
        output_root / setting / stage / "units",
        output_root / setting / stage / "sealed",
    )


def _run_resumable_stage(
    args: argparse.Namespace,
    *,
    stage: str,
    campaign: Path,
    evidence: Path,
    runtime: Path,
) -> Path:
    manifest_path, unit_root, sealed = _stage_paths(
        args.output_root, args.setting, stage
    )
    _run(
        [
            str(args.python),
            "experiments/paper/init_v4_stage.py",
            "--stage",
            stage,
            "--setting",
            args.setting,
            "--campaign",
            str(campaign),
            "--evidence",
            str(evidence),
            "--runtime",
            str(runtime),
            "--python",
            str(args.python),
            "--unit-root",
            str(unit_root),
            "--out",
            str(manifest_path),
        ]
    )
    manifest = _load_yaml(manifest_path)
    units = manifest.get("units")
    if not isinstance(units, list):
        raise FinalizationError(f"{manifest_path} has no unit list")
    completion = [
        (
            unit,
            *_unit_completion_status(
                unit,
                campaign_hash=str(manifest["campaign_config_sha256"]),
                evidence_hash=str(manifest["evidence_config_sha256"]),
                runtime_hash=str(manifest["runtime_config_sha256"]),
                stage=stage,
            ),
        )
        for unit in units
        if isinstance(unit, Mapping)
    ]
    pending = [
        unit for unit, complete, _reason in completion if not complete
    ]
    print(
        f"[RESUME] stage={stage} valid={len(completion) - len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    for unit, complete, reason in completion:
        identity = (
            f"{unit.get('parent')}__{unit.get('request')}__seed-{unit.get('seed')}"
        )
        if complete:
            print(
                f"[UNIT REUSED] stage={stage} unit={identity} retraining=0",
                flush=True,
            )
        else:
            print(
                f"[UNIT PENDING] stage={stage} unit={identity} reason={reason}",
                flush=True,
            )
    _run_lanes(
        pending,
        gpus=args.gpus,
        trial_dir=args.output_root / args.setting / stage,
        events=_EventLog(args.output_root / "events.jsonl"),
        progress_interval=args.progress_interval,
    )
    _run(
        [
            str(args.python),
            "experiments/paper/run_v4_stage.py",
            "--campaign",
            str(campaign),
            "--evidence",
            str(evidence),
            "--manifest",
            str(manifest_path),
            "--output-dir",
            str(sealed),
            "--action",
            "verify",
        ]
    )
    return sealed


def _validate_existing_freeze(
    path: Path, prediction_input: Path, protection_input: Path
) -> bool:
    if not path.is_file():
        return False
    freeze = _load_yaml(path)
    if freeze.get("status") != "frozen" or freeze.get("frozen_before_target") is not True:
        raise FinalizationError(f"existing selection freeze is not frozen: {path}")
    expected = {
        str(candidate.resolve()): _sha256(candidate)
        for candidate in (prediction_input, protection_input)
    }
    artifacts = freeze.get("development_artifacts")
    observed = (
        {
            str(Path(str(item.get("path", ""))).resolve()): item.get("sha256")
            for item in artifacts
            if isinstance(item, Mapping)
        }
        if isinstance(artifacts, list)
        else {}
    )
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != len(expected)
        or observed != expected
    ):
        raise FinalizationError(
            "existing selection freeze does not match current prediction/protection inputs"
        )
    print(f"[RESUME] reusing selection freeze: {path}", flush=True)
    return True


def _publish_combined(args: argparse.Namespace) -> None:
    ledger = args.output_root / args.setting / "evidence_ledger.json"
    if not ledger.is_file():
        raise FinalizationError(f"cannot publish missing evidence ledger: {ledger}")
    _run(
        [
            str(args.python),
            "experiments/paper/publish_evidence.py",
            "--ledger",
            str(ledger),
            "--combined-root",
            str(args.combined_root),
            "--fidelity-input",
            str(args.fidelity_input),
        ]
    )


def run(args: argparse.Namespace) -> None:
    complete, reason = finalization_completion(
        joint_root=args.joint_root,
        output_root=args.output_root,
        table_out=args.table_out,
        fidelity_input=args.fidelity_input,
        setting=args.setting,
    )
    if complete:
        print(
            f"[FINALIZATION SKIPPED] already_complete=true rerun=0 reason={reason}",
            flush=True,
        )
        _publish_combined(args)
        return
    print(
        f"[FINALIZATION RESUME] already_complete=false reason={reason}",
        flush=True,
    )
    _final_stage(args.output_root, 1, 7, "validate-joint-winner")
    winner = resolve_joint_winner(args.joint_root)
    selected = _load_json(args.joint_root / "BEST.json")
    strict_joint_dominance = bool(
        selected.get("strict_joint_dominance", True)
    )
    print(
        "[SELECTION] "
        f"status={selected.get('status')} "
        f"trial={selected.get('trial_id')} "
        f"strict_joint_dominance={str(strict_joint_dominance).lower()}",
        flush=True,
    )
    _record_joint_best_freeze(args.joint_root, winner)

    args.output_root.mkdir(parents=True, exist_ok=True)
    selection_freeze = (
        args.output_root / args.setting / "selection_freeze.yaml"
    ).resolve()
    final_campaign = (args.output_root / "config" / "campaign.final.yaml").resolve()
    _write_final_campaign(winner["campaign"], final_campaign, selection_freeze)

    _final_stage(args.output_root, 2, 7, "prediction")
    prediction_sealed = _run_resumable_stage(
        args,
        stage="prediction",
        campaign=final_campaign,
        evidence=winner["evidence"],
        runtime=winner["runtime"],
    )
    prediction_input = prediction_sealed / "selection_inputs.jsonl"
    _final_stage(args.output_root, 3, 7, "selection-freeze")
    if not _validate_existing_freeze(
        selection_freeze, prediction_input, winner["protection_input"]
    ):
        _run(
            [
                str(args.python),
                "experiments/paper/select_tofu_v4.py",
                "--kind",
                "claims",
                "--input",
                str(prediction_input),
                "--input",
                str(winner["protection_input"]),
                "--setting",
                args.setting,
                "--campaign",
                str(final_campaign),
                "--evidence",
                str(winner["evidence"]),
                "--runtime",
                str(winner["runtime"]),
                "--freeze",
                "--out",
                str(selection_freeze),
            ]
        )

    _final_stage(args.output_root, 4, 7, "target-evaluation")
    target_sealed = _run_resumable_stage(
        args,
        stage="target_evaluation",
        campaign=final_campaign,
        evidence=winner["evidence"],
        runtime=winner["runtime"],
    )
    setting_root = args.output_root / args.setting
    raw_plan = setting_root / "raw_plan.json"
    ledger = setting_root / "evidence_ledger.json"
    readiness = setting_root / "evidence_readiness.json"
    campaign_contract = _load_yaml(final_campaign)
    evidence_contract = _load_yaml(winner["evidence"])
    bootstrap = campaign_contract.get("execution", {}).get("bootstrap")
    decision = evidence_contract.get("decision")
    if not isinstance(bootstrap, Mapping) or not isinstance(decision, Mapping):
        raise FinalizationError("campaign/evidence lacks bootstrap decision contract")
    fidelity_summary = args.fidelity_input
    fidelity_payload = _load_json(fidelity_summary)
    if fidelity_payload.get("setting") != args.setting:
        raise FinalizationError(
            "declared fidelity summary setting does not match finalization setting"
        )
    if fidelity_payload.get("support") != "declared_setting_fidelity":
        raise FinalizationError(
            "RQ2 requires a declared setting-level fidelity support summary; "
            "target-support diagnostics are not accepted"
        )
    certificate_path = Path(
        str(fidelity_payload.get("source_certificate", ""))
    ).resolve()
    if (
        not certificate_path.is_file()
        or fidelity_payload.get("source_certificate_sha256")
        != _sha256(certificate_path)
    ):
        raise FinalizationError(
            "declared fidelity source certificate path/hash validation failed"
        )
    if type(fidelity_payload.get("certificate_passed")) is not bool:
        raise FinalizationError(
            "declared fidelity summary lacks a boolean measured outcome"
        )
    print(
        "[RESULT] declared fidelity passed="
        f"{str(fidelity_payload['certificate_passed']).lower()}",
        flush=True,
    )
    print(f"[DONE] declared fidelity summary: {fidelity_summary}", flush=True)
    _final_stage(args.output_root, 5, 7, "raw-evidence")
    _run(
        [
            str(args.python),
            "experiments/paper/init_raw_plan.py",
            "--evidence",
            str(winner["evidence"]),
            "--campaign",
            str(final_campaign),
            "--selection-freeze",
            str(selection_freeze),
            "--setting",
            args.setting,
            "--out",
            str(raw_plan),
        ]
    )
    _final_stage(args.output_root, 6, 7, "aggregate-evidence")
    _run(
        [
            str(args.python),
            "experiments/paper/aggregate_raw.py",
            "--plan",
            str(raw_plan),
            "--prediction-raw",
            str(target_sealed / "prediction_raw.jsonl"),
            "--fidelity-raw",
            str(target_sealed / "fidelity_raw.jsonl"),
            "--protection-raw",
            str(target_sealed / "protection_raw.jsonl"),
            "--core-only",
            "--out",
            str(ledger),
        ]
    )
    _final_stage(args.output_root, 7, 7, "table1-latex")
    _run(
        [
            str(args.python),
            "experiments/paper/build_evidence.py",
            "--config",
            str(winner["evidence"]),
            "--ledger",
            str(ledger),
            "--readiness-out",
            str(readiness),
            "--fidelity-input",
            f"{args.setting}={fidelity_summary}",
            "--table1-setting",
            args.setting,
            "--table1-out",
            str(args.table_out),
        ]
    )
    print(f"[DONE] Table 1 LaTeX: {args.table_out}", flush=True)
    print(f"[DONE] evidence readiness: {readiness}", flush=True)
    _atomic_json(
        args.output_root / "RESULT_CONCLUSION.json",
        {
            "schema_version": 1,
            "setting": args.setting,
            "selection_status": selected.get("status"),
            "selected_trial": selected.get("trial_id"),
            "strict_joint_dominance": strict_joint_dominance,
            "result_interpretation": (
                "strict_joint_winner"
                if strict_joint_dominance
                else "best_observed_with_strict_joint_criterion_failed"
            ),
            "table1": str(args.table_out.resolve()),
            "readiness": str(readiness.resolve()),
        },
    )
    artifact_paths = {
        "joint_best": args.joint_root / "BEST.json",
        "joint_status": args.joint_root / "SWEEP_STATUS.json",
        "fidelity_summary": fidelity_summary,
        "table1": args.table_out,
        "readiness": readiness,
        "ledger": ledger,
        "selection_freeze": selection_freeze,
        "conclusion": args.output_root / "RESULT_CONCLUSION.json",
    }
    _atomic_json(
        args.output_root / "FINALIZATION_STATUS.json",
        {
            "schema_version": 1,
            "status": "complete",
            "setting": args.setting,
            "artifacts": {
                name: {
                    "path": str(path.resolve()),
                    "sha256": _sha256(path.resolve()),
                }
                for name, path in artifact_paths.items()
            },
        },
    )
    _atomic_json(
        args.output_root / "FINAL_CURRENT_STAGE.json",
        {
            "state": "complete",
            "stage": "table1-latex",
            "stage_index": 7,
            "stage_total": 7,
        },
    )
    _publish_combined(args)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", default="tofu_qwen25_1p5b")
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument(
        "--joint-root",
        type=Path,
        default=Path(
            "/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/joint_sweep"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/rdata/minsoo3.kim/results/paper/tofu_qwen25_1p5b/final"),
    )
    parser.add_argument("--table-out", type=Path, default=None)
    parser.add_argument("--combined-root", type=Path, default=None)
    parser.add_argument("--fidelity-input", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--progress-interval", type=float, default=15.0)
    parser.add_argument(
        "--approve-joint-best",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--check-complete",
        action="store_true",
        help="validate completed Table 1 outputs without running any stage",
    )
    args = parser.parse_args(argv)
    try:
        args.gpus = tuple(int(value.strip()) for value in args.gpus.split(","))
    except ValueError as error:
        parser.error(f"--gpus must be comma-separated integers: {error}")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain unique GPU ids")
    args.python = _absolute_executable(args.python)
    args.joint_root = args.joint_root.resolve()
    args.output_root = args.output_root.resolve()
    args.fidelity_input = args.fidelity_input.resolve()
    if not args.fidelity_input.is_file():
        parser.error(f"--fidelity-input does not exist: {args.fidelity_input}")
    args.table_out = (
        args.table_out.resolve()
        if args.table_out is not None
        else args.output_root / "table1.tex"
    )
    args.combined_root = (
        args.combined_root.resolve()
        if args.combined_root is not None
        else args.output_root.parent.parent / "paper_v4"
    )
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.check_complete:
            complete, reason = finalization_completion(
                joint_root=args.joint_root,
                output_root=args.output_root,
                table_out=args.table_out,
                fidelity_input=args.fidelity_input,
                setting=args.setting,
            )
            print(
                f"[FINALIZATION STATUS] complete={str(complete).lower()} "
                f"reason={reason}",
                flush=True,
            )
            if complete:
                _publish_combined(args)
            return 0 if complete else 1
        run(args)
        return 0
    except (
        FinalizationError,
        SweepError,
        KeyError,
        OSError,
        subprocess.SubprocessError,
        TypeError,
    ) as error:
        print(f"joint sweep finalization failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
