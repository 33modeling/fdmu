#!/usr/bin/env python3
"""Continue a frozen joint development winner through Table 1 LaTeX."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paper.run_joint_dev_sweep import (  # noqa: E402
    SweepError,
    _EventLog,
    _run_lanes,
    _unit_complete,
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


def _required_file(value: object, name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise FinalizationError(f"{name} must be a non-empty path")
    path = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if not path.is_file():
        raise FinalizationError(f"{name} is missing: {path}")
    return path


def resolve_joint_winner(joint_root: Path) -> dict[str, Path]:
    joint_root = joint_root.resolve()
    best = _load_json(joint_root / "BEST.json")
    status = _load_json(joint_root / "SWEEP_STATUS.json")
    if (
        status.get("status") != "joint_best"
        or status.get("terminal") is not True
        or status.get("target_used") is not False
    ):
        raise FinalizationError(
            f"joint sweep is not a target-free terminal winner: {joint_root}"
        )
    if (
        best.get("status") != "draft"
        or best.get("human_review_required") is not True
        or best.get("target_used") is not False
    ):
        raise FinalizationError("BEST.json does not describe a reviewable joint winner")

    trial_dir = Path(str(best.get("trial_dir", ""))).resolve()
    if (
        not trial_dir.is_dir()
        or trial_dir != Path(str(status.get("trial_dir", ""))).resolve()
    ):
        raise FinalizationError("BEST.json and SWEEP_STATUS.json disagree on trial_dir")
    runtime = _required_file(best.get("recommended_runtime"), "recommended_runtime")
    comparison = _required_file(best.get("joint_comparison"), "joint_comparison")
    comparison_payload = _load_json(comparison)
    if comparison_payload.get("passed") is not True:
        raise FinalizationError("winning joint comparison is not passing")

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
    pending = [
        unit
        for unit in units
        if isinstance(unit, Mapping)
        and not _unit_complete(
            unit,
            campaign_hash=str(manifest["campaign_config_sha256"]),
            evidence_hash=str(manifest["evidence_config_sha256"]),
            runtime_hash=str(manifest["runtime_config_sha256"]),
            stage=stage,
        )
    ]
    if len(pending) != sum(isinstance(unit, Mapping) for unit in units):
        print(
            f"[RESUME] stage={stage} valid={len(units) - len(pending)} "
            f"pending={len(pending)}",
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


def run(args: argparse.Namespace) -> None:
    winner = resolve_joint_winner(args.joint_root)
    if not args.approve_joint_best:
        raise FinalizationError(
            "target evaluation requires explicit review approval: "
            "pass --approve-joint-best"
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    selection_freeze = (
        args.output_root / args.setting / "selection_freeze.yaml"
    ).resolve()
    final_campaign = (args.output_root / "config" / "campaign.final.yaml").resolve()
    _write_final_campaign(winner["campaign"], final_campaign, selection_freeze)

    prediction_sealed = _run_resumable_stage(
        args,
        stage="prediction",
        campaign=final_campaign,
        evidence=winner["evidence"],
        runtime=winner["runtime"],
    )
    prediction_input = prediction_sealed / "selection_inputs.jsonl"
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
            "--table1-setting",
            args.setting,
            "--table1-out",
            str(args.table_out),
        ]
    )
    print(f"[DONE] Table 1 LaTeX: {args.table_out}", flush=True)
    print(f"[DONE] evidence readiness: {readiness}", flush=True)


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
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--progress-interval", type=float, default=15.0)
    parser.add_argument("--approve-joint-best", action="store_true")
    args = parser.parse_args(argv)
    try:
        args.gpus = tuple(int(value.strip()) for value in args.gpus.split(","))
    except ValueError as error:
        parser.error(f"--gpus must be comma-separated integers: {error}")
    if not args.gpus or len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must contain unique GPU ids")
    args.python = args.python.resolve()
    args.joint_root = args.joint_root.resolve()
    args.output_root = args.output_root.resolve()
    args.table_out = (
        args.table_out.resolve()
        if args.table_out is not None
        else args.output_root / "table1.tex"
    )
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
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
