#!/usr/bin/env python3
"""Run the complete TOFU Table 1 workflow in its required freeze order."""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]


def _run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one mapping")
    return value


def _stage_paths(root: Path, setting: str, stage: str) -> tuple[Path, Path, Path]:
    return (
        root / "manifests" / f"{setting}__{stage}.yaml",
        root / setting / stage / "units",
        root / setting / stage / "sealed",
    )


def _initialize_stage(
    args: argparse.Namespace,
    root: Path,
    stage: str,
) -> tuple[Path, Path, Path]:
    manifest, unit_root, sealed = _stage_paths(root, args.setting, stage)
    _run(
        [
            args.python,
            "experiments/paper/init_v4_stage.py",
            "--stage",
            stage,
            "--setting",
            args.setting,
            "--campaign",
            str(args.campaign),
            "--evidence",
            str(args.evidence),
            "--runtime",
            str(args.runtime),
            "--python",
            args.python,
            "--unit-root",
            str(unit_root),
            "--out",
            str(manifest),
        ]
    )
    return manifest, unit_root, sealed


def _execute_stage(
    args: argparse.Namespace,
    manifest: Path,
    sealed: Path,
    *,
    action: str,
) -> None:
    _run(
        [
            args.python,
            "experiments/paper/run_v4_stage.py",
            "--campaign",
            str(args.campaign),
            "--evidence",
            str(args.evidence),
            "--manifest",
            str(manifest),
            "--output-dir",
            str(sealed),
            "--action",
            action,
        ]
    )


def run_workflow(args: argparse.Namespace) -> None:
    root = args.output_root.resolve()
    campaign = _load(args.campaign)
    runtime = _load(args.runtime)
    setting_runtime = runtime.get("settings", {}).get(args.setting)
    if not isinstance(setting_runtime, dict):
        raise ValueError(f"runtime has no setting {args.setting!r}")
    parent_freeze = Path(setting_runtime["parent_freeze"])
    if not parent_freeze.is_absolute():
        parent_freeze = ROOT / parent_freeze
    selection_freeze = Path(campaign["execution"]["selection_freeze"])
    if not selection_freeze.is_absolute():
        selection_freeze = ROOT / selection_freeze

    stages = {
        stage: _initialize_stage(args, root, stage)
        for stage in ("calibration", "prediction", "protection", "target_evaluation")
    }
    if args.action == "plan":
        print(f"wrote {len(stages)} exact stage manifests below {root / 'manifests'}")
        return

    calibration_manifest, _calibration_units, calibration_sealed = stages["calibration"]
    _execute_stage(
        args, calibration_manifest, calibration_sealed, action="run"
    )
    _run(
        [
            args.python,
            "experiments/paper/select_tofu_v4.py",
            "--kind",
            "parent",
            "--input",
            str(calibration_sealed / "parent_selection_inputs.jsonl"),
            "--setting",
            args.setting,
            "--campaign",
            str(args.campaign),
            "--evidence",
            str(args.evidence),
            "--runtime",
            str(args.runtime),
            "--freeze",
            "--out",
            str(parent_freeze),
        ]
    )

    for stage in ("prediction", "protection"):
        manifest, _units, sealed = stages[stage]
        _execute_stage(args, manifest, sealed, action="run")
    prediction_sealed = stages["prediction"][2]
    protection_sealed = stages["protection"][2]
    _run(
        [
            args.python,
            "experiments/paper/select_tofu_v4.py",
            "--kind",
            "claims",
            "--input",
            str(prediction_sealed / "selection_inputs.jsonl"),
            "--input",
            str(protection_sealed / "selection_inputs.jsonl"),
            "--setting",
            args.setting,
            "--campaign",
            str(args.campaign),
            "--evidence",
            str(args.evidence),
            "--runtime",
            str(args.runtime),
            "--freeze",
            "--out",
            str(selection_freeze),
        ]
    )

    target_manifest, _target_units, target_sealed = stages["target_evaluation"]
    _execute_stage(args, target_manifest, target_sealed, action="run")
    raw_plan = root / args.setting / "raw_plan.json"
    ledger = root / args.setting / "evidence_ledger.json"
    readiness = root / args.setting / "evidence_readiness.json"
    _run(
        [
            args.python,
            "experiments/paper/init_raw_plan.py",
            "--evidence",
            str(args.evidence),
            "--campaign",
            str(args.campaign),
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
            args.python,
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
            args.python,
            "experiments/paper/build_evidence.py",
            "--config",
            str(args.evidence),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("plan", "run"), default="plan")
    parser.add_argument("--setting", default="tofu_qwen25_1p5b")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "configs/paper/campaign.yaml",
    )
    parser.add_argument(
        "--evidence",
        type=Path,
        default=ROOT / "configs/paper/evidence.yaml",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "configs/paper/tofu_v4.yaml",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "runs/paper/tofu_table1",
    )
    parser.add_argument(
        "--table-out",
        type=Path,
        default=ROOT / "paper/sections/generated/table1.tex",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        args.campaign = args.campaign.resolve()
        args.evidence = args.evidence.resolve()
        args.runtime = args.runtime.resolve()
        args.table_out = args.table_out.resolve()
        run_workflow(args)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"TOFU Table 1 workflow failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
