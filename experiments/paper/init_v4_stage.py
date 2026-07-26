#!/usr/bin/env python3
"""Generate the exact command manifest for one PDF-v4 paper stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
STAGE_OUTPUTS = {
    "calibration": ("fidelity_raw", "parent_selection_inputs"),
    "prediction": ("prediction_raw", "selection_inputs"),
    "protection": ("protection_raw", "selection_inputs"),
    "target_evaluation": ("prediction_raw", "fidelity_raw", "protection_raw"),
}


class ManifestInitError(ValueError):
    """The campaign cannot be expanded into an exact stage manifest."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ManifestInitError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise ManifestInitError(f"{path} must contain one mapping")
    return value


def _resolve(value: str | Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    return path if path.is_absolute() else (ROOT / path).resolve()


def absolute_executable(value: str | Path, *, base: Path = ROOT) -> Path:
    """Return an absolute command path without dereferencing a venv symlink."""
    expanded = os.path.expanduser(os.path.expandvars(str(value)))
    path = Path(expanded)
    if path.is_absolute():
        return Path(os.path.abspath(path))
    if len(path.parts) == 1:
        located = shutil.which(expanded)
        if located:
            return Path(os.path.abspath(located))
    return Path(os.path.abspath(base / path))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    campaign: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    stage: str,
    setting_id: str,
    campaign_path: Path,
    evidence_path: Path,
    runtime_path: Path,
    unit_root: Path,
    python: str,
) -> dict[str, Any]:
    if stage not in STAGE_OUTPUTS:
        raise ManifestInitError(f"unsupported stage {stage!r}")
    settings = {
        item.get("id"): item
        for item in evidence.get("settings", [])
        if isinstance(item, Mapping)
    }
    setting = settings.get(setting_id)
    if not isinstance(setting, Mapping):
        raise ManifestInitError(f"unknown evidence setting {setting_id!r}")
    if setting.get("dataset") != "TOFU":
        raise ManifestInitError("the TOFU unit producer only accepts TOFU settings")
    stage_config = campaign.get("stages", {}).get(stage)
    dataset = campaign.get("datasets", {}).get("TOFU")
    execution = campaign.get("execution")
    if not all(
        isinstance(value, Mapping)
        for value in (stage_config, dataset, execution)
    ):
        raise ManifestInitError("campaign lacks stage, TOFU, or execution config")
    roster_name = stage_config.get("roster")
    requests = dataset.get("rosters", {}).get(roster_name)
    seeds = execution.get("seeds")
    parents = setting.get("parents")
    if not all(isinstance(value, list) and value for value in (requests, seeds, parents)):
        raise ManifestInitError("stage roster, seeds, and parents must be non-empty lists")

    producer = _resolve(dataset.get("unit_producer", ""))
    unit_python = str(absolute_executable(python))
    if not producer.is_file():
        raise ManifestInitError(f"TOFU unit producer is missing: {producer}")
    selection_freeze = _resolve(execution.get("selection_freeze", ""))
    units = []
    for parent in parents:
        for request in requests:
            for seed in seeds:
                identity = f"{parent}__{request}__seed-{seed}"
                output_dir = (unit_root / identity).resolve()
                command = [
                    unit_python,
                    "-u",
                    str(producer),
                    "--campaign",
                    str(campaign_path),
                    "--evidence",
                    str(evidence_path),
                    "--runtime",
                    str(runtime_path),
                    "--stage",
                    stage,
                    "--setting",
                    setting_id,
                    "--parent",
                    str(parent),
                    "--request",
                    str(request),
                    "--seed",
                    str(seed),
                    "--output-dir",
                    str(output_dir),
                ]
                if stage == "target_evaluation":
                    command.extend(
                        ("--selection-freeze", str(selection_freeze))
                    )
                units.append(
                    {
                        "parent": str(parent),
                        "request": str(request),
                        "seed": str(seed),
                        "command": command,
                        "run_manifest": str(output_dir / "run_manifest.json"),
                        "outputs": {
                            kind: str(output_dir / f"{kind}.jsonl")
                            for kind in STAGE_OUTPUTS[stage]
                        },
                    }
                )
    return {
        "schema_version": 1,
        "contract": "kdd-unlearning-fail-pdf-v4-stage",
        "status": "frozen",
        "campaign_id": campaign.get("campaign_id"),
        "stage": stage,
        "setting": setting_id,
        "unit_producer": str(producer),
        "runtime_config": str(runtime_path),
        "campaign_config_sha256": _sha256(campaign_path),
        "evidence_config_sha256": _sha256(evidence_path),
        "runtime_config_sha256": _sha256(runtime_path),
        "units": units,
    }


def _atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_OUTPUTS), required=True)
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
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--unit-root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        campaign_path = args.campaign.resolve()
        evidence_path = args.evidence.resolve()
        runtime_path = args.runtime.resolve()
        unit_root = (
            args.unit_root.resolve()
            if args.unit_root is not None
            else ROOT
            / "runs/paper/tofu_v4"
            / args.setting
            / args.stage
            / "units"
        )
        destination = (
            args.out.resolve()
            if args.out is not None
            else ROOT
            / "results/paper/manifests"
            / f"{args.setting}__{args.stage}.yaml"
        )
        manifest = build_manifest(
            _load(campaign_path),
            _load(evidence_path),
            stage=args.stage,
            setting_id=args.setting,
            campaign_path=campaign_path,
            evidence_path=evidence_path,
            runtime_path=runtime_path,
            unit_root=unit_root,
            python=args.python,
        )
        _atomic_write(destination, manifest)
        print(f"wrote stage manifest: {destination}")
        print(f"planned units: {len(manifest['units'])}")
        print(json.dumps({"stage": args.stage, "setting": args.setting}))
        return 0
    except ManifestInitError as error:
        print(f"stage manifest initialization failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
