#!/usr/bin/env python3
"""Run resumable 1.5B parent calibration and emit a reviewable freeze draft."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paper.init_v4_stage import build_manifest  # noqa: E402
from experiments.paper.run_joint_dev_sweep import (  # noqa: E402
    _EventLog,
    _absolute_executable,
    _atomic_json,
    _atomic_text,
    _environment_snapshot,
    _load_json,
    _load_yaml,
    _recover_git_snapshot,
    _run_lanes,
    _seal_trial,
    _sha256,
    _snapshot_current_config,
    _status,
    _unit_completion_status,
    _write_once,
    _write_or_rebind_manifest,
    SweepError,
)
from experiments.paper.select_tofu_v4 import (  # noqa: E402
    _records,
    select_parents,
)


SETTING = "tofu_qwen25_1p5b"


def calibration_completion(
    output_root: Path,
    setting: str,
) -> tuple[bool, str]:
    """Validate the terminal calibration marker without loading a model."""
    output_root = output_root.resolve()
    status_path = output_root / "CALIBRATION_STATUS.json"
    if not status_path.is_file():
        return False, f"missing status marker {status_path}"
    try:
        status = _load_json(status_path)
        if status.get("schema_version") != 1:
            return False, "CALIBRATION_STATUS.json schema_version is not 1"
        if status.get("setting") != setting:
            return False, (
                "CALIBRATION_STATUS.json setting mismatch: "
                f"expected={setting} actual={status.get('setting')}"
            )
        if status.get("approval_ready") is not True:
            return False, "calibration selection is not approval-ready"
        if status.get("unresolved") != []:
            return False, (
                "calibration selection remains unresolved: "
                f"{status.get('unresolved')}"
            )

        expected_proposal = (
            output_root
            / "freeze_proposals"
            / "tofu_parent_freeze_1p5b.recommended.yaml"
        ).resolve()
        expected_selection = (
            output_root / "stage" / "parent_selection_inputs.jsonl"
        ).resolve()
        proposal = Path(str(status.get("proposal", ""))).resolve()
        selection = Path(str(status.get("selection_input", ""))).resolve()
        if proposal != expected_proposal or not proposal.is_file():
            return False, f"missing or mismatched proposal {proposal}"
        if selection != expected_selection or not selection.is_file():
            return False, f"missing or mismatched selection input {selection}"
        selection_digest = str(status.get("selection_input_sha256", ""))
        if len(selection_digest) != 64 or _sha256(selection) != selection_digest:
            return False, "selection input SHA-256 mismatch"

        proposal_digest = status.get("proposal_sha256")
        if proposal_digest is not None and (
            not isinstance(proposal_digest, str)
            or len(proposal_digest) != 64
            or _sha256(proposal) != proposal_digest
        ):
            return False, "proposal SHA-256 mismatch"
        proposal_payload = _load_yaml(proposal)
        if (
            proposal_payload.get("schema_version") != 1
            or proposal_payload.get("contract") != "tofu-pdf-v4-parent-freeze"
            or proposal_payload.get("status") != "draft"
            or proposal_payload.get("unresolved") != []
        ):
            return False, "proposal is not a resolved parent-freeze draft"
        expected_source = {
            "path": str(selection),
            "sha256": selection_digest,
        }
        if proposal_payload.get("development_artifacts") != [expected_source]:
            return False, "proposal does not reference the sealed selection input"
    except (OSError, SweepError, TypeError, ValueError) as error:
        return False, f"{type(error).__name__}: {error}"
    return True, f"validated terminal marker {status_path}"


def _setting(
    evidence: Mapping[str, Any],
    runtime: Mapping[str, Any],
    setting: str,
) -> tuple[str, Mapping[str, Any]]:
    settings = {
        item.get("id"): item
        for item in evidence.get("settings", [])
        if isinstance(item, Mapping)
    }
    evidence_setting = settings.get(setting)
    runtime_setting = runtime.get("settings", {}).get(setting)
    if not isinstance(evidence_setting, Mapping) or not isinstance(
        runtime_setting, Mapping
    ):
        raise SweepError(f"setting {setting!r} is missing from evidence or runtime")
    model = evidence_setting.get("model")
    if not isinstance(model, str):
        raise SweepError(f"setting {setting!r} has no model")
    return model, runtime_setting


def _overlay(
    campaign: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    model: str,
    model_source: Path,
    sft_cache_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    campaign_local = json.loads(json.dumps(campaign))
    runtime_local = json.loads(json.dumps(runtime))
    model_config = campaign_local.get("models", {}).get(model)
    if not isinstance(model_config, dict):
        raise SweepError(f"campaign has no model {model!r}")
    model_config["source"] = str(model_source)
    model_config["source_kind"] = "local_path"
    runtime_local["runtime"]["sft_cache_root"] = str(sft_cache_root)
    return campaign_local, runtime_local


def _write_proposal(
    *,
    campaign: Mapping[str, Any],
    runtime: Mapping[str, Any],
    input_path: Path,
    setting: str,
    destination: Path,
) -> dict[str, Any]:
    rows, sources = _records([input_path])
    proposal = select_parents(
        rows,
        campaign=campaign,
        runtime=runtime,
        setting=setting,
        sources=sources,
        frozen=False,
    )
    rendered = yaml.safe_dump(proposal, sort_keys=False)
    if destination.is_file():
        existing = _load_yaml(destination)
        if existing == proposal:
            return proposal
        if existing.get("unresolved") and not proposal.get("unresolved"):
            _atomic_text(destination, rendered)
            _status(
                "CALIBRATION_PROPOSAL_UPGRADED "
                f"path={destination} fallback_parents="
                f"{proposal.get('fallback_parents', [])}"
            )
            return proposal
    _write_once(destination, rendered)
    return proposal


def run(args: argparse.Namespace) -> int:
    complete, completion_reason = calibration_completion(
        args.output_root,
        args.setting,
    )
    if complete:
        _status(
            "CALIBRATION_SKIPPED already_complete=true retraining=0 "
            f"reason={completion_reason}"
        )
        return 0
    _status(
        "CALIBRATION_RESUME already_complete=false "
        f"reason={completion_reason}"
    )
    if args.progress_interval < 1.0:
        raise SweepError("--progress-interval must be at least 1 second")
    try:
        gpus = [int(value) for value in args.gpus.split(",")]
    except ValueError as error:
        raise SweepError("--gpus must be a comma-separated integer list") from error
    if (
        not gpus
        or any(gpu < 0 for gpu in gpus)
        or len(set(gpus)) != len(gpus)
    ):
        raise SweepError("--gpus must contain unique non-negative integers")

    campaign_source = args.campaign.resolve()
    evidence_source = args.evidence.resolve()
    runtime_source = args.runtime.resolve()
    python = _absolute_executable(args.python)
    model_source = args.model_source.resolve()
    output_root = args.output_root.resolve()
    sft_cache_root = args.sft_cache_root.resolve()
    _status(
        f"CALIBRATION_CONFIG setting={args.setting} gpus={gpus} "
        f"output={output_root}"
    )
    for name, path in (
        ("python", python),
        ("model", model_source),
    ):
        if not path.exists():
            raise SweepError(f"{name} is missing: {path}")

    campaign = _load_yaml(campaign_source)
    evidence = _load_yaml(evidence_source)
    runtime = _load_yaml(runtime_source)
    model, _runtime_setting = _setting(evidence, runtime, args.setting)
    campaign_local, runtime_local = _overlay(
        campaign,
        runtime,
        model=model,
        model_source=model_source,
        sft_cache_root=sft_cache_root,
    )

    config_dir = output_root / "config"
    campaign_path = config_dir / "campaign.local.yaml"
    runtime_path = config_dir / "tofu_v4.local.yaml"
    evidence_snapshot = config_dir / "evidence.frozen.yaml"
    _write_once(campaign_path, yaml.safe_dump(campaign_local, sort_keys=False))
    _write_once(runtime_path, yaml.safe_dump(runtime_local, sort_keys=False))
    manifest_path = output_root / "manifest.yaml"
    if manifest_path.is_file():
        existing_manifest = _load_yaml(manifest_path)
        expected_evidence_hash = str(
            existing_manifest.get("evidence_config_sha256", "")
        )
        if len(expected_evidence_hash) != 64:
            raise SweepError("existing calibration manifest lacks evidence SHA-256")
        evidence_path = _recover_git_snapshot(
            evidence_source,
            expected_evidence_hash,
            evidence_snapshot,
        )
    else:
        evidence_path = _snapshot_current_config(evidence_source, evidence_snapshot)
    manifest = build_manifest(
        campaign_local,
        _load_yaml(evidence_path),
        stage="calibration",
        setting_id=args.setting,
        campaign_path=campaign_path,
        evidence_path=evidence_path,
        runtime_path=runtime_path,
        unit_root=output_root / "units",
        python=str(python),
    )
    for unit in manifest["units"]:
        unit["setting"] = args.setting
    manifest = _write_or_rebind_manifest(manifest_path, manifest)

    _environment_snapshot(output_root, python, gpus)
    events = _EventLog(output_root / "events.jsonl")
    events.write(
        {
            "event": "parent_calibration_invoked",
            "setting": args.setting,
            "gpus": gpus,
            "planned_units": len(manifest["units"]),
        }
    )
    campaign_hash = _sha256(campaign_path)
    evidence_hash = _sha256(evidence_path)
    runtime_hash = _sha256(runtime_path)
    completion = [
        (
            unit,
            *_unit_completion_status(
                unit,
                campaign_hash=campaign_hash,
                evidence_hash=evidence_hash,
                runtime_hash=runtime_hash,
                stage="calibration",
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
            f"{unit.get('parent')}__{unit.get('request')}__seed-{unit.get('seed')}"
        )
        if complete:
            _status(f"CALIBRATION_REUSE unit={identity} retraining=0")
        else:
            _status(f"CALIBRATION_PENDING unit={identity} reason={reason}")
    _status(
        f"CALIBRATION_UNITS planned={len(manifest['units'])} "
        f"valid={len(manifest['units']) - len(pending)} pending={len(pending)}"
    )
    if args.dry_run:
        _status(f"CALIBRATION_DRY_RUN manifest={manifest_path}")
        return 0

    _run_lanes(
        pending,
        gpus=gpus,
        trial_dir=output_root,
        events=events,
        progress_interval=args.progress_interval,
    )
    _seal_trial(
        python=python,
        campaign_path=campaign_path,
        evidence_path=evidence_path,
        manifest_path=manifest_path,
        trial_dir=output_root,
    )
    selection_input = output_root / "stage" / "parent_selection_inputs.jsonl"
    proposal_path = (
        output_root
        / "freeze_proposals"
        / "tofu_parent_freeze_1p5b.recommended.yaml"
    )
    proposal = _write_proposal(
        campaign=campaign_local,
        runtime=runtime_local,
        input_path=selection_input,
        setting=args.setting,
        destination=proposal_path,
    )
    unresolved = [str(value) for value in proposal.get("unresolved", [])]
    status = {
        "schema_version": 1,
        "setting": args.setting,
        "proposal": str(proposal_path),
        "proposal_sha256": _sha256(proposal_path),
        "selection_input": str(selection_input),
        "selection_input_sha256": _sha256(selection_input),
        "unresolved": unresolved,
        "approval_ready": not unresolved,
    }
    _atomic_json(output_root / "CALIBRATION_STATUS.json", status)
    events.write({"event": "parent_calibration_selected", **status})
    if unresolved:
        _status(
            "PARENT_CALIBRATION_UNRESOLVED "
            f"parents={unresolved} proposal={proposal_path}"
        )
        return 3
    _status(f"AUTOMATIC_FREEZE_READY proposal={proposal_path}")
    _status("NEXT_AUTOMATIC_STAGE parent-freeze (no operator input required)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--setting", default=SETTING)
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
    parser.add_argument("--python", type=Path, default=ROOT / ".venv/bin/python")
    parser.add_argument("--model-source", type=Path, required=True)
    parser.add_argument("--sft-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--progress-interval", type=float, default=15.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--check-complete",
        action="store_true",
        help="validate the terminal marker without starting calibration",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        if args.check_complete:
            complete, reason = calibration_completion(
                args.output_root,
                args.setting,
            )
            _status(
                f"CALIBRATION_STATUS complete={str(complete).lower()} "
                f"reason={reason}"
            )
            return 0 if complete else 1
        return run(args)
    except (SweepError, OSError, ValueError, KeyError, TypeError) as error:
        print(f"parent calibration failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
