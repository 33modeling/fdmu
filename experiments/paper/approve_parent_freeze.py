#!/usr/bin/env python3
"""Approve a resolved parent-freeze proposal without hand-editing YAML."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from experiments.paper.select_tofu_v4 import (  # noqa: E402
    _load,
    _records,
    select_parents,
)


class ApprovalError(ValueError):
    """The proposal is unresolved or differs from its sealed source."""


METADATA_FIELDS = {"freeze_id", "status", "frozen_at_utc"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _body(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key not in METADATA_FIELDS}


def validate_draft(
    proposal: Mapping[str, Any],
    recomputed: Mapping[str, Any],
    *,
    input_path: Path,
) -> None:
    if (
        proposal.get("schema_version") != 1
        or proposal.get("contract") != "tofu-pdf-v4-parent-freeze"
        or proposal.get("status") != "draft"
        or proposal.get("frozen_before_prediction") is not False
    ):
        raise ApprovalError("proposal is not a valid draft parent freeze")
    unresolved = proposal.get("unresolved")
    if not isinstance(unresolved, list) or unresolved:
        raise ApprovalError(f"proposal is unresolved: {unresolved}")
    sources = proposal.get("development_artifacts")
    expected_source = {
        "path": str(input_path.resolve()),
        "sha256": _sha256(input_path),
    }
    if sources != [expected_source]:
        raise ApprovalError("proposal does not match the sealed calibration input")
    if _body(proposal) != _body(recomputed):
        raise ApprovalError("proposal differs from a fresh target-free recomputation")


def _atomic_yaml(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(yaml.safe_dump(dict(value), sort_keys=False), encoding="utf-8")
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    if not args.approve:
        raise ApprovalError("explicit --approve is required")
    proposal_path = args.proposal.resolve()
    input_path = args.input.resolve()
    output_path = args.out.resolve()
    proposal = _load(proposal_path)
    campaign = _load(args.campaign.resolve())
    runtime = _load(args.runtime.resolve())
    rows, sources = _records([input_path])
    recomputed = select_parents(
        rows,
        campaign=campaign,
        runtime=runtime,
        setting=args.setting,
        sources=sources,
        frozen=False,
    )
    validate_draft(proposal, recomputed, input_path=input_path)
    frozen = select_parents(
        rows,
        campaign=campaign,
        runtime=runtime,
        setting=args.setting,
        sources=sources,
        frozen=True,
    )
    if output_path.is_file():
        current = _load(output_path)
        if current.get("status") == "frozen":
            if _body(current) != _body(frozen):
                raise ApprovalError("existing frozen parent contract differs")
            print(f"parent freeze already approved: {output_path}")
            return 0
    _atomic_yaml(output_path, frozen)
    record = {
        "schema_version": 1,
        "approved": True,
        "proposal": str(proposal_path),
        "proposal_sha256": _sha256(proposal_path),
        "selection_input": str(input_path),
        "selection_input_sha256": _sha256(input_path),
        "output": str(output_path),
        "output_sha256": _sha256(output_path),
        "freeze_id": frozen["freeze_id"],
    }
    record_path = proposal_path.parent / "parent_freeze_approval.json"
    record_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"approved parent freeze: {output_path}")
    print(f"freeze id: {frozen['freeze_id']}")
    print(f"approval record: {record_path}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approve", action="store_true")
    parser.add_argument("--setting", default="tofu_qwen25_1p5b")
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=ROOT / "configs/paper/campaign.yaml",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=ROOT / "configs/paper/tofu_v4.yaml",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "configs/paper/tofu_parent_freeze_1p5b.yaml",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except (ApprovalError, ValueError, OSError, KeyError, TypeError) as error:
        print(f"parent freeze approval failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
