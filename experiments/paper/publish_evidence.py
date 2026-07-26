#!/usr/bin/env python3
"""Merge one setting ledger into the shared paper ledger and rerender tables."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.evidence.decisions import evaluate_evidence  # noqa: E402
from rsus.evidence.raw import write_ledger  # noqa: E402
from rsus.evidence.registry import load_contract  # noqa: E402
from rsus.evidence.rendering import write_readiness_json  # noqa: E402
from rsus.evidence.schemas import EvidenceLedger, EvidenceValidationError  # noqa: E402
from rsus.evidence.table1 import write_table1  # noqa: E402
from rsus.evidence.tables import render_robustness_table  # noqa: E402


def _read_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"{path} must contain a mapping")
    return value


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row["setting"]), str(row["parent"])


def merge_ledgers(
    existing: Mapping[str, Any] | None,
    incoming: Mapping[str, Any],
) -> dict[str, Any]:
    """Merge by setting/parent, replacing only keys present in ``incoming``."""
    EvidenceLedger.from_mapping(incoming)
    if existing is None:
        merged = {
            "schema_version": 2,
            "rows": list(incoming.get("rows", [])),
            "artifacts": dict(incoming.get("artifacts", {}) or {}),
        }
        EvidenceLedger.from_mapping(merged)
        return merged
    EvidenceLedger.from_mapping(existing)
    rows = {
        _row_key(row): dict(row)
        for row in existing.get("rows", [])
        if isinstance(row, Mapping)
    }
    for row in incoming.get("rows", []):
        if not isinstance(row, Mapping):
            raise EvidenceValidationError("incoming ledger row must be a mapping")
        rows[_row_key(row)] = dict(row)

    artifacts = dict(existing.get("artifacts", {}) or {})
    for artifact_id, artifact in (incoming.get("artifacts", {}) or {}).items():
        if artifact_id in artifacts and artifacts[artifact_id] != artifact:
            raise EvidenceValidationError(
                f"artifact conflict while publishing {artifact_id!r}"
            )
        artifacts[artifact_id] = artifact
    merged = {
        "schema_version": 2,
        "rows": [rows[key] for key in sorted(rows)],
        "artifacts": artifacts,
    }
    EvidenceLedger.from_mapping(merged)
    return merged


def _load_fidelity(directory: Path) -> dict[str, dict[str, Any]]:
    result = {}
    if not directory.is_dir():
        return result
    for path in sorted(directory.glob("*.json")):
        value = _read_mapping(path)
        setting = value.get("setting")
        if (
            isinstance(setting, str)
            and value.get("support") == "declared_setting_fidelity"
        ):
            result[setting] = value
    return result


def publish(
    *,
    ledger_path: Path,
    combined_root: Path,
    evidence_config: Path,
    fidelity_input: Path | None = None,
    primary_setting: str = "tofu_qwen25_7b",
    print_tables: bool = True,
) -> dict[str, str]:
    incoming = _read_mapping(ledger_path.resolve())
    EvidenceLedger.from_mapping(incoming)
    combined_root = combined_root.resolve()
    combined_root.mkdir(parents=True, exist_ok=True)
    lock_path = combined_root / ".publish.lock"
    ledger_out = combined_root / "evidence_ledger.json"

    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = _read_mapping(ledger_out) if ledger_out.is_file() else None
        merged_mapping = merge_ledgers(existing, incoming)
        write_ledger(merged_mapping, ledger_out)

        fidelity_dir = combined_root / "fidelity"
        if fidelity_input is not None:
            summary = _read_mapping(fidelity_input.resolve())
            setting = summary.get("setting")
            if (
                not isinstance(setting, str)
                or summary.get("support") != "declared_setting_fidelity"
            ):
                raise EvidenceValidationError(
                    f"invalid setting fidelity summary: {fidelity_input}"
                )
            fidelity_dir.mkdir(parents=True, exist_ok=True)
            destination = fidelity_dir / f"{setting}.json"
            temporary = destination.with_name(
                f".{destination.name}.{os.getpid()}.tmp"
            )
            shutil.copyfile(fidelity_input.resolve(), temporary)
            os.replace(temporary, destination)

        ledger = EvidenceLedger.from_mapping(merged_mapping)
        contract = load_contract(evidence_config.resolve())
        fidelity = _load_fidelity(fidelity_dir)
        report = evaluate_evidence(contract, ledger, fidelity=fidelity)
        readiness = combined_root / "evidence_readiness.json"
        write_readiness_json(report, readiness)

        table1 = combined_root / "table1.tex"
        table2 = combined_root / "table2.tex"
        write_table1(
            ledger,
            report,
            table1,
            setting=primary_setting,
            allow_incomplete=True,
            fidelity_summary=fidelity.get(primary_setting),
        )
        _atomic_text(
            table2,
            render_robustness_table(
                contract,
                ledger,
                report,
                fidelity=fidelity,
            ),
        )
        status = {
            "schema_version": 1,
            "status": "complete",
            "row_count": len(merged_mapping["rows"]),
            "settings": sorted(
                {str(row["setting"]) for row in merged_mapping["rows"]}
            ),
            "outputs": {
                "ledger": str(ledger_out),
                "readiness": str(readiness),
                "table1": str(table1),
                "table2": str(table2),
            },
        }
        _atomic_text(
            combined_root / "PUBLISH_STATUS.json",
            json.dumps(status, indent=2, sort_keys=True) + "\n",
        )
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    if print_tables:
        print(f"\n===== PAPER TABLE 1: {table1} =====", flush=True)
        print(table1.read_text(encoding="utf-8"), end="", flush=True)
        print("===== END PAPER TABLE 1 =====\n", flush=True)
        print(f"===== PAPER TABLE 2: {table2} =====", flush=True)
        print(table2.read_text(encoding="utf-8"), end="", flush=True)
        print("===== END PAPER TABLE 2 =====\n", flush=True)
    return status["outputs"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument(
        "--evidence-config",
        type=Path,
        default=ROOT / "configs/paper/evidence.yaml",
    )
    parser.add_argument("--fidelity-input", type=Path, default=None)
    parser.add_argument("--primary-setting", default="tofu_qwen25_7b")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        outputs = publish(
            ledger_path=args.ledger,
            combined_root=args.combined_root,
            evidence_config=args.evidence_config,
            fidelity_input=args.fidelity_input,
            primary_setting=args.primary_setting,
        )
        print(f"[PUBLISH] merged ledger: {outputs['ledger']}", flush=True)
        print(f"[PUBLISH] paper Table 1: {outputs['table1']}", flush=True)
        print(f"[PUBLISH] paper Table 2: {outputs['table2']}", flush=True)
        return 0
    except (EvidenceValidationError, OSError, ValueError, KeyError) as error:
        print(f"evidence publish failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
