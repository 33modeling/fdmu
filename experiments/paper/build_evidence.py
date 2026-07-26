"""Validate the complete paper evidence ledger and render claim-safe outputs.

Examples
--------
Readiness only (missing ledger rows remain in the planned denominator)::

    python experiments/paper/build_evidence.py

Validate all artifacts and atomically update the authoritative paper macros::

    python experiments/paper/build_evidence.py --paper-root ../paper --require-ready
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.evidence.decisions import evaluate_evidence  # noqa: E402
from rsus.evidence.registry import load_contract  # noqa: E402
from rsus.evidence.rendering import (  # noqa: E402
    write_readiness_json,
    write_tex_macros,
)
from rsus.evidence.table1 import write_table1  # noqa: E402
from rsus.evidence.tables import write_tex_tables  # noqa: E402
from rsus.evidence.schemas import (  # noqa: E402
    EvidenceLedger,
    EvidenceValidationError,
    validate_artifact_files,
)


def _fidelity_paths(
    contract, overrides: list[str]
) -> dict[str, tuple[Path, bool]]:
    paths = {
        setting_id: (_resolve_repo_path(relative), False)
        for setting_id, relative in contract.fidelity_inputs.items()
    }
    seen: set[str] = set()
    for raw in overrides:
        if "=" not in raw:
            raise EvidenceValidationError(
                "--fidelity-input must use SETTING=PATH syntax"
            )
        setting_id, value = (part.strip() for part in raw.split("=", 1))
        if setting_id not in contract.fidelity_inputs:
            raise EvidenceValidationError(
                f"--fidelity-input setting {setting_id!r} is not predeclared"
            )
        if setting_id in seen:
            raise EvidenceValidationError(
                f"--fidelity-input setting {setting_id!r} is duplicated"
            )
        if not value:
            raise EvidenceValidationError("--fidelity-input path must be non-empty")
        seen.add(setting_id)
        paths[setting_id] = (_resolve_repo_path(value), True)
    return paths


def _load_fidelity_inputs(
    contract, overrides: list[str] | None = None
) -> dict[str, dict]:
    """Load per-setting fidelity summaries named by the frozen contract.

    Missing files keep RQ2 incomplete. Failed certificates are retained so
    readiness distinguishes measured failure from absent evidence.
    """
    import json

    result: dict[str, dict] = {}
    for setting_id, (path, required) in _fidelity_paths(
        contract, overrides or []
    ).items():
        if not path.is_file():
            if required:
                raise EvidenceValidationError(
                    f"explicit fidelity input is missing: {path}"
                )
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise EvidenceValidationError(
                f"fidelity_inputs.{setting_id} is not valid JSON: {error}"
            ) from error
        if not isinstance(payload, dict):
            raise EvidenceValidationError(
                f"fidelity_inputs.{setting_id} root must be a mapping"
            )
        if payload.get("setting") != setting_id:
            raise EvidenceValidationError(
                f"fidelity_inputs.{setting_id} carries setting "
                f"{payload.get('setting')!r}; refusing a mismatched summary"
            )
        if payload.get("support") != "declared_setting_fidelity":
            raise EvidenceValidationError(
                f"fidelity_inputs.{setting_id} is not measured on the "
                "declared setting-level fidelity support"
            )
        result[setting_id] = payload
    return result


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "paper" / "evidence.yaml"),
        help="predeclared evidence registry",
    )
    parser.add_argument(
        "--ledger",
        default=None,
        help="normalized evidence ledger JSON; defaults to config.ledger",
    )
    parser.add_argument(
        "--readiness-out",
        default=None,
        help="readiness JSON path; defaults to config.outputs.readiness_json",
    )
    parser.add_argument(
        "--paper-root",
        default=None,
        help="authoritative paper root containing main.tex; writes five macros",
    )
    parser.add_argument(
        "--require-ready",
        action="store_true",
        help="return exit status 2 unless every registered table is data-ready",
    )
    parser.add_argument(
        "--fidelity-input",
        action="append",
        default=[],
        metavar="SETTING=PATH",
        help=(
            "override one predeclared setting-level fidelity summary; repeatable. "
            "Explicit paths are required to exist and never alter the frozen roster"
        ),
    )
    parser.add_argument(
        "--table1-out",
        default=None,
        help="optional generated Table 1 LaTeX path",
    )
    parser.add_argument(
        "--table1-setting",
        default="tofu_qwen25_1p5b",
        help="setting rendered into --table1-out",
    )
    parser.add_argument(
        "--allow-incomplete-table1",
        action="store_true",
        help="render explicit dashes instead of failing on incomplete Table 1 rows",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config_path = _resolve_repo_path(args.config).resolve()
        contract = load_contract(config_path)
        ledger_path = _resolve_repo_path(args.ledger or contract.ledger_path).resolve()
        ledger = EvidenceLedger.read(ledger_path) if ledger_path.is_file() else EvidenceLedger.empty()
        validate_artifact_files(ledger, repository_root=ROOT)
        fidelity = _load_fidelity_inputs(contract, args.fidelity_input)
        report = evaluate_evidence(contract, ledger, fidelity=fidelity)
        report["sources"] = {
            "config": str(config_path),
            "ledger": str(ledger_path),
            "ledger_exists": ledger_path.is_file(),
        }
        readiness_path = _resolve_repo_path(
            args.readiness_out or contract.readiness_output
        ).resolve()
        write_readiness_json(report, readiness_path)
        print(f"wrote readiness: {readiness_path}")
        if args.paper_root:
            paper_root = _resolve_repo_path(args.paper_root)
            macro_path = write_tex_macros(contract, ledger, report, paper_root)
            print(f"wrote paper macros: {macro_path}")
            table_paths = write_tex_tables(
                contract,
                ledger,
                report,
                paper_root,
                fidelity=fidelity,
            )
            for table_path in table_paths:
                print(f"wrote paper table: {table_path}")
        if args.table1_out:
            table_path = write_table1(
                ledger,
                report,
                _resolve_repo_path(args.table1_out),
                setting=args.table1_setting,
                allow_incomplete=args.allow_incomplete_table1,
                fidelity_summary=fidelity.get(args.table1_setting),
            )
            print(f"wrote Table 1: {table_path}")
        denominators = report["denominators"]
        print(
            "rows planned/attempted/completed: "
            f"{denominators['planned_rows']}/"
            f"{denominators['attempted_rows']}/"
            f"{denominators['completed_rows']}"
        )
        print(
            "multi-setting claim: "
            + ("PASS" if report["multi_setting"]["pass"] else "NOT LICENSED")
        )
        if args.require_ready and not report["all_tables_ready"]:
            return 2
        return 0
    except EvidenceValidationError as error:
        print(f"evidence validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
