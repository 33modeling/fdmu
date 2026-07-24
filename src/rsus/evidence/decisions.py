"""Claim and readiness decisions for the PDF-v4 evidence registry."""
from __future__ import annotations

from dataclasses import asdict, replace
from typing import Mapping

from .pdf_v4 import V4Decision, decide_rq1, decide_rq2, decide_rq3
from .registry import CLAIMS, EvidenceContract
from .schemas import EvidenceLedger, EvidenceRow, EvidenceValidationError


def _missing() -> V4Decision:
    return V4Decision(False, False, False, False, None, ("missing row",))


def _require_completed(
    decision: V4Decision, row: EvidenceRow | None
) -> V4Decision:
    if row is None:
        return _missing()
    reasons = list(decision.reasons)
    if not row.attempted:
        reasons.append("not attempted")
    if not row.completed:
        reasons.append("incomplete planned trajectories")
    eligible = decision.eligible and row.attempted and row.completed
    return replace(
        decision,
        eligible=eligible,
        claim_pass=eligible and decision.statistical_pass,
        reasons=tuple(reasons),
    )


def _row_decisions(
    row: EvidenceRow | None, *, alpha: float, minimum_support: int
) -> dict[str, V4Decision]:
    if row is None:
        return {claim: _missing() for claim in CLAIMS}
    return {
        "rq1": _require_completed(
            decide_rq1(
                row.rq1,
                alpha=alpha,
                minimum_support=minimum_support,
            ),
            row,
        ),
        "rq2": _require_completed(
            decide_rq2(
                row.rq2,
                alpha=alpha,
                minimum_support=minimum_support,
            ),
            row,
        ),
        "rq3": _require_completed(
            decide_rq3(
                row.rq3,
                alpha=alpha,
                minimum_support=minimum_support,
            ),
            row,
        ),
    }


def _setting_summary(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    row_decisions: Mapping[tuple[str, str], Mapping[str, V4Decision]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for setting_id, setting in contract.settings.items():
        setting_rows = [
            ledger.rows.get((setting_id, parent)) for parent in setting.parents
        ]
        summary: dict[str, object] = {
            "denominators": {
                "planned_rows": len(setting.parents),
                "attempted_rows": sum(
                    bool(row and row.attempted) for row in setting_rows
                ),
                "completed_rows": sum(
                    bool(row and row.completed) for row in setting_rows
                ),
            }
        }
        for claim in CLAIMS:
            decisions = [
                row_decisions[(setting_id, parent)][claim]
                for parent in setting.parents
            ]
            summary[claim] = {
                "planned": len(decisions),
                "data_complete": sum(item.data_complete for item in decisions),
                "eligible": sum(item.eligible for item in decisions),
                "passed": sum(item.claim_pass for item in decisions),
                "pass": bool(decisions) and all(
                    item.claim_pass for item in decisions
                ),
            }

        parent_groups = []
        for group in contract.multi_setting.parent_groups:
            corrected_alpha = contract.alpha / len(group.parents)
            passed: list[str] = []
            for parent in group.parents:
                decisions = [
                    row_decisions[(setting_id, parent)][claim]
                    for claim in CLAIMS
                ]
                if all(
                    decision.claim_pass
                    and decision.p_iut is not None
                    and decision.p_iut <= corrected_alpha
                    for decision in decisions
                ):
                    passed.append(parent)
            parent_groups.append(
                {
                    "id": group.group_id,
                    "minimum_joint_pass": group.minimum_joint_pass,
                    "multiplicity": group.multiplicity,
                    "familywise_alpha": contract.alpha,
                    "per_parent_iut_alpha": corrected_alpha,
                    "passed_parents": passed,
                    "pass_count": len(passed),
                    "planned_count": len(group.parents),
                    "pass": len(passed) >= group.minimum_joint_pass,
                }
            )
        summary["chain"] = {
            "required_claims": list(CLAIMS),
            "parent_groups": parent_groups,
            "pass": all(group["pass"] for group in parent_groups),
        }
        result[setting_id] = summary
    return result


def _multi_setting_summary(
    contract: EvidenceContract,
    setting_summary: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    rule = contract.multi_setting
    primary_pass = {
        setting: bool(setting_summary[setting]["chain"]["pass"])
        for setting in rule.primary_required
    }
    groups = []
    for group in rule.groups:
        passed = [
            setting
            for setting in group.settings
            if bool(setting_summary[setting]["chain"]["pass"])
        ]
        groups.append(
            {
                "id": group.group_id,
                "minimum_pass": group.minimum_pass,
                "passed_settings": passed,
                "pass_count": len(passed),
                "planned_count": len(group.settings),
                "pass": len(passed) >= group.minimum_pass,
            }
        )
    return {
        "rule_id": rule.rule_id,
        "stress_excluded": list(rule.stress_excluded),
        "primary": primary_pass,
        "groups": groups,
        "setting_support": (
            "minimum joint RQ1+RQ2+RQ3 passes per readout group after "
            "within-group Bonferroni correction"
        ),
        "pass": all(primary_pass.values())
        and all(group["pass"] for group in groups),
    }


def evaluate_evidence(
    contract: EvidenceContract, ledger: EvidenceLedger
) -> dict[str, object]:
    """Evaluate every predeclared RQ1/RQ2/RQ3 row, including missing rows."""
    planned = set(contract.planned_keys)
    extra = set(ledger.rows) - planned
    if extra:
        raise EvidenceValidationError(
            f"ledger contains unregistered setting/parent rows: {sorted(extra)}"
        )
    unknown_artifacts = set(ledger.artifacts) - set(contract.artifacts)
    if unknown_artifacts:
        raise EvidenceValidationError(
            f"ledger contains unregistered artifacts: {sorted(unknown_artifacts)}"
        )

    row_decisions: dict[tuple[str, str], dict[str, V4Decision]] = {}
    rows_json: list[dict[str, object]] = []
    for key in contract.planned_keys:
        row = ledger.rows.get(key)
        decisions = _row_decisions(
            row,
            alpha=contract.alpha,
            minimum_support=contract.minimum_support_units,
        )
        row_decisions[key] = decisions
        rows_json.append(
            {
                "setting": key[0],
                "parent": key[1],
                "attempted": bool(row and row.attempted),
                "completed": bool(row and row.completed),
                "funnel": asdict(row.funnel) if row else None,
                "prediction_alpha": (
                    row.prediction_selection.alpha if row else None
                ),
                "protection_alpha": (
                    row.protection_selection.alpha if row else None
                ),
                **{
                    claim: asdict(decisions[claim])
                    for claim in CLAIMS
                },
            }
        )

    settings = _setting_summary(contract, ledger, row_decisions)
    multi = _multi_setting_summary(contract, settings)
    tables: dict[str, object] = {}
    for table_id, table in contract.tables.items():
        selected_keys = [
            (setting, parent)
            for setting in table.settings
            for parent in contract.settings[setting].parents
        ]
        incomplete_rows = []
        for key in selected_keys:
            row = ledger.rows.get(key)
            if row is None or not row.completed or any(
                not row_decisions[key][claim].data_complete
                for claim in table.claims
            ):
                incomplete_rows.append(f"{key[0]}::{key[1]}")
        missing_artifacts = [
            artifact
            for artifact in table.artifacts
            if not ledger.artifacts.get(artifact)
            or not ledger.artifacts[artifact].completed
        ]
        tables[table_id] = {
            "label": table.label,
            "location": table.location,
            "producer": table.producer,
            "planned_rows": len(selected_keys),
            "completed_rows": len(selected_keys) - len(incomplete_rows),
            "incomplete_rows": incomplete_rows,
            "missing_artifacts": missing_artifacts,
            "ready": not incomplete_rows and not missing_artifacts,
        }

    attempted = sum(
        bool(ledger.rows.get(key) and ledger.rows[key].attempted)
        for key in planned
    )
    completed = sum(
        bool(ledger.rows.get(key) and ledger.rows[key].completed)
        for key in planned
    )
    return {
        "schema_version": 2,
        "decision_alpha": contract.alpha,
        "minimum_support_units": contract.minimum_support_units,
        "denominators": {
            "planned_rows": len(planned),
            "attempted_rows": attempted,
            "completed_rows": completed,
            "missing_rows": len(planned - set(ledger.rows)),
        },
        "rows": rows_json,
        "settings": settings,
        "multi_setting": multi,
        "tables": tables,
        "all_tables_ready": all(bool(table["ready"]) for table in tables.values()),
    }
