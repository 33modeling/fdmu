"""Claim and readiness decisions for the PDF-v4 evidence registry."""
from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Mapping

from .pdf_v4 import V4Decision, decide_rq1, decide_rq2, decide_rq3
from .registry import CLAIMS, EvidenceContract
from .schemas import EvidenceLedger, EvidenceRow, EvidenceValidationError


def _missing() -> V4Decision:
    return V4Decision(False, False, False, False, None, ("missing row",))


def _number(summary: Mapping[str, object], key: str) -> float | None:
    value = summary.get(key)
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _fidelity_decision(
    summary: Mapping[str, object] | None, *, alpha: float
) -> V4Decision:
    """Evaluate the latest paper's setting-level RQ2 certificate."""
    if summary is None:
        return V4Decision(
            False, False, False, False, None, ("fidelity summary missing",)
        )
    values = {
        key: _number(summary, key)
        for key in ("f_rho", "f_k", "f_rho_lb", "f_k_lb", "tau_rho", "tau_k")
    }
    certificate = summary.get("certificate_passed")
    data_complete = (
        all(value is not None for value in values.values())
        and type(certificate) is bool
    )
    reasons: list[str] = []
    if not data_complete:
        reasons.append("fidelity measurements or bounds incomplete")
    thresholds_frozen = bool(
        data_complete
        and math.isclose(values["tau_rho"], 0.80, abs_tol=1e-12)
        and math.isclose(values["tau_k"], 0.70, abs_tol=1e-12)
    )
    if data_complete and not thresholds_frozen:
        reasons.append("fidelity thresholds differ from frozen contract")

    # The certificate owns the outcome-blind split-bank, perturbation, and
    # integrity checks. A failed certificate is therefore ineligible even if
    # its rank bounds happen to exceed the statistical floors.
    eligible = bool(data_complete and thresholds_frozen and certificate)
    statistical_pass = bool(
        data_complete
        and values["f_rho_lb"] > values["tau_rho"]
        and values["f_k_lb"] > values["tau_k"]
    )
    if data_complete and not certificate:
        reasons.append("fidelity certificate eligibility failed")
    if data_complete and not statistical_pass:
        reasons.append("fidelity lower-bound IUT failed")
    p_values = [
        _number(summary, "f_rho_p_one_sided"),
        _number(summary, "f_k_p_one_sided"),
    ]
    p_iut = max(p_values) if all(value is not None for value in p_values) else None
    if p_iut is not None:
        statistical_pass = statistical_pass and p_iut <= alpha
    return V4Decision(
        data_complete,
        eligible,
        statistical_pass,
        eligible and statistical_pass,
        p_iut,
        tuple(reasons),
    )


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
    row: EvidenceRow | None,
    *,
    alpha: float,
    minimum_support: int,
    fidelity_decision: V4Decision,
    setting_level_fidelity: bool,
) -> dict[str, V4Decision]:
    if row is None:
        return {
            "rq1": _missing(),
            "rq2": fidelity_decision,
            "rq3": _missing(),
        }
    return {
        "rq1": _require_completed(
            decide_rq1(
                row.rq1,
                control_gain=row.rq2.g_ctl,
                alpha=alpha,
                minimum_support=minimum_support,
            ),
            row,
        ),
        "rq2": (
            fidelity_decision
            if setting_level_fidelity
            else _require_completed(
                decide_rq2(
                    row.rq2,
                    alpha=alpha,
                    minimum_support=minimum_support,
                ),
                row,
            )
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
    fidelity_decisions: Mapping[str, V4Decision],
    setting_level_fidelity: bool,
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
        parent_claims = ("rq1", "rq3") if setting_level_fidelity else CLAIMS
        for claim in parent_claims:
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
        fidelity = fidelity_decisions[setting_id]
        if setting_level_fidelity:
            fidelity_planned = int(setting_id in contract.fidelity_inputs)
            summary["rq2"] = {
                "planned": fidelity_planned,
                "data_complete": (
                    int(fidelity.data_complete) if fidelity_planned else 0
                ),
                "eligible": int(fidelity.eligible) if fidelity_planned else 0,
                "passed": int(fidelity.claim_pass) if fidelity_planned else 0,
                "pass": fidelity.claim_pass if fidelity_planned else None,
            }

        parent_groups = []
        for group in contract.multi_setting.parent_groups:
            corrected_alpha = contract.alpha / len(group.parents)
            passed: list[str] = []
            for parent in group.parents:
                decisions = [
                    row_decisions[(setting_id, parent)][claim]
                    for claim in parent_claims
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
            "required_claims": ["rq1", "rq2", "rq3"],
            "parent_groups": parent_groups,
            "pass": (
                bool(fidelity.claim_pass)
                if setting_level_fidelity
                else True
            )
            and all(group["pass"] for group in parent_groups),
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
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    fidelity: Mapping[str, Mapping[str, object]] | None = None,
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

    setting_level_fidelity = fidelity is not None
    fidelity = fidelity or {}
    extra_fidelity = set(fidelity) - set(contract.settings)
    if extra_fidelity:
        raise EvidenceValidationError(
            f"fidelity summaries contain unregistered settings: "
            f"{sorted(extra_fidelity)}"
        )
    fidelity_decisions = {
        setting_id: _fidelity_decision(
            fidelity.get(setting_id), alpha=contract.alpha
        )
        for setting_id in contract.settings
    }

    row_decisions: dict[tuple[str, str], dict[str, V4Decision]] = {}
    rows_json: list[dict[str, object]] = []
    for key in contract.planned_keys:
        row = ledger.rows.get(key)
        decisions = _row_decisions(
            row,
            alpha=contract.alpha,
            minimum_support=contract.minimum_support_units,
            fidelity_decision=fidelity_decisions[key[0]],
            setting_level_fidelity=setting_level_fidelity,
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

    settings = _setting_summary(
        contract,
        ledger,
        row_decisions,
        fidelity_decisions,
        setting_level_fidelity,
    )
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
            parent_claims = [
                claim
                for claim in table.claims
                if not (setting_level_fidelity and claim == "rq2")
            ]
            if row is None or not row.completed or any(
                not row_decisions[key][claim].data_complete
                for claim in parent_claims
            ):
                incomplete_rows.append(f"{key[0]}::{key[1]}")
        incomplete_fidelity_settings = [
            setting
            for setting in table.settings
            if setting_level_fidelity
            and "rq2" in table.claims
            and setting in contract.fidelity_inputs
            and not fidelity_decisions[setting].data_complete
        ]
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
            "incomplete_fidelity_settings": incomplete_fidelity_settings,
            "missing_artifacts": missing_artifacts,
            "ready": (
                not incomplete_rows
                and not incomplete_fidelity_settings
                and not missing_artifacts
            ),
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
