"""Render the claim-bearing Table 1 directly from a validated evidence ledger."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

from .schemas import Effect, EvidenceLedger, EvidenceRow, EvidenceValidationError


PARENT_LABELS = {
    "graddiff": "GradDiff",
    "npo": "NPO",
    "simnpo": "SimNPO",
    "gru": "GRU",
    "rmu": "RMU",
    "repnoise": "RepNoise",
    "circuit_breakers": "CB",
}
PARENT_GROUPS = (
    ("Output-readout parents", ("graddiff", "npo", "simnpo", "gru")),
    (
        "Representation-readout parents",
        ("rmu", "repnoise", "circuit_breakers"),
    ),
)


def _number(value: float | None, *, digits: int = 3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


def _effect(
    effect: Effect,
    *,
    bound: str,
    offset: float = 0.0,
) -> str:
    boundary = (
        effect.lower_bound if bound == "lower" else effect.upper_bound
    )
    if effect.estimate is None or boundary is None:
        return "--"
    return f"{effect.estimate + offset:.3f} [{boundary + offset:.3f}]"


def _decision(value: Mapping[str, object] | None) -> str:
    if value is None:
        return "--/--"
    eligible = "Y" if value.get("eligible") is True else "N"
    passed = "Y" if value.get("claim_pass") is True else "N"
    return f"{eligible}/{passed}"


def _panel_a_complete(row: EvidenceRow) -> bool:
    effects = (
        row.rq1.joint_rho,
        row.rq1.joint_minus_s0,
        row.rq1.joint_minus_s1,
        row.rq1.tail_lift,
        row.rq2.f_rho_minus_0p80,
        row.rq2.f_k_minus_0p70,
        row.rq2.g_h,
        row.rq2.g_ctl,
    )
    return all(
        effect.estimate is not None and effect.lower_bound is not None
        for effect in effects
    )


def _panel_b_complete(row: EvidenceRow) -> bool:
    comparisons = [
        effect
        for outcomes in row.rq3.comparisons.values()
        for effect in outcomes.values()
    ]
    native = list(row.rq3.native_noninferiority.values())
    return (
        all(
            value is not None
            for value in (
                row.rq3.profile_mean,
                row.rq3.profile_cvar95,
                row.rq3.no_repair_mean,
                row.rq3.no_repair_cvar95,
                row.rq3.min_forget_margin,
                row.rq3.min_utility_margin,
                row.rq3.repair_updates,
                row.rq3.repair_rollbacks,
            )
        )
        and len(comparisons) == 8
        and all(
            effect.estimate is not None and effect.upper_bound is not None
            for effect in comparisons
        )
        and len(native) == 4
        and all(
            effect.estimate is not None and effect.lower_bound is not None
            for effect in native
        )
    )


def _required(row: EvidenceRow, setting: str, parent: str) -> None:
    missing = []
    if not _panel_a_complete(row):
        missing.append("RQ1/RQ2 effects")
    if not _panel_b_complete(row):
        missing.append("RQ3 summaries/effects")
    if missing:
        raise EvidenceValidationError(
            f"Table 1 row {setting}/{parent} is incomplete: {', '.join(missing)}"
        )


def _panel_a_row(
    row: EvidenceRow,
    decisions: Mapping[str, Mapping[str, object]],
) -> str:
    gains = (row.rq1.joint_minus_s0, row.rq1.joint_minus_s1)
    min_gain = min(float(effect.estimate) for effect in gains)
    min_gain_lb = min(float(effect.lower_bound) for effect in gains)
    fidelity = (
        _effect(row.rq2.f_rho_minus_0p80, bound="lower", offset=0.80)
        + " / "
        + _effect(row.rq2.f_k_minus_0p70, bound="lower", offset=0.70)
    )
    eligible = (
        f"{row.rq1.tail_eligible_units}/{row.rq1.reached_valid_units}"
    )
    return " & ".join(
        (
            PARENT_LABELS.get(row.parent, row.parent),
            _effect(row.rq1.joint_rho, bound="lower"),
            f"{min_gain:.3f} [{min_gain_lb:.3f}]",
            fidelity,
            _effect(row.rq2.g_ctl, bound="lower"),
            f"{_effect(row.rq1.tail_lift, bound='lower')}; {eligible}",
            _decision(decisions.get("rq1")),
            _decision(decisions.get("rq2")),
        )
    ) + r" \\"


def _panel_b_row(
    row: EvidenceRow,
    decisions: Mapping[str, Mapping[str, object]],
) -> str:
    comparisons = [
        effect
        for outcomes in row.rq3.comparisons.values()
        for effect in outcomes.values()
    ]
    worst = max(
        comparisons,
        key=lambda effect: float(effect.upper_bound),
    )
    native = min(
        row.rq3.native_noninferiority.values(),
        key=lambda effect: float(effect.lower_bound),
    )
    return " & ".join(
        (
            PARENT_LABELS.get(row.parent, row.parent),
            f"{_number(row.rq3.profile_mean)}; {_number(row.rq3.profile_cvar95)}",
            (
                f"{_number(row.rq3.no_repair_mean)}; "
                f"{_number(row.rq3.no_repair_cvar95)}"
            ),
            _effect(worst, bound="upper"),
            _effect(native, bound="lower"),
            (
                f"{_number(row.rq3.min_forget_margin)} / "
                f"{_number(row.rq3.min_utility_margin)}"
            ),
            (
                f"{_number(row.rq3.repair_updates, digits=1)} / "
                f"{_number(row.rq3.repair_rollbacks, digits=1)}"
            ),
            _decision(decisions.get("rq3")),
        )
    ) + r" \\"


def render_table1(
    ledger: EvidenceLedger,
    report: Mapping[str, object],
    *,
    setting: str,
    allow_incomplete: bool = False,
) -> str:
    """Return a two-panel LaTeX table for one predeclared setting."""
    report_rows = report.get("rows")
    if not isinstance(report_rows, list):
        raise EvidenceValidationError("readiness report lacks row decisions")
    decision_by_parent = {
        str(item["parent"]): item
        for item in report_rows
        if isinstance(item, Mapping) and item.get("setting") == setting
    }
    rows: dict[str, EvidenceRow | None] = {
        parent: ledger.rows.get((setting, parent))
        for _label, parents in PARENT_GROUPS
        for parent in parents
    }
    if not allow_incomplete:
        for parent, row in rows.items():
            if row is None or not _panel_a_complete(row):
                raise EvidenceValidationError(
                    f"Table 1 row {setting}/{parent} is missing"
                )
            _required(row, setting, parent)

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Claim-bearing evidence by parent. E/P denotes eligible/pass.}",
        r"\label{tab:core-evidence}",
        r"\small",
        r"\setlength{\tabcolsep}{3pt}",
        r"\textbf{A. Prospective prediction and loss-shake validation}\\[-2pt]",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        (
            r"Parent & Joint $\rho$ [LB] & $\min(g_G,g_H)$ [min LB] & "
            r"$f_\rho/f_K$ [LB] & $g_{\rm ctl}$ [LB] & "
            r"$L_{\rm tail}$ [LB]; eligible $n/N$ & RQ1 E/P & RQ2 E/P \\"
        ),
        r"\midrule",
    ]
    for group_label, parents in PARENT_GROUPS:
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{group_label}}}}} \\")
        for parent in parents:
            row = rows[parent]
            if row is None or not _panel_a_complete(row):
                lines.append(
                    f"{PARENT_LABELS[parent]} & "
                    + " & ".join(["--"] * 7)
                    + r" \\"
                )
            else:
                lines.append(
                    _panel_a_row(row, decision_by_parent.get(parent, {}))
                )
    lines.extend(
        (
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{4pt}",
            r"\textbf{B. Constraint-matched fixed-budget protection}\\[-2pt]",
            r"\begin{tabular}{lccccccc}",
            r"\toprule",
            (
                r"Parent & Profile mean; CVaR & No-repair mean; CVaR & "
                r"$\max_{a,k}\Delta_{a,k}$ [UCB] & $\min_a h_a$ [LB] & "
                r"min F/U slack & updates/rollback & RQ3 E/P \\"
            ),
            r"\midrule",
        )
    )
    for group_label, parents in PARENT_GROUPS:
        lines.append(rf"\multicolumn{{8}}{{l}}{{\textit{{{group_label}}}}} \\")
        for parent in parents:
            row = rows[parent]
            if row is None or not _panel_b_complete(row):
                lines.append(
                    f"{PARENT_LABELS[parent]} & "
                    + " & ".join(["--"] * 7)
                    + r" \\"
                )
            else:
                lines.append(
                    _panel_b_row(row, decision_by_parent.get(parent, {}))
                )
    lines.extend((r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""))
    return "\n".join(lines)


def write_table1(
    ledger: EvidenceLedger,
    report: Mapping[str, object],
    path: str | Path,
    *,
    setting: str,
    allow_incomplete: bool = False,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        render_table1(
            ledger,
            report,
            setting=setting,
            allow_incomplete=allow_incomplete,
        ),
        encoding="utf-8",
    )
    os.replace(temporary, destination)
    return destination
