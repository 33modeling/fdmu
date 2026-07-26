"""Render the paper's two main claim tables from the validated ledger.

Every cell falls back to the explicit ``\\tblph`` placeholder when its
evidence block is incomplete; a partial campaign can therefore regenerate the
tables at any time without a favorable number ever appearing ahead of its
eligibility checks.  Claim eligibility/pass flags come exclusively from
:mod:`rsus.evidence.decisions`; this module formats, it never decides.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .registry import EvidenceContract
from .schemas import (
    PROTECTION_COMPARATORS,
    PROTECTION_OUTCOMES,
    Effect,
    EvidenceLedger,
    EvidenceRow,
    EvidenceValidationError,
)

PLACEHOLDER = r"\tblph"

# Frozen outcome-blind engineering floors for loss-shake fidelity (RQ2).
FIDELITY_TAU_RHO = 0.80
FIDELITY_TAU_K = 0.70

PARENT_LABELS = {
    "graddiff": "GradDiff",
    "npo": "NPO",
    "simnpo": "SimNPO",
    "gru": "GRU",
    "rmu": "RMU",
    "repnoise": "RepNoise",
    "circuit_breakers": "CB",
}

READOUT_HEADINGS = {
    "output_readout": r"\emph{Output-readout parents}",
    "representation_readout": r"\emph{Representation-readout parents}",
}

AXIS_BY_ROLE = {
    "primary": "Request",
    "scale_boundary": "Model",
    "model_scale": "Model",
    "model_family": "Model",
    "dataset_replication": "Dataset",
    "stress": "Dataset",
}


def _fmt(value: float | None, digits: int = 3, sign: bool = False) -> str:
    if value is None:
        return PLACEHOLDER
    pattern = f"{{:+.{digits}f}}" if sign else f"{{:.{digits}f}}"
    return pattern.format(value)


def _fmt_effect(effect: Effect, *, bound: str, sign: bool = False) -> str:
    """``estimate [bound]`` with the one-sided bound the claim consumes."""
    limit = effect.lower_bound if bound == "lower" else effect.upper_bound
    if effect.estimate is None or limit is None:
        return PLACEHOLDER
    return f"{_fmt(effect.estimate, sign=sign)} [{_fmt(limit, sign=sign)}]"


def _ep(decision: Mapping[str, Any] | None) -> str:
    if decision is None:
        return PLACEHOLDER
    eligible = "y" if decision.get("eligible") else "n"
    passed = "y" if decision.get("claim_pass") else "n"
    if not decision.get("eligible"):
        passed = "--"
    return f"{eligible}/{passed}"


def _row_lookup(report: Mapping[str, Any]) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (record["setting"], record["parent"]): record
        for record in report["rows"]
    }


def _parent_order(contract: EvidenceContract, parents: Sequence[str]) -> list[tuple[str, list[str]]]:
    """Group a setting's parents by declared readout group, keeping order."""
    grouped: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for group in contract.multi_setting.parent_groups:
        members = [parent for parent in group.parents if parent in parents]
        if members:
            grouped.append((group.group_id, members))
            seen.update(members)
    leftover = [parent for parent in parents if parent not in seen]
    if leftover:
        grouped.append(("other", leftover))
    return grouped


def _parent_rows(
    lines: list[str],
    contract: EvidenceContract,
    parents: Sequence[str],
    columns: int,
    cell_fn,
) -> None:
    for group_index, (group_id, members) in enumerate(
        _parent_order(contract, parents)
    ):
        if group_index:
            lines.append(r"\addlinespace[2pt]")
        heading = READOUT_HEADINGS.get(group_id)
        if heading:
            lines.append(rf"\multicolumn{{{columns}}}{{@{{}}l}}{{{heading}}} \\")
        for parent in members:
            label = PARENT_LABELS.get(parent, parent)
            lines.append(f"{label} & " + " & ".join(cell_fn(parent)) + r" \\")


def _sub_ep(decision: Mapping[str, Any] | None, field: str) -> str:
    if decision is None:
        return PLACEHOLDER
    eligible = bool(decision.get("eligible"))
    condition = decision.get(field)
    if not eligible or condition is None:
        return f"{'y' if eligible else 'n'}/--"
    return f"y/{'y' if condition else 'n'}"


def _alpha_cell(row: EvidenceRow) -> str:
    selection = row.prediction_selection
    if selection.alpha is None:
        return PLACEHOLDER
    dagger = r"^\dagger" if selection.fallback else ""
    return rf"${selection.alpha:.2f}{dagger}$"


def _derived_endpoint_rho(row: EvidenceRow, endpoint: str) -> str:
    gain = row.rq1.joint_minus_s0 if endpoint == "s0" else row.rq1.joint_minus_s1
    if row.rq1.joint_rho.estimate is None or gain.estimate is None:
        return PLACEHOLDER
    return _fmt(row.rq1.joint_rho.estimate - gain.estimate)


def _swap_cell(row: EvidenceRow) -> str:
    effect = row.rq1.swap_delta
    if (
        effect.estimate is None
        or effect.lower_bound is None
        or effect.upper_bound is None
    ):
        return PLACEHOLDER
    return (
        f"{_fmt(effect.estimate, sign=True)} "
        f"[{_fmt(effect.lower_bound, sign=True)}, "
        f"{_fmt(effect.upper_bound, sign=True)}]"
    )


def _recall_cell(value: float | None) -> str:
    return PLACEHOLDER if value is None else f"{value:.2f}"


def _fidelity_margin_cell(
    summary: Mapping[str, Any] | None,
    value_key: str,
    bound_key: str,
    tau: float,
) -> str:
    if not summary or summary.get(value_key) is None:
        return PLACEHOLDER
    value = float(summary[value_key])
    bound = summary.get(bound_key)
    if bound is None:
        return f"{value:.2f} [{PLACEHOLDER}]"
    return f"{value:.2f} [{float(bound) - tau:+.2f}]"


def _cost_entries(
    summary: Mapping[str, Any] | None,
) -> dict[str, Mapping[str, Any]]:
    entries: dict[str, Mapping[str, Any]] = {}
    if not summary:
        return entries
    for entry in summary.get("cost", ()) or ():
        if not isinstance(entry, Mapping):
            continue
        profiler = str(entry.get("profiler", ""))
        current = entries.get(profiler)
        if current is None or int(entry.get("R", 0)) > int(current.get("R", 0)):
            entries[profiler] = entry
    return entries


def _cost_number(
    entry: Mapping[str, Any] | None, key: str, *, digits: int = 1
) -> str:
    if not entry or entry.get(key) is None:
        return PLACEHOLDER
    return f"{float(entry[key]):.{digits}f}"


def _fidelity_ep(
    contract: EvidenceContract,
    setting_id: str,
    summary: Mapping[str, Any] | None,
) -> str:
    if setting_id not in contract.fidelity_inputs:
        return r"\textit{n.a.}"
    if not summary or type(summary.get("certificate_passed")) is not bool:
        return "n/--"
    return "y/y" if summary["certificate_passed"] else "y/n"


def _fidelity_count(
    contract: EvidenceContract,
    setting_id: str,
    setting_report: Mapping[str, Any],
) -> str:
    if setting_id not in contract.fidelity_inputs:
        rq2 = setting_report["rq2"]
        if rq2.get("planned", 0) > 0:
            return f"{rq2['eligible']}/{rq2['passed']}"
        return r"\textit{n.a.}"
    rq2 = setting_report["rq2"]
    return f"{rq2['eligible']}/{rq2['passed']}"


def _arm_absolute_cell(row: EvidenceRow, comparator: str) -> str:
    if comparator == "no_repair":
        return _absolute_cell(row, comparator)
    joint = {
        "mean": row.rq3.profile_mean,
        "cvar95": row.rq3.profile_cvar95,
    }
    effects = row.rq3.comparisons.get(comparator, {})
    values: list[float] = []
    for outcome in PROTECTION_OUTCOMES:
        effect = effects.get(outcome)
        if joint[outcome] is None or effect is None or effect.estimate is None:
            return PLACEHOLDER
        values.append(float(joint[outcome]) - effect.estimate)
    return f"{_fmt(values[0])}; {_fmt(values[1])}"


def _feasible_arms_cell(row: EvidenceRow) -> str:
    return "5/5" if row.rq3.all_five_arms_feasible else f"{PLACEHOLDER}/5"


def _slack_value(value: float | None) -> str:
    return PLACEHOLDER if value is None else _fmt(value, digits=2, sign=True)


def _fidelity_cell(
    row: EvidenceRow | None,
    fidelity: Mapping[str, Any] | None,
) -> str:
    """Render frozen fidelity summaries, preferring the validated ledger row."""
    if row is not None:
        rho = row.rq2.f_rho_minus_0p80
        top_k = row.rq2.f_k_minus_0p70
        if (
            rho.estimate is not None
            and rho.lower_bound is not None
            and top_k.estimate is not None
            and top_k.lower_bound is not None
        ):
            return (
                f"{rho.estimate + FIDELITY_TAU_RHO:.2f}/"
                f"{top_k.estimate + FIDELITY_TAU_K:.2f} "
                f"[{rho.lower_bound:+.2f}/{top_k.lower_bound:+.2f}]"
            )
    if not fidelity:
        return PLACEHOLDER
    f_rho = fidelity.get("f_rho")
    f_k = fidelity.get("f_k")
    rho_lb = fidelity.get("f_rho_lb")
    k_lb = fidelity.get("f_k_lb")
    if f_rho is None or f_k is None:
        return PLACEHOLDER
    if rho_lb is None or k_lb is None:
        return f"{f_rho:.2f}/{f_k:.2f} [{PLACEHOLDER}]"
    return (
        f"{f_rho:.2f}/{f_k:.2f} "
        f"[{rho_lb - FIDELITY_TAU_RHO:+.2f}/{k_lb - FIDELITY_TAU_K:+.2f}]"
    )


def _min_gain_cell(row: EvidenceRow) -> str:
    effects = (row.rq1.joint_minus_s0, row.rq1.joint_minus_s1)
    if not all(effect.complete_for_gain() for effect in effects):
        return PLACEHOLDER
    estimate = min(effect.estimate for effect in effects)
    bound = min(effect.lower_bound for effect in effects)
    return f"{_fmt(estimate)} [{_fmt(bound)}]"


def _tail_cell(row: EvidenceRow) -> str:
    tail = row.rq1.tail_lift
    eligible_n = row.rq1.tail_eligible_units
    total_n = row.rq1.reached_valid_units
    counts = f"{eligible_n}/{total_n}"
    if not tail.complete_for_gain():
        return f"{PLACEHOLDER}; {counts}"
    return f"{_fmt(tail.estimate, sign=True)} [{_fmt(tail.lower_bound, sign=True)}]; {counts}"


def _joint_cell(row: EvidenceRow) -> str:
    return _fmt_effect(row.rq1.joint_rho, bound="lower")


def _absolute_cell(row: EvidenceRow, arm: str) -> str:
    if arm == "joint":
        mean, cvar = row.rq3.profile_mean, row.rq3.profile_cvar95
    elif arm == "no_repair":
        mean, cvar = row.rq3.no_repair_mean, row.rq3.no_repair_cvar95
    else:
        raise EvidenceValidationError(f"unsupported absolute protection arm {arm!r}")
    if mean is None or cvar is None:
        return PLACEHOLDER
    return f"{_fmt(mean)}; {_fmt(cvar)}"


def _max_delta_cell(row: EvidenceRow) -> str:
    estimates: list[float] = []
    uppers: list[float] = []
    for comparator in PROTECTION_COMPARATORS:
        outcomes = row.rq3.comparisons.get(comparator, {})
        for outcome in PROTECTION_OUTCOMES:
            effect = outcomes.get(outcome)
            if effect is None or not effect.complete_for_reduction():
                return PLACEHOLDER
            estimates.append(effect.estimate)
            uppers.append(effect.upper_bound)
    return f"{_fmt(max(estimates), sign=True)} [{_fmt(max(uppers), sign=True)}]"


def _min_native_cell(row: EvidenceRow) -> str:
    effects = [
        row.rq3.native_noninferiority.get(comparator)
        for comparator in PROTECTION_COMPARATORS
    ]
    if any(effect is None or not effect.complete_for_gain() for effect in effects):
        return PLACEHOLDER
    estimate = min(effect.estimate for effect in effects)
    bound = min(effect.lower_bound for effect in effects)
    return f"{_fmt(estimate, sign=True)} [{_fmt(bound, sign=True)}]"


def _slack_cell(row: EvidenceRow) -> str:
    forget = row.rq3.min_forget_margin
    utility = row.rq3.min_utility_margin
    if forget is None or utility is None:
        return PLACEHOLDER
    return f"{_fmt(forget, digits=2, sign=True)}/{_fmt(utility, digits=2, sign=True)}"


def _updates_cell(row: EvidenceRow) -> str:
    accepted = row.rq3.repair_updates
    rolled = row.rq3.repair_rollbacks
    if accepted is None or rolled is None:
        return PLACEHOLDER
    def compact(value: float) -> str:
        return f"{value:.2f}".rstrip("0").rstrip(".")

    return f"{compact(accepted)}/{compact(rolled)}"


def render_core_evidence_table(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    report: Mapping[str, Any],
    *,
    fidelity: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Render the five claim-bearing tables in the current paper contract."""
    table = contract.tables.get("main_core_evidence")
    if table is None:
        raise EvidenceValidationError("contract does not register main_core_evidence")
    decisions = _row_lookup(report)
    fidelity = fidelity or {}
    lines = [
        "% Generated by experiments/paper/build_evidence.py; do not edit by hand.",
        "% Incomplete evidence remains an explicit \\tblph placeholder.",
    ]

    def decision(setting_id: str, parent: str, claim: str):
        record = decisions.get((setting_id, parent))
        return record.get(claim) if record else None

    # Table I: prospective prediction and the RQ1 rank condition.
    for setting_id in table.settings:
        setting = contract.settings[setting_id]
        lines += [
            r"\begin{table*}[!t]",
            r"\caption{\textbf{Prospective damage prediction (RQ1, rank condition):",
            r"the sealed mixture against its own ingredients.} Scores are sealed",
            r"before parent runs. Point correlations for both endpoints, the sealed",
            r"mixture with its one-sided 95\% lower bound, and paired gains over",
            r"both endpoints and the frozen simple control are shown.",
            r"\texttt{E/P} denotes eligible/pass.}",
            r"\label{tab:pred-value}",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
        ]
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{@{}lcccccccc@{}}")
        lines.append(r"\toprule")
        lines.append(
            r"\multicolumn{9}{@{}l}{\textbf{Prospective prediction "
            rf"loss-shake validation ({setting.dataset}, {setting.model})}}}} \\"
        )
        lines.append(
            r"Parent & $\widehat\alpha_{\mathrm{pred}}$ & $S_0$ & $S_1$ (prox.) &"
            r" $S_{\widehat\alpha_{\mathrm{pred}}}$ (ours) [LB] &"
            r" $g_G$ [LB] & $g_H$ [LB] & $g_{\rm ctl}$ [LB] & Rank E/P \\"
        )
        lines.append(r"\midrule")

        def prediction_cells(parent: str) -> list[str]:
            row = ledger.rows.get((setting_id, parent))
            rq1 = decision(setting_id, parent, "rq1")
            if row is None:
                return [PLACEHOLDER] * 7 + [_sub_ep(rq1, "rank_pass")]
            return [
                _alpha_cell(row),
                _derived_endpoint_rho(row, "s0"),
                _derived_endpoint_rho(row, "s1"),
                _joint_cell(row),
                _fmt_effect(row.rq1.joint_minus_s0, bound="lower", sign=True),
                _fmt_effect(row.rq1.joint_minus_s1, bound="lower", sign=True),
                _fmt_effect(row.rq2.g_ctl, bound="lower", sign=True),
                _sub_ep(rq1, "rank_pass"),
            ]

        _parent_rows(lines, contract, setting.parents, 9, prediction_cells)
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
        lines.append(r"\end{table*}")

        # Table II: setting-level RQ2 certificate and no-refit swap diagnostic.
        setting_fidelity = fidelity.get(setting_id)
        costs = _cost_entries(setting_fidelity)
        exact = costs.get("exact_energy")
        shake = costs.get("loss_shake")
        lines += [
            "",
            r"\begin{table*}[t]",
            r"\caption{\textbf{Loss Susceptibility estimation fidelity against",
            r"its exact target (RQ2).} Panel A reports the registered setting-level",
            r"certificate. Panel B substitutes exact gradient energy inside the",
            r"frozen profile without refitting and is diagnostic only.}",
            r"\label{tab:loss-susceptibility-fidelity}",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{5pt}",
        ]
        lines.append(r"\resizebox{\textwidth}{!}{%")
        lines.append(r"\begin{tabular}{@{}llccccccc@{}}")
        lines.append(r"\toprule")
        lines.append(
            r"\multicolumn{9}{@{}l}{\textbf{A. Estimator versus direct-gradient "
            r"reference on declared setting-level fidelity support}} \\"
        )
        lines.append(
            r"Estimator & Access & $f_\rho$ [LB$-\tau_\rho$] &"
            r" $f_K$ [LB$-\tau_K$] & Split-bank $\rho$ & Survival &"
            r" Time (s) & Mem (GB) & RQ2 E/P \\"
        )
        lines.append(r"\midrule")
        lines.append(
            r"Direct-gradient reference & reverse, per-cand. & 1.00 [--] &"
            rf" 1.00 [--] & -- & -- & {_cost_number(exact, 'time_seconds_median', digits=2)} &"
            rf" {_cost_number(exact, 'peak_memory_gb_median')} & -- \\"
        )
        split = (
            f"{float(setting_fidelity['split_half_rho']):.2f}"
            if setting_fidelity and setting_fidelity.get("split_half_rho") is not None
            else _cost_number(shake, "split_half_rho_median", digits=2)
        )
        survival = (
            f"{float(setting_fidelity['perturbation_survival']):.2f}"
            if setting_fidelity
            and setting_fidelity.get("perturbation_survival") is not None
            else _cost_number(shake, "survival_median", digits=2)
        )
        lines.append(
            r"Loss Susceptibility, $2R$ forward sweeps (\textbf{ours}) & forward-only &"
            rf" {_fidelity_margin_cell(setting_fidelity, 'f_rho', 'f_rho_lb', FIDELITY_TAU_RHO)} &"
            rf" {_fidelity_margin_cell(setting_fidelity, 'f_k', 'f_k_lb', FIDELITY_TAU_K)} &"
            rf" {split} & {survival} &"
            rf" {_cost_number(shake, 'time_seconds_median', digits=2)} &"
            rf" {_cost_number(shake, 'peak_memory_gb_median')} &"
            rf" {_fidelity_ep(contract, setting_id, setting_fidelity)} \\"
        )
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}}")
        lines += [
            "",
            r"\vspace{6pt}",
            r"\begin{tabular}{@{}lcc@{}}",
            r"\toprule",
            r"\multicolumn{3}{@{}l}{\textbf{B. $q_G^\star$-substituted mixture",
            r"inside the frozen profile (diagnostic only)}} \\",
            r"\midrule",
            r"Parent & $\Delta\rho_{\mathrm{swap}}$ [95\% CI] &"
            r" $g_H$ [LB] (Tab.~\ref{tab:pred-value}) \\",
            r"\midrule",
        ]

        def swap_cells(parent: str) -> list[str]:
            row = ledger.rows.get((setting_id, parent))
            if row is None:
                return [PLACEHOLDER, PLACEHOLDER]
            return [
                _swap_cell(row),
                _fmt_effect(row.rq1.joint_minus_s1, bound="lower", sign=True),
            ]

        _parent_rows(lines, contract, setting.parents, 3, swap_cells)
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

        # Table III: RQ1 harmful-tail condition.
        lines += [
            "",
            r"\begin{table*}[t]",
            r"\caption{\textbf{Harmful-tail recovery (RQ1, tail condition).}",
            r"The harmful tail contains the largest positive realized damages.",
            r"Endpoint and sealed-mixture recall are compared with exact finite-sample",
            r"chance; lift carries its one-sided 95\% lower bound.}",
            r"\label{tab:tail-recovery}",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{@{}lccccccc@{}}",
            r"\toprule",
            r"Parent & Tail-eligible $n/N$ & Chance $q_{\mathrm{tail}}^{\mathrm{eff}}$ &"
            r" $S_0$ & $S_1$ (prox.) & $S_{\widehat\alpha_{\mathrm{pred}}}$ (ours) &"
            r" $L_{\mathrm{tail}}$ [LB] & RQ1 E/P \\",
            r"\midrule",
        ]

        def tail_cells(parent: str) -> list[str]:
            row = ledger.rows.get((setting_id, parent))
            rq1 = decision(setting_id, parent, "rq1")
            if row is None:
                return [PLACEHOLDER] * 6 + [_ep(rq1)]
            tail = row.rq1.tail_lift
            lift = (
                f"{_fmt(tail.estimate, sign=True)} "
                f"[{_fmt(tail.lower_bound, sign=True)}]"
                if tail.complete_for_gain()
                else PLACEHOLDER
            )
            return [
                f"{row.rq1.tail_eligible_units}/{row.rq1.reached_valid_units}",
                _recall_cell(row.rq1.chance_q),
                _recall_cell(row.rq1.tail_recall_s0),
                _recall_cell(row.rq1.tail_recall_s1),
                _recall_cell(row.rq1.tail_recall_joint),
                lift,
                _ep(rq1),
            ]

        _parent_rows(lines, contract, setting.parents, 8, tail_cells)
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

        # Table IV: RQ3 effect condition.
        lines += [
            "",
            r"\begin{table*}[t]",
            r"\caption{\textbf{Fixed-budget protection (RQ3, effect condition):",
            r"allocation is the only free variable.} Damage cells report",
            r"mean;\,$\mathrm{CVaR}_{.95}$ on common held-out support.}",
            r"\label{tab:prot-effect}",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{@{}lccccccc@{}}",
            r"\toprule",
            r"Parent & No repair & Random & $S_0$ & $S_1$ (prox.) &"
            r" $S_{\widehat\alpha_{\mathrm{pred}}}$ (ours) &"
            r" $\max\widehat\Delta_{a,k}$ / $\max U^{95}_{a,k}$ & Effect E/P \\",
            r"\midrule",
        ]

        def effect_cells(parent: str) -> list[str]:
            row = ledger.rows.get((setting_id, parent))
            rq3 = decision(setting_id, parent, "rq3")
            if row is None:
                return [PLACEHOLDER] * 6 + [_sub_ep(rq3, "effect_pass")]
            return [
                _absolute_cell(row, "no_repair"),
                _arm_absolute_cell(row, "repeated_random"),
                _arm_absolute_cell(row, "s0"),
                _arm_absolute_cell(row, "s1"),
                _absolute_cell(row, "joint"),
                _max_delta_cell(row).replace(" [", "/").replace("]", ""),
                _sub_ep(rq3, "effect_pass"),
            ]

        _parent_rows(lines, contract, setting.parents, 8, effect_cells)
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]

        # Table V: RQ3 common-support and feasibility contract.
        lines += [
            "",
            r"\begin{table*}[t]",
            r"\caption{\textbf{Fixed-budget protection (RQ3, contract condition):",
            r"the comparison is constraint-matched.} Native non-inferiority,",
            r"forgetting and utility slack, common-support feasibility, and guarded",
            r"repair updates are reported separately from the damage effect.}",
            r"\label{tab:prot-contract}",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{5pt}",
            r"\begin{tabular}{@{}lcccccc@{}}",
            r"\toprule",
            r"Parent & $\min\widehat h_a$ / $\min L^{95}_a$ & Forget slack &"
            r" Utility slack & Feasible arms & Updates/rollback & RQ3 E/P \\",
            r"\midrule",
        ]

        def contract_cells(parent: str) -> list[str]:
            row = ledger.rows.get((setting_id, parent))
            rq3 = decision(setting_id, parent, "rq3")
            if row is None:
                return [PLACEHOLDER] * 5 + [_ep(rq3)]
            return [
                _min_native_cell(row).replace(" [", "/").replace("]", ""),
                _slack_value(row.rq3.min_forget_margin),
                _slack_value(row.rq3.min_utility_margin),
                _feasible_arms_cell(row),
                _updates_cell(row),
                _ep(rq3),
            ]

        _parent_rows(lines, contract, setting.parents, 7, contract_cells)
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines) + "\n"


def _setting_label(setting) -> str:
    if setting.role in ("model_scale", "model_family", "scale_boundary"):
        label = setting.model
    else:
        label = setting.dataset
    if setting.role == "stress":
        label += " (stress)"
    if setting.role == "primary":
        label = f"held-out {setting.dataset} requests"
    if setting.role == "scale_boundary":
        label += " (boundary)"
    return label


FAILURE_FLAGS = (
    ("missing row", "not attempted"),
    ("not attempted", "not attempted"),
    ("incomplete planned trajectories", "incomplete"),
    ("not all predeclared profiles are valid", "invalid profile"),
    ("prediction support below frozen minimum", "non-reach"),
    ("protection support below frozen minimum", "non-reach"),
    ("prediction lacks complete common support", "no common support"),
    ("five arms do not share complete outcome support", "no common support"),
    ("not all five claim arms are feasible", "infeasible arm"),
    ("forgetting or utility constraint failed", "constraint fail"),
    ("tail lift bound or eligible coverage failed", "tail miss"),
    ("IUT failed", "IUT fail"),
    ("effects incomplete", "incomplete"),
    ("bounds incomplete", "incomplete"),
    ("weight unresolved or fallback", "fallback weight"),
)


def _failure_modes(decisions: Sequence[Mapping[str, Any]]) -> str:
    flags: list[str] = []
    for decision in decisions:
        for claim in ("rq1", "rq2", "rq3"):
            for reason in decision[claim].get("reasons", ()):
                for needle, flag in FAILURE_FLAGS:
                    if needle in reason:
                        if flag not in flags:
                            flags.append(flag)
                        break
    if not flags:
        return "none"
    return "; ".join(flags[:3])


def _sum_funnel(rows: Sequence[EvidenceRow | None], field: str) -> int:
    return sum(getattr(row.funnel, field) for row in rows if row is not None)


def render_robustness_table(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    report: Mapping[str, Any],
    *,
    fidelity: Mapping[str, Mapping[str, Any]] | None = None,
) -> str:
    """Render the split breadth-of-claim and evidence-funnel tables."""
    table = contract.tables.get("main_robustness")
    if table is None:
        raise EvidenceValidationError("contract does not register main_robustness")
    decisions = _row_lookup(report)
    fidelity = fidelity or {}
    lines = [
        "% Generated by experiments/paper/build_evidence.py; do not edit by hand.",
        "% Incomplete evidence remains an explicit \\tblph placeholder.",
        r"\begin{table*}[!t]",
        r"\caption{\textbf{Breadth of the claim across predeclared settings.}",
        r"Each row is one setting on a declared breadth axis; its seven parents",
        r"are fixed in advance and \emph{Plan/done} keeps unfinished rows in the",
        r"denominator. RQ1 and RQ3 count eligible/passing parents; RQ2 reports",
        r"the setting's single registered fidelity certificate.",
        r"\emph{Chain} requires a passing RQ2 certificate plus at least one",
        r"output-readout and one representation-readout parent passing RQ1 and",
        r"RQ3 after the predeclared correction. Stress settings cannot",
        r"rescue a primary failure; failure diagnosis lives in",
        r"Table~\ref{tab:robustness-funnel}.}",
        r"\label{tab:robustness}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\resizebox{0.85\textwidth}{!}{%",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"Axis & Setting & Plan/done & RQ1 parents & RQ2 cert. & RQ3 parents & Chain \\",
        r"\midrule",
    ]
    claim_lines: list[str] = []
    funnel_lines: list[str] = []
    previous_axis: str | None = None
    for setting_id in table.settings:
        setting = contract.settings[setting_id]
        axis = AXIS_BY_ROLE.get(setting.role, "Setting")
        summary = report["settings"][setting_id]
        parents = setting.parents
        rows = [ledger.rows.get((setting_id, parent)) for parent in parents]
        row_decisions = [decisions[(setting_id, parent)] for parent in parents]
        planned = summary["denominators"]["planned_rows"]
        completed = summary["denominators"]["completed_rows"]

        rq1 = summary["rq1"]
        rq3 = summary["rq3"]

        profiles_planned = _sum_funnel(rows, "profiles_planned")
        profiles_valid = _sum_funnel(rows, "profiles_valid")
        reached = _sum_funnel(rows, "trajectories_reached")
        trajectories_planned = _sum_funnel(rows, "trajectories_planned")
        prediction_common = _sum_funnel(rows, "prediction_common")
        feasible = _sum_funnel(rows, "protection_feasible_all_arms")
        reached_valid = _sum_funnel(rows, "reached_with_valid_profile")
        tail_eligible = sum(
            row.rq1.tail_eligible_units for row in rows if row is not None
        )

        # Least-favorable descriptive extrema per research question.  RQ1 and
        # RQ2 members are one-sided lower bounds (worst = min); the RQ3 damage
        # contrasts are one-sided upper bounds (worst = max).  The four native
        # non-inferiority lower bounds live on a different scale and stay out
        # of the single RQ3 scalar.
        rq1_bounds = [
            bound
            for row in rows
            if row is not None
            for bound in (
                row.rq1.joint_rho.lower_bound,
                row.rq1.joint_minus_s0.lower_bound,
                row.rq1.joint_minus_s1.lower_bound,
                row.rq1.tail_lift.lower_bound,
            )
            if bound is not None
        ]
        setting_fidelity = fidelity.get(setting_id) or {}
        rq2_bounds = []
        if setting_fidelity.get("f_rho_lb") is not None:
            rq2_bounds.append(
                float(setting_fidelity["f_rho_lb"]) - FIDELITY_TAU_RHO
            )
        if setting_fidelity.get("f_k_lb") is not None:
            rq2_bounds.append(
                float(setting_fidelity["f_k_lb"]) - FIDELITY_TAU_K
            )
        protection_bounds = [
            effect.upper_bound
            for row in rows
            if row is not None
            for outcomes in row.rq3.comparisons.values()
            for effect in outcomes.values()
            if effect.upper_bound is not None
        ]
        worst_rq1 = min(rq1_bounds) if rq1_bounds else None
        worst_rq2 = min(rq2_bounds) if rq2_bounds else None
        worst_rq3 = max(protection_bounds) if protection_bounds else None
        if worst_rq1 is None and worst_rq2 is None and worst_rq3 is None:
            worst_cell = PLACEHOLDER
        else:
            worst_cell = " / ".join(
                _fmt(value, sign=True) if value is not None else PLACEHOLDER
                for value in (worst_rq1, worst_rq2, worst_rq3)
            )

        attempted_any = any(row is not None for row in rows)
        chain = summary.get("chain", {})
        chain_cell = (
            ("y" if chain.get("pass") else "n") if attempted_any else PLACEHOLDER
        )
        axis_cell = axis if axis != previous_axis else ""
        setting_cell = _setting_label(setting)
        claim_cells = [
            axis_cell,
            setting_cell,
            f"{planned}/{completed}",
            f"{rq1['eligible']}/{rq1['passed']}",
            _fidelity_count(contract, setting_id, summary),
            f"{rq3['eligible']}/{rq3['passed']}",
            chain_cell,
        ]
        claim_lines.append(" & ".join(claim_cells) + r" \\")
        funnel_cells = [
            axis_cell,
            setting_cell,
            (
                f"{profiles_valid}/{profiles_planned}"
                if attempted_any
                else PLACEHOLDER
            ),
            (
                f"{reached}/{trajectories_planned}"
                if attempted_any
                else PLACEHOLDER
            ),
            f"{prediction_common}" if attempted_any else PLACEHOLDER,
            f"{tail_eligible}" if attempted_any else PLACEHOLDER,
            f"{feasible}/{reached_valid}" if attempted_any else PLACEHOLDER,
            worst_cell,
            _failure_modes(row_decisions),
        ]
        funnel_lines.append(" & ".join(funnel_cells) + r" \\")
        previous_axis = axis
    lines.extend(claim_lines)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    lines += [
        "",
        r"\begin{table*}[!t]",
        r"\caption{\textbf{Predeclared evidence funnel and failure boundaries.}",
        r"Columns follow the stages a setting--parent row must survive before",
        r"any claim can pass, so a drop between adjacent columns localizes the",
        r"failure: sealed profiles valid (of planned), trajectories reaching the",
        r"forgetting gate (of planned), audit cells with complete common support,",
        r"tail-eligible cells among them, and rows with all five repair arms",
        r"feasible (of reached rows with a valid profile). No stage removes a",
        r"row from the declared denominator. Worst bounds are least-favorable",
        r"one-sided bounds among attempted rows --- descriptive extrema, not",
        r"claims. Failure modes name the dominant predeclared reasons.}",
        r"\label{tab:robustness-funnel}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.6pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{@{}llccccccl@{}}",
        r"\toprule",
        r"Axis & Setting & Profiles valid & Gate reached & Common $n$ &"
        r" Tail-elig.\ $n$ & All-arm feas. & Worst RQ1/RQ2/RQ3 & Failure modes \\",
        r"\midrule",
    ]
    lines.extend(funnel_lines)
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table*}")
    return "\n".join(lines) + "\n"


def write_tex_tables(
    contract: EvidenceContract,
    ledger: EvidenceLedger,
    report: Mapping[str, Any],
    paper_root: str | Path,
    *,
    fidelity: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[Path]:
    from .rendering import _atomic_write  # shared atomic-replace helper

    root = Path(paper_root).resolve()
    if not root.is_dir() or not (root / "main.tex").is_file():
        raise EvidenceValidationError(
            f"--paper-root must contain main.tex, got {root}"
        )
    outputs = []
    rendered = {
        contract.core_table_output: render_core_evidence_table(
            contract, ledger, report, fidelity=fidelity
        ),
        contract.robustness_table_output: render_robustness_table(
            contract, ledger, report, fidelity=fidelity
        ),
    }
    for relative, text in rendered.items():
        target = (root / relative).resolve()
        try:
            target.relative_to(root)
        except ValueError as error:
            raise EvidenceValidationError(
                "outputs.tex_tables must remain inside --paper-root"
            ) from error
        _atomic_write(target, text)
        outputs.append(target)
    return outputs
