"""Fail-closed RQ1/RQ2/RQ3 decisions for the July-24 PDF contract.

The schema-v2 raw aggregator populates these effects on common support. This
module is the single statistical decision layer used by the paper registry.
"""
from __future__ import annotations

from dataclasses import dataclass

from .schemas import Effect, RQ1Evidence, RQ2Evidence, RQ3Evidence
from .statistics import intersection_union_p


COMPARATORS = ("no_repair", "repeated_random", "s0", "s1")
DAMAGE_OUTCOMES = ("mean", "cvar95")


@dataclass(frozen=True)
class V4Decision:
    data_complete: bool
    eligible: bool
    statistical_pass: bool
    claim_pass: bool
    p_iut: float | None
    reasons: tuple[str, ...]
    rank_pass: bool | None = None
    tail_pass: bool | None = None
    effect_pass: bool | None = None
    contract_pass: bool | None = None


def _gain_complete(effect: Effect) -> bool:
    return effect.complete_for_gain()


def _reduction_complete(effect: Effect) -> bool:
    return effect.complete_for_reduction()


def _gain_iut(effects: list[Effect], alpha: float) -> tuple[float | None, bool]:
    if not all(_gain_complete(effect) for effect in effects):
        return None, False
    p_iut = intersection_union_p(
        effect.p_one_sided for effect in effects if effect.p_one_sided is not None
    )
    passed = all(
        effect.lower_bound is not None and effect.lower_bound > 0.0
        for effect in effects
    ) and p_iut <= alpha
    return p_iut, passed


def decide_rq1(
    evidence: RQ1Evidence,
    *,
    control_gain: Effect | None = None,
    alpha: float = 0.05,
    minimum_support: int = 2,
    minimum_tail_coverage: float = 0.80,
) -> V4Decision:
    rank_effects = [
        evidence.joint_rho,
        evidence.joint_minus_s0,
        evidence.joint_minus_s1,
    ]
    if control_gain is not None:
        rank_effects.append(control_gain)
    effects = [*rank_effects, evidence.tail_lift]
    data_complete = evidence.paired and all(_gain_complete(effect) for effect in effects)
    coverage = (
        evidence.tail_eligible_units / evidence.reached_valid_units
        if evidence.reached_valid_units > 0
        else 0.0
    )
    support_ok = (
        evidence.common_support_units >= minimum_support
        and evidence.common_support_units == evidence.reached_valid_units
    )
    reasons: list[str] = []
    if not data_complete:
        reasons.append("RQ1 four lower-bound effects incomplete")
    if not evidence.selection_valid:
        reasons.append("alpha_pred unresolved or fallback")
    if not evidence.profile_valid:
        reasons.append("profile validity failed")
    if not support_ok:
        reasons.append("RQ1 common support incomplete or below minimum")
    if coverage < minimum_tail_coverage:
        reasons.append("positive-damage tail eligibility coverage below 0.80")
    eligible = all(
        (data_complete, evidence.selection_valid, evidence.profile_valid, support_ok,
         coverage >= minimum_tail_coverage)
    )
    _rank_p, rank_pass = _gain_iut(rank_effects, alpha)
    _tail_p, tail_pass = _gain_iut([evidence.tail_lift], alpha)
    tail_pass = tail_pass and coverage >= minimum_tail_coverage
    p_iut, statistical_pass = _gain_iut(effects, alpha)
    statistical_pass = statistical_pass and tail_pass
    if data_complete and not statistical_pass:
        reasons.append("RQ1 four-way one-sided IUT failed")
    return V4Decision(
        data_complete,
        eligible,
        statistical_pass,
        eligible and statistical_pass,
        p_iut,
        tuple(reasons),
        rank_pass=rank_pass,
        tail_pass=tail_pass,
    )


def decide_rq2(
    evidence: RQ2Evidence,
    *,
    alpha: float = 0.05,
    minimum_support: int = 2,
) -> V4Decision:
    effects = [
        evidence.f_rho_minus_0p80,
        evidence.f_k_minus_0p70,
        evidence.g_h,
        evidence.g_ctl,
    ]
    data_complete = evidence.paired and all(_gain_complete(effect) for effect in effects)
    validity = all(
        (
            evidence.perturbations_valid,
            evidence.exact_reference_valid,
            evidence.common_control_support,
        )
    )
    support_ok = evidence.common_support_units >= minimum_support
    reasons: list[str] = []
    if not data_complete:
        reasons.append("RQ2 fidelity/add-value effects incomplete")
    if not validity:
        reasons.append("RQ2 perturbation, exact-reference, or control validity failed")
    if not support_ok:
        reasons.append("RQ2 common support below minimum")
    eligible = data_complete and validity and support_ok
    p_iut, statistical_pass = _gain_iut(effects, alpha)
    if data_complete and not statistical_pass:
        reasons.append("RQ2 four-way one-sided IUT failed")
    return V4Decision(
        data_complete,
        eligible,
        statistical_pass,
        eligible and statistical_pass,
        p_iut,
        tuple(reasons),
    )


def decide_rq3(
    evidence: RQ3Evidence,
    *,
    alpha: float = 0.05,
    minimum_support: int = 2,
) -> V4Decision:
    damage_effects: list[Effect] = []
    complete_shape = set(evidence.comparisons) == set(COMPARATORS)
    if complete_shape:
        for comparator in COMPARATORS:
            outcomes = evidence.comparisons[comparator]
            if set(outcomes) != set(DAMAGE_OUTCOMES):
                complete_shape = False
                break
            damage_effects.extend(outcomes[outcome] for outcome in DAMAGE_OUTCOMES)
    native_shape = set(evidence.native_noninferiority) == set(COMPARATORS)
    native_effects = [
        evidence.native_noninferiority[comparator]
        for comparator in COMPARATORS
        if comparator in evidence.native_noninferiority
    ]
    data_complete = (
        evidence.paired
        and complete_shape
        and native_shape
        and len(damage_effects) == 8
        and all(_reduction_complete(effect) for effect in damage_effects)
        and len(native_effects) == 4
        and all(_gain_complete(effect) for effect in native_effects)
        and evidence.min_forget_margin is not None
        and evidence.min_utility_margin is not None
    )
    constraints_ok = (
        evidence.min_forget_margin is not None
        and evidence.min_forget_margin >= 0.0
        and evidence.min_utility_margin is not None
        and evidence.min_utility_margin >= 0.0
    )
    support_ok = evidence.common_support and evidence.common_support_units >= minimum_support
    reasons: list[str] = []
    if not data_complete:
        reasons.append("RQ3 eight damage and four native-NI effects incomplete")
    if not evidence.selection_valid:
        reasons.append("alpha_prot unresolved or fallback")
    if not evidence.all_random_draws_complete:
        reasons.append("repeated-random draw roster incomplete")
    if not evidence.all_five_arms_feasible:
        reasons.append("not all five arms feasible")
    if not constraints_ok:
        reasons.append("forgetting or utility margin failed")
    if not support_ok:
        reasons.append("RQ3 common support incomplete or below minimum")
    eligible = all(
        (
            data_complete,
            evidence.selection_valid,
            evidence.all_random_draws_complete,
            evidence.all_five_arms_feasible,
            constraints_ok,
            support_ok,
        )
    )

    p_iut = statistical_pass = None
    effect_pass: bool | None = None
    contract_pass: bool | None = None
    if data_complete:
        all_p = [
            effect.p_one_sided
            for effect in damage_effects + native_effects
            if effect.p_one_sided is not None
        ]
        p_iut = intersection_union_p(all_p)
        effect_pass = all(
            effect.upper_bound is not None and effect.upper_bound < 0.0
            for effect in damage_effects
        )
        contract_pass = all(
            effect.lower_bound is not None and effect.lower_bound > 0.0
            for effect in native_effects
        )
        statistical_pass = effect_pass and contract_pass and p_iut <= alpha
        if not statistical_pass:
            reasons.append("RQ3 twelve-way one-sided IUT failed")
    else:
        statistical_pass = False
    return V4Decision(
        data_complete,
        eligible,
        statistical_pass,
        eligible and statistical_pass,
        p_iut,
        tuple(reasons),
        effect_pass=effect_pass,
        contract_pass=contract_pass,
    )
