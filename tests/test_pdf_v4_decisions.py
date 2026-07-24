"""Exact three-claim decision contract from PDF Sections 4.6--4.8."""
from __future__ import annotations

from rsus.evidence.pdf_v4 import (
    COMPARATORS,
    DAMAGE_OUTCOMES,
    RQ1Evidence,
    RQ2Evidence,
    RQ3Evidence,
    decide_rq1,
    decide_rq2,
    decide_rq3,
)
from rsus.evidence.schemas import Effect


def _gain() -> Effect:
    return Effect(estimate=0.2, lower_bound=0.1, p_one_sided=0.01)


def _reduction() -> Effect:
    return Effect(estimate=-0.2, upper_bound=-0.1, p_one_sided=0.01)


def test_rq1_requires_absolute_rho_tail_lift_and_coverage():
    evidence = RQ1Evidence(
        paired=True,
        selection_valid=True,
        profile_valid=True,
        common_support_units=10,
        reached_valid_units=10,
        tail_eligible_units=8,
        joint_rho=_gain(),
        joint_minus_s0=_gain(),
        joint_minus_s1=_gain(),
        tail_lift=_gain(),
    )
    assert decide_rq1(evidence).claim_pass
    assert not decide_rq1(
        RQ1Evidence(**{**evidence.__dict__, "tail_eligible_units": 7})
    ).eligible


def test_rq2_is_separate_four_way_iut():
    evidence = RQ2Evidence(
        paired=True,
        perturbations_valid=True,
        exact_reference_valid=True,
        common_control_support=True,
        common_support_units=4,
        f_rho_minus_0p80=_gain(),
        f_k_minus_0p70=_gain(),
        g_h=_gain(),
        g_ctl=_gain(),
    )
    assert decide_rq2(evidence).claim_pass
    failed = RQ2Evidence(**{**evidence.__dict__, "common_control_support": False})
    assert not decide_rq2(failed).claim_pass


def test_rq3_requires_eight_damage_and_four_native_bounds():
    evidence = RQ3Evidence(
        paired=True,
        selection_valid=True,
        all_random_draws_complete=True,
        all_five_arms_feasible=True,
        common_support=True,
        common_support_units=5,
        min_forget_margin=0.1,
        min_utility_margin=0.1,
        damage={
            comparator: {outcome: _reduction() for outcome in DAMAGE_OUTCOMES}
            for comparator in COMPARATORS
        },
        native_noninferiority={comparator: _gain() for comparator in COMPARATORS},
    )
    decision = decide_rq3(evidence)
    assert decision.claim_pass
    assert decision.p_iut == 0.01

    missing_native = dict(evidence.native_noninferiority)
    missing_native.pop("s1")
    incomplete = RQ3Evidence(
        **{**evidence.__dict__, "native_noninferiority": missing_native}
    )
    assert not decide_rq3(incomplete).data_complete


def test_rq3_native_noninferiority_cannot_be_replaced_by_damage_pass():
    adverse_native = {
        comparator: Effect(estimate=-0.1, lower_bound=-0.2, p_one_sided=0.5)
        for comparator in COMPARATORS
    }
    evidence = RQ3Evidence(
        paired=True,
        selection_valid=True,
        all_random_draws_complete=True,
        all_five_arms_feasible=True,
        common_support=True,
        common_support_units=5,
        min_forget_margin=0.1,
        min_utility_margin=0.1,
        damage={
            comparator: {outcome: _reduction() for outcome in DAMAGE_OUTCOMES}
            for comparator in COMPARATORS
        },
        native_noninferiority=adverse_native,
    )
    decision = decide_rq3(evidence)
    assert decision.eligible
    assert not decision.statistical_pass
    assert not decision.claim_pass
