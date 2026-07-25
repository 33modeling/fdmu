from __future__ import annotations

import pytest

from rsus.evidence.schemas import EvidenceLedger, EvidenceValidationError
from rsus.evidence.table1 import render_table1


PARENTS = (
    "graddiff",
    "npo",
    "simnpo",
    "gru",
    "rmu",
    "repnoise",
    "circuit_breakers",
)


def _gain():
    return {
        "estimate": 0.2,
        "lower_bound": 0.1,
        "p_one_sided": 0.01,
    }


def _reduction():
    return {
        "estimate": -0.2,
        "upper_bound": -0.1,
        "p_one_sided": 0.01,
    }


def _row(parent: str):
    return {
        "setting": "tofu",
        "parent": parent,
        "attempted": True,
        "completed": True,
        "prediction_selection": {
            "valid": True,
            "fallback": False,
            "alpha": 0.5,
        },
        "protection_selection": {
            "valid": True,
            "fallback": False,
            "alpha": 0.5,
        },
        "funnel": {
            "profiles_planned": 2,
            "profiles_valid": 2,
            "fidelity_planned": 2,
            "fidelity_valid": 2,
            "trajectories_planned": 2,
            "trajectories_attempted": 2,
            "trajectories_completed": 2,
            "trajectories_reached": 2,
            "reached_with_valid_profile": 2,
            "prediction_common": 2,
            "tail_eligible": 2,
            "fidelity_common": 2,
            "protection_feasible_all_arms": 2,
            "protection_common": 2,
        },
        "rq1": {
            "paired": True,
            "selection_valid": True,
            "profile_valid": True,
            "common_support_units": 2,
            "reached_valid_units": 2,
            "tail_eligible_units": 2,
            "joint_rho": _gain(),
            "joint_minus_s0": _gain(),
            "joint_minus_s1": _gain(),
            "tail_lift": _gain(),
        },
        "rq2": {
            "paired": True,
            "perturbations_valid": True,
            "exact_reference_valid": True,
            "common_control_support": True,
            "common_support_units": 2,
            "f_rho_minus_0p80": _gain(),
            "f_k_minus_0p70": _gain(),
            "g_h": _gain(),
            "g_ctl": _gain(),
        },
        "rq3": {
            "paired": True,
            "comparisons": {
                comparator: {
                    "mean": _reduction(),
                    "cvar95": _reduction(),
                }
                for comparator in ("no_repair", "repeated_random", "s0", "s1")
            },
            "native_noninferiority": {
                comparator: _gain()
                for comparator in ("no_repair", "repeated_random", "s0", "s1")
            },
            "selection_valid": True,
            "all_random_draws_complete": True,
            "all_five_arms_feasible": True,
            "common_support": True,
            "common_support_units": 2,
            "min_forget_margin": 0.05,
            "min_utility_margin": 0.04,
            "profile_mean": 0.2,
            "profile_cvar95": 0.4,
            "no_repair_mean": 0.5,
            "no_repair_cvar95": 0.8,
            "repair_updates": 10,
            "repair_rollbacks": 2,
        },
    }


def _report():
    return {
        "rows": [
            {
                "setting": "tofu",
                "parent": parent,
                "rq1": {"eligible": True, "claim_pass": True},
                "rq2": {"eligible": True, "claim_pass": True},
                "rq3": {"eligible": True, "claim_pass": True},
            }
            for parent in PARENTS
        ]
    }


def test_table1_renders_both_panels_and_all_seven_parents():
    ledger = EvidenceLedger.from_mapping(
        {
            "schema_version": 2,
            "rows": [_row(parent) for parent in PARENTS],
            "artifacts": {},
        }
    )
    rendered = render_table1(ledger, _report(), setting="tofu")
    assert "A. Prospective prediction" in rendered
    assert "B. Constraint-matched" in rendered
    assert "GradDiff" in rendered
    assert "RepNoise" in rendered
    assert "10.0 / 2.0" in rendered
    assert rendered.count("Y/Y") == 21


def test_table1_fails_closed_on_missing_panel_b_summary():
    raw = _row("graddiff")
    raw["rq3"]["profile_mean"] = None
    ledger = EvidenceLedger.from_mapping(
        {"schema_version": 2, "rows": [raw], "artifacts": {}}
    )
    with pytest.raises(EvidenceValidationError, match="incomplete"):
        render_table1(ledger, _report(), setting="tofu")


def test_table1_can_render_explicit_dashes_for_incomplete_rows():
    raw = _row("graddiff")
    raw["rq1"]["joint_rho"] = {
        "estimate": None,
        "lower_bound": None,
        "p_one_sided": None,
    }
    ledger = EvidenceLedger.from_mapping(
        {"schema_version": 2, "rows": [raw], "artifacts": {}}
    )
    rendered = render_table1(
        ledger, _report(), setting="tofu", allow_incomplete=True
    )
    assert "GradDiff & -- & -- & -- & -- & -- & -- & --" in rendered
