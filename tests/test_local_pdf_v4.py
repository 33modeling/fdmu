from __future__ import annotations

import copy

import pytest

from rsus.blocks import mlp_down_last_layers
from rsus.data.substrate import make_substrate
from rsus.local_pdf_v4 import (
    LocalPDFV4Error,
    block_identity,
    prepare_manifest,
    run_local_pdf_v4,
    validate_manifest,
    validate_run_config,
)


def test_prepare_manifest_freezes_disjoint_score_independent_streams():
    request, _ = make_substrate(
        seed=7, n_adjacent=6, n_remote=6, n_decoy=4
    )
    manifest = prepare_manifest(
        request,
        {"seed": 19, "audit_groups": 3, "neutral_groups": 2, "utility_groups": 2},
    )
    checked = validate_manifest(request, manifest)
    sets = checked["_sets"]
    assert sets["discovery_ids"].isdisjoint(sets["audit_ids"])
    assert sets["neutral_ids"].isdisjoint(sets["utility_guard_ids"])
    assert sets["repair_eligible_ids"].isdisjoint(sets["neutral_ids"])
    assert sets["repair_eligible_ids"].isdisjoint(sets["utility_guard_ids"])
    assert sets["parent_retain_ids"] == sets["discovery_ids"]


def test_manifest_hash_detects_post_freeze_edit():
    request, _ = make_substrate(seed=4, n_adjacent=5, n_remote=5, n_decoy=3)
    manifest = prepare_manifest(
        request,
        {"seed": 1, "audit_groups": 2, "neutral_groups": 2, "utility_groups": 2},
    )
    manifest["repair_eligible_ids"] = manifest["repair_eligible_ids"][:-1]
    with pytest.raises(LocalPDFV4Error, match="content_sha256"):
        validate_manifest(request, manifest)


def _resolved_config() -> dict:
    return {
        "schema_version": 1,
        "contract": "kdd-unlearning-fail-pdf-v4-local",
        "status": "frozen_for_local_diagnostic",
        "claim_eligible": False,
        "run": {"seed": 1, "deterministic_algorithms": True},
        "model": {
            "source": "model",
            "source_revision_or_sha256": "a" * 64,
            "tokenizer_source": "tokenizer",
            "dtype": "float32",
            "device": "cpu",
            "block": {
                "parameter_regex": ".*weight",
                "parameter_names_sha256": "b" * 64,
                "parameter_count_d_B": 3,
            },
        },
        "data": {
            "adapter": "substrate",
            "request": {"seed": 1},
            "manifest": {"path": "manifest.yaml"},
        },
        "probe": {
            "gradient_scorer": "fd_norm",
            "proximity_scorer": "knn_feature",
            "R": 8,
            "eta": 0.001,
            "norm_eta": 0.003,
            "seed": 0,
            "batch_size": 2,
            "k": 2,
            "alpha_prot": 0.5,
            "Kp": 2,
        },
        "parent": {
            "objective": "npo",
            "recall_max": 0.1,
            "trainable_scope": "block",
            "trajectory": {
                "max_steps": 4,
                "checkpoint_every": 1,
                "batch_size": 2,
                "lr": 0.001,
                "seed": 1,
                "beta": 0.1,
            },
        },
        "repair": {
            "step_size": 0.001,
            "beta": 1.0,
            "momentum": 0.0,
            "max_steps": 2,
            "batch_size": 2,
            "m_ref": 1,
            "ridge_lambda": 0.001,
            "kappa_tok": 0.0,
            "kappa_ex": 0.0,
            "epsilon_tok": 0.01,
            "epsilon_ex": 0.01,
            "max_retries": 1,
            "retry_shrink": 0.5,
            "token_budget": 10000,
            "save_every": 1,
            "constraint_reduction": "per_token_and_per_example",
        },
        "output": {"directory": "runs/test", "save_final_block": True},
    }


def test_run_config_rejects_unresolved_or_legacy_contracts():
    config = _resolved_config()
    config["repair"]["step_size"] = None
    with pytest.raises(LocalPDFV4Error, match="placeholders"):
        validate_run_config(config)

    config = _resolved_config()
    config["repair"]["s2_eta2"] = config["repair"].pop("step_size")
    with pytest.raises(TypeError):
        validate_run_config(config)


def test_run_config_requires_diagnostic_not_claim_label():
    config = copy.deepcopy(_resolved_config())
    config["claim_eligible"] = True
    with pytest.raises(LocalPDFV4Error, match="claim_eligible"):
        validate_run_config(config)


def test_local_runner_connects_profile_parent_and_pdf_repair(tiny_model):
    model = copy.deepcopy(tiny_model).float()
    request, _ = make_substrate(
        seed=8,
        n_forget=3,
        n_adjacent=4,
        n_remote=4,
        n_decoy=0,
        seq_len=10,
        prompt_len=5,
        vocab=128,
    )
    manifest = prepare_manifest(
        request,
        {"seed": 2, "audit_groups": 1, "neutral_groups": 1, "utility_groups": 1},
    )
    config = _resolved_config()
    block = mlp_down_last_layers(model, 1)
    identity = block_identity(model, block)
    config["model"]["block"] = {
        "parameter_regex": block.pattern,
        "parameter_names_sha256": identity["parameter_names_sha256"],
        "parameter_count_d_B": identity["parameter_count_d_B"],
    }
    config["probe"].update({"R": 2, "k": 2, "Kp": 2, "batch_size": 4})
    config["parent"].update({"recall_max": 1.0})
    config["parent"]["trajectory"].update(
        {"max_steps": 1, "checkpoint_every": 1, "batch_size": 4, "lr": 1e-5}
    )
    config["repair"].update(
        {
            "step_size": 1e-5,
            "max_steps": 1,
            "batch_size": 4,
            "m_ref": 1,
            "kappa_tok": 0.0,
            "kappa_ex": 0.0,
            "epsilon_tok": 100.0,
            "epsilon_ex": 100.0,
            "token_budget": 1_000_000,
            "save_every": 1,
        }
    )

    payload, final_block = run_local_pdf_v4(
        config, request, manifest, model, log=lambda _message: None
    )

    assert payload["status"] == "completed_local_diagnostic"
    assert payload["claim_eligible"] is False
    assert payload["repair"]["metadata"]["pdf_v4_repair"]["contract"].endswith(
        "eq7-eq8"
    )
    assert len(payload["allocation"]["protect_ids"]) == 2
    assert payload["allocation"]["eligibility_status"] == "provisional_local_diagnostic"
    assert final_block is not None
