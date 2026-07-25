"""July-24 PDF repair invariants (Equations 7--8 and Appendix D.1)."""
from __future__ import annotations

import pytest
import torch

from conftest import build_tiny
from rsus.blocks import mlp_down_last_layers, save_params
from rsus.data.substrate import make_substrate
from rsus.repair import (
    RepairConfig,
    RepairContractError,
    build_repair_reference,
    damped_constraint_filter,
    run_repair,
)


def _streams(seed: int = 91):
    request, truth = make_substrate(seed=seed)
    adjacent = [
        example for example in request.universe.examples
        if truth[example.example_id] == "adjacent"
    ]
    remote = [
        example for example in request.universe.examples
        if truth[example.example_id] == "remote"
    ]
    return request, adjacent[:4], remote[:4], remote[4:8]


def test_damped_filter_matches_closed_form():
    gradient = {"w": torch.tensor([2.0, 3.0], dtype=torch.float64)}
    basis = [{"w": torch.tensor([1.0, 0.0], dtype=torch.float64)}]
    output = damped_constraint_filter(gradient, basis, ridge_lambda=1.0)
    # [I - e1 (e1^T e1 + 1)^-1 e1^T] [2, 3] = [1, 3]
    assert torch.equal(output["w"], torch.tensor([1.0, 3.0], dtype=torch.float64))


def test_reference_seals_full_neutral_distributions():
    model = build_tiny(3).eval()
    request, _protect, neutral, utility = _streams()
    reference = build_repair_reference(
        model, request.forget, neutral, utility, batch_size=2
    )
    assert reference.sha256
    assert len(reference.neutral.entry_log_probs) == len(neutral)
    assert all(rows.ndim == 2 for rows in reference.neutral.entry_log_probs)
    assert not reference.forget.entry_log_probs
    assert not reference.utility.entry_log_probs


def test_repair_requires_frozen_token_budget():
    model = build_tiny(5).eval()
    request, protect, neutral, utility = _streams()
    with pytest.raises(RepairContractError, match="token_budget"):
        run_repair(
            model,
            mlp_down_last_layers(model, 1),
            protect=protect,
            forget_guard=request.forget,
            neutral=neutral,
            utility_guard=utility,
            cfg=RepairConfig(token_budget=None),
        )


def test_repair_accepts_only_hard_feasible_updates_and_counts_tokens():
    model = build_tiny(7).eval()
    request, protect, neutral, utility = _streams()
    block = mlp_down_last_layers(model, 1)
    before = save_params(block.select(model))
    result = run_repair(
        model,
        block,
        protect=protect,
        forget_guard=request.forget,
        neutral=neutral,
        utility_guard=utility,
        cfg=RepairConfig(
            step_size=1e-5,
            beta=0.5,
            max_steps=2,
            batch_size=4,
            m_ref=1,
            ridge_lambda=1e-3,
            # Loose guards make the smoke test about mechanics rather than a
            # tuned scientific operating point.
            epsilon_tok=10.0,
            epsilon_ex=10.0,
            token_budget=2_000_000,
            save_every=1,
        ),
    )
    assert result.n_accepted == 2
    assert result.stopped_reason == "max_steps"
    assert result.saved_steps == [1, 2]
    assert result.cost.tokens_fwd > 0 and result.cost.tokens_bwd > 0
    assert result.cost.tokens_fwd + result.cost.tokens_bwd <= 2_000_000
    assert all(event.margins is None or event.margins.feasible for event in result.events if event.accepted)
    after = save_params(block.select(model))
    assert any(not torch.equal(before[name], after[name]) for name in before)


def test_external_feasibility_rejects_and_rolls_back_tentative_update():
    model = build_tiny(11).eval()
    request, protect, neutral, utility = _streams()
    block = mlp_down_last_layers(model, 1)
    before = save_params(block.select(model))
    calls = 0

    def reject(_model):
        nonlocal calls
        calls += 1
        return False

    result = run_repair(
        model,
        block,
        protect=protect,
        forget_guard=request.forget,
        neutral=neutral,
        utility_guard=utility,
        cfg=RepairConfig(
            step_size=1e-5,
            max_steps=1,
            batch_size=4,
            m_ref=1,
            ridge_lambda=1e-3,
            epsilon_tok=10.0,
            epsilon_ex=10.0,
            max_retries=1,
            token_budget=2_000_000,
            save_every=1,
        ),
        external_feasibility=reject,
    )
    assert calls == 2
    assert result.n_accepted == 0
    assert result.n_rejected == 2
    assert result.stopped_reason == "retry_exhausted"
    assert {event.reason for event in result.events} == {
        "external_feasibility_rejected"
    }
    after = save_params(block.select(model))
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_retry_exhaustion_saves_last_accepted_state():
    model = build_tiny(13).eval()
    request, protect, neutral, utility = _streams()
    accepted_checks = 0
    snapshots = []

    def accept_once(_model):
        nonlocal accepted_checks
        accepted_checks += 1
        return accepted_checks == 1

    result = run_repair(
        model,
        mlp_down_last_layers(model, 1),
        protect=protect,
        forget_guard=request.forget,
        neutral=neutral,
        utility_guard=utility,
        cfg=RepairConfig(
            step_size=1e-5,
            max_steps=2,
            batch_size=4,
            m_ref=1,
            ridge_lambda=1e-3,
            epsilon_tok=10.0,
            epsilon_ex=10.0,
            max_retries=0,
            token_budget=2_000_000,
            save_every=10,
        ),
        snapshot_hook=snapshots.append,
        external_feasibility=accept_once,
    )

    assert result.n_accepted == 1
    assert result.stopped_reason == "retry_exhausted"
    assert result.saved_steps == [1]
    assert snapshots == [1]
