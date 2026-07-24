"""July-24 PDF constrained repair (Section 4.4, Equations 7--8).

This module is intentionally separate from :mod:`rsus.stage2`.  ``stage2`` is
the repository's superseded mean-hinge diagnostic; translating its parameter
names would silently change the estimand.  The paper-facing implementation
here instead provides one fail-closed contract:

* protected-example NLL plus selector-independent, entry-anchored neutral KL;
* active gradients of oriented forget/neutral/utility constraints;
* a fixed-ridge damped constraint-gradient filter;
* exact token and example-average hard guards after every tentative update;
* rollback, step-size halving, and same-step retries; and
* a processed-model-token budget measured at the model boundary.

The PDF does not specify how scalar differentiable constraints are formed from
the hard margins.  This implementation takes the conservative literal choice:
every frozen token margin and every example-average margin is a separate
``c_j <= 0``.  The convention is recorded in :class:`RepairConfig` and must be
frozen with a run; it must not be changed after target outcomes are opened.
"""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable, Iterator, Literal, Sequence

import torch

from rsus.blocks import (
    BlockSpec,
    ParamVec,
    grads_of,
    load_params_,
    only_block_grads,
    save_params,
    vec_dot,
)
from rsus.costs import CostRecord, Meter
from rsus.data.base import Example, collate
from rsus.losses import answer_token_log_probs, seq_mean_answer_nll


GuardRole = Literal["forget", "neutral", "utility"]


class RepairContractError(ValueError):
    """A paper-facing repair contract is incomplete or internally invalid."""


class TokenBudgetExhausted(RuntimeError):
    """The next model operation would exceed the frozen repair budget."""


@dataclass(frozen=True)
class RepairConfig:
    """Frozen Equation (7)--(8) and hard-acceptance settings.

    ``token_budget`` has no default on purpose: the supplied PDF leaves its
    value open, and a target run must not invent one.  Token accounting uses
    forward-token equivalents plus backward-token equivalents.  Root-model
    forwards are counted from their actual attention masks, including calls
    made by checkpoint hooks and autoregressive generation.
    """

    step_size: float = 3.0e-5
    beta: float = 1.0
    momentum: float = 0.0
    max_steps: int = 120
    batch_size: int = 8
    m_ref: int = 4
    ridge_lambda: float = 1.0e-4
    kappa_tok: float = 0.0
    kappa_ex: float = 0.0
    epsilon_tok: float = 0.0
    epsilon_ex: float = 0.0
    max_retries: int = 3
    retry_shrink: float = 0.5
    token_budget: int | None = None
    save_every: int = 1
    constraint_reduction: str = "per_token_and_per_example"

    def validate(self) -> None:
        if self.token_budget is None or self.token_budget <= 0:
            raise RepairContractError("token_budget must be a frozen positive integer")
        if self.step_size <= 0:
            raise RepairContractError("step_size must be positive")
        if self.beta < 0:
            raise RepairContractError("beta must be non-negative")
        if not 0.0 <= self.momentum < 1.0:
            raise RepairContractError("momentum must be in [0, 1)")
        if self.max_steps <= 0 or self.batch_size <= 0:
            raise RepairContractError("max_steps and batch_size must be positive")
        if self.m_ref <= 0 or self.save_every <= 0:
            raise RepairContractError("m_ref and save_every must be positive")
        if self.ridge_lambda <= 0:
            raise RepairContractError("ridge_lambda must be strictly positive")
        if min(self.kappa_tok, self.kappa_ex, self.epsilon_tok, self.epsilon_ex) < 0:
            raise RepairContractError("activation margins and hard tolerances must be non-negative")
        if self.max_retries < 0:
            raise RepairContractError("max_retries must be non-negative")
        if not 0.0 < self.retry_shrink < 1.0:
            raise RepairContractError("retry_shrink must be in (0, 1)")
        if self.constraint_reduction != "per_token_and_per_example":
            raise RepairContractError(
                "unsupported constraint_reduction; the frozen v4 convention is "
                "per_token_and_per_example"
            )


@dataclass(frozen=True)
class GuardReference:
    role: GuardRole
    example_ids: tuple[str, ...]
    token_index: tuple[tuple[str, int], ...]
    token_nll: torch.Tensor
    example_nll: torch.Tensor
    token_counts: tuple[int, ...]
    # Only the neutral stream needs full entry distributions for Eq. (7).
    entry_log_probs: tuple[torch.Tensor, ...] = ()
    sha256: str = ""


@dataclass(frozen=True)
class RepairReference:
    forget: GuardReference
    neutral: GuardReference
    utility: GuardReference
    sha256: str


@dataclass(frozen=True)
class MarginReport:
    forget_token_max: float
    forget_example_max: float
    neutral_token_max: float
    neutral_example_max: float
    utility_token_max: float
    utility_example_max: float

    @property
    def feasible(self) -> bool:
        return max(
            self.forget_token_max,
            self.forget_example_max,
            self.neutral_token_max,
            self.neutral_example_max,
            self.utility_token_max,
            self.utility_example_max,
        ) <= 0.0


@dataclass(frozen=True)
class RepairEvent:
    target_step: int
    retry: int
    accepted: bool
    refreshed: bool
    step_size: float
    active_constraints: int
    zero_gradient_constraints: int
    margins: MarginReport | None
    reason: str
    processed_tokens: int


@dataclass
class RepairResult:
    n_accepted: int
    n_rejected: int
    step_size_final: float
    stopped_reason: str
    reference_sha256: str
    events: list[RepairEvent] = field(default_factory=list)
    saved_steps: list[int] = field(default_factory=list)
    cost: CostRecord = field(default_factory=CostRecord)


class _TokenBudget:
    def __init__(self, limit: int, rec: CostRecord):
        self.limit = int(limit)
        self.rec = rec

    @property
    def used(self) -> int:
        return self.rec.tokens_fwd + self.rec.tokens_bwd

    def _reserve(self, amount: int) -> None:
        amount = int(amount)
        if amount < 0:
            raise ValueError("token charge cannot be negative")
        if self.used + amount > self.limit:
            raise TokenBudgetExhausted(
                f"repair token budget exhausted: used={self.used}, "
                f"next={amount}, limit={self.limit}"
            )

    def forward(self, amount: int) -> None:
        self._reserve(amount)
        self.rec.tokens_fwd += int(amount)
        self.rec.fwd_passes += 1

    def backward(self, amount: int) -> None:
        self._reserve(amount)
        self.rec.tokens_bwd += int(amount)
        self.rec.bwd_passes += 1


def _model_call_tokens(args: tuple, kwargs: dict) -> int:
    mask = kwargs.get("attention_mask")
    if torch.is_tensor(mask):
        return int(mask.detach().sum().item())
    ids = kwargs.get("input_ids")
    if ids is None and args:
        ids = args[0]
    if torch.is_tensor(ids):
        return int(ids.numel())
    raise RepairContractError(
        "cannot count a model forward without attention_mask or input_ids"
    )


@contextmanager
def _count_model_tokens(model: torch.nn.Module, budget: _TokenBudget):
    """Count every root-model call, including calls made by snapshot hooks."""

    def hook(_module, args, kwargs):
        budget.forward(_model_call_tokens(args, kwargs))

    handle = model.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def _batches(examples: Sequence[Example], batch_size: int) -> Iterator[list[Example]]:
    for start in range(0, len(examples), batch_size):
        yield list(examples[start : start + batch_size])


def _attention_tokens(examples: Sequence[Example]) -> int:
    return sum(int(example.input_ids.numel()) for example in examples)


def _validate_streams(
    protect: Sequence[Example],
    forget: Sequence[Example],
    neutral: Sequence[Example],
    utility: Sequence[Example],
) -> None:
    streams = {
        "protect": protect,
        "forget": forget,
        "neutral": neutral,
        "utility": utility,
    }
    id_sets: dict[str, set[str]] = {}
    for name, examples in streams.items():
        ids = [example.example_id for example in examples]
        if not ids:
            raise RepairContractError(f"{name} stream must be non-empty")
        if len(ids) != len(set(ids)):
            raise RepairContractError(f"{name} stream contains duplicate IDs")
        if any(example.n_answer_tokens() <= 0 for example in examples):
            raise RepairContractError(f"{name} stream contains an empty answer")
        id_sets[name] = set(ids)
    for left, right in (
        ("protect", "forget"),
        ("protect", "neutral"),
        ("protect", "utility"),
        ("forget", "neutral"),
        ("forget", "utility"),
        ("neutral", "utility"),
    ):
        overlap = id_sets[left] & id_sets[right]
        if overlap:
            raise RepairContractError(
                f"{left}/{right} streams overlap: {sorted(overlap)[:5]}"
            )


def _hash_reference(
    role: GuardRole,
    example_ids: tuple[str, ...],
    token_index: tuple[tuple[str, int], ...],
    token_nll: torch.Tensor,
    example_nll: torch.Tensor,
    entry_log_probs: tuple[torch.Tensor, ...],
) -> str:
    digest = hashlib.sha256()
    digest.update(role.encode())
    digest.update(repr(example_ids).encode())
    digest.update(repr(token_index).encode())
    digest.update(token_nll.cpu().double().numpy().tobytes())
    digest.update(example_nll.cpu().double().numpy().tobytes())
    for value in entry_log_probs:
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.cpu().float().numpy().tobytes())
    return digest.hexdigest()


def _build_guard_reference(
    model: torch.nn.Module,
    role: GuardRole,
    examples: Sequence[Example],
    batch_size: int,
    *,
    distributions: bool,
) -> GuardReference:
    example_ids: list[str] = []
    token_index: list[tuple[str, int]] = []
    token_nll: list[torch.Tensor] = []
    example_nll: list[torch.Tensor] = []
    token_counts: list[int] = []
    entry_log_probs: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in _batches(examples, batch_size):
            logp, nll, index, counts = answer_token_log_probs(model, collate(chunk))
            token_nll.append(nll.detach().cpu().float())
            token_index.extend(index)
            offset = 0
            for example, count in zip(chunk, counts):
                rows = logp[offset : offset + count]
                losses = nll[offset : offset + count]
                example_ids.append(example.example_id)
                token_counts.append(count)
                example_nll.append(losses.mean().detach().cpu().float())
                if distributions:
                    entry_log_probs.append(rows.detach().cpu().float().contiguous())
                offset += count
    token_tensor = torch.cat(token_nll)
    example_tensor = torch.stack(example_nll)
    ids_tuple = tuple(example_ids)
    index_tuple = tuple(token_index)
    logp_tuple = tuple(entry_log_probs)
    sha = _hash_reference(
        role, ids_tuple, index_tuple, token_tensor, example_tensor, logp_tuple
    )
    return GuardReference(
        role,
        ids_tuple,
        index_tuple,
        token_tensor,
        example_tensor,
        tuple(token_counts),
        logp_tuple,
        sha,
    )


def build_repair_reference(
    model: torch.nn.Module,
    forget: Sequence[Example],
    neutral: Sequence[Example],
    utility: Sequence[Example],
    batch_size: int,
) -> RepairReference:
    """Seal all entry-state references used by Eq. (7) and hard guards."""
    forget_ref = _build_guard_reference(
        model, "forget", forget, batch_size, distributions=False
    )
    neutral_ref = _build_guard_reference(
        model, "neutral", neutral, batch_size, distributions=True
    )
    utility_ref = _build_guard_reference(
        model, "utility", utility, batch_size, distributions=False
    )
    digest = hashlib.sha256()
    for value in (forget_ref.sha256, neutral_ref.sha256, utility_ref.sha256):
        digest.update(value.encode())
    return RepairReference(forget_ref, neutral_ref, utility_ref, digest.hexdigest())


def _current_losses(
    model: torch.nn.Module,
    examples: Sequence[Example],
    batch_size: int,
) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...], torch.Tensor, torch.Tensor]:
    ids: list[str] = []
    index: list[tuple[str, int]] = []
    tokens: list[torch.Tensor] = []
    sequences: list[torch.Tensor] = []
    for chunk in _batches(examples, batch_size):
        _logp, nll, chunk_index, counts = answer_token_log_probs(model, collate(chunk))
        tokens.append(nll)
        index.extend(chunk_index)
        offset = 0
        for example, count in zip(chunk, counts):
            ids.append(example.example_id)
            sequences.append(nll[offset : offset + count].mean())
            offset += count
    return tuple(ids), tuple(index), torch.cat(tokens), torch.stack(sequences)


def _constraint_values(
    model: torch.nn.Module,
    examples: Sequence[Example],
    reference: GuardReference,
    cfg: RepairConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids, index, token_nll, example_nll = _current_losses(
        model, examples, cfg.batch_size
    )
    if ids != reference.example_ids or index != reference.token_index:
        raise RepairContractError(f"{reference.role} guard identity map changed")
    token_ref = reference.token_nll.to(token_nll.device)
    example_ref = reference.example_nll.to(example_nll.device)
    if reference.role == "forget":
        token_adverse = token_ref - token_nll
        example_adverse = example_ref - example_nll
    else:
        token_adverse = token_nll - token_ref
        example_adverse = example_nll - example_ref
    return token_adverse - cfg.epsilon_tok, example_adverse - cfg.epsilon_ex


def _slice_reference(
    reference: GuardReference, start: int, stop: int
) -> GuardReference:
    token_start = sum(reference.token_counts[:start])
    token_stop = token_start + sum(reference.token_counts[start:stop])
    return GuardReference(
        role=reference.role,
        example_ids=reference.example_ids[start:stop],
        token_index=reference.token_index[token_start:token_stop],
        token_nll=reference.token_nll[token_start:token_stop],
        example_nll=reference.example_nll[start:stop],
        token_counts=reference.token_counts[start:stop],
        entry_log_probs=reference.entry_log_probs[start:stop]
        if reference.entry_log_probs
        else (),
        sha256=reference.sha256,
    )


def _active_basis(
    model: torch.nn.Module,
    selected: dict[str, torch.nn.Parameter],
    streams,
    cfg: RepairConfig,
    budget: _TokenBudget,
) -> tuple[list[ParamVec], int, int]:
    """Return gradients of every active exact token/example constraint."""
    model.zero_grad(set_to_none=True)
    basis: list[ParamVec] = []
    active_count = zero = 0
    with only_block_grads(model, selected):
        for examples, reference in streams:
            for start in range(0, len(examples), cfg.batch_size):
                stop = min(len(examples), start + cfg.batch_size)
                chunk = list(examples[start:stop])
                chunk_ref = _slice_reference(reference, start, stop)
                token_values, example_values = _constraint_values(
                    model, chunk, chunk_ref, cfg
                )
                chunk_tokens = _attention_tokens(chunk)
                active = [
                    value
                    for value in token_values.unbind()
                    if float(value.detach()) >= -cfg.kappa_tok
                ] + [
                    value
                    for value in example_values.unbind()
                    if float(value.detach()) >= -cfg.kappa_ex
                ]
                active_count += len(active)
                for position, value in enumerate(active):
                    model.zero_grad(set_to_none=True)
                    budget.backward(chunk_tokens)
                    value.backward(retain_graph=position + 1 < len(active))
                    gradient = grads_of(selected)
                    norm_sq = float(vec_dot(gradient, gradient))
                    if norm_sq > 0.0:
                        basis.append(gradient)
                    else:
                        zero += 1
    model.zero_grad(set_to_none=True)
    return basis, active_count, zero


def _neutral_kl_batch(
    model: torch.nn.Module,
    chunk: Sequence[Example],
    reference: GuardReference,
    offset_examples: int,
) -> torch.Tensor:
    current, _nll, index, counts = answer_token_log_probs(model, collate(list(chunk)))
    expected_index: list[tuple[str, int]] = []
    for example, count in zip(chunk, counts):
        expected_index.extend((example.example_id, position) for position in range(count))
    if tuple(index) != tuple(expected_index):
        raise RepairContractError("neutral KL token identity changed")

    per_example: list[torch.Tensor] = []
    offset_tokens = 0
    for local, count in enumerate(counts):
        entry = reference.entry_log_probs[offset_examples + local].to(current.device)
        rows = current[offset_tokens : offset_tokens + count]
        if entry.shape != rows.shape:
            raise RepairContractError("neutral KL vocabulary/token shape changed")
        probabilities = entry.exp()
        per_token = (probabilities * (entry - rows)).sum(dim=-1)
        per_example.append(per_token.mean())
        offset_tokens += count
    return torch.stack(per_example)


def _repair_gradient(
    model: torch.nn.Module,
    selected: dict[str, torch.nn.Parameter],
    protect: Sequence[Example],
    neutral: Sequence[Example],
    neutral_ref: GuardReference,
    cfg: RepairConfig,
    budget: _TokenBudget,
) -> ParamVec:
    model.zero_grad(set_to_none=True)
    with only_block_grads(model, selected):
        for chunk in _batches(protect, cfg.batch_size):
            loss = seq_mean_answer_nll(model, collate(chunk)).mean()
            loss = loss * (len(chunk) / len(protect))
            budget.backward(_attention_tokens(chunk))
            loss.backward()

        offset = 0
        for chunk in _batches(neutral, cfg.batch_size):
            kl = _neutral_kl_batch(model, chunk, neutral_ref, offset).mean()
            weighted = cfg.beta * kl * (len(chunk) / len(neutral))
            budget.backward(_attention_tokens(chunk))
            weighted.backward()
            offset += len(chunk)
    gradient = grads_of(selected)
    model.zero_grad(set_to_none=True)
    return gradient


def damped_constraint_filter(
    gradient: ParamVec, basis: Sequence[ParamVec], ridge_lambda: float
) -> ParamVec:
    """Apply Eq. (8) with a fixed absolute ridge ``lambda``."""
    if ridge_lambda <= 0:
        raise RepairContractError("ridge_lambda must be strictly positive")
    if not basis:
        return {name: value.clone() for name, value in gradient.items()}
    size = len(basis)
    gram = torch.empty((size, size), dtype=torch.float64)
    rhs = torch.empty(size, dtype=torch.float64)
    for i, left in enumerate(basis):
        rhs[i] = float(vec_dot(left, gradient))
        for j, right in enumerate(basis):
            gram[i, j] = float(vec_dot(left, right))
    coefficients = torch.linalg.solve(
        gram + ridge_lambda * torch.eye(size, dtype=torch.float64), rhs
    )
    filtered = {name: value.clone() for name, value in gradient.items()}
    for coefficient, constraint_gradient in zip(coefficients.tolist(), basis):
        filtered = {
            name: filtered[name] - coefficient * constraint_gradient[name]
            for name in filtered
        }
    return filtered


def _margin_report(
    model: torch.nn.Module,
    streams,
    cfg: RepairConfig,
) -> MarginReport:
    maxima: list[float] = []
    for examples, reference in streams:
        token_max = example_max = float("-inf")
        for start in range(0, len(examples), cfg.batch_size):
            stop = min(len(examples), start + cfg.batch_size)
            chunk = list(examples[start:stop])
            chunk_ref = _slice_reference(reference, start, stop)
            tokens, sequences = _constraint_values(model, chunk, chunk_ref, cfg)
            token_max = max(token_max, float(tokens.max().detach()))
            example_max = max(example_max, float(sequences.max().detach()))
        maxima.extend((token_max, example_max))
    return MarginReport(*maxima)


SnapshotHook = Callable[[int], None]


def run_repair(
    model: torch.nn.Module,
    block: BlockSpec,
    *,
    protect: Sequence[Example],
    forget_guard: Sequence[Example],
    neutral: Sequence[Example],
    utility_guard: Sequence[Example],
    cfg: RepairConfig,
    snapshot_hook: SnapshotHook | None = None,
) -> RepairResult:
    """Run the PDF repair operator from the model's current entry state.

    ``snapshot_hook`` is called only after accepted updates on the frozen save
    schedule.  Model calls made by the hook are included automatically in
    ``B_tok``.  The hook must not catch :class:`TokenBudgetExhausted`.
    """
    cfg.validate()
    _validate_streams(protect, forget_guard, neutral, utility_guard)
    rec = CostRecord()
    budget = _TokenBudget(int(cfg.token_budget), rec)
    events: list[RepairEvent] = []
    saved_steps: list[int] = []
    n_accepted = n_rejected = 0
    step_size = cfg.step_size
    stopped_reason = "max_steps"
    reference: RepairReference | None = None

    with Meter(rec), _count_model_tokens(model, budget):
        selected = block.select(model)
        velocity: ParamVec = {name: torch.zeros_like(param) for name, param in selected.items()}
        streams = None
        basis: list[ParamVec] = []
        active_count = zero_count = 0
        try:
            reference = build_repair_reference(
                model, forget_guard, neutral, utility_guard, cfg.batch_size
            )
            streams = (
                (forget_guard, reference.forget),
                (neutral, reference.neutral),
                (utility_guard, reference.utility),
            )
            # Entry must be exactly feasible relative to its own references.
            entry_margins = _margin_report(model, streams, cfg)
            if not entry_margins.feasible:
                raise RepairContractError("entry reference is not hard-guard feasible")

            while n_accepted < cfg.max_steps:
                refreshed = n_accepted % cfg.m_ref == 0
                if refreshed:
                    basis, active_count, zero_count = _active_basis(
                        model, selected, streams, cfg, budget
                    )

                gradient = _repair_gradient(
                    model,
                    selected,
                    protect,
                    neutral,
                    reference.neutral,
                    cfg,
                    budget,
                )
                candidate_velocity = {
                    name: cfg.momentum * velocity[name] + gradient[name]
                    for name in velocity
                }
                direction = damped_constraint_filter(
                    candidate_velocity, basis, cfg.ridge_lambda
                )
                before = save_params(selected)
                target_step = n_accepted + 1
                accepted = False

                for retry in range(cfg.max_retries + 1):
                    load_params_(selected, before)
                    with torch.no_grad():
                        for name, parameter in selected.items():
                            parameter.add_(direction[name], alpha=-step_size)
                    try:
                        margins = _margin_report(model, streams, cfg)
                    except TokenBudgetExhausted:
                        load_params_(selected, before)
                        raise
                    if margins.feasible:
                        n_accepted += 1
                        velocity = {name: value.clone() for name, value in direction.items()}
                        events.append(
                            RepairEvent(
                                target_step,
                                retry,
                                True,
                                refreshed,
                                step_size,
                                active_count,
                                zero_count,
                                margins,
                                "accepted",
                                budget.used,
                            )
                        )
                        accepted = True
                        if n_accepted % cfg.save_every == 0 or n_accepted == cfg.max_steps:
                            if snapshot_hook is not None:
                                snapshot_hook(n_accepted)
                            saved_steps.append(n_accepted)
                        break

                    load_params_(selected, before)
                    n_rejected += 1
                    events.append(
                        RepairEvent(
                            target_step,
                            retry,
                            False,
                            refreshed,
                            step_size,
                            active_count,
                            zero_count,
                            margins,
                            "hard_guard_rejected",
                            budget.used,
                        )
                    )
                    step_size *= cfg.retry_shrink

                if not accepted:
                    stopped_reason = "retry_exhausted"
                    break
        except TokenBudgetExhausted:
            stopped_reason = "token_budget_exhausted"
            model.zero_grad(set_to_none=True)

    if reference is None:
        # No target run may silently continue without a sealed entry anchor.
        raise RepairContractError("token budget cannot even build the entry reference")
    rec.notes.update(
        {
            "token_budget": int(cfg.token_budget),
            "processed_tokens": budget.used,
            "accounting": "forward_tokens_plus_backward_token_equivalents",
            "constraint_reduction": cfg.constraint_reduction,
        }
    )
    return RepairResult(
        n_accepted,
        n_rejected,
        step_size,
        stopped_reason,
        reference.sha256,
        events,
        saved_steps,
        rec,
    )
