"""Fail-closed local executor for the July-24 PDF v4 protection path.

This module deliberately does not translate legacy Stage-2 knobs.  It builds
the score-independent streams from a frozen manifest, profiles ``theta0``,
allocates the exact Top-Kp protection pool, runs an unchanged parent to its
first saved reaching checkpoint, and only then invokes Equation (7)--(8).

The local executor is a diagnostic runner.  It does not manufacture the
cross-request calibration, comparator arms, native metrics, or confidence
bounds required for a paper claim.
"""
from __future__ import annotations

import dataclasses as dc
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml

from rsus.analysis.mixture import channel_mixture_scores
from rsus.analysis.prediction import cvar_upper
from rsus.blocks import BlockSpec, save_params
from rsus.data.base import Example, Request
from rsus.data.registry import get_adapter
from rsus.generators import TrajectoryConfig, objective_names, run_trajectory
from rsus.generators.repaired import PDFRepairedConfig, run_pdf_repair_from_reached
from rsus.partition import PartitionParams, build_pdf_protection_partition
from rsus.probe import ProbeSpec, ScoreProfile, get_scorer
from rsus.repair import RepairConfig


class LocalPDFV4Error(ValueError):
    """The local v4 execution contract is missing or inconsistent."""


def load_yaml_mapping(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise LocalPDFV4Error(f"{source} must contain one YAML mapping")
    return value


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LocalPDFV4Error(f"{name} must be a mapping")
    return value


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise LocalPDFV4Error(f"{name} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise LocalPDFV4Error(f"{name} must be an integer") from error
    floor = 0 if allow_zero else 1
    if parsed < floor or parsed != value:
        qualifier = "non-negative" if allow_zero else "positive"
        raise LocalPDFV4Error(f"{name} must be a {qualifier} integer")
    return parsed


def _placeholder_paths(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if value is None:
        return [prefix or "<root>"]
    if isinstance(value, str) and value.strip().upper() in {"TBD", "TODO", "CHANGEME"}:
        return [prefix or "<root>"]
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(_placeholder_paths(child, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            paths.extend(_placeholder_paths(child, f"{prefix}[{index}]"))
    return paths


def validate_run_config(config: Mapping[str, Any]) -> None:
    """Validate fields that must be resolved before any model scoring."""
    placeholders = _placeholder_paths(config)
    if placeholders:
        raise LocalPDFV4Error(
            "run config still contains null/TBD placeholders: " + ", ".join(placeholders[:12])
        )
    if config.get("schema_version") != 1:
        raise LocalPDFV4Error("schema_version must be 1")
    if config.get("contract") != "kdd-unlearning-fail-pdf-v4-local":
        raise LocalPDFV4Error("contract must be kdd-unlearning-fail-pdf-v4-local")
    if config.get("status") != "frozen_for_local_diagnostic":
        raise LocalPDFV4Error("status must be frozen_for_local_diagnostic")
    if config.get("claim_eligible") is not False:
        raise LocalPDFV4Error("local runner must keep claim_eligible: false")

    model = _mapping(config.get("model"), "model")
    if model.get("dtype") != "float32":
        raise LocalPDFV4Error("PDF-v4 confirmatory local runner requires model.dtype: float32")
    for key in ("source", "source_revision_or_sha256", "tokenizer_source"):
        if not isinstance(model.get(key), str) or not model[key].strip():
            raise LocalPDFV4Error(f"model.{key} must be a non-empty frozen string")
    block = _mapping(model.get("block"), "model.block")
    if not isinstance(block.get("parameter_regex"), str) or not block["parameter_regex"]:
        raise LocalPDFV4Error("model.block.parameter_regex must be non-empty")
    if not isinstance(block.get("parameter_names_sha256"), str) or len(block["parameter_names_sha256"]) != 64:
        raise LocalPDFV4Error("model.block.parameter_names_sha256 must be a SHA-256 hex digest")
    _positive_int(block.get("parameter_count_d_B"), "model.block.parameter_count_d_B")

    data = _mapping(config.get("data"), "data")
    if not isinstance(data.get("adapter"), str):
        raise LocalPDFV4Error("data.adapter must name a registered adapter")
    _mapping(data.get("request"), "data.request")
    manifest = _mapping(data.get("manifest"), "data.manifest")
    if not isinstance(manifest.get("path"), str) or not manifest["path"]:
        raise LocalPDFV4Error("data.manifest.path must be non-empty")

    probe = _mapping(config.get("probe"), "probe")
    if probe.get("gradient_scorer") != "fd_norm":
        raise LocalPDFV4Error("probe.gradient_scorer must be fd_norm for PDF loss-shake energy")
    if probe.get("proximity_scorer") != "knn_feature":
        raise LocalPDFV4Error("probe.proximity_scorer must be knn_feature for the native hidden channel")
    _positive_int(probe.get("R"), "probe.R")
    _positive_int(probe.get("k"), "probe.k")
    for key in ("eta", "norm_eta"):
        value = float(probe.get(key))
        if not math.isfinite(value) or value <= 0:
            raise LocalPDFV4Error(f"probe.{key} must be finite and positive")
    alpha = float(probe.get("alpha_prot"))
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise LocalPDFV4Error("probe.alpha_prot must be in [0, 1]")
    _positive_int(probe.get("Kp"), "probe.Kp")

    parent = _mapping(config.get("parent"), "parent")
    supported_parents = set(objective_names()) - {"idkdpo"}
    if parent.get("objective") not in supported_parents:
        raise LocalPDFV4Error(
            f"unsupported parent.objective {parent.get('objective')!r}; "
            f"local v4 runner supports: {sorted(supported_parents)}"
        )
    if parent.get("trainable_scope") not in {"full", "block"}:
        raise LocalPDFV4Error("parent.trainable_scope must be full or block")
    recall_max = float(parent.get("recall_max"))
    if not math.isfinite(recall_max) or not 0 <= recall_max <= 1:
        raise LocalPDFV4Error("parent.recall_max must be in [0, 1]")
    trajectory = _mapping(parent.get("trajectory"), "parent.trajectory")
    _positive_int(trajectory.get("max_steps"), "parent.trajectory.max_steps")
    checkpoint_every = _positive_int(
        trajectory.get("checkpoint_every"), "parent.trajectory.checkpoint_every"
    )
    if checkpoint_every > int(trajectory["max_steps"]):
        raise LocalPDFV4Error("parent checkpoint_every cannot exceed max_steps")

    repair = _mapping(config.get("repair"), "repair")
    RepairConfig(**repair).validate()
    output = _mapping(config.get("output"), "output")
    if not isinstance(output.get("directory"), str) or not output["directory"]:
        raise LocalPDFV4Error("output.directory must be non-empty")


def _content_sha(payload: Mapping[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "content_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_manifest(request: Request, split: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze exact score-independent streams before profiling.

    Counts are at semantic-group granularity.  Neutral and utility groups stay
    in the discovery fold but are excluded from repair eligibility.
    """
    seed = int(split.get("seed"))
    n_audit = _positive_int(split.get("audit_groups"), "data.split.audit_groups")
    n_neutral = _positive_int(split.get("neutral_groups"), "data.split.neutral_groups")
    n_utility = _positive_int(split.get("utility_groups"), "data.split.utility_groups")
    examples = list(request.universe.examples)
    universe_ids = {example.example_id for example in examples}
    eligibility = _mapping(split.get("eligibility", {}), "data.split.eligibility")
    eligibility_status = str(
        eligibility.get("status", "provisional_local_diagnostic")
    )
    if eligibility_status not in {
        "provisional_local_diagnostic",
        "frozen_semantic_review",
    }:
        raise LocalPDFV4Error(
            "data.split.eligibility.status must be provisional_local_diagnostic "
            "or frozen_semantic_review"
        )
    ineligible_raw = eligibility.get("ineligible_ids", [])
    if not isinstance(ineligible_raw, list):
        raise LocalPDFV4Error("data.split.eligibility.ineligible_ids must be a list")
    ineligible_ids = {str(value) for value in ineligible_raw}
    if len(ineligible_ids) != len(ineligible_raw):
        raise LocalPDFV4Error("data.split.eligibility.ineligible_ids contains duplicates")
    unknown_ineligible = ineligible_ids - universe_ids
    if unknown_ineligible:
        raise LocalPDFV4Error(
            "semantic ineligibility references unknown IDs: "
            f"{sorted(unknown_ineligible)[:5]}"
        )
    review_sha = str(eligibility.get("review_sha256", "provisional_local_diagnostic"))
    if eligibility_status == "frozen_semantic_review" and len(review_sha) != 64:
        raise LocalPDFV4Error(
            "frozen semantic eligibility requires a 64-character review_sha256"
        )
    by_group: dict[str, list[str]] = {}
    for example in examples:
        if not example.group:
            raise LocalPDFV4Error(f"candidate {example.example_id} has no semantic group")
        by_group.setdefault(example.group, []).append(example.example_id)
    groups = sorted(by_group)
    required = n_audit + n_neutral + n_utility + 1
    if len(groups) < required:
        raise LocalPDFV4Error(
            f"need at least {required} groups for audit/neutral/utility/eligible streams; "
            f"request has {len(groups)}"
        )
    generator = torch.Generator().manual_seed(seed)
    order = [groups[index] for index in torch.randperm(len(groups), generator=generator).tolist()]
    audit_groups = set(order[:n_audit])
    neutral_groups = set(order[n_audit : n_audit + n_neutral])
    utility_groups = set(
        order[n_audit + n_neutral : n_audit + n_neutral + n_utility]
    )
    discovery_groups = set(groups) - audit_groups

    def ids_for(selected_groups: set[str]) -> list[str]:
        return sorted(
            example_id
            for group in selected_groups
            for example_id in by_group[group]
        )

    audit_ids = ids_for(audit_groups)
    discovery_ids = ids_for(discovery_groups)
    neutral_ids = ids_for(neutral_groups)
    utility_ids = ids_for(utility_groups)
    excluded = set(neutral_ids) | set(utility_ids)
    eligible_ids = sorted(set(discovery_ids) - excluded - ineligible_ids)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": "pdf-v4-score-independent-local-streams",
        "status": "frozen_before_scoring",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": request.request_id,
        "forget_sha256": request.forget_sha,
        "candidate_universe_sha256": request.universe.sha,
        "split_seed": seed,
        "fold_by_group": {
            group: ("audit" if group in audit_groups else "discovery")
            for group in groups
        },
        "discovery_ids": discovery_ids,
        "audit_ids": audit_ids,
        "neutral_ids": neutral_ids,
        "utility_guard_ids": utility_ids,
        "semantic_ineligible_ids": sorted(ineligible_ids),
        "eligibility_status": eligibility_status,
        "eligibility_review_sha256": review_sha,
        "repair_eligible_ids": eligible_ids,
        # Parent input is fixed before scores and excludes the audit fold.  It
        # intentionally includes neutral/utility examples; those roles become
        # disjoint only inside the repair operator.
        "parent_retain_ids": discovery_ids,
    }
    payload["content_sha256"] = _content_sha(payload)
    return payload


def validate_manifest(request: Request, manifest: Mapping[str, Any]) -> dict[str, Any]:
    if manifest.get("schema_version") != 1:
        raise LocalPDFV4Error("manifest schema_version must be 1")
    if manifest.get("contract") != "pdf-v4-score-independent-local-streams":
        raise LocalPDFV4Error("wrong local manifest contract")
    if manifest.get("status") != "frozen_before_scoring":
        raise LocalPDFV4Error("manifest status must be frozen_before_scoring")
    expected_sha = _content_sha(manifest)
    if manifest.get("content_sha256") != expected_sha:
        raise LocalPDFV4Error("manifest content_sha256 does not match its content")
    if manifest.get("request_id") != request.request_id:
        raise LocalPDFV4Error("manifest request_id does not match the loaded request")
    if manifest.get("forget_sha256") != request.forget_sha:
        raise LocalPDFV4Error("manifest forget hash does not match the loaded request")
    if manifest.get("candidate_universe_sha256") != request.universe.sha:
        raise LocalPDFV4Error("manifest candidate-universe hash does not match")

    universe = {example.example_id for example in request.universe.examples}
    fields = (
        "discovery_ids",
        "audit_ids",
        "neutral_ids",
        "utility_guard_ids",
        "semantic_ineligible_ids",
        "repair_eligible_ids",
        "parent_retain_ids",
    )
    sets: dict[str, set[str]] = {}
    for field in fields:
        raw = manifest.get(field)
        if not isinstance(raw, list) or (not raw and field != "semantic_ineligible_ids"):
            qualifier = "a list" if field == "semantic_ineligible_ids" else "a non-empty list"
            raise LocalPDFV4Error(f"manifest {field} must be {qualifier}")
        values = [str(value) for value in raw]
        if len(values) != len(set(values)):
            raise LocalPDFV4Error(f"manifest {field} contains duplicate IDs")
        unknown = set(values) - universe
        if unknown:
            raise LocalPDFV4Error(f"manifest {field} has unknown IDs: {sorted(unknown)[:5]}")
        sets[field] = set(values)

    discovery, audit = sets["discovery_ids"], sets["audit_ids"]
    if discovery & audit or discovery | audit != universe:
        raise LocalPDFV4Error("discovery_ids and audit_ids must exactly partition the universe")
    neutral, utility = sets["neutral_ids"], sets["utility_guard_ids"]
    if neutral & utility:
        raise LocalPDFV4Error("neutral and utility guard streams overlap")
    if not neutral <= discovery or not utility <= discovery:
        raise LocalPDFV4Error("neutral and utility guards must be in discovery")
    ineligible = sets["semantic_ineligible_ids"]
    eligible = sets["repair_eligible_ids"]
    expected_eligible = discovery - neutral - utility - ineligible
    if eligible != expected_eligible:
        raise LocalPDFV4Error(
            "repair eligibility must exactly equal discovery minus neutral, utility, "
            "and semantic-ineligible IDs"
        )
    eligibility_status = manifest.get("eligibility_status")
    if eligibility_status not in {
        "provisional_local_diagnostic",
        "frozen_semantic_review",
    }:
        raise LocalPDFV4Error("manifest eligibility_status is invalid")
    review_sha = manifest.get("eligibility_review_sha256")
    if eligibility_status == "frozen_semantic_review" and (
        not isinstance(review_sha, str) or len(review_sha) != 64
    ):
        raise LocalPDFV4Error("frozen semantic eligibility review hash is invalid")
    if sets["parent_retain_ids"] != discovery:
        raise LocalPDFV4Error("parent_retain_ids must equal the score-independent discovery fold")

    group_of = {example.example_id: example.group for example in request.universe.examples}
    fold_by_group = _mapping(manifest.get("fold_by_group"), "manifest.fold_by_group")
    for example_id in discovery:
        if fold_by_group.get(group_of[example_id]) != "discovery":
            raise LocalPDFV4Error("discovery IDs disagree with fold_by_group")
    for example_id in audit:
        if fold_by_group.get(group_of[example_id]) != "audit":
            raise LocalPDFV4Error("audit IDs disagree with fold_by_group")
    return {**manifest, "_sets": sets}


def block_identity(model: torch.nn.Module, block: BlockSpec) -> dict[str, Any]:
    selected = block.select(model)
    names = sorted(selected)
    digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
    return {
        "parameter_regex": block.pattern,
        "parameter_names": names,
        "parameter_names_sha256": digest,
        "parameter_count_d_B": sum(parameter.numel() for parameter in selected.values()),
        "devices": sorted({str(parameter.device) for parameter in selected.values()}),
        "dtypes": sorted({str(parameter.dtype) for parameter in selected.values()}),
    }


def validate_block_identity(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    for key in ("parameter_names_sha256", "parameter_count_d_B"):
        if expected.get(key) != actual.get(key):
            raise LocalPDFV4Error(
                f"model block {key} mismatch: expected {expected.get(key)!r}, "
                f"loaded {actual.get(key)!r}"
            )
    if actual["dtypes"] != ["torch.float32"]:
        raise LocalPDFV4Error(f"selected block must be fp32, found {actual['dtypes']}")
    if len(actual["devices"]) != 1:
        raise LocalPDFV4Error(
            "Equation (8) block must reside on one device in the current executor; "
            f"found {actual['devices']}"
        )


def _examples(request: Request, ids: Sequence[str]) -> list[Example]:
    by_id = {example.example_id: example for example in request.universe.examples}
    return [by_id[example_id] for example_id in ids]


def _jsonable(value: Any) -> Any:
    if dc.is_dataclass(value):
        return _jsonable(dc.asdict(value))
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    return value


def trajectory_payload(record) -> dict[str, Any]:
    return {
        "objective": record.objective,
        "request_id": record.request_id,
        "nll0": record.nll0,
        "cost": _jsonable(record.cost),
        "metadata": _jsonable(record.metadata),
        "snapshots": [
            {
                "step": snapshot.step,
                "forget_recall": snapshot.forget_recall,
                "nll": snapshot.nll,
                "extra": _jsonable(snapshot.extra),
            }
            for snapshot in record.snapshots
        ],
    }


def run_local_pdf_v4(
    config: Mapping[str, Any],
    request: Request,
    manifest: Mapping[str, Any],
    model: torch.nn.Module,
    *,
    log=print,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    """Execute one frozen local diagnostic; return JSON payload and final block."""
    validate_run_config(config)
    checked = validate_manifest(request, manifest)
    model_cfg = _mapping(config["model"], "model")
    block_cfg = _mapping(model_cfg["block"], "model.block")
    block = BlockSpec(str(block_cfg["parameter_regex"]))
    identity = block_identity(model, block)
    validate_block_identity(block_cfg, identity)

    probe_cfg = _mapping(config["probe"], "probe")
    if int(probe_cfg["k"]) > len(request.forget):
        raise LocalPDFV4Error(
            f"probe.k={probe_cfg['k']} exceeds |Df|={len(request.forget)}; clipping is forbidden"
        )
    spec = ProbeSpec(
        block=block,
        eta=float(probe_cfg["eta"]),
        seed=int(probe_cfg["seed"]),
        batch_size=int(probe_cfg["batch_size"]),
        n_dirs=int(probe_cfg["R"]),
        norm_eta=float(probe_cfg["norm_eta"]),
        representation_k=int(probe_cfg["k"]),
        representation_layer=int(probe_cfg.get("representation_layer", -1)),
        representation_pooling=str(probe_cfg.get("representation_pooling", "answer_mean")),
    )
    log("[1/4] profiling theta0: fd_norm + knn_feature")
    gradient = get_scorer(str(probe_cfg["gradient_scorer"]))(model, request, spec)
    proximity = get_scorer(str(probe_cfg["proximity_scorer"]))(model, request, spec)
    all_ids = [example.example_id for example in request.universe.examples]
    discovery_ids = checked["discovery_ids"]
    mixture = channel_mixture_scores(
        gradient.scores,
        proximity.scores,
        float(probe_cfg["alpha_prot"]),
        candidate_ids=all_ids,
        normalization_ids=discovery_ids,
    )
    profile = ScoreProfile(
        request.request_id,
        f"pdf_v4_mixture_alpha_{float(probe_cfg['alpha_prot']):g}",
        mixture,
        spec,
    )
    folds = dict(checked["fold_by_group"])
    partition = build_pdf_protection_partition(
        profile,
        request,
        folds,
        PartitionParams(
            pool_size=int(probe_cfg["Kp"]),
            min_pool_size=int(probe_cfg["Kp"]),
            seed=int(probe_cfg["seed"]),
        ),
        neutral_ids=checked["neutral_ids"],
        repair_eligible_ids=set(checked["repair_eligible_ids"]),
    )
    log(f"[2/4] sealed exact Top-Kp allocation: Kp={len(partition.protect)}")

    parent_cfg = _mapping(config["parent"], "parent")
    trajectory_values = dict(_mapping(parent_cfg["trajectory"], "parent.trajectory"))
    trajectory_values["trainable_pattern"] = (
        block.pattern if parent_cfg["trainable_scope"] == "block" else None
    )
    trajectory = TrajectoryConfig(**trajectory_values)
    parent = run_trajectory(
        model,
        str(parent_cfg["objective"]),
        request,
        _examples(request, checked["parent_retain_ids"]),
        trajectory,
        stop_at_recall=float(parent_cfg["recall_max"]),
    )
    reached = bool(parent.snapshots) and (
        parent.snapshots[-1].forget_recall <= float(parent_cfg["recall_max"])
    )
    log(
        f"[3/4] parent {parent_cfg['objective']}: "
        f"reached={reached} "
        f"step={parent.snapshots[-1].step if parent.snapshots else 'none'}"
    )

    base_payload: dict[str, Any] = {
        "schema": "pdf-v4-local-diagnostic-v1",
        "claim_eligible": False,
        "request_id": request.request_id,
        "manifest_sha256": checked["content_sha256"],
        "block": identity,
        "profile": {
            "gradient": _jsonable(gradient),
            "proximity": _jsonable(proximity),
            "mixture_scores": mixture,
            "alpha_prot": float(probe_cfg["alpha_prot"]),
        },
        "allocation": {
            "manifest_sha256": partition.manifest_sha,
            "protect_ids": list(partition.protect),
            "neutral_ids": list(checked["neutral_ids"]),
            "utility_guard_ids": list(checked["utility_guard_ids"]),
            "audit_ids": list(checked["audit_ids"]),
            "eligibility_status": checked["eligibility_status"],
            "semantic_ineligible_ids": list(checked["semantic_ineligible_ids"]),
        },
        "parent": trajectory_payload(parent),
    }
    if not reached:
        base_payload["status"] = "parent_not_reached_no_repair"
        return base_payload, None

    log("[4/4] running PDF v4 Equation (7)-(8) constrained repair")
    repaired = run_pdf_repair_from_reached(
        model,
        block,
        request,
        _examples(request, partition.protect),
        _examples(request, checked["neutral_ids"]),
        _examples(request, checked["utility_guard_ids"]),
        str(parent_cfg["objective"]),
        PDFRepairedConfig(
            repair=RepairConfig(**_mapping(config["repair"], "repair")),
            recall_max=float(parent_cfg["recall_max"]),
            batch_size=int(probe_cfg["batch_size"]),
        ),
        parent,
        log=log,
    )
    terminal = repaired.snapshots[-1]
    damage = {
        candidate: terminal.nll[candidate] - repaired.nll0[candidate]
        for candidate in repaired.nll0
    }
    audit_damage = [damage[candidate] for candidate in checked["audit_ids"]]
    utility_damage = [damage[candidate] for candidate in checked["utility_guard_ids"]]
    base_payload.update(
        {
            "status": "completed_local_diagnostic",
            "repair": trajectory_payload(repaired),
            "diagnostics": {
                "terminal_step": terminal.step,
                "terminal_forget_recall": terminal.forget_recall,
                "audit_mean_damage": sum(audit_damage) / len(audit_damage),
                "audit_cvar95_damage": cvar_upper(audit_damage, 0.05),
                "audit_n": len(audit_damage),
                "utility_guard_mean_damage": sum(utility_damage) / len(utility_damage),
            },
        }
    )
    final_block = {
        name: value.detach().cpu()
        for name, value in save_params(block.select(model)).items()
    }
    return base_payload, final_block


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
