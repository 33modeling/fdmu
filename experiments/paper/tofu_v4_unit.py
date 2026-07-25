#!/usr/bin/env python3
"""Execute one claim-bearing TOFU PDF-v4 paper unit.

Heavy ML imports are intentionally local to :func:`run_unit`, allowing the
paper preflight and controller tests to inspect this executable on CPU-only
hosts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


PAPER_UNIT_CONTRACT = {
    "schema_version": 1,
    "adapters": ["tofu"],
    "stages": ["calibration", "prediction", "protection", "target_evaluation"],
    "consumes_campaign_config": True,
    "consumes_frozen_unit_identity": True,
    "uses_adapter_registry": True,
    "executes_model": True,
    "emits_fidelity_raw": True,
    "emits_parent_selection_inputs": True,
    "computes_exact_gradient_reference": True,
    "emits_candidate_level_prediction_raw": True,
    "emits_selection_inputs": True,
    "runs_parent_to_first_reaching_checkpoint": True,
    "emits_candidate_level_protection_raw": True,
    "runs_pdf_v4_repair": True,
    "runs_all_comparator_arms": True,
    "emits_dataset_native_retention": True,
}

PARENTS = (
    "graddiff",
    "npo",
    "simnpo",
    "gru",
    "rmu",
    "repnoise",
    "circuit_breakers",
)
STAGE_ROSTER = {
    "calibration": "D_cal",
    "prediction": "D_pred",
    "protection": "D_prot",
    "target_evaluation": "target",
}
REQUEST_RE = re.compile(r"tofu-a(18[0-9]|19[0-9])\Z")


class TOFUUnitError(ValueError):
    """A unit identity or frozen runtime contract is invalid."""


def _load(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise TOFUUnitError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise TOFUUnitError(f"{path} must contain one mapping")
    return value


def _resolve(value: str | Path, *, base: Path = ROOT) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(value))))
    return path if path.is_absolute() else (base / path).resolve()


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TOFUUnitError(f"{name} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha(value: object) -> str:
    body = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(body.encode()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _unit_contract(
    campaign: Mapping[str, Any],
    evidence: Mapping[str, Any],
    runtime: Mapping[str, Any],
    *,
    stage: str,
    setting_id: str,
    parent: str,
    request_id: str,
    seed: int,
) -> dict[str, Any]:
    if runtime.get("schema_version") != 1 or runtime.get("status") != "executable":
        raise TOFUUnitError("TOFU runtime must have schema_version 1 and status=executable")
    if runtime.get("contract") != "kdd-unlearning-fail-tofu-pdf-v4":
        raise TOFUUnitError("wrong TOFU runtime contract")
    if stage not in STAGE_ROSTER:
        raise TOFUUnitError(f"unsupported stage {stage!r}")
    settings = {
        item.get("id"): item
        for item in evidence.get("settings", [])
        if isinstance(item, Mapping)
    }
    setting = settings.get(setting_id)
    if not isinstance(setting, Mapping) or setting.get("dataset") != "TOFU":
        raise TOFUUnitError(f"{setting_id!r} is not a registered TOFU setting")
    if parent not in setting.get("parents", []) or parent not in PARENTS:
        raise TOFUUnitError(f"parent {parent!r} is outside the frozen seven-parent roster")
    setting_runtime = runtime.get("settings", {}).get(setting_id)
    if not isinstance(setting_runtime, Mapping):
        raise TOFUUnitError(f"runtime has no setting {setting_id!r}")
    if setting_runtime.get("model") != setting.get("model"):
        raise TOFUUnitError("runtime/evidence model mismatch")
    model = campaign.get("models", {}).get(setting.get("model"))
    if not isinstance(model, Mapping) or model.get("provisioned") is not True:
        raise TOFUUnitError(f"model {setting.get('model')!r} is not provisioned")
    if model.get("dtype") != "float32":
        raise TOFUUnitError("claim-bearing loss-shake execution requires float32")
    dataset = campaign.get("datasets", {}).get("TOFU")
    execution = campaign.get("execution")
    if not isinstance(dataset, Mapping) or not isinstance(execution, Mapping):
        raise TOFUUnitError("campaign lacks TOFU/execution mappings")
    roster = dataset.get("rosters", {}).get(STAGE_ROSTER[stage])
    if not isinstance(roster, list) or request_id not in roster:
        raise TOFUUnitError(f"{request_id!r} is outside the exact {stage} roster")
    if seed not in execution.get("seeds", []):
        raise TOFUUnitError(f"seed {seed} is outside the exact campaign seed roster")
    match = REQUEST_RE.fullmatch(request_id)
    if match is None:
        raise TOFUUnitError(f"invalid TOFU request id {request_id!r}")
    return {
        "setting": dict(setting),
        "runtime": dict(setting_runtime),
        "model": dict(model),
        "dataset": dict(dataset),
        "execution": dict(execution),
        "author": int(match.group(1)),
    }


def _candidate_authors(
    runtime: Mapping[str, Any], *, stage: str, author: int, seed: int
) -> list[int]:
    design = _mapping(runtime.get("candidate_design"), "candidate_design")
    ranges = _mapping(design.get("stage_author_ranges"), "stage_author_ranges")
    bounds = ranges.get(stage)
    if (
        not isinstance(bounds, list)
        or len(bounds) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in bounds)
    ):
        raise TOFUUnitError(f"stage_author_ranges.{stage} must be [first, last]")
    pool = list(range(bounds[0], bounds[1] + 1))
    count = int(design.get("authors_per_request"))
    if count < 1 or count > len(pool):
        raise TOFUUnitError("authors_per_request exceeds its stage author range")
    digest = hashlib.sha256(f"{stage}|{author}|{seed}".encode()).digest()
    offset = int.from_bytes(digest[:8], "big") % len(pool)
    ordered = pool[offset:] + pool[:offset]
    selected = ordered[:count]
    if author in selected:
        raise TOFUUnitError("candidate pool contains its forget author")
    return selected


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _semantic_ineligible(request) -> list[str]:
    forget_text = {
        _normalized_text(example.text)
        for example in request.forget
        if example.text
    }
    seen: dict[str, str] = {}
    excluded: set[str] = set()
    for example in sorted(request.universe.examples, key=lambda item: item.example_id):
        normalized = _normalized_text(example.text)
        if not normalized:
            excluded.add(example.example_id)
        elif normalized in forget_text:
            excluded.add(example.example_id)
        elif normalized in seen:
            excluded.add(example.example_id)
        else:
            seen[normalized] = example.example_id
    return sorted(excluded)


def _parent_grid(setting_runtime: Mapping[str, Any], parent: str) -> list[dict[str, Any]]:
    source = _resolve(str(setting_runtime.get("channel_config", "")))
    channel = _load(source)
    raw = channel.get("calibration", {}).get("objective_grid", {}).get(parent)
    if not isinstance(raw, list) or not raw:
        raise TOFUUnitError(f"channel config lacks calibration grid for {parent}")
    values = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise TOFUUnitError(f"{source} objective_grid.{parent}[{index}] is invalid")
        values.append(dict(item))
    return values


def _frozen_parent(
    setting_runtime: Mapping[str, Any],
    *,
    setting_id: str,
    parent: str,
) -> tuple[dict[str, Any], Path]:
    path = _resolve(str(setting_runtime.get("parent_freeze", "")))
    freeze = _load(path)
    if freeze.get("status") != "frozen":
        raise TOFUUnitError(
            f"parent freeze is not frozen for {setting_id}: {path}"
        )
    if freeze.get("contract") == "tofu-pdf-v4-parent-freeze":
        if (
            freeze.get("schema_version") != 1
            or freeze.get("setting") != setting_id
            or freeze.get("frozen_before_prediction") is not True
        ):
            raise TOFUUnitError(f"invalid paper parent freeze: {path}")
        value = freeze.get("parents", {}).get(parent)
    else:
        model_id = str(setting_runtime.get("channel_model_id"))
        if freeze.get("frozen_before_audit") is not True:
            raise TOFUUnitError(f"legacy parent freeze is not frozen: {path}")
        value = freeze.get("models", {}).get(model_id, {}).get(parent)
    if not isinstance(value, Mapping):
        raise TOFUUnitError(f"parent freeze lacks {setting_id}/{parent}")
    required = ("lr", "steps")
    if any(
        isinstance(value.get(field), bool)
        or not isinstance(value.get(field), (int, float))
        for field in required
    ):
        raise TOFUUnitError(f"parent freeze has unresolved {setting_id}/{parent}")
    return dict(value), path


def _selection(
    path: Path,
    *,
    campaign_id: str,
    setting: str,
    parent: str,
) -> dict[str, Any]:
    freeze = _load(path)
    if (
        freeze.get("schema_version") != 1
        or freeze.get("status") != "frozen"
        or freeze.get("source_campaign") != campaign_id
        or freeze.get("frozen_before_target") is not True
    ):
        raise TOFUUnitError(f"selection freeze is not target-ready: {path}")
    value = freeze.get("selections", {}).get(setting, {}).get(parent)
    if not isinstance(value, Mapping):
        raise TOFUUnitError(f"selection freeze lacks {setting}/{parent}")
    prediction = value.get("prediction")
    protection = value.get("protection")
    if not isinstance(prediction, Mapping) or not isinstance(protection, Mapping):
        raise TOFUUnitError(f"selection freeze lacks prediction/protection for {parent}")
    for name, selection in (
        ("prediction", prediction),
        ("protection", protection),
    ):
        valid = selection.get("valid")
        fallback = selection.get("fallback")
        alpha = selection.get("alpha")
        if type(valid) is not bool or type(fallback) is not bool or valid == fallback:
            raise TOFUUnitError(
                f"selection freeze {setting}/{parent}/{name} is unresolved"
            )
        if (
            isinstance(alpha, bool)
            or not isinstance(alpha, (int, float))
            or not math.isfinite(float(alpha))
            or not 0.0 <= float(alpha) <= 1.0
        ):
            raise TOFUUnitError(
                f"selection freeze {setting}/{parent}/{name}.alpha is invalid"
            )
    if (
        not isinstance(prediction.get("simple_control"), str)
        or not prediction["simple_control"].strip()
    ):
        raise TOFUUnitError(
            f"selection freeze {setting}/{parent} lacks simple_control"
        )
    kp = protection.get("Kp")
    if isinstance(kp, bool) or not isinstance(kp, int) or kp < 1:
        raise TOFUUnitError(
            f"selection freeze {setting}/{parent} has invalid Kp"
        )
    return {
        "prediction": dict(prediction),
        "protection": dict(protection),
    }


def _selection_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "valid": value.get("valid") is True,
        "fallback": value.get("fallback") is True,
        "alpha": value.get("alpha"),
    }


def _trajectory_config(values: Mapping[str, Any], runtime: Mapping[str, Any], block, seed: int):
    from rsus.generators import TrajectoryConfig

    return TrajectoryConfig(
        max_steps=int(values["steps"]),
        checkpoint_every=min(
            int(runtime["parent"]["checkpoint_every"]), int(values["steps"])
        ),
        batch_size=int(runtime["runtime"]["batch_size"]),
        lr=float(values["lr"]),
        seed=seed,
        beta=float(values.get("beta", 1.0)),
        simnpo_gamma=float(values.get("simnpo_gamma", 0.0)),
        rmu_alpha=float(values.get("rmu_alpha", 10.0)),
        rmu_c=float(values.get("rmu_c", 3.0)),
        trainable_pattern=block.pattern,
        forget_weight=float(values.get("forget_weight", 1.0)),
        retain_weight=float(values.get("retain_weight", 1.0)),
        representation_retain_mode="stream_cached",
    )


def _checkpoint_id(block_values: Mapping[str, Any], *, request: str, parent: str, step: int) -> str:
    digest = hashlib.sha256(f"{request}|{parent}|{step}".encode())
    for name in sorted(block_values):
        tensor = block_values[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.view(-1).numpy().tobytes())
    return f"first-reaching-{step}-{digest.hexdigest()[:16]}"


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise TOFUUnitError("cannot average an empty sequence")
    return sum(values) / len(values)


def _top_recall(scores: Mapping[str, float], damage: Mapping[str, float], q: float) -> float:
    count = max(1, math.ceil(q * len(scores)))
    score_top = set(sorted(scores, key=lambda key: (-scores[key], key))[:count])
    damage_top = set(sorted(damage, key=lambda key: (-damage[key], key))[:count])
    return len(score_top & damage_top) / count


def _random_scores(ids: Sequence[str], seed_text: str) -> dict[str, float]:
    return {
        candidate: int.from_bytes(
            hashlib.sha256(f"{seed_text}|{candidate}".encode()).digest()[:8],
            "big",
        )
        / float(2**64 - 1)
        for candidate in ids
    }


def run_unit(args: argparse.Namespace) -> None:
    # Heavy dependencies stay below this line.
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(ROOT / "experiments/gate_1p5b"))
    import gate as gate_runtime  # type: ignore  # noqa: E402

    from rsus.analysis.mixture import channel_mixture_scores
    from rsus.analysis.prediction import cvar_upper, spearman, top_k_ids
    from rsus.blocks import load_params_, mlp_down_last_layers, save_params
    from rsus.data.base import CandidateUniverse, Request
    from rsus.data.registry import get_adapter
    from rsus.data.tofu import load_tofu_examples, load_tofu_paraphrases
    from rsus.evalx.metrics import (
        greedy_generation_recall,
        mean_recall,
    )
    from rsus.generators import run_trajectory
    from rsus.generators.repaired import (
        PDFRepairedConfig,
        run_pdf_repair_from_reached,
    )
    from rsus.local_pdf_v4 import (
        block_identity,
        prepare_manifest,
        trajectory_payload,
        validate_manifest,
    )
    from rsus.losses import IGNORE, seq_mean_answer_nll
    from rsus.partition import PartitionParams, build_pdf_protection_partition
    from rsus.probe import ProbeSpec, ScoreProfile, get_scorer
    from rsus.probe.baselines import set_embed_encoder
    from rsus.probe.fidelity import perturbation_report
    from rsus.repair import RepairConfig

    campaign_path = args.campaign.resolve()
    evidence_path = args.evidence.resolve()
    runtime_path = args.runtime.resolve()
    campaign = _load(campaign_path)
    evidence = _load(evidence_path)
    runtime = _load(runtime_path)
    contract = _unit_contract(
        campaign,
        evidence,
        runtime,
        stage=args.stage,
        setting_id=args.setting,
        parent=args.parent,
        request_id=args.request,
        seed=args.seed,
    )
    selected = None
    if args.stage == "target_evaluation":
        if args.selection_freeze is None:
            raise TOFUUnitError("target_evaluation requires --selection-freeze")
        selected = _selection(
            args.selection_freeze.resolve(),
            campaign_id=str(campaign["campaign_id"]),
            setting=args.setting,
            parent=args.parent,
        )
    setting_runtime = contract["runtime"]
    model_cfg = contract["model"]
    design = _mapping(runtime["candidate_design"], "candidate_design")
    common = _mapping(runtime["runtime"], "runtime")
    probe_cfg = _mapping(runtime["probe"], "probe")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "unit.log"

    def log(message: str) -> None:
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(message + "\n")

    model_source = _resolve(str(model_cfg["source"]))
    if not model_source.is_dir():
        raise TOFUUnitError(f"model source is unavailable: {model_source}")
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(model_source))
    examples = load_tofu_examples(tokenizer, max_length=int(common["max_length"]))
    native_metric = _mapping(
        contract["dataset"].get("native_metric"), "TOFU.native_metric"
    )
    if (
        native_metric.get("name") != "retain_answer_token_recall"
        or native_metric.get("orientation") != "higher"
    ):
        raise TOFUUnitError(
            "TOFU producer requires higher-is-better "
            "retain_answer_token_recall"
        )
    candidate_authors = _candidate_authors(
        runtime,
        stage=args.stage,
        author=contract["author"],
        seed=args.seed,
    )
    adapter = get_adapter(str(contract["dataset"]["adapter"]))
    if not adapter.capabilities.supports(args.stage):
        raise TOFUUnitError(
            f"adapter {adapter.key!r} does not support stage {args.stage!r}"
        )
    request = adapter.build_request(
        author_id=contract["author"],
        examples=examples,
        universe_authors=len(candidate_authors),
        seed=args.seed,
        candidate_authors=candidate_authors,
    )
    by_id = {example.example_id: example for example in request.universe.examples}

    eligibility_path = _resolve(str(design["semantic_eligibility"]))
    eligibility = _load(eligibility_path)
    if (
        eligibility.get("status") != "frozen_rule_review"
        or eligibility.get("frozen_before_scoring") is not True
    ):
        raise TOFUUnitError("semantic eligibility rules are not frozen")
    split = dict(_mapping(design["split"], "candidate_design.split"))
    split.update(
        {
            "seed": args.seed + contract["author"] * 100_003,
            "eligibility": {
                "status": "frozen_semantic_review",
                "ineligible_ids": _semantic_ineligible(request),
                "review_sha256": _sha256(eligibility_path),
            },
        }
    )
    stream_path = output_dir / "score_independent_manifest.json"
    if stream_path.is_file():
        stream_manifest = json.loads(stream_path.read_text(encoding="utf-8"))
    else:
        stream_manifest = prepare_manifest(request, split)
        _atomic_json(stream_path, stream_manifest)
    checked = validate_manifest(request, stream_manifest)

    utility_first, utility_last = design["native_utility_authors"]
    utility_authors = set(range(int(utility_first), int(utility_last) + 1))
    if utility_authors & set(candidate_authors) or contract["author"] in utility_authors:
        raise TOFUUnitError(
            "native utility authors must be disjoint from candidate and forget authors"
        )
    per_author = int(design["native_utility_examples_per_author"])
    native_utility = []
    for author in sorted(utility_authors):
        members = sorted(
            (
                example
                for example in examples
                if example.group == f"author-{author:03d}"
            ),
            key=lambda example: example.example_id,
        )
        native_utility.extend(members[:per_author])
    expected_native_utility = len(utility_authors) * per_author
    if len(native_utility) != expected_native_utility:
        raise TOFUUnitError(
            "native utility roster is incomplete: "
            f"expected {expected_native_utility}, found {len(native_utility)}"
        )
    candidate_groups = {
        candidate: by_id[candidate].group for candidate in checked["audit_ids"]
    }

    runtime_args = SimpleNamespace(
        model=str(model_source),
        model_id=str(setting_runtime["channel_model_id"]),
        device=str(common["device"]),
        dtype=str(common["dtype"]),
        attn_impl=str(common.get("attn_impl", "")),
        smoke=False,
        seed=args.seed,
        trainable_scope=str(common["trainable_scope"]),
        sft_lr=float(setting_runtime["sft"]["lr"]),
        sft_steps=int(setting_runtime["sft"]["steps"]),
        sft_target_loss=float(setting_runtime["sft"]["target_loss"]),
        sft_eval_every=int(setting_runtime["sft"]["eval_every"]),
        batch_size=int(common["batch_size"]),
    )
    if runtime_args.trainable_scope != "probe_block":
        raise TOFUUnitError("paper parent and repair must share probe_block scope")

    log("loading/SFT-memorizing theta0")
    model0 = gate_runtime.load_model(runtime_args, tokenizer)
    block = mlp_down_last_layers(model0, int(common["block_last_n"]))
    sft_examples = (
        list(request.forget)
        + list(request.universe.examples)
        + list(native_utility)
    )
    cache_root = _resolve(str(common["sft_cache_root"]))
    cache_path = (
        cache_root
        / args.setting
        / f"{args.request}__seed-{args.seed}.pt"
    )
    cache_contract = gate_runtime._sft_cache_contract(runtime_args, request, block)
    cache_contract["paper_native_utility_ids_sha256"] = _json_sha(
        [example.example_id for example in native_utility]
    )
    sft_result = gate_runtime._load_sft_cache(
        model0, cache_path, cache_contract, log
    )
    if sft_result is None:
        sft_result = gate_runtime.sft(
            model0, sft_examples, runtime_args, log, block
        )
    if not sft_result["reached"]:
        raise TOFUUnitError(
            f"SFT target not reached: {sft_result['full_mean_nll']} > "
            f"{sft_result['target']}"
        )
    state0 = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model0.state_dict().items()
    }
    if not cache_path.exists():
        gate_runtime._write_sft_cache(
            cache_path, cache_contract, sft_result, state0, log
        )

    def fresh():
        model = gate_runtime.load_model(runtime_args, tokenizer)
        model.load_state_dict(state0)
        return model

    identity = block_identity(model0, block)
    native_baseline = mean_recall(
        model0, native_utility, int(common["batch_size"])
    )
    spec = ProbeSpec(
        block=block,
        eta=float(probe_cfg["eta"]),
        seed=int(probe_cfg["seed"]),
        batch_size=int(common["batch_size"]),
        n_dirs=int(probe_cfg["R"]),
        norm_eta=float(probe_cfg["norm_eta"]),
        representation_k=int(probe_cfg["representation_k"]),
        representation_layer=int(probe_cfg["representation_layer"]),
        representation_pooling=str(probe_cfg["representation_pooling"]),
    )
    log("profiling theta0: fd_norm, knn_feature, and simple controls")
    gradient = get_scorer(str(probe_cfg["gradient_scorer"]))(
        model0, request, spec
    )
    proximity = get_scorer(str(probe_cfg["proximity_scorer"]))(
        model0, request, spec
    )
    controls: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        initial_nll: dict[str, float] = {}
        for batch in request.universe.batches(int(common["batch_size"])):
            values = seq_mean_answer_nll(model0, batch).detach().cpu().tolist()
            initial_nll.update(zip(batch["example_ids"], values))
    controls["initial_nll"] = {
        candidate: -value for candidate, value in initial_nll.items()
    }
    controls["answer_length"] = {
        example.example_id: float((example.labels != IGNORE).sum())
        for example in request.universe.examples
    }
    if "knn_lexical" in probe_cfg["simple_controls"]:
        controls["knn_lexical"] = get_scorer("knn_lexical")(
            model0, request, spec
        ).scores
    if "knn_embed" in probe_cfg["simple_controls"]:
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(
            str(probe_cfg["sentence_encoder"]), device="cpu"
        )

        def encode_text(items):
            return torch.as_tensor(
                encoder.encode(
                    [item.text for item in items],
                    batch_size=int(common["batch_size"]),
                    convert_to_numpy=True,
                )
            )

        set_embed_encoder(encode_text)
        controls["knn_embed"] = get_scorer("knn_embed")(
            model0, request, spec
        ).scores
    expected_controls = tuple(str(value) for value in probe_cfg["simple_controls"])
    if set(controls) != set(expected_controls):
        raise TOFUUnitError(
            f"simple control coverage differs: expected={expected_controls}, "
            f"observed={sorted(controls)}"
        )
    all_ids = sorted(by_id)
    profile_valid = all(
        set(scores) == set(all_ids)
        and all(math.isfinite(float(value)) for value in scores.values())
        for scores in (gradient.scores, proximity.scores, *controls.values())
    )
    if not profile_valid:
        raise TOFUUnitError("profile/control score coverage is invalid")
    profile_path = output_dir / "profiles.json"
    _atomic_json(
        profile_path,
        {
            "schema_version": 1,
            "request": args.request,
            "seed": args.seed,
            "candidate_universe_sha256": request.universe.sha,
            "score_independent_manifest_sha256": checked["content_sha256"],
            "probe": {
                "block": identity,
                "R": spec.n_dirs,
                "eta": spec.eta,
                "norm_eta": spec.norm_eta,
                "seed": spec.seed,
            },
            "gradient": {
                "scorer": gradient.scorer,
                "scores": gradient.scores,
                "cost": vars(gradient.cost),
                "artifacts": gradient.artifacts,
            },
            "proximity": {
                "scorer": proximity.scorer,
                "scores": proximity.scores,
                "cost": vars(proximity.cost),
                "artifacts": proximity.artifacts,
            },
            "simple_controls": controls,
            "profile_valid": profile_valid,
        },
    )

    def fidelity_row() -> dict[str, Any]:
        fidelity_cfg = _mapping(
            contract["execution"]["fidelity"], "execution.fidelity"
        )
        ids = sorted(all_ids)
        count = min(int(fidelity_cfg["n_candidates"]), len(ids))
        order = sorted(
            ids,
            key=lambda candidate: hashlib.sha256(
                f"{fidelity_cfg['candidate_seed']}|{candidate}".encode()
            ).digest(),
        )[:count]
        subset = Request.build(
            request.request_id,
            list(request.forget),
            CandidateUniverse.freeze([by_id[candidate] for candidate in order]),
        )
        exact = get_scorer("grad_norm")(model0, subset, spec)
        left = [gradient.scores[candidate] for candidate in order]
        right = [exact.scores[candidate] for candidate in order]
        rho = spearman(left, right)
        raw_k = int(fidelity_cfg["k"])
        k = (
            int(selected["protection"]["Kp"])
            if selected is not None
            else raw_k
            or max(
                1,
                math.ceil(
                    float(contract["execution"]["bootstrap"]["top_q"]) * count
                ),
            )
        )
        if k > count:
            raise TOFUUnitError(
                f"fidelity Kp={k} exceeds exact-reference pool size {count}"
            )
        overlap = len(
            top_k_ids({candidate: gradient.scores[candidate] for candidate in order}, k)
            & top_k_ids(exact.scores, k)
        ) / k
        perturb = perturbation_report(
            model0,
            spec,
            float(fidelity_cfg["norm_eta"]),
            seed=int(fidelity_cfg["seed"]),
        )
        survival = min(perturb["eff_over_eta"], perturb["frac_changed"])
        _atomic_json(
            output_dir / "fidelity_diagnostics.json",
            {
                "schema_version": 1,
                "request": args.request,
                "seed": args.seed,
                "candidate_ids": order,
                "Kp": k,
                "loss_shake_scores": {
                    candidate: gradient.scores[candidate]
                    for candidate in order
                },
                "exact_gradient_scores": exact.scores,
                "perturbation_report": perturb,
                "exact_cost": vars(exact.cost),
            },
        )
        return {
            "setting": args.setting,
            "parent": args.parent,
            "request": args.request,
            "seed": args.seed,
            "f_rho": rho,
            "f_k": overlap,
            "perturbations_valid": (
                survival >= float(fidelity_cfg["min_perturbation_survival"])
            ),
            "exact_reference_valid": (
                set(exact.scores) == set(order)
                and all(math.isfinite(value) for value in exact.scores.values())
            ),
            "common_control_support": all(
                set(scores) >= set(order) for scores in controls.values()
            ),
        }

    fidelity = fidelity_row() if args.stage in {"calibration", "target_evaluation"} else None
    del model0
    gate_runtime.clear_cuda_cache()
    parent_retain = [by_id[candidate] for candidate in checked["parent_retain_ids"]]
    audit_ids = list(checked["audit_ids"])

    def run_parent(values: Mapping[str, Any]):
        model = fresh()
        record = run_trajectory(
            model,
            args.parent,
            request,
            parent_retain,
            _trajectory_config(values, runtime, block, args.seed),
            stop_at_recall=float(runtime["parent"]["recall_max"]),
        )
        reached = bool(record.snapshots) and (
            record.snapshots[-1].forget_recall
            <= float(runtime["parent"]["recall_max"])
        )
        saved = {
            name: value.detach().cpu()
            for name, value in save_params(block.select(model)).items()
        }
        return model, record, reached, saved

    if args.stage == "calibration":
        parent_rows = []
        for values in _parent_grid(setting_runtime, args.parent):
            model, record, reached, _saved = run_parent(values)
            terminal = record.snapshots[-1]
            damage = record.damage_at()
            audit_damage = [damage[candidate] for candidate in audit_ids]
            parent_rows.append(
                {
                    "campaign_phase": "calibration",
                    "setting": args.setting,
                    "parent": args.parent,
                    "request": args.request,
                    "seed": args.seed,
                    "candidate_setting": values,
                    "reached": reached,
                    "forget_recall": terminal.forget_recall,
                    "mean_damage": _mean(audit_damage),
                    "cvar95_damage": cvar_upper(audit_damage, 0.05),
                    "step": terminal.step,
                }
            )
            del model
            gate_runtime.clear_cuda_cache()
        assert fidelity is not None
        _atomic_jsonl(output_dir / "fidelity_raw.jsonl", [fidelity])
        _atomic_jsonl(
            output_dir / "parent_selection_inputs.jsonl", parent_rows
        )
        parent_values = None
        parent_freeze_path = None
        parent_record = None
        parent_checkpoint = None
    else:
        parent_values, parent_freeze_path = _frozen_parent(
            setting_runtime,
            setting_id=args.setting,
            parent=args.parent,
        )
        parent_model, parent_record, reached, parent_block = run_parent(parent_values)
        terminal = parent_record.snapshots[-1]
        checkpoint_id = _checkpoint_id(
            parent_block,
            request=args.request,
            parent=args.parent,
            step=terminal.step,
        )
        parent_checkpoint = {
            "id": checkpoint_id,
            "reached": reached,
            "first_reaching": reached,
            "step": terminal.step,
            "forget_recall": terminal.forget_recall,
            "block": parent_block,
        }
        del parent_model
        gate_runtime.clear_cuda_cache()
        if not reached and args.stage in {"protection", "target_evaluation"}:
            raise TOFUUnitError(
                f"{args.parent}/{args.request}/seed-{args.seed} did not reach "
                "the direct forgetting gate"
            )

    prediction_rows: list[dict[str, Any]] = []
    protection_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    protection_diagnostics: dict[str, Any] | None = None
    if args.stage in {"prediction", "target_evaluation"}:
        assert parent_record is not None and parent_checkpoint is not None
        damage_all = parent_record.damage_at()
        damage = {candidate: damage_all[candidate] for candidate in audit_ids}
        alpha_diagnostics = []
        mixtures: dict[float, dict[str, float]] = {}
        for raw_alpha in probe_cfg["alpha_grid"]:
            alpha = float(raw_alpha)
            mixture = channel_mixture_scores(
                gradient.scores,
                proximity.scores,
                alpha,
                candidate_ids=audit_ids,
                normalization_ids=checked["discovery_ids"],
            )
            mixtures[alpha] = mixture
            alpha_diagnostics.append(
                {
                    "alpha": alpha,
                    "spearman": spearman(
                        [mixture[candidate] for candidate in audit_ids],
                        [damage[candidate] for candidate in audit_ids],
                    ),
                    "top_q_recall": _top_recall(
                        mixture,
                        damage,
                        float(contract["execution"]["bootstrap"]["top_q"]),
                    ),
                }
            )
        control_spearman = {
            name: spearman(
                [scores[candidate] for candidate in audit_ids],
                [damage[candidate] for candidate in audit_ids],
            )
            for name, scores in controls.items()
        }
        if selected is None:
            winner = min(
                alpha_diagnostics,
                key=lambda item: (
                    -item["spearman"],
                    -item["top_q_recall"],
                    abs(item["alpha"] - 0.5),
                    item["alpha"],
                ),
            )
            control_name = min(
                expected_controls,
                key=lambda name: (
                    -control_spearman[name],
                    expected_controls.index(name),
                ),
            )
            prediction_selection = {
                "valid": True,
                "fallback": False,
                "alpha": winner["alpha"],
            }
            selection_rows.append(
                {
                    "setting": args.setting,
                    "parent": args.parent,
                    "request": args.request,
                    "seed": args.seed,
                    "target_free": True,
                    "selection_kind": "prediction",
                    "alpha_pred": winner["alpha"],
                    "campaign_phase": "development",
                    "reached": parent_checkpoint["reached"],
                    "alpha_grid": alpha_diagnostics,
                    "control_spearman": control_spearman,
                    "simple_controls": list(expected_controls),
                    "local_simple_control": control_name,
                }
            )
        else:
            prediction_selection = _selection_mapping(selected["prediction"])
            alpha = float(prediction_selection["alpha"])
            control_name = str(
                selected["prediction"].get(
                    "simple_control", runtime["selection"].get("fallback_control", "initial_nll")
                )
            )
            if control_name not in controls:
                raise TOFUUnitError(
                    f"frozen simple control {control_name!r} was not computed"
                )
            if alpha not in mixtures:
                mixtures[alpha] = channel_mixture_scores(
                    gradient.scores,
                    proximity.scores,
                    alpha,
                    candidate_ids=audit_ids,
                    normalization_ids=checked["discovery_ids"],
                )
        alpha = float(prediction_selection["alpha"])
        joint = mixtures[alpha]
        for candidate in audit_ids:
            prediction_rows.append(
                {
                    "setting": args.setting,
                    "parent": args.parent,
                    "request": args.request,
                    "seed": args.seed,
                    "candidate_id": candidate,
                    "group": candidate_groups[candidate],
                    "s0": gradient.scores[candidate],
                    "s1": proximity.scores[candidate],
                    "joint": joint[candidate],
                    "simple_control": controls[control_name][candidate],
                    "simple_control_name": control_name,
                    "damage": damage[candidate],
                    "profile_valid": profile_valid,
                    "reached": parent_checkpoint["reached"],
                    "trajectory_completed": True,
                    "parent_checkpoint_id": parent_checkpoint["id"],
                    "parent_checkpoint_first_reaching": parent_checkpoint[
                        "first_reaching"
                    ],
                    "prediction_selection": prediction_selection,
                }
            )

    if args.stage in {"protection", "target_evaluation"}:
        assert parent_record is not None and parent_checkpoint is not None
        protection_cfg = _mapping(runtime["protection"], "protection")
        final_cfg = _mapping(
            protection_cfg["final_constraints"], "final_constraints"
        )
        repair_config = RepairConfig(
            **dict(_mapping(protection_cfg["repair"], "protection.repair"))
        )
        try:
            paraphrases = load_tofu_paraphrases(
                tokenizer, max_length=int(common["max_length"])
            )
            para_examples = [
                paraphrases[example.example_id] for example in request.forget
            ]
        except KeyError as error:
            raise TOFUUnitError("TOFU paraphrase coverage is incomplete") from error

        def evaluate_constraints(model) -> dict[str, float | bool]:
            direct = mean_recall(
                model, list(request.forget), int(common["batch_size"])
            )
            paraphrase = mean_recall(
                model, para_examples, int(common["batch_size"])
            )
            extraction = greedy_generation_recall(model, list(request.forget))
            native = mean_recall(
                model, native_utility, int(common["batch_size"])
            )
            margins = {
                "direct_forget_margin": float(final_cfg["direct_recall_max"])
                - direct,
                "paraphrase_forget_margin": float(
                    final_cfg["paraphrase_recall_max"]
                )
                - paraphrase,
                "extraction_generation_margin": float(
                    final_cfg["extraction_generation_max"]
                )
                - extraction,
                "utility_margin": native
                - float(final_cfg["native_utility_fraction_min"]) * native_baseline,
            }
            return {
                **margins,
                "native_retention": native,
                "feasible": all(value >= 0.0 for value in margins.values()),
            }

        folds = dict(checked["fold_by_group"])

        def partition_for(scores: Mapping[str, float], kp: int, label: str):
            profile = ScoreProfile(
                request.request_id,
                label,
                dict(scores),
                spec,
            )
            return build_pdf_protection_partition(
                profile,
                request,
                folds,
                PartitionParams(pool_size=kp, min_pool_size=kp, seed=args.seed),
                neutral_ids=checked["neutral_ids"],
                repair_eligible_ids=set(checked["repair_eligible_ids"]),
            )

        def execute_arm(
            *,
            arm: str,
            scores: Mapping[str, float] | None,
            kp: int,
            draw_id: str | None = None,
        ) -> dict[str, Any]:
            model = fresh()
            load_params_(block.select(model), parent_checkpoint["block"])
            if arm == "no_repair":
                record = parent_record
                updates = rollbacks = 0.0
                partition_sha = None
            else:
                if scores is None:
                    raise AssertionError("repair arm requires allocation scores")
                partition = partition_for(
                    scores,
                    kp,
                    f"{arm}:{draw_id or 'fixed'}",
                )
                latest: dict[str, float | bool] = {}

                def external(candidate_model) -> bool:
                    latest.clear()
                    latest.update(evaluate_constraints(candidate_model))
                    return bool(latest["feasible"])

                record = run_pdf_repair_from_reached(
                    model,
                    block,
                    request,
                    [by_id[candidate] for candidate in partition.protect],
                    [by_id[candidate] for candidate in checked["neutral_ids"]],
                    [
                        by_id[candidate]
                        for candidate in checked["utility_guard_ids"]
                    ],
                    args.parent,
                    PDFRepairedConfig(
                        repair=repair_config,
                        recall_max=float(runtime["parent"]["recall_max"]),
                        batch_size=int(common["batch_size"]),
                    ),
                    parent_record,
                    external_feasibility=external,
                    log=log,
                )
                metadata = record.metadata["pdf_v4_repair"]
                if metadata["stopped_reason"] == "token_budget_exhausted":
                    raise TOFUUnitError(
                        f"{arm} repair exhausted its token budget before a "
                        "validated terminal snapshot"
                    )
                updates = float(metadata["n_accepted"])
                rollbacks = float(metadata["n_rejected"])
                partition_sha = partition.manifest_sha
            terminal = record.snapshots[-1]
            metrics = evaluate_constraints(model)
            damage_all = {
                candidate: terminal.nll[candidate] - record.nll0[candidate]
                for candidate in record.nll0
            }
            audit_damage = {
                candidate: damage_all[candidate] for candidate in audit_ids
            }
            result = {
                "arm": arm,
                "draw_id": draw_id,
                "candidate_damage": audit_damage,
                "metrics": metrics,
                "mean_damage": _mean(list(audit_damage.values())),
                "cvar95_damage": cvar_upper(list(audit_damage.values()), 0.05),
                "repair_updates": updates,
                "repair_rollbacks": rollbacks,
                "partition_sha256": partition_sha,
            }
            del model
            gate_runtime.clear_cuda_cache()
            return result

        discovery_ids = list(checked["discovery_ids"])
        mixtures_all: dict[float, dict[str, float]] = {}
        for raw_alpha in probe_cfg["alpha_grid"]:
            alpha = float(raw_alpha)
            mixtures_all[alpha] = channel_mixture_scores(
                gradient.scores,
                proximity.scores,
                alpha,
                candidate_ids=all_ids,
                normalization_ids=discovery_ids,
            )
        if selected is None:
            grid_results: dict[tuple[float, int], dict[str, Any]] = {}
            diagnostics = []
            for alpha, scores in mixtures_all.items():
                for kp in protection_cfg["Kp_grid"]:
                    result = execute_arm(
                        arm="joint", scores=scores, kp=int(kp)
                    )
                    grid_results[(alpha, int(kp))] = result
                    diagnostics.append(
                        {
                            "alpha": alpha,
                            "Kp": int(kp),
                            "feasible": result["metrics"]["feasible"],
                            "mean_damage": result["mean_damage"],
                            "cvar95_damage": result["cvar95_damage"],
                        }
                    )
            feasible = [
                item for item in diagnostics if item["feasible"] is True
            ]
            winner = min(
                feasible or diagnostics,
                key=lambda item: (
                    item["feasible"] is not True,
                    item["cvar95_damage"],
                    item["mean_damage"],
                    abs(item["alpha"] - 0.5),
                    item["alpha"],
                    item["Kp"],
                ),
            )
            alpha_prot = float(winner["alpha"])
            kp = int(winner["Kp"])
            protection_selection = {
                "valid": bool(feasible),
                "fallback": not bool(feasible),
                "alpha": alpha_prot,
            }
            selection_rows.append(
                {
                    "setting": args.setting,
                    "parent": args.parent,
                    "request": args.request,
                    "seed": args.seed,
                    "target_free": True,
                    "selection_kind": "protection",
                    "alpha_prot": alpha_prot,
                    "Kp": kp,
                    "native_metric_name": native_metric["name"],
                    "campaign_phase": "development",
                    "grid": diagnostics,
                }
            )
            joint_result = grid_results[(alpha_prot, kp)]
        else:
            protection_selection = _selection_mapping(selected["protection"])
            alpha_prot = float(protection_selection["alpha"])
            kp = int(selected["protection"]["Kp"])
            joint_scores = mixtures_all.get(alpha_prot)
            if joint_scores is None:
                joint_scores = channel_mixture_scores(
                    gradient.scores,
                    proximity.scores,
                    alpha_prot,
                    candidate_ids=all_ids,
                    normalization_ids=discovery_ids,
                )
            joint_result = execute_arm(
                arm="joint", scores=joint_scores, kp=kp
            )

        arm_results = [
            joint_result,
            execute_arm(arm="no_repair", scores=None, kp=kp),
            execute_arm(arm="s0", scores=gradient.scores, kp=kp),
            execute_arm(arm="s1", scores=proximity.scores, kp=kp),
        ]
        for draw_id in contract["execution"]["repeated_random_draws"]:
            arm_results.append(
                execute_arm(
                    arm="repeated_random",
                    scores=_random_scores(
                        all_ids,
                        (
                            f"{campaign['campaign_id']}|{args.setting}|{args.parent}|"
                            f"{args.request}|{args.seed}|{draw_id}"
                        ),
                    ),
                    kp=kp,
                    draw_id=str(draw_id),
                )
            )
        for result in arm_results:
            metrics = result["metrics"]
            for candidate in audit_ids:
                row = {
                    "setting": args.setting,
                    "parent": args.parent,
                    "request": args.request,
                    "seed": args.seed,
                    "candidate_id": candidate,
                    "group": candidate_groups[candidate],
                    "arm": result["arm"],
                    "Kp": kp,
                    "damage": result["candidate_damage"][candidate],
                    "native_retention": metrics["native_retention"],
                    "feasible": metrics["feasible"],
                    "direct_forget_margin": metrics["direct_forget_margin"],
                    "paraphrase_forget_margin": metrics[
                        "paraphrase_forget_margin"
                    ],
                    "extraction_generation_margin": metrics[
                        "extraction_generation_margin"
                    ],
                    "utility_margin": metrics["utility_margin"],
                    "repair_updates": result["repair_updates"],
                    "repair_rollbacks": result["repair_rollbacks"],
                    "parent_checkpoint_id": parent_checkpoint["id"],
                    "parent_checkpoint_first_reaching": True,
                    "protection_selection": protection_selection,
                }
                if result["arm"] == "repeated_random":
                    row.update(
                        {
                            "draw_id": result["draw_id"],
                            "draw_complete": True,
                        }
                    )
                protection_rows.append(row)
        protection_diagnostics = {
            "schema_version": 1,
            "request": args.request,
            "seed": args.seed,
            "parent": args.parent,
            "parent_checkpoint_id": parent_checkpoint["id"],
            "alpha_prot": alpha_prot,
            "Kp": kp,
            "arms": [
                {
                    key: value
                    for key, value in result.items()
                    if key != "candidate_damage"
                }
                for result in arm_results
            ],
        }
        _atomic_json(
            output_dir / "protection_diagnostics.json",
            protection_diagnostics,
        )

    if prediction_rows:
        _atomic_jsonl(output_dir / "prediction_raw.jsonl", prediction_rows)
    if fidelity is not None:
        _atomic_jsonl(output_dir / "fidelity_raw.jsonl", [fidelity])
    if protection_rows:
        _atomic_jsonl(output_dir / "protection_raw.jsonl", protection_rows)
    if selection_rows:
        _atomic_jsonl(output_dir / "selection_inputs.jsonl", selection_rows)

    manifest_payload = {
        "schema_version": 1,
        "contract": "tofu-pdf-v4-unit-output",
        "campaign_id": campaign["campaign_id"],
        "stage": args.stage,
        "setting": args.setting,
        "parent": args.parent,
        "request": args.request,
        "seed": args.seed,
        "model_source": str(model_source),
        "model_dtype": model_cfg["dtype"],
        "candidate_authors": candidate_authors,
        "candidate_universe_sha256": request.universe.sha,
        "score_independent_manifest_sha256": checked["content_sha256"],
        "score_independent_manifest": {
            "path": str(stream_path),
            "sha256": _sha256(stream_path),
        },
        "semantic_eligibility_sha256": _sha256(eligibility_path),
        "block": identity,
        "sft": sft_result,
        "native_metric": dict(native_metric),
        "native_utility_ids_sha256": _json_sha(
            [example.example_id for example in native_utility]
        ),
        "native_utility_baseline": native_baseline,
        "campaign_config_sha256": _sha256(campaign_path),
        "evidence_config_sha256": _sha256(evidence_path),
        "runtime_config_sha256": _sha256(runtime_path),
        "profile_artifact": {
            "path": str(profile_path),
            "sha256": _sha256(profile_path),
        },
        "fidelity_diagnostics": (
            {
                "path": str(output_dir / "fidelity_diagnostics.json"),
                "sha256": _sha256(output_dir / "fidelity_diagnostics.json"),
            }
            if (output_dir / "fidelity_diagnostics.json").is_file()
            else None
        ),
        "protection_diagnostics": (
            {
                "path": str(output_dir / "protection_diagnostics.json"),
                "sha256": _sha256(output_dir / "protection_diagnostics.json"),
            }
            if protection_diagnostics is not None
            else None
        ),
        "parent_freeze": (
            str(parent_freeze_path) if args.stage != "calibration" else None
        ),
        "parent_freeze_sha256": (
            _sha256(parent_freeze_path)
            if args.stage != "calibration"
            else None
        ),
        "selection_freeze": (
            {
                "path": str(args.selection_freeze.resolve()),
                "sha256": _sha256(args.selection_freeze.resolve()),
            }
            if args.stage == "target_evaluation"
            and args.selection_freeze is not None
            else None
        ),
        "parent_trajectory": (
            trajectory_payload(parent_record)
            if args.stage != "calibration"
            else None
        ),
        "outputs": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
            }
            for name in (
                "prediction_raw.jsonl",
                "fidelity_raw.jsonl",
                "protection_raw.jsonl",
                "selection_inputs.jsonl",
                "parent_selection_inputs.jsonl",
            )
            if (path := output_dir / name).is_file()
        },
    }
    _atomic_json(output_dir / "run_manifest.json", manifest_payload)
    gate_runtime.clear_cuda_cache()
    log(f"completed {args.stage}/{args.parent}/{args.request}/seed-{args.seed}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign", type=Path, default=ROOT / "configs/paper/campaign.yaml"
    )
    parser.add_argument(
        "--evidence", type=Path, default=ROOT / "configs/paper/evidence.yaml"
    )
    parser.add_argument(
        "--runtime", type=Path, default=ROOT / "configs/paper/tofu_v4.yaml"
    )
    parser.add_argument("--stage", choices=tuple(STAGE_ROSTER), required=True)
    parser.add_argument("--setting", required=True)
    parser.add_argument("--parent", choices=PARENTS, required=True)
    parser.add_argument("--request", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--selection-freeze", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run_unit(parse_args(argv))
        return 0
    except (TOFUUnitError, ValueError, RuntimeError, OSError) as error:
        print(f"TOFU PDF-v4 unit failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
