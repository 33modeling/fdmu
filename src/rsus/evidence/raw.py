"""Aggregate candidate-level campaign records into the paper evidence ledger.

The input plan is the denominator: it enumerates every
``(setting, parent, request, seed)`` unit before results are inspected.  Raw
files may be incomplete, but may never introduce an unplanned unit.  Paired
contrasts are computed only when a unit has exact candidate support for every
required arm; a missing arm is recorded in the funnel instead of being hidden
by a pairwise intersection.

The implementation is deliberately dependency-free so aggregation can run on
a CPU login node after GPU jobs have written JSONL shards.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Callable, Iterable, Mapping, Sequence

from ..analysis.prediction import cvar_upper
from .schemas import (
    PROTECTION_COMPARATORS,
    EvidenceLedger,
    EvidenceValidationError,
    Selection,
)
from .statistics import percentile, summarize_bootstrap_effect


CLAIM_ARMS = ("joint", "no_repair", "repeated_random", "s0", "s1")
NON_RANDOM_ARMS = tuple(arm for arm in CLAIM_ARMS if arm != "repeated_random")
METRIC_OUTCOMES = ("mean", "cvar95")


UnitKey = tuple[str, str, str, str]
RowKey = tuple[str, str]


def _required_text(raw: Mapping[str, Any], field: str, *, where: str) -> str:
    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise EvidenceValidationError(f"{where}.{field} must be a string or integer")
    result = str(value).strip()
    if not result:
        raise EvidenceValidationError(f"{where}.{field} must be non-empty")
    return result


def _required_bool(raw: Mapping[str, Any], field: str, *, where: str) -> bool:
    value = raw.get(field)
    if type(value) is not bool:
        raise EvidenceValidationError(f"{where}.{field} must be a boolean")
    return value


def _optional_bool(
    raw: Mapping[str, Any], field: str, *, where: str, default: bool
) -> bool:
    if field not in raw:
        return default
    return _required_bool(raw, field, where=where)


def _number(raw: Mapping[str, Any], field: str, *, where: str) -> float:
    value = raw.get(field)
    if isinstance(value, bool):
        raise EvidenceValidationError(f"{where}.{field} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise EvidenceValidationError(f"{where}.{field} must be numeric") from error
    if not math.isfinite(result):
        raise EvidenceValidationError(f"{where}.{field} must be finite")
    return result


def _probability(raw: Mapping[str, Any], field: str, *, default: float) -> float:
    if field not in raw:
        return default
    value = _number(raw, field, where="bootstrap")
    if not 0.0 < value < 1.0:
        raise EvidenceValidationError(f"bootstrap.{field} must be in (0, 1)")
    return value


def _unit_key(raw: Mapping[str, Any], *, where: str) -> UnitKey:
    return (
        _required_text(raw, "setting", where=where),
        _required_text(raw, "parent", where=where),
        _required_text(raw, "request", where=where),
        _required_text(raw, "seed", where=where),
    )


def _selection(raw: object, *, where: str) -> Selection:
    if not isinstance(raw, Mapping):
        raise EvidenceValidationError(f"{where} must be a mapping")
    return Selection.from_mapping(raw, name=where)


def _selection_mapping(selection: Selection) -> dict[str, object]:
    return {
        "valid": selection.valid,
        "fallback": selection.fallback,
        "alpha": selection.alpha,
    }


@dataclass(frozen=True)
class PlanUnit:
    key: UnitKey
    prediction_selection: Selection
    protection_selection: Selection
    simple_control_name: str
    repeated_random_draws: tuple[str, ...]
    tail_m: int
    native_metric_name: str
    native_metric_orientation: str
    native_noninferiority_margin: float

    @property
    def row_key(self) -> RowKey:
        return self.key[0], self.key[1]

    @property
    def request(self) -> str:
        return self.key[2]

    @property
    def seed(self) -> str:
        return self.key[3]


@dataclass(frozen=True)
class RawPlan:
    units: Mapping[UnitKey, PlanUnit]
    bootstrap_replicates: int
    bootstrap_seed: int
    alpha: float
    top_q: float
    cvar_q: float
    artifact_contracts: Mapping[str, Mapping[str, Any]]
    source_sha256: str | None = None

    @property
    def row_keys(self) -> tuple[RowKey, ...]:
        return tuple(sorted({unit.row_key for unit in self.units.values()}))


def load_raw_plan(path: str | Path) -> RawPlan:
    """Load and validate an immutable unit plan JSON file."""
    source = Path(path)
    try:
        payload = source.read_bytes()
        raw = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceValidationError(f"cannot read raw plan {source}: {error}") from error
    plan = raw_plan_from_mapping(raw)
    return RawPlan(
        units=plan.units,
        bootstrap_replicates=plan.bootstrap_replicates,
        bootstrap_seed=plan.bootstrap_seed,
        alpha=plan.alpha,
        top_q=plan.top_q,
        cvar_q=plan.cvar_q,
        artifact_contracts=plan.artifact_contracts,
        source_sha256=hashlib.sha256(payload).hexdigest(),
    )


def raw_plan_from_mapping(raw: object) -> RawPlan:
    if not isinstance(raw, Mapping):
        raise EvidenceValidationError("raw plan root must be a mapping")
    if raw.get("schema_version") != 2:
        raise EvidenceValidationError("raw plan schema_version must be 2")
    settings = raw.get("bootstrap", {}) or {}
    if not isinstance(settings, Mapping):
        raise EvidenceValidationError("raw plan bootstrap must be a mapping")
    replicates = settings.get("replicates", 2000)
    seed = settings.get("seed", 1729)
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise EvidenceValidationError("bootstrap.replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise EvidenceValidationError("bootstrap.seed must be an integer")
    alpha = _probability(settings, "alpha", default=0.05)
    top_q = _probability(settings, "top_q", default=0.05)
    cvar_q = _probability(settings, "cvar_q", default=0.95)

    raw_units = raw.get("units")
    if not isinstance(raw_units, list) or not raw_units:
        raise EvidenceValidationError("raw plan units must be a non-empty list")
    units: dict[UnitKey, PlanUnit] = {}
    for index, item in enumerate(raw_units):
        where = f"units[{index}]"
        if not isinstance(item, Mapping):
            raise EvidenceValidationError(f"{where} must be a mapping")
        key = _unit_key(item, where=where)
        if key in units:
            raise EvidenceValidationError(f"duplicate planned unit {key!r}")
        prediction_selection = _selection(
            item.get("prediction_selection"),
            where=f"{where}.prediction_selection",
        )
        protection_selection = _selection(
            item.get("protection_selection"),
            where=f"{where}.protection_selection",
        )
        simple_control_name = _required_text(
            item, "simple_control_name", where=where
        )
        raw_draws = item.get("repeated_random_draws")
        if not isinstance(raw_draws, list) or not raw_draws:
            raise EvidenceValidationError(
                f"{where}.repeated_random_draws must be a non-empty list"
            )
        draws = tuple(
            _required_text({"draw": draw}, "draw", where=f"{where}.repeated_random_draws")
            for draw in raw_draws
        )
        if len(set(draws)) != len(draws):
            raise EvidenceValidationError(
                f"{where}.repeated_random_draws contains duplicates"
            )
        tail_m = item.get("tail_m")
        if isinstance(tail_m, bool) or not isinstance(tail_m, int) or tail_m < 1:
            raise EvidenceValidationError(f"{where}.tail_m must be a positive integer")
        native_margin = _number(
            item, "native_noninferiority_margin", where=where
        )
        if not 0.0 <= native_margin <= 0.1:
            raise EvidenceValidationError(
                f"{where}.native_noninferiority_margin must lie in [0, 0.1]"
            )
        native_name = _required_text(item, "native_metric_name", where=where)
        native_orientation = item.get("native_metric_orientation")
        if native_orientation not in {"higher", "lower"}:
            raise EvidenceValidationError(
                f"{where}.native_metric_orientation must be higher or lower"
            )
        units[key] = PlanUnit(
            key=key,
            prediction_selection=prediction_selection,
            protection_selection=protection_selection,
            simple_control_name=simple_control_name,
            repeated_random_draws=draws,
            tail_m=tail_m,
            native_metric_name=native_name,
            native_metric_orientation=native_orientation,
            native_noninferiority_margin=native_margin,
        )

    # A parent-level mixture weight is frozen once and therefore cannot vary
    # by request or seed inside the same paper row.
    by_row: dict[RowKey, list[PlanUnit]] = defaultdict(list)
    for unit in units.values():
        by_row[unit.row_key].append(unit)
    for row_key, row_units in by_row.items():
        prediction = {unit.prediction_selection for unit in row_units}
        protection = {unit.protection_selection for unit in row_units}
        controls = {unit.simple_control_name for unit in row_units}
        if len(prediction) != 1:
            raise EvidenceValidationError(
                f"planned row {row_key!r} changes frozen prediction selection"
            )
        if len(protection) != 1:
            raise EvidenceValidationError(
                f"planned row {row_key!r} changes frozen protection selection"
            )
        if len(controls) != 1:
            raise EvidenceValidationError(
                f"planned row {row_key!r} changes frozen simple control"
            )

    raw_contracts = raw.get("artifact_contracts", {}) or {}
    if not isinstance(raw_contracts, Mapping):
        raise EvidenceValidationError("raw plan artifact_contracts must be a mapping")
    artifact_contracts: dict[str, Mapping[str, Any]] = {}
    allowed_artifacts = {
        "campaign_manifest",
        "tail_structure",
        "lse_fidelity_cost",
        "protection_budget_sweep",
        "specificity_negative_controls",
    }
    for artifact_id, contract in raw_contracts.items():
        name = str(artifact_id)
        if name not in allowed_artifacts:
            raise EvidenceValidationError(
                f"artifact_contracts has unsupported artifact {name!r}"
            )
        if not isinstance(contract, Mapping):
            raise EvidenceValidationError(
                f"artifact_contracts.{name} must be a mapping"
            )
        kind = contract.get("kind")
        expected_kind = {
            "campaign_manifest": "plan_manifest",
            "tail_structure": "tail_from_prediction",
            "lse_fidelity_cost": "measurements",
            "protection_budget_sweep": "measurements",
            "specificity_negative_controls": "measurements",
        }[name]
        if kind != expected_kind:
            raise EvidenceValidationError(
                f"artifact_contracts.{name}.kind must be {expected_kind!r}"
            )
        artifact_contracts[name] = dict(contract)

    return RawPlan(
        units=units,
        bootstrap_replicates=replicates,
        bootstrap_seed=seed,
        alpha=alpha,
        top_q=top_q,
        cvar_q=cvar_q,
        artifact_contracts=artifact_contracts,
    )


def read_raw_records(paths: Iterable[str | Path]) -> list[Mapping[str, Any]]:
    """Read JSONL shards (or JSON arrays) without treating absent files as data."""
    records: list[Mapping[str, Any]] = []
    for value in paths:
        path = Path(value)
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as error:
            raise EvidenceValidationError(f"cannot read raw records {path}: {error}") from error
        if path.suffix.lower() == ".json":
            try:
                payload = json.loads(text)
            except json.JSONDecodeError as error:
                raise EvidenceValidationError(f"invalid JSON in {path}: {error}") from error
            if not isinstance(payload, list):
                raise EvidenceValidationError(f"{path} must contain a JSON list")
            items = payload
        else:
            items = []
            for line_number, line in enumerate(text.splitlines(), start=1):
                if not line.strip():
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise EvidenceValidationError(
                        f"invalid JSON at {path}:{line_number}: {error}"
                    ) from error
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise EvidenceValidationError(f"{path} record {index} must be a mapping")
            records.append(item)
    return records


@dataclass(frozen=True)
class PredictionCandidate:
    candidate_id: str
    group: str
    s0: float
    s1: float
    joint: float
    simple_control: float
    damage: float
    joint_exact: float | None = None


@dataclass(frozen=True)
class PredictionUnitData:
    key: UnitKey
    profile_valid: bool
    reached: bool
    trajectory_completed: bool
    parent_checkpoint_id: str
    parent_checkpoint_first_reaching: bool
    candidates: tuple[PredictionCandidate, ...]


@dataclass(frozen=True)
class FidelityUnitData:
    key: UnitKey
    perturbations_valid: bool
    exact_reference_valid: bool
    common_control_support: bool
    f_rho: float
    f_k: float


@dataclass(frozen=True)
class ProtectionRecord:
    candidate_id: str
    group: str
    arm: str
    draw_id: str | None
    damage: float
    native_retention: float
    feasible: bool
    direct_forget_margin: float
    paraphrase_forget_margin: float
    extraction_generation_margin: float
    utility_margin: float
    draw_complete: bool
    parent_checkpoint_id: str
    parent_checkpoint_first_reaching: bool
    repair_updates: float
    repair_rollbacks: float

    @property
    def forget_margin(self) -> float:
        """Least favorable slack across all three forgetting audits."""
        return min(
            self.direct_forget_margin,
            self.paraphrase_forget_margin,
            self.extraction_generation_margin,
        )


@dataclass(frozen=True)
class ProtectionUnitData:
    key: UnitKey
    candidates: tuple[str, ...]
    groups: Mapping[str, str]
    damages: Mapping[str, Mapping[str, float]]
    random_draw_damages: Mapping[str, Mapping[str, float]]
    native_retention: Mapping[str, float]
    random_draw_native_retention: Mapping[str, float]
    native_metric_orientation: str
    native_noninferiority_margin: float
    all_random_draws_complete: bool
    feasible: bool
    common: bool
    min_forget_margin: float | None
    min_utility_margin: float | None
    repair_updates: Mapping[str, float]
    repair_rollbacks: Mapping[str, float]


def _validate_frozen_selection(
    raw: Mapping[str, Any],
    field: str,
    planned: Selection,
    *,
    where: str,
) -> None:
    observed = _selection(raw.get(field), where=f"{where}.{field}")
    if observed != planned:
        raise EvidenceValidationError(
            f"{where}.{field} does not match the frozen unit plan"
        )


def _parse_prediction_records(
    plan: RawPlan, records: Iterable[Mapping[str, Any]]
) -> dict[UnitKey, PredictionUnitData]:
    grouped: dict[UnitKey, list[PredictionCandidate]] = defaultdict(list)
    status: dict[UnitKey, tuple[bool, bool, bool, str, bool]] = {}
    seen: set[tuple[UnitKey, str]] = set()
    for index, raw in enumerate(records):
        where = f"prediction[{index}]"
        key = _unit_key(raw, where=where)
        unit = plan.units.get(key)
        if unit is None:
            raise EvidenceValidationError(f"{where} has unplanned unit {key!r}")
        _validate_frozen_selection(
            raw,
            "prediction_selection",
            unit.prediction_selection,
            where=where,
        )
        observed_control = _required_text(
            raw, "simple_control_name", where=where
        )
        if observed_control != unit.simple_control_name:
            raise EvidenceValidationError(
                f"{where}.simple_control_name does not match the frozen unit plan"
            )
        candidate_id = _required_text(raw, "candidate_id", where=where)
        identity = key, candidate_id
        if identity in seen:
            raise EvidenceValidationError(
                f"duplicate prediction candidate {candidate_id!r} for {key!r}"
            )
        seen.add(identity)
        reached = _required_bool(raw, "reached", where=where)
        first_reaching = _required_bool(
            raw, "parent_checkpoint_first_reaching", where=where
        )
        if reached and not first_reaching:
            raise EvidenceValidationError(
                f"{where} reached damage is not from the first reaching checkpoint"
            )
        observed_status = (
            _required_bool(raw, "profile_valid", where=where),
            reached,
            _optional_bool(raw, "trajectory_completed", where=where, default=True),
            _required_text(raw, "parent_checkpoint_id", where=where),
            first_reaching,
        )
        if key in status and status[key] != observed_status:
            raise EvidenceValidationError(f"{where} has inconsistent unit status")
        status[key] = observed_status
        grouped[key].append(
            PredictionCandidate(
                candidate_id=candidate_id,
                group=_required_text(raw, "group", where=where),
                s0=_number(raw, "s0", where=where),
                s1=_number(raw, "s1", where=where),
                joint=_number(raw, "joint", where=where),
                simple_control=_number(raw, "simple_control", where=where),
                damage=_number(raw, "damage", where=where),
                joint_exact=(
                    _number(raw, "joint_exact", where=where)
                    if raw.get("joint_exact") is not None
                    else None
                ),
            )
        )
    for key, candidates in grouped.items():
        exact_count = sum(item.joint_exact is not None for item in candidates)
        if exact_count not in (0, len(candidates)):
            raise EvidenceValidationError(
                f"prediction unit {key!r} has a partial joint_exact column"
            )
    return {
        key: PredictionUnitData(
            key=key,
            profile_valid=status[key][0],
            reached=status[key][1],
            trajectory_completed=status[key][2],
            parent_checkpoint_id=status[key][3],
            parent_checkpoint_first_reaching=status[key][4],
            candidates=tuple(sorted(values, key=lambda value: value.candidate_id)),
        )
        for key, values in grouped.items()
    }


def _parse_fidelity_records(
    plan: RawPlan, records: Iterable[Mapping[str, Any]]
) -> dict[UnitKey, FidelityUnitData]:
    parsed: dict[UnitKey, FidelityUnitData] = {}
    for index, raw in enumerate(records):
        where = f"fidelity[{index}]"
        key = _unit_key(raw, where=where)
        if key not in plan.units:
            raise EvidenceValidationError(f"{where} has unplanned unit {key!r}")
        if key in parsed:
            raise EvidenceValidationError(f"duplicate fidelity unit {key!r}")
        f_rho = _number(raw, "f_rho", where=where)
        f_k = _number(raw, "f_k", where=where)
        if not -1.0 <= f_rho <= 1.0:
            raise EvidenceValidationError(f"{where}.f_rho must be in [-1, 1]")
        if not 0.0 <= f_k <= 1.0:
            raise EvidenceValidationError(f"{where}.f_k must be in [0, 1]")
        parsed[key] = FidelityUnitData(
            key=key,
            perturbations_valid=_required_bool(
                raw, "perturbations_valid", where=where
            ),
            exact_reference_valid=_required_bool(
                raw, "exact_reference_valid", where=where
            ),
            common_control_support=_required_bool(
                raw, "common_control_support", where=where
            ),
            f_rho=f_rho,
            f_k=f_k,
        )
    return parsed


def _parse_protection_records(
    plan: RawPlan, records: Iterable[Mapping[str, Any]]
) -> tuple[dict[UnitKey, ProtectionUnitData], set[UnitKey]]:
    grouped: dict[UnitKey, list[ProtectionRecord]] = defaultdict(list)
    seen: set[tuple[UnitKey, str, str | None, str]] = set()
    for index, raw in enumerate(records):
        where = f"protection[{index}]"
        key = _unit_key(raw, where=where)
        unit = plan.units.get(key)
        if unit is None:
            raise EvidenceValidationError(f"{where} has unplanned unit {key!r}")
        _validate_frozen_selection(
            raw,
            "protection_selection",
            unit.protection_selection,
            where=where,
        )
        raw_kp = raw.get("Kp")
        if isinstance(raw_kp, bool) or not isinstance(raw_kp, int):
            raise EvidenceValidationError(f"{where}.Kp must be an integer")
        if raw_kp != unit.tail_m:
            raise EvidenceValidationError(
                f"{where}.Kp does not match the frozen unit plan"
            )
        native_name = _required_text(
            raw, "native_metric_name", where=where
        )
        if native_name != unit.native_metric_name:
            raise EvidenceValidationError(
                f"{where}.native_metric_name does not match the frozen unit plan"
            )
        arm = _required_text(raw, "arm", where=where)
        if arm not in CLAIM_ARMS:
            raise EvidenceValidationError(f"{where}.arm is not one of {CLAIM_ARMS}")
        raw_draw = raw.get("draw_id")
        draw_id: str | None
        if arm == "repeated_random":
            if raw_draw is None:
                raise EvidenceValidationError(
                    f"{where}.draw_id is required for repeated_random"
                )
            draw_id = _required_text({"draw_id": raw_draw}, "draw_id", where=where)
            if draw_id not in unit.repeated_random_draws:
                raise EvidenceValidationError(
                    f"{where}.draw_id {draw_id!r} is not in the frozen plan"
                )
            draw_complete = _required_bool(raw, "draw_complete", where=where)
        else:
            if raw_draw is not None:
                raise EvidenceValidationError(
                    f"{where}.draw_id is only valid for repeated_random"
                )
            draw_id = None
            draw_complete = True
        candidate_id = _required_text(raw, "candidate_id", where=where)
        identity = key, arm, draw_id, candidate_id
        if identity in seen:
            raise EvidenceValidationError(
                f"duplicate protection record {identity!r}"
            )
        seen.add(identity)
        feasible = _required_bool(raw, "feasible", where=where)
        direct_margin = _number(raw, "direct_forget_margin", where=where)
        paraphrase_margin = _number(raw, "paraphrase_forget_margin", where=where)
        extraction_margin = _number(
            raw, "extraction_generation_margin", where=where
        )
        utility_margin = _number(raw, "utility_margin", where=where)
        checkpoint_id = _required_text(raw, "parent_checkpoint_id", where=where)
        first_reaching = _required_bool(
            raw, "parent_checkpoint_first_reaching", where=where
        )
        if not first_reaching:
            raise EvidenceValidationError(
                f"{where} did not start from the first direct-criterion-reaching "
                "parent checkpoint"
            )
        margins_feasible = all(
            value >= 0.0
            for value in (
                direct_margin,
                paraphrase_margin,
                extraction_margin,
                utility_margin,
            )
        )
        if feasible != margins_feasible:
            raise EvidenceValidationError(
                f"{where}.feasible must equal the conjunction of direct, "
                "paraphrase, extraction/generation, and utility margins"
            )
        grouped[key].append(
            ProtectionRecord(
                candidate_id=candidate_id,
                group=_required_text(raw, "group", where=where),
                arm=arm,
                draw_id=draw_id,
                damage=_number(raw, "damage", where=where),
                native_retention=_number(raw, "native_retention", where=where),
                feasible=feasible,
                direct_forget_margin=direct_margin,
                paraphrase_forget_margin=paraphrase_margin,
                extraction_generation_margin=extraction_margin,
                utility_margin=utility_margin,
                draw_complete=draw_complete,
                parent_checkpoint_id=checkpoint_id,
                parent_checkpoint_first_reaching=first_reaching,
                repair_updates=_number(raw, "repair_updates", where=where),
                repair_rollbacks=_number(
                    raw, "repair_rollbacks", where=where
                ),
            )
        )

    normalized = {
        key: _normalize_protection_unit(plan.units[key], values)
        for key, values in grouped.items()
    }
    return normalized, set(grouped)


def _normalize_protection_unit(
    unit: PlanUnit, records: Sequence[ProtectionRecord]
) -> ProtectionUnitData:
    arm_draws: dict[str, dict[str | None, dict[str, ProtectionRecord]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in records:
        arm_draws[record.arm][record.draw_id][record.candidate_id] = record
    checkpoint_ids = {record.parent_checkpoint_id for record in records}
    if len(checkpoint_ids) != 1:
        raise EvidenceValidationError(
            f"protection unit {unit.key!r} mixes parent checkpoints: "
            f"{sorted(checkpoint_ids)}"
        )

    regular_sets: dict[str, set[str]] = {}
    for arm in NON_RANDOM_ARMS:
        regular_sets[arm] = set(arm_draws.get(arm, {}).get(None, {}))
    random_sets = {
        draw: set(arm_draws.get("repeated_random", {}).get(draw, {}))
        for draw in unit.repeated_random_draws
    }
    present = all(regular_sets.values()) and all(random_sets.values())
    support_sets = list(regular_sets.values()) + list(random_sets.values())
    common = bool(present and support_sets and all(value == support_sets[0] for value in support_sets))

    all_records_expected: list[ProtectionRecord] = []
    for arm in NON_RANDOM_ARMS:
        all_records_expected.extend(arm_draws.get(arm, {}).get(None, {}).values())
    for draw in unit.repeated_random_draws:
        all_records_expected.extend(
            arm_draws.get("repeated_random", {}).get(draw, {}).values()
        )
    draw_ids_observed = set(arm_draws.get("repeated_random", {}))
    draw_complete = draw_ids_observed == set(unit.repeated_random_draws) and all(
        record.draw_complete
        for draw in unit.repeated_random_draws
        for record in arm_draws.get("repeated_random", {}).get(draw, {}).values()
    )
    # Feasibility and common candidate support are separate funnels. A unit can
    # have every arm complete and constraint-feasible yet fail the paired claim
    # because one arm changed candidate support.
    feasible = bool(present and draw_complete and all_records_expected) and all(
        record.feasible for record in all_records_expected
    )

    groups: dict[str, str] = {}
    if common:
        for record in all_records_expected:
            previous = groups.setdefault(record.candidate_id, record.group)
            if previous != record.group:
                raise EvidenceValidationError(
                    f"protection unit {unit.key!r} changes semantic group for "
                    f"candidate {record.candidate_id!r}"
                )

    damages: dict[str, dict[str, float]] = {}
    random_draw_damages: dict[str, dict[str, float]] = {}
    native_retention: dict[str, float] = {}
    random_draw_native_retention: dict[str, float] = {}
    repair_updates: dict[str, float] = {}
    repair_rollbacks: dict[str, float] = {}
    candidates = tuple(sorted(support_sets[0])) if common else ()
    if common:
        for arm in NON_RANDOM_ARMS:
            damages[arm] = {
                candidate: arm_draws[arm][None][candidate].damage
                for candidate in candidates
            }
            native_values = {
                arm_draws[arm][None][candidate].native_retention
                for candidate in candidates
            }
            if len(native_values) != 1:
                raise EvidenceValidationError(
                    f"protection unit {unit.key!r} changes native_retention "
                    f"within arm {arm!r}"
                )
            native_retention[arm] = native_values.pop()
            update_values = {
                arm_draws[arm][None][candidate].repair_updates
                for candidate in candidates
            }
            rollback_values = {
                arm_draws[arm][None][candidate].repair_rollbacks
                for candidate in candidates
            }
            if len(update_values) != 1 or len(rollback_values) != 1:
                raise EvidenceValidationError(
                    f"protection unit {unit.key!r} changes repair diagnostics "
                    f"within arm {arm!r}"
                )
            repair_updates[arm] = update_values.pop()
            repair_rollbacks[arm] = rollback_values.pop()
        damages["repeated_random"] = {
            candidate: sum(
                arm_draws["repeated_random"][draw][candidate].damage
                for draw in unit.repeated_random_draws
            )
            / len(unit.repeated_random_draws)
            for candidate in candidates
        }
        random_draw_damages = {
            draw: {
                candidate: arm_draws["repeated_random"][draw][candidate].damage
                for candidate in candidates
            }
            for draw in unit.repeated_random_draws
        }
        for draw in unit.repeated_random_draws:
            native_values = {
                arm_draws["repeated_random"][draw][candidate].native_retention
                for candidate in candidates
            }
            if len(native_values) != 1:
                raise EvidenceValidationError(
                    f"protection unit {unit.key!r} changes native_retention "
                    f"within repeated-random draw {draw!r}"
                )
            random_draw_native_retention[draw] = native_values.pop()
        native_retention["repeated_random"] = sum(
            random_draw_native_retention.values()
        ) / len(random_draw_native_retention)

    margins = all_records_expected if present and draw_complete else []
    return ProtectionUnitData(
        key=unit.key,
        candidates=candidates,
        groups=groups,
        damages=damages,
        random_draw_damages=random_draw_damages,
        native_retention=native_retention,
        random_draw_native_retention=random_draw_native_retention,
        native_metric_orientation=unit.native_metric_orientation,
        native_noninferiority_margin=unit.native_noninferiority_margin,
        all_random_draws_complete=draw_complete,
        feasible=feasible,
        common=common,
        min_forget_margin=min((record.forget_margin for record in margins), default=None),
        min_utility_margin=min((record.utility_margin for record in margins), default=None),
        repair_updates=repair_updates,
        repair_rollbacks=repair_rollbacks,
    )


def _midranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + 1 + end) / 2.0
        for position in range(cursor, end):
            ranks[order[position]] = rank
        cursor = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    variance_left = sum(value * value for value in centered_left)
    variance_right = sum(value * value for value in centered_right)
    if variance_left <= 0.0 or variance_right <= 0.0:
        return None
    return sum(a * b for a, b in zip(centered_left, centered_right)) / math.sqrt(
        variance_left * variance_right
    )


def _spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    return _correlation(_midranks(left), _midranks(right))


def _top_q_recall(
    scores: Sequence[float], damage: Sequence[float], candidate_ids: Sequence[str], q: float
) -> float | None:
    if len(scores) != len(damage) or len(scores) != len(candidate_ids) or not scores:
        return None
    count = max(1, math.ceil(q * len(scores)))
    score_top = {
        index
        for index in sorted(
            range(len(scores)), key=lambda i: (-scores[i], candidate_ids[i], i)
        )[:count]
    }
    damage_top = {
        index
        for index in sorted(
            range(len(damage)), key=lambda i: (-damage[i], candidate_ids[i], i)
        )[:count]
    }
    return len(score_top & damage_top) / count


def _prediction_metrics(
    candidates: Sequence[PredictionCandidate],
    *,
    top_q: float,
    tail_m: int,
    include_swap: bool = False,
) -> dict[str, float] | None:
    if len(candidates) < 2:
        return None
    damage = [candidate.damage for candidate in candidates]
    joint = [candidate.joint for candidate in candidates]
    s0 = [candidate.s0 for candidate in candidates]
    s1 = [candidate.s1 for candidate in candidates]
    control = [candidate.simple_control for candidate in candidates]
    rho_joint = _spearman(joint, damage)
    rho_s0 = _spearman(s0, damage)
    rho_s1 = _spearman(s1, damage)
    rho_control = _spearman(control, damage)
    recall = _top_q_recall(
        joint, damage, [candidate.candidate_id for candidate in candidates], top_q
    )
    if any(
        value is None
        for value in (rho_joint, rho_s0, rho_s1, rho_control, recall)
    ):
        return None
    assert rho_joint is not None and rho_s0 is not None and rho_s1 is not None
    assert rho_control is not None and recall is not None
    result = {
        "joint_rho": rho_joint,
        "top_q_recall": recall,
        "gain_s0": rho_joint - rho_s0,
        "gain_s1": rho_joint - rho_s1,
        "gain_control": rho_joint - rho_control,
    }
    if include_swap:
        if any(candidate.joint_exact is None for candidate in candidates):
            return None
        rho_exact = _spearman(
            [float(candidate.joint_exact) for candidate in candidates],
            damage,
        )
        if rho_exact is None:
            return None
        result["swap_delta"] = rho_exact - rho_joint
    positive = [
        index for index, candidate in enumerate(candidates) if candidate.damage > 0.0
    ]
    if len(positive) >= tail_m and tail_m <= len(candidates):
        harmful = set(
            sorted(
                positive,
                key=lambda index: (
                    -candidates[index].damage,
                    candidates[index].candidate_id,
                    index,
                ),
            )[:tail_m]
        )
        predicted = set(
            sorted(
                range(len(candidates)),
                key=lambda index: (
                    -candidates[index].joint,
                    candidates[index].candidate_id,
                    index,
                ),
            )[:tail_m]
        )
        recall_m = len(harmful & predicted) / tail_m
        predicted_s0 = set(
            sorted(
                range(len(candidates)),
                key=lambda index: (
                    -candidates[index].s0,
                    candidates[index].candidate_id,
                    index,
                ),
            )[:tail_m]
        )
        predicted_s1 = set(
            sorted(
                range(len(candidates)),
                key=lambda index: (
                    -candidates[index].s1,
                    candidates[index].candidate_id,
                    index,
                ),
            )[:tail_m]
        )
        q = tail_m / len(candidates)
        result["tail_lift"] = recall_m / q - 1.0
        result["chance_q"] = q
        result["tail_recall_joint"] = recall_m
        result["tail_recall_s0"] = len(harmful & predicted_s0) / tail_m
        result["tail_recall_s1"] = len(harmful & predicted_s1) / tail_m
    return result


@dataclass(frozen=True)
class RQ2UnitData:
    prediction: PredictionUnitData
    fidelity: FidelityUnitData

    @property
    def key(self) -> UnitKey:
        return self.prediction.key


def _rq2_metrics(
    data: RQ2UnitData,
    *,
    top_q: float,
    tail_m: int,
    rng: random.Random | None = None,
) -> dict[str, float] | None:
    candidates = (
        _resample_groups_prediction(data.prediction.candidates, rng)
        if rng is not None
        else list(data.prediction.candidates)
    )
    prediction = _prediction_metrics(
        candidates, top_q=top_q, tail_m=tail_m
    )
    if prediction is None:
        return None
    return {
        "f_rho_minus_0p80": data.fidelity.f_rho - 0.80,
        "f_k_minus_0p70": data.fidelity.f_k - 0.70,
        "g_h": prediction["gain_s1"],
        "g_ctl": prediction["gain_control"],
    }


def _prediction_core_metrics(
    candidates: Sequence[PredictionCandidate],
    *,
    top_q: float,
    tail_m: int,
    include_swap: bool = False,
) -> dict[str, float] | None:
    metrics = _prediction_metrics(
        candidates,
        top_q=top_q,
        tail_m=tail_m,
        include_swap=include_swap,
    )
    if metrics is None:
        return None
    keys = [
        "joint_rho",
        "top_q_recall",
        "gain_s0",
        "gain_s1",
        "gain_control",
    ]
    if include_swap:
        keys.append("swap_delta")
    return {key: metrics[key] for key in keys}


def _prediction_tail_metrics(
    candidates: Sequence[PredictionCandidate], *, top_q: float, tail_m: int
) -> dict[str, float] | None:
    metrics = _prediction_metrics(candidates, top_q=top_q, tail_m=tail_m)
    if metrics is None or "tail_lift" not in metrics:
        return None
    return {
        key: metrics[key]
        for key in (
            "tail_lift",
            "chance_q",
            "tail_recall_joint",
            "tail_recall_s0",
            "tail_recall_s1",
        )
    }


def _tail_mean(values: Sequence[float], *, cvar_q: float) -> float:
    return cvar_upper(list(values), 1.0 - cvar_q)


def _protection_metrics(
    unit: ProtectionUnitData,
    candidate_ids: Sequence[str],
    *,
    cvar_q: float,
    rng: random.Random | None = None,
) -> dict[str, float] | None:
    if not candidate_ids or not unit.common:
        return None
    result: dict[str, float] = {}
    joint = [unit.damages["joint"][candidate] for candidate in candidate_ids]
    joint_outcomes = {
        "mean": sum(joint) / len(joint),
        "cvar95": _tail_mean(joint, cvar_q=cvar_q),
    }
    result["profile.mean"] = joint_outcomes["mean"]
    result["profile.cvar95"] = joint_outcomes["cvar95"]
    result["profile.repair_updates"] = unit.repair_updates["joint"]
    result["profile.repair_rollbacks"] = unit.repair_rollbacks["joint"]
    for comparator in PROTECTION_COMPARATORS:
        if comparator == "repeated_random" and rng is not None:
            draws = sorted(unit.random_draw_damages)
            sampled_draws = [rng.choice(draws) for _ in draws]
            values = [
                sum(
                    unit.random_draw_damages[draw][candidate]
                    for draw in sampled_draws
                )
                / len(sampled_draws)
                for candidate in candidate_ids
            ]
        else:
            values = [unit.damages[comparator][candidate] for candidate in candidate_ids]
        outcomes = {
            "mean": sum(values) / len(values),
            "cvar95": _tail_mean(values, cvar_q=cvar_q),
        }
        if comparator == "no_repair":
            result["no_repair_absolute.mean"] = outcomes["mean"]
            result["no_repair_absolute.cvar95"] = outcomes["cvar95"]
        for outcome in METRIC_OUTCOMES:
            result[f"{comparator}.{outcome}"] = (
                joint_outcomes[outcome] - outcomes[outcome]
            )
        comparator_native = (
            sum(
                unit.random_draw_native_retention[draw]
                for draw in sampled_draws
            )
            / len(sampled_draws)
            if comparator == "repeated_random" and rng is not None
            else unit.native_retention[comparator]
        )
        oriented_difference = (
            unit.native_retention["joint"] - comparator_native
            if unit.native_metric_orientation == "higher"
            else comparator_native - unit.native_retention["joint"]
        )
        result[f"{comparator}.native_noninferiority"] = (
            oriented_difference + unit.native_noninferiority_margin
        )
    return result


def _average_metric_maps(values: Sequence[Mapping[str, float]]) -> dict[str, float]:
    if not values:
        raise ValueError("metric maps must be non-empty")
    keys = set(values[0])
    if any(set(value) != keys for value in values):
        raise ValueError("metric maps must have identical keys")
    return {
        key: sum(value[key] for value in values) / len(values)
        for key in sorted(keys)
    }


def _equal_request_average(
    units: Sequence[tuple[str, str, Mapping[str, float]]]
) -> dict[str, float]:
    by_request: dict[str, list[Mapping[str, float]]] = defaultdict(list)
    for request, _seed, metrics in units:
        by_request[request].append(metrics)
    request_metrics = [
        _average_metric_maps(seed_metrics)
        for _request, seed_metrics in sorted(by_request.items())
    ]
    return _average_metric_maps(request_metrics)


def _resample_groups_prediction(
    candidates: Sequence[PredictionCandidate], rng: random.Random
) -> list[PredictionCandidate]:
    by_group: dict[str, list[PredictionCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_group[candidate.group].append(candidate)
    groups = sorted(by_group)
    sampled: list[PredictionCandidate] = []
    for _ in groups:
        sampled.extend(by_group[rng.choice(groups)])
    return sampled


def _resample_groups_protection(
    unit: ProtectionUnitData, rng: random.Random
) -> list[str]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for candidate in unit.candidates:
        by_group[unit.groups[candidate]].append(candidate)
    groups = sorted(by_group)
    sampled: list[str] = []
    for _ in groups:
        sampled.extend(by_group[rng.choice(groups)])
    return sampled


def _hierarchical_bootstrap(
    units: Sequence[Any],
    *,
    replicates: int,
    seed: int,
    request_of: Callable[[Any], str],
    seed_of: Callable[[Any], str],
    resampled_metrics: Callable[[Any, random.Random], Mapping[str, float] | None],
    max_rejection_fraction: float = 0.5,
) -> list[dict[str, float]]:
    by_request: dict[str, list[Any]] = defaultdict(list)
    for unit in units:
        by_request[request_of(unit)].append(unit)
    requests = sorted(by_request)
    if not requests:
        return []
    for request in requests:
        by_request[request].sort(key=seed_of)
    rng = random.Random(seed)
    draws: list[dict[str, float]] = []
    attempts = 0
    max_attempts = max(replicates * 100, 1000)
    while len(draws) < replicates and attempts < max_attempts:
        attempts += 1
        request_draws: list[dict[str, float]] = []
        valid = True
        for _ in requests:
            request = rng.choice(requests)
            seeds = by_request[request]
            seed_metrics: list[Mapping[str, float]] = []
            for _ in seeds:
                unit = rng.choice(seeds)
                metrics = resampled_metrics(unit, rng)
                if metrics is None:
                    valid = False
                    break
                seed_metrics.append(metrics)
            if not valid:
                break
            request_draws.append(_average_metric_maps(seed_metrics))
        if valid:
            draws.append(_average_metric_maps(request_draws))
    if len(draws) != replicates:
        raise EvidenceValidationError(
            "hierarchical bootstrap could not produce enough finite paired draws; "
            "increase within-group candidate variation or inspect the raw shard"
        )
    rejection_fraction = 1.0 - replicates / attempts
    if rejection_fraction > max_rejection_fraction:
        raise EvidenceValidationError(
            f"hierarchical bootstrap rejected {rejection_fraction:.0%} of "
            f"replicates (cap {max_rejection_fraction:.0%}); the resampled "
            "statistic is too fragile to report"
        )
    return draws


def _empty_rq1(*, selection_valid: bool = False) -> dict[str, object]:
    return {
        "paired": False,
        "selection_valid": selection_valid,
        "profile_valid": False,
        "common_support_units": 0,
        "reached_valid_units": 0,
        "tail_eligible_units": 0,
        "joint_rho": {},
        "joint_minus_s0": {},
        "joint_minus_s1": {},
        "tail_lift": {},
        "chance_q": None,
        "tail_recall_joint": None,
        "tail_recall_s0": None,
        "tail_recall_s1": None,
        "swap_delta": {},
    }


def _empty_rq2() -> dict[str, object]:
    return {
        "paired": False,
        "perturbations_valid": False,
        "exact_reference_valid": False,
        "common_control_support": False,
        "common_support_units": 0,
        "f_rho_minus_0p80": {},
        "f_k_minus_0p70": {},
        "g_h": {},
        "g_ctl": {},
    }


def _empty_rq3(*, selection_valid: bool = False) -> dict[str, object]:
    return {
        "paired": False,
        "comparisons": {},
        "native_noninferiority": {},
        "selection_valid": selection_valid,
        "all_random_draws_complete": False,
        "all_five_arms_feasible": False,
        "common_support": False,
        "common_support_units": 0,
        "low_tail_support": False,
        "min_forget_margin": None,
        "min_utility_margin": None,
        "profile_mean": None,
        "profile_cvar95": None,
        "no_repair_mean": None,
        "no_repair_cvar95": None,
        "repair_updates": None,
        "repair_rollbacks": None,
    }


def aggregate_raw_evidence(
    plan: RawPlan,
    prediction_records: Iterable[Mapping[str, Any]],
    protection_records: Iterable[Mapping[str, Any]],
    *,
    fidelity_records: Iterable[Mapping[str, Any]] = (),
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a schema-v2 RQ1/RQ2/RQ3 ledger from frozen raw records."""
    prediction = _parse_prediction_records(plan, prediction_records)
    fidelity = _parse_fidelity_records(plan, fidelity_records)
    protection, protection_attempted = _parse_protection_records(
        plan, protection_records
    )
    rows: list[dict[str, Any]] = []
    for row_index, row_key in enumerate(plan.row_keys):
        row_units = sorted(
            (unit for unit in plan.units.values() if unit.row_key == row_key),
            key=lambda unit: (unit.request, unit.seed),
        )
        prediction_selection = row_units[0].prediction_selection
        protection_selection = row_units[0].protection_selection

        attempted_units = {
            unit.key
            for unit in row_units
            if (
                unit.key in prediction
                or unit.key in fidelity
                or unit.key in protection_attempted
            )
        }
        completed_units = {
            unit.key
            for unit in row_units
            if unit.key in prediction and prediction[unit.key].trajectory_completed
        }
        reached_units = {
            key
            for key in completed_units
            if prediction[key].reached
        }
        valid_profiles = {
            unit.key
            for unit in row_units
            if unit.key in prediction and prediction[unit.key].profile_valid
        }
        reached_valid = reached_units & valid_profiles

        prediction_common_data: list[PredictionUnitData] = []
        prediction_points: list[tuple[str, str, Mapping[str, float]]] = []
        tail_common_data: list[PredictionUnitData] = []
        tail_points: list[tuple[str, str, Mapping[str, float]]] = []
        row_has_swap = bool(row_units) and all(
            unit.key in prediction
            and bool(prediction[unit.key].candidates)
            and all(
                candidate.joint_exact is not None
                for candidate in prediction[unit.key].candidates
            )
            for unit in row_units
        )
        for unit in row_units:
            data = prediction.get(unit.key)
            if data is None or unit.key not in reached_valid:
                continue
            metrics = _prediction_core_metrics(
                data.candidates,
                top_q=plan.top_q,
                tail_m=unit.tail_m,
                include_swap=row_has_swap,
            )
            if metrics is not None:
                prediction_common_data.append(data)
                prediction_points.append((unit.request, unit.seed, metrics))
                tail_metrics = _prediction_tail_metrics(
                    data.candidates, top_q=plan.top_q, tail_m=unit.tail_m
                )
                if tail_metrics is not None:
                    tail_common_data.append(data)
                    tail_points.append((unit.request, unit.seed, tail_metrics))

        rq2_common_data: list[RQ2UnitData] = []
        rq2_points: list[tuple[str, str, Mapping[str, float]]] = []
        for unit in row_units:
            pred = prediction.get(unit.key)
            fid = fidelity.get(unit.key)
            if pred is None or fid is None or unit.key not in reached_valid:
                continue
            data = RQ2UnitData(pred, fid)
            metrics = _rq2_metrics(
                data, top_q=plan.top_q, tail_m=unit.tail_m
            )
            if metrics is not None:
                rq2_common_data.append(data)
                rq2_points.append((unit.request, unit.seed, metrics))

        protection_feasible: list[ProtectionUnitData] = []
        protection_common_data: list[ProtectionUnitData] = []
        protection_points: list[tuple[str, str, Mapping[str, float]]] = []
        for unit in row_units:
            data = protection.get(unit.key)
            if data is None or unit.key not in reached_valid:
                continue
            if data.feasible:
                protection_feasible.append(data)
                metrics = _protection_metrics(
                    data, data.candidates, cvar_q=plan.cvar_q
                )
                if metrics is not None:
                    protection_common_data.append(data)
                    protection_points.append((unit.request, unit.seed, metrics))

        prediction_evidence = _empty_rq1(
            selection_valid=prediction_selection.valid
            and not prediction_selection.fallback
        )
        if prediction_points:
            point = _equal_request_average(prediction_points)
            bootstrap = _hierarchical_bootstrap(
                prediction_common_data,
                replicates=plan.bootstrap_replicates,
                seed=plan.bootstrap_seed + row_index * 2,
                request_of=lambda data: data.key[2],
                seed_of=lambda data: data.key[3],
                resampled_metrics=lambda data, rng: _prediction_core_metrics(
                    _resample_groups_prediction(data.candidates, rng),
                    top_q=plan.top_q,
                    tail_m=plan.units[data.key].tail_m,
                    include_swap=row_has_swap,
                ),
            )
            tail_effect: dict[str, float] = {}
            if tail_points:
                tail_point = _equal_request_average(tail_points)
                tail_bootstrap = _hierarchical_bootstrap(
                    tail_common_data,
                    replicates=plan.bootstrap_replicates,
                    seed=plan.bootstrap_seed + row_index * 3 + 1,
                    request_of=lambda data: data.key[2],
                    seed_of=lambda data: data.key[3],
                    resampled_metrics=lambda data, rng: _prediction_tail_metrics(
                        _resample_groups_prediction(data.candidates, rng),
                        top_q=plan.top_q,
                        tail_m=plan.units[data.key].tail_m,
                    ),
                )
                tail_effect = summarize_bootstrap_effect(
                    tail_point["tail_lift"],
                    [draw["tail_lift"] for draw in tail_bootstrap],
                    beneficial="positive",
                    alpha=plan.alpha,
                )
                tail_details = {
                    key: tail_point[key]
                    for key in (
                        "chance_q",
                        "tail_recall_joint",
                        "tail_recall_s0",
                        "tail_recall_s1",
                    )
                }
            else:
                tail_details = {}
            swap_effect = {}
            if row_has_swap and bootstrap:
                swap_values = [draw["swap_delta"] for draw in bootstrap]
                swap_effect = {
                    "estimate": point["swap_delta"],
                    "lower_bound": percentile(swap_values, plan.alpha / 2.0),
                    "upper_bound": percentile(
                        swap_values, 1.0 - plan.alpha / 2.0
                    ),
                }
            prediction_evidence = {
                "paired": True,
                "selection_valid": prediction_selection.valid
                and not prediction_selection.fallback,
                "profile_valid": len(valid_profiles) == len(row_units),
                "common_support_units": len(prediction_common_data),
                "reached_valid_units": len(reached_valid),
                "tail_eligible_units": len(tail_common_data),
                "joint_rho": summarize_bootstrap_effect(
                    point["joint_rho"],
                    [draw["joint_rho"] for draw in bootstrap],
                    beneficial="positive",
                    alpha=plan.alpha,
                ),
                "joint_minus_s0": summarize_bootstrap_effect(
                    point["gain_s0"],
                    [draw["gain_s0"] for draw in bootstrap],
                    beneficial="positive",
                    alpha=plan.alpha,
                ),
                "joint_minus_s1": summarize_bootstrap_effect(
                    point["gain_s1"],
                    [draw["gain_s1"] for draw in bootstrap],
                    beneficial="positive",
                    alpha=plan.alpha,
                ),
                "tail_lift": tail_effect,
                **tail_details,
                "swap_delta": swap_effect,
            }

        rq2_evidence = _empty_rq2()
        if rq2_points:
            point = _equal_request_average(rq2_points)
            bootstrap = _hierarchical_bootstrap(
                rq2_common_data,
                replicates=plan.bootstrap_replicates,
                seed=plan.bootstrap_seed + row_index * 3 + 2,
                request_of=lambda data: data.key[2],
                seed_of=lambda data: data.key[3],
                resampled_metrics=lambda data, rng: _rq2_metrics(
                    data,
                    top_q=plan.top_q,
                    tail_m=plan.units[data.key].tail_m,
                    rng=rng,
                ),
            )
            rq2_evidence = {
                "paired": True,
                "perturbations_valid": all(
                    data.fidelity.perturbations_valid for data in rq2_common_data
                ),
                "exact_reference_valid": all(
                    data.fidelity.exact_reference_valid for data in rq2_common_data
                ),
                "common_control_support": all(
                    data.fidelity.common_control_support for data in rq2_common_data
                ),
                "common_support_units": len(rq2_common_data),
                **{
                    metric: summarize_bootstrap_effect(
                        point[metric],
                        [draw[metric] for draw in bootstrap],
                        beneficial="positive",
                        alpha=plan.alpha,
                    )
                    for metric in (
                        "f_rho_minus_0p80",
                        "f_k_minus_0p70",
                        "g_h",
                        "g_ctl",
                    )
                },
            }

        protection_evidence = _empty_rq3(
            selection_valid=protection_selection.valid
            and not protection_selection.fallback
        )
        if protection_points:
            point = _equal_request_average(protection_points)
            bootstrap = _hierarchical_bootstrap(
                protection_common_data,
                replicates=plan.bootstrap_replicates,
                seed=plan.bootstrap_seed + row_index * 2 + 1,
                request_of=lambda data: data.key[2],
                seed_of=lambda data: data.key[3],
                resampled_metrics=lambda data, rng: _protection_metrics(
                    data,
                    _resample_groups_protection(data, rng),
                    cvar_q=plan.cvar_q,
                    rng=rng,
                ),
            )
            comparisons: dict[str, dict[str, dict[str, float]]] = {}
            native_noninferiority: dict[str, dict[str, float]] = {}
            for comparator in PROTECTION_COMPARATORS:
                comparisons[comparator] = {}
                for outcome in METRIC_OUTCOMES:
                    metric = f"{comparator}.{outcome}"
                    comparisons[comparator][outcome] = summarize_bootstrap_effect(
                        point[metric],
                        [draw[metric] for draw in bootstrap],
                        beneficial="negative",
                        alpha=plan.alpha,
                    )
                native_metric = f"{comparator}.native_noninferiority"
                native_noninferiority[comparator] = summarize_bootstrap_effect(
                    point[native_metric],
                    [draw[native_metric] for draw in bootstrap],
                    beneficial="positive",
                    alpha=plan.alpha,
                )
            protection_evidence = {
                "paired": True,
                "comparisons": comparisons,
                "native_noninferiority": native_noninferiority,
                "selection_valid": protection_selection.valid
                and not protection_selection.fallback,
                "all_random_draws_complete": all(
                    data.all_random_draws_complete
                    for data in protection_common_data
                ),
                "all_five_arms_feasible": len(protection_feasible)
                == len(reached_valid),
                "common_support": len(protection_common_data)
                == len(reached_valid),
                "common_support_units": len(protection_common_data),
                "low_tail_support": any(
                    len(data.candidates) < 20 for data in protection_common_data
                ),
                "min_forget_margin": min(
                    data.min_forget_margin
                    for data in protection_common_data
                    if data.min_forget_margin is not None
                ),
                "min_utility_margin": min(
                    data.min_utility_margin
                    for data in protection_common_data
                    if data.min_utility_margin is not None
                ),
                "profile_mean": point["profile.mean"],
                "profile_cvar95": point["profile.cvar95"],
                "no_repair_mean": point["no_repair_absolute.mean"],
                "no_repair_cvar95": point["no_repair_absolute.cvar95"],
                "repair_updates": point["profile.repair_updates"],
                "repair_rollbacks": point["profile.repair_rollbacks"],
            }

        planned = len(row_units)
        row = {
            "setting": row_key[0],
            "parent": row_key[1],
            "attempted": bool(attempted_units),
            "completed": len(completed_units) == planned,
            "prediction_selection": _selection_mapping(prediction_selection),
            "protection_selection": _selection_mapping(protection_selection),
            "funnel": {
                "profiles_planned": planned,
                "profiles_valid": len(valid_profiles),
                "fidelity_planned": planned,
                "fidelity_valid": sum(
                    key in fidelity
                    and fidelity[key].perturbations_valid
                    and fidelity[key].exact_reference_valid
                    and fidelity[key].common_control_support
                    for key in (unit.key for unit in row_units)
                ),
                "trajectories_planned": planned,
                "trajectories_attempted": len(attempted_units),
                "trajectories_completed": len(completed_units),
                "trajectories_reached": len(reached_units),
                "reached_with_valid_profile": len(reached_valid),
                "prediction_common": len(prediction_common_data),
                "tail_eligible": len(tail_common_data),
                "fidelity_common": len(rq2_common_data),
                "protection_feasible_all_arms": len(protection_feasible),
                "protection_common": len(protection_common_data),
            },
            "rq1": prediction_evidence,
            "rq2": rq2_evidence,
            "rq3": protection_evidence,
        }
        rows.append(row)

    ledger: dict[str, Any] = {
        "schema_version": 2,
        "rows": rows,
        "artifacts": dict(artifacts or {}),
    }
    # Validate the exact object consumed by build_evidence.py before returning.
    EvidenceLedger.from_mapping(ledger)
    return ledger


def write_ledger(ledger: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and atomically write a normalized evidence ledger."""
    EvidenceLedger.from_mapping(ledger)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(
        json.dumps(ledger, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


def _pair_rate(groups: Sequence[str]) -> float | None:
    if len(groups) < 2:
        return None
    pairs = len(groups) * (len(groups) - 1) // 2
    same = 0
    for left in range(len(groups)):
        for right in range(left + 1, len(groups)):
            same += groups[left] == groups[right]
    return same / pairs


def _tail_metrics(
    candidates: Sequence[PredictionCandidate], *, q: float, cvar_q: float
) -> dict[str, float] | None:
    if len(candidates) < 2:
        return None
    ordered = sorted(candidates, key=lambda item: (-item.damage, item.candidate_id))
    tail_count = max(2, math.ceil(q * len(ordered)))
    tail = ordered[:tail_count]
    positive_total = sum(max(candidate.damage, 0.0) for candidate in ordered)
    if positive_total <= 0.0:
        return None
    mass = sum(max(candidate.damage, 0.0) for candidate in tail) / positive_total
    full_rate = _pair_rate([candidate.group for candidate in ordered])
    tail_rate = _pair_rate([candidate.group for candidate in tail])
    if full_rate is None or full_rate <= 0.0 or tail_rate is None:
        return None
    damage = [candidate.damage for candidate in ordered]
    return {
        "mean_damage": sum(damage) / len(damage),
        "cvar95": _tail_mean(damage, cvar_q=cvar_q),
        "mass_ratio": mass / q,
        "group_lift": tail_rate / full_rate,
        "tail_n": float(tail_count),
    }


def _tail_permutation_p(
    units: Sequence[PredictionUnitData],
    *,
    q: float,
    cvar_q: float,
    permutations: int,
    seed: int,
) -> float:
    points = []
    for data in units:
        metrics = _tail_metrics(data.candidates, q=q, cvar_q=cvar_q)
        if metrics is not None:
            points.append((data.key[2], data.key[3], metrics))
    observed = _equal_request_average(points)["group_lift"]
    rng = random.Random(seed)
    adverse = 0
    valid = 0
    attempts = 0
    max_attempts = max(permutations * 100, 1000)
    while valid < permutations and attempts < max_attempts:
        attempts += 1
        permuted_points = []
        okay = True
        for data in units:
            groups = [candidate.group for candidate in data.candidates]
            rng.shuffle(groups)
            permuted = [
                PredictionCandidate(
                    candidate_id=candidate.candidate_id,
                    group=groups[index],
                    s0=candidate.s0,
                    s1=candidate.s1,
                    joint=candidate.joint,
                    simple_control=candidate.simple_control,
                    damage=candidate.damage,
                )
                for index, candidate in enumerate(data.candidates)
            ]
            metrics = _tail_metrics(permuted, q=q, cvar_q=cvar_q)
            if metrics is None:
                okay = False
                break
            permuted_points.append((data.key[2], data.key[3], metrics))
        if okay:
            valid += 1
            value = _equal_request_average(permuted_points)["group_lift"]
            adverse += value >= observed
    if valid != permutations:
        raise EvidenceValidationError(
            "tail permutation could not produce enough finite group-lift draws"
        )
    return (1.0 + adverse) / (permutations + 1.0)


def _canonical_key(record: Mapping[str, Any], fields: Sequence[str], *, where: str) -> tuple[str, ...]:
    values: list[str] = []
    for field in fields:
        if field not in record:
            raise EvidenceValidationError(f"{where}.{field} is required")
        value = record[field]
        if isinstance(value, (Mapping, list)) or value is None:
            raise EvidenceValidationError(
                f"{where}.{field} must be a scalar string, number, or boolean"
            )
        values.append(json.dumps(value, sort_keys=True, ensure_ascii=False))
    return tuple(values)


_MEASUREMENT_REQUIRED_FIELDS: dict[str, set[str]] = {
    "reference_roster": {
        "rho_all",
        "rho_output",
        "rho_representation",
        "top_q_recall",
        "common_support",
    },
    "construction_checks": {
        "joint_rho",
        "min_gain",
        "min_lower_bound",
        "top_q_recall",
        "common_support",
        "eligible",
        "claim_pass",
    },
    "lse_fidelity_cost": {
        "rho_exact",
        "overlap_k",
        "split_half_rho",
        "perturbation_survival",
        "time_seconds",
        "peak_memory_bytes",
        "integrity_valid",
        "candidate_backward",
    },
    "protection_budget_sweep": {
        "worst_effect",
        "max_upper_bound",
        "bottleneck",
        "eligible",
        "claim_pass",
        "min_forget_margin",
        "min_utility_margin",
        "accepted_updates",
        "common_support",
        "random_draws_complete",
    },
    "specificity_negative_controls": {
        "rho_g",
        "rho_h",
        "rho_joint",
        "top_q_lift",
        "displacement_matched",
        "common_support",
    },
}


def _validate_measurement_value(value: object, kind: str, *, where: str) -> object:
    if kind == "number":
        if isinstance(value, bool):
            raise EvidenceValidationError(f"{where} must be numeric")
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise EvidenceValidationError(f"{where} must be numeric") from error
        if not math.isfinite(result):
            raise EvidenceValidationError(f"{where} must be finite")
        return result
    if kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise EvidenceValidationError(f"{where} must be an integer")
        return value
    if kind == "boolean":
        if type(value) is not bool:
            raise EvidenceValidationError(f"{where} must be a boolean")
        return value
    if kind == "text":
        if not isinstance(value, str) or not value.strip():
            raise EvidenceValidationError(f"{where} must be non-empty text")
        return value.strip()
    raise EvidenceValidationError(f"{where} has unsupported declared type {kind!r}")


def _aggregate_measurement(values: Sequence[object], operation: str) -> object:
    if not values:
        raise ValueError("measurement values must be non-empty")
    if operation == "mean":
        return sum(float(value) for value in values) / len(values)
    if operation == "median":
        ordered = sorted(float(value) for value in values)
        middle = len(ordered) // 2
        return (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2.0
        )
    if operation == "min":
        return min(values)
    if operation == "max":
        return max(values)
    if operation == "sum":
        return sum(float(value) for value in values)
    if operation == "range":
        return [min(values), max(values)]
    if operation == "count_true":
        return {"n": sum(value is True for value in values), "N": len(values)}
    if operation == "all_true":
        return all(value is True for value in values)
    if operation == "first":
        if any(value != values[0] for value in values[1:]):
            raise EvidenceValidationError(
                "measurement aggregation 'first' requires identical values"
            )
        return values[0]
    raise EvidenceValidationError(f"unsupported measurement aggregation {operation!r}")


def _measurement_artifact(
    artifact_id: str,
    contract: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    key_fields = contract.get("key_fields")
    group_by = contract.get("group_by")
    raw_metrics = contract.get("metrics")
    planned = contract.get("planned")
    if not isinstance(key_fields, list) or not key_fields or not all(
        isinstance(field, str) and field for field in key_fields
    ):
        raise EvidenceValidationError(
            f"artifact_contracts.{artifact_id}.key_fields must be non-empty text list"
        )
    if not isinstance(group_by, list) or not group_by or not set(group_by) <= set(key_fields):
        raise EvidenceValidationError(
            f"artifact_contracts.{artifact_id}.group_by must be a non-empty key subset"
        )
    if not isinstance(planned, list) or not planned:
        raise EvidenceValidationError(
            f"artifact_contracts.{artifact_id}.planned must be non-empty"
        )
    if not isinstance(raw_metrics, Mapping) or not raw_metrics:
        raise EvidenceValidationError(
            f"artifact_contracts.{artifact_id}.metrics must be non-empty"
        )
    missing_schema = _MEASUREMENT_REQUIRED_FIELDS[artifact_id] - set(raw_metrics)
    if missing_schema:
        raise EvidenceValidationError(
            f"artifact_contracts.{artifact_id}.metrics lacks table fields "
            f"{sorted(missing_schema)}"
        )
    metric_specs: dict[str, tuple[str, str]] = {}
    for metric, spec in raw_metrics.items():
        if not isinstance(spec, Mapping):
            raise EvidenceValidationError(
                f"artifact_contracts.{artifact_id}.metrics.{metric} must be a mapping"
            )
        value_type = spec.get("type")
        aggregate = spec.get("aggregate")
        if not isinstance(value_type, str) or not isinstance(aggregate, str):
            raise EvidenceValidationError(
                f"artifact_contracts.{artifact_id}.metrics.{metric} needs type/aggregate"
            )
        metric_specs[str(metric)] = value_type, aggregate

    expected: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for index, item in enumerate(planned):
        if not isinstance(item, Mapping):
            raise EvidenceValidationError(
                f"artifact_contracts.{artifact_id}.planned[{index}] must be a mapping"
            )
        key = _canonical_key(item, key_fields, where=f"{artifact_id}.planned[{index}]")
        if key in expected:
            raise EvidenceValidationError(
                f"artifact_contracts.{artifact_id} duplicates planned key {key!r}"
            )
        expected[key] = item

    observed: dict[tuple[str, ...], dict[str, object]] = {}
    for index, item in enumerate(records):
        where = f"{artifact_id}.raw[{index}]"
        key = _canonical_key(item, key_fields, where=where)
        if key not in expected:
            raise EvidenceValidationError(f"{where} has unplanned key {key!r}")
        if key in observed:
            raise EvidenceValidationError(f"{where} duplicates raw key {key!r}")
        observed[key] = {
            metric: _validate_measurement_value(
                item.get(metric), value_type, where=f"{where}.{metric}"
            )
            for metric, (value_type, _aggregate) in metric_specs.items()
        }

    grouped: dict[tuple[str, ...], list[tuple[Mapping[str, Any], Mapping[str, object]]]] = defaultdict(list)
    for key, values in observed.items():
        plan_item = expected[key]
        group_key = _canonical_key(plan_item, group_by, where=f"{artifact_id}.planned")
        grouped[group_key].append((plan_item, values))
    cells = []
    for group_key, items in sorted(grouped.items()):
        first_plan = items[0][0]
        cells.append(
            {
                "key": {field: first_plan[field] for field in group_by},
                "values": {
                    metric: _aggregate_measurement(
                        [values[metric] for _plan, values in items], operation
                    )
                    for metric, (_kind, operation) in metric_specs.items()
                },
                "source_keys": [
                    {field: plan_item[field] for field in key_fields}
                    for plan_item, _values in items
                ],
            }
        )
    missing = [
        {field: item[field] for field in key_fields}
        for key, item in expected.items()
        if key not in observed
    ]
    return {
        "schema_version": 1,
        "artifact_id": artifact_id,
        "kind": "measurements",
        "complete": not missing,
        "planned_units": len(expected),
        "observed_units": len(observed),
        "missing_units": missing,
        "group_by": list(group_by),
        "cells": cells,
    }


def _tail_artifact(
    plan: RawPlan,
    contract: Mapping[str, Any],
    prediction: Mapping[UnitKey, PredictionUnitData],
    supplementary_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    raw_rows = contract.get("rows")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise EvidenceValidationError(
            "artifact_contracts.tail_structure.rows must be non-empty"
        )
    q_raw = contract.get("q", plan.top_q)
    cvar_raw = contract.get("cvar_q", plan.cvar_q)
    try:
        q = float(q_raw)
        cvar_q = float(cvar_raw)
    except (TypeError, ValueError) as error:
        raise EvidenceValidationError("tail q/cvar_q must be numeric") from error
    if not 0.0 < q < 1.0 or not 0.0 < cvar_q < 1.0:
        raise EvidenceValidationError("tail q/cvar_q must be in (0, 1)")
    permutations = contract.get("permutations", plan.bootstrap_replicates)
    if isinstance(permutations, bool) or not isinstance(permutations, int) or permutations < 1:
        raise EvidenceValidationError("tail permutations must be a positive integer")

    planned_rows: list[RowKey] = []
    for index, raw_row in enumerate(raw_rows):
        if not isinstance(raw_row, Mapping):
            raise EvidenceValidationError(f"tail rows[{index}] must be a mapping")
        row_key = (
            _required_text(raw_row, "setting", where=f"tail.rows[{index}]"),
            _required_text(raw_row, "parent", where=f"tail.rows[{index}]"),
        )
        if row_key not in plan.row_keys:
            raise EvidenceValidationError(f"tail row {row_key!r} is not planned")
        if row_key in planned_rows:
            raise EvidenceValidationError(f"duplicate tail row {row_key!r}")
        planned_rows.append(row_key)

    output_rows = []
    artifact_complete = True
    for row_index, row_key in enumerate(planned_rows):
        units = sorted(
            (unit for unit in plan.units.values() if unit.row_key == row_key),
            key=lambda unit: (unit.request, unit.seed),
        )
        common: list[PredictionUnitData] = []
        points: list[tuple[str, str, Mapping[str, float]]] = []
        for unit in units:
            data = prediction.get(unit.key)
            if data is None or not data.trajectory_completed or not data.reached:
                continue
            metrics = _tail_metrics(data.candidates, q=q, cvar_q=cvar_q)
            if metrics is not None:
                common.append(data)
                points.append((unit.request, unit.seed, metrics))
        artifact_complete = artifact_complete and len(common) == len(units)
        if not points:
            output_rows.append(
                {
                    "setting": row_key[0],
                    "parent": row_key[1],
                    "support": {"n": 0, "N": len(units)},
                    "metrics": None,
                }
            )
            continue
        point = _equal_request_average(points)
        draws = _hierarchical_bootstrap(
            common,
            replicates=plan.bootstrap_replicates,
            seed=plan.bootstrap_seed + 10000 + row_index,
            request_of=lambda data: data.key[2],
            seed_of=lambda data: data.key[3],
            resampled_metrics=lambda data, rng: _tail_metrics(
                _resample_groups_prediction(data.candidates, rng),
                q=q,
                cvar_q=cvar_q,
            ),
        )
        output_rows.append(
            {
                "setting": row_key[0],
                "parent": row_key[1],
                "support": {"n": len(common), "N": len(units)},
                "tail_n": int(sum(draw[2]["tail_n"] for draw in points)),
                "mean_damage": point["mean_damage"],
                "cvar95": point["cvar95"],
                "mass_ratio": {
                    "estimate": point["mass_ratio"],
                    "lower_bound": sorted(draw["mass_ratio"] for draw in draws)[
                        max(0, math.floor(plan.alpha / 2 * len(draws)))
                    ],
                    "upper_bound": sorted(draw["mass_ratio"] for draw in draws)[
                        min(len(draws) - 1, math.ceil((1 - plan.alpha / 2) * len(draws)) - 1)
                    ],
                },
                "group_lift": {
                    "estimate": point["group_lift"],
                    "lower_bound": sorted(draw["group_lift"] for draw in draws)[
                        max(0, math.floor(plan.alpha / 2 * len(draws)))
                    ],
                    "upper_bound": sorted(draw["group_lift"] for draw in draws)[
                        min(len(draws) - 1, math.ceil((1 - plan.alpha / 2) * len(draws)) - 1)
                    ],
                },
                "permutation_p": _tail_permutation_p(
                    common,
                    q=q,
                    cvar_q=cvar_q,
                    permutations=permutations,
                    seed=plan.bootstrap_seed + 20000 + row_index,
                ),
            }
        )
    raw_blocks = contract.get("supplementary_blocks")
    if not isinstance(raw_blocks, Mapping):
        raise EvidenceValidationError(
            "tail_structure requires supplementary_blocks for reference_roster "
            "and construction_checks"
        )
    required_blocks = {"reference_roster", "construction_checks"}
    if set(raw_blocks) != required_blocks:
        raise EvidenceValidationError(
            "tail_structure.supplementary_blocks must contain exactly "
            "reference_roster and construction_checks"
        )
    records_by_block: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, record in enumerate(supplementary_records):
        block = _required_text(record, "block", where=f"tail_structure.raw[{index}]")
        if block not in required_blocks:
            raise EvidenceValidationError(
                f"tail_structure.raw[{index}] has unknown block {block!r}"
            )
        records_by_block[block].append(record)
    supplementary = {}
    for block in sorted(required_blocks):
        block_contract = raw_blocks[block]
        if not isinstance(block_contract, Mapping):
            raise EvidenceValidationError(
                f"tail_structure.supplementary_blocks.{block} must be a mapping"
            )
        supplementary[block] = _measurement_artifact(
            block,
            block_contract,
            records_by_block.get(block, ()),
        )
        artifact_complete = artifact_complete and supplementary[block]["complete"]

    return {
        "schema_version": 1,
        "artifact_id": "tail_structure",
        "kind": "tail_from_prediction",
        "complete": artifact_complete,
        "q": q,
        "cvar_q": cvar_q,
        "rows": output_rows,
        "supplementary": supplementary,
    }


def _artifact_headline(artifact_id: str, artifact: Mapping[str, Any]) -> str | None:
    if not artifact.get("complete"):
        return None
    if artifact_id == "tail_structure":
        rows = [row for row in artifact.get("rows", []) if row.get("metrics") is not None or "mass_ratio" in row]
        if not rows:
            return None
        mass = [float(row["mass_ratio"]["estimate"]) for row in rows]
        lift = [float(row["group_lift"]["estimate"]) for row in rows]
        support_n = sum(int(row["support"]["n"]) for row in rows)
        support_N = sum(int(row["support"]["N"]) for row in rows)
        return (
            "Across the predeclared parent rows, positive-damage tail mass "
            f"concentration ranged from {min(mass):.2f} to {max(mass):.2f} "
            f"times the diffuse reference and semantic-group lift from "
            f"{min(lift):.2f} to {max(lift):.2f} (support "
            f"{support_n}/{support_N})."
        )
    if artifact_id == "lse_fidelity_cost":
        return "The frozen loss-shake operating points passed the predeclared fidelity, perturbation, and integrity measurements reported in the appendix."
    return None


def build_raw_artifacts(
    plan: RawPlan,
    prediction_records: Sequence[Mapping[str, Any]],
    measurement_records: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    output_dir: str | Path,
) -> tuple[dict[str, dict[str, object]], dict[str, Path]]:
    """Produce contracted non-row artifacts and ledger status entries.

    Every artifact is backed by an explicit plan contract.  Missing contracted
    records still write an inspectable readiness artifact but receive
    ``completed: false`` in the ledger, so ``build_evidence.py`` cannot license
    its table or headline.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    parsed_prediction = _parse_prediction_records(plan, prediction_records)
    statuses: dict[str, dict[str, object]] = {}
    paths: dict[str, Path] = {}
    for artifact_id, contract in sorted(plan.artifact_contracts.items()):
        kind = contract["kind"]
        if kind == "plan_manifest":
            scope_rows = contract.get("scope_rows")
            feasibility_rows = contract.get("feasibility_rows")
            if not isinstance(scope_rows, list) or not scope_rows:
                raise EvidenceValidationError(
                    "campaign_manifest.scope_rows must be a non-empty list"
                )
            if not isinstance(feasibility_rows, list) or not feasibility_rows:
                raise EvidenceValidationError(
                    "campaign_manifest.feasibility_rows must be a non-empty list"
                )
            scope_required = {
                "setting",
                "dataset_role",
                "model_precision",
                "folds",
                "target_requests",
                "parents",
                "seeds",
                "candidate_counts",
                "planned_cells",
            }
            feasibility_required = {
                "audit",
                "metric",
                "direction",
                "boundary",
                "stopping_role",
                "reported_slack",
            }
            for index, row in enumerate(scope_rows):
                if not isinstance(row, Mapping) or not scope_required <= set(row):
                    raise EvidenceValidationError(
                        f"campaign_manifest.scope_rows[{index}] lacks physical-table columns"
                    )
            for index, row in enumerate(feasibility_rows):
                if not isinstance(row, Mapping) or not feasibility_required <= set(row):
                    raise EvidenceValidationError(
                        f"campaign_manifest.feasibility_rows[{index}] lacks physical-table columns"
                    )
            artifact = {
                "schema_version": 1,
                "artifact_id": artifact_id,
                "kind": kind,
                "complete": True,
                "plan_sha256": plan.source_sha256,
                "scope_rows": scope_rows,
                "feasibility_rows": feasibility_rows,
                "planned_units": [
                    {
                        "setting": unit.key[0],
                        "parent": unit.key[1],
                        "request": unit.key[2],
                        "seed": unit.key[3],
                        "prediction_selection": _selection_mapping(unit.prediction_selection),
                        "protection_selection": _selection_mapping(unit.protection_selection),
                        "simple_control_name": unit.simple_control_name,
                        "Kp": unit.tail_m,
                        "native_metric_name": unit.native_metric_name,
                        "repeated_random_draws": list(unit.repeated_random_draws),
                    }
                    for unit in sorted(plan.units.values(), key=lambda value: value.key)
                ],
            }
        elif kind == "tail_from_prediction":
            artifact = _tail_artifact(
                plan,
                contract,
                parsed_prediction,
                list(measurement_records.get(artifact_id, ())),
            )
        elif kind == "measurements":
            artifact = _measurement_artifact(
                artifact_id,
                contract,
                list(measurement_records.get(artifact_id, ())),
            )
        else:  # pragma: no cover - load-time validation owns this branch.
            raise EvidenceValidationError(f"unknown artifact kind {kind!r}")
        artifact["source_plan_sha256"] = plan.source_sha256
        path = destination / f"{artifact_id}.json"
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        complete = bool(artifact.get("complete"))
        status: dict[str, object] = {
            "completed": complete,
            "path": str(path.resolve()),
            "sha256": digest,
        }
        headline = _artifact_headline(artifact_id, artifact)
        if headline:
            status["headline_tex"] = headline
        statuses[artifact_id] = status
        paths[artifact_id] = path
    # Raw files for an undeclared artifact are an error rather than an ignored
    # result shard that could later be mistaken for a denominator.
    unknown = set(measurement_records) - set(plan.artifact_contracts)
    if unknown:
        raise EvidenceValidationError(
            f"raw measurement files supplied for undeclared artifacts {sorted(unknown)}"
        )
    return statuses, paths
