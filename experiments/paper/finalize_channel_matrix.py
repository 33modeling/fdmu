#!/usr/bin/env python3
"""Convert completed channel-matrix runs into PDF-v4 paper tables.

This is a CPU-only backfill path.  It never launches training and never mutates
the sealed campaign.  Available evidence is rendered cell by cell; unavailable
RQ2/RQ3 blocks remain explicit placeholders instead of blocking LaTeX output.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from experiments.paper.export_channel_matrix_raw import (  # noqa: E402
    export_fidelity_summary,
    export_prediction,
    export_protection,
)
from experiments.paper.publish_evidence import publish  # noqa: E402
from rsus.analysis.channels import DECLARED_CHANNEL  # noqa: E402
from rsus.evidence.decisions import evaluate_evidence  # noqa: E402
from rsus.evidence.raw import (  # noqa: E402
    aggregate_raw_evidence,
    raw_plan_from_mapping,
    read_raw_records,
    write_ledger,
)
from rsus.evidence.registry import load_contract  # noqa: E402
from rsus.evidence.rendering import write_readiness_json  # noqa: E402
from rsus.evidence.schemas import EvidenceLedger, EvidenceValidationError  # noqa: E402


PAPER_PARENTS = {
    "graddiff",
    "npo",
    "simnpo",
    "gru",
    "rmu",
    "repnoise",
    "circuit_breakers",
}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise EvidenceValidationError(f"cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"{path} must contain a mapping")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_obsolete_latex(output: Path, model_id: str) -> list[str]:
    """Remove duplicate LaTeX names emitted by the pre-unified finalizer."""
    tag = model_id.replace("/", "_")
    removed = []
    for path in (
        output / "table1.tex",
        output / "table2.tex",
        output / f"table1_core_evidence_{tag}.tex",
        output / f"table2_robustness_{tag}.tex",
    ):
        if path.exists() or path.is_symlink():
            path.unlink()
            removed.append(str(path))
    return removed


def _materialize_final_latex(
    output: Path, combined_outputs: Mapping[str, str]
) -> dict[str, str]:
    """Materialize shared final tables in the per-run result directory."""
    outputs = {}
    for key in ("table1", "table2"):
        target = Path(combined_outputs[key]).resolve()
        destination = output / target.name
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.tmp"
        )
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()
        shutil.copyfile(target, temporary)
        os.replace(temporary, destination)
        outputs[key] = str(destination)
    return outputs


def _request_id(dataset: str, author: int) -> str:
    if dataset == "rwku":
        return f"rwku-t{author:03d}"
    if dataset == "wmdp_bio_mmlu":
        return f"wmdp-r{author:03d}"
    return f"tofu-a{author}"


def _relative_config_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else config_path.parent / path


def _prediction_parents(
    cfg: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> tuple[list[str], dict[str, float]]:
    if freeze.get("status") != "frozen" or freeze.get("frozen_before_audit") is not True:
        raise EvidenceValidationError(
            "prediction alpha freeze must be frozen before audit"
        )
    raw_alphas = freeze.get("prediction_alpha")
    if not isinstance(raw_alphas, Mapping):
        raise EvidenceValidationError("prediction alpha freeze lacks prediction_alpha")
    audit = cfg.get("audit")
    if not isinstance(audit, Mapping):
        raise EvidenceValidationError("campaign config lacks audit")
    roster = list(audit.get("objectives", [])) + list(
        audit.get("stress_objectives", [])
    )
    alphas = {
        str(parent): float(alpha)
        for parent, alpha in raw_alphas.items()
        if str(parent) in PAPER_PARENTS
    }
    parents = [
        str(parent)
        for parent in roster
        if str(parent) in alphas and str(parent) in PAPER_PARENTS
    ]
    if not parents:
        raise EvidenceValidationError(
            "no paper parent has a frozen prediction alpha in this campaign"
        )
    return parents, alphas


def _prediction_roster(
    cfg: Mapping[str, Any],
    freeze: Mapping[str, Any],
) -> tuple[list[str], dict[str, dict[str, object]]]:
    """Return every paper parent with frozen or descriptive selection metadata."""
    frozen_parents, frozen_alphas = _prediction_parents(cfg, freeze)
    frozen_set = set(frozen_parents)
    raw_descriptive = freeze.get("descriptive_prediction_alpha")
    if not isinstance(raw_descriptive, Mapping):
        raise EvidenceValidationError(
            "prediction alpha freeze lacks descriptive_prediction_alpha"
        )
    descriptive_alphas = {
        str(parent): float(alpha)
        for parent, alpha in raw_descriptive.items()
        if str(parent) in PAPER_PARENTS
    }
    audit = cfg["audit"]
    roster = list(audit.get("objectives", [])) + list(
        audit.get("stress_objectives", [])
    )
    parents = []
    for parent in roster:
        parent = str(parent)
        if parent in PAPER_PARENTS and parent not in parents:
            parents.append(parent)
    missing = [
        parent
        for parent in parents
        if parent not in frozen_set and parent not in descriptive_alphas
    ]
    if missing:
        raise EvidenceValidationError(
            "paper parents lack frozen or descriptive prediction alpha: "
            f"{missing}"
        )
    selections = {
        parent: {
            "valid": parent in frozen_set,
            "fallback": parent not in frozen_set,
            "alpha": (
                frozen_alphas[parent]
                if parent in frozen_set
                else descriptive_alphas[parent]
            ),
        }
        for parent in parents
    }
    return parents, selections


def _protection_selections(
    cfg: Mapping[str, Any],
    config_path: Path,
    model_id: str,
    parents: list[str],
) -> tuple[dict[str, dict[str, object]], bool, Path | None]:
    phase = cfg.get("alpha_protection")
    if not isinstance(phase, Mapping):
        fallback = {
            parent: {"valid": False, "fallback": True, "alpha": 0.5}
            for parent in parents
        }
        return fallback, False, None
    freeze_name = phase.get("alpha_freeze")
    if not isinstance(freeze_name, str):
        raise EvidenceValidationError("alpha_protection.alpha_freeze is missing")
    freeze_path = _relative_config_path(config_path, freeze_name)
    freeze = _load_yaml(freeze_path)
    models = freeze.get("models")
    model_values = models.get(model_id) if isinstance(models, Mapping) else None
    frozen = (
        freeze.get("status") == "frozen"
        and freeze.get("frozen_before_alpha_audit") is True
        and isinstance(model_values, Mapping)
        and all(model_values.get(parent) is not None for parent in parents)
    )
    if frozen:
        return (
            {
                parent: {
                    "valid": True,
                    "fallback": False,
                    "alpha": float(model_values[parent]),
                }
                for parent in parents
            },
            True,
            freeze_path,
        )
    priors = phase.get("declared_prior", {})
    selections = {}
    for parent in parents:
        channel = DECLARED_CHANNEL[parent]
        alpha = float(priors.get(channel, 0.5)) if isinstance(priors, Mapping) else 0.5
        selections[parent] = {
            "valid": False,
            "fallback": True,
            "alpha": alpha,
        }
    return selections, False, freeze_path


def _build_plan(
    cfg: Mapping[str, Any],
    *,
    setting_id: str,
    parents: list[str],
    prediction_selections: Mapping[str, Mapping[str, object]],
    protection: Mapping[str, Mapping[str, object]],
    control: str,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    audit = cfg["audit"]
    phase = cfg.get("alpha_protection") or {}
    partition = phase.get("partition") or {}
    comparators = phase.get("comparators") or {}
    random_cfg = comparators.get("repeated_random") or {}
    b_rand = int(random_cfg.get("B_rand", 5))
    tail_m = int(partition.get("pool_size", 30))
    dataset = str(cfg.get("dataset", "tofu"))
    units = []
    for parent in parents:
        for author in audit["authors"]:
            for seed in audit["seeds"]:
                units.append(
                    {
                        "setting": setting_id,
                        "parent": parent,
                        "request": _request_id(dataset, int(author)),
                        "seed": str(seed),
                        "prediction_selection": dict(
                            prediction_selections[parent]
                        ),
                        "protection_selection": dict(protection[parent]),
                        "simple_control_name": control,
                        "repeated_random_draws": [
                            f"rand-{index:03d}" for index in range(b_rand)
                        ],
                        "tail_m": tail_m,
                        "native_metric_name": "retain_answer_token_recall",
                        "native_metric_orientation": "higher",
                        "native_noninferiority_margin": 0.02,
                    }
                )
    return {
        "schema_version": 2,
        "campaign_id": cfg.get("campaign_id"),
        "selection_freeze_id": "channel-matrix-backfill-from-committed-freezes",
        "bootstrap": {
            "replicates": bootstrap_replicates,
            "seed": 2027,
            "alpha": 0.05,
            "top_q": 0.10,
            "cvar_q": 0.95,
        },
        "units": units,
        "artifact_contracts": {},
    }


def _has_complete_protection_root(campaign_root: Path) -> bool:
    audit_root = campaign_root / "alpha_protection" / "audit"
    return audit_root.is_dir() and any(audit_root.rglob("results.json"))


def finalize(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.campaign_config.resolve()
    campaign_root = args.campaign_root.resolve()
    evidence_path = args.evidence_config.resolve()
    output = (
        args.out_dir.resolve()
        if args.out_dir
        else campaign_root / "aggregate" / "paper_v4"
    )
    raw_dir = output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_yaml(config_path)
    freeze = _load_yaml(args.prediction_alpha_freeze.resolve())
    parents, prediction_selections = _prediction_roster(cfg, freeze)
    protection, protection_frozen, alpha_freeze_path = _protection_selections(
        cfg, config_path, args.model_id, parents
    )
    plan_mapping = _build_plan(
        cfg,
        setting_id=args.setting_id,
        parents=parents,
        prediction_selections=prediction_selections,
        protection=protection,
        control=args.control_predictor,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    plan = raw_plan_from_mapping(plan_mapping)
    plan_path = output / "raw_plan.json"
    _atomic_json(plan_path, plan_mapping)

    export_args = SimpleNamespace(
        campaign_root=str(campaign_root),
        parents=parents,
        gradient_predictor=args.gradient_predictor,
        proximity_predictor=args.proximity_predictor,
        exact_predictor=args.exact_predictor,
        control_predictor=args.control_predictor,
        prediction_alpha=None,
        prediction_alpha_freeze=str(args.prediction_alpha_freeze.resolve()),
        prediction_selections=prediction_selections,
        setting_id=args.setting_id,
        native_metric_name="retain_answer_token_recall",
        top_q=0.10,
        fidelity_certificate=(
            str(args.fidelity_certificate.resolve())
            if args.fidelity_certificate
            else None
        ),
    )
    prediction_path = raw_dir / "prediction.jsonl"
    prediction_count, prediction_cells = export_prediction(
        export_args, cfg, prediction_path
    )

    protection_path = raw_dir / "protection.jsonl"
    protection_count = 0
    protection_cells = 0
    protection_available = (
        protection_frozen and _has_complete_protection_root(campaign_root)
    )
    if protection_available:
        protection_count, protection_cells = export_protection(
            export_args, cfg, protection_path
        )
    elif protection_path.exists():
        protection_path.unlink()

    fidelity: dict[str, dict[str, Any]] = {}
    fidelity_path: Path | None = None
    fidelity_source = args.fidelity_certificate
    if fidelity_source is None:
        candidate = campaign_root / "fidelity" / f"{args.model_id}.json"
        if candidate.is_file():
            fidelity_source = candidate
            export_args.fidelity_certificate = str(candidate)
    fidelity_warning: str | None = None
    if fidelity_source:
        fidelity_path = output / "fidelity_summary.json"
        try:
            export_fidelity_summary(export_args, cfg, fidelity_path)
            fidelity[args.setting_id] = json.loads(
                fidelity_path.read_text(encoding="utf-8")
            )
        except (EvidenceValidationError, OSError, ValueError, KeyError) as error:
            fidelity_warning = str(error)
            fidelity_path = None

    prediction_records = read_raw_records([prediction_path])
    protection_records = (
        read_raw_records([protection_path]) if protection_path.is_file() else []
    )
    ledger_mapping = aggregate_raw_evidence(
        plan,
        prediction_records,
        protection_records,
    )
    ledger_path = output / "evidence_ledger.json"
    write_ledger(ledger_mapping, ledger_path)
    ledger = EvidenceLedger.from_mapping(ledger_mapping)

    contract = load_contract(evidence_path)
    report = evaluate_evidence(contract, ledger, fidelity=fidelity)
    report["sources"] = {
        "campaign_config": str(config_path),
        "campaign_root": str(campaign_root),
        "raw_plan": str(plan_path),
        "ledger": str(ledger_path),
    }
    readiness_path = output / "evidence_readiness.json"
    write_readiness_json(report, readiness_path)

    removed_obsolete_latex = _remove_obsolete_latex(output, args.model_id)

    status = {
        "schema_version": 1,
        "status": "complete",
        "setting": args.setting_id,
        "model_id": args.model_id,
        "parents_exported": parents,
        "prediction_records": prediction_count,
        "prediction_cells": prediction_cells,
        "protection_records": protection_count,
        "protection_cells": protection_cells,
        "protection_complete": protection_available,
        "protection_freeze": str(alpha_freeze_path) if alpha_freeze_path else None,
        "fidelity_summary": str(fidelity_path) if fidelity_path else None,
        "fidelity_warning": fidelity_warning,
        "removed_obsolete_latex": removed_obsolete_latex,
        "outputs": {
            "ledger": str(ledger_path),
            "readiness": str(readiness_path),
        },
        "note": (
            "The shared Table 1 fixes all seven parent rows and Table 2 fixes "
            "all registered settings. Missing evidence is an explicit placeholder."
        ),
    }
    combined_root = (
        args.combined_root.resolve()
        if getattr(args, "combined_root", None)
        else campaign_root.parent / "paper_v4"
    )
    combined_outputs = publish(
        ledger_path=ledger_path,
        combined_root=combined_root,
        evidence_config=evidence_path,
        fidelity_input=fidelity_path,
    )
    status["combined_outputs"] = combined_outputs
    status["outputs"].update(_materialize_final_latex(output, combined_outputs))
    _atomic_json(output / "FINALIZATION_STATUS.json", status)
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign-config", type=Path, required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--setting-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prediction-alpha-freeze", type=Path, required=True)
    parser.add_argument(
        "--evidence-config",
        type=Path,
        default=ROOT / "configs/paper/evidence.yaml",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--combined-root",
        type=Path,
        default=None,
        help=(
            "shared cross-setting ledger/table directory; defaults to "
            "<campaign-root parent>/paper_v4"
        ),
    )
    parser.add_argument("--gradient-predictor", default="fd_norm")
    parser.add_argument("--proximity-predictor", default="knn_feature")
    parser.add_argument("--exact-predictor", default="grad_norm")
    parser.add_argument("--control-predictor", default="knn_lexical")
    parser.add_argument("--fidelity-certificate", type=Path, default=None)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        status = finalize(args)
        print(
            "[RESULT] PDF-v4 Table 1: "
            f"{status['outputs']['table1']}",
            flush=True,
        )
        print(
            "[RESULT] PDF-v4 Table 2: "
            f"{status['outputs']['table2']}",
            flush=True,
        )
        if not status["protection_complete"]:
            print(
                "[RESULT] RQ3 protection evidence is incomplete; "
                "Table 1 keeps all parent rows and marks missing cells explicitly.",
                flush=True,
            )
        return 0
    except (EvidenceValidationError, OSError, ValueError, KeyError) as error:
        print(f"paper-v4 finalization failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
