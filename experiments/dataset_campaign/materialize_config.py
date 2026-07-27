#!/usr/bin/env python3
"""Materialize one immutable dataset/model channel-matrix campaign config."""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = ROOT / "configs" / "dataset_campaign" / "matrix.yaml"


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one mapping")
    return value


def build_config(
    matrix: dict,
    *,
    dataset: str,
    scale: str,
    model_path: Path,
    output_root: Path,
) -> dict:
    if matrix.get("schema_version") != 1:
        raise ValueError("dataset campaign matrix must have schema_version: 1")
    try:
        dataset_cfg = deepcopy(matrix["datasets"][dataset])
    except KeyError as error:
        raise ValueError(
            f"unknown dataset {dataset!r}; available={sorted(matrix['datasets'])}"
        ) from error
    try:
        scale_cfg = deepcopy(matrix["scales"][scale])
    except KeyError as error:
        raise ValueError(
            f"unknown scale {scale!r}; available={sorted(matrix['scales'])}"
        ) from error

    common = deepcopy(matrix["common"])
    common.update(
        {
            "universe_authors": int(dataset_cfg["universe_groups"]),
            "candidate_author_pools": deepcopy(dataset_cfg["candidate_pools"]),
            "batch_size": int(scale_cfg["batch_size"]),
            "sft_steps": int(scale_cfg["sft_steps"]),
        }
    )
    if "pistol_sample" in dataset_cfg:
        common["pistol_sample"] = int(dataset_cfg["pistol_sample"])

    objective_grid = deepcopy(matrix["objective_grid"])
    objectives = list(objective_grid)
    freeze_path = output_root / "freeze" / "objective_freeze.yaml"
    return {
        "campaign_id": f"fdmu-dataset-{dataset}-{scale}-v1",
        "campaign_kind": "dataset_expansion_diagnostic",
        "dataset_label": str(dataset_cfg["label"]),
        "model_label": str(scale_cfg["model_label"]),
        "dataset": str(dataset_cfg["adapter"]),
        "output_root": str(output_root),
        "models": [
            {
                "id": str(scale_cfg["model_id"]),
                "path": str(model_path),
                "enabled": True,
            }
        ],
        "common": common,
        "fidelity": None,
        "calibration": {
            "authors": list(dataset_cfg["calibration_requests"]),
            "seeds": [int(matrix["audit_seeds"][0])],
            "selection": deepcopy(matrix["selection"]),
            "objective_grid": objective_grid,
        },
        "audit": {
            "offline": True,
            "authors": list(dataset_cfg["audit_requests"]),
            "seeds": list(matrix["audit_seeds"]),
            "objectives": objectives,
            "stress_objectives": [],
            "predictors": list(matrix["predictors"]),
            "objective_freeze": str(freeze_path),
        },
    }


def _write_immutable(path: Path, config: dict) -> str:
    rendered = yaml.safe_dump(config, sort_keys=False)
    if path.exists():
        current = path.read_text(encoding="utf-8")
        if current != rendered:
            raise RuntimeError(
                f"runtime config already exists with different content: {path}; "
                "use a new RUN_ROOT to protect existing results"
            )
        return "reused"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)
    return "created"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        matrix = _load(args.matrix.resolve())
        config = build_config(
            matrix,
            dataset=args.dataset,
            scale=args.scale,
            model_path=args.model_path.expanduser().resolve(),
            output_root=args.output_root.expanduser().resolve(),
        )
        state = _write_immutable(args.out.expanduser().resolve(), config)
        print(
            f"[CONFIG] {state} dataset={args.dataset} scale={args.scale} "
            f"path={args.out.expanduser().resolve()}"
        )
        return 0
    except (OSError, ValueError, RuntimeError, yaml.YAMLError) as error:
        print(
            f"dataset campaign config failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
