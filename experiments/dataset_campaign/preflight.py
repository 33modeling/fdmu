#!/usr/bin/env python3
"""Validate one generated dataset campaign before allocating model weights."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from rsus.data.indexed import build_indexed_request  # noqa: E402
from rsus.data.registry import get_adapter  # noqa: E402


def _load(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain one mapping")
    return value


def _campaign_module():
    path = ROOT / "experiments" / "channel_matrix" / "run_campaign.py"
    spec = importlib.util.spec_from_file_location("dataset_preflight_campaign", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load campaign module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dataset_tables(dataset: str):
    if dataset == "rwku":
        from rsus.data.rwku import load_rwku_tables

        return load_rwku_tables()
    if dataset == "wmdp_bio_mmlu":
        from rsus.data.wmdp import load_wmdp_tables

        return load_wmdp_tables()
    if dataset in {"muse_news", "muse_books"}:
        from rsus.data.muse import load_muse_knowmem

        return load_muse_knowmem(
            "news" if dataset == "muse_news" else "books"
        )
    if dataset == "pistol":
        from rsus.data.pistol import load_pistol_rows

        return load_pistol_rows()
    return None


def _build_requests(config: dict, tokenizer) -> None:
    campaign = _campaign_module()
    dataset = str(config["dataset"])
    common = config["common"]
    tables = _dataset_tables(dataset)
    seen: set[str] = set()
    for phase_name in ("calibration", "audit"):
        phase = config[phase_name]
        pool_spec = common["candidate_author_pools"][phase_name]
        for request_index in phase["authors"]:
            raw_pool = (
                pool_spec[str(request_index)]
                if isinstance(pool_spec, dict)
                else pool_spec
            )
            candidates = sorted(campaign._expand_int_ranges(str(raw_pool)))
            request = build_indexed_request(
                dataset,
                tokenizer,
                int(request_index),
                candidates,
                universe_groups=int(common["universe_authors"]),
                seed=int(phase["seeds"][0]),
                pistol_sample=int(common.get("pistol_sample", 2)),
                tables=tables,
            )
            if request.request_id in seen:
                raise ValueError(f"duplicate request id: {request.request_id}")
            seen.add(request.request_id)
            print(
                f"[DATA] phase={phase_name} request={request.request_id} "
                f"forget={len(request.forget)} candidates={len(request.universe)} "
                f"groups={len({item.group for item in request.universe.examples})} "
                f"native_audit={len(request.native_audit_ids)}"
            )


def validate(config: dict, *, load_data: bool) -> None:
    campaign = _campaign_module()
    campaign._validate_campaign(config)
    if config.get("campaign_kind") != "dataset_expansion_diagnostic":
        raise ValueError("generated config has the wrong campaign_kind")
    if config.get("fidelity") is not None:
        raise ValueError("dataset expansion must not require a fidelity certificate")
    if config["audit"].get("fidelity_certificates"):
        raise ValueError("dataset expansion audit must not declare fidelity certificates")
    predictors = set(config["audit"]["predictors"])
    if "knn_embed" in predictors or config["common"].get("sentence_encoder"):
        raise ValueError("dataset expansion must not depend on an external sentence encoder")
    adapter = get_adapter(str(config["dataset"]))
    if not adapter.capabilities.grouped_candidates:
        raise ValueError(f"adapter {adapter.key} lacks grouped candidates")

    models = [model for model in config["models"] if model.get("enabled", True)]
    if len(models) != 1:
        raise ValueError("dataset expansion config must enable exactly one model")
    model_path = Path(models[0]["path"])
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory is missing: {model_path}")
    print(
        f"[MODEL] id={models[0]['id']} path={model_path} "
        f"dtype={config['common']['dtype']}"
    )
    if load_data:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        _build_requests(config, tokenizer)
    print(
        f"[PREFLIGHT] PASS campaign={config['campaign_id']} "
        f"python={sys.executable} adapter={adapter.key}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--skip-data", action="store_true")
    args = parser.parse_args(argv)
    try:
        validate(_load(args.config.resolve()), load_data=not args.skip_data)
        return 0
    except Exception as error:  # noqa: BLE001 - preflight prints the real class
        print(
            f"dataset campaign preflight failed: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
