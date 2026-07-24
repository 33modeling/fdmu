#!/usr/bin/env python3
"""Command-line entrypoint used by ``local_run/local_run.sh``."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rsus.blocks import BlockSpec  # noqa: E402
from rsus.data.registry import get_adapter  # noqa: E402
from rsus.local_pdf_v4 import (  # noqa: E402
    LocalPDFV4Error,
    block_identity,
    load_yaml_mapping,
    prepare_manifest,
    run_local_pdf_v4,
    validate_run_config,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail-closed local runner for KDD_UnlearningFail PDF v4"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--action",
        choices=("prepare-manifest", "inspect-model", "validate", "run"),
        default="run",
    )
    parser.add_argument("--force", action="store_true", help="overwrite a prepared manifest")
    return parser.parse_args()


def resolve(path: str, *, base: Path = ROOT) -> Path:
    candidate = Path(os.path.expandvars(os.path.expanduser(path)))
    return candidate if candidate.is_absolute() else base / candidate


def load_tokenizer(model_cfg: dict):
    from transformers import AutoTokenizer

    kwargs = {}
    revision = model_cfg.get("tokenizer_revision")
    if revision:
        kwargs["revision"] = revision
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["tokenizer_source"], **kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise LocalPDFV4Error("tokenizer has neither PAD nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_request(config: dict, tokenizer):
    data_cfg = config["data"]
    kwargs = dict(data_cfg["request"])
    if data_cfg["adapter"] in {"tofu", "rwku"}:
        kwargs["tokenizer"] = tokenizer
    return get_adapter(data_cfg["adapter"]).build_request(**kwargs)


def load_model(model_cfg: dict):
    from transformers import AutoModelForCausalLM

    kwargs = {"torch_dtype": torch.float32}
    revision = model_cfg.get("revision")
    if revision:
        kwargs["revision"] = revision
    if model_cfg.get("attention_implementation"):
        kwargs["attn_implementation"] = model_cfg["attention_implementation"]
    if model_cfg.get("device_map"):
        kwargs["device_map"] = model_cfg["device_map"]
    model = AutoModelForCausalLM.from_pretrained(model_cfg["source"], **kwargs)
    if not model_cfg.get("device_map"):
        model = model.to(model_cfg["device"])
    return model.eval()


def main() -> int:
    args = parse_args()
    config_path = resolve(args.config, base=Path.cwd())
    config = load_yaml_mapping(config_path)
    model_cfg = config.get("model", {})

    if args.action == "prepare-manifest":
        tokenizer = load_tokenizer(model_cfg)
        request = build_request(config, tokenizer)
        payload = prepare_manifest(request, config["data"]["split"])
        destination = resolve(config["data"]["manifest"]["path"])
        if destination.exists() and not args.force:
            raise LocalPDFV4Error(
                f"refusing to overwrite frozen manifest {destination}; pass --force explicitly"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
        print(f"prepared score-independent manifest: {destination}")
        print(f"content_sha256={payload['content_sha256']}")
        return 0

    if args.action == "inspect-model":
        model = load_model(model_cfg)
        identity = block_identity(model, BlockSpec(model_cfg["block"]["parameter_regex"]))
        print(json.dumps(identity, indent=2))
        return 0

    validate_run_config(config)
    manifest_path = resolve(config["data"]["manifest"]["path"])
    manifest = load_yaml_mapping(manifest_path)
    if args.action == "validate":
        # Request hashes and stream identities are checked too; no model or
        # susceptibility score is loaded/computed.
        from rsus.local_pdf_v4 import validate_manifest

        tokenizer = load_tokenizer(model_cfg)
        request = build_request(config, tokenizer)
        validate_manifest(request, manifest)
        print("PDF v4 local config and frozen manifest are valid")
        return 0

    output_dir = resolve(config["output"]["directory"])
    if output_dir.exists() and any(output_dir.iterdir()) and not bool(
        config["output"].get("overwrite", False)
    ):
        raise LocalPDFV4Error(
            f"refusing to overwrite non-empty output directory {output_dir}; "
            "choose a new directory or set output.overwrite: true"
        )

    seed = int(config["run"]["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if bool(config["run"].get("deterministic_algorithms", True)):
        torch.use_deterministic_algorithms(True)

    tokenizer = load_tokenizer(model_cfg)
    request = build_request(config, tokenizer)
    model = load_model(model_cfg)
    payload, final_block = run_local_pdf_v4(config, request, manifest, model)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", payload)
    write_json(output_dir / "resolved_config.json", config)
    if final_block is not None and bool(config["output"].get("save_final_block", True)):
        torch.save(final_block, output_dir / "final_block.pt")
    (output_dir / "DONE").write_text(payload["status"] + "\n", encoding="utf-8")
    print(f"result: {output_dir / 'result.json'}")
    return 0 if payload["status"] == "completed_local_diagnostic" else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (LocalPDFV4Error, FileNotFoundError, KeyError, TypeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
