"""Deterministic paths for reusable request-level SFT checkpoints."""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Mapping


_DISABLED = {"off", "none", "disable", "disabled"}


def contract_sha256(contract: Mapping[str, object]) -> str:
    payload = json.dumps(
        dict(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _component(value: str, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
    return cleaned or fallback


def automatic_sft_cache_path(
    root: str | Path,
    *,
    model: str,
    model_id: str,
    dataset: str,
    request_id: str,
    contract: Mapping[str, object],
) -> Path:
    """Return a collision-resistant cache path for one exact SFT contract."""
    model_label = model_id.strip() or Path(model).name or "model"
    model_source = hashlib.sha256(model.encode("utf-8")).hexdigest()[:12]
    contract_digest = contract_sha256(contract)[:16]
    return (
        Path(root)
        / f"{_component(model_label, 'model')}-{model_source}"
        / _component(dataset, "dataset")
        / f"{_component(request_id, 'request')}__{contract_digest}.pt"
    )


def request_sft_cache_path(
    root: str | Path,
    *,
    setting: str,
    request_id: str,
    seed: str | int,
    contract: Mapping[str, object],
) -> Path:
    """Return the paper runner's exact-contract request/seed cache path."""
    contract_digest = contract_sha256(contract)[:16]
    request = _component(request_id, "request")
    seed_label = _component(str(seed), "seed")
    return (
        Path(root)
        / _component(setting, "setting")
        / f"{request}__seed-{seed_label}__{contract_digest}.pt"
    )


def sft_cache_pair_status(
    path: str | Path,
    contract: Mapping[str, object],
) -> str:
    """Inspect a cache pair without loading model weights.

    This is used only to adopt legacy fixed-name caches. New cache paths
    already contain the contract digest and remain fail-closed when corrupted.
    """
    cache_path = Path(path)
    meta_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not cache_path.exists() and not meta_path.exists():
        return "missing"
    if not cache_path.exists() or not meta_path.exists():
        return "incomplete"
    try:
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "invalid-metadata"
    if not isinstance(metadata, dict) or metadata.get("contract") != dict(contract):
        return "contract-mismatch"
    return "match"


def resolve_sft_cache_path(
    option: str | None,
    *,
    automatic_root: str | Path,
    model: str,
    model_id: str,
    dataset: str,
    request_id: str,
    contract: Mapping[str, object],
) -> Path | None:
    """Resolve ``auto``, an explicit path, or an explicit cache opt-out."""
    raw = (option or "auto").strip()
    if raw.lower() in _DISABLED:
        return None
    if raw.lower() == "auto":
        return automatic_sft_cache_path(
            automatic_root,
            model=model,
            model_id=model_id,
            dataset=dataset,
            request_id=request_id,
            contract=contract,
        ).resolve()
    return Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
