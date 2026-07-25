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
