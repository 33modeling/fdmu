"""Lock-free reads of prepared Hugging Face dataset Arrow caches."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile


def find_cached_arrow_shards(
    repo_cache_dir: str,
    config: str,
    split: str,
    *,
    hf_home: str | Path | None = None,
) -> list[Path]:
    """Return one cached config fingerprint's Arrow shards for ``split``.

    Only prepared ``dataset_info.json`` and ``*.arrow`` files are considered.
    Builder ``*.lock`` files in the shared cache are never opened.
    """
    if hf_home is not None:
        datasets_root = Path(hf_home) / "datasets"
    elif os.environ.get("HF_DATASETS_CACHE"):
        datasets_root = Path(
            os.path.expandvars(os.path.expanduser(os.environ["HF_DATASETS_CACHE"]))
        )
    else:
        home = Path(
            os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
        )
        datasets_root = home / "datasets"
    if not datasets_root.is_dir():
        return []

    expected = repo_cache_dir.casefold()
    repo_dirs = sorted(
        path
        for path in datasets_root.iterdir()
        if path.is_dir() and path.name.casefold() == expected
    )
    infos: list[Path] = []
    for repo_dir in repo_dirs:
        config_dir = repo_dir / config
        if config_dir.is_dir():
            infos.extend(config_dir.glob("*/*/dataset_info.json"))
    if not infos:
        return []

    fingerprint_dir = sorted(infos)[-1].parent
    arrows = sorted(fingerprint_dir.glob("*.arrow"))
    return [
        arrow
        for arrow in arrows
        if arrow.stem.endswith(f"-{split}") or f"-{split}-" in arrow.stem
    ]


def writable_datasets_cache() -> Path:
    """Return a process-user writable fallback for dataset preparation."""
    configured = os.environ.get("RSUS_DATASETS_CACHE")
    if configured:
        root = Path(os.path.expandvars(os.path.expanduser(configured)))
    else:
        root = Path(os.environ.get("TMPDIR", tempfile.gettempdir()))
        root = root / f"rsus-datasets-{os.getuid()}"
    root.mkdir(parents=True, exist_ok=True)
    return root
