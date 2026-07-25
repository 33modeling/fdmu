"""Lock-free reads of prepared Hugging Face dataset Arrow caches."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile


def dataset_cache_roots(
    hf_home: str | Path | None = None,
) -> list[Path]:
    """Return every configured prepared-dataset cache root in priority order."""
    if hf_home is not None:
        return [Path(hf_home) / "datasets"]
    candidates: list[Path] = []
    configured = os.environ.get("HF_DATASETS_CACHE")
    if configured:
        candidates.append(
            Path(os.path.expandvars(os.path.expanduser(configured)))
        )
    home = Path(
        os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    candidates.append(home / "datasets")
    result: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in result:
            result.append(resolved)
    return result


def _split_arrows(fingerprint_dir: Path, split: str) -> list[Path]:
    return [
        arrow
        for arrow in sorted(fingerprint_dir.glob("*.arrow"))
        if arrow.stem.endswith(f"-{split}") or f"-{split}-" in arrow.stem
    ]


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
    expected = repo_cache_dir.casefold()
    for datasets_root in dataset_cache_roots(hf_home):
        candidates: list[tuple[int, str, list[Path]]] = []
        if not datasets_root.is_dir():
            continue
        repo_dirs = sorted(
            path
            for path in datasets_root.iterdir()
            if path.is_dir() and path.name.casefold() == expected
        )
        for repo_dir in repo_dirs:
            config_dir = repo_dir / config
            if not config_dir.is_dir():
                continue
            for info in config_dir.glob("*/*/dataset_info.json"):
                arrows = _split_arrows(info.parent, split)
                if arrows:
                    candidates.append(
                        (
                            info.stat().st_mtime_ns,
                            str(info.parent),
                            arrows,
                        )
                    )
        if candidates:
            return max(candidates, key=lambda item: (item[0], item[1]))[2]
    return []


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
