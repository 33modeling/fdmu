from pathlib import Path

from rsus.data.hf_cache import (
    find_cached_arrow_shards,
    writable_datasets_cache,
)


def _fingerprint(root: Path, name: str) -> Path:
    path = root / "datasets" / "locuslab___TOFU" / "full" / "0.0.0" / name
    path.mkdir(parents=True)
    (path / "dataset_info.json").write_text("{}", encoding="utf-8")
    return path


def test_cached_arrow_lookup_ignores_locks_and_filters_the_split(tmp_path):
    old = _fingerprint(tmp_path, "aaa")
    new = _fingerprint(tmp_path, "bbb")
    for path in (old, new):
        (path / "tofu-train.arrow").write_text("", encoding="utf-8")
        (path / "tofu-test.arrow").write_text("", encoding="utf-8")
        (path / "dataset_info.json.lock").write_text("", encoding="utf-8")
    (tmp_path / "datasets" / "shared-builder.lock").write_text("", encoding="utf-8")

    arrows = find_cached_arrow_shards(
        "locuslab___tofu",
        "full",
        "train",
        hf_home=tmp_path,
    )

    assert arrows == [new / "tofu-train.arrow"]
    assert all(path.suffix == ".arrow" for path in arrows)


def test_cached_arrow_lookup_returns_empty_for_missing_config(tmp_path):
    _fingerprint(tmp_path, "aaa")
    assert find_cached_arrow_shards(
        "locuslab___tofu",
        "forget10_perturbed",
        "train",
        hf_home=tmp_path,
    ) == []


def test_cached_arrow_lookup_honors_hf_datasets_cache(tmp_path, monkeypatch):
    new = _fingerprint(tmp_path, "bbb")
    (new / "tofu-train.arrow").write_text("", encoding="utf-8")
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "datasets"))

    assert find_cached_arrow_shards(
        "locuslab___tofu",
        "full",
        "train",
    ) == [new / "tofu-train.arrow"]


def test_writable_fallback_cache_honors_override(tmp_path, monkeypatch):
    configured = tmp_path / "owned-cache"
    monkeypatch.setenv("RSUS_DATASETS_CACHE", str(configured))
    assert writable_datasets_cache() == configured
    assert configured.is_dir()
