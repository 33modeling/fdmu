from rsus.sft_cache import (
    automatic_sft_cache_path,
    contract_sha256,
    resolve_sft_cache_path,
)


def _contract(**overrides):
    value = {
        "schema": "sft-cache-v2",
        "model": "/models/Qwen2.5-1.5B",
        "request": "tofu-a188",
        "candidate_universe_sha": "candidate-sha",
        "forget_sha": "forget-sha",
        "seed": 2025,
    }
    value.update(overrides)
    return value


def test_automatic_cache_path_reuses_only_the_exact_contract(tmp_path):
    first = automatic_sft_cache_path(
        tmp_path,
        model="/models/Qwen2.5-1.5B",
        model_id="qwen25/1p5b",
        dataset="tofu",
        request_id="tofu-a188",
        contract=_contract(),
    )
    repeated = automatic_sft_cache_path(
        tmp_path,
        model="/models/Qwen2.5-1.5B",
        model_id="qwen25/1p5b",
        dataset="tofu",
        request_id="tofu-a188",
        contract=_contract(),
    )
    changed = automatic_sft_cache_path(
        tmp_path,
        model="/models/Qwen2.5-1.5B",
        model_id="qwen25/1p5b",
        dataset="tofu",
        request_id="tofu-a188",
        contract=_contract(seed=2026),
    )

    assert first == repeated
    assert first != changed
    assert first.suffix == ".pt"
    assert "qwen25-1p5b" in first.parts


def test_cache_resolution_supports_auto_explicit_and_off(tmp_path, monkeypatch):
    contract = _contract()
    auto = resolve_sft_cache_path(
        "auto",
        automatic_root=tmp_path / "automatic",
        model="/models/Qwen2.5-1.5B",
        model_id="qwen25_1p5b",
        dataset="tofu",
        request_id="tofu-a188",
        contract=contract,
    )
    assert auto is not None
    assert auto.parent.parent.parent == (tmp_path / "automatic").resolve()

    monkeypatch.setenv("SFT_ROOT", str(tmp_path))
    explicit = resolve_sft_cache_path(
        "$SFT_ROOT/shared/theta0.pt",
        automatic_root=tmp_path / "unused",
        model="model",
        model_id="",
        dataset="tofu",
        request_id="request",
        contract=contract,
    )
    assert explicit == (tmp_path / "shared/theta0.pt").resolve()

    assert resolve_sft_cache_path(
        "off",
        automatic_root=tmp_path,
        model="model",
        model_id="",
        dataset="tofu",
        request_id="request",
        contract=contract,
    ) is None


def test_contract_hash_is_order_independent():
    left = _contract()
    right = dict(reversed(list(left.items())))
    assert contract_sha256(left) == contract_sha256(right)
