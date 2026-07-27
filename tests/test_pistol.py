"""CPU-only tests for the official PISTOL edge-level adapter."""
from __future__ import annotations

from pathlib import Path

import pytest

from rsus.data.pistol import (
    edge_names,
    find_cached_arrows,
    pistol_request,
)
from rsus.data.registry import get_adapter


class MockTokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(char) % 50) for char in text[:80]]}


EDGES = ("A_B", "A_C", "B_D", "E_F")


def fake_rows() -> list[dict]:
    return [
        {
            "question": f"Question {edge} number {index}?",
            "answer": f"Answer {edge} {index}.",
            "edge": edge,
        }
        for edge in EDGES
        for index in range(20)
    ]


def test_pistol_request_preserves_edge_groups_and_native_neighbors():
    request = pistol_request(
        MockTokenizer(),
        request_index=0,
        candidate_edges=[1, 2, 3],
        rows=fake_rows(),
    )
    assert request.request_id == "pistol-s2-e000"
    assert len(request.forget) == 20
    assert len(request.universe) == 60
    assert {item.group for item in request.universe.examples} == {
        "pistol-s2-edge-001",
        "pistol-s2-edge-002",
        "pistol-s2-edge-003",
    }
    native_groups = {
        item.group
        for item in request.universe.examples
        if item.example_id in request.native_audit_ids
    }
    assert native_groups == {"pistol-s2-edge-001", "pistol-s2-edge-002"}
    assert edge_names(fake_rows()) == list(EDGES)


def test_pistol_request_rejects_bad_edge_pools():
    with pytest.raises(ValueError, match="appears in candidate_edges"):
        pistol_request(
            MockTokenizer(),
            request_index=0,
            candidate_edges=[0, 1],
            rows=fake_rows(),
        )
    with pytest.raises(ValueError, match="duplicates"):
        pistol_request(
            MockTokenizer(),
            request_index=0,
            candidate_edges=[1, 1],
            rows=fake_rows(),
        )
    with pytest.raises(ValueError, match="outside"):
        pistol_request(
            MockTokenizer(),
            request_index=0,
            candidate_edges=[4],
            rows=fake_rows(),
        )


def test_pistol_registry_and_roster_ids():
    adapter = get_adapter("xinchiqiu/PISTOL")
    assert adapter.key == "pistol"
    assert adapter.capabilities.independent_target_roster
    assert adapter.capabilities.native_audit
    assert adapter.accepts_roster_id("pistol-s1-e019")
    assert adapter.accepts_roster_id("pistol-s2-e074")
    assert not adapter.accepts_roster_id("pistol-s1-e020")
    assert not adapter.accepts_roster_id("pistol-s2-e075")
    request = adapter.build_request(
        tokenizer=MockTokenizer(),
        request_index=1,
        candidate_edges=[0, 2],
        rows=fake_rows(),
    )
    assert request.request_id == "pistol-s2-e001"


def test_pistol_cached_arrow_lookup_uses_newest_fingerprint(tmp_path):
    base = tmp_path / "datasets" / "xinchiqiu___pistol" / "pistol_data_2"
    old = base / "0.0.0" / "aaa"
    new = base / "0.0.0" / "bbb"
    for fingerprint in (old, new):
        fingerprint.mkdir(parents=True)
        (fingerprint / "dataset_info.json").write_text("{}", encoding="utf-8")
        (fingerprint / "pistol-train.arrow").write_text("", encoding="utf-8")
        (fingerprint / "pistol-test.arrow").write_text("", encoding="utf-8")
    assert find_cached_arrows("pistol_data_2", hf_home=tmp_path) == [
        new / "pistol-train.arrow"
    ]
