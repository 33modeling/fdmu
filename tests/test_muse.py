"""CPU-only tests for the MUSE adapter (knowmem forget_qa/retain_qa schema)."""
from __future__ import annotations

import pytest

from rsus.data.base import Request
from rsus.data.muse import CORPORA, find_cached_arrows, muse_request
from rsus.data.registry import get_adapter
from rsus.losses import IGNORE


class MockTokenizer:
    eos_token_id = 9

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [10 + (ord(c) % 50) for c in text[:40]]}


def fake_qa(n: int, tag: str) -> list[dict]:
    return [{"question": f"{tag} question number {i}?", "answer": f"{tag}-answer-{i}"}
            for i in range(n)]


def test_muse_request_maps_forget_and_retain():
    tok = MockTokenizer()
    forget, retain = fake_qa(8, "F"), fake_qa(20, "R")
    req = muse_request(tok, corpus="books", forget=forget, retain=retain, retain_group_size=5)

    assert isinstance(req, Request)
    assert req.request_id == "muse-books"
    assert len(req.forget) == 8
    assert len(req.universe) == 20
    # forget ids and retain ids are disjoint (contract of Request.build)
    fids = {e.example_id for e in req.forget}
    rids = {e.example_id for e in req.universe.examples}
    assert not (fids & rids)
    # MUSE has no benchmark-native audit set
    assert req.native_audit_ids == frozenset()


def test_muse_retain_groups_are_chunked():
    tok = MockTokenizer()
    req = muse_request(tok, corpus="news", forget=fake_qa(4, "F"),
                       retain=fake_qa(20, "R"), retain_group_size=5)
    groups = [e.group for e in req.universe.examples]
    # 20 retain / group_size 5 -> 4 distinct groups, each seen 5 times
    assert len(set(groups)) == 4
    assert all(groups.count(g) == 5 for g in set(groups))
    # forget lives in a single request-level group
    assert {e.group for e in req.forget} == {"muse-news-forget"}


def test_muse_prompt_is_masked():
    tok = MockTokenizer()
    req = muse_request(tok, corpus="books", forget=fake_qa(2, "F"),
                       retain=fake_qa(5, "R"), retain_group_size=5)
    ex = req.forget[0]
    # some prompt positions masked, some answer positions supervised
    assert (ex.labels == IGNORE).any()
    assert (ex.labels != IGNORE).any()
    assert ex.input_ids.shape == ex.labels.shape


def test_muse_rejects_unknown_corpus():
    with pytest.raises(ValueError):
        muse_request(MockTokenizer(), corpus="wikipedia",
                     forget=fake_qa(2, "F"), retain=fake_qa(2, "R"))
    assert set(CORPORA) == {"news", "books"}


def test_muse_cached_arrow_lookup_is_case_insensitive_and_split_specific(tmp_path):
    fingerprint = (
        tmp_path
        / "datasets"
        / "muse-bench___MUSE-News"
        / "knowmem"
        / "0.0.0"
        / "fingerprint"
    )
    fingerprint.mkdir(parents=True)
    (fingerprint / "dataset_info.json").write_text("{}", encoding="utf-8")
    forget = fingerprint / "muse-news-forget_qa.arrow"
    retain = fingerprint / "muse-news-retain_qa.arrow"
    forget.write_text("", encoding="utf-8")
    retain.write_text("", encoding="utf-8")
    assert find_cached_arrows("news", "forget_qa", tmp_path) == [forget]
    assert find_cached_arrows("news", "retain_qa", tmp_path) == [retain]


def test_muse_chunk_request_has_disjoint_forget_and_candidate_groups():
    req = muse_request(
        MockTokenizer(),
        corpus="news",
        forget=fake_qa(30, "F"),
        retain=fake_qa(50, "R"),
        request_index=2,
        candidate_groups=[0, 3],
        forget_chunk_size=10,
        retain_group_size=10,
    )
    assert req.request_id == "muse-news-r002"
    assert len(req.forget) == 10
    assert {item.example_id for item in req.forget} == {
        f"muse-news-f{index:04d}" for index in range(20, 30)
    }
    assert len(req.universe) == 20
    assert {item.group for item in req.universe.examples} == {
        "muse-news-rg000",
        "muse-news-rg003",
    }
    assert req.native_audit_ids == {
        item.example_id for item in req.universe.examples
    }


def test_muse_chunk_request_rejects_invalid_or_implicit_candidate_pool():
    values = {
        "forget": fake_qa(20, "F"),
        "retain": fake_qa(20, "R"),
        "retain_group_size": 10,
    }
    with pytest.raises(ValueError, match="candidate_groups"):
        muse_request(MockTokenizer(), corpus="books", request_index=0, **values)
    with pytest.raises(ValueError, match="outside"):
        muse_request(
            MockTokenizer(),
            corpus="books",
            request_index=0,
            candidate_groups=[2],
            **values,
        )


@pytest.mark.parametrize(
    ("name", "request_id"),
    (("MUSE-News", "muse-news-r009"), ("MUSE-Books", "muse-books-r009")),
)
def test_muse_registry_exposes_deterministic_chunk_requests(name, request_id):
    adapter = get_adapter(name)
    assert adapter.accepts_roster_id(request_id)
    assert adapter.capabilities.independent_target_roster
    assert adapter.capabilities.native_audit
