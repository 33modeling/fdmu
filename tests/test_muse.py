"""CPU-only tests for the MUSE adapter (knowmem forget_qa/retain_qa schema)."""
from __future__ import annotations

import pytest

from rsus.data.base import Request
from rsus.data.muse import CORPORA, muse_request
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
