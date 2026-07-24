"""MUSE adapter: knowmem forget_qa -> forget set, retain_qa -> candidate universe.

Layout (validated against ``muse-bench/MUSE-News`` and ``muse-bench/MUSE-Books``
``knowmem``, probe dump 2026-07-24): ``forget_qa`` and ``retain_qa`` are each 100
rows of ``{question, answer}``.

Unlike TOFU (one request per forget10 author) and RWKU (one request per
forget_target), MUSE's knowmem split is corpus-level: a single forget/retain QA
partition. We therefore model the *whole* ``forget_qa`` set as one deletion
request and use ``retain_qa`` as the frozen retained-candidate universe (the
knowledge that must survive). ``knowmem`` carries no finer grouping, so retain
candidates are chunked into synthetic equal-size groups purely to give the
discovery/audit fold split a granularity unit (mirrors TOFU's author groups).

QA are tokenized with the shared TOFU ``Question:/Answer:`` layout (prompt
masked), so probes and trajectories need no MUSE-specific branch downstream.
"""
from __future__ import annotations

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.data.tofu import ANSWER_PREFIX, QUESTION_PREFIX, format_qa

QA_PER_SPLIT = 100
CORPORA = {"news": "muse-bench/MUSE-News", "books": "muse-bench/MUSE-Books"}


def load_muse_knowmem(corpus: str) -> tuple[list[dict], list[dict]]:
    """Return (forget_qa, retain_qa) rows for one MUSE corpus (offline cache)."""
    if corpus not in CORPORA:
        raise ValueError(f"unknown MUSE corpus {corpus!r}; expected one of {sorted(CORPORA)}")
    from datasets import load_dataset

    ds = load_dataset(CORPORA[corpus], "knowmem")
    forget, retain = list(ds["forget_qa"]), list(ds["retain_qa"])
    for name, rows in (("forget_qa", forget), ("retain_qa", retain)):
        if not rows:
            raise ValueError(f"MUSE {corpus} knowmem {name} is empty")
    return forget, retain


def _qa_example(row: dict, example_id: str, group: str, tokenizer, max_length: int) -> Example:
    q, a = str(row["question"]), str(row["answer"])
    ids, labels = format_qa(q, a, tokenizer, max_length)
    return Example(
        example_id=example_id,
        input_ids=ids,
        labels=labels,
        group=group,
        text=f"{QUESTION_PREFIX}{q}{ANSWER_PREFIX} {a}",
    )


def muse_request(
    tokenizer,
    corpus: str = "news",
    max_length: int = 256,
    retain_group_size: int = 10,
    forget: list[dict] | None = None,
    retain: list[dict] | None = None,
) -> Request:
    """Build the single deletion request for one MUSE corpus.

    ``retain_group_size`` chunks ``retain_qa`` into fold-granularity groups
    (default 10 -> ten groups of ten for 100 retain QA).
    """
    if corpus not in CORPORA:
        raise ValueError(f"unknown MUSE corpus {corpus!r}; expected one of {sorted(CORPORA)}")
    if retain_group_size < 1:
        raise ValueError("retain_group_size must be >= 1")
    if forget is None or retain is None:
        forget, retain = load_muse_knowmem(corpus)

    forget_ex = [
        _qa_example(row, f"muse-{corpus}-f{idx:04d}", f"muse-{corpus}-forget", tokenizer, max_length)
        for idx, row in enumerate(forget)
    ]
    retain_ex = [
        _qa_example(row, f"muse-{corpus}-r{idx:04d}",
                    f"muse-{corpus}-rg{idx // retain_group_size:03d}", tokenizer, max_length)
        for idx, row in enumerate(retain)
    ]
    return Request.build(
        request_id=f"muse-{corpus}",
        forget=forget_ex,
        universe=CandidateUniverse.freeze(retain_ex),
    )
