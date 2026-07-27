"""MUSE adapter: knowmem forget_qa -> forget set, retain_qa -> candidate universe.

Layout (validated against ``muse-bench/MUSE-News`` and ``muse-bench/MUSE-Books``
``knowmem``, probe dump 2026-07-24): ``forget_qa`` and ``retain_qa`` are each 100
rows of ``{question, answer}``.

MUSE publishes one corpus-level forget/retain partition rather than independent
deletion requests.  ``muse_request`` keeps that original corpus request as its
backward-compatible default.  Dataset-expansion campaigns may additionally set
``request_index``: the forget split is then partitioned into deterministic,
non-overlapping chunks while retained QA are selected by deterministic group
indices.  These chunked requests are diagnostic request replications; they do
not relabel MUSE's corpus split as a benchmark-native request roster.

QA are tokenized with the shared TOFU ``Question:/Answer:`` layout (prompt
masked), so probes and trajectories need no MUSE-specific branch downstream.
"""
from __future__ import annotations

import os
from pathlib import Path

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.data.tofu import ANSWER_PREFIX, QUESTION_PREFIX, format_qa

QA_PER_SPLIT = 100
FORGET_CHUNK_SIZE = 10
CORPORA = {"news": "muse-bench/MUSE-News", "books": "muse-bench/MUSE-Books"}
CACHE_REPO_DIRS = {
    "news": "muse-bench___muse-news",
    "books": "muse-bench___muse-books",
}


def find_cached_arrows(
    corpus: str,
    split: str,
    hf_home: str | Path | None = None,
) -> list[Path]:
    """Return cached split shards without acquiring a datasets builder lock."""
    if corpus not in CORPORA:
        raise ValueError(f"unknown MUSE corpus {corpus!r}; expected one of {sorted(CORPORA)}")
    home = Path(
        hf_home
        or os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    datasets_root = home / "datasets"
    expected = CACHE_REPO_DIRS[corpus].casefold()
    repo_dirs = [
        path
        for path in datasets_root.glob("*")
        if path.is_dir() and path.name.casefold() == expected
    ]
    if not repo_dirs:
        return []
    config_dir = repo_dirs[0] / "knowmem"
    infos = sorted(config_dir.glob("*/*/dataset_info.json"))
    if not infos:
        return []
    arrows = sorted(infos[-1].parent.glob("*.arrow"))
    return [
        arrow
        for arrow in arrows
        if arrow.stem.endswith(f"-{split}") or f"-{split}-" in arrow.stem
    ]


def load_muse_knowmem(corpus: str) -> tuple[list[dict], list[dict]]:
    """Return (forget_qa, retain_qa) rows for one MUSE corpus (offline cache)."""
    if corpus not in CORPORA:
        raise ValueError(f"unknown MUSE corpus {corpus!r}; expected one of {sorted(CORPORA)}")
    forget_arrows = find_cached_arrows(corpus, "forget_qa")
    retain_arrows = find_cached_arrows(corpus, "retain_qa")
    if forget_arrows and retain_arrows:
        from datasets import Dataset

        forget = [
            row
            for arrow in forget_arrows
            for row in Dataset.from_file(str(arrow))
        ]
        retain = [
            row
            for arrow in retain_arrows
            for row in Dataset.from_file(str(arrow))
        ]
        return forget, retain

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
    request_index: int | None = None,
    candidate_groups: list[int] | None = None,
    forget_chunk_size: int = FORGET_CHUNK_SIZE,
    forget: list[dict] | None = None,
    retain: list[dict] | None = None,
) -> Request:
    """Build a corpus-level or deterministic chunked MUSE deletion request.

    ``retain_group_size`` chunks ``retain_qa`` into fold-granularity groups
    (default 10 -> ten groups of ten for 100 retain QA).  When
    ``request_index`` is supplied, ``candidate_groups`` must explicitly select
    retained groups and only one ``forget_chunk_size`` block is forgotten.
    """
    if corpus not in CORPORA:
        raise ValueError(f"unknown MUSE corpus {corpus!r}; expected one of {sorted(CORPORA)}")
    if retain_group_size < 1 or forget_chunk_size < 1:
        raise ValueError("retain_group_size and forget_chunk_size must be >= 1")
    if forget is None or retain is None:
        forget, retain = load_muse_knowmem(corpus)

    request_id = f"muse-{corpus}"
    forget_offset = 0
    selected_forget = list(forget)
    selected_retain = list(enumerate(retain))
    if request_index is not None:
        request_count = (len(forget) + forget_chunk_size - 1) // forget_chunk_size
        if not 0 <= request_index < request_count:
            raise ValueError(
                f"request_index {request_index} outside 0..{request_count - 1}"
            )
        if not candidate_groups:
            raise ValueError(
                "chunked MUSE requests require an explicit candidate_groups pool"
            )
        if len(set(candidate_groups)) != len(candidate_groups):
            raise ValueError("candidate_groups contains duplicates")
        retain_group_count = (
            len(retain) + retain_group_size - 1
        ) // retain_group_size
        invalid = [
            index
            for index in candidate_groups
            if not 0 <= index < retain_group_count
        ]
        if invalid:
            raise ValueError(
                f"candidate group indices outside 0..{retain_group_count - 1}: "
                f"{invalid}"
            )
        request_id = f"muse-{corpus}-r{request_index:03d}"
        forget_offset = request_index * forget_chunk_size
        selected_forget = list(
            forget[forget_offset : forget_offset + forget_chunk_size]
        )
        selected_group_set = set(candidate_groups)
        selected_retain = [
            (index, row)
            for index, row in enumerate(retain)
            if index // retain_group_size in selected_group_set
        ]

    forget_ex = [
        _qa_example(
            row,
            f"muse-{corpus}-f{forget_offset + idx:04d}",
            f"{request_id}-forget",
            tokenizer,
            max_length,
        )
        for idx, row in enumerate(selected_forget)
    ]
    retain_ex = [
        _qa_example(
            row,
            f"muse-{corpus}-r{idx:04d}",
            f"muse-{corpus}-rg{idx // retain_group_size:03d}",
            tokenizer,
            max_length,
        )
        for idx, row in selected_retain
    ]
    return Request.build(
        request_id=request_id,
        forget=forget_ex,
        universe=CandidateUniverse.freeze(retain_ex),
        native_audit_ids=(
            {example.example_id for example in retain_ex}
            if request_index is not None
            else frozenset()
        ),
    )
