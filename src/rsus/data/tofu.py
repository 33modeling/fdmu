"""TOFU forget10 adapter: one deletion request per forget10 author.

Layout (validated against ``locuslab/TOFU`` ``full``): 4,000 QA rows, 200
authors x 20 QA in contiguous blocks of 20; forget10 = authors 180-199.
example_id = "tofu-<row:04d>", group = "author-<id:03d>" (fold granularity),
text = raw QA string for external-encoder baselines.

The benchmark-native audit rule is a preregistration choice that is still
open (paper: 'metadata rule'); until frozen, requests carry an empty native
set and the gate experiment audits on the untouched random fold plus the
complete candidate distribution.
"""
from __future__ import annotations

from pathlib import Path

import torch

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.data.hf_cache import find_cached_arrow_shards, writable_datasets_cache
from rsus.losses import IGNORE

QA_PER_AUTHOR = 20
AUTHORS_TOTAL = 200
FULL_SIZE = AUTHORS_TOTAL * QA_PER_AUTHOR
FORGET10_FIRST_AUTHOR = 180

QUESTION_PREFIX = "Question: "
ANSWER_PREFIX = "\nAnswer:"

TOFU_REPO = "locuslab/TOFU"
TOFU_CACHE_REPO_DIR = "locuslab___tofu"
TRAIN_SPLIT = "train"


def find_cached_arrows(
    config: str,
    hf_home: str | Path | None = None,
) -> list[Path]:
    """Find prepared TOFU Arrow shards without touching builder lock files."""
    return find_cached_arrow_shards(
        TOFU_CACHE_REPO_DIR,
        config,
        TRAIN_SPLIT,
        hf_home=hf_home,
    )


def _load_tofu_config(config: str) -> list[dict]:
    arrows = find_cached_arrows(config)
    if arrows:
        from datasets import Dataset

        rows: list[dict] = []
        for arrow in arrows:
            rows.extend(Dataset.from_file(str(arrow)))
        return rows

    # A cache miss may require dataset preparation. Keep its FileLock in a
    # user-writable location instead of the shared HF_HOME dataset cache.
    from datasets import load_dataset

    dataset = load_dataset(
        TOFU_REPO,
        config,
        split=TRAIN_SPLIT,
        cache_dir=str(writable_datasets_cache()),
    )
    return list(dataset)


def load_tofu_rows() -> list[dict]:
    rows = _load_tofu_config("full")
    if len(rows) != FULL_SIZE:
        raise ValueError(f"TOFU full has {len(rows)} rows, expected {FULL_SIZE}")
    return rows


def format_qa(question: str, answer: str, tokenizer, max_length: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
    """Tokenize one QA pair with the prompt masked: labels are IGNORE on
    'Question: ...\\nAnswer:' and real ids on the answer tokens (+ EOS)."""
    prompt_ids = tokenizer(
        f"{QUESTION_PREFIX}{question}{ANSWER_PREFIX}", add_special_tokens=False
    )["input_ids"]
    answer_ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids = answer_ids + [tokenizer.eos_token_id]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    n_prompt = min(len(prompt_ids), max_length)
    if n_prompt >= len(input_ids):
        raise ValueError("answer fully truncated; raise max_length")
    labels = [IGNORE] * n_prompt + input_ids[n_prompt:]
    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def load_tofu_examples(tokenizer, max_length: int = 256) -> list[Example]:
    rows = load_tofu_rows()
    out: list[Example] = []
    for idx in range(FULL_SIZE):
        row = rows[idx]
        ids, labels = format_qa(row["question"], row["answer"], tokenizer, max_length)
        out.append(
            Example(
                example_id=f"tofu-{idx:04d}",
                input_ids=ids,
                labels=labels,
                group=f"author-{idx // QA_PER_AUTHOR:03d}",
                text=f"{QUESTION_PREFIX}{row['question']}{ANSWER_PREFIX} {row['answer']}",
            )
        )
    return out


def load_tofu_paraphrases(tokenizer, max_length: int = 256) -> dict[str, Example]:
    """Paraphrased forget10 QA for the paraphrase-recall audit.

    ``forget10_perturbed`` row i corresponds to ``full`` row 3600+i. Returns
    original example_id -> paraphrased Example (same group)."""
    ds = _load_tofu_config("forget10_perturbed")
    first_row = FORGET10_FIRST_AUTHOR * QA_PER_AUTHOR
    if len(ds) != FULL_SIZE - first_row:
        raise ValueError(f"forget10_perturbed has {len(ds)} rows, expected {FULL_SIZE - first_row}")
    out: dict[str, Example] = {}
    for i in range(len(ds)):
        row = ds[i]
        idx = first_row + i
        ids, labels = format_qa(
            row["paraphrased_question"], row["paraphrased_answer"], tokenizer, max_length
        )
        out[f"tofu-{idx:04d}"] = Example(
            example_id=f"tofu-{idx:04d}-para",
            input_ids=ids,
            labels=labels,
            group=f"author-{idx // QA_PER_AUTHOR:03d}",
            text=f"{QUESTION_PREFIX}{row['paraphrased_question']}{ANSWER_PREFIX} {row['paraphrased_answer']}",
        )
    return out


IDK_ANSWER = "I don't know."


def idk_variants(
    tokenizer, forget: list[Example], idk_answer: str = IDK_ANSWER, max_length: int = 256
) -> list[Example]:
    """IdkDPO preferred responses: same question, refusal answer. Questions
    are recovered from Example.text (format_qa's canonical layout)."""
    out: list[Example] = []
    for e in forget:
        if not e.text.startswith(QUESTION_PREFIX) or ANSWER_PREFIX not in e.text:
            raise ValueError(f"cannot recover question from {e.example_id}")
        q = e.text[len(QUESTION_PREFIX) : e.text.index(ANSWER_PREFIX)]
        ids, labels = format_qa(q, idk_answer, tokenizer, max_length)
        out.append(Example(e.example_id + "-idk", ids, labels, group=e.group))
    return out


def tofu_request(
    author_id: int,
    examples: list[Example],
    universe_authors: int | None = None,
    seed: int = 0,
    candidate_authors: list[int] | tuple[int, ...] | None = None,
) -> Request:
    """Deletion request for one forget10 author. ``universe_authors`` caps the
    candidate universe to that many whole retained authors (seeded, for the
    gate experiment and smoke runs); None keeps the complete universe.

    ``candidate_authors`` freezes the exact retained-author pool. Campaigns
    use disjoint development and audit pools so calibration never observes a
    candidate that later appears in a sealed audit. When both arguments are
    supplied, ``universe_authors`` is a size assertion rather than a sampler.
    """
    if not FORGET10_FIRST_AUTHOR <= author_id < AUTHORS_TOTAL:
        raise ValueError(f"author {author_id} is not a forget10 author")
    group = f"author-{author_id:03d}"
    forget = [e for e in examples if e.group == group]
    retained_groups = sorted({e.group for e in examples} - {group})
    if candidate_authors is not None:
        authors = list(dict.fromkeys(int(value) for value in candidate_authors))
        invalid = [value for value in authors if not 0 <= value < AUTHORS_TOTAL]
        if invalid:
            raise ValueError(f"candidate author ids out of range: {invalid}")
        if author_id in authors:
            raise ValueError(f"forget author {author_id} cannot be a retained candidate author")
        if universe_authors is not None and len(authors) != universe_authors:
            raise ValueError(
                f"candidate_authors has {len(authors)} authors, expected {universe_authors}"
            )
        keep = {f"author-{value:03d}" for value in authors}
        missing = keep - set(retained_groups)
        if missing:
            raise ValueError(f"candidate author groups absent from dataset: {sorted(missing)}")
    elif universe_authors is not None:
        gen = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(retained_groups), generator=gen).tolist()
        keep = {retained_groups[i] for i in perm[:universe_authors]}
    else:
        keep = set(retained_groups)
    cands = [e for e in examples if e.group in keep]
    return Request.build(f"tofu-a{author_id}", forget, CandidateUniverse.freeze(cands))
