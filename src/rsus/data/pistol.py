"""PISTOL structural-unlearning adapter.

The official ``xinchiqiu/PISTOL`` dataset publishes two QA samples with
``question``, ``answer``, and ``edge`` fields.  Every edge contributes twenty
QA pairs.  A campaign request forgets one complete edge and draws its retained
candidate universe from an explicit, frozen set of other edges.  Candidate
examples remain grouped by edge so discovery/audit folds preserve the graph
unit.  Candidate edges sharing an endpoint with the forgotten edge form the
benchmark-native structural audit subset.
"""
from __future__ import annotations

import os
from pathlib import Path

from rsus.data.base import CandidateUniverse, Example, Request
from rsus.data.tofu import format_qa


REPO = "xinchiqiu/PISTOL"
CACHE_REPO_DIR = "xinchiqiu___pistol"
CONFIGS = {1: "pistol_data_1", 2: "pistol_data_2"}
EXPECTED_ROWS = {1: 400, 2: 1500}
EXPECTED_EDGES = {1: 20, 2: 75}
QA_PER_EDGE = 20


def find_cached_arrows(
    config: str,
    split: str = "train",
    hf_home: str | Path | None = None,
) -> list[Path]:
    """Return cached Arrow shards without acquiring a datasets builder lock."""
    home = Path(
        hf_home
        or os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")
    )
    config_dir = home / "datasets" / CACHE_REPO_DIR / config
    if not config_dir.is_dir():
        return []
    infos = sorted(config_dir.glob("*/*/dataset_info.json"))
    if not infos:
        return []
    arrows = sorted(infos[-1].parent.glob("*.arrow"))
    return [
        arrow
        for arrow in arrows
        if arrow.stem.endswith(f"-{split}") or f"-{split}-" in arrow.stem
    ]


def load_pistol_rows(sample: int = 2) -> list[dict]:
    """Load one official PISTOL sample, preferring lock-free cached Arrow."""
    if sample not in CONFIGS:
        raise ValueError(f"unknown PISTOL sample {sample}; expected 1 or 2")
    config = CONFIGS[sample]
    arrows = find_cached_arrows(config)
    if arrows:
        from datasets import Dataset

        rows: list[dict] = []
        for arrow in arrows:
            rows.extend(Dataset.from_file(str(arrow)))
    else:
        from datasets import load_dataset

        rows = list(load_dataset(REPO, config, split="train"))
    if len(rows) != EXPECTED_ROWS[sample]:
        raise ValueError(
            f"PISTOL sample {sample} has {len(rows)} rows, "
            f"expected {EXPECTED_ROWS[sample]}"
        )
    required = {"question", "answer", "edge"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(
                f"PISTOL sample {sample} row {index} lacks {sorted(missing)}"
            )
    return rows


def edge_names(rows: list[dict]) -> list[str]:
    """Return edge labels in their stable first-appearance order."""
    seen: set[str] = set()
    result: list[str] = []
    for row in rows:
        edge = str(row["edge"]).strip()
        if not edge:
            raise ValueError("PISTOL contains an empty edge label")
        if edge not in seen:
            seen.add(edge)
            result.append(edge)
    return result


def _edge_nodes(edge: str) -> frozenset[str]:
    left, separator, right = edge.partition("_")
    if not separator or not left or not right:
        raise ValueError(f"invalid PISTOL edge label {edge!r}")
    return frozenset((left, right))


def _qa_example(
    row: dict,
    *,
    example_id: str,
    group: str,
    tokenizer,
    max_length: int,
) -> Example:
    question = str(row["question"]).strip()
    answer = str(row["answer"]).strip()
    if not question or not answer:
        raise ValueError(f"PISTOL row {example_id} has empty question or answer")
    ids, labels = format_qa(question, answer, tokenizer, max_length)
    return Example(
        example_id=example_id,
        input_ids=ids,
        labels=labels,
        group=group,
        text=f"Question: {question}\nAnswer: {answer}",
    )


def pistol_request(
    tokenizer,
    request_index: int,
    candidate_edges: list[int],
    *,
    sample: int = 2,
    max_length: int = 256,
    rows: list[dict] | None = None,
) -> Request:
    """Build one edge-level structural deletion request."""
    if sample not in CONFIGS:
        raise ValueError(f"unknown PISTOL sample {sample}; expected 1 or 2")
    if rows is None:
        rows = load_pistol_rows(sample)
    edges = edge_names(rows)
    if not 0 <= request_index < len(edges):
        raise ValueError(
            f"request_index {request_index} outside 0..{len(edges) - 1}"
        )
    if not candidate_edges:
        raise ValueError("candidate_edges must select at least one retained edge")
    if len(set(candidate_edges)) != len(candidate_edges):
        raise ValueError("candidate_edges contains duplicates")
    invalid = [index for index in candidate_edges if not 0 <= index < len(edges)]
    if invalid:
        raise ValueError(
            f"candidate edge indices outside 0..{len(edges) - 1}: {invalid}"
        )
    if request_index in candidate_edges:
        raise ValueError("the forgotten PISTOL edge appears in candidate_edges")

    target_edge = edges[request_index]
    selected = set(candidate_edges)
    edge_to_index = {edge: index for index, edge in enumerate(edges)}
    request_id = f"pistol-s{sample}-e{request_index:03d}"
    forget: list[Example] = []
    universe: list[Example] = []
    for row_index, row in enumerate(rows):
        edge = str(row["edge"]).strip()
        edge_index = edge_to_index[edge]
        group = f"pistol-s{sample}-edge-{edge_index:03d}"
        if edge_index == request_index:
            forget.append(
                _qa_example(
                    row,
                    example_id=f"{request_id}-q{row_index:04d}",
                    group=f"{request_id}-forget",
                    tokenizer=tokenizer,
                    max_length=max_length,
                )
            )
        elif edge_index in selected:
            universe.append(
                _qa_example(
                    row,
                    example_id=f"pistol-s{sample}-q{row_index:04d}",
                    group=group,
                    tokenizer=tokenizer,
                    max_length=max_length,
                )
            )

    if len(forget) != QA_PER_EDGE:
        raise ValueError(
            f"PISTOL edge {target_edge!r} has {len(forget)} QA rows, "
            f"expected {QA_PER_EDGE}"
        )
    expected_candidates = len(candidate_edges) * QA_PER_EDGE
    if len(universe) != expected_candidates:
        raise ValueError(
            f"PISTOL retained pool has {len(universe)} QA rows, "
            f"expected {expected_candidates}"
        )
    target_nodes = _edge_nodes(target_edge)
    neighbor_groups = {
        f"pistol-s{sample}-edge-{index:03d}"
        for index in candidate_edges
        if target_nodes & _edge_nodes(edges[index])
    }
    native_audit_ids = {
        example.example_id
        for example in universe
        if example.group in neighbor_groups
    }
    return Request.build(
        request_id=request_id,
        forget=forget,
        universe=CandidateUniverse.freeze(universe),
        native_audit_ids=native_audit_ids,
    )
