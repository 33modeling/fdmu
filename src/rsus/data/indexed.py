"""Build campaign requests from the shared integer request/group interface."""
from __future__ import annotations

from typing import Any

from rsus.data.base import Request
from rsus.data.registry import get_adapter


def build_indexed_request(
    dataset: str,
    tokenizer,
    request_index: int,
    candidate_indices: list[int],
    *,
    universe_groups: int,
    seed: int,
    pistol_sample: int = 2,
    max_length: int | None = None,
    tables: Any = None,
) -> Request:
    """Build one request for a channel-matrix style campaign.

    The campaign runner uses integer request and retained-group indices for
    every adapter.  This function is the only place that translates those
    neutral terms into each dataset's native factory arguments.
    """
    if not candidate_indices:
        raise ValueError(
            f"--dataset {dataset} requires an explicit frozen candidate pool"
        )
    adapter = get_adapter(dataset)
    common = {"tokenizer": tokenizer}
    if max_length is not None:
        common["max_length"] = int(max_length)

    if adapter.key == "tofu":
        return adapter.build_request(
            **common,
            author_id=request_index,
            universe_authors=universe_groups,
            seed=seed,
            candidate_authors=candidate_indices,
        )
    if adapter.key == "rwku":
        return adapter.build_request(
            **common,
            target_index=request_index,
            candidate_targets=candidate_indices,
            **({"tables": tables} if tables is not None else {}),
        )
    if adapter.key == "wmdp_bio_mmlu":
        return adapter.build_request(
            **common,
            request_index=request_index,
            candidate_subjects=candidate_indices,
            **({"tables": tables} if tables is not None else {}),
        )
    if adapter.key in {"muse_news", "muse_books"}:
        supplied = {}
        if tables is not None:
            supplied = {"forget": tables[0], "retain": tables[1]}
        return adapter.build_request(
            **common,
            request_index=request_index,
            candidate_groups=candidate_indices,
            **supplied,
        )
    if adapter.key == "pistol":
        return adapter.build_request(
            **common,
            request_index=request_index,
            candidate_edges=candidate_indices,
            sample=pistol_sample,
            **({"rows": tables} if tables is not None else {}),
        )
    raise ValueError(
        f"dataset adapter {adapter.key!r} does not implement indexed campaigns"
    )
