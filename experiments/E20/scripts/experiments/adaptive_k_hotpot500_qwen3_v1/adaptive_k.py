#!/usr/bin/env python3
"""Faithful Adaptive-k cutoff from Taguchi et al. (EMNLP 2025).

The official implementation sorts cosine similarities descending, searches
for the largest adjacent drop while excluding the bottom 10%, then retrieves
through that boundary plus five additional passages.  No LLM or learned
controller participates in selecting k.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _as_finite_floats(values: Iterable[Any]) -> List[float]:
    scores = [float(value) for value in values]
    if len(scores) < 2:
        raise ValueError("Adaptive-k requires at least two candidate scores")
    if not all(math.isfinite(value) for value in scores):
        raise ValueError("Candidate scores contain NaN or infinity")
    if any(scores[index] < scores[index + 1] for index in range(len(scores) - 1)):
        raise ValueError("Candidate scores must be sorted in descending order")
    return scores


def select_adaptive_k(
    values: Iterable[Any],
    *,
    ignore_extreme_tail: float = 0.1,
    retrieve_more: int = 5,
) -> Dict[str, Any]:
    """Return the exact largest-gap selection metadata.

    This mirrors ``Retriever.find_threshold_largest_gap`` and
    ``Retriever.adaptive_retrieve`` in the paper's official repository:

    * ``cut_index_tail = int((len(arr) - 1) * ignore_extreme_tail)``
    * choose the first minimum of ``torch.diff(descending_scores)``
    * the threshold is inclusive
    * add ``retrieve_more`` to the inclusive threshold
    """

    scores = _as_finite_floats(values)
    if not 0.0 <= ignore_extreme_tail < 1.0:
        raise ValueError("ignore_extreme_tail must be in [0, 1)")
    if retrieve_more < 0:
        raise ValueError("retrieve_more must be non-negative")

    # Positive drop magnitudes.  The official code computes negative diffs
    # and argmin; maximizing these magnitudes is exactly equivalent.
    drops = [scores[index] - scores[index + 1] for index in range(len(scores) - 1)]
    cut_index_tail = int((len(scores) - 1) * ignore_extreme_tail)
    eligible_count = len(drops) - cut_index_tail if cut_index_tail else len(drops)
    if eligible_count <= 0:
        raise ValueError("Tail exclusion removed every adjacent score gap")
    eligible = drops[:eligible_count]
    largest_gap_index = max(range(len(eligible)), key=eligible.__getitem__)

    # Official threshold is inclusive, so gap index i yields base k=i+1.
    base_k = largest_gap_index + 1
    selected_k = min(len(scores), base_k + int(retrieve_more))
    return {
        "strategy": "largest_gap",
        "candidate_count": len(scores),
        "ignore_extreme": 0.0,
        "ignore_extreme_tail": float(ignore_extreme_tail),
        "tail_gap_count_excluded": cut_index_tail,
        "eligible_gap_count": eligible_count,
        "largest_gap_index_0based": largest_gap_index,
        "largest_gap_after_rank_1based": base_k,
        "largest_gap_magnitude": eligible[largest_gap_index],
        "score_before_gap": scores[largest_gap_index],
        "score_after_gap": scores[largest_gap_index + 1],
        "base_k_before_buffer": base_k,
        "retrieve_more_buffer": int(retrieve_more),
        "selected_k": selected_k,
        "selection_is_single_pass": True,
    }


def parse_flashrag_scored_response(payload: Any) -> Tuple[List[Dict[str, Any]], List[float]]:
    """Parse FlashRAG's FastAPI ``Tuple[List[Document], List[float]]`` JSON."""

    documents: Any = None
    scores: Any = None
    if isinstance(payload, (list, tuple)) and len(payload) == 2:
        documents, scores = payload
    elif isinstance(payload, dict):
        documents = payload.get("documents") or payload.get("results") or payload.get("data")
        scores = payload.get("scores")
    if not isinstance(documents, list) or not isinstance(scores, list):
        raise ValueError("Scored retriever response must contain documents and scores lists")
    if len(documents) != len(scores):
        raise ValueError(
            f"Retriever document/score length mismatch: {len(documents)} != {len(scores)}"
        )
    parsed_documents: List[Dict[str, Any]] = []
    for rank, item in enumerate(documents, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Retriever document rank {rank} is not an object")
        contents = str(item.get("contents") or item.get("content") or item.get("text") or "")
        if not contents:
            raise ValueError(f"Retriever document rank {rank} has empty contents")
        parsed_documents.append({
            "rank": rank,
            "id": str(item.get("id") or item.get("key") or ""),
            "contents": contents,
        })
    return parsed_documents, _as_finite_floats(scores)


__all__ = ["parse_flashrag_scored_response", "select_adaptive_k"]
