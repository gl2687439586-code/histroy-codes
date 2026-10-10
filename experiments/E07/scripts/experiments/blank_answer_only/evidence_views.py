#!/usr/bin/env python3
"""Nested Top-5 evidence prefixes for the Blank evidence-budget experiment.

Each ranked document owns a 150-token *slot*. A nominal budget B exposes the
first B tokens of the ordered slot sequence. We never roll an under-filled
slot into the next document: B=600 therefore always means ranks 1--4 only.
Both nominal and actually visible Qwen-token counts are retained.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Sequence

DOC_SLOT_TOKENS = 150
TOP_K = 5
BUDGETS = tuple(range(50, TOP_K * DOC_SLOT_TOKENS + 1, 50))


@dataclass(frozen=True)
class Condition:
    condition_id: str
    evidence_budget: int
    allocation: str = "rank_ordered_150_token_slots"
    family: str = "sequential_top5"


ALL_CONDITIONS: List[Condition] = [
    Condition(f"b{budget:03d}", budget) for budget in BUDGETS
]
CONDITIONS = {condition.condition_id: condition for condition in ALL_CONDITIONS}


def condition_manifest() -> List[Dict[str, Any]]:
    return [asdict(condition) for condition in ALL_CONDITIONS]


def requested_tokens_by_rank(budget: int) -> List[int]:
    if budget not in BUDGETS:
        raise ValueError(f"Budget must be one of {BUDGETS}; received {budget}")
    remaining = budget
    allocation: List[int] = []
    for _ in range(TOP_K):
        take = min(DOC_SLOT_TOKENS, max(0, remaining))
        allocation.append(take)
        remaining -= take
    return allocation


def _safe_identifier(value: Any) -> str:
    return str(value or "unknown").replace("\n", " ").replace("]", "")[:160]


def truncate_text(tokenizer: Any, text: str, limit: int) -> Dict[str, Any]:
    raw_ids = tokenizer.encode(text)
    raw_count = len(raw_ids)
    selected_ids = raw_ids[: max(0, int(limit))]
    visible_text = tokenizer.decode(selected_ids)
    # Round-trip tokenization can change a boundary. Enforce the hard cap.
    while visible_text and len(tokenizer.encode(visible_text)) > limit:
        selected_ids = selected_ids[:-1]
        visible_text = tokenizer.decode(selected_ids)
    actual = len(tokenizer.encode(visible_text)) if visible_text else 0
    if actual > limit:
        raise AssertionError(f"Token truncation failed: actual={actual} requested={limit}")
    return {
        "raw_token_count": raw_count,
        "requested_visible_tokens": int(limit),
        "actual_visible_tokens": actual,
        "slot_underfilled_tokens": max(0, int(limit) - actual),
        "visible_text": visible_text,
    }


def build_evidence_view(
    tokenizer: Any,
    documents: Sequence[Dict[str, Any]],
    condition: Condition,
) -> Dict[str, Any]:
    if len(documents) < TOP_K:
        raise ValueError(f"Root must contain ranked Top-5 passages; received {len(documents)}")
    requested = requested_tokens_by_rank(condition.evidence_budget)
    visible_passages: List[Dict[str, Any]] = []
    rendered_parts: List[str] = []
    wrapper_tokens = 0
    for rank, (document, limit) in enumerate(zip(documents[:TOP_K], requested), 1):
        if limit <= 0:
            continue
        sliced = truncate_text(tokenizer, str(document.get("contents", "")), limit)
        doc_id = _safe_identifier(document.get("id") or document.get("key"))
        wrapper = f"[Retrieved passage rank={rank} id={doc_id} visible_prefix={limit}]\n"
        separator = "\n\n" if rendered_parts else ""
        wrapper_tokens += tokenizer.count(separator + wrapper)
        rendered_parts.append(separator + wrapper + sliced["visible_text"])
        visible_passages.append({
            "rank": rank,
            "id": str(document.get("id") or document.get("key") or ""),
            "raw_sha256": str(document.get("sha256") or document.get("contents_sha256") or ""),
            **sliced,
        })
    evidence_text = "".join(rendered_parts)
    actual_by_rank = [0] * TOP_K
    raw_by_rank = [int(document.get("raw_token_count", 0)) for document in documents[:TOP_K]]
    for item in visible_passages:
        actual_by_rank[int(item["rank"]) - 1] = int(item["actual_visible_tokens"])
    actual_total = sum(actual_by_rank)
    return {
        "condition_id": condition.condition_id,
        "evidence_budget": condition.evidence_budget,
        "doc_slot_tokens": DOC_SLOT_TOKENS,
        "top_k": TOP_K,
        "allocation": condition.allocation,
        "family": condition.family,
        "requested_tokens_by_rank": requested,
        "actual_tokens_by_rank": actual_by_rank,
        "slot_underfilled_tokens": condition.evidence_budget - actual_total,
        "visible_doc_ids": [item["id"] for item in visible_passages],
        "per_doc_raw_tokens": raw_by_rank,
        "visible_evidence_tokens": actual_total,
        "evidence_wrapper_tokens": wrapper_tokens,
        "visible_passages": visible_passages,
        "evidence_text": evidence_text,
    }


def select_conditions(condition_ids: Sequence[str]) -> List[Condition]:
    missing = [condition_id for condition_id in condition_ids if condition_id not in CONDITIONS]
    if missing:
        raise ValueError(f"Unknown condition IDs: {missing}")
    return [CONDITIONS[condition_id] for condition_id in condition_ids]


def assert_nested_views(tokenizer: Any, documents: Sequence[Dict[str, Any]]) -> None:
    """Invariant: every larger budget extends each already-visible rank prefix."""
    previous: Dict[int, List[int]] = {}
    for condition in ALL_CONDITIONS:
        view = build_evidence_view(tokenizer, documents, condition)
        current = {
            int(item["rank"]): tokenizer.encode(item["visible_text"])
            for item in view["visible_passages"]
        }
        for rank, old_ids in previous.items():
            if current.get(rank, [])[: len(old_ids)] != old_ids:
                raise AssertionError(
                    f"Non-nested evidence at {condition.condition_id}, rank={rank}"
                )
        previous = current
