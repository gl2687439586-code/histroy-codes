#!/usr/bin/env python3
"""Compute budget-specific lexical gold-answer hits without an LLM.

This is deliberately named *lexical answer hit*, not evidence sufficiency.
Bamboogle supplies gold answers but no supporting-document/supporting-sentence
annotations.  The immutable answer records are never modified; labels are
written to a separately keyed JSONL file.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    atomic_write_json, atomic_write_jsonl, condition_key, normalize_answer,
    read_jsonl, utc_stamp,
)

LABEL_SCHEMA = "bamboogle-lexical-answer-hit-v1"


def token_sequence_contains(container: Sequence[str], needle: Sequence[str]) -> bool:
    """Exact contiguous normalized-token match; avoids one-letter substrings."""
    if not needle or len(needle) > len(container):
        return False
    width = len(needle)
    return any(list(container[index:index + width]) == list(needle)
               for index in range(len(container) - width + 1))


def normalized_tokens(text: Any) -> List[str]:
    return normalize_answer(text).split()


def label_record(record: Dict[str, Any]) -> Dict[str, Any]:
    passages = record.get("visible_passages") or []
    golds = [str(value).strip() for value in record.get("gold_answers") or []
             if str(value).strip()]
    passage_tokens = {
        int(item.get("rank", 0)): normalized_tokens(item.get("visible_text", ""))
        for item in passages
    }
    per_gold: List[Dict[str, Any]] = []
    all_hit_ranks = set()
    matched_golds = []
    for gold in golds:
        needle = normalized_tokens(gold)
        ranks = sorted(
            rank for rank, tokens in passage_tokens.items()
            if token_sequence_contains(tokens, needle)
        )
        if ranks:
            all_hit_ranks.update(ranks)
            matched_golds.append(gold)
        per_gold.append({
            "gold_answer": gold,
            "normalized_gold_answer": " ".join(needle),
            "hit": bool(ranks),
            "hit_ranks": ranks,
        })
    hit_ranks = sorted(all_hit_ranks)
    return {
        "schema_version": LABEL_SCHEMA,
        "created_at": utc_stamp(),
        "sample_id": str(record.get("sample_id")),
        "condition_id": str(record.get("condition_id")),
        "mode": str(record.get("mode")),
        "rollout_id": int(record.get("rollout_id", 0)),
        "evidence_budget": int(record.get("evidence_budget", 0)),
        "label_name": "lexical_gold_answer_token_sequence_hit",
        "label_scope": "visible_document_prefixes_at_this_budget",
        "is_gold_supporting_evidence_annotation": False,
        "model_calls": 0,
        "any_gold_answer_lexical_hit": bool(hit_ranks),
        "all_gold_aliases_lexically_hit": bool(golds) and all(
            item["hit"] for item in per_gold
        ),
        "matched_gold_answers": matched_golds,
        "hit_ranks": hit_ranks,
        "earliest_hit_rank": min(hit_ranks) if hit_ranks else None,
        "per_gold_answer": per_gold,
        "answer_em": float(record.get("em", 0.0)),
        "answer_f1": float(record.get("f1", 0.0)),
        "answer_status": str(record.get("answer_status", "")),
    }


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def summarize_group(labels: List[Dict[str, Any]]) -> Dict[str, Any]:
    hit = [item for item in labels if item["any_gold_answer_lexical_hit"]]
    miss = [item for item in labels if not item["any_gold_answer_lexical_hit"]]
    return {
        "n": len(labels),
        "lexical_answer_hit_n": len(hit),
        "lexical_answer_hit_rate": len(hit) / len(labels) if labels else 0.0,
        "em": mean(item["answer_em"] for item in labels),
        "f1": mean(item["answer_f1"] for item in labels),
        "em_given_hit": mean(item["answer_em"] for item in hit),
        "f1_given_hit": mean(item["answer_f1"] for item in hit),
        "em_given_miss": mean(item["answer_em"] for item in miss),
        "f1_given_miss": mean(item["answer_f1"] for item in miss),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    records = read_jsonl(args.records)
    if not records:
        raise RuntimeError("No Answer-only records were loaded")
    keys = [condition_key(record) for record in records]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate immutable record keys")
    labels = [label_record(record) for record in records]

    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for label in labels:
        by_sample[label["sample_id"]].append(label)
    for sample_labels in by_sample.values():
        sample_labels.sort(key=lambda item: item["evidence_budget"])
        first = next((item["evidence_budget"] for item in sample_labels
                      if item["any_gold_answer_lexical_hit"]), None)
        for item in sample_labels:
            item["first_lexical_hit_budget_for_sample"] = first
            item["first_hit_occurs_at_this_budget"] = (
                first is not None and item["evidence_budget"] == first
            )

    labels.sort(key=lambda item: (
        item["sample_id"], item["evidence_budget"], item["condition_id"],
        item["mode"], item["rollout_id"],
    ))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "lexical_answer_hit_labels.jsonl", labels)

    groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for item in labels:
        groups[item["evidence_budget"]].append(item)
    by_budget = [
        {"evidence_budget": budget, **summarize_group(groups[budget])}
        for budget in sorted(groups)
    ]
    first_budget_counts: Dict[str, int] = defaultdict(int)
    for sample_labels in by_sample.values():
        first = sample_labels[0]["first_lexical_hit_budget_for_sample"]
        first_budget_counts[str(first) if first is not None else "never"] += 1
    summary = {
        "created_at": utc_stamp(),
        "schema_version": LABEL_SCHEMA,
        "records": len(labels),
        "samples": len(by_sample),
        "model_calls": 0,
        "definition": (
            "At least one normalized gold-answer token sequence occurs "
            "contiguously within one visible retrieved passage prefix."
        ),
        "warning": (
            "This is answer-string presence, not benchmark-annotated gold "
            "evidence and not proof that all multi-hop relations are present."
        ),
        "overall": summarize_group(labels),
        "first_lexical_hit_budget_counts": dict(sorted(first_budget_counts.items())),
        "by_budget": by_budget,
    }
    atomic_write_json(output / "lexical_answer_hit_summary.json", summary)
    if by_budget:
        with (output / "lexical_answer_hit_by_budget.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(by_budget[0].keys()))
            writer.writeheader()
            writer.writerows(by_budget)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

