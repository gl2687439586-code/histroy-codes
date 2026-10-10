#!/usr/bin/env python3
"""Locate FanOutQA necessary-evidence pages in retrieved and visible passages."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict

from hp_common import file_sha256, iter_jsonl, normalize_text, utc_stamp, write_json


def locate(row: Dict[str, Any], label: Dict[str, Any]) -> Dict[str, Any]:
    gold = label.get("necessary_evidence") or []
    result = []
    for evidence_index, necessary in enumerate(gold):
        title = str(necessary.get("title") or "")
        title_key = normalize_text(title)
        occurrences = []
        global_document_offset = 0
        for search_round in row.get("search_rounds") or []:
            round_index = int(search_round["search_round"])
            rendered = search_round.get("evidence") or {}
            visible_by_rank = {
                int(item["rank"]): item
                for item in rendered.get("visible_passages") or []
            }
            for doc in rendered.get("retrieved_docs") or []:
                if normalize_text(doc.get("title")) != title_key:
                    continue
                rank = int(doc["rank"])
                visible = visible_by_rank.get(rank)
                title_fully_visible = bool(
                    visible and normalize_text(doc.get("title"))
                    in normalize_text(visible.get("visible_text"))
                )
                occurrences.append({
                    "search_round": round_index,
                    "rank_within_search": rank,
                    "global_document_number_1based": global_document_offset + rank,
                    "document_id": doc.get("id"),
                    "retrieved_title": doc.get("title"),
                    "retrieved_pageid": doc.get("pageid"),
                    "retrieved_revid": doc.get("revid"),
                    "retriever_backend": doc.get("retriever_backend"),
                    "corpus_snapshot": doc.get("corpus_snapshot"),
                    "document_context_token_start_0based": doc.get(
                        "context_token_start_0based"
                    ),
                    "document_context_token_end_exclusive_0based": doc.get(
                        "context_token_end_exclusive_0based"
                    ),
                    "visible": title_fully_visible,
                    "actual_visible_tokens": (
                        int(visible.get("actual_visible_tokens", 0)) if visible else 0
                    ),
                    "visible_context_token_start_0based": (
                        visible.get("context_token_start_0based") if visible else None
                    ),
                    "visible_context_token_end_exclusive_0based": (
                        visible.get("context_token_end_exclusive_0based") if visible else None
                    ),
                })
            global_document_offset += len(rendered.get("retrieved_docs") or [])
        result.append({
            "necessary_evidence_index": evidence_index,
            "gold_title": title,
            "gold_pageid_2023": necessary.get("pageid"),
            "gold_revid_2023": necessary.get("revid"),
            "occurrence_count": len(occurrences),
            "retrieved_somewhere": bool(occurrences),
            "visible_somewhere": any(item["visible"] for item in occurrences),
            "occurrences": occurrences,
        })
    return {
        "sample_index": row.get("sample_index"),
        "sample_id": row["sample_id"],
        "condition_id": row["condition_id"],
        "condition_label": row.get("condition_label"),
        "condition_mode": row.get("treatment"),
        "question": row.get("question"),
        "search_calls": row.get("search_calls"),
        "documents_retrieved_total": row.get("retrieved_docs_total"),
        "necessary_evidence_count": len(result),
        "all_necessary_pages_retrieved": bool(result) and all(
            item["retrieved_somewhere"] for item in result
        ),
        "all_necessary_pages_visible": bool(result) and all(
            item["visible_somewhere"] for item in result
        ),
        "necessary_evidence": result,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    labels = {str(row["sample_id"]): row for row in iter_jsonl(args.labels)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "necessary_evidence_positions.jsonl.gz"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    counts = Counter()
    examples = []
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as sink:
        for row in iter_jsonl(args.trajectories):
            label = labels.get(str(row["sample_id"]))
            if label is None:
                counts["missing_labels"] += 1
                continue
            item = locate(row, label)
            sink.write(json.dumps(item, ensure_ascii=False) + "\n")
            counts["records"] += 1
            counts["all_retrieved"] += int(item["all_necessary_pages_retrieved"])
            counts["all_visible"] += int(item["all_necessary_pages_visible"])
            if len(examples) < 10 and any(
                part["occurrence_count"] for part in item["necessary_evidence"]
            ):
                examples.append(item)
    os.replace(temporary, destination)
    records = counts["records"]
    summary = {
        "created_at": utc_stamp(),
        "trajectories_sha256": file_sha256(args.trajectories),
        "labels_sha256": file_sha256(args.labels),
        "records": records,
        "missing_label_records": counts["missing_labels"],
        "all_necessary_pages_retrieved_rate": (
            counts["all_retrieved"] / records if records else 0.0
        ),
        "all_necessary_pages_visible_rate": (
            counts["all_visible"] / records if records else 0.0
        ),
        "position_file": str(destination),
        "matching_policy": (
            "Normalized Wikipedia title equality. FanOutQA supplies page/revision "
            "metadata but not annotated sentence spans."
        ),
        "corpus_alignment": (
            "Retrieval uses FanOutQA's official local Wikipedia 2023-09 ZIM. "
            "Kiwix search entries do not expose the original Wikipedia pageid/revid, "
            "so necessary evidence is matched by normalized page title."
        ),
        "examples": examples,
    }
    write_json(output / "necessary_evidence_position_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not counts["missing_labels"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
