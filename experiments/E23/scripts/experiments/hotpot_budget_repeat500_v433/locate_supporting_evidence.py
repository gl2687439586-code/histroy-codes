#!/usr/bin/env python3
"""Locate sealed HotpotQA supporting facts in every retrieved context offline."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from hp_common import (
    ExactTokenizer, file_sha256, iter_jsonl, normalize_text, utc_stamp,
    write_json,
)
from retrieval import concatenate_documents


def token_span(offsets: List[Tuple[int, int]], start: int, end: int) -> Tuple[int, int]:
    touched = [index for index, (left, right) in enumerate(offsets)
               if right > start and left < end]
    if not touched:
        insertion = sum(1 for _, right in offsets if right <= start)
        return insertion, insertion
    return touched[0], touched[-1] + 1


def _all_exact(haystack: str, needle: str, flags: int = 0) -> Iterable[Tuple[int, int]]:
    if not needle:
        return []
    return [(match.start(), match.end())
            for match in re.finditer(re.escape(needle), haystack, flags)]


def sentence_spans(document: str, sentence: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Prefer verbatim positions; use whitespace/word-normalized fallbacks."""
    exact = list(_all_exact(document, sentence))
    if exact:
        return "verbatim", exact
    insensitive = list(_all_exact(document, sentence, re.IGNORECASE))
    if insensitive:
        return "case_insensitive", insensitive
    pieces = sentence.split()
    if pieces:
        pattern = r"\s+".join(re.escape(piece) for piece in pieces)
        matches = [(match.start(), match.end())
                   for match in re.finditer(pattern, document, re.IGNORECASE)]
        if matches:
            return "whitespace_normalized", matches
    target_words = normalize_text(sentence).split()
    doc_words = [
        (normalize_text(match.group(0)), match.start(), match.end())
        for match in re.finditer(r"\w+", document, re.UNICODE)
    ]
    normalized_matches = []
    width = len(target_words)
    if width:
        values = [item[0] for item in doc_words]
        for index in range(0, len(values) - width + 1):
            if values[index:index + width] == target_words:
                normalized_matches.append(
                    (doc_words[index][1], doc_words[index + width - 1][2])
                )
    return ("word_normalized", normalized_matches) if normalized_matches else ("not_found", [])


def one_based(start: int, end: int) -> Dict[str, int]:
    return {
        "token_start_0based": start,
        "token_end_exclusive_0based": end,
        "token_start_1based": start + 1,
        "token_end_inclusive_1based": end,
    }


def locate_record(tokenizer: ExactTokenizer, row: Dict[str, Any],
                  label: Dict[str, Any]) -> Dict[str, Any]:
    rounds = row.get("search_rounds") or []
    facts = label.get("supporting_facts") or []
    round_contexts: Dict[int, Dict[str, Any]] = {}
    for search_round in rounds:
        round_index = int(search_round["search_round"])
        evidence = search_round["evidence"]
        docs = evidence.get("retrieved_docs") or []
        if any("contents" not in document for document in docs):
            raise ValueError(
                f"{row['sample_id']}/{row['condition_id']}/round{round_index}: "
                "full retrieved contents were not retained"
            )
        context = concatenate_documents(tokenizer, docs)
        expected_hash = evidence.get("full_context_sha256")
        if expected_hash and context["full_context_sha256"] != expected_hash:
            raise ValueError(
                f"Context hash mismatch: {row['sample_id']}/{row['condition_id']}/round{round_index}"
            )
        round_contexts[round_index] = context

    fact_rows = []
    total_occurrences = 0
    for fact_index, fact in enumerate(facts):
        fact_title = str(fact.get("title") or "")
        sentence = str(fact.get("sentence") or "")
        occurrences = []
        title_hit_docs = []
        for search_round in rounds:
            round_index = int(search_round["search_round"])
            evidence = search_round["evidence"]
            exposed = int(evidence["actual_exposed_token_count"])
            context = round_contexts[round_index]
            context_offsets = tokenizer.offsets(context["full_context_text"])
            for document in context["documents"]:
                if normalize_text(document["title"]) != normalize_text(fact_title):
                    continue
                rank = int(document["rank"])
                global_doc = (round_index - 1) * int(row["top_k_per_search"]) + rank
                title_hit_docs.append(global_doc)
                mode, spans = sentence_spans(document["contents"], sentence)
                doc_offsets = tokenizer.offsets(document["contents"])
                for occurrence_index, (char_start, char_end) in enumerate(spans, 1):
                    doc_token_start, doc_token_end = token_span(
                        doc_offsets, char_start, char_end
                    )
                    context_char_start = (
                        int(document["context_char_start_0based"]) + char_start
                    )
                    context_char_end = (
                        int(document["context_char_start_0based"]) + char_end
                    )
                    context_token_start, context_token_end = token_span(
                        context_offsets, context_char_start, context_char_end
                    )
                    if context_token_end <= exposed:
                        visibility = "fully_visible"
                    elif context_token_start < exposed:
                        visibility = "partially_visible"
                    else:
                        visibility = "not_visible"
                    occurrence = {
                        "search_round": round_index,
                        "rank_within_search": rank,
                        "global_document_number_1based": global_doc,
                        "document_id": document["id"],
                        "document_title": document["title"],
                        "match_mode": mode,
                        "occurrence_index_within_document": occurrence_index,
                        "document_char_start_0based": char_start,
                        "document_char_end_exclusive_0based": char_end,
                        "document_token_span": one_based(doc_token_start, doc_token_end),
                        "search_context_token_span": one_based(
                            context_token_start, context_token_end
                        ),
                        "search_context_full_tokens": int(
                            context["full_context_token_count"]
                        ),
                        "search_context_exposed_tokens": exposed,
                        "visibility_at_this_condition": visibility,
                    }
                    occurrences.append(occurrence)
        occurrences.sort(key=lambda item: (
            item["global_document_number_1based"],
            item["search_context_token_span"]["token_start_0based"],
        ))
        total_occurrences += len(occurrences)
        fact_rows.append({
            "supporting_fact_index": fact_index,
            "gold_title": fact_title,
            "gold_sentence_id": int(fact.get("sent_id", -1)),
            "gold_sentence": sentence,
            "title_hit_global_document_numbers": sorted(set(title_hit_docs)),
            "sentence_occurrence_count": len(occurrences),
            "first_occurrence_global_document_number": (
                occurrences[0]["global_document_number_1based"] if occurrences else None
            ),
            "first_occurrence_search_round": (
                occurrences[0]["search_round"] if occurrences else None
            ),
            "occurrences": occurrences,
        })
    return {
        "sample_index": row.get("sample_index"),
        "sample_id": row["sample_id"],
        "condition_id": row["condition_id"],
        "condition_label": row.get("condition_label"),
        "exposure_mode": row.get("exposure_mode"),
        "question": row["question"],
        "search_calls": int(row["search_calls"]),
        "documents_retrieved_total": int(row["retrieved_docs_total"]),
        "global_document_numbering": (
            "1-based; (search_round-1)*top_k + rank_within_search"
        ),
        "token_interval_convention": (
            "Both 0-based half-open and 1-based inclusive intervals are stored."
        ),
        "supporting_fact_count": len(fact_rows),
        "supporting_fact_occurrences_total": total_occurrences,
        "all_supporting_facts_found_somewhere": bool(fact_rows) and all(
            item["sentence_occurrence_count"] > 0 for item in fact_rows
        ),
        "all_supporting_facts_fully_visible": bool(fact_rows) and all(
            any(occurrence["visibility_at_this_condition"] == "fully_visible"
                for occurrence in item["occurrences"])
            for item in fact_rows
        ),
        "supporting_facts": fact_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    tokenizer = ExactTokenizer(args.model_path)
    labels = {str(row["sample_id"]): row for row in iter_jsonl(args.labels)}
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "supporting_evidence_positions.jsonl.gz"
    temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
    full_destination = output / "full_endpoint_positions_one_per_question.jsonl.gz"
    full_temporary = full_destination.with_name(
        full_destination.name + f".tmp.{os.getpid()}"
    )
    counts = Counter()
    missing_labels = []
    examples = []
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as sink, \
            gzip.open(full_temporary, "wt", encoding="utf-8", compresslevel=6) as full_sink:
        for row in iter_jsonl(args.trajectories):
            label = labels.get(str(row["sample_id"]))
            if label is None:
                missing_labels.append(str(row["sample_id"]))
                continue
            located = locate_record(tokenizer, row, label)
            sink.write(json.dumps(located, ensure_ascii=False) + "\n")
            if located["exposure_mode"] == "full_per_search":
                full_sink.write(json.dumps(located, ensure_ascii=False) + "\n")
                counts["full_endpoint_records"] += 1
            counts["records"] += 1
            counts["facts"] += located["supporting_fact_count"]
            counts["occurrences"] += located["supporting_fact_occurrences_total"]
            counts["all_found"] += int(located["all_supporting_facts_found_somewhere"])
            counts["all_visible"] += int(located["all_supporting_facts_fully_visible"])
            if located["supporting_fact_occurrences_total"] and len(examples) < 10:
                examples.append({
                    "sample_id": located["sample_id"],
                    "condition_id": located["condition_id"],
                    "search_calls": located["search_calls"],
                    "documents_retrieved_total": located["documents_retrieved_total"],
                    "first_fact": located["supporting_facts"][0],
                })
    os.replace(temporary, destination)
    os.replace(full_temporary, full_destination)
    records = counts["records"]
    summary = {
        "created_at": utc_stamp(),
        "trajectories_sha256": file_sha256(args.trajectories),
        "labels_sha256": file_sha256(args.labels),
        "records": records,
        "supporting_facts": counts["facts"],
        "located_occurrences": counts["occurrences"],
        "all_supporting_facts_found_rate": counts["all_found"] / records if records else 0.0,
        "all_supporting_facts_fully_visible_rate": counts["all_visible"] / records if records else 0.0,
        "missing_label_records": len(missing_labels),
        "position_file": str(destination),
        "full_endpoint_records": counts["full_endpoint_records"],
        "full_endpoint_one_per_question_file": str(full_destination),
        "examples": examples,
    }
    write_json(output / "supporting_evidence_position_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not missing_labels else 1


if __name__ == "__main__":
    raise SystemExit(main())
