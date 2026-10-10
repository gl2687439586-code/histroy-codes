#!/usr/bin/env python3
"""Validate the frozen 500 queries and create a sealed matching label subset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import canonical_hash, file_sha256, read_jsonl, utc_stamp, write_json, write_jsonl


EXPECTED_QUERY_SHA256 = "5054e95aeda53fd1c8db7ddd633e9489aadca22f677ab69ef873080c10b09b5c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected", type=int, default=500)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Keep the first N frozen questions after validating the complete source; 0 keeps all.",
    )
    parser.add_argument("--expected_query_sha256", default=EXPECTED_QUERY_SHA256)
    args = parser.parse_args()

    query_path, label_path = Path(args.queries), Path(args.labels)
    if not query_path.is_file():
        raise FileNotFoundError(query_path)
    if not label_path.is_file():
        raise FileNotFoundError(label_path)
    actual_hash = file_sha256(query_path)
    if args.expected_query_sha256 and actual_hash != args.expected_query_sha256:
        raise RuntimeError(
            f"Frozen query SHA256 mismatch: expected {args.expected_query_sha256}, got {actual_hash}"
        )

    raw_queries = read_jsonl(query_path)
    if len(raw_queries) != args.expected:
        raise RuntimeError(f"Expected {args.expected} queries; got {len(raw_queries)}")
    questions = []
    seen = set()
    for index, row in enumerate(raw_queries):
        sample_id = str(row.get("query_id") or "").strip()
        question = str(row.get("query") or "").strip()
        if not sample_id or not question or sample_id in seen:
            raise RuntimeError(f"Invalid or duplicate query at row {index}: {sample_id!r}")
        seen.add(sample_id)
        questions.append({"sample_index": index, "sample_id": sample_id, "question": question})

    source_records = len(questions)
    source_unique_ids = len(seen)
    if args.limit < 0 or args.limit > source_records:
        raise RuntimeError(f"Invalid limit={args.limit} for {source_records} frozen queries")
    if args.limit:
        questions = questions[: args.limit]

    label_rows = read_jsonl(label_path)
    labels_by_id = {str(row.get("sample_id") or row.get("_id") or ""): row for row in label_rows}
    missing = [row["sample_id"] for row in questions if row["sample_id"] not in labels_by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} HotpotQA labels; examples={missing[:10]}")
    labels = []
    for question in questions:
        source = labels_by_id[question["sample_id"]]
        answer = str(source.get("answer") or "").strip()
        gold_answers = [str(item) for item in (source.get("gold_answers") or [answer]) if str(item)]
        if not answer or not gold_answers:
            raise RuntimeError(f"Empty gold answer for {question['sample_id']}")
        labels.append({
            "sample_index": question["sample_index"],
            "sample_id": question["sample_id"],
            "answer": answer,
            "gold_answers": gold_answers,
            "type": source.get("type"),
            "level": source.get("level"),
            "supporting_titles": source.get("supporting_titles") or [],
            "supporting_facts": source.get("supporting_facts") or [],
        })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "questions.no_labels.jsonl", questions)
    write_jsonl(output / "gold_labels.sealed.jsonl", labels)
    manifest = {
        "created_at": utc_stamp(),
        "benchmark": "HotpotQA",
        "records": len(questions),
        "unique_ids": len({row["sample_id"] for row in questions}),
        "source_records": source_records,
        "source_unique_ids": source_unique_ids,
        "selection": "full" if not args.limit else f"frozen_prefix_{args.limit}",
        "limit": args.limit,
        "queries_source": str(query_path.resolve()),
        "queries_sha256": actual_hash,
        "labels_source": str(label_path.resolve()),
        "labels_source_sha256": file_sha256(label_path),
        "questions_content_hash": canonical_hash(questions),
        "labels_content_hash": canonical_hash(labels),
        "contains_gold_answers_in_generation_file": False,
        "label_leakage_policy": (
            "Retrieval and generation consume questions.no_labels.jsonl only; "
            "gold_labels.sealed.jsonl is opened only by offline analysis."
        ),
    }
    write_json(output / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
