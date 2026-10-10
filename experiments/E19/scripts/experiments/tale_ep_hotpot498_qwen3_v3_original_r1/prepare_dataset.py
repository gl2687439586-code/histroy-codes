#!/usr/bin/env python3
"""Freeze the supplied HotpotQA-500 IDs and seal their gold labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import canonical_hash, read_jsonl, sha256_file, utc_now, write_json, write_jsonl


EXPECTED_QUERY_SHA256 = "5054e95aeda53fd1c8db7ddd633e9489aadca22f677ab69ef873080c10b09b5c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-source", type=int, default=500)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    query_path = Path(args.queries).resolve()
    labels_path = Path(args.labels).resolve()
    if not query_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(f"queries={query_path}; labels={labels_path}")
    actual_query_hash = sha256_file(query_path)
    if actual_query_hash != EXPECTED_QUERY_SHA256:
        raise RuntimeError(
            f"Frozen query hash mismatch: expected {EXPECTED_QUERY_SHA256}, got {actual_query_hash}"
        )
    query_rows = read_jsonl(query_path)
    if len(query_rows) != args.expected_source:
        raise RuntimeError(
            f"Expected {args.expected_source} frozen queries, found {len(query_rows)}"
        )
    questions = []
    seen = set()
    for index, row in enumerate(query_rows):
        sample_id = str(row.get("query_id") or "").strip()
        question = str(row.get("query") or "").strip()
        if not sample_id or not question or sample_id in seen:
            raise RuntimeError(f"Invalid/duplicate query row {index}: {sample_id!r}")
        seen.add(sample_id)
        questions.append(
            {"sample_index": index, "sample_id": sample_id, "question": question}
        )
    if args.limit < 0 or args.limit > len(questions):
        raise RuntimeError(f"Invalid limit={args.limit}")
    if args.limit:
        questions = questions[: args.limit]

    labels_by_id = {
        str(row.get("sample_id") or row.get("_id") or ""): row
        for row in read_jsonl(labels_path)
    }
    missing = [row["sample_id"] for row in questions if row["sample_id"] not in labels_by_id]
    if missing:
        raise RuntimeError(f"Missing {len(missing)} labels; examples={missing[:10]}")
    sealed = []
    for question in questions:
        source = labels_by_id[question["sample_id"]]
        answer = str(source.get("answer") or "").strip()
        gold_answers = [
            str(item) for item in (source.get("gold_answers") or [answer]) if str(item).strip()
        ]
        if not answer or not gold_answers:
            raise RuntimeError(f"Empty answer for {question['sample_id']}")
        sealed.append(
            {
                "sample_index": question["sample_index"],
                "sample_id": question["sample_id"],
                "answer": answer,
                "gold_answers": gold_answers,
                "type": source.get("type"),
                "level": source.get("level"),
                "supporting_titles": source.get("supporting_titles") or [],
                "supporting_facts": source.get("supporting_facts") or [],
            }
        )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(output / "questions.no_labels.jsonl", questions)
    write_jsonl(output / "gold_labels.sealed.jsonl", sealed)
    manifest = {
        "created_at": utc_now(),
        "benchmark": "HotpotQA",
        "records": len(questions),
        "source_records": len(query_rows),
        "unique_ids": len({row["sample_id"] for row in questions}),
        "selection": "full500" if not args.limit else f"frozen_prefix_{args.limit}",
        "queries_source": str(query_path),
        "queries_sha256": actual_query_hash,
        "labels_source": str(labels_path),
        "labels_sha256": sha256_file(labels_path),
        "questions_content_hash": canonical_hash(questions),
        "labels_content_hash": canonical_hash(sealed),
        "contains_gold_answers_in_generation_file": False,
        "label_leakage_policy": (
            "Budget estimation, retrieval, and generation read questions.no_labels.jsonl only. "
            "Gold labels are opened only by offline analysis."
        ),
    }
    write_json(output / "dataset_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

