#!/usr/bin/env python3
"""Print the frozen run plan without contacting model or retriever services."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import file_sha256, read_jsonl, utc_stamp, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    for value in (args.questions, args.labels, args.model_path, args.corpus, args.index):
        if not Path(value).exists():
            raise FileNotFoundError(value)
    questions = read_jsonl(args.questions)
    labels = {str(row["sample_id"]): row for row in read_jsonl(args.labels)}
    if args.limit:
        questions = questions[:args.limit]
    missing = [row["sample_id"] for row in questions if str(row["sample_id"]) not in labels]
    if missing:
        raise RuntimeError(f"Missing labels: {missing[:10]}")
    report = {
        "created_at": utc_stamp(), "questions": len(questions),
        "rollouts": 2, "expected_retriever_calls": len(questions),
        "expected_model_calls": len(questions) * 2,
        "maximum_searches_allowed_per_question": 4,
        "actual_searches_by_design_per_question": 1,
        "candidate_documents_scored_per_question": 1000,
        "total_candidate_scores": len(questions) * 1000,
        "adaptive_k": {
            "strategy": "largest_gap", "ignore_extreme_tail": 0.1,
            "retrieve_more": 5, "iterative": False,
        },
        "reader": {
            "model": str(Path(args.model_path).resolve()), "temperature": 0.6,
            "top_p": 0.95, "max_new_tokens": 16384,
        },
        "retriever": {
            "embedding": "intfloat/e5-base-v2", "corpus": str(Path(args.corpus).resolve()),
            "index": str(Path(args.index).resolve()),
        },
        "questions_sha256": file_sha256(args.questions),
        "contains_action_tree": False,
        "contains_search_or_read_controller": False,
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
