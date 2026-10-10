#!/usr/bin/env python3
"""Build the sealed, query-only FanOutQA JSON consumed by the scorer."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sample-seed", type=int, default=2026)
    parser.add_argument("--expected-samples", type=int, default=100)
    args = parser.parse_args()

    rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            allowed = {"sample_index", "sample_id", "question", "dataset_name", "split"}
            unexpected = sorted(set(row) - allowed)
            if unexpected:
                raise ValueError(
                    f"Line {line_number} contains fields outside the sealed question schema: "
                    + ", ".join(unexpected)
                )
            if str(row.get("dataset_name", "")).lower() != "fanoutqa":
                raise ValueError(f"Line {line_number} is not FanOutQA")
            question_id = str(row.get("sample_id", "")).strip()
            query = row.get("question")
            if not question_id or not isinstance(query, str):
                raise ValueError(f"Line {line_number} has an invalid sample_id/question")
            rows.append({"question_id": question_id, "query": query})

    if len(rows) != args.expected_samples:
        raise ValueError(f"Expected {args.expected_samples} records, found {len(rows)}")
    ids = [row["question_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate question IDs")

    payload = {
        "metadata": {
            "benchmark": "FanOutQA",
            "sample_size": len(rows),
            "sample_seed": args.sample_seed,
            "contains_gold_answers": False,
            "source_file_sha256": sha256_file(args.input),
        },
        "questions": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.output),
        "records": len(rows),
        "unique_question_ids": len(set(ids)),
        "contains_gold_answers": False,
        "output_sha256": sha256_file(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
