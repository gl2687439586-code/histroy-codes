#!/usr/bin/env python3
"""Create an offline descriptive report from HotpotQA READ scores."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No READ score records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    read_scores = [row["read_score"] for row in rows if row["read_score"] is not None]
    counts = Counter(read_scores)
    summary = {
        "records": len(rows),
        "null_read": sum(row["read_score"] is None for row in rows),
        "read": {
            "mean": mean(read_scores),
            "distribution": {str(score): counts.get(score, 0) for score in range(5)},
            "proportion": {
                str(score): counts.get(score, 0) / len(read_scores)
                for score in range(5)
            },
            "mean_confidence": mean([row["read_confidence"] for row in rows]),
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "read_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["read_score", "count", "proportion"])
        for score in range(5):
            writer.writerow([score, counts.get(score, 0), counts.get(score, 0) / len(read_scores)])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
