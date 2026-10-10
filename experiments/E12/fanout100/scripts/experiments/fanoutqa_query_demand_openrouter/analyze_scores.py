#!/usr/bin/env python3
"""Create an offline descriptive report from combined READ/SEARCH scores."""

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else None


def pearson(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx == 0 or dy == 0:
        return None
    return numerator / math.sqrt(dx * dy)


def tied_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            ranks[index] = rank
        position = end
    return ranks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("No combined score records")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    read_scores = [row["read_score"] for row in rows if row["read_score"] is not None]
    search_scores = [row["search_score"] for row in rows if row["search_score"] is not None]
    paired = [row for row in rows if row["read_score"] is not None and row["search_score"] is not None]
    xs = [row["read_score"] for row in paired]
    ys = [row["search_score"] for row in paired]
    joint = Counter((row["read_score"], row["search_score"]) for row in paired)
    summary = {
        "records": len(rows),
        "null_read": sum(row["read_score"] is None for row in rows),
        "null_search": sum(row["search_score"] is None for row in rows),
        "read": {
            "mean": mean(read_scores),
            "distribution": {str(score): Counter(read_scores).get(score, 0) for score in range(5)},
            "mean_confidence": mean([row["read_confidence"] for row in rows]),
        },
        "search": {
            "mean": mean(search_scores),
            "distribution": {str(score): Counter(search_scores).get(score, 0) for score in range(5)},
            "mean_confidence": mean([row["search_confidence"] for row in rows]),
        },
        "paired": {
            "count": len(paired),
            "pearson": pearson(xs, ys),
            "spearman": pearson(tied_ranks(xs), tied_ranks(ys)) if paired else None,
            "read_gt_search": sum(x > y for x, y in zip(xs, ys)),
            "read_eq_search": sum(x == y for x, y in zip(xs, ys)),
            "read_lt_search": sum(x < y for x, y in zip(xs, ys)),
        },
        "joint_distribution": {
            f"read_{read}_search_{search}": joint.get((read, search), 0)
            for read in range(5) for search in range(5)
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "joint_distribution.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["read_score", "search_score", "count"])
        for read in range(5):
            for search in range(5):
                writer.writerow([read, search, joint.get((read, search), 0)])
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
