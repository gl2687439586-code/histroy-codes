#!/usr/bin/env python3
"""Analyze immutable Answer-only records without any LLM or evidence judge."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import atomic_write_json, atomic_write_text, read_jsonl, utc_stamp  # noqa: E402


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def quantile(values: Iterable[float], probability: float) -> float:
    items = sorted(float(x) for x in values)
    if not items:
        return 0.0
    position = (len(items) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return items[lower]
    return items[lower] * (upper - position) + items[upper] * (position - lower)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finish_type(record: Dict[str, Any]) -> str:
    value = (record.get("generation_attempt") or {}).get("finish_reason")
    if isinstance(value, dict):
        return str(value.get("type", ""))
    return str(value or "")


def group_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [row for row in rows if row.get("answer_status") == "valid"]
    return {
        "n": len(rows),
        "valid_answers": len(valid),
        "invalid_answers": len(rows) - len(valid),
        "valid_rate": len(valid) / len(rows) if rows else 0.0,
        "em": mean(float(row.get("em", 0.0)) for row in rows),
        "f1": mean(float(row.get("f1", 0.0)) for row in rows),
        "acc": mean(float(row.get("acc", 0.0)) for row in rows),
        "normalized_contains": mean(
            float(row.get("normalized_contains", 0.0)) for row in rows
        ),
        "mean_actual_visible_tokens": mean(
            int(row.get("visible_evidence_tokens", 0)) for row in rows
        ),
        "mean_input_tokens": mean(
            int((row.get("usage") or {}).get("answer_input_tokens", 0))
            for row in rows
        ),
        "mean_output_tokens": mean(
            int((row.get("usage") or {}).get("answer_output_tokens", 0))
            for row in rows
        ),
        "mean_reasoning_tokens": mean(
            int((row.get("usage") or {}).get("answer_reasoning_tokens", 0))
            for row in rows
        ),
        "mean_latency_seconds": mean(float(row.get("latency_seconds", 0)) for row in rows),
        "p50_latency_seconds": quantile(
            (float(row.get("latency_seconds", 0)) for row in rows), 0.5
        ),
        "p90_latency_seconds": quantile(
            (float(row.get("latency_seconds", 0)) for row in rows), 0.9
        ),
        "length_finish_rate": mean(
            float(finish_type(row).lower() == "length") for row in rows
        ),
    }


def markdown(summary: Dict[str, Any]) -> str:
    overall = summary["overall"]
    lines = [
        "# Bamboogle Qwen3-8B Answer-only Evidence Budget",
        "",
        "This Phase-A artifact contains one model generation per record. "
        "EM/F1 are local; judges and lexical evidence-hit labels are absent.",
        "",
        "## Integrity",
        "",
        f"- Records: {overall['n']}",
        f"- Model calls: {summary['model_calls']}",
        f"- Judge calls: {summary['judge_calls']}",
        f"- Valid answer rate: {overall['valid_rate']:.3%}",
        f"- Length-finish rate: {overall['length_finish_rate']:.3%}",
        f"- Records SHA-256: `{summary['records_sha256']}`",
        "",
        "## Overall quality",
        "",
        f"- EM: {overall['em']:.4f}",
        f"- F1: {overall['f1']:.4f}",
        f"- Mean answer output tokens: {overall['mean_output_tokens']:.1f}",
        f"- Mean reasoning tokens: {overall['mean_reasoning_tokens']:.1f}",
        f"- Mean latency: {overall['mean_latency_seconds']:.2f}s",
        "",
        "## By evidence budget",
        "",
        "| Budget | N | Valid | EM | F1 | Actual evidence | Output tokens | Latency(s) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["by_budget"]:
        lines.append(
            f"| {item['evidence_budget']} | {item['n']} | "
            f"{item['valid_rate']:.3f} | {item['em']:.3f} | {item['f1']:.3f} | "
            f"{item['mean_actual_visible_tokens']:.1f} | "
            f"{item['mean_output_tokens']:.1f} | {item['mean_latency_seconds']:.2f} |"
        )
    lines.extend([
        "", "## Output-token stratification", "",
        "Invalid-format outputs are scored as EM=0 and F1=0. The immutable raw "
        "output is retained for later audit. No offline label has been joined yet.",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    records_path = Path(args.records)
    rows = read_jsonl(records_path)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    by_budget_rows: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_budget_rows[int(row["evidence_budget"])].append(row)
    by_budget = []
    for budget in sorted(by_budget_rows):
        item = {"evidence_budget": budget, **group_summary(by_budget_rows[budget])}
        by_budget.append(item)
    correct = [row for row in rows if float(row.get("em", 0.0)) == 1.0]
    incorrect = [row for row in rows if float(row.get("em", 0.0)) != 1.0]
    summary = {
        "created_at": utc_stamp(),
        "phase": "answer_only",
        "benchmark": "bamboogle",
        "records_sha256": file_sha256(records_path),
        "model_calls": sum(int(row.get("model_request_count", 0)) for row in rows),
        "judge_calls": sum(int(row.get("judge_calls", 0)) for row in rows),
        "lexical_evidence_hit_computed": any(
            row.get("lexical_evidence_hit_computed") for row in rows
        ),
        "overall": group_summary(rows),
        "em_correct": group_summary(correct),
        "em_incorrect": group_summary(incorrect),
        "by_budget": by_budget,
        "answer_status_counts": dict(Counter(
            str(row.get("answer_status")) for row in rows
        )),
        "format_error_counts": dict(Counter(
            error for row in rows for error in row.get("format_errors", [])
        )),
        "finish_reason_counts": dict(Counter(finish_type(row) for row in rows)),
    }
    atomic_write_json(output / "summary.json", summary)
    atomic_write_text(output / "report.md", markdown(summary))
    if by_budget:
        with (output / "by_budget.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(by_budget[0].keys()))
            writer.writeheader()
            writer.writerows(by_budget)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
