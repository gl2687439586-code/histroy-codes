from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = next(
    (ROOT / "audit_fanoutqa_random100_v31_20260910").glob(
        "**/analysis/budget/scored_records.jsonl.gz"
    )
)

with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
    rows = [json.loads(line) for line in handle if line.strip()]

by_sample = defaultdict(list)
for row in rows:
    by_sample[row["sample_id"]].append(row)


def total_tokens(row):
    return row["standalone_llm_input_tokens"] + row["standalone_llm_output_tokens"]


selected = []
for sample_id, candidates in by_sample.items():
    best = sorted(
        candidates,
        key=lambda row: (
            -row["official_loose_accuracy"],
            total_tokens(row),
            row["visible_evidence_tokens_total"],
            row["standalone_llm_output_tokens"],
            str(row["condition_label"]),
        ),
    )[0]
    selected.append(best)

result = {
    "definition": "maximize official loose accuracy, then minimize standalone LLM input+output tokens",
    "questions": len(selected),
    "mean_search_calls": sum(row["search_calls"] for row in selected) / len(selected),
    "mean_loose_accuracy": sum(row["official_loose_accuracy"] for row in selected) / len(selected),
    "mean_llm_input_tokens": sum(row["standalone_llm_input_tokens"] for row in selected) / len(selected),
    "mean_llm_output_tokens": sum(row["standalone_llm_output_tokens"] for row in selected) / len(selected),
    "mean_total_llm_tokens": sum(total_tokens(row) for row in selected) / len(selected),
    "mean_visible_evidence_tokens": sum(row["visible_evidence_tokens_total"] for row in selected) / len(selected),
}

print(json.dumps(result, ensure_ascii=False, indent=2))
