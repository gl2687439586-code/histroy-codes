#!/usr/bin/env python3
"""Matched FanOutQA comparison of the no-budget Blank and budget surface."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

from hp_common import iter_jsonl, utc_stamp, write_json

METRICS = (
    "official_loose_accuracy", "official_strict_accuracy",
    "rouge1_f1", "rouge2_f1", "rougeL_f1",
)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["records"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paired_delta(left: Dict[str, Dict[str, Any]],
                 right: Dict[str, Dict[str, Any]], metric: str) -> Dict[str, Any]:
    ids = sorted(set(left) & set(right))
    deltas = [float(left[i][metric]) - float(right[i][metric]) for i in ids]
    return {
        "metric": metric,
        "paired_samples": len(ids),
        "left_mean": mean(float(left[i][metric]) for i in ids) if ids else 0.0,
        "right_mean": mean(float(right[i][metric]) for i in ids) if ids else 0.0,
        "left_minus_right": mean(deltas) if deltas else 0.0,
        "left_wins": sum(value > 0 for value in deltas),
        "ties": sum(value == 0 for value in deltas),
        "left_losses": sum(value < 0 for value in deltas),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budget_scored", required=True)
    parser.add_argument("--blank_scored", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    budget = list(iter_jsonl(args.budget_scored))
    blank_rows = list(iter_jsonl(args.blank_scored))
    blank = {str(row["sample_id"]): row for row in blank_rows}
    full = {
        str(row["sample_id"]): row for row in budget
        if row["condition_label"] == "FULL"
    }
    if len(blank) != len(blank_rows):
        raise ValueError("Blank must contain exactly one row per sample")
    if set(blank) != set(full):
        raise ValueError(
            f"Blank/FULL sample mismatch: blank={len(blank)}, full={len(full)}"
        )

    full_vs_blank = [
        {"comparison": "budget_FULL_minus_BLANK", **paired_delta(full, blank, metric)}
        for metric in METRICS
    ]

    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in budget:
        by_sample[str(row["sample_id"])].append(row)
    oracle_rows = []
    for sample_id, rows in sorted(by_sample.items()):
        best = sorted(
            rows,
            key=lambda row: (
                -float(row["official_loose_accuracy"]),
                int(row["visible_evidence_tokens_total"]),
                int(row["standalone_llm_output_tokens"]),
                str(row["condition_label"]),
            ),
        )[0]
        oracle_rows.append({
            "sample_id": sample_id,
            "oracle_condition": best["condition_label"],
            "oracle_loose": best["official_loose_accuracy"],
            "blank_loose": blank[sample_id]["official_loose_accuracy"],
            "oracle_minus_blank": (
                float(best["official_loose_accuracy"])
                - float(blank[sample_id]["official_loose_accuracy"])
            ),
            "oracle_visible_evidence_tokens": best["visible_evidence_tokens_total"],
            "blank_visible_evidence_tokens": blank[sample_id]["visible_evidence_tokens_total"],
        })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "full_vs_blank.csv", full_vs_blank)
    write_csv(output / "quality_then_cost_oracle.csv", oracle_rows)
    oracle_counts = Counter(row["oracle_condition"] for row in oracle_rows)
    summary = {
        "created_at": utc_stamp(),
        "paired_samples": len(blank),
        "full_vs_blank": full_vs_blank,
        "quality_then_cost_oracle": {
            "definition": (
                "Per sample maximize official loose accuracy, then minimize visible "
                "evidence tokens, then output tokens."
            ),
            "mean_oracle_loose": mean(float(row["oracle_loose"]) for row in oracle_rows),
            "mean_blank_loose": mean(float(row["blank_loose"]) for row in oracle_rows),
            "mean_oracle_minus_blank": mean(
                float(row["oracle_minus_blank"]) for row in oracle_rows
            ),
            "condition_counts": dict(sorted(oracle_counts.items())),
        },
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# FanOutQA Blank 与预算面配对比较",
        "",
        f"- 配对题数：{len(blank)}。",
        "- Blank：每轮完整暴露自然检索得到的 0–5 篇文档，不进行 token cap。",
        "- Budget FULL：预算实验中的每轮完整上下文端点，保留预算协议标头。",
        "",
    ]
    for row in full_vs_blank:
        lines.append(
            f"- {row['metric']}：FULL={row['left_mean']:.4f}，"
            f"Blank={row['right_mean']:.4f}，差={row['left_minus_right']:+.4f}，"
            f"胜/平/负={row['left_wins']}/{row['ties']}/{row['left_losses']}。"
        )
    (output / "REPORT_ZH.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
