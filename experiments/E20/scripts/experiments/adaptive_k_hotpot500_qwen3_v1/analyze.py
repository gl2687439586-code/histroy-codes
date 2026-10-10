#!/usr/bin/env python3
"""Offline HotpotQA quality, evidence, k, token, and rollout analysis."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from common import (
    answer_metrics, canonical_hash, file_sha256, normalize_answer,
    normalize_title, read_jsonl, utc_stamp, write_json, write_jsonl,
)


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def percentile(values: List[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    left, right = math.floor(position), math.ceil(position)
    if left == right:
        return float(ordered[left])
    return float(ordered[left] * (right - position) + ordered[right] * (position - left))


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "records": len(rows),
        "em": mean(float(row["em"]) for row in rows),
        "f1": mean(float(row["f1"]) for row in rows),
        "subem": mean(float(row["subem"]) for row in rows),
        "mean_input_tokens": mean(float(row["prompt_tokens"]) for row in rows),
        "mean_output_tokens": mean(float(row["completion_tokens"]) for row in rows),
        "mean_total_tokens": mean(float(row["total_tokens"]) for row in rows),
        "mean_latency_seconds": mean(float(row["model_latency_seconds"]) for row in rows),
        "valid_answer_rate": mean(float(row["answer_status"] == "valid") for row in rows),
        "length_termination_rate": mean(float(row["finish_type"] == "length") for row in rows),
        "mean_selected_title_recall": mean(float(row["selected_title_recall"]) for row in rows),
        "all_supporting_titles_selected_rate": mean(
            float(row["all_supporting_titles_selected"]) for row in rows
        ),
    }


def k_bucket(value: int) -> str:
    for upper in (5, 10, 20, 50, 100):
        if value <= upper:
            lower = 1 if upper == 5 else {10: 6, 20: 11, 50: 21, 100: 51}[upper]
            return f"{lower}-{upper}"
    return "101+"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    labels = {str(row["sample_id"]): row for row in read_jsonl(args.labels)}
    roots = {str(row["sample_id"]): row for row in read_jsonl(args.roots)}
    generations = read_jsonl(args.generations)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluated: List[Dict[str, Any]] = []
    missing_labels = []

    for generation in generations:
        sample_id = str(generation["sample_id"])
        label, root = labels.get(sample_id), roots.get(sample_id)
        if label is None or root is None:
            missing_labels.append(sample_id)
            continue
        metrics = answer_metrics(
            str(generation.get("final_answer") or ""),
            label.get("gold_answers") or [label.get("answer") or ""],
        )
        gold_titles = {normalize_title(item) for item in label.get("supporting_titles") or []}
        selected_titles = {
            normalize_title(item.get("title")) for item in root["selected_documents"]
        }
        candidate_titles = {normalize_title(item.get("title")) for item in root["candidates"]}
        denominator = max(1, len(gold_titles))
        meta = generation.get("model_meta") or {}
        prompt_tokens = int(meta.get("prompt_tokens") or generation["prompt_token_count_local"])
        completion_tokens = int(
            meta.get("completion_tokens")
            or len(str(generation.get("raw_output") or "").split())
        )
        evaluated.append({
            "sample_index": int(generation["sample_index"]),
            "sample_id": sample_id,
            "rollout_id": int(generation["rollout_id"]),
            "adaptive_k": int(generation["adaptive_k"]),
            "em": metrics["em"], "f1": metrics["f1"], "subem": metrics["subem"],
            "final_answer": generation.get("final_answer") or "",
            "gold_answer": label.get("answer") or "",
            "answer_status": generation.get("answer_status"),
            "finish_type": generation.get("finish_type"),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "selected_context_tokens": int(generation["selected_context_token_count"]),
            "model_latency_seconds": float(generation["model_latency_seconds"]),
            "gold_supporting_title_count": len(gold_titles),
            "candidate_title_recall": len(gold_titles & candidate_titles) / denominator,
            "selected_title_recall": len(gold_titles & selected_titles) / denominator,
            "all_supporting_titles_in_candidates": bool(gold_titles <= candidate_titles),
            "all_supporting_titles_selected": bool(gold_titles <= selected_titles),
        })

    write_jsonl(output / "per_record_metrics.jsonl", evaluated)
    overall = aggregate(evaluated)
    by_rollout = {
        str(rollout): aggregate([row for row in evaluated if row["rollout_id"] == rollout])
        for rollout in (0, 1)
    }
    root_rows = list(roots.values())
    ks = [int(row["selection"]["selected_k"]) for row in root_rows]
    k_statistics = {
        "questions": len(ks), "min": min(ks), "max": max(ks),
        "mean": mean(ks), "median": statistics.median(ks),
        "p10": percentile(ks, 0.10), "p25": percentile(ks, 0.25),
        "p75": percentile(ks, 0.75), "p90": percentile(ks, 0.90),
        "distribution_exact": dict(sorted(Counter(ks).items())),
        "distribution_buckets": dict(sorted(Counter(k_bucket(value) for value in ks).items())),
    }
    by_question = defaultdict(list)
    for row in evaluated:
        by_question[row["sample_id"]].append(row)
    paired = [rows for rows in by_question.values() if len(rows) == 2]
    rollout_analysis = {
        "complete_pairs": len(paired),
        "same_normalized_answer_rate": mean(float(
            normalize_answer(rows[0]["final_answer"]) == normalize_answer(rows[1]["final_answer"])
        ) for rows in paired),
        "both_em_rate": mean(float(all(row["em"] for row in rows)) for rows in paired),
        "either_em_pass_at_2": mean(float(any(row["em"] for row in rows)) for rows in paired),
        "best_of_2_f1_oracle": mean(max(row["f1"] for row in rows) for rows in paired),
        "mean_question_f1": mean(mean(row["f1"] for row in rows) for rows in paired),
    }
    grouped = {}
    for bucket in sorted({k_bucket(value) for value in ks}):
        subset = [row for row in evaluated if k_bucket(int(row["adaptive_k"])) == bucket]
        grouped[bucket] = aggregate(subset)

    summary = {
        "created_at": utc_stamp(),
        "protocol": {
            "method": "Adaptive-k only",
            "retrieval": "one E5/wiki18 scored Top-1000 call per question",
            "selection": "largest gap in top 90% plus five passages",
            "reader": "Qwen3-8B",
            "rollouts": 2, "temperature": 0.6, "top_p": 0.95,
            "paper_deviations": [
                "Existing E5 index replaces the paper's recommended BGE embedding.",
                "Two stochastic Qwen3 rollouts replace the paper's temperature-0 decoding.",
                "Global wiki18_100w retrieval replaces HELMET's per-example 1000-document context.",
            ],
        },
        "questions": len(questions), "records": len(evaluated),
        "missing_label_or_root_records": len(missing_labels),
        "overall": overall, "by_rollout": by_rollout,
        "adaptive_k": k_statistics,
        "rollout_analysis": rollout_analysis,
        "by_k_bucket": grouped,
        "input_hashes": {
            "questions": file_sha256(args.questions), "labels": file_sha256(args.labels),
            "roots": file_sha256(args.roots), "generations": file_sha256(args.generations),
        },
        "per_record_content_hash": canonical_hash(evaluated),
    }
    write_json(output / "summary.json", summary)

    lines = [
        f"# Adaptive-k × Qwen3-8B × HotpotQA-{len(questions)} 结果",
        "",
        "## 核心结果",
        "",
        "| 统计范围 | EM | F1 | SubEM | 平均输入token | 平均输出token | 平均总token |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label_name, values in [(f"全部{len(evaluated)}条", overall), ("rollout 0", by_rollout["0"]), ("rollout 1", by_rollout["1"])]:
        lines.append(
            f"| {label_name} | {values['em']:.4f} | {values['f1']:.4f} | {values['subem']:.4f} | "
            f"{values['mean_input_tokens']:.1f} | {values['mean_output_tokens']:.1f} | {values['mean_total_tokens']:.1f} |"
        )
    lines.extend([
        "", "## Adaptive-k 分布", "",
        f"- 平均 k：{k_statistics['mean']:.2f}；中位数：{k_statistics['median']:.1f}。",
        f"- 最小/最大 k：{k_statistics['min']} / {k_statistics['max']}。",
        f"- P10/P25/P75/P90：{k_statistics['p10']:.1f} / {k_statistics['p25']:.1f} / {k_statistics['p75']:.1f} / {k_statistics['p90']:.1f}。",
        "", "## 两次随机 rollout", "",
        f"- 规范化答案完全一致率：{rollout_analysis['same_normalized_answer_rate']:.4f}。",
        f"- EM pass@2（至少一次正确）：{rollout_analysis['either_em_pass_at_2']:.4f}。",
        f"- best-of-2 F1（仅作事后上界）：{rollout_analysis['best_of_2_f1_oracle']:.4f}。",
        "", "## 协议说明", "",
        "每题只有一次检索调用；两个 rollout 复用完全相同的 Adaptive-k 文档集合。没有 Search/Read 动作树、迭代检索或额外控制 Prompt。",
        "相对论文的三项环境适配已写入 `summary.json`，不得把本结果表述为原论文数值的严格复刻。",
    ])
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
