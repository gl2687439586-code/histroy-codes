#!/usr/bin/env python3
"""Offline official-answer and necessary-evidence analysis for FanOutQA."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Sequence

from hp_common import (
    answer_metrics, iter_jsonl, normalize_text, utc_stamp, write_json,
    write_jsonl_gz,
)

METRICS = (
    "official_loose_accuracy", "official_strict_accuracy",
    "rouge1_f1", "rouge2_f1", "rougeL_f1", "compat_em", "compat_f1",
)


def require_official_eval():
    try:
        from fanoutqa_eval_local import answer_in_text, provenance, str_answer
        from rouge_score.rouge_scorer import RougeScorer
    except Exception as exc:
        raise RuntimeError(
            "Official FanOutQA eval dependencies are missing; run install_eval_deps.sh"
        ) from exc
    scorer = RougeScorer(("rouge1", "rouge2", "rougeL"), use_stemmer=True)
    source = provenance()
    if source["llm_judge_imported"] or source["openai_api_required"]:
        raise RuntimeError(f"Unexpected FanOutQA LLM-judge dependency: {source}")
    return answer_in_text, str_answer, scorer, source


def flatten_answer(value: Any) -> List[str]:
    if isinstance(value, dict):
        items: List[str] = []
        for key, child in value.items():
            items.extend((str(key), str(child)))
        return items
    if isinstance(value, list):
        return [str(child) for child in value]
    if isinstance(value, bool):
        return ["yes" if value else "no"]
    return [str(value)]


def normalized_titles(values: Iterable[Any]) -> set[str]:
    return {normalize_text(value) for value in values if normalize_text(value)}


def average(rows: Sequence[Dict[str, Any]], key: str) -> float:
    return mean(float(row.get(key, 0.0)) for row in rows) if rows else 0.0


def aggregate(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"records": 0}
    result: Dict[str, Any] = {"records": len(rows)}
    for key in METRICS + (
        "answer_valid", "necessary_title_any_retrieved",
        "necessary_title_all_retrieved", "necessary_title_any_visible",
        "necessary_title_all_visible", "necessary_title_retrieval_recall",
        "necessary_title_visibility_recall",
        "answer_component_evidence_coverage", "search_calls",
        "retrieved_docs_total", "visible_evidence_tokens_total",
        "mean_docs_per_search", "retrieval_shortfall_round_rate",
        "zero_result_round_rate", "underfilled_evidence_tokens_total",
        "standalone_llm_input_tokens", "standalone_llm_output_tokens",
        "standalone_model_calls",
    ):
        result[key] = average(rows, key)
    result["local_fallback_rate"] = 1.0 - result["answer_valid"]
    return result


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["records"]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def condition_sort(value: str):
    if value == "FULL":
        return (1, 0)
    if value == "BLANK":
        return (2, 0)
    try:
        return (0, int(value))
    except ValueError:
        return (3, value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_mode", choices=("budget", "blank"), required=True)
    args = parser.parse_args()

    answer_in_text, str_answer, rouge, evaluator_source = require_official_eval()
    labels = {str(row["sample_id"]): row for row in iter_jsonl(args.labels)}
    trajectories = list(iter_jsonl(args.trajectories))
    backends = sorted({str(row.get("retriever_backend") or "") for row in trajectories})
    snapshots = sorted({str(row.get("wikipedia_snapshot") or "") for row in trajectories})
    expected_backend = (
        "fanoutqa_wikipedia_en_all_nopic_2023-09_kiwix_passage100_upto5"
    )
    if backends != [expected_backend] or snapshots != ["2023-09"]:
        raise RuntimeError(
            f"Corpus provenance mismatch: backends={backends}, snapshots={snapshots}"
        )
    scored: List[Dict[str, Any]] = []
    missing_labels: List[str] = []

    for row in trajectories:
        sample_id = str(row["sample_id"])
        label = labels.get(sample_id)
        if label is None:
            missing_labels.append(sample_id)
            continue
        prediction = str(row.get("final_answer") or "")
        valid = row.get("answer_status") == "valid"
        # Score the stored final_answer independently of its protocol/format
        # status.  A local fallback remains explicitly marked below, but it is
        # not silently replaced with an empty answer.  This keeps answer
        # quality and output-format reliability as two separate measurements.
        official = answer_in_text(label["answer"], prediction)
        rouge_scores = rouge.score(
            str_answer(label["answer"]), str_answer(prediction)
        )
        loose = float(official.score)
        strict = float(official.found)
        missing_components = list(official.missing)

        compatibility = answer_metrics(prediction, [str_answer(label["answer"])])
        gold_titles = normalized_titles(label.get("necessary_titles") or [])
        retrieved_titles: set[str] = set()
        visible_titles: set[str] = set()
        visible_text_parts: List[str] = []
        for search_round in row.get("search_rounds") or []:
            evidence = search_round.get("evidence") or {}
            for doc in evidence.get("retrieved_docs") or []:
                title = normalize_text(doc.get("title"))
                if title:
                    retrieved_titles.add(title)
            for doc in evidence.get("visible_passages") or []:
                if int(doc.get("actual_visible_tokens", 0)) > 0:
                    title = normalize_text(doc.get("title"))
                    visible_text = str(doc.get("visible_text") or "")
                    if title and title in normalize_text(visible_text):
                        visible_titles.add(title)
                    visible_text_parts.append(visible_text)

        retrieved_hits = gold_titles & retrieved_titles
        visible_hits = gold_titles & visible_titles
        denominator = len(gold_titles) or 1
        evidence_components = answer_in_text(
            label["answer"], "\n".join(visible_text_parts)
        )
        usage = row.get("usage") or {}
        rounds = row.get("search_rounds") or []
        round_doc_counts = [
            len((item.get("evidence") or {}).get("retrieved_docs") or [])
            for item in rounds
        ]
        round_count = len(round_doc_counts) or 1
        shortfall_rounds = sum(count < 5 for count in round_doc_counts)
        zero_rounds = sum(count == 0 for count in round_doc_counts)
        scored.append({
            "sample_id": sample_id,
            "question": row.get("question"),
            "condition_id": row.get("condition_id"),
            "condition_label": str(row.get("condition_label")),
            "exposure_mode": row.get("exposure_mode"),
            "budget": int(row.get("root_exposure_tokens", 0)),
            "condition_mode": args.condition_mode,
            "treatment": row.get("treatment"),
            "retriever_backend": row.get("retriever_backend"),
            "wikipedia_snapshot": row.get("wikipedia_snapshot"),
            "categories": label.get("categories") or [],
            "decomposition_leaf_count": int(label.get("decomposition_leaf_count", 0)),
            "necessary_evidence_count": len(gold_titles),
            "gold_answer": label["answer"],
            "final_answer": prediction,
            "answer_status": row.get("answer_status"),
            "local_fallback": not valid,
            "answer_valid": float(valid),
            "official_loose_accuracy": loose,
            "official_strict_accuracy": strict,
            "official_missing_components": missing_components,
            "rouge1_f1": float(rouge_scores["rouge1"].fmeasure),
            "rouge2_f1": float(rouge_scores["rouge2"].fmeasure),
            "rougeL_f1": float(rouge_scores["rougeL"].fmeasure),
            "compat_em": float(compatibility["em"]),
            "compat_f1": float(compatibility["f1"]),
            "answer_f1": float(compatibility["f1"]),
            "necessary_titles": sorted(gold_titles),
            "retrieved_necessary_titles": sorted(retrieved_hits),
            "visible_necessary_titles": sorted(visible_hits),
            "necessary_title_retrieval_recall": len(retrieved_hits) / denominator,
            "necessary_title_visibility_recall": len(visible_hits) / denominator,
            "necessary_title_any_retrieved": float(bool(retrieved_hits)),
            "necessary_title_all_retrieved": float(bool(gold_titles) and gold_titles <= retrieved_titles),
            "necessary_title_any_visible": float(bool(visible_hits)),
            "necessary_title_all_visible": float(bool(gold_titles) and gold_titles <= visible_titles),
            "answer_component_evidence_coverage": float(evidence_components.score),
            "search_calls": int(row.get("search_calls", 0)),
            "retrieved_docs_total": int(row.get("retrieved_docs_total", 0)),
            "retrieved_docs_by_round": round_doc_counts,
            "mean_docs_per_search": sum(round_doc_counts) / round_count,
            "retrieval_shortfall_rounds": shortfall_rounds,
            "retrieval_shortfall_round_rate": shortfall_rounds / round_count,
            "zero_result_rounds": zero_rounds,
            "zero_result_round_rate": zero_rounds / round_count,
            "any_retrieval_shortfall": float(shortfall_rounds > 0),
            "any_zero_result": float(zero_rounds > 0),
            "underfilled_evidence_tokens_total": int(
                row.get("requested_evidence_token_caps_total", 0)
            ) - int(row.get("visible_evidence_tokens_total", 0)),
            "visible_evidence_tokens_total": int(row.get("visible_evidence_tokens_total", 0)),
            "standalone_llm_input_tokens": int(usage.get("standalone_llm_input_tokens", 0)),
            "standalone_llm_output_tokens": int(usage.get("standalone_llm_output_tokens", 0)),
            "standalone_model_calls": int(usage.get("standalone_model_calls", 0)),
            "context_compactions": int(row.get("context_compactions", 0)),
        })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl_gz(output / "scored_records.jsonl.gz", scored)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        grouped[row["condition_label"]].append(row)
    by_condition = [
        {"condition_label": label, **aggregate(values)}
        for label, values in sorted(
            grouped.items(), key=lambda item: condition_sort(item[0])
        )
    ]
    write_csv(output / "by_condition.csv", by_condition)

    categories: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        for category in row["categories"] or ["uncategorized"]:
            categories[str(category)].append(row)
    write_csv(output / "by_category.csv", [
        {"category": category, **aggregate(values)}
        for category, values in sorted(categories.items())
    ])
    condition_categories: Dict[tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        for category in row["categories"] or ["uncategorized"]:
            condition_categories[(row["condition_label"], str(category))].append(row)
    write_csv(output / "by_condition_category.csv", [
        {
            "condition_label": condition,
            "category": category,
            **aggregate(values),
        }
        for (condition, category), values in sorted(
            condition_categories.items(),
            key=lambda item: (condition_sort(item[0][0]), item[0][1]),
        )
    ])

    hit_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        name = "all" if row["necessary_title_all_visible"] else (
            "partial" if row["necessary_title_any_visible"] else "none"
        )
        hit_groups[name].append(row)
    write_csv(output / "by_evidence_visibility.csv", [
        {"visible_necessary_evidence": name, **aggregate(values)}
        for name, values in sorted(hit_groups.items())
    ])

    cardinality_groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        name = "any_zero" if row["any_zero_result"] else (
            "shortfall_1_to_4" if row["any_retrieval_shortfall"] else "all_rounds_k5"
        )
        cardinality_groups[name].append(row)
    write_csv(output / "by_retrieval_cardinality.csv", [
        {"retrieval_cardinality": name, **aggregate(values)}
        for name, values in sorted(cardinality_groups.items())
    ])

    summary = {
        "created_at": utc_stamp(),
        "dataset": "FanOutQA",
        "split": "dev",
        "condition_mode": args.condition_mode,
        "retriever_backend": expected_backend,
        "wikipedia_snapshot": "2023-09",
        "records": len(scored),
        "unique_samples": len({row["sample_id"] for row in scored}),
        "missing_label_records": len(missing_labels),
        "overall": aggregate(scored),
        "by_condition": by_condition,
        "metric_policy": {
            "primary": "FanOutQA official loose and strict accuracy",
            "secondary": "ROUGE-1/2/L F1",
            "diagnostic": "Hotpot/SQuAD-style compatibility EM/F1",
            "invalid_outputs": (
                "score the stored recovered final_answer normally; preserve "
                "answer_status, answer_valid, and local_fallback separately"
            ),
            "not_run": "BLEURT and proprietary GPT judge",
        },
        "official_evaluator_source": evaluator_source,
        "evidence_policy": (
            "Title matching between necessary_evidence and actually retrieved or "
            "visible Wikipedia passages; title recall is not sentence entailment."
        ),
        "retrieval_cardinality_policy": {
            "requested_max_k": 5,
            "accepted_actual_k": "0..5",
            "query_rewriting": False,
            "backfill": False,
            "zero_result_is_system_failure": False,
        },
    }
    write_json(output / "summary.json", summary)
    report = [
        "# FanOutQA 实验报告",
        "",
        f"- 条件：{args.condition_mode}；记录 {len(scored)}；题目 {summary['unique_samples']}。",
        f"- Official Loose Accuracy：{summary['overall']['official_loose_accuracy']:.4f}。",
        f"- Official Strict Accuracy：{summary['overall']['official_strict_accuracy']:.4f}。",
        f"- ROUGE-L F1：{summary['overall']['rougeL_f1']:.4f}。",
        f"- 有效格式率：{summary['overall']['answer_valid']:.4f}。",
        f"- 必要证据标题检索召回：{summary['overall']['necessary_title_retrieval_recall']:.4f}。",
        f"- 必要证据标题可见召回：{summary['overall']['necessary_title_visibility_recall']:.4f}。",
        f"- 平均每轮实际检索文档数：{summary['overall']['mean_docs_per_search']:.4f}。",
        f"- 检索不足 5 篇的轮次率：{summary['overall']['retrieval_shortfall_round_rate']:.4f}。",
        f"- 零结果轮次率：{summary['overall']['zero_result_round_rate']:.4f}。",
        "",
        "主指标复用 FanOutQA answer_in_text 与 ROUGE 实现；兼容 EM/F1 仅用于旧实验横向诊断。",
        "BLEURT 与需要外部 API 的 GPT judge 未运行。",
    ]
    (output / "REPORT_ZH.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    if missing_labels:
        raise RuntimeError(f"Missing sealed labels for {len(missing_labels)} records")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
