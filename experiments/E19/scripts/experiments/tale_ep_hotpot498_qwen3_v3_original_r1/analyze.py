#!/usr/bin/env python3
"""Offline scoring plus retry-complete physical cost accounting."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from common import (
    ROLLOUTS,
    read_jsonl,
    request_attempts,
    score_prediction,
    summarize_request_attempts,
    utc_now,
    write_json,
    write_jsonl,
)


USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def percentile(values: Sequence[int], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def ledger_paths(directory: str | Path) -> List[Path]:
    paths = sorted(Path(directory).resolve().glob("shard_*.physical_requests.jsonl"))
    if not paths:
        raise RuntimeError(f"No physical request ledgers found under {directory}")
    return paths


def context_of(attempt: Dict[str, Any]) -> Dict[str, Any]:
    for event in (attempt.get("started"), attempt.get("finished")):
        if event and isinstance(event.get("context"), dict):
            return dict(event["context"])
    return {}


def usage_of(attempts: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    result = {key: 0 for key in USAGE_KEYS}
    for attempt in attempts:
        finished = attempt.get("finished") or {}
        if not finished.get("usage_available"):
            continue
        usage = finished.get("usage") or {}
        for key in USAGE_KEYS:
            result[key] += int(usage.get(key) or 0)
    return result


def unknown_usage_count(attempts: Iterable[Dict[str, Any]]) -> int:
    return sum(
        1
        for attempt in attempts
        if not (attempt.get("finished") or {}).get("usage_available")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--estimates", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--estimator-ledger-dir", required=True)
    parser.add_argument("--solver-ledger-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    labels = {str(row["sample_id"]): row for row in read_jsonl(args.labels)}
    estimates = {str(row["sample_id"]): row for row in read_jsonl(args.estimates)}
    trajectories = read_jsonl(args.trajectories)
    question_ids = {str(row["sample_id"]) for row in questions}
    if len(labels) != len(questions):
        raise RuntimeError("Question and label counts are not aligned")
    if set(estimates) - question_ids:
        raise RuntimeError("Estimate file contains unknown sample ids")
    missing_estimate_ids = sorted(question_ids - set(estimates))

    estimator_attempts = request_attempts(ledger_paths(args.estimator_ledger_dir))
    solver_attempts = request_attempts(ledger_paths(args.solver_ledger_dir))
    estimator_summary = summarize_request_attempts(estimator_attempts)
    solver_summary = summarize_request_attempts(solver_attempts)
    if not estimator_summary["cost_ledger_complete"]:
        raise RuntimeError("Estimator ledger has an unmatched request event")
    if not solver_summary["cost_ledger_complete"]:
        raise RuntimeError("Solver ledger has an unmatched request event")

    estimator_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for attempt in estimator_attempts:
        estimator_by_sample[str(context_of(attempt).get("sample_id") or "")].append(
            attempt
        )
    solver_by_key: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for attempt in solver_attempts:
        context = context_of(attempt)
        key = (
            str(context.get("sample_id") or ""),
            int(context.get("rollout_id", -1)),
        )
        solver_by_key[key].append(attempt)

    scored: List[Dict[str, Any]] = []
    for row in trajectories:
        sample_id = str(row["sample_id"])
        rollout_id = int(row["rollout_id"])
        label = labels[sample_id]
        metrics = score_prediction(
            str(row.get("prediction") or ""),
            [str(item) for item in label.get("gold_answers") or [label["answer"]]],
        )
        estimator_sample_attempts = estimator_by_sample[sample_id]
        solver_record_attempts = solver_by_key[(sample_id, rollout_id)]
        if not estimator_sample_attempts:
            raise RuntimeError(f"No estimator ledger attempts for {sample_id}")
        if not solver_record_attempts:
            raise RuntimeError(f"No solver ledger attempts for {(sample_id, rollout_id)}")
        estimator_usage = usage_of(estimator_sample_attempts)
        solver_physical_usage = usage_of(solver_record_attempts)
        amortized_estimator_total = estimator_usage["total_tokens"] / ROLLOUTS
        end_to_end_physical_total = (
            solver_physical_usage["total_tokens"] + amortized_estimator_total
        )
        scored.append(
            {
                "record_kind": "solver_trajectory",
                "sample_index": int(row["sample_index"]),
                "sample_id": sample_id,
                "rollout_id": rollout_id,
                "estimated_budget": int(row["estimated_budget"]),
                "prediction": str(row.get("prediction") or ""),
                "answer_status": row.get("answer_status"),
                **metrics,
                "search_calls": int(row["solver_usage"]["search_calls"]),
                "successful_trajectory_model_calls": int(
                    row["solver_usage"]["model_calls"]
                ),
                "successful_trajectory_solver_prompt_tokens": int(
                    row["solver_usage"]["prompt_tokens"]
                ),
                "successful_trajectory_solver_completion_tokens": int(
                    row["solver_usage"]["completion_tokens"]
                ),
                "successful_trajectory_solver_total_tokens": int(
                    row["solver_usage"]["total_tokens"]
                ),
                "physical_solver_requests_including_failed_attempts": len(
                    solver_record_attempts
                ),
                "physical_solver_prompt_tokens_lower_bound": solver_physical_usage[
                    "prompt_tokens"
                ],
                "physical_solver_completion_tokens_lower_bound": solver_physical_usage[
                    "completion_tokens"
                ],
                "physical_solver_total_tokens_lower_bound": solver_physical_usage[
                    "total_tokens"
                ],
                "physical_solver_unknown_usage_requests": unknown_usage_count(
                    solver_record_attempts
                ),
                "physical_estimator_requests_for_sample": len(
                    estimator_sample_attempts
                ),
                "physical_estimator_total_tokens_for_sample_lower_bound": estimator_usage[
                    "total_tokens"
                ],
                "physical_estimator_unknown_usage_requests_for_sample": unknown_usage_count(
                    estimator_sample_attempts
                ),
                "amortized_estimator_total_tokens_lower_bound": amortized_estimator_total,
                "end_to_end_amortized_total_tokens_lower_bound": end_to_end_physical_total,
                "budget_compliant": bool(row["budget_compliant"]),
                "budget_overrun_tokens": int(row["budget_overrun_tokens"]),
            }
        )

    questions_by_id = {str(row["sample_id"]): row for row in questions}
    for sample_id in missing_estimate_ids:
        estimator_sample_attempts = estimator_by_sample[sample_id]
        if not estimator_sample_attempts:
            raise RuntimeError(f"No estimator attempts for failed sample {sample_id}")
        estimator_usage = usage_of(estimator_sample_attempts)
        amortized_estimator_total = estimator_usage["total_tokens"] / ROLLOUTS
        for rollout_id in range(ROLLOUTS):
            scored.append(
                {
                    "record_kind": "estimator_failure",
                    "sample_index": int(questions_by_id[sample_id]["sample_index"]),
                    "sample_id": sample_id,
                    "rollout_id": rollout_id,
                    "estimated_budget": None,
                    "prediction": "",
                    "answer_status": "estimator_failure",
                    "em": 0.0,
                    "f1": 0.0,
                    "acc": 0.0,
                    "search_calls": 0,
                    "successful_trajectory_model_calls": 0,
                    "successful_trajectory_solver_prompt_tokens": 0,
                    "successful_trajectory_solver_completion_tokens": 0,
                    "successful_trajectory_solver_total_tokens": 0,
                    "physical_solver_requests_including_failed_attempts": 0,
                    "physical_solver_prompt_tokens_lower_bound": 0,
                    "physical_solver_completion_tokens_lower_bound": 0,
                    "physical_solver_total_tokens_lower_bound": 0,
                    "physical_solver_unknown_usage_requests": 0,
                    "physical_estimator_requests_for_sample": len(
                        estimator_sample_attempts
                    ),
                    "physical_estimator_total_tokens_for_sample_lower_bound": estimator_usage[
                        "total_tokens"
                    ],
                    "physical_estimator_unknown_usage_requests_for_sample": unknown_usage_count(
                        estimator_sample_attempts
                    ),
                    "amortized_estimator_total_tokens_lower_bound": amortized_estimator_total,
                    "end_to_end_amortized_total_tokens_lower_bound": amortized_estimator_total,
                    "budget_compliant": None,
                    "budget_overrun_tokens": None,
                }
            )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scored.sort(key=lambda row: (row["sample_index"], row["rollout_id"]))
    write_jsonl(output / "scored_records.jsonl", scored)

    budgets = [int(row["estimated_budget"]) for row in estimates.values()]
    solver_scored = [row for row in scored if row["record_kind"] == "solver_trajectory"]
    physical_usage = {
        key: int(estimator_summary["usage_lower_bound"][key])
        + int(solver_summary["usage_lower_bound"][key])
        for key in USAGE_KEYS
    }
    by_rollout: Dict[str, Any] = {}
    for rollout_id in range(ROLLOUTS):
        subset = [row for row in scored if row["rollout_id"] == rollout_id]
        by_rollout[str(rollout_id)] = {
            "records": len(subset),
            "em": mean([row["em"] for row in subset]),
            "f1": mean([row["f1"] for row in subset]),
            "acc": mean([row["acc"] for row in subset]),
            "mean_search_calls": mean([row["search_calls"] for row in subset]),
            "mean_successful_trajectory_solver_total_tokens": mean(
                [row["successful_trajectory_solver_total_tokens"] for row in subset]
            ),
            "mean_end_to_end_amortized_total_tokens_lower_bound": mean(
                [row["end_to_end_amortized_total_tokens_lower_bound"] for row in subset]
            ),
        }
    by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in scored:
        by_sample[row["sample_id"]].append(row)
    pair_agreement = None if ROLLOUTS != 2 else {
        "both_em_correct": sum(
            1
            for rows in by_sample.values()
            if len(rows) == ROLLOUTS and all(row["em"] for row in rows)
        ),
        "one_em_correct": sum(
            1
            for rows in by_sample.values()
            if len(rows) == ROLLOUTS and sum(row["em"] for row in rows) == 1
        ),
        "neither_em_correct": sum(
            1
            for rows in by_sample.values()
            if len(rows) == ROLLOUTS and not any(row["em"] for row in rows)
        ),
    }
    budget_sum = sum(budgets)
    estimator_completion = int(
        estimator_summary["usage_lower_bound"]["completion_tokens"]
    )
    summary = {
        "created_at": utc_now(),
        "method": "Original TALE-EP prompts on the shared search-agent interface",
        "benchmark": "HotpotQA frozen 500",
        "questions": len(questions),
        "records": len(scored),
        "rollouts_per_question": ROLLOUTS,
        "primary_aggregation": (
            f"mean over {len(questions)} questions x {ROLLOUTS} rollout(s); "
            "persistent estimator failures are explicit zero-quality end-to-end failures"
        ),
        "estimator_coverage": {
            "valid_estimates": len(estimates),
            "failed_estimates": len(missing_estimate_ids),
            "coverage_rate": len(estimates) / len(questions) if questions else 0.0,
            "failed_sample_ids": missing_estimate_ids,
            "synthetic_zero_quality_rollout_records": len(missing_estimate_ids) * ROLLOUTS,
        },
        "overall": {
            "em": mean([row["em"] for row in scored]),
            "f1": mean([row["f1"] for row in scored]),
            "acc": mean([row["acc"] for row in scored]),
            "mean_search_calls": mean([row["search_calls"] for row in scored]),
            "search_call_distribution": dict(
                sorted(Counter(row["search_calls"] for row in scored).items())
            ),
            "mean_successful_trajectory_solver_completion_tokens": mean(
                [row["successful_trajectory_solver_completion_tokens"] for row in scored]
            ),
            "mean_successful_trajectory_solver_total_tokens": mean(
                [row["successful_trajectory_solver_total_tokens"] for row in scored]
            ),
            "mean_physical_solver_total_tokens_including_failed_attempts_lower_bound": mean(
                [row["physical_solver_total_tokens_lower_bound"] for row in scored]
            ),
            "mean_end_to_end_amortized_total_tokens_lower_bound": mean(
                [row["end_to_end_amortized_total_tokens_lower_bound"] for row in scored]
            ),
            "budget_compliance_rate": mean(
                [float(row["budget_compliant"]) for row in solver_scored]
            ),
            "mean_budget_overrun_tokens": mean(
                [row["budget_overrun_tokens"] for row in solver_scored]
            ),
            "valid_answer_rate": mean(
                [float(row["answer_status"] == "valid") for row in scored]
            ),
        },
        "solver_only_valid_estimates": {
            "records": len(solver_scored),
            "questions": len({row["sample_id"] for row in solver_scored}),
            "em": mean([row["em"] for row in solver_scored]),
            "f1": mean([row["f1"] for row in solver_scored]),
            "acc": mean([row["acc"] for row in solver_scored]),
            "mean_search_calls": mean([row["search_calls"] for row in solver_scored]),
            "mean_solver_total_tokens": mean(
                [row["physical_solver_total_tokens_lower_bound"] for row in solver_scored]
            ),
            "mean_end_to_end_total_tokens": mean(
                [row["end_to_end_amortized_total_tokens_lower_bound"] for row in solver_scored]
            ),
        },
        "by_rollout": by_rollout,
        "rollout_pair_em_agreement": pair_agreement,
        "budget_estimates": {
            "count": len(budgets),
            "min": min(budgets),
            "p25": percentile(budgets, 0.25),
            "median": percentile(budgets, 0.5),
            "p75": percentile(budgets, 0.75),
            "max": max(budgets),
            "mean": mean([float(value) for value in budgets]),
            "sum": budget_sum,
            "physical_estimator_completion_to_predicted_budget_ratio": (
                estimator_completion / budget_sum if budget_sum else None
            ),
        },
        "physical_cost_entire_run": {
            "estimator": estimator_summary,
            "solver": solver_summary,
            "end_to_end": {
                "physical_model_requests": estimator_summary["physical_requests"]
                + solver_summary["physical_requests"],
                "usage_unknown_requests": estimator_summary["usage_unknown_requests"]
                + solver_summary["usage_unknown_requests"],
                "usage_lower_bound": physical_usage,
                "cost_is_exact": estimator_summary["usage_unknown_requests"] == 0
                and solver_summary["usage_unknown_requests"] == 0,
            },
            "note": (
                "Every physical request is counted, including parse failures and reruns. "
                "If a failed transport has no server usage, totals are explicitly lower bounds."
            ),
        },
        "metric_definitions": {
            "em": "Hotpot normalized exact match, max over gold aliases",
            "f1": "Hotpot normalized token F1, max over gold aliases",
            "acc": "normalized gold answer appears as a substring of normalized prediction",
        },
    }
    write_json(output / "summary.json", summary)
    report = f"""# Original-Prompt TALE-EP × Qwen3-8B × HotpotQA Agent

## 主结果

- Records: {len(scored)} = {len(questions)} questions × {ROLLOUTS} rollout(s).
- Real solver records: {len(solver_scored)} over {len(estimates)} valid-estimate questions.
- Valid TALE estimates: {len(estimates)}/{len(questions)} ({len(estimates) / len(questions):.2%}); persistent estimator failures: {len(missing_estimate_ids)}.
- Estimator-failure rollout records counted as zero quality: {len(missing_estimate_ids) * ROLLOUTS}.
- EM / F1 / ACC: {summary['overall']['em']:.4f} / {summary['overall']['f1']:.4f} / {summary['overall']['acc']:.4f}.
- Solver-only 498 EM / F1 / ACC: {summary['solver_only_valid_estimates']['em']:.4f} / {summary['solver_only_valid_estimates']['f1']:.4f} / {summary['solver_only_valid_estimates']['acc']:.4f}.
- Mean searches: {summary['overall']['mean_search_calls']:.3f} / 4.
- Mean successful-trajectory solver total tokens: {summary['overall']['mean_successful_trajectory_solver_total_tokens']:.1f}.
- Mean end-to-end amortized total tokens (including failed attempts, lower bound if unknown): {summary['overall']['mean_end_to_end_amortized_total_tokens_lower_bound']:.1f}.
- Cumulative trajectory output within the estimated budget (post-hoc diagnostic): {summary['overall']['budget_compliance_rate']:.2%}.
- Valid tagged answer rate: {summary['overall']['valid_answer_rate']:.2%}.

## 物理成本账本

- Estimator physical requests: {estimator_summary['physical_requests']}.
- Solver physical requests: {solver_summary['physical_requests']}.
- Requests with unknown usage: {summary['physical_cost_entire_run']['end_to_end']['usage_unknown_requests']}.
- End-to-end known total tokens: {physical_usage['total_tokens']}.
- Exact cost: {summary['physical_cost_entire_run']['end_to_end']['cost_is_exact']}.

所有已发出的物理请求均按 request_id 记账。请求成功但预算解析失败、轨迹失败后重跑等成本不再从汇总中消失。网络/HTTP 失败若服务端未返回 usage，会保留在请求数中，并把 token 汇总明确标记为 lower bound。

## 方法边界

估计器只使用论文 TALE-EP 原始零样本指令，没有加入 agent 轨迹定义、数值校准、预算档位或缩放。求解阶段原样使用论文的 `less than <Token-Budget> tokens` 预算句；Search/Answer 标签和 Top-5 检索仅用于接入共同实验环境。由于论文原任务是单次 CoT，而这里是多轮搜索，累计轨迹预算服从率仅作为事后诊断，不能解释成论文承诺的硬上限。
"""
    (output / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
