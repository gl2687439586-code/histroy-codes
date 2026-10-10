from __future__ import annotations

import argparse
import gzip
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    pick_answer,
    pick_id,
    read_jsonl,
    score_prediction,
    sha256_file,
    utc_now,
    write_json,
)


def mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_unique(path: Path, expected: int) -> Dict[str, Dict[str, Any]]:
    rows = read_jsonl(path)
    result = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in result:
            raise RuntimeError(f"Duplicate sample in {path}: {sample_id}")
        result[sample_id] = row
    if len(result) != expected:
        raise RuntimeError(f"Expected {expected} records in {path}, found {len(result)}")
    return result


def planner_usage(plan: Dict[str, Any]) -> Dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for phase in ("decomposition", "difficulty_and_credits"):
        attempts = plan.get("planner_calls", {}).get(phase, [])
        successes = [x for x in attempts if x.get("ok")]
        if not successes:
            continue
        usage = successes[-1].get("response", {}).get("usage", {})
        for key in totals:
            totals[key] += int(usage.get(key) or 0)
    return totals


def validate_trajectory(
    row: Dict[str, Any], plan: Dict[str, Any], scheduler: str
) -> List[str]:
    errors: List[str] = []
    sample_id = str(row.get("sample_id"))
    budget = int(plan["total_output_budget"])
    if row.get("scheduler") != scheduler:
        errors.append(f"{sample_id}: wrong scheduler")
    if int(row.get("rollout", -1)) != 0:
        errors.append(f"{sample_id}: rollout is not 0")
    if int(row.get("official_total_output_budget", -1)) != budget:
        errors.append(f"{sample_id}: total budget differs from plan")
    if row.get("budget_control") != "prompt_soft_target_no_enforcement":
        errors.append(f"{sample_id}: budget control is not the v2 soft policy")
    allocations = [int(x) for x in row.get("subquestion_output_allocations", [])]
    if sum(allocations) != budget:
        errors.append(f"{sample_id}: allocations sum {sum(allocations)} != {budget}")
    if len(allocations) != len(plan.get("subquestions", [])):
        errors.append(f"{sample_id}: allocation/plan step count mismatch")
    usage = row.get("usage", {})
    completion_tokens = int(usage.get("completion_tokens") or 0)
    expected_overrun = max(0, completion_tokens - budget)
    if int(usage.get("soft_budget_target_tokens") or -1) != budget:
        errors.append(f"{sample_id}: usage soft-budget target differs from plan")
    if int(usage.get("soft_budget_overrun_tokens") or 0) != expected_overrun:
        errors.append(f"{sample_id}: incorrect record-level soft-budget overrun")
    if bool(usage.get("soft_budget_compliant")) != (completion_tokens <= budget):
        errors.append(f"{sample_id}: incorrect record-level compliance flag")
    if row.get("gold_labels_loaded_during_generation") is not False:
        errors.append(f"{sample_id}: generation may have loaded gold labels")
    if row.get("raw_evidence_crosses_subquestion_boundary") is not False:
        errors.append(f"{sample_id}: raw evidence crossed a subquestion boundary")
    if row.get("retrieval_calls_shared_across_schedulers") is not False:
        errors.append(f"{sample_id}: retrieval was marked shared")
    step_actual_total = 0
    for step_index, step in enumerate(row.get("steps", []), 1):
        step_target = int(step.get("soft_budget_target_tokens") or -1)
        step_actual = int(step.get("consumed_output_tokens") or 0)
        step_actual_total += step_actual
        if step_target != int(step.get("allocated_output_tokens") or -1):
            errors.append(f"{sample_id}: step {step_index} target/allocation mismatch")
        if int(step.get("soft_budget_overrun_tokens") or 0) != max(
            0, step_actual - step_target
        ):
            errors.append(f"{sample_id}: step {step_index} incorrect overrun")
        if bool(step.get("soft_budget_compliant")) != (step_actual <= step_target):
            errors.append(f"{sample_id}: step {step_index} incorrect compliance flag")
        rounds = step.get("search_rounds", [])
        if len(rounds) > 4:
            errors.append(f"{sample_id}: step {step_index} exceeded 4 searches")
        for search in rounds:
            if len(search.get("documents", [])) != 5:
                errors.append(
                    f"{sample_id}: step {step_index} search did not return Top-5"
                )
    if step_actual_total != completion_tokens:
        errors.append(
            f"{sample_id}: step actual total {step_actual_total} != record completion {completion_tokens}"
        )
    return errors


def summarize(
    rows: Dict[str, Dict[str, Any]],
    labels: Dict[str, Any],
    plans: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    scored = []
    for sample_id, row in rows.items():
        em, f1 = score_prediction(str(row.get("prediction") or ""), labels[sample_id])
        usage = row["usage"]
        plan_usage = planner_usage(plans[sample_id])
        statuses = [step.get("answer_status") for step in row.get("steps", [])]
        scored.append(
            {
                "sample_id": sample_id,
                "em": em,
                "f1": f1,
                "official_budget": int(row["official_total_output_budget"]),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "soft_budget_overrun_tokens": int(
                    usage.get("soft_budget_overrun_tokens") or 0
                ),
                "soft_budget_compliant": bool(usage.get("soft_budget_compliant")),
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "server_total_tokens": int(usage.get("server_total_tokens") or 0),
                "evidence_tokens": int(
                    usage.get("evidence_tokens_raw_once_per_search_result") or 0
                ),
                "search_calls": int(usage.get("search_calls") or 0),
                "model_calls": int(usage.get("model_calls") or 0),
                "latency_seconds": float(row.get("latency_seconds") or 0.0),
                "planner_prompt_tokens": plan_usage["prompt_tokens"],
                "planner_completion_tokens": plan_usage["completion_tokens"],
                "planner_total_tokens": plan_usage["total_tokens"],
                "has_fallback": any(
                    status not in ("valid", "valid_subanswer_as_final_fallback")
                    for status in statuses
                ),
                "final_answer_status": statuses[-1] if statuses else "missing",
            }
        )

    em = mean([x["em"] for x in scored])
    f1 = mean([x["f1"] for x in scored])
    avg_completion = mean([x["completion_tokens"] for x in scored])
    avg_reasoner_total = mean([x["server_total_tokens"] for x in scored])
    avg_planner_total = mean([x["planner_total_tokens"] for x in scored])
    by_level = {}
    for level in range(1, 6):
        subset = [x for x in scored if plans[x["sample_id"]]["level"] == level]
        by_level[str(level)] = {
            "n": len(subset),
            "em": mean([x["em"] for x in subset]),
            "f1": mean([x["f1"] for x in subset]),
            "mean_completion_tokens": mean([x["completion_tokens"] for x in subset]),
            "mean_search_calls": mean([x["search_calls"] for x in subset]),
        }

    return {
        "n": len(scored),
        "em": em,
        "f1": f1,
        "mean_official_budget": mean([x["official_budget"] for x in scored]),
        "mean_reasoner_completion_tokens": avg_completion,
        "mean_soft_budget_overrun_tokens": mean(
            [x["soft_budget_overrun_tokens"] for x in scored]
        ),
        "soft_budget_compliance_rate": mean(
            [float(x["soft_budget_compliant"]) for x in scored]
        ),
        "soft_budget_overrun_question_rate": mean(
            [float(not x["soft_budget_compliant"]) for x in scored]
        ),
        "mean_reasoner_prompt_tokens": mean([x["prompt_tokens"] for x in scored]),
        "mean_reasoner_total_tokens": avg_reasoner_total,
        "mean_raw_evidence_tokens_once_per_search_result": mean(
            [x["evidence_tokens"] for x in scored]
        ),
        "mean_search_calls": mean([x["search_calls"] for x in scored]),
        "mean_model_calls": mean([x["model_calls"] for x in scored]),
        "mean_latency_seconds": mean([x["latency_seconds"] for x in scored]),
        "mean_planner_prompt_tokens": mean(
            [x["planner_prompt_tokens"] for x in scored]
        ),
        "mean_planner_completion_tokens": mean(
            [x["planner_completion_tokens"] for x in scored]
        ),
        "mean_planner_total_tokens": avg_planner_total,
        "mean_end_to_end_model_tokens": avg_reasoner_total + avg_planner_total,
        "fallback_question_rate": mean([float(x["has_fallback"]) for x in scored]),
        "e3_em_squared_over_completion": (em * em / avg_completion)
        if avg_completion
        else 0.0,
        "e3_f1_squared_over_completion": (f1 * f1 / avg_completion)
        if avg_completion
        else 0.0,
        "by_planner_level": by_level,
        "per_sample": scored,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--expected", type=int, default=500)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    labels_path = Path(args.labels).resolve()
    plans_path = Path(args.plans).resolve()
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    labels = {pick_id(row): pick_answer(row) for row in read_jsonl(labels_path)}
    plans = {row["sample_id"]: row for row in read_jsonl(plans_path)}
    if len(labels) < args.expected or len(plans) != args.expected:
        raise RuntimeError(
            f"Expected at least {args.expected} labels and exactly {args.expected} plans, "
            f"got labels={len(labels)} plans={len(plans)}"
        )
    labels = {sample_id: labels[sample_id] for sample_id in plans}

    scheduler_rows = {}
    summaries = {}
    source_hashes = {}
    integrity_errors: List[str] = []
    for scheduler in ("polynomial", "cosine"):
        path = run_dir / "trajectories" / scheduler / "records.jsonl"
        rows = load_unique(path, args.expected)
        if set(rows) != set(labels) or set(rows) != set(plans):
            raise RuntimeError(f"ID mismatch for {scheduler}")
        for sample_id, row in rows.items():
            integrity_errors.extend(validate_trajectory(row, plans[sample_id], scheduler))
        scheduler_rows[scheduler] = rows
        summaries[scheduler] = summarize(rows, labels, plans)
        source_hashes[scheduler] = sha256_file(path)

        gz_path = analysis_dir / f"{scheduler}.records.jsonl.gz"
        with path.open("rb") as source, gzip.open(gz_path, "wb", compresslevel=6) as target:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                target.write(chunk)

    paired = []
    for sample_id in sorted(labels):
        p = next(
            x for x in summaries["polynomial"]["per_sample"] if x["sample_id"] == sample_id
        )
        c = next(x for x in summaries["cosine"]["per_sample"] if x["sample_id"] == sample_id)
        paired.append(
            {
                "sample_id": sample_id,
                "polynomial_em": p["em"],
                "cosine_em": c["em"],
                "polynomial_f1": p["f1"],
                "cosine_f1": c["f1"],
                "delta_f1_polynomial_minus_cosine": p["f1"] - c["f1"],
                "polynomial_completion_tokens": p["completion_tokens"],
                "cosine_completion_tokens": c["completion_tokens"],
                "polynomial_soft_budget_overrun_tokens": p[
                    "soft_budget_overrun_tokens"
                ],
                "cosine_soft_budget_overrun_tokens": c[
                    "soft_budget_overrun_tokens"
                ],
                "polynomial_soft_budget_compliant": p["soft_budget_compliant"],
                "cosine_soft_budget_compliant": c["soft_budget_compliant"],
                "polynomial_search_calls": p["search_calls"],
                "cosine_search_calls": c["search_calls"],
            }
        )

    # Keep detailed per-sample data in a separate file; summary stays compact.
    for scheduler in summaries:
        detail = summaries[scheduler].pop("per_sample")
        output = analysis_dir / f"{scheduler}.scored.jsonl"
        with output.open("w", encoding="utf-8", newline="\n") as f:
            for row in detail:
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    paired_path = analysis_dir / "paired.jsonl"
    with paired_path.open("w", encoding="utf-8", newline="\n") as f:
        for row in paired:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    if integrity_errors:
        raise RuntimeError(
            f"Trajectory integrity failed with {len(integrity_errors)} errors: "
            + json.dumps(integrity_errors[:50], ensure_ascii=False)
        )

    summary = {
        "created_at": utc_now(),
        "passed": True,
        "expected_questions": args.expected,
        "records_per_scheduler": args.expected,
        "trajectory_integrity_errors": [],
        "schedulers": summaries,
        "paired": {
            "mean_f1_delta_polynomial_minus_cosine": mean(
                [x["delta_f1_polynomial_minus_cosine"] for x in paired]
            ),
            "polynomial_em_only": sum(
                x["polynomial_em"] > x["cosine_em"] for x in paired
            ),
            "cosine_em_only": sum(
                x["cosine_em"] > x["polynomial_em"] for x in paired
            ),
            "both_em": sum(
                x["cosine_em"] == 1 and x["polynomial_em"] == 1 for x in paired
            ),
            "neither_em": sum(
                x["cosine_em"] == 0 and x["polynomial_em"] == 0 for x in paired
            ),
        },
        "source_sha256": {
            "labels": sha256_file(labels_path),
            "plans": sha256_file(plans_path),
            **source_hashes,
        },
        "metric_notes": {
            "em_f1": "Standard normalized HotpotQA answer EM and token F1.",
            "e3": "E^3=A^2/T, with T=mean actual reasoner decoding/completion tokens; nominal soft budgets are never substituted for actual usage.",
            "planner_cost": "Reported separately and in end-to-end model tokens; the same frozen plan is shared by both schedulers.",
            "retrieval": "Each scheduler independently invokes deterministic Top-5 retrieval; evidence is not frozen or shared.",
            "budget_control": "P&B allocations are prompt-level soft targets. They never set API max_tokens, stop a turn, or stop the experiment. Overruns are only measured and reported.",
        },
    }
    write_json(analysis_dir / "summary.json", summary)

    lines = [
        f"# Plan-and-Budget × HotpotQA {args.expected} 题结果",
        "",
        "## 完整性",
        "",
        f"- 每种调度轨迹：{args.expected}；Polynomial/Cosine 共 {args.expected * 2} 条。",
        "- 两种调度共享同一份 Llama-3.1-8B-Instruct 规划；检索与 Qwen3 推理各自独立执行。",
        "- 生成阶段未加载金标答案。",
        "",
        "## 主结果",
        "",
        "| 调度 | EM | F1 | 平均软预算 | 平均实际解码 token | 平均超额 token | 遵从率 | 平均 Search | 平均模型总 token | E³(EM) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scheduler in ("polynomial", "cosine"):
        s = summaries[scheduler]
        lines.append(
            f"| {scheduler.title()} | {s['em']:.4f} | {s['f1']:.4f} | "
            f"{s['mean_official_budget']:.1f} | "
            f"{s['mean_reasoner_completion_tokens']:.1f} | "
            f"{s['mean_soft_budget_overrun_tokens']:.1f} | "
            f"{s['soft_budget_compliance_rate']:.1%} | "
            f"{s['mean_search_calls']:.2f} | "
            f"{s['mean_end_to_end_model_tokens']:.1f} | "
            f"{s['e3_em_squared_over_completion']:.8f} |"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "这不是原论文的逐字复现：原论文没有检索工具。本实验保留其 Planner、五档总预算、credits 与 Polynomial/Cosine 调度，并按实验要求加入 Qwen3 自主 Search。Top-5 全文属于输入证据；200–600 及其子问题分配是 Prompt 软目标。超过目标不触发截断、停止、重试或删除，只记录实际解码 token、超额量与遵从率。API max_tokens 仅为统一的技术安全上限。",
            "",
        ]
    )
    (analysis_dir / "REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
