#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "full498_r1_t01"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(rows, key):
    return statistics.fmean(float(row[key]) for row in rows) if rows else 0.0


def rankdata(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def pearson(xs, ys):
    if len(xs) < 2:
        return 0.0
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else 0.0


def summarize(rows):
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "em": mean(rows, "em"),
        "f1": mean(rows, "f1"),
        "acc": mean(rows, "acc"),
        "valid_rate": statistics.fmean(
            1.0 if row["answer_status"] == "valid" else 0.0 for row in rows
        ),
        "mean_budget": mean(rows, "estimated_budget"),
        "mean_solver_completion_tokens": mean(rows, "solver_completion_tokens"),
        "mean_solver_total_tokens": mean(rows, "solver_total_tokens"),
        "mean_search_calls": mean(rows, "search_calls"),
        "budget_compliance_rate": statistics.fmean(
            1.0 if row["budget_compliant"] else 0.0 for row in rows
        ),
    }


summary = read_json(RUN / "analysis" / "summary.json")
audit = read_json(RUN / "analysis" / "AUDIT.json")
trajectory_manifest = read_json(
    RUN / "trajectories" / "trajectories.merged.jsonl.manifest.json"
)
estimate_manifest = read_json(RUN / "estimates" / "estimates.merged.jsonl.manifest.json")
questions = read_jsonl(RUN / "dataset" / "questions.no_labels.jsonl")
labels = read_jsonl(RUN / "dataset" / "gold_labels.sealed.jsonl")
estimates = read_jsonl(RUN / "estimates" / "estimates.merged.jsonl")
trajectories = read_jsonl(RUN / "trajectories" / "trajectories.merged.jsonl")
scored = read_jsonl(RUN / "analysis" / "scored_records.jsonl")

real_scored = [row for row in scored if row["record_kind"] == "solver_trajectory"]
score_by_key = {(row["sample_id"], row["rollout_id"]): row for row in real_scored}
joined = []
for trajectory in trajectories:
    score = score_by_key[(trajectory["sample_id"], trajectory["rollout_id"])]
    joined.append(
        {
            "sample_id": trajectory["sample_id"],
            "estimated_budget": int(trajectory["estimated_budget"]),
            "em": float(score["em"]),
            "f1": float(score["f1"]),
            "acc": float(score["acc"]),
            "answer_status": trajectory["answer_status"],
            "search_calls": int(trajectory["solver_usage"]["search_calls"]),
            "model_calls": int(trajectory["solver_usage"]["model_calls"]),
            "solver_completion_tokens": int(
                trajectory["solver_usage"]["completion_tokens"]
            ),
            "solver_total_tokens": int(trajectory["solver_usage"]["total_tokens"]),
            "budget_compliant": bool(trajectory["budget_compliant"]),
            "budget_overrun_tokens": int(trajectory["budget_overrun_tokens"]),
            "finish_reasons": [
                turn.get("finish_reason") for turn in trajectory.get("model_turns", [])
            ],
        }
    )

assert sha256(RUN / "trajectories" / "trajectories.merged.jsonl") == trajectory_manifest[
    "output_sha256"
]
assert sha256(RUN / "estimates" / "estimates.merged.jsonl") == estimate_manifest[
    "output_sha256"
]
assert len(questions) == len(labels) == 500
assert len(estimates) == len(trajectories) == len(real_scored) == len(joined) == 498
assert len({row["sample_id"] for row in estimates}) == 498
assert len({row["sample_id"] for row in trajectories}) == 498
assert audit["passed"] is True

status_groups = defaultdict(list)
search_groups = defaultdict(list)
for row in joined:
    status_groups[row["answer_status"]].append(row)
    search_groups[str(row["search_calls"])].append(row)


def budget_bucket(value):
    if value <= 3:
        return "01-03"
    if value <= 11:
        return "04-11"
    if value == 12:
        return "12"
    if value <= 20:
        return "13-20"
    return ">20"


budget_groups = defaultdict(list)
for row in joined:
    budget_groups[budget_bucket(row["estimated_budget"])].append(row)

exact_budget = defaultdict(list)
for row in joined:
    exact_budget[str(row["estimated_budget"])].append(row)

budgets = [row["estimated_budget"] for row in joined]
completion = [row["solver_completion_tokens"] for row in joined]
f1 = [row["f1"] for row in joined]

result = {
    "integrity": {
        "audit_passed": audit["passed"],
        "audit_errors": audit["errors"],
        "questions": len(questions),
        "valid_estimates": len(estimates),
        "persistent_estimator_failures": 500 - len(estimates),
        "real_trajectory_records": len(trajectories),
        "analysis_records_including_zero_failures": len(scored),
        "trajectory_manifest_passed": trajectory_manifest["passed"],
        "trajectory_sha256_verified": True,
        "estimate_sha256_verified": True,
        "gold_absent_from_generation_inputs": audit[
            "gold_labels_absent_from_generation_inputs"
        ],
        "rollouts": audit["rollouts_per_question"],
        "temperature": audit["temperature"],
        "top_k": audit["top_k_per_search"],
        "maximum_searches": audit["maximum_searches"],
    },
    "quality": {
        "solver_only_498": summary["solver_only_valid_estimates"],
        "end_to_end_500": {
            key: summary["overall"][key]
            for key in (
                "em",
                "f1",
                "acc",
                "valid_answer_rate",
                "mean_search_calls",
            )
        },
    },
    "cost": {
        "estimator": summary["physical_cost_entire_run"]["estimator"],
        "solver": summary["physical_cost_entire_run"]["solver"],
        "end_to_end": summary["physical_cost_entire_run"]["end_to_end"],
        "mean_end_to_end_tokens_solver_only_498": summary[
            "solver_only_valid_estimates"
        ]["mean_end_to_end_total_tokens"],
        "mean_end_to_end_tokens_all_500": summary["overall"][
            "mean_end_to_end_amortized_total_tokens_lower_bound"
        ],
        "predicted_budget_sum": sum(budgets),
        "actual_solver_completion_sum": sum(completion),
        "actual_to_predicted_solver_completion_ratio": sum(completion) / sum(budgets),
    },
    "budget_control": {
        "distribution": summary["budget_estimates"],
        "compliant_records": sum(row["budget_compliant"] for row in joined),
        "compliance_rate": statistics.fmean(
            1.0 if row["budget_compliant"] else 0.0 for row in joined
        ),
        "median_completion_to_budget_ratio": statistics.median(
            row["solver_completion_tokens"] / row["estimated_budget"] for row in joined
        ),
        "spearman_budget_vs_completion": pearson(
            rankdata(budgets), rankdata(completion)
        ),
        "spearman_budget_vs_f1": pearson(rankdata(budgets), rankdata(f1)),
        "buckets": {
            key: summarize(rows) for key, rows in sorted(budget_groups.items())
        },
        "most_common_exact_budgets": [
            {"budget": int(value), **summarize(rows)}
            for value, rows in sorted(
                exact_budget.items(), key=lambda item: (-len(item[1]), int(item[0]))
            )[:12]
        ],
    },
    "behavior": {
        "answer_status": {
            key: summarize(rows) for key, rows in sorted(status_groups.items())
        },
        "search_calls": {
            key: summarize(rows) for key, rows in sorted(search_groups.items())
        },
        "model_call_distribution": dict(
            sorted(Counter(row["model_calls"] for row in joined).items())
        ),
        "all_turn_finish_reasons": dict(
            sorted(
                Counter(
                    reason
                    for row in joined
                    for reason in row["finish_reasons"]
                ).items()
            )
        ),
        "trajectory_answer_status_counts": dict(
            sorted(Counter(row["answer_status"] for row in joined).items())
        ),
    },
}

(ROOT / "INDEPENDENT_AUDIT.json").write_text(
    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(result, ensure_ascii=False, indent=2))
