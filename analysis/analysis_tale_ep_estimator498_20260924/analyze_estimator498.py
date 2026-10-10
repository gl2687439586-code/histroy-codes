#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values, probability):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def ranks(values):
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2 + 1
        for offset in range(cursor, end):
            result[order[offset]] = rank
        cursor = end
    return result


def pearson(left, right):
    if len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - mean_left) ** 2 for x in left)
        * sum((y - mean_right) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def created_at(event):
    return datetime.fromisoformat(str(event["created_at"]).replace("Z", "+00:00"))


questions = read_jsonl(ROOT / "dataset/questions.no_labels.jsonl")
question_by_id = {str(row["sample_id"]): row for row in questions}
estimates = []
failures = []
events = []
manifests = []
for shard in range(2):
    estimates.extend(read_jsonl(ROOT / f"estimates/shards/shard_{shard}.jsonl"))
    failures.extend(read_jsonl(ROOT / f"estimates/shards/shard_{shard}.failures.jsonl"))
    events.extend(read_jsonl(ROOT / f"estimates/shards/shard_{shard}.physical_requests.jsonl"))
    manifests.append(json.loads((ROOT / f"estimates/shards/shard_{shard}.manifest.json").read_text(encoding="utf-8")))

estimate_by_id = {str(row["sample_id"]): row for row in estimates}
question_ids = set(question_by_id)
missing_ids = sorted(question_ids - set(estimate_by_id))

attempts = {}
for event in events:
    request_id = str(event["request_id"])
    attempt = attempts.setdefault(request_id, {"events": []})
    attempt["events"].append(event)
    if event["event"] == "request_started":
        attempt["started"] = event
        attempt["sample_id"] = str((event.get("context") or {}).get("sample_id") or "")
    elif event["event"] == "request_finished":
        attempt["finished"] = event
    elif event["event"] == "consumer_result":
        attempt["consumer"] = event

by_sample = defaultdict(list)
for attempt in attempts.values():
    by_sample[attempt["sample_id"]].append(attempt)
for rows in by_sample.values():
    rows.sort(key=lambda row: created_at(row["started"]))

accepted = [row for row in attempts.values() if (row.get("consumer") or {}).get("outcome") == "accepted_estimate"]
parse_failed = [row for row in attempts.values() if (row.get("consumer") or {}).get("outcome") == "downstream_parse_error"]
finished = [row for row in attempts.values() if row.get("finished")]
unknown_usage = [row for row in finished if not row["finished"].get("usage_available")]

usage = Counter()
for attempt in finished:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        usage[key] += int(((attempt["finished"].get("usage") or {}).get(key)) or 0)

first_attempt_success = 0
recovered_after_retry = []
attempt_counts = {}
sample_total_tokens = {}
success_attempt_number = {}
for sample_id in sorted(question_ids):
    rows = by_sample.get(sample_id, [])
    attempt_counts[sample_id] = len(rows)
    sample_total_tokens[sample_id] = sum(
        int((((row.get("finished") or {}).get("usage") or {}).get("total_tokens")) or 0)
        for row in rows
    )
    outcomes = [(row.get("consumer") or {}).get("outcome") for row in rows]
    if outcomes and outcomes[0] == "accepted_estimate":
        first_attempt_success += 1
        success_attempt_number[sample_id] = 1
    elif "accepted_estimate" in outcomes:
        number = outcomes.index("accepted_estimate") + 1
        success_attempt_number[sample_id] = number
        recovered_after_retry.append(sample_id)

budgets = [int(row["estimated_budget"]) for row in estimates]
question_words = [len(str(row["question"]).split()) for row in estimates]
accepted_completion = [int(row["usage"]["completion_tokens"]) for row in estimates]
accepted_total = [int(row["usage"]["total_tokens"]) for row in estimates]

failure_finish = Counter(str(row.get("finish_reason")) for row in failures)
failure_brackets = Counter()
for row in failures:
    raw = str(row.get("raw_output") or "")
    bracket_values = re.findall(r"\[\[(.*?)\]\]", raw, flags=re.S)
    if not bracket_values:
        failure_brackets["no_double_bracket"] += 1
    elif any(re.fullmatch(r"\s*[0-9][0-9,]*\s*", item) for item in bracket_values):
        failure_brackets["integer_bracket_but_rejected"] += 1
    else:
        failure_brackets["non_integer_double_bracket"] += 1

finish_reasons_accepted = Counter(str(row.get("finish_reason")) for row in estimates)
budget_counts = Counter(budgets)
attempt_distribution = Counter(attempt_counts.values())

top_attempts = sorted(
    (
        {
            "sample_id": sample_id,
            "question": str(question_by_id[sample_id]["question"]),
            "attempts": attempts_count,
            "accepted": sample_id in estimate_by_id,
            "success_attempt_number": success_attempt_number.get(sample_id),
            "physical_estimator_tokens": sample_total_tokens[sample_id],
            "estimated_budget": (estimate_by_id.get(sample_id) or {}).get("estimated_budget"),
        }
        for sample_id, attempts_count in attempt_counts.items()
    ),
    key=lambda row: (-row["attempts"], -row["physical_estimator_tokens"], row["sample_id"]),
)[:20]

highest_budgets = sorted(
    (
        {
            "sample_id": str(row["sample_id"]),
            "question": str(row["question"]),
            "budget": int(row["estimated_budget"]),
            "completion_tokens": int(row["usage"]["completion_tokens"]),
            "attempts": attempt_counts[str(row["sample_id"])],
        }
        for row in estimates
    ),
    key=lambda row: (-row["budget"], row["sample_id"]),
)[:20]

summary = {
    "integrity": {
        "questions": len(questions),
        "unique_questions": len(question_ids),
        "valid_estimates": len(estimates),
        "unique_estimates": len(estimate_by_id),
        "missing_estimates": len(missing_ids),
        "missing_sample_ids": missing_ids,
        "missing_questions": [
            {"sample_id": sample_id, "question": question_by_id[sample_id]["question"]}
            for sample_id in missing_ids
        ],
        "gold_keys_in_generation_questions": sorted(
            {
                key
                for row in questions
                for key in row
                if key in {"answer", "answers", "gold_answer", "gold_answers", "supporting_facts"}
            }
        ),
        "models": sorted({str(row.get("model_path_verified")) for row in estimates}),
        "temperatures": sorted({float(row["temperature"]) for row in estimates}),
        "top_p": sorted({float(row["top_p"]) for row in estimates}),
        "ledger_request_ids_unique": len(attempts) == len([row for row in events if row["event"] == "request_started"]),
        "all_requests_finished": len(finished) == len(attempts),
        "unknown_usage_requests": len(unknown_usage),
        "failure_rows": len(failures),
        "parse_failure_attempts": len(parse_failed),
        "failure_rows_match_parse_attempts": len(failures) == len(parse_failed),
    },
    "coverage_and_recovery": {
        "first_attempt_valid": first_attempt_success,
        "first_attempt_valid_rate": first_attempt_success / len(questions),
        "first_attempt_invalid": len(questions) - first_attempt_success,
        "recovered_after_retry": len(recovered_after_retry),
        "recovery_rate_among_first_attempt_failures": len(recovered_after_retry) / (len(questions) - first_attempt_success),
        "persistent_failures": len(missing_ids),
        "final_coverage_rate": len(estimates) / len(questions),
        "physical_requests": len(attempts),
        "extra_requests_above_one_per_question": len(attempts) - len(questions),
        "request_overhead_rate": (len(attempts) - len(questions)) / len(questions),
        "accepted_requests": len(accepted),
        "parse_failed_requests": len(parse_failed),
        "physical_request_parse_failure_rate": len(parse_failed) / len(attempts),
        "attempts_per_question_distribution": dict(sorted(attempt_distribution.items())),
        "mean_attempts_per_question": statistics.fmean(attempt_counts.values()),
        "max_attempts_for_one_question": max(attempt_counts.values()),
    },
    "physical_cost": {
        **dict(usage),
        "mean_prompt_tokens_per_request": usage["prompt_tokens"] / len(attempts),
        "mean_completion_tokens_per_request": usage["completion_tokens"] / len(attempts),
        "mean_total_tokens_per_request": usage["total_tokens"] / len(attempts),
        "mean_total_tokens_per_question": usage["total_tokens"] / len(questions),
        "sum_predicted_budgets": sum(budgets),
        "estimator_completion_to_sum_predicted_budget_ratio": usage["completion_tokens"] / sum(budgets),
    },
    "budget_distribution": {
        "count": len(budgets),
        "min": min(budgets),
        "p05": percentile(budgets, 0.05),
        "p25": percentile(budgets, 0.25),
        "median": percentile(budgets, 0.5),
        "mean": statistics.fmean(budgets),
        "p75": percentile(budgets, 0.75),
        "p95": percentile(budgets, 0.95),
        "max": max(budgets),
        "counts": {str(key): value for key, value in sorted(budget_counts.items())},
        "budget_le_3": sum(value <= 3 for value in budgets),
        "budget_le_5": sum(value <= 5 for value in budgets),
        "budget_le_10": sum(value <= 10 for value in budgets),
        "budget_le_20": sum(value <= 20 for value in budgets),
    },
    "generation_behavior": {
        "accepted_finish_reasons": dict(sorted(finish_reasons_accepted.items())),
        "failure_finish_reasons": dict(sorted(failure_finish.items())),
        "failure_bracket_classes": dict(sorted(failure_brackets.items())),
        "accepted_completion_tokens": {
            "mean": statistics.fmean(accepted_completion),
            "median": percentile(accepted_completion, 0.5),
            "p95": percentile(accepted_completion, 0.95),
            "max": max(accepted_completion),
        },
        "accepted_total_tokens": {
            "mean": statistics.fmean(accepted_total),
            "median": percentile(accepted_total, 0.5),
            "p95": percentile(accepted_total, 0.95),
            "max": max(accepted_total),
        },
    },
    "correlations": {
        "pearson_question_words_vs_budget": pearson(question_words, budgets),
        "spearman_question_words_vs_budget": pearson(ranks(question_words), ranks(budgets)),
        "pearson_estimator_completion_vs_budget": pearson(accepted_completion, budgets),
        "spearman_estimator_completion_vs_budget": pearson(ranks(accepted_completion), ranks(budgets)),
    },
    "persistent_failure_details": [
        {
            "sample_id": sample_id,
            "question": question_by_id[sample_id]["question"],
            "attempts": attempt_counts[sample_id],
            "physical_estimator_tokens": sample_total_tokens[sample_id],
            "finish_reasons": dict(
                Counter(str((row.get("finished") or {}).get("finish_reason")) for row in by_sample[sample_id])
            ),
            "consumer_outcomes": dict(
                Counter(str((row.get("consumer") or {}).get("outcome")) for row in by_sample[sample_id])
            ),
        }
        for sample_id in missing_ids
    ],
    "top_attempt_cost_cases": top_attempts,
    "highest_budget_cases": highest_budgets,
}

(ROOT / "ESTIMATOR_AUDIT.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
