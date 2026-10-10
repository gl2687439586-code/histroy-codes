#!/usr/bin/env python3
"""Strict protocol, integrity, and physical-request accounting audit."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from common import (
    MAX_SEARCHES,
    METHOD,
    PROTOCOL_VARIANT,
    ROLLOUTS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    canonical_hash,
    read_jsonl,
    request_attempts,
    summarize_request_attempts,
    utc_now,
    write_json,
)
from prompts import TALE_EP_BUDGET_SENTENCE, TALE_EP_ESTIMATOR_INSTRUCTION, estimator_prompt


GOLD_KEYS = {"answer", "gold_answer", "answers", "gold_answers", "supporting_facts"}


def ledger_paths(directory: str | Path) -> list[Path]:
    paths = sorted(Path(directory).resolve().glob("shard_*.physical_requests.jsonl"))
    if not paths:
        raise RuntimeError(f"No physical request ledgers under {directory}")
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--estimates", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--estimator-ledger-dir", required=True)
    parser.add_argument("--solver-ledger-dir", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-questions", type=int, required=True)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    labels = read_jsonl(args.labels)
    estimates = read_jsonl(args.estimates)
    trajectories = read_jsonl(args.trajectories)
    analysis = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    estimator_attempts = request_attempts(ledger_paths(args.estimator_ledger_dir))
    solver_attempts = request_attempts(ledger_paths(args.solver_ledger_dir))
    estimator_accounting = summarize_request_attempts(estimator_attempts)
    solver_accounting = summarize_request_attempts(solver_attempts)
    errors, warnings = [], []

    qids = [str(row["sample_id"]) for row in questions]
    if len(questions) != args.expected_questions or len(set(qids)) != len(qids):
        errors.append(
            f"Question integrity mismatch: records={len(questions)}, unique={len(set(qids))}"
        )
    if len(labels) != len(questions):
        errors.append("Label count mismatch")
    estimate_by_id = {str(row["sample_id"]): row for row in estimates}
    if len(estimate_by_id) != len(estimates) or set(estimate_by_id) - set(qids):
        errors.append("TALE estimate uniqueness or sample-id mismatch")
    missing_estimate_ids = sorted(set(qids) - set(estimate_by_id))
    if missing_estimate_ids:
        warnings.append(
            f"Persistent estimator format failures: {len(missing_estimate_ids)}; "
            "these are retained as zero-quality end-to-end failures"
        )
    trajectory_by_key = {
        (str(row["sample_id"]), int(row["rollout_id"])): row for row in trajectories
    }
    expected_keys = {
        (sample_id, rollout)
        for sample_id in estimate_by_id
        for rollout in range(ROLLOUTS)
    }
    if len(trajectories) != len(expected_keys) or set(trajectory_by_key) != expected_keys:
        errors.append(
            f"Trajectory keys mismatch: records={len(trajectories)}, unique={len(trajectory_by_key)}"
        )

    estimator_request_ids = {row["request_id"] for row in estimator_attempts}
    solver_request_ids = {row["request_id"] for row in solver_attempts}
    estimator_attempt_by_id = {
        str(row["request_id"]): row for row in estimator_attempts
    }
    estimator_attempts_by_sample = {}
    for attempt in estimator_attempts:
        context = {}
        for event in (attempt.get("started"), attempt.get("finished")):
            if event and isinstance(event.get("context"), dict):
                context = event["context"]
                break
        sample_id = str(context.get("sample_id") or "")
        estimator_attempts_by_sample.setdefault(sample_id, []).append(attempt)
    if estimator_request_ids & solver_request_ids:
        errors.append("A request_id appears in both estimator and solver ledgers")
    if not estimator_accounting["cost_ledger_complete"]:
        errors.append("Estimator request ledger has unmatched start/finish events")
    if not solver_accounting["cost_ledger_complete"]:
        errors.append("Solver request ledger has unmatched start/finish events")
    if estimator_accounting["usage_unknown_requests"]:
        warnings.append(
            f"Estimator has {estimator_accounting['usage_unknown_requests']} physical requests "
            "without server usage; token cost is a lower bound, but calls are counted"
        )
    if solver_accounting["usage_unknown_requests"]:
        warnings.append(
            f"Solver has {solver_accounting['usage_unknown_requests']} physical requests "
            "without server usage; token cost is a lower bound, but calls are counted"
        )

    for sample_id in missing_estimate_ids:
        attempts = estimator_attempts_by_sample.get(sample_id) or []
        if not attempts:
            errors.append(f"Missing estimate has no physical attempts: {sample_id}")
            continue
        outcomes = {
            str(event.get("outcome") or "")
            for attempt in attempts
            for event in (attempt.get("consumer") or [])
        }
        if "accepted_estimate" in outcomes:
            errors.append(f"Missing estimate has an accepted ledger event: {sample_id}")
        if "downstream_parse_error" not in outcomes:
            errors.append(f"Missing estimate lacks parse-failure evidence: {sample_id}")

    for question in questions:
        sample_id = str(question["sample_id"])
        if GOLD_KEYS.intersection(question):
            errors.append(f"Gold leakage in generation question: {sample_id}")
        estimate = estimate_by_id.get(sample_id)
        if estimate is None:
            continue
        if estimate.get("method") != METHOD:
            errors.append(f"Unexpected method in estimate: {sample_id}")
        if estimate.get("protocol_variant") != PROTOCOL_VARIANT:
            errors.append(f"Unexpected estimator protocol variant: {sample_id}")
        expected_prompt = estimator_prompt(str(question["question"]))
        if estimate.get("estimator_prompt") != expected_prompt:
            errors.append(f"Estimator prompt mismatch: {sample_id}")
        prompt = str(estimate.get("estimator_prompt") or "")
        if TALE_EP_ESTIMATOR_INSTRUCTION not in prompt:
            errors.append(f"Official estimator instruction missing: {sample_id}")
        if prompt != expected_prompt:
            errors.append(f"Non-paper text found in estimator prompt: {sample_id}")
        if "COMPLETE MULTI-TURN TRAJECTORY" in prompt:
            errors.append(f"Agent-specific estimator modification found: {sample_id}")
        if int(estimate.get("estimated_budget", 0)) <= 0:
            errors.append(f"Non-positive budget: {sample_id}")
        success_request_id = str(estimate.get("successful_estimator_request_id") or "")
        if success_request_id not in estimator_request_ids:
            errors.append(f"Successful estimator request absent from ledger: {sample_id}")
        else:
            attempt = estimator_attempt_by_id[success_request_id]
            if (attempt.get("finished") or {}).get("outcome") != "success":
                errors.append(f"Accepted estimate did not come from successful HTTP response: {sample_id}")
            consumer_outcomes = {
                str(event.get("outcome") or "")
                for event in attempt.get("consumer") or []
            }
            if "accepted_estimate" not in consumer_outcomes:
                errors.append(f"Accepted estimate lacks consumer ledger event: {sample_id}")
        if estimate.get("gold_labels_loaded") is not False:
            errors.append(f"Estimator label leakage flag: {sample_id}")

        for rollout in range(ROLLOUTS):
            key = (sample_id, rollout)
            row = trajectory_by_key.get(key)
            if row is None:
                continue
            if row.get("method") != METHOD:
                errors.append(f"Unexpected trajectory method: {key}")
            if row.get("protocol_variant") != PROTOCOL_VARIANT:
                errors.append(f"Unexpected trajectory protocol variant: {key}")
            if (
                float(row.get("temperature", -1)) != TEMPERATURE
                or float(row.get("top_p", -1)) != TOP_P
            ):
                errors.append(f"Sampling mismatch: {key}")
            if int(row.get("estimated_budget", -1)) != int(estimate["estimated_budget"]):
                errors.append(f"Rollouts do not share estimate: {key}")
            if row.get("estimate_record_hash") != estimate.get("record_hash"):
                errors.append(f"Estimate hash mismatch: {key}")
            if int(row.get("maximum_allowed_searches", -1)) != MAX_SEARCHES:
                errors.append(f"Max-search declaration mismatch: {key}")
            rounds = row.get("search_rounds") or []
            if len(rounds) > MAX_SEARCHES:
                errors.append(f"More than four Search calls: {key}")
            for search_round in rounds:
                documents = search_round.get("documents") or []
                if int(search_round.get("top_k", -1)) != TOP_K or len(documents) != TOP_K:
                    errors.append(f"Non-Top-5 retrieval: {key}")
            expected_sentence = TALE_EP_BUDGET_SENTENCE.format(
                budget=int(estimate["estimated_budget"])
            )
            if row.get("tale_budget_sentence") != expected_sentence:
                errors.append(f"Official budget sentence mismatch: {key}")
            for turn in row.get("model_turns") or []:
                request_id = str(turn.get("physical_request_id") or "")
                if request_id not in solver_request_ids:
                    errors.append(f"Solver turn absent from physical ledger: {key}")
            if row.get("gold_labels_loaded") is not False:
                errors.append(f"Trajectory label leakage flag: {key}")
            if row.get("answer_status") != "valid":
                warnings.append(f"Non-valid answer {key}: {row.get('answer_status')}")

    if int(analysis.get("questions", -1)) != len(questions):
        errors.append("Analysis question count mismatch")
    if int(analysis.get("records", -1)) != len(questions) * ROLLOUTS:
        errors.append("Analysis trajectory count mismatch")
    reported_cost = analysis.get("physical_cost_entire_run") or {}
    if int((reported_cost.get("estimator") or {}).get("physical_requests", -1)) != int(
        estimator_accounting["physical_requests"]
    ):
        errors.append("Analysis estimator physical request count mismatch")
    if int((reported_cost.get("solver") or {}).get("physical_requests", -1)) != int(
        solver_accounting["physical_requests"]
    ):
        errors.append("Analysis solver physical request count mismatch")

    budgets = [int(row["estimated_budget"]) for row in estimates]
    compliance = float((analysis.get("overall") or {}).get("budget_compliance_rate", 0))
    report = {
        "created_at": utc_now(),
        "passed": not errors,
        "method": "Original TALE-EP prompts on the shared search-agent interface",
        "protocol_variant": PROTOCOL_VARIANT,
        "questions": len(questions),
        "estimates": len(estimates),
        "estimator_coverage": {
            "valid_estimates": len(estimates),
            "failed_estimates": len(missing_estimate_ids),
            "coverage_rate": len(estimates) / len(questions) if questions else 0.0,
            "failed_sample_ids": missing_estimate_ids,
            "main_metric_policy": f"{ROLLOUTS} zero-quality rollout record(s) per failed estimate",
        },
        "trajectory_records": len(trajectories),
        "rollouts_per_question": ROLLOUTS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "maximum_searches": MAX_SEARCHES,
        "top_k_per_search": TOP_K,
        "total_search_calls": sum(
            len(row.get("search_rounds") or []) for row in trajectories
        ),
        "only_tale_ep": all(row.get("method") == METHOD for row in estimates + trajectories),
        "budgets_shared_across_rollouts": None if ROLLOUTS == 1 else all(
            trajectory_by_key[(sample_id, 0)]["estimated_budget"]
            == trajectory_by_key[(sample_id, 1)]["estimated_budget"]
            for sample_id in qids
            if (sample_id, 0) in trajectory_by_key
            and (sample_id, 1) in trajectory_by_key
        ),
        "gold_labels_absent_from_generation_inputs": not any(
            GOLD_KEYS.intersection(row) for row in questions
        ),
        "physical_request_accounting": {
            "estimator": estimator_accounting,
            "solver": solver_accounting,
            "all_requests_have_terminal_events": (
                estimator_accounting["cost_ledger_complete"]
                and solver_accounting["cost_ledger_complete"]
            ),
            "successful_response_parse_failures_counted": int(
                estimator_accounting["consumer_outcomes"].get(
                    "downstream_parse_error", 0
                )
            ),
        },
        "budget_semantics_diagnostic": {
            "min": min(budgets) if budgets else None,
            "median": statistics.median(budgets) if budgets else None,
            "max": max(budgets) if budgets else None,
            "budget_compliance_rate": compliance,
            "note": (
                "Interpret after the run; values are never silently clamped or calibrated. "
                "The estimator prompt is the paper prompt without agent-specific semantics."
            ),
        },
        "warnings": warnings,
        "warning_count": len(warnings),
        "errors": errors,
        "error_count": len(errors),
        "content_hashes": {
            "questions": canonical_hash(questions),
            "estimates": canonical_hash(estimates),
            "trajectories": canonical_hash(trajectories),
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
