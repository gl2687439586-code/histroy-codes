from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    LEVEL_BUDGETS,
    append_jsonl,
    chat_completion,
    extract_json_object,
    read_jsonl,
    sha256_file,
    utc_now,
    verify_model_path,
    write_json,
)
from .prompts import (
    PLANNER_DECOMPOSE_SYSTEM,
    PLANNER_EVALUATE_SYSTEM,
    planner_decompose_user,
    planner_evaluate_user,
)


def validate_decomposition(value: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = value.get("subquestions")
    if not isinstance(raw, list) or not 2 <= len(raw) <= 5:
        raise ValueError("Planner must return 2 to 5 subquestions")
    result = []
    for index, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Subquestion {index} is not an object")
        question = str(item.get("subquestion") or "").strip()
        hint = str(item.get("hint") or "").strip()
        if not question or not hint:
            raise ValueError(f"Subquestion {index} lacks subquestion or hint")
        result.append({"index": index, "subquestion": question, "hint": hint})
    return result


def validate_evaluation(
    value: Dict[str, Any], subquestions: List[Dict[str, str]]
) -> Dict[str, Any]:
    problem = value.get("problem")
    rows = value.get("subquestions")
    if not isinstance(problem, dict) or not isinstance(rows, list):
        raise ValueError("Evaluation lacks problem/subquestions")
    level = int(problem.get("evaluated_level"))
    if level not in LEVEL_BUDGETS:
        raise ValueError(f"Invalid problem level: {level}")
    problem_reason = str(problem.get("reason") or "").strip()
    forbidden_reasons = {
        "at most 50 words",
        "brief concrete justification",
    }
    if not problem_reason or problem_reason.lower() in forbidden_reasons:
        raise ValueError("Problem difficulty reason is empty or a schema placeholder")
    if len(rows) != len(subquestions):
        raise ValueError("Evaluation subquestion count changed")
    evaluated = []
    for expected_index, item in enumerate(rows, 1):
        if not isinstance(item, dict):
            raise ValueError(f"Evaluation item {expected_index} is not an object")
        index = int(item.get("index", expected_index))
        item_level = int(item.get("evaluated_level"))
        credit = int(item.get("credit"))
        reason = str(item.get("reason") or "").strip()
        if index != expected_index:
            raise ValueError("Evaluation reordered subquestions")
        if item_level not in LEVEL_BUDGETS or credit <= 0:
            raise ValueError(f"Invalid level/credit at subquestion {index}")
        if not reason or reason.lower() in forbidden_reasons:
            raise ValueError(
                f"Difficulty reason at subquestion {index} is empty or a schema placeholder"
            )
        evaluated.append(
            {
                **subquestions[index - 1],
                "evaluated_level": item_level,
                "credit": credit,
                "difficulty_reason": reason,
            }
        )
    if sum(x["credit"] for x in evaluated) != 100:
        raise ValueError(
            f"Credits must sum to 100, got {sum(x['credit'] for x in evaluated)}"
        )
    return {
        "level": level,
        "total_output_budget": LEVEL_BUDGETS[level],
        "problem_difficulty_reason": problem_reason,
        "subquestions": evaluated,
    }


def invoke_with_validation(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    validator,
    max_attempts: int,
    max_tokens: int,
) -> tuple[Any, List[Dict[str, Any]]]:
    attempts = []
    for attempt_index in range(1, max_attempts + 1):
        started = time.time()
        try:
            response = chat_completion(
                base_url,
                model,
                messages,
                max_tokens=max_tokens,
            )
            parsed = extract_json_object(response["text"])
            result = validator(parsed)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "ok": True,
                    "latency_seconds": time.time() - started,
                    "response": response,
                }
            )
            return result, attempts
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt_index,
                    "ok": False,
                    "latency_seconds": time.time() - started,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    raise RuntimeError(json.dumps(attempts, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--planner-url", default="http://127.0.0.1:9103")
    parser.add_argument("--planner-model", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--expected-model-fragment", default="Llama-3.1-8B-Instruct")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    args = parser.parse_args()

    questions_path = Path(args.questions).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    plans_path = output_dir / "plans.jsonl"
    failures_path = output_dir / "failures.jsonl"
    questions = read_jsonl(questions_path)
    if args.limit is not None:
        questions = questions[: args.limit]

    model_info = verify_model_path(args.planner_url, args.expected_model_fragment)
    existing = {}
    if plans_path.is_file():
        existing = {row["sample_id"]: row for row in read_jsonl(plans_path)}

    succeeded = 0
    failed = 0
    for ordinal, question in enumerate(questions, 1):
        sample_id = str(question["sample_id"])
        if sample_id in existing:
            print(f"[skip] {ordinal}/{len(questions)} {sample_id}", flush=True)
            continue
        text = str(question["question"])
        try:
            decomposition, decomposition_attempts = invoke_with_validation(
                base_url=args.planner_url,
                model=args.planner_model,
                messages=[
                    {"role": "system", "content": PLANNER_DECOMPOSE_SYSTEM},
                    {"role": "user", "content": planner_decompose_user(text)},
                ],
                validator=validate_decomposition,
                max_attempts=args.max_attempts,
                max_tokens=args.max_tokens,
            )
            evaluation, evaluation_attempts = invoke_with_validation(
                base_url=args.planner_url,
                model=args.planner_model,
                messages=[
                    {"role": "system", "content": PLANNER_EVALUATE_SYSTEM},
                    {
                        "role": "user",
                        "content": planner_evaluate_user(text, decomposition),
                    },
                ],
                validator=lambda value: validate_evaluation(value, decomposition),
                max_attempts=args.max_attempts,
                max_tokens=args.max_tokens,
            )
            row = {
                "schema_version": "pandb-hotpot-plan-v2.0",
                "created_at": utc_now(),
                "sample_id": sample_id,
                "source_index": question.get("source_index"),
                "question": text,
                "planner_model_info": model_info,
                "planner_model_requested": args.planner_model,
                "level": evaluation["level"],
                "total_output_budget": evaluation["total_output_budget"],
                "problem_difficulty_reason": evaluation["problem_difficulty_reason"],
                "subquestions": evaluation["subquestions"],
                "planner_calls": {
                    "decomposition": decomposition_attempts,
                    "difficulty_and_credits": evaluation_attempts,
                },
            }
            append_jsonl(plans_path, row)
            existing[sample_id] = row
            succeeded += 1
            print(
                f"[ok] {ordinal}/{len(questions)} {sample_id} "
                f"level={row['level']} B={row['total_output_budget']} "
                f"steps={len(row['subquestions'])}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            append_jsonl(
                failures_path,
                {
                    "created_at": utc_now(),
                    "sample_id": sample_id,
                    "question": text,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"[failed] {sample_id}: {exc}", flush=True)

    complete = len(existing) == len(questions)
    manifest = {
        "created_at": utc_now(),
        "questions": len(questions),
        "complete_plans": len(existing),
        "succeeded_this_attempt": succeeded,
        "failed_this_attempt": failed,
        "complete": complete,
        "questions_sha256": sha256_file(questions_path),
        "plans": str(plans_path),
        "model_info": model_info,
        "planner_calls_per_successful_question": 2,
        "planner_shared_across_schedulers": True,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
