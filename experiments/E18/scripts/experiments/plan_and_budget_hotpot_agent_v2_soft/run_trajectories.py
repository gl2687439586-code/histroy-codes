from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .common import (
    allocate_tokens,
    append_jsonl,
    chat_completion,
    parse_action,
    read_jsonl,
    retriever_search,
    sha256_file,
    sha256_text,
    utc_now,
    verify_model_path,
    write_json,
)
from .prompts import REASONER_SYSTEM, reasoner_user


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def health(base_url: str) -> Dict[str, Any]:
    response = requests.get(base_url.rstrip("/") + "/health", timeout=20)
    response.raise_for_status()
    try:
        return response.json()
    except Exception:
        return {"body": response.text}


def solve_subquestion(
    *,
    tokenizer,
    model_url: str,
    model_name: str,
    retriever_url: str,
    question: str,
    level: int,
    total_budget: int,
    scheduler: str,
    step: Dict[str, Any],
    step_index: int,
    step_count: int,
    step_budget: int,
    previous_summaries: List[Dict[str, Any]],
    max_searches: int,
    top_k: int,
    generation_max_tokens: int,
) -> Dict[str, Any]:
    actual_output_tokens = 0
    search_rounds: List[Dict[str, Any]] = []
    model_turns: List[Dict[str, Any]] = []
    subanswer: Optional[str] = None
    final_answer: Optional[str] = None
    citations: List[str] = []
    answer_status = "missing"
    forced_finish_retry_used = False

    # The P&B allocation is a prompt-level soft target. It must never be used as
    # the API generation cap or as a reason to interrupt a complete action.
    while True:
        search_limit_reached = len(search_rounds) >= max_searches
        user_prompt = reasoner_user(
            original_question=question,
            level=level,
            total_budget=total_budget,
            scheduler=scheduler,
            subquestion_index=step_index,
            subquestion_count=step_count,
            subquestion=step["subquestion"],
            hint=step["hint"],
            subquestion_budget=step_budget,
            actual_output_tokens_so_far=actual_output_tokens,
            previous_summaries=previous_summaries,
            search_rounds=search_rounds,
            search_limit_reached=search_limit_reached,
        )
        response = chat_completion(
            model_url,
            model_name,
            [
                {"role": "system", "content": REASONER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=generation_max_tokens,
        )
        completion_tokens = int(response["usage"]["completion_tokens"])
        # Some old SGLang versions omit usage. Count locally only in that case.
        usage_source = "server"
        if completion_tokens <= 0 and response["text"]:
            completion_tokens = token_count(tokenizer, response["text"])
            usage_source = "local_tokenizer_fallback"
        actual_before = actual_output_tokens
        actual_output_tokens += completion_tokens
        action = parse_action(response["text"])
        model_turns.append(
            {
                "turn_index": len(model_turns) + 1,
                "soft_budget_target_tokens": step_budget,
                "actual_output_tokens_before": actual_before,
                "actual_output_tokens_after": actual_output_tokens,
                "soft_budget_remaining_before": step_budget - actual_before,
                "soft_budget_remaining_after": step_budget - actual_output_tokens,
                "soft_budget_overrun_after": max(
                    0, actual_output_tokens - step_budget
                ),
                "completion_tokens_charged": completion_tokens,
                "technical_generation_max_tokens": generation_max_tokens,
                "usage_source": usage_source,
                "prompt_tokens": response["usage"]["prompt_tokens"],
                "server_total_tokens": response["usage"]["total_tokens"],
                "finish_reason": response["finish_reason"],
                "latency_seconds": response["latency_seconds"],
                "raw_output": response["text"],
                "raw_output_sha256": sha256_text(response["text"]),
                "parsed_action": action,
            }
        )

        if action["type"] == "search" and not search_limit_reached:
            result = retriever_search(retriever_url, action["query"], top_k=top_k)
            result["search_round"] = len(search_rounds) + 1
            result["evidence_tokens"] = sum(
                token_count(tokenizer, doc["contents"])
                for doc in result["documents"]
            )
            search_rounds.append(result)
            continue

        if action["type"] == "finish":
            subanswer = action["subanswer"]
            final_answer = action["final_answer"]
            citations = action["citations"]
            if step_index == step_count and not final_answer:
                final_answer = subanswer
                answer_status = "valid_subanswer_as_final_fallback"
            else:
                answer_status = "valid"
            break

        if action["type"] == "search" and search_limit_reached:
            # This guard is unrelated to the soft token budget. Give one recorded
            # retry after the model ignores the explicit Finish-only instruction,
            # then terminate deterministically if it ignores the instruction again.
            if not forced_finish_retry_used:
                forced_finish_retry_used = True
                continue
            answer_status = "search_limit_violation"
            break

        # No valid tag. Preserve the single physical output as an auditable local
        # fallback; never issue a format-repair request outside the budget.
        cleaned = response["text"].strip()
        if cleaned:
            subanswer = cleaned
            if step_index == step_count:
                final_answer = cleaned
            answer_status = (
                "technical_length_truncated_local_fallback"
                if response["finish_reason"] == "length"
                else "invalid_format_local_fallback"
            )
        else:
            answer_status = "empty_output"
        break

    return {
        "step_index": step_index,
        "subquestion": step["subquestion"],
        "hint": step["hint"],
        "credit": step["credit"],
        "evaluated_level": step["evaluated_level"],
        "soft_budget_target_tokens": step_budget,
        "allocated_output_tokens": step_budget,
        "consumed_output_tokens": actual_output_tokens,
        "soft_budget_remaining_tokens": step_budget - actual_output_tokens,
        "soft_budget_overrun_tokens": max(0, actual_output_tokens - step_budget),
        "soft_budget_compliant": actual_output_tokens <= step_budget,
        "technical_generation_max_tokens": generation_max_tokens,
        "search_rounds": search_rounds,
        "model_turns": model_turns,
        "subanswer": subanswer or "",
        "final_answer": final_answer,
        "citations": citations,
        "answer_status": answer_status,
    }


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True)
    parser.add_argument("--plans", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scheduler", choices=["polynomial", "cosine"], required=True)
    parser.add_argument("--model-url", default="http://127.0.0.1:9104")
    parser.add_argument("--model-name", default="Qwen3-8B")
    parser.add_argument("--expected-model-fragment", default="Qwen3-8B")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:6008")
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--max-searches", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generation-max-tokens", type=int, default=8192)
    parser.add_argument("--rollout", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if args.max_searches != 4:
        raise RuntimeError("Protocol requires MAX_SEARCHES_PER_SUBQUESTION=4")
    if args.top_k != 5:
        raise RuntimeError("Protocol requires Top-5 retrieval")
    if args.rollout != 0:
        raise RuntimeError("Protocol requires N_ROLLOUTS=1 (rollout index 0)")
    if args.generation_max_tokens < 256:
        raise ValueError("generation-max-tokens is a technical cap and must be >= 256")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard id")

    questions_path = Path(args.questions).resolve()
    plans_path = Path(args.plans).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    failures_path = output_dir / "failures.jsonl"

    model_info = verify_model_path(args.model_url, args.expected_model_fragment)
    retriever_health = health(args.retriever_url)
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, trust_remote_code=True, local_files_only=True
    )

    questions = read_jsonl(questions_path)
    if args.limit is not None:
        questions = questions[: args.limit]
    plans = {row["sample_id"]: row for row in read_jsonl(plans_path)}
    missing_plans = [row["sample_id"] for row in questions if row["sample_id"] not in plans]
    if missing_plans:
        raise RuntimeError(f"Missing {len(missing_plans)} plans: {missing_plans[:20]}")
    shard_questions = [
        row
        for index, row in enumerate(questions)
        if index % args.num_shards == args.shard_id
    ]
    existing = {}
    if records_path.is_file():
        existing = {row["sample_id"]: row for row in read_jsonl(records_path)}

    succeeded = 0
    failed = 0
    skipped = 0
    for ordinal, question_row in enumerate(shard_questions, 1):
        sample_id = str(question_row["sample_id"])
        if sample_id in existing:
            skipped += 1
            print(f"[skip] {ordinal}/{len(shard_questions)} {sample_id}", flush=True)
            continue
        plan = plans[sample_id]
        started = time.time()
        try:
            credits = [int(step["credit"]) for step in plan["subquestions"]]
            allocations = allocate_tokens(
                credits, int(plan["total_output_budget"]), args.scheduler
            )
            step_records = []
            summaries: List[Dict[str, Any]] = []
            for step_index, (step, step_budget) in enumerate(
                zip(plan["subquestions"], allocations), 1
            ):
                step_record = solve_subquestion(
                    tokenizer=tokenizer,
                    model_url=args.model_url,
                    model_name=args.model_name,
                    retriever_url=args.retriever_url,
                    question=str(question_row["question"]),
                    level=int(plan["level"]),
                    total_budget=int(plan["total_output_budget"]),
                    scheduler=args.scheduler,
                    step=step,
                    step_index=step_index,
                    step_count=len(plan["subquestions"]),
                    step_budget=step_budget,
                    previous_summaries=summaries,
                    max_searches=args.max_searches,
                    top_k=args.top_k,
                    generation_max_tokens=args.generation_max_tokens,
                )
                step_records.append(step_record)
                summaries.append(
                    {
                        "step_index": step_index,
                        "subanswer": step_record["subanswer"],
                        "cited_document_ids": step_record["citations"],
                    }
                )

            prediction = step_records[-1].get("final_answer") or step_records[-1].get(
                "subanswer", ""
            )
            model_turns = [turn for step in step_records for turn in step["model_turns"]]
            search_rounds = [
                search for step in step_records for search in step["search_rounds"]
            ]
            completion_tokens = sum(
                int(turn["completion_tokens_charged"]) for turn in model_turns
            )
            prompt_tokens = sum(int(turn["prompt_tokens"]) for turn in model_turns)
            server_total_tokens = sum(
                int(turn["server_total_tokens"]) for turn in model_turns
            )
            evidence_tokens = sum(
                int(search["evidence_tokens"]) for search in search_rounds
            )
            soft_budget_target = int(plan["total_output_budget"])
            soft_budget_overrun = max(0, completion_tokens - soft_budget_target)
            row = {
                "schema_version": "pandb-hotpot-agent-trajectory-soft-v2.0",
                "created_at": utc_now(),
                "sample_id": sample_id,
                "source_index": question_row.get("source_index"),
                "question": question_row["question"],
                "scheduler": args.scheduler,
                "rollout": args.rollout,
                "level": plan["level"],
                "official_total_output_budget": plan["total_output_budget"],
                "budget_control": "prompt_soft_target_no_enforcement",
                "technical_generation_max_tokens_per_turn": args.generation_max_tokens,
                "credits": credits,
                "subquestion_output_allocations": allocations,
                "max_searches_per_subquestion": args.max_searches,
                "top_k": args.top_k,
                "prediction": prediction,
                "prediction_sha256": sha256_text(prediction),
                "steps": step_records,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "server_total_tokens": server_total_tokens,
                    "evidence_tokens_raw_once_per_search_result": evidence_tokens,
                    "search_calls": len(search_rounds),
                    "model_calls": len(model_turns),
                    "soft_budget_target_tokens": soft_budget_target,
                    "soft_budget_overrun_tokens": soft_budget_overrun,
                    "soft_budget_compliant": completion_tokens <= soft_budget_target,
                },
                "latency_seconds": time.time() - started,
                "reasoner_model_info": model_info,
                "retriever_health": retriever_health,
                "gold_labels_loaded_during_generation": False,
                "raw_evidence_crosses_subquestion_boundary": False,
                "plan_shared_across_schedulers": True,
                "retrieval_calls_shared_across_schedulers": False,
            }
            append_jsonl(records_path, row)
            existing[sample_id] = row
            succeeded += 1
            print(
                f"[ok] {ordinal}/{len(shard_questions)} {sample_id} "
                f"{args.scheduler} B={plan['total_output_budget']} "
                f"out={completion_tokens} overrun={soft_budget_overrun} "
                f"searches={len(search_rounds)}",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            append_jsonl(
                failures_path,
                {
                    "created_at": utc_now(),
                    "sample_id": sample_id,
                    "scheduler": args.scheduler,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            print(f"[failed] {sample_id}: {exc}", flush=True)

    manifest = {
        "created_at": utc_now(),
        "scheduler": args.scheduler,
        "rollout": args.rollout,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "questions_global": len(questions),
        "questions_this_shard": len(shard_questions),
        "records_total": len(existing),
        "expected_records": len(shard_questions),
        "succeeded_this_attempt": succeeded,
        "failed_this_attempt": failed,
        "skipped_existing": skipped,
        "complete": len(existing) == len(shard_questions),
        "questions_sha256": sha256_file(questions_path),
        "plans_sha256": sha256_file(plans_path),
        "model_info": model_info,
        "retriever_health": retriever_health,
        "top_k": args.top_k,
        "max_searches_per_subquestion": args.max_searches,
        "budget_control": "prompt_soft_target_no_enforcement",
        "technical_generation_max_tokens_per_turn": args.generation_max_tokens,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    return 0 if manifest["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
