#!/usr/bin/env python3
"""Run a configured number of TALE-EP rollouts with at most four Search actions."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List

from common import (
    MAX_SEARCHES,
    METHOD,
    PROTOCOL_VARIANT,
    ROLLOUTS,
    SCHEMA_VERSION,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    append_jsonl,
    canonical_hash,
    chat_completion,
    new_request_id,
    parse_action,
    read_jsonl,
    request_attempts,
    retriever_search,
    sha256_file,
    sha256_text,
    summarize_request_attempts,
    utc_now,
    verify_qwen3,
    write_json,
)
from prompts import AGENT_SYSTEM, TALE_EP_BUDGET_SENTENCE, agent_prompt


def local_token_count(tokenizer: Any, messages: List[Dict[str, str]]) -> int:
    try:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    return len(tokenizer.encode(rendered, add_special_tokens=False))


def fallback_answer(raw: str) -> str:
    text = str(raw or "").strip()
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1].strip()
    text = re.sub(r"</?think>", "", text, flags=re.I).strip()
    return text


def solve(
    *,
    tokenizer: Any,
    question: Dict[str, Any],
    estimate: Dict[str, Any],
    rollout_id: int,
    model_url: str,
    model_name: str,
    retriever_url: str,
    max_searches: int,
    top_k: int,
    technical_max_tokens: int,
    server_max_total_tokens: int,
    min_completion_tokens: int,
    timeout: int,
    request_ledger_path: str | Path,
    signature_hash: str,
    rollouts: int,
) -> Dict[str, Any]:
    search_rounds: List[Dict[str, Any]] = []
    model_turns: List[Dict[str, Any]] = []
    completion_tokens_used = 0
    prediction = ""
    answer_status = "missing"
    forced_finish_retry_used = False

    while True:
        user_prompt = agent_prompt(
            question=str(question["question"]),
            estimated_budget=int(estimate["estimated_budget"]),
            completion_tokens_used=completion_tokens_used,
            search_rounds=search_rounds,
            max_searches=max_searches,
        )
        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]
        prompt_tokens_local = local_token_count(tokenizer, messages)
        available = server_max_total_tokens - prompt_tokens_local - 128
        effective_max_tokens = min(technical_max_tokens, available)
        if effective_max_tokens < min_completion_tokens:
            raise RuntimeError(
                "Full accumulated Top-5 evidence exceeds verified model capacity without "
                f"truncation: prompt={prompt_tokens_local}, pool={server_max_total_tokens}, "
                f"available_completion={available}"
            )
        request_id = new_request_id()
        result = chat_completion(
            base_url=model_url,
            model=model_name,
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            timeout=timeout,
            request_ledger_path=request_ledger_path,
            request_id=request_id,
            request_context={
                "stage": "agent_trajectories",
                "signature_hash": signature_hash,
                "sample_index": int(question["sample_index"]),
                "sample_id": str(question["sample_id"]),
                "rollout_id": rollout_id,
                "turn_index": len(model_turns) + 1,
            },
        )
        completion_tokens = int(result["usage"].get("completion_tokens") or 0)
        usage_source = "server"
        if completion_tokens <= 0 and result["text"]:
            completion_tokens = len(
                tokenizer.encode(result["text"], add_special_tokens=False)
            )
            usage_source = "local_tokenizer_fallback"
        before = completion_tokens_used
        completion_tokens_used += completion_tokens
        action_text = str(result["text"] or "")
        if "</think>" in action_text:
            action_text = action_text.rsplit("</think>", 1)[-1].strip()
        action = parse_action(action_text)
        append_jsonl(
            request_ledger_path,
            {
                "event": "consumer_result",
                "created_at": utc_now(),
                "request_id": request_id,
                "outcome": f"parsed_{action['type']}",
                "signature_hash": signature_hash,
                "sample_id": str(question["sample_id"]),
                "rollout_id": rollout_id,
                "turn_index": len(model_turns) + 1,
            },
        )
        model_turns.append(
            {
                "physical_request_id": request_id,
                "turn_index": len(model_turns) + 1,
                "estimated_total_output_budget": int(estimate["estimated_budget"]),
                "completion_tokens_before": before,
                "completion_tokens_after": completion_tokens_used,
                "budget_remaining_after": int(estimate["estimated_budget"])
                - completion_tokens_used,
                "budget_overrun_after": max(
                    0, completion_tokens_used - int(estimate["estimated_budget"])
                ),
                "prompt_tokens_local": prompt_tokens_local,
                "technical_max_tokens": technical_max_tokens,
                "effective_max_tokens": effective_max_tokens,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "usage": result["usage"],
                "completion_tokens_charged": completion_tokens,
                "usage_source": usage_source,
                "finish_reason": result["finish_reason"],
                "latency_seconds": result["latency_seconds"],
                "raw_output": result["text"],
                "raw_output_sha256": sha256_text(result["text"]),
                "visible_action_text": action_text,
                "parsed_action": action,
                "user_prompt_sha256": sha256_text(user_prompt),
            }
        )

        if action["type"] == "search" and len(search_rounds) < max_searches:
            retrieval = retriever_search(retriever_url, action["query"], top_k=top_k)
            retrieval["search_round"] = len(search_rounds) + 1
            retrieval["evidence_tokens_local"] = sum(
                len(tokenizer.encode(document["contents"], add_special_tokens=False))
                for document in retrieval["documents"]
            )
            search_rounds.append(retrieval)
            continue

        if action["type"] == "answer":
            prediction = action["answer"]
            answer_status = "valid"
            break

        if action["type"] == "search" and len(search_rounds) >= max_searches:
            if not forced_finish_retry_used:
                forced_finish_retry_used = True
                continue
            answer_status = "search_limit_violation"
            prediction = fallback_answer(result["text"])
            break

        prediction = fallback_answer(result["text"])
        if prediction:
            answer_status = (
                "technical_length_truncated_local_fallback"
                if result["finish_reason"] == "length"
                else "invalid_format_local_fallback"
            )
        else:
            answer_status = "empty_output"
        break

    solver_usage = {
        "prompt_tokens": sum(int(turn["usage"].get("prompt_tokens") or 0) for turn in model_turns),
        "completion_tokens": sum(
            int(turn["completion_tokens_charged"]) for turn in model_turns
        ),
        "total_tokens": sum(int(turn["usage"].get("total_tokens") or 0) for turn in model_turns),
        "model_calls": len(model_turns),
        "search_calls": len(search_rounds),
        "retrieved_documents": len(search_rounds) * top_k,
        "raw_evidence_tokens_once_per_search": sum(
            int(row["evidence_tokens_local"]) for row in search_rounds
        ),
    }
    estimator_usage = estimate.get("usage") or {}
    amortized_estimator = {
        key: float(estimator_usage.get(key) or 0) / rollouts
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "protocol_variant": PROTOCOL_VARIANT,
        "sample_index": int(question["sample_index"]),
        "sample_id": str(question["sample_id"]),
        "question": str(question["question"]),
        "rollout_id": rollout_id,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "estimated_budget": int(estimate["estimated_budget"]),
        "estimate_record_hash": estimate["record_hash"],
        "logical_estimator_record_shared_by_rollouts": 1,
        "budget_control": "TALE-EP prompt-level soft output-token target",
        "tale_budget_sentence": TALE_EP_BUDGET_SENTENCE.format(
            budget=int(estimate["estimated_budget"])
        ),
        "maximum_allowed_searches": max_searches,
        "top_k_per_search": top_k,
        "search_rounds": search_rounds,
        "model_turns": model_turns,
        "prediction": prediction,
        "prediction_sha256": sha256_text(prediction),
        "answer_status": answer_status,
        "solver_usage": solver_usage,
        "successful_estimator_record_usage_shared": estimator_usage,
        "successful_estimator_record_usage_amortized_per_rollout": amortized_estimator,
        "successful_attempt_end_to_end_amortized_total_tokens": float(
            solver_usage["total_tokens"]
        )
        + amortized_estimator["total_tokens"],
        "budget_compliant": solver_usage["completion_tokens"]
        <= int(estimate["estimated_budget"]),
        "budget_overrun_tokens": max(
            0, solver_usage["completion_tokens"] - int(estimate["estimated_budget"])
        ),
        "gold_labels_loaded": False,
    }


def main() -> int:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--estimates", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="Qwen3-8B")
    parser.add_argument("--retriever-url", default="http://127.0.0.1:6008")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=ROLLOUTS)
    parser.add_argument("--max-searches", type=int, default=MAX_SEARCHES)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--technical-max-tokens", type=int, default=8192)
    parser.add_argument("--server-max-total-tokens", type=int, default=24497)
    parser.add_argument("--min-completion-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    if args.rollouts != ROLLOUTS or args.rollouts not in (1, 2):
        raise RuntimeError(
            f"Configured protocol requires {ROLLOUTS} rollout(s); got {args.rollouts}"
        )
    if args.max_searches != 4 or args.top_k != 5:
        raise RuntimeError("Registered protocol requires max 4 searches and Top-5")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard id")
    questions_path = Path(args.questions).resolve()
    estimates_path = Path(args.estimates).resolve()
    questions = read_jsonl(questions_path)
    estimates = {str(row["sample_id"]): row for row in read_jsonl(estimates_path)}
    question_ids = {str(row["sample_id"]) for row in questions}
    unknown_estimates = sorted(set(estimates) - question_ids)
    if unknown_estimates:
        raise RuntimeError(f"Estimates contain unknown sample ids: {unknown_estimates}")
    missing_estimate_ids = sorted(question_ids - set(estimates))
    if not estimates:
        raise RuntimeError("No valid TALE estimates are available")
    tasks = []
    for question in questions:
        if str(question["sample_id"]) not in estimates:
            continue
        for rollout_id in range(args.rollouts):
            global_index = int(question["sample_index"]) * args.rollouts + rollout_id
            if global_index % args.num_shards == args.shard_id:
                tasks.append((question, rollout_id))

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / f"shard_{args.shard_id}.jsonl"
    failures_path = output / f"shard_{args.shard_id}.failures.jsonl"
    ledger_path = output / f"shard_{args.shard_id}.physical_requests.jsonl"
    manifest_path = output / f"shard_{args.shard_id}.manifest.json"
    info = verify_qwen3(args.model_url, args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    signature = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "protocol_variant": PROTOCOL_VARIANT,
        "stage": "agent_trajectories",
        "questions_sha256": sha256_file(questions_path),
        "estimates_sha256": sha256_file(estimates_path),
        "model_path": str(Path(args.model_path).resolve()),
        "model_url": args.model_url,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "rollouts": args.rollouts,
        "max_searches": args.max_searches,
        "top_k": args.top_k,
        "technical_max_tokens": args.technical_max_tokens,
        "server_max_total_tokens": args.server_max_total_tokens,
        "seed_policy": (
            "No explicit seed: the installed SGLang backend previously rejected seed; "
            f"{args.rollouts} physical temperature=0.1 draw(s) define the rollout set."
        ),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    signature_hash = canonical_hash(signature)
    existing_rows = read_jsonl(records_path) if records_path.exists() else []
    existing = {
        (str(row["sample_id"]), int(row["rollout_id"])): row for row in existing_rows
    }
    if len(existing) != len(existing_rows):
        raise RuntimeError("Duplicate trajectory key")
    if any(row.get("signature_hash") != signature_hash for row in existing_rows):
        raise RuntimeError("Existing trajectory shard has a different signature")

    succeeded = failed = 0
    started_all = time.time()
    for ordinal, (question, rollout_id) in enumerate(tasks, 1):
        key = (str(question["sample_id"]), rollout_id)
        if key in existing:
            continue
        try:
            record = solve(
                tokenizer=tokenizer,
                question=question,
                estimate=estimates[key[0]],
                rollout_id=rollout_id,
                model_url=args.model_url,
                model_name=args.model_name,
                retriever_url=args.retriever_url,
                max_searches=args.max_searches,
                top_k=args.top_k,
                technical_max_tokens=args.technical_max_tokens,
                server_max_total_tokens=args.server_max_total_tokens,
                min_completion_tokens=args.min_completion_tokens,
                timeout=args.timeout,
                request_ledger_path=ledger_path,
                signature_hash=signature_hash,
                rollouts=args.rollouts,
            )
            record["signature_hash"] = signature_hash
            record["model_path_verified"] = str(info.get("model_path") or "")
            record["record_hash"] = canonical_hash(record)
            append_jsonl(records_path, record)
            existing[key] = record
            succeeded += 1
            elapsed = max(time.time() - started_all, 1e-9)
            remaining = len(tasks) - len(existing)
            eta = elapsed / max(succeeded + failed, 1) * remaining / 3600
            print(
                f"[trajectory-ok] {ordinal}/{len(tasks)} {key[0]} r={rollout_id} "
                f"B={record['estimated_budget']} searches={record['solver_usage']['search_calls']} "
                f"status={record['answer_status']} eta={eta:.2f}h",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            append_jsonl(
                failures_path,
                {
                    "created_at": utc_now(),
                    "sample_id": key[0],
                    "rollout_id": key[1],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "signature_hash": signature_hash,
                },
            )
            print(f"[trajectory-failed] {key}: {exc}", flush=True)

    expected_keys = {(str(question["sample_id"]), rollout) for question, rollout in tasks}
    missing = sorted(expected_keys - set(existing))
    ledger_summary = summarize_request_attempts(request_attempts([ledger_path]))
    manifest = {
        "created_at": utc_now(),
        "signature": signature,
        "signature_hash": signature_hash,
        "model_info": info,
        "expected_records": len(tasks),
        "records": len(existing),
        "succeeded_this_attempt": succeeded,
        "failed_this_attempt": failed,
        "unresolved_failures": len(missing),
        "unresolved_keys": missing,
        "complete": not missing,
        "output": str(records_path),
        "physical_request_ledger": str(ledger_path),
        "physical_request_accounting": ledger_summary,
        "estimator_coverage": {
            "questions": len(questions),
            "valid_estimates": len(estimates),
            "missing_estimates": len(missing_estimate_ids),
            "missing_sample_ids": missing_estimate_ids,
        },
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
