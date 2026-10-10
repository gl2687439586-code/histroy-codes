#!/usr/bin/env python3
"""Run 30 evidence-budget branches with natural multi-search continuation."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

from actions import generate_action, plain_answer_fallback
from hp_common import (
    BUDGETS, ExactTokenizer, SCHEMA_VERSION, SQLiteRecordStore,
    SafeSGLangClient, canonical_hash, file_sha256, normalize_text, probe_qwen,
    read_json, read_jsonl, utc_stamp, write_json,
)
from prompts import decision_prompt, forced_answer_prompt
from retrieval import CachedRetriever, render_evidence


def parse_budgets(value: str) -> List[int]:
    budgets = [int(item) for item in value.split(",") if item.strip()] if value else BUDGETS
    unknown = [item for item in budgets if item not in BUDGETS]
    if unknown:
        raise ValueError(f"Unsupported budgets: {unknown}")
    return sorted(set(budgets))


def fit_prompt(tokenizer: ExactTokenizer, question: str, rounds: List[Dict[str, Any]],
               maximum: int, effective_limit: int, force_answer: bool,
               max_searches: int) -> Tuple[str, int, bool]:
    builder = forced_answer_prompt if force_answer else None
    for retained in range(len(rounds), 0, -1):
        selected = rounds[-retained:]
        prompt = (
            builder(question, selected) if force_answer
            else decision_prompt(question, selected, max(0, max_searches - len(rounds)))
        )
        if tokenizer.count(prompt) + maximum <= effective_limit:
            return prompt, retained, retained != len(rounds)
    raise RuntimeError("Even the last evidence round exceeds the model token pool")


def attempts_usage(events: List[Dict[str, Any]]) -> Dict[str, int]:
    attempts = [attempt for event in events for attempt in event.get("attempts", [])]
    return {
        "llm_input_tokens": sum(int(item["input_tokens"]) for item in attempts),
        "llm_output_tokens": sum(int(item["output_tokens"]) for item in attempts),
        "thinking_tokens": sum(int(item.get("thinking_tokens", 0)) for item in attempts),
        "model_calls": len(attempts),
        "format_repair_calls": sum(event.get("parse_mode") == "format_repair" for event in events),
    }


def run_one(root: Dict[str, Any], budget: int, args: argparse.Namespace,
            tokenizer: ExactTokenizer, model: SafeSGLangClient,
            retriever: CachedRetriever) -> Dict[str, Any]:
    first_evidence = render_evidence(tokenizer, root["documents"], budget, 1)
    rounds: List[Dict[str, Any]] = [{
        "search_round": 1,
        "query": root["query"],
        "query_source": "shared_first_search_root",
        "query_parse_mode": root.get("query_parse_mode"),
        "retrieval_cache_hit": root.get("retrieval_cache_hit"),
        "retrieval_latency_seconds": root.get("retrieval_latency_seconds", 0.0),
        "evidence": first_evidence,
    }]
    events: List[Dict[str, Any]] = []
    context_compactions = 0
    repeated_queries = 0
    final_answer = ""
    answer_parse_mode = ""
    effective_limit = min(args.context_length, args.server_max_total_tokens)
    while True:
        force_answer = len(rounds) >= args.max_searches
        maximum = args.answer_max_new_tokens if force_answer else args.action_max_new_tokens
        prompt, retained, compacted = fit_prompt(
            tokenizer, root["question"], rounds, maximum, effective_limit,
            force_answer, args.max_searches,
        )
        context_compactions += int(compacted)
        allowed = ["answer"] if force_answer else ["search", "answer"]
        action = generate_action(model, tokenizer, prompt, allowed, maximum)
        event = {
            "decision_index": len(events) + 1,
            "searches_completed_before_decision": len(rounds),
            "forced_answer": force_answer,
            "history_rounds_retained": retained,
            "context_compacted": compacted,
            "action": action["action"],
            "payload": action["payload"],
            "parse_mode": action["parse_mode"],
            "attempts": action["attempts"],
        }
        events.append(event)
        if action["action"] == "answer" and action["payload"]:
            final_answer = action["payload"]
            answer_parse_mode = action["parse_mode"]
            break
        if action["action"] == "search" and action["payload"] and not force_answer:
            query = action["payload"]
            repeated_queries += int(
                normalize_text(query) in {normalize_text(item["query"]) for item in rounds}
            )
            documents, cache_hit, latency = retriever.search(query, 10)
            evidence = render_evidence(tokenizer, documents, budget, len(rounds) + 1)
            rounds.append({
                "search_round": len(rounds) + 1,
                "query": query,
                "query_source": "natural_model_decision",
                "query_parse_mode": action["parse_mode"],
                "retrieval_cache_hit": cache_hit,
                "retrieval_latency_seconds": latency,
                "evidence": evidence,
            })
            continue

        # No second LLM repair prompt: recover locally and mark the record explicitly.
        last_text = action["attempts"][-1]["output"] if action["attempts"] else ""
        final_answer = plain_answer_fallback(last_text) or "unknown"
        answer_parse_mode = "local_plaintext_fallback"
        break

    continuation_usage = attempts_usage(events)
    root_usage = root.get("usage") or {}
    logical_searches = len(rounds)
    visible_total = sum(item["evidence"]["visible_evidence_tokens"] for item in rounds)
    retrieved_ids = [
        doc["id"] for item in rounds for doc in item["evidence"]["retrieved_docs"]
    ]
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "natural_multisearch_budget_trajectory",
        "created_at": utc_stamp(),
        "sample_index": root["sample_index"],
        "sample_id": root["sample_id"],
        "question": root["question"],
        "type": root.get("type"), "level": root.get("level"),
        "gold_labels_loaded": False,
        "condition_id": f"b{budget:04d}",
        "evidence_budget_per_search": budget,
        "top_k_per_search": 10,
        "doc_slot_tokens": 150,
        "max_searches": args.max_searches,
        "root_checkpoint_sha256": root["root_checkpoint_sha256"],
        "root_query": root["query"],
        "root_retrieval_hash": root["retrieval_hash"],
        "search_rounds": rounds,
        "decision_events": events,
        "search_calls": logical_searches,
        "additional_search_calls_after_root": logical_searches - 1,
        "repeated_query_count": repeated_queries,
        "retrieved_docs_total": logical_searches * 10,
        "retrieved_docs_unique": len(set(retrieved_ids)),
        "visible_evidence_tokens_total": visible_total,
        "nominal_evidence_tokens_total": budget * logical_searches,
        "context_compactions": context_compactions,
        "final_answer": final_answer,
        "answer_status": (
            "local_fallback" if answer_parse_mode == "local_plaintext_fallback" else "valid"
        ),
        "answer_parse_mode": answer_parse_mode,
        "protocol_compliant": answer_parse_mode != "local_plaintext_fallback",
        "temperature": 0.0,
        "thinking_control": "qwen3_native_thinking",
        "usage": {
            "shared_root_model_calls": int(root_usage.get("model_calls", 0)),
            "shared_root_llm_input_tokens": int(root_usage.get("llm_input_tokens", 0)),
            "shared_root_llm_output_tokens": int(root_usage.get("llm_output_tokens", 0)),
            "shared_root_thinking_tokens": int(root_usage.get("thinking_tokens", 0)),
            "shared_root_format_repair_calls": int(root_usage.get("format_repair_calls", 0)),
            "continuation_model_calls": continuation_usage["model_calls"],
            "continuation_llm_input_tokens": continuation_usage["llm_input_tokens"],
            "continuation_llm_output_tokens": continuation_usage["llm_output_tokens"],
            "continuation_thinking_tokens": continuation_usage["thinking_tokens"],
            "format_repair_calls": continuation_usage["format_repair_calls"],
            "standalone_model_calls": int(root_usage.get("model_calls", 0))
            + continuation_usage["model_calls"],
            "standalone_llm_input_tokens": int(root_usage.get("llm_input_tokens", 0))
            + continuation_usage["llm_input_tokens"],
            "standalone_llm_output_tokens": int(root_usage.get("llm_output_tokens", 0))
            + continuation_usage["llm_output_tokens"],
            "standalone_thinking_tokens": int(root_usage.get("thinking_tokens", 0))
            + continuation_usage["thinking_tokens"],
            "standalone_format_repair_calls": int(root_usage.get("format_repair_calls", 0))
            + continuation_usage["format_repair_calls"],
        },
    }
    record["record_sha256"] = canonical_hash(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--retrieval_cache", required=True)
    parser.add_argument("--model_url", required=True)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:6008")
    parser.add_argument("--model_path", default="/home/researcher/smartsearch_rebuild/models/Qwen3-8B")
    parser.add_argument("--budgets", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--max_searches", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=32768)
    parser.add_argument("--server_max_total_tokens", type=int, default=14336)
    parser.add_argument("--action_max_new_tokens", type=int, default=8192)
    parser.add_argument("--answer_max_new_tokens", type=int, default=8192)
    parser.add_argument("--repair_max_new_tokens", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retry_failures", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not 2 <= args.max_searches <= 4:
        raise ValueError("max_searches must be between 2 and 4 for this protocol")
    roots = read_jsonl(args.roots)
    roots.sort(key=lambda row: int(row.get("sample_index", 0)))
    if args.limit:
        roots = roots[:args.limit]
    budgets = parse_budgets(args.budgets)
    all_jobs = [(root, budget) for root in roots for budget in budgets]
    # Keep every budget for one question on one worker/GPU so the model can
    # reuse question/evidence prefixes through SGLang's radix cache.
    jobs = [
        (root, budget) for root_index, root in enumerate(roots)
        if root_index % args.num_shards == args.shard_id
        for budget in budgets
    ]
    if args.dry_run:
        print(json.dumps({
            "stage": "natural_multisearch_budget_trajectories",
            "roots": len(roots), "budgets": budgets,
            "trajectory_records_global": len(all_jobs),
            "trajectory_records_this_shard": len(jobs),
            "minimum_continuation_model_calls_global": len(all_jobs),
            "maximum_continuation_model_calls_without_repairs_global": len(all_jobs) * args.max_searches,
            "maximum_logical_search_calls_per_trajectory": args.max_searches,
            "gold_labels_loaded": False,
        }, ensure_ascii=False, indent=2))
        return 0

    identity = probe_qwen(args.model_url)
    if not identity["verified_qwen3_8b"]:
        raise RuntimeError("Trajectory endpoint is not verified Qwen3-8B")
    tokenizer = ExactTokenizer(args.model_path)
    model = SafeSGLangClient(args.model_url, args.timeout, 3)
    retriever = CachedRetriever(args.retriever_url, args.retrieval_cache, args.timeout, 3)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": SCHEMA_VERSION, "stage": "trajectories",
        "roots_sha256": file_sha256(args.roots), "budgets": budgets,
        "max_searches": args.max_searches, "top_k": 10, "doc_slot_tokens": 150,
        "context_length": args.context_length,
        "server_max_total_tokens": args.server_max_total_tokens,
        "action_max_new_tokens": args.action_max_new_tokens,
        "answer_max_new_tokens": args.answer_max_new_tokens,
        "repair_max_new_tokens": 0,
        "temperature": 0.0, "thinking_control": "qwen3_native_thinking",
        "model_path": args.model_path, "shard_id": args.shard_id,
        "num_shards": args.num_shards, "limit": args.limit,
        "gold_labels_loaded": False,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path).get("resume_signature") != canonical_hash(signature):
            raise RuntimeError("Resume signature mismatch; use a new RUN_DIR")
    else:
        write_json(manifest_path, {
            "created_at": utc_stamp(), "signature": signature,
            "resume_signature": canonical_hash(signature),
            "expected_records": len(jobs), "model_identity": identity,
        })

    store = SQLiteRecordStore(output / "state.sqlite")
    completed = store.completed_keys()
    failures = store.failed_keys()
    succeeded = failed = skipped = 0
    started = time.time()
    for root, budget in jobs:
        key = f"{root['sample_id']}|b{budget:04d}"
        if key in completed or (key in failures and not args.retry_failures):
            skipped += 1
            continue
        try:
            record = run_one(root, budget, args, tokenizer, model, retriever)
            store.put_record(key, record)
            succeeded += 1
            elapsed = time.time() - started
            rate = succeeded / elapsed if elapsed else 0.0
            remaining = max(0, len(jobs) - skipped - succeeded - failed)
            eta = remaining / rate if rate else 0.0
            print(
                f"[ok] {root['sample_id']} b{budget:04d} "
                f"searches={record['search_calls']} status={record['answer_status']} "
                f"repairs={record['usage']['format_repair_calls']} eta={eta/3600:.2f}h",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            store.put_failure(key, {
                "created_at": utc_stamp(), "sample_id": root["sample_id"],
                "budget": budget, "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
            })
            print(f"[failed] {key}: {exc}", flush=True)
    records_total, failures_total = store.export(output)
    status = {
        "finished_at": utc_stamp(), "stage": "trajectories",
        "shard_id": args.shard_id, "num_shards": args.num_shards,
        "expected_records": len(jobs), "records_total": records_total,
        "unresolved_failures": failures_total,
        "succeeded_this_attempt": succeeded, "failed_this_attempt": failed,
        "skipped_existing": skipped,
        "elapsed_seconds_this_attempt": time.time() - started,
        "complete": records_total == len(jobs) and failures_total == 0,
    }
    write_json(output / "run_status.json", status)
    store.close()
    retriever.close()
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
