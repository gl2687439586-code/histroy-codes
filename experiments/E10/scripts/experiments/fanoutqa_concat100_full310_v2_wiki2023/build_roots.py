#!/usr/bin/env python3
"""Build one shared, label-free first-Search checkpoint per FanOutQA question."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

from actions import generate_action
from hp_common import (
    TOP_K, RETRIEVER_BACKEND, ExactTokenizer, SCHEMA_VERSION, SQLiteRecordStore, SafeSGLangClient,
    canonical_hash, file_sha256, probe_qwen, read_json, read_jsonl,
    sha256_text, utc_stamp, write_json,
)
from prompts import initial_search_prompt
from retrieval import (
    CachedRetriever, concatenate_documents, enrich_documents,
    exposure_conditions,
)


def selected_questions(path: str, limit: int, sample_ids: str):
    rows = read_jsonl(path)
    wanted = {item.strip() for item in sample_ids.split(",") if item.strip()}
    if wanted:
        rows = [row for row in rows if str(row["sample_id"]) in wanted]
    rows.sort(key=lambda row: int(row.get("sample_index", 0)))
    if limit:
        rows = rows[:limit]
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--retrieval_cache", required=True)
    parser.add_argument("--model_url", required=True)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:6008")
    parser.add_argument("--retriever_backend", default=RETRIEVER_BACKEND)
    parser.add_argument("--model_path", default="/home/researcher/smartsearch_rebuild/models/Qwen3-8B")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_ids", default="")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--action_max_new_tokens", type=int, default=8192)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retry_failures", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard")
    all_rows = selected_questions(args.questions, args.limit, args.sample_ids)
    jobs = [row for index, row in enumerate(all_rows)
            if index % args.num_shards == args.shard_id]
    if args.dry_run:
        print(json.dumps({
            "stage": "shared_first_search_roots",
            "questions_global": len(all_rows),
            "questions_this_shard": len(jobs),
            "model_calls_minimum_this_shard": len(jobs),
            "retriever_calls_maximum_this_shard": len(jobs),
            "gold_labels_loaded": False,
        }, ensure_ascii=False, indent=2))
        return 0

    identity = probe_qwen(args.model_url)
    if not identity["verified_qwen3_8b"]:
        raise RuntimeError("Root endpoint is not verified Qwen3-8B")
    tokenizer = ExactTokenizer(args.model_path)
    model = SafeSGLangClient(args.model_url, args.timeout, 3)
    retriever = CachedRetriever(
        args.retriever_url, args.retrieval_cache, args.timeout, 3,
        backend_id=args.retriever_backend,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": SCHEMA_VERSION,
        "stage": "shared_first_search_roots",
        "questions_sha256": file_sha256(args.questions),
        "gold_labels_loaded": False,
        "requested_max_k": TOP_K,
        "retrieval_cardinality": "natural_zero_to_requested_max_no_backfill",
        "retriever_backend": args.retriever_backend,
        "wikipedia_snapshot": "2023-09",
        "evidence_exposure": "concat_then_global_prefix_100_to_exact_full",
        "temperature": 0.0,
        "thinking_control": "qwen3_native_thinking",
        "action_max_new_tokens": args.action_max_new_tokens,
        "model_path": args.model_path,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "limit": args.limit,
        "sample_ids": args.sample_ids,
    }
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path).get("resume_signature") != canonical_hash(signature):
            raise RuntimeError("Resume signature mismatch; use a new ROOT_RUN_DIR")
    else:
        write_json(manifest_path, {
            "created_at": utc_stamp(), "signature": signature,
            "resume_signature": canonical_hash(signature),
            "model_identity": identity,
            "expected_records": len(jobs),
        })

    store = SQLiteRecordStore(output / "state.sqlite")
    completed = store.completed_keys()
    failures = store.failed_keys()
    succeeded = failed = skipped = 0
    started = time.time()
    for row in jobs:
        key = str(row["sample_id"])
        if key in completed or (key in failures and not args.retry_failures):
            skipped += 1
            continue
        try:
            prompt = initial_search_prompt(row["question"])
            action = generate_action(
                model, tokenizer, prompt, ["search"], args.action_max_new_tokens
            )
            if action["action"] == "search" and action["payload"]:
                query = action["payload"]
                query_parse_mode = action["parse_mode"]
            else:
                query = row["question"]
                query_parse_mode = "fallback_question_as_query"
            documents, cache_hit, retrieval_latency = retriever.search(query, TOP_K)
            documents = enrich_documents(tokenizer, documents)
            root_context = concatenate_documents(tokenizer, documents)
            conditions = exposure_conditions(tokenizer, documents)
            usage = {
                "llm_input_tokens": sum(a["input_tokens"] for a in action["attempts"]),
                "llm_output_tokens": sum(a["output_tokens"] for a in action["attempts"]),
                "thinking_tokens": sum(a["thinking_tokens"] for a in action["attempts"]),
                "model_calls": len(action["attempts"]),
                "format_repair_calls": max(0, len(action["attempts"]) - 1),
                "retriever_calls": 0 if cache_hit else 1,
            }
            record = {
                "schema_version": SCHEMA_VERSION,
                "stage": "shared_first_search_root",
                "created_at": utc_stamp(),
                "sample_index": row["sample_index"],
                "sample_id": row["sample_id"],
                "question": row["question"],
                "categories": row.get("categories") or [],
                "dataset_name": "fanoutqa", "split": "dev",
                "gold_labels_loaded": False,
                "query": query,
                "query_parse_mode": query_parse_mode,
                "action_attempts": action["attempts"],
                "documents": documents,
                "requested_max_k": TOP_K,
                "root_actual_retrieved_k": len(documents),
                "root_retrieval_shortfall": len(documents) < TOP_K,
                "root_zero_result": len(documents) == 0,
                "root_full_context_tokens": root_context["full_context_token_count"],
                "root_full_context_sha256": root_context["full_context_sha256"],
                "exposure_conditions": conditions,
                "evidence_exposure_protocol": (
                    f"Concatenate 0..{TOP_K} naturally retrieved contents with two-newline separators; "
                    "evaluate 100-token global prefixes and one exact FULL endpoint."
                ),
                "retrieval_hash": canonical_hash([
                    [doc["id"], doc["contents_sha256"]] for doc in documents
                ]),
                "retrieval_cache_hit": cache_hit,
                "retrieval_latency_seconds": retrieval_latency,
                "usage": usage,
                "model_url": args.model_url,
                "model_path": args.model_path,
                "temperature": 0.0,
                "thinking_control": "qwen3_native_thinking",
                "requested_max_k": TOP_K,
                "retrieval_cardinality": "natural_zero_to_requested_max_no_backfill",
                "retriever_backend": args.retriever_backend,
                "wikipedia_snapshot": "2023-09",
            }
            record["root_checkpoint_sha256"] = canonical_hash(record)
            store.put_record(key, record)
            succeeded += 1
            elapsed = time.time() - started
            rate = succeeded / elapsed if elapsed else 0.0
            remaining = max(0, len(jobs) - skipped - succeeded - failed)
            eta = remaining / rate if rate else 0.0
            print(
                f"[root-ok] {key} query={query!r} cache={cache_hit} "
                f"eta={eta/3600:.2f}h", flush=True,
            )
        except Exception as exc:
            failed += 1
            store.put_failure(key, {
                "created_at": utc_stamp(), "sample_id": key,
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            print(f"[root-failed] {key}: {exc}", flush=True)
    records_total, failures_total = store.export(output)
    status = {
        "finished_at": utc_stamp(), "stage": "roots",
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
