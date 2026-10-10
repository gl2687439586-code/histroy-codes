#!/usr/bin/env python3
"""Run the query-only TALE-EP budget estimator exactly once per question."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (
    METHOD,
    PROTOCOL_VARIANT,
    SCHEMA_VERSION,
    TEMPERATURE,
    TOP_P,
    append_jsonl,
    canonical_hash,
    chat_completion,
    new_request_id,
    parse_budget,
    read_jsonl,
    request_attempts,
    sha256_file,
    sha256_text,
    summarize_request_attempts,
    utc_now,
    verify_qwen3,
    write_json,
)
from prompts import (
    TALE_EP_ESTIMATOR_INSTRUCTION,
    estimator_prompt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--model-name", default="Qwen3-8B")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=TOP_P)
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Technical cap only; Qwen3 thinking may precede the short [[budget]] answer.",
    )
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()

    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard id")
    if args.temperature != TEMPERATURE or args.top_p != TOP_P:
        raise RuntimeError("Registered TALE-EP estimator requires temperature=0.1, top_p=1.0")
    questions_path = Path(args.questions).resolve()
    questions = read_jsonl(questions_path)
    tasks = [row for index, row in enumerate(questions) if index % args.num_shards == args.shard_id]
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / f"shard_{args.shard_id}.jsonl"
    failures_path = output / f"shard_{args.shard_id}.failures.jsonl"
    ledger_path = output / f"shard_{args.shard_id}.physical_requests.jsonl"
    manifest_path = output / f"shard_{args.shard_id}.manifest.json"
    info = verify_qwen3(args.model_url, args.model_path)
    signature = {
        "schema_version": SCHEMA_VERSION,
        "method": METHOD,
        "protocol_variant": PROTOCOL_VARIANT,
        "stage": "query_only_budget_estimation",
        "questions_sha256": sha256_file(questions_path),
        "model_path": str(Path(args.model_path).resolve()),
        "model_url": args.model_url,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "estimator_instruction": TALE_EP_ESTIMATOR_INSTRUCTION,
        "estimates_per_question": 1,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    signature_hash = canonical_hash(signature)
    existing_rows = read_jsonl(records_path) if records_path.exists() else []
    existing = {str(row["sample_id"]): row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise RuntimeError("Duplicate estimate sample id")
    if any(row.get("signature_hash") != signature_hash for row in existing_rows):
        raise RuntimeError("Existing estimate shard has a different signature")

    succeeded = failed = 0
    started_all = time.time()
    for ordinal, question in enumerate(tasks, 1):
        sample_id = str(question["sample_id"])
        if sample_id in existing:
            continue
        prompt = estimator_prompt(str(question["question"]))
        request_id = new_request_id()
        result = None
        try:
            result = chat_completion(
                base_url=args.model_url,
                model=args.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                timeout=args.timeout,
                request_ledger_path=ledger_path,
                request_id=request_id,
                request_context={
                    "stage": "query_only_budget_estimation",
                    "signature_hash": signature_hash,
                    "sample_index": int(question["sample_index"]),
                    "sample_id": sample_id,
                },
            )
            visible_output = str(result["text"] or "")
            if "</think>" in visible_output:
                visible_output = visible_output.rsplit("</think>", 1)[-1].strip()
            budget = parse_budget(visible_output)
            append_jsonl(
                ledger_path,
                {
                    "event": "consumer_result",
                    "created_at": utc_now(),
                    "request_id": request_id,
                    "outcome": "accepted_estimate",
                    "estimated_budget": budget,
                    "signature_hash": signature_hash,
                    "sample_id": sample_id,
                },
            )
            record = {
                "schema_version": SCHEMA_VERSION,
                "signature_hash": signature_hash,
                "method": METHOD,
                "protocol_variant": PROTOCOL_VARIANT,
                "stage": "query_only_budget_estimation",
                "sample_index": int(question["sample_index"]),
                "sample_id": sample_id,
                "question": str(question["question"]),
                "estimator_prompt": prompt,
                "estimator_prompt_sha256": sha256_text(prompt),
                "raw_output": result["text"],
                "visible_output": visible_output,
                "estimated_budget": budget,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "usage": result["usage"],
                "finish_reason": result["finish_reason"],
                "latency_seconds": result["latency_seconds"],
                "model_path_verified": str(info.get("model_path") or ""),
                "gold_labels_loaded": False,
                "successful_estimator_request_id": request_id,
                "logical_estimate_records": 1,
            }
            record["record_hash"] = canonical_hash(record)
            append_jsonl(records_path, record)
            existing[sample_id] = record
            succeeded += 1
            elapsed = max(time.time() - started_all, 1e-9)
            remaining = len(tasks) - len(existing)
            eta = elapsed / max(succeeded + failed, 1) * remaining / 3600
            print(
                f"[estimate-ok] {ordinal}/{len(tasks)} {sample_id} B={budget} eta={eta:.2f}h",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            append_jsonl(
                ledger_path,
                {
                    "event": "consumer_result",
                    "created_at": utc_now(),
                    "request_id": request_id,
                    "outcome": (
                        "downstream_parse_error" if result is not None else "request_error"
                    ),
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                    "signature_hash": signature_hash,
                    "sample_id": sample_id,
                },
            )
            append_jsonl(
                failures_path,
                {
                    "created_at": utc_now(),
                    "sample_id": sample_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "signature_hash": signature_hash,
                    "physical_request_id": request_id,
                    "usage": (result or {}).get("usage"),
                    "finish_reason": (result or {}).get("finish_reason"),
                    "raw_output": (result or {}).get("text"),
                },
            )
            print(f"[estimate-failed] {sample_id}: {exc}", flush=True)

    expected_ids = {str(row["sample_id"]) for row in tasks}
    missing = sorted(expected_ids - set(existing))
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
        "unresolved_ids": missing,
        "complete": not missing,
        "output": str(records_path),
        "physical_request_ledger": str(ledger_path),
        "physical_request_accounting": ledger_summary,
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
