#!/usr/bin/env python3
"""Run the Qwen3 reader for frozen Adaptive-k roots (two stochastic rollouts)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import (
    SCHEMA_VERSION, SGLangClient, append_jsonl, canonical_hash,
    extract_qwen_answer, file_sha256, paper_qa_prompt, qwen_chat_prompt,
    read_jsonl, utc_stamp, write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_url", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=2)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_new_tokens", type=int, default=16384)
    parser.add_argument("--server_max_total_tokens", type=int, default=24497)
    parser.add_argument("--min_completion_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard_id")
    if args.rollouts != 2:
        raise ValueError("Registered protocol requires exactly two rollouts")
    if args.temperature != 0.6 or args.top_p != 0.95:
        raise ValueError("Registered stochastic protocol requires temperature=0.6, top_p=0.95")

    from transformers import AutoTokenizer

    roots = read_jsonl(args.roots)
    if not roots:
        raise RuntimeError("No roots loaded")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    client = SGLangClient(args.model_url, args.timeout, args.retries)
    model_info = client.model_info()
    actual_model_path = str(model_info.get("model_path") or "")
    if Path(actual_model_path).name != Path(args.model_path).name or "Qwen3-8B" not in actual_model_path:
        raise RuntimeError(f"Endpoint is not the expected Qwen3-8B: {model_info}")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / f"shard_{args.shard_id}.jsonl"
    failures_path = output / f"shard_{args.shard_id}.failures.jsonl"
    manifest_path = output / f"shard_{args.shard_id}.manifest.json"
    signature = {
        "schema_version": SCHEMA_VERSION,
        "roots_sha256": file_sha256(args.roots),
        "root_signature_hash": roots[0]["signature_hash"],
        "model_path": str(Path(args.model_path).resolve()),
        "model_url": args.model_url,
        "rollouts": args.rollouts,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "server_max_total_tokens": args.server_max_total_tokens,
        "seed_policy": "no explicit seed; SGLang 0.4.6 rejects seed; stochastic sampling supplies rollout variation",
        "paper_decoding_deviation": "paper uses temperature=0/top_p=1; user requested two stochastic rollouts",
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    signature_hash = canonical_hash(signature)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("signature_hash") != signature_hash:
            raise RuntimeError("Generation shard directory has a different signature")
    existing_rows = read_jsonl(records_path) if records_path.exists() else []
    for row in existing_rows:
        if row.get("signature_hash") != signature_hash:
            raise RuntimeError("Existing generation record has a different signature")
    existing = {(str(row["sample_id"]), int(row["rollout_id"])): row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise RuntimeError("Duplicate sample/rollout key in existing generation records")

    tasks = []
    for root in roots:
        for rollout_id in range(args.rollouts):
            global_index = int(root["sample_index"]) * args.rollouts + rollout_id
            if global_index % args.num_shards == args.shard_id:
                tasks.append((root, rollout_id))
    succeeded = failed = 0
    started_all = time.time()
    for root, rollout_id in tasks:
        key = (str(root["sample_id"]), rollout_id)
        if key in existing:
            continue
        try:
            context = "\n\n".join(item["contents"] for item in root["selected_documents"])
            user_prompt = paper_qa_prompt(str(root["question"]), context)
            chat_prompt = qwen_chat_prompt(tokenizer, user_prompt)
            prompt_tokens = len(tokenizer.encode(chat_prompt, add_special_tokens=False))
            available = args.server_max_total_tokens - prompt_tokens - 128
            effective_max_new_tokens = min(args.max_new_tokens, available)
            if effective_max_new_tokens < args.min_completion_tokens:
                raise RuntimeError(
                    "Adaptive-k context exceeds the verified Qwen server capacity without truncation: "
                    f"prompt_tokens={prompt_tokens}, pool={args.server_max_total_tokens}, "
                    f"available_completion={available}"
                )
            generated = client.generate(
                chat_prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=effective_max_new_tokens,
            )
            answer = extract_qwen_answer(generated.text, generated.meta)
            record = {
                "schema_version": SCHEMA_VERSION,
                "signature_hash": signature_hash,
                "sample_index": int(root["sample_index"]),
                "sample_id": str(root["sample_id"]),
                "question": str(root["question"]),
                "rollout_id": rollout_id,
                "root_hash": root["root_hash"],
                "method": "adaptive-k-noclass",
                "retrieval_calls": 1,
                "maximum_allowed_searches": 4,
                "candidate_count": int(root["candidate_count"]),
                "adaptive_k": int(root["selection"]["selected_k"]),
                "selection": root["selection"],
                "selected_document_ids": [item["id"] for item in root["selected_documents"]],
                "selected_document_titles": [item["title"] for item in root["selected_documents"]],
                "selected_context_token_count": int(root["selected_context_token_count"]),
                "prompt_token_count_local": prompt_tokens,
                "requested_max_new_tokens": args.max_new_tokens,
                "effective_max_new_tokens": effective_max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "raw_output": generated.text,
                **answer,
                "model_meta": generated.meta,
                "model_latency_seconds": generated.elapsed_seconds,
                "model_url": args.model_url,
                "model_path_verified": actual_model_path,
            }
            record["record_hash"] = canonical_hash(record)
            append_jsonl(records_path, record)
            existing[key] = record
            succeeded += 1
            elapsed = max(time.time() - started_all, 1e-9)
            remaining = len(tasks) - len(existing)
            eta_hours = elapsed / max(succeeded + failed, 1) * remaining / 3600
            print(
                f"[gen-ok] {key[0]} r={rollout_id} k={record['adaptive_k']} "
                f"status={record['answer_status']} out={generated.meta.get('completion_tokens')} "
                f"eta={eta_hours:.2f}h",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            append_jsonl(failures_path, {
                "created_at": utc_stamp(), "sample_id": key[0], "rollout_id": key[1],
                "error_type": type(exc).__name__, "error": str(exc),
                "signature_hash": signature_hash,
            })
            print(f"[gen-failed] {key}: {exc}", flush=True)

    expected_keys = {(str(root["sample_id"]), rollout) for root, rollout in tasks}
    missing = sorted(expected_keys - set(existing))
    manifest = {
        "created_at": utc_stamp(), "signature": signature,
        "signature_hash": signature_hash, "shard_id": args.shard_id,
        "num_shards": args.num_shards, "expected_records": len(tasks),
        "records": len(existing), "succeeded_this_attempt": succeeded,
        "failed_this_attempt": failed, "unresolved_failures": len(missing),
        "unresolved_keys": missing, "complete": not missing,
        "model_info": model_info, "output": str(records_path.resolve()),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
