#!/usr/bin/env python3
"""Rerun only invalid-format Answer-only v5 records with the exact saved prompt.

This is a separately labelled supplementary stage.  It never overwrites the
one-call primary experiment.  Each selected key receives exactly one new model
request with the same prompt, temperature, stop string, and token limit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "blank_answer_only"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common import (  # noqa: E402
    ExactTokenizer, ModelClient, RecordStore, answer_metrics,
    assert_qwen_configuration, atomic_write_json, canonical_json_hash,
    condition_key, probe_model_identity, probe_server_context,
    public_model_meta, read_json, read_jsonl, sha256_text, utc_stamp,
)
from run_answer_only import (  # noqa: E402
    attempt_record, check_capacity, strict_answer_parse,
)

RETRY_SCHEMA = "qwen3-blank-answer-only-format-retry-v1"
RETRY_STAGE = "invalid_format_exact_prompt_retry"
EXPECTED_SOURCE_SHA256 = (
    "68c5be70bed953053cc9cdcddba8cc08ab6922f1f7aed09319d9b41dbdfc6042"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finish_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type", "")).lower()
    return str(value or "").lower()


def validate_parent(record: Dict[str, Any]) -> None:
    if record.get("answer_status") != "invalid_format":
        raise ValueError(f"Selected parent is not invalid: {condition_key(record)}")
    supplied = record.get("record_sha256")
    unhashed = dict(record)
    unhashed.pop("record_sha256", None)
    if supplied != canonical_json_hash(unhashed):
        raise ValueError(f"Parent record hash mismatch: {condition_key(record)}")
    if float(record.get("temperature", -1)) != 0.0:
        raise ValueError(f"Parent temperature is not zero: {condition_key(record)}")
    if int(record.get("max_new_tokens", -1)) != 6144:
        raise ValueError(f"Parent max_new_tokens is not 6144: {condition_key(record)}")
    attempts = (record.get("generation_attempts") or {}).get("answer") or []
    if len(attempts) != 1 or attempts[0] != record.get("generation_attempt"):
        raise ValueError(f"Parent attempt invariant failed: {condition_key(record)}")


def build_retry_record(
    parent: Dict[str, Any], model: ModelClient, tokenizer: ExactTokenizer,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    validate_parent(parent)
    parent_attempt = parent["generation_attempt"]
    prompt = str(parent_attempt["prompt"])
    prompt_tokens = check_capacity(
        tokenizer, prompt, args.max_new_tokens, args.context_length,
        args.server_max_total_tokens,
    )
    if sha256_text(prompt) != parent_attempt.get("prompt_sha256"):
        raise ValueError(f"Saved parent prompt hash mismatch: {condition_key(parent)}")
    planned_seed = int(parent.get("planned_seed_not_sent", args.seed))
    response = model.generate(
        prompt, args.temperature, args.max_new_tokens, ("</answer>",), planned_seed
    )
    parsed = strict_answer_parse(response.text, response.finish_reason, tokenizer)
    attempt = attempt_record(tokenizer, prompt, response, args.max_new_tokens, parsed)
    metrics = answer_metrics(
        parsed["final_answer"], parent.get("gold_answers") or []
    ) if parsed["valid"] else {
        "em": 0.0, "f1": 0.0, "acc": 0.0,
        "normalized_exact": 0.0, "normalized_contains": 0.0,
    }
    retry_usage = {
        "answer_input_tokens": attempt["usage"]["input_tokens"],
        "answer_output_tokens": attempt["usage"]["output_tokens"],
        "answer_reasoning_tokens": attempt["usage"]["reasoning_tokens"],
        "total_tokens": (
            attempt["usage"]["input_tokens"] + attempt["usage"]["output_tokens"]
        ),
        "generation_calls": 1,
    }
    parent_usage = parent.get("usage") or {}
    cumulative_usage = {
        "answer_input_tokens": int(parent_usage.get("answer_input_tokens", 0))
        + retry_usage["answer_input_tokens"],
        "answer_output_tokens": int(parent_usage.get("answer_output_tokens", 0))
        + retry_usage["answer_output_tokens"],
        "total_tokens": int(parent_usage.get("total_tokens", 0))
        + retry_usage["total_tokens"],
        "generation_calls": int(parent_usage.get("generation_calls", 1)) + 1,
    }
    record: Dict[str, Any] = {
        "schema_version": RETRY_SCHEMA,
        "created_at": utc_stamp(),
        "benchmark": parent.get("benchmark", "bamboogle"),
        "stage": RETRY_STAGE,
        "sample_id": parent["sample_id"],
        "condition_id": parent["condition_id"],
        "mode": parent["mode"],
        "rollout_id": int(parent.get("rollout_id", 0)),
        "question": parent["question"],
        "gold_answers": parent.get("gold_answers") or [],
        "evidence_budget": int(parent["evidence_budget"]),
        "root_type": parent.get("root_type"),
        "root_checkpoint_sha256": parent.get("root_checkpoint_sha256"),
        "retrieval_hash": parent.get("retrieval_hash"),
        "visible_evidence_hash": parent.get("visible_evidence_hash"),
        "model_name": "qwen3-8b",
        "model_path": args.model_path,
        "model_url": args.model_url,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "server_max_total_tokens": args.server_max_total_tokens,
        "planned_seed_not_sent": planned_seed,
        "retry_round": 1,
        "retry_policy": "same_saved_prompt_same_model_same_sampling",
        "primary_analysis_eligible": False,
        "retry_of_record_sha256": parent["record_sha256"],
        "original_answer_status": parent["answer_status"],
        "original_format_errors": parent.get("format_errors") or [],
        "original_finish_type": finish_type(parent_attempt.get("finish_reason")),
        "original_output_sha256": parent_attempt.get("output_sha256"),
        "original_output_tokens": int(
            (parent_usage or {}).get("answer_output_tokens", 0)
        ),
        "same_prompt_verified": attempt["prompt_sha256"] == parent_attempt["prompt_sha256"],
        "same_visible_evidence_verified": bool(parent.get("visible_evidence_hash")),
        "answer_status": "valid" if parsed["valid"] else "invalid_format",
        **parsed,
        "metrics": metrics,
        **metrics,
        "generation_attempt": attempt,
        "generation_attempts": {"format_retry": [attempt]},
        "retry_usage": retry_usage,
        "cumulative_usage_for_key": cumulative_usage,
        "model_request_count_this_stage": 1,
        "cumulative_model_request_count_for_key": 2,
        "judge_calls": 0,
        "prompt_tokens": prompt_tokens,
        "latency_seconds": float(response.latency_seconds),
        "retry_output_identical_to_original": (
            attempt["output_sha256"] == parent_attempt.get("output_sha256")
        ),
        "model_meta": public_model_meta(response.raw),
    }
    record["record_sha256"] = canonical_json_hash(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_records", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_url", required=True)
    parser.add_argument("--model_path", default=(
        "/home/researcher/smartsearch_rebuild/models/Qwen3-8B"))
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--server_max_total_tokens", type=int, default=10086)
    parser.add_argument("--max_new_tokens", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--expected_invalid", type=int, default=271)
    parser.add_argument("--expected_source_sha256", default=EXPECTED_SOURCE_SHA256)
    parser.add_argument("--retry_failures", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.temperature != 0.0 or args.max_new_tokens != 6144:
        raise ValueError("Exact retry freezes temperature=0 and max_new_tokens=6144")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Invalid shard_id")
    source = Path(args.original_records)
    source_sha = file_sha256(source)
    if args.expected_source_sha256 and source_sha != args.expected_source_sha256:
        raise ValueError(
            f"Original records SHA256 mismatch: {source_sha} != "
            f"{args.expected_source_sha256}"
        )
    original = read_jsonl(source)
    invalid = [row for row in original if row.get("answer_status") == "invalid_format"]
    invalid.sort(key=condition_key)
    if len(invalid) != args.expected_invalid:
        raise ValueError(f"Expected {args.expected_invalid} invalid records; got {len(invalid)}")
    for parent in invalid:
        validate_parent(parent)
    jobs = [row for index, row in enumerate(invalid)
            if index % args.num_shards == args.shard_id]
    mean_latency = sum(float(row.get("latency_seconds", 0)) for row in invalid) / len(invalid)
    if args.dry_run:
        print(json.dumps({
            "source_sha256": source_sha,
            "global_invalid_records": len(invalid),
            "shard_id": args.shard_id,
            "records_this_shard": len(jobs),
            "expected_new_model_calls_this_shard": len(jobs),
            "mean_original_invalid_latency_seconds": mean_latency,
            "four_gpu_base_estimated_wall_hours": mean_latency * len(invalid) / 4 / 3600,
            "four_gpu_planning_hours_at_1.25x": mean_latency * len(invalid) / 4 / 3600 * 1.25,
        }, ensure_ascii=False, indent=2))
        return 0

    assert_qwen_configuration(args.model_path, args.model_url)
    identity = probe_model_identity(args.model_url)
    if not identity.get("verified_qwen3_8b"):
        raise RuntimeError("Retry endpoint must be verified Qwen3-8B")
    tokenizer = ExactTokenizer(args.model_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": RETRY_SCHEMA,
        "source_records_sha256": source_sha,
        "retry_policy": "same_saved_prompt_same_model_same_sampling",
        "temperature": 0.0,
        "max_new_tokens": 6144,
        "context_length": args.context_length,
        "server_max_total_tokens": args.server_max_total_tokens,
        "model_path": args.model_path,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "expected_global_invalid": len(invalid),
        "expected_records_this_shard": len(jobs),
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path).get("resume_signature") != canonical_json_hash(signature):
            raise RuntimeError("Resume signature mismatch; use a new RETRY_RUN_DIR")
    else:
        atomic_write_json(manifest_path, {
            "created_at": utc_stamp(),
            "design": "One exact saved-prompt retry for each primary invalid-format record",
            "resume_signature": canonical_json_hash(signature),
            "signature": signature,
            "model_identity": identity,
            "server_context": probe_server_context(args.model_url),
        })

    store = RecordStore(output_dir)
    completed = store.completed_keys()
    failed_keys = store.failed_keys()
    model = ModelClient(args.model_url, args.model_path, "sglang", args.timeout, 3)
    succeeded = failed = skipped = 0
    started = time.time()
    for parent in jobs:
        key = condition_key(parent)
        if key in completed or (key in failed_keys and not args.retry_failures):
            skipped += 1
            continue
        try:
            record = build_retry_record(parent, model, tokenizer, args)
            store.put_record(record)
            succeeded += 1
            elapsed = time.time() - started
            rate = succeeded / elapsed if elapsed else 0.0
            remaining = max(0, len(jobs) - skipped - succeeded - failed)
            eta = remaining / rate if rate else 0.0
            print(
                f"[retry-ok] {key[0]} {key[1]} status={record['answer_status']} "
                f"same_output={record['retry_output_identical_to_original']} "
                f"f1={record['f1']:.3f} eta={eta/3600:.2f}h",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            failure = {
                "created_at": utc_stamp(),
                "sample_id": key[0], "condition_id": key[1],
                "mode": key[2], "rollout_id": key[3],
                "retry_of_record_sha256": parent.get("record_sha256"),
                "error_type": type(exc).__name__, "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            store.put_failure(failure)
            print(f"[retry-failed] {key}: {exc}", file=sys.stderr, flush=True)
    records_total, failures_total = store.consolidate()
    rows = read_jsonl(output_dir / "records.jsonl")
    status = {
        "finished_at": utc_stamp(), "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "expected_records": len(jobs), "records_total": records_total,
        "valid_retries": sum(row.get("answer_status") == "valid" for row in rows),
        "remaining_invalid_retries": sum(row.get("answer_status") != "valid" for row in rows),
        "identical_output_retries": sum(row.get("retry_output_identical_to_original") for row in rows),
        "succeeded_this_attempt": succeeded, "failed_this_attempt": failed,
        "skipped_existing": skipped, "unresolved_failures_total": failures_total,
        "elapsed_seconds_this_attempt": time.time() - started,
        "complete": records_total == len(jobs) and failures_total == 0,
    }
    atomic_write_json(output_dir / "run_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
