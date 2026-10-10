#!/usr/bin/env python3
"""Run the frozen Qwen3-8B Bamboogle Answer-only phase.

There is exactly one semantic model generation per successful record.  Format
errors are retained as invalid answer records rather than repaired by another
model call.  EM/F1 are computed locally.  Judges and evidence-hit labels are
intentionally absent and must be added by immutable-keyed offline stages.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    PROTOCOL_VERSION,
    ExactTokenizer,
    ModelClient,
    RecordStore,
    answer_metrics,
    assert_qwen_configuration,
    atomic_write_json,
    canonical_json_hash,
    condition_key,
    probe_model_identity,
    probe_server_context,
    public_model_meta,
    read_json,
    read_jsonl,
    sha256_text,
    stable_hash,
    utc_stamp,
)
from evidence_views import (  # noqa: E402
    ALL_CONDITIONS,
    CONDITIONS,
    assert_nested_views,
    build_evidence_view,
    condition_manifest,
)
from prompts import answer_prompt  # noqa: E402

MODE = "blank_answer_only"
STAGE = "answer_generation"
MAX_SHORT_ANSWER_CHARS = 256
MAX_SHORT_ANSWER_TOKENS = 64


def finish_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type", "")).lower()
    return str(value or "").lower()


def strict_answer_parse(output: str, finish_reason: Any,
                        tokenizer: ExactTokenizer) -> Dict[str, Any]:
    """Validate only the final segment after Qwen's optional </think>."""
    text = str(output or "")
    final_segment = text.rsplit("</think>", 1)[-1]
    match = re.fullmatch(
        r"\s*<answer>(.*?)</answer>\s*", final_segment,
        flags=re.I | re.S,
    )
    payload = match.group(1).strip() if match else ""
    errors: List[str] = []
    if finish_type(finish_reason) == "length":
        errors.append("generation_hit_length_limit")
    if re.search(r"<(?:search|read|result)\b", text, flags=re.I):
        errors.append("forbidden_action_tag")
    if match is None:
        errors.append("final_segment_not_exactly_one_answer_tag")
    if match is not None and not payload:
        errors.append("empty_answer_payload")
    if payload and ("<" in payload or ">" in payload):
        errors.append("markup_inside_answer_payload")
    if len(payload) > MAX_SHORT_ANSWER_CHARS:
        errors.append("answer_payload_too_many_characters")
    payload_tokens = tokenizer.count(payload) if payload else 0
    if payload_tokens > MAX_SHORT_ANSWER_TOKENS:
        errors.append("answer_payload_too_many_tokens")
    errors = sorted(set(errors))
    return {
        "valid": not errors,
        "candidate_answer": payload,
        "final_answer": payload if not errors else "",
        "format_errors": errors,
        "canonical_answer_tag": match is not None,
        "clean_answer_payload": bool(match is not None and payload and
                                      "<" not in payload and ">" not in payload),
        "payload_characters": len(payload),
        "payload_tokens": payload_tokens,
        "final_segment_sha256": sha256_text(final_segment),
    }


def attempt_record(tokenizer: ExactTokenizer, prompt: str, response: Any,
                   maximum: int, parsed: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "output": response.text,
        "output_sha256": sha256_text(response.text),
        "max_new_tokens": maximum,
        "finish_reason": response.finish_reason,
        "latency_seconds": response.latency_seconds,
        "usage": {
            "input_tokens": tokenizer.count(prompt),
            "output_tokens": tokenizer.count(response.text),
            "reasoning_tokens": tokenizer.count(
                re.findall(r"<think>(.*?)</think>", response.text,
                           flags=re.I | re.S)[-1]
            ) if re.findall(r"<think>(.*?)</think>", response.text,
                            flags=re.I | re.S) else 0,
        },
        "model_meta": public_model_meta(response.raw),
        "format_errors": parsed["format_errors"],
    }


def check_capacity(tokenizer: ExactTokenizer, prompt: str, maximum: int,
                   context_length: int, server_max_total_tokens: int) -> int:
    prompt_tokens = tokenizer.count(prompt)
    combined = prompt_tokens + maximum
    effective_limit = min(context_length, server_max_total_tokens)
    if combined > effective_limit:
        raise ValueError(
            f"prompt_tokens({prompt_tokens}) + max_new_tokens({maximum}) "
            f"> effective_limit({effective_limit})"
        )
    return prompt_tokens


def selected_roots(root_cache: str, limit: int, sample_ids: Sequence[str],
                   root_scope: str, shard_id: int,
                   num_shards: int) -> Tuple[List[Dict[str, Any]], int]:
    roots = read_jsonl(root_cache)
    wanted = {str(item) for item in sample_ids}
    if wanted:
        roots = [root for root in roots if str(root.get("sample_id")) in wanted]
    if root_scope != "all":
        roots = [root for root in roots if root.get("root_type") == root_scope]
    roots.sort(key=lambda root: str(root.get("sample_id")))
    if limit:
        roots = roots[:limit]
    total = len(roots)
    return [root for index, root in enumerate(roots)
            if index % num_shards == shard_id], total


def file_sha256(path: Path) -> str:
    return sha256_text(path.read_text(encoding="utf-8"))


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def provenance(model_path: str) -> Dict[str, Any]:
    config = Path(model_path) / "config.json"
    code_files = [
        SCRIPT_DIR / "run_answer_only.py", SCRIPT_DIR / "common.py",
        SCRIPT_DIR / "evidence_views.py", SCRIPT_DIR / "prompts.py",
    ]
    return {
        "model_config_sha256": file_sha256(config) if config.exists() else "missing",
        "experiment_code_sha256": canonical_json_hash({
            item.name: file_sha256(item) for item in code_files
        }),
        "git_commit": git_commit(),
        "model_server_command": os.environ.get(
            "MODEL_SERVER_COMMAND", "unrecorded"
        ),
        "service_snapshot_path": os.environ.get(
            "SERVICE_SNAPSHOT_PATH", "unrecorded"
        ),
    }


def run_job(root: Dict[str, Any], condition_id: str, rollout_id: int,
            model: ModelClient, tokenizer: ExactTokenizer,
            args: argparse.Namespace) -> Dict[str, Any]:
    view = build_evidence_view(tokenizer, root["documents"],
                               CONDITIONS[condition_id])
    prompt = answer_prompt(root["question"], view["evidence_text"])
    prompt_tokens = check_capacity(
        tokenizer, prompt, args.max_new_tokens, args.context_length,
        args.server_max_total_tokens,
    )
    planned_seed = args.seed + rollout_id + int(
        stable_hash(f"{root['sample_id']}|{condition_id}", 8), 16
    ) % 100000000
    response = model.generate(
        prompt, args.temperature, args.max_new_tokens,
        ("</answer>",), planned_seed,
    )
    parsed = strict_answer_parse(response.text, response.finish_reason, tokenizer)
    attempt = attempt_record(
        tokenizer, prompt, response, args.max_new_tokens, parsed
    )
    metrics = answer_metrics(
        parsed["final_answer"], root.get("gold_answers") or []
    ) if parsed["valid"] else {
        "em": 0.0, "f1": 0.0, "acc": 0.0,
        "normalized_exact": 0.0, "normalized_contains": 0.0,
    }
    usage = {
        "answer_input_tokens": attempt["usage"]["input_tokens"],
        "answer_output_tokens": attempt["usage"]["output_tokens"],
        "answer_reasoning_tokens": attempt["usage"]["reasoning_tokens"],
        "llm_input_tokens": attempt["usage"]["input_tokens"],
        "llm_output_tokens": attempt["usage"]["output_tokens"],
        "reasoning_tokens": attempt["usage"]["reasoning_tokens"],
        "answer_tokens": parsed["payload_tokens"] if parsed["valid"] else 0,
        "total_tokens": (
            attempt["usage"]["input_tokens"] + attempt["usage"]["output_tokens"]
        ),
        "generation_calls": 1,
        "format_repair_calls": 0,
    }
    record: Dict[str, Any] = {
        "schema_version": PROTOCOL_VERSION,
        "created_at": utc_stamp(),
        "benchmark": "bamboogle",
        "stage": STAGE,
        "sample_id": root["sample_id"],
        "condition_id": condition_id,
        "mode": MODE,
        "rollout_id": rollout_id,
        "question": root["question"],
        "gold_answers": root.get("gold_answers") or [],
        "root_type": root["root_type"],
        "root_origin_detail": root.get("root_origin_detail", root["root_type"]),
        "root_query": root.get("root_query", ""),
        "root_prompt_hash": root.get("root_prompt_hash", ""),
        "root_checkpoint_sha256": root.get("root_checkpoint_sha256", ""),
        "retrieval_hash": root["retrieval_hash"],
        "root_cache_record_hash": root.get("root_cache_record_hash", ""),
        "model_name": "qwen3-8b",
        "model_path": args.model_path,
        "model_url": args.model_url,
        "temperature": args.temperature,
        "planned_seed_not_sent": planned_seed,
        "answer_thinking_control": "native_qwen3_thinking_enabled",
        "action_controller": False,
        "neutral_prompt": False,
        "action_tree": False,
        "judge_calls": 0,
        "lexical_evidence_hit_computed": False,
        "evidence_budget": view["evidence_budget"],
        "doc_slot_tokens": view["doc_slot_tokens"],
        "top_k": view["top_k"],
        "allocation": view["allocation"],
        "requested_tokens_by_rank": view["requested_tokens_by_rank"],
        "actual_tokens_by_rank": view["actual_tokens_by_rank"],
        "slot_underfilled_tokens": view["slot_underfilled_tokens"],
        "visible_doc_ids": view["visible_doc_ids"],
        "per_doc_raw_tokens": view["per_doc_raw_tokens"],
        "visible_evidence_tokens": view["visible_evidence_tokens"],
        "evidence_wrapper_tokens": view["evidence_wrapper_tokens"],
        "visible_passages": view["visible_passages"],
        "visible_evidence_hash": sha256_text(view["evidence_text"]),
        "prompt_tokens": prompt_tokens,
        "max_new_tokens": args.max_new_tokens,
        "server_max_total_tokens": args.server_max_total_tokens,
        "answer_status": "valid" if parsed["valid"] else "invalid_format",
        **parsed,
        "metrics": metrics,
        **metrics,
        "generation_attempt": attempt,
        "generation_attempts": {"answer": [attempt]},
        "usage": usage,
        "model_request_count": 1,
        "retrieval_calls_for_frozen_root": 1,
        "retrieval_calls_during_condition": 0,
        "retrieved_docs_total": 5,
        "visible_docs": sum(x > 0 for x in view["requested_tokens_by_rank"]),
        "protocol_violations": parsed["format_errors"],
        "protocol_compliant": parsed["valid"],
        "latency_seconds": float(response.latency_seconds),
    }
    record["record_sha256"] = canonical_json_hash(record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root_cache", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--condition_ids", default="")
    parser.add_argument("--root_scope", choices=(
        "all", "natural_search", "forced_search"), default="all")
    parser.add_argument("--model_url", default="http://127.0.0.1:9101")
    parser.add_argument("--model_path", default=(
        "/home/researcher/smartsearch_rebuild/models/Qwen3-8B"))
    parser.add_argument("--context_length", type=int, default=16384)
    parser.add_argument("--server_max_total_tokens", type=int, default=10086)
    parser.add_argument("--max_new_tokens", type=int, default=6144)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_ids", default="")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_failures", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--allow_unverified_server", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    if args.n_rollouts != 1:
        raise ValueError("Answer-only v5 freezes n_rollouts=1")
    if args.temperature != 0.0:
        raise ValueError("Answer-only v5 freezes temperature=0")
    assert_qwen_configuration(args.model_path, args.model_url)
    condition_ids = [x.strip() for x in args.condition_ids.split(",") if x.strip()]
    condition_ids = condition_ids or [x.condition_id for x in ALL_CONDITIONS]
    unknown = [x for x in condition_ids if x not in CONDITIONS]
    if unknown:
        raise ValueError(f"Unknown conditions: {unknown}")
    sample_ids = [x.strip() for x in args.sample_ids.split(",") if x.strip()]
    roots, global_count = selected_roots(
        args.root_cache, args.limit, sample_ids, args.root_scope,
        args.shard_id, args.num_shards,
    )
    if not roots:
        raise RuntimeError("No roots selected for this shard")
    tokenizer = ExactTokenizer(args.model_path)
    assert_nested_views(tokenizer, roots[0]["documents"])
    jobs = [
        {"sample_id": str(root["sample_id"]), "condition_id": condition_id,
         "mode": MODE, "rollout_id": 0}
        for root in roots for condition_id in condition_ids
    ]
    if args.dry_run:
        print(json.dumps({
            "schema_version": PROTOCOL_VERSION,
            "answer_only": True,
            "global_questions": global_count,
            "questions_this_shard": len(roots),
            "budgets": [CONDITIONS[x].evidence_budget for x in condition_ids],
            "records_this_shard": len(jobs),
            "expected_model_calls": len(jobs),
            "judge_calls": 0,
            "lexical_evidence_hit_computed": False,
        }, ensure_ascii=False, indent=2))
        return 0
    identity = probe_model_identity(args.model_url)
    if not identity["verified_qwen3_8b"] and not args.allow_unverified_server:
        raise RuntimeError("Answer endpoint must be verified Qwen3-8B")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": PROTOCOL_VERSION,
        "root_cache_sha256": sha256_text(
            Path(args.root_cache).read_text(encoding="utf-8")
        ),
        "condition_ids": condition_ids,
        "mode": MODE,
        "stage": STAGE,
        "root_scope": args.root_scope,
        "n_rollouts": 1,
        "temperature": 0.0,
        "context_length": args.context_length,
        "server_max_total_tokens": args.server_max_total_tokens,
        "max_new_tokens": args.max_new_tokens,
        "max_model_calls_per_record": 1,
        "model_path": args.model_path,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        if read_json(manifest_path).get("resume_signature") != canonical_json_hash(signature):
            raise RuntimeError("Resume signature mismatch; use a new RUN_DIR")
    else:
        atomic_write_json(manifest_path, {
            "created_at": utc_stamp(),
            "design": (
                "Qwen3-8B Blank Answer-only Phase A; exactly one generation "
                "per record; local EM/F1; no judges or evidence-hit labels"
            ),
            "resume_signature": canonical_json_hash(signature),
            "signature": signature,
            "conditions": condition_manifest(),
            "expected_records": len(jobs),
            "expected_model_calls": len(jobs),
            "model_identity": identity,
            "server_context": probe_server_context(args.model_url),
            "provenance": provenance(args.model_path),
        })
    store = RecordStore(output_dir)
    completed, failed_keys = store.completed_keys(), store.failed_keys()
    root_by_id = {str(root["sample_id"]): root for root in roots}
    model = ModelClient(
        args.model_url, args.model_path, "sglang", args.timeout, 1
    )
    succeeded = failed = skipped = invalid = 0
    started = time.time()
    for job_index, job in enumerate(jobs, 1):
        key = condition_key(job)
        if key in completed or (key in failed_keys and not args.retry_failures):
            skipped += 1
            continue
        try:
            record = run_job(
                root_by_id[job["sample_id"]], job["condition_id"], 0,
                model, tokenizer, args,
            )
            store.put_record(record)
            succeeded += 1
            invalid += int(record["answer_status"] != "valid")
            elapsed = time.time() - started
            rate = succeeded / elapsed if elapsed > 0 else 0.0
            pending = max(0, len(jobs) - skipped - succeeded - failed)
            eta = pending / rate if rate > 0 else 0.0
            print(
                f"[ok] {job['sample_id']} {job['condition_id']} "
                f"status={record['answer_status']} f1={record['f1']:.3f} "
                f"tokens={record['usage']['answer_output_tokens']} "
                f"latency={record['latency_seconds']:.1f}s "
                f"shard_eta={eta/3600:.2f}h", flush=True,
            )
        except Exception as exc:
            failed += 1
            root = root_by_id[job["sample_id"]]
            failure: Dict[str, Any] = {
                "created_at": utc_stamp(), **job,
                "stage": STAGE,
                "root_type": root.get("root_type"),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "model_requests_planned": 1,
            }
            try:
                view = build_evidence_view(
                    tokenizer, root["documents"], CONDITIONS[job["condition_id"]]
                )
                prompt = answer_prompt(root["question"], view["evidence_text"])
                failure.update({
                    "prompt": prompt,
                    "prompt_sha256": sha256_text(prompt),
                    "prompt_tokens": tokenizer.count(prompt),
                    "visible_evidence_hash": sha256_text(view["evidence_text"]),
                })
            except Exception as audit_exc:
                failure["failure_audit_error"] = str(audit_exc)
            store.put_failure(failure)
            print(f"[failed] {key}: {exc}", file=sys.stderr, flush=True)
    records_total, failures_total = store.consolidate()
    all_records = read_jsonl(output_dir / "records.jsonl")
    elapsed = time.time() - started
    status = {
        "finished_at": utc_stamp(),
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "succeeded_this_attempt": succeeded,
        "failed_this_attempt": failed,
        "skipped_existing": skipped,
        "records_total": records_total,
        "invalid_format_total": sum(
            row.get("answer_status") != "valid" for row in all_records
        ),
        "unresolved_failures_total": failures_total,
        "expected_records": len(jobs),
        "expected_model_calls": len(jobs),
        "recorded_model_calls": sum(
            int(row.get("model_request_count", 0)) for row in all_records
        ),
        "elapsed_seconds_this_attempt": elapsed,
        "complete": records_total == len(jobs) and failures_total == 0,
    }
    atomic_write_json(output_dir / "run_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
