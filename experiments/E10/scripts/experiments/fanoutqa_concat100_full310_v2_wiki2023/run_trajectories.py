#!/usr/bin/env python3
"""Run exact concatenated-context exposure branches with natural multi-search."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

from actions import generate_action, plain_answer_fallback
from hp_common import (
    TOP_K, RETRIEVER_BACKEND, ExactTokenizer, SCHEMA_VERSION, SQLiteRecordStore,
    SafeSGLangClient, canonical_hash, normalize_text, probe_qwen,
    read_json, read_jsonl, utc_stamp, write_json,
)
from prompts import decision_prompt, forced_answer_prompt
from retrieval import CachedRetriever, render_evidence


ROOT_HASH_FIELDS = {"roots_sha256", "roots_semantic_sha256"}


def legacy_root_signature_is_migratable(previous: Dict[str, Any],
                                         current: Dict[str, Any]) -> bool:
    """Accept only the v4.3 gzip-artifact-hash to semantic-hash migration.

    gzip embeds artifact metadata, so hashing the compressed file made an
    unchanged root set look different after every merge.  All non-root-hash
    protocol fields must still match exactly.  Stored record checkpoints are
    verified separately before the manifest is migrated.
    """
    if "roots_sha256" not in previous:
        return False
    if "roots_semantic_sha256" not in current:
        return False
    old = {key: value for key, value in previous.items()
           if key not in ROOT_HASH_FIELDS}
    new = {key: value for key, value in current.items()
           if key not in ROOT_HASH_FIELDS}
    return old == new


def conditions_for_root(root: Dict[str, Any], value: str,
                        condition_mode: str = "budget") -> List[Dict[str, Any]]:
    if condition_mode == "blank":
        if value:
            raise ValueError("--exposures cannot be combined with --condition_mode blank")
        return [blank_condition_from_root(root)]
    if condition_mode != "budget":
        raise ValueError(f"Unknown condition mode: {condition_mode}")
    conditions = list(root.get("exposure_conditions") or [])
    if not conditions:
        raise ValueError(f"Root {root.get('sample_id')} has no exposure conditions")
    if not value:
        return conditions
    requested = {item.strip().upper() for item in value.split(",") if item.strip()}
    selected = [condition for condition in conditions if (
        str(condition["condition_label"]).upper() in requested
        or str(condition["root_exposure_tokens"]) in requested
        or str(condition["condition_id"]).upper() in requested
    )]
    missing = requested - {
        token for condition in selected for token in {
            str(condition["condition_label"]).upper(),
            str(condition["root_exposure_tokens"]).upper(),
            str(condition["condition_id"]).upper(),
        }
    }
    if missing:
        raise ValueError(f"Unavailable exposure selectors for this root: {sorted(missing)}")
    return selected


def blank_condition_from_root(root: Dict[str, Any]) -> Dict[str, Any]:
    """Construct the matched no-budget condition without retokenizing the root."""
    total = int(root["root_full_context_tokens"])
    return {
        "condition_id": "blank_full_context",
        "condition_label": "BLANK",
        "exposure_mode": "blank_full_context",
        "token_cap_per_search": None,
        "root_exposure_tokens": total,
        "root_full_context_tokens": total,
        "step_tokens": 100,
    }


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


def run_one(root: Dict[str, Any], condition: Dict[str, Any], args: argparse.Namespace,
            tokenizer: ExactTokenizer, model: SafeSGLangClient,
            retriever: CachedRetriever) -> Dict[str, Any]:
    first_evidence = render_evidence(tokenizer, root["documents"], condition, 1)
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
            documents, cache_hit, latency = retriever.search(query, TOP_K)
            evidence = render_evidence(
                tokenizer, documents, condition, len(rounds) + 1
            )
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
    retrieved_docs_by_round = [
        len(item["evidence"]["retrieved_docs"]) for item in rounds
    ]
    underfilled_total = sum(
        int(item["evidence"].get("underfilled_tokens", 0)) for item in rounds
    )
    record: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "fanoutqa_natural_multisearch_trajectory",
        "created_at": utc_stamp(),
        "sample_index": root["sample_index"],
        "sample_id": root["sample_id"],
        "question": root["question"],
        "categories": root.get("categories") or [],
        "dataset_name": "fanoutqa", "split": "dev",
        "gold_labels_loaded": False,
        "condition_id": condition["condition_id"],
        "condition_label": condition["condition_label"],
        "exposure_mode": condition["exposure_mode"],
        "root_exposure_tokens": int(condition["root_exposure_tokens"]),
        "root_full_context_tokens": int(condition["root_full_context_tokens"]),
        "token_cap_per_search": condition.get("token_cap_per_search"),
        "exposure_step_tokens": int(condition["step_tokens"]),
        # Compatibility alias.  Actual per-round exposure is always recorded
        # in search_rounds[*].evidence.actual_exposed_token_count.
        "evidence_budget_per_search": int(condition["root_exposure_tokens"]),
        # Compatibility name: this is the requested maximum, not an assertion
        # that every sparse Kiwix query returned exactly this many pages.
        "top_k_per_search": TOP_K,
        "requested_max_k_per_search": TOP_K,
        "retrieval_cardinality": "natural_zero_to_requested_max_no_backfill",
        "retriever_backend": args.retriever_backend,
        "wikipedia_snapshot": "2023-09",
        "doc_slot_tokens": None,
        "evidence_exposure_protocol": (
            "complete_natural_up_to5_no_budget_truncation" if condition["exposure_mode"] == "blank_full_context"
            else "concat_then_global_prefix_100_to_exact_full"
        ),
        "treatment": (
            "blank_no_budget" if condition["exposure_mode"] == "blank_full_context"
            else "evidence_token_budget"
        ),
        "max_searches": args.max_searches,
        "root_checkpoint_sha256": root["root_checkpoint_sha256"],
        "root_query": root["query"],
        "root_retrieval_hash": root["retrieval_hash"],
        "search_rounds": rounds,
        "decision_events": events,
        "search_calls": logical_searches,
        "additional_search_calls_after_root": logical_searches - 1,
        "repeated_query_count": repeated_queries,
        "retrieved_docs_total": sum(retrieved_docs_by_round),
        "retrieved_docs_by_round": retrieved_docs_by_round,
        "retrieval_shortfall_rounds": sum(
            count < TOP_K for count in retrieved_docs_by_round
        ),
        "zero_result_rounds": sum(count == 0 for count in retrieved_docs_by_round),
        "mean_retrieved_docs_per_search": (
            sum(retrieved_docs_by_round) / logical_searches
        ),
        "retrieved_docs_unique": len(set(retrieved_ids)),
        "visible_evidence_tokens_total": visible_total,
        "requested_evidence_token_caps_total": sum(
            int(item["evidence"]["requested_token_cap"]) for item in rounds
        ),
        "underfilled_evidence_tokens_total": underfilled_total,
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
    parser.add_argument("--retriever_backend", default=RETRIEVER_BACKEND)
    parser.add_argument("--model_path", default="/home/researcher/smartsearch_rebuild/models/Qwen3-8B")
    parser.add_argument(
        "--exposures", default="",
        help="Optional comma list such as 100,200,FULL; default is every root-specific condition",
    )
    parser.add_argument(
        "--condition_mode", choices=("budget", "blank"), default="budget",
        help="budget: all 100-token/FULL branches; blank: one untruncated up-to-5 branch",
    )
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
    conditions_by_root = {
        str(root["sample_id"]): conditions_for_root(
            root, args.exposures, args.condition_mode
        )
        for root in roots
    }
    all_jobs = [
        (root, condition) for root in roots
        for condition in conditions_by_root[str(root["sample_id"])]
    ]
    # Keep every budget for one question on one worker/GPU so the model can
    # reuse question/evidence prefixes through SGLang's radix cache.
    jobs = [
        (root, condition) for root_index, root in enumerate(roots)
        if root_index % args.num_shards == args.shard_id
        for condition in conditions_by_root[str(root["sample_id"])]
    ]
    if args.dry_run:
        print(json.dumps({
            "stage": "fanoutqa_natural_multisearch_trajectories",
            "condition_mode": args.condition_mode,
            "roots": len(roots),
            "exposure_selector": args.exposures or "all_root_specific_conditions",
            "exposure_records_per_root_min": min(
                len(value) for value in conditions_by_root.values()
            ) if conditions_by_root else 0,
            "exposure_records_per_root_max": max(
                len(value) for value in conditions_by_root.values()
            ) if conditions_by_root else 0,
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
    wrong_roots = [
        str(root.get("sample_id")) for root in roots
        if root.get("retriever_backend") != args.retriever_backend
        or root.get("wikipedia_snapshot") != "2023-09"
    ]
    if wrong_roots:
        raise RuntimeError(
            "Roots were not built with the required 2023-09 retriever backend; "
            f"first mismatches={wrong_roots[:5]}"
        )
    retriever = CachedRetriever(
        args.retriever_url, args.retrieval_cache, args.timeout, 3,
        backend_id=args.retriever_backend,
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    root_identity = [
        {
            "sample_id": str(root["sample_id"]),
            "root_checkpoint_sha256": str(root["root_checkpoint_sha256"]),
        }
        for root in roots
    ]
    signature = {
        "schema_version": SCHEMA_VERSION, "stage": "trajectories",
        "roots_semantic_sha256": canonical_hash(root_identity),
        "condition_mode": args.condition_mode,
        "exposure_selector": args.exposures,
        "exposure_condition_hash": canonical_hash(conditions_by_root),
        "max_searches": args.max_searches, "requested_max_k": TOP_K,
        "retrieval_cardinality": "natural_zero_to_requested_max_no_backfill",
        "retriever_backend": args.retriever_backend,
        "wikipedia_snapshot": "2023-09",
        "evidence_exposure": (
            "complete_natural_up_to5_no_budget_truncation" if args.condition_mode == "blank"
            else "concat_then_global_prefix_100_to_exact_full"
        ),
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
    store = SQLiteRecordStore(output / "state.sqlite")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing_manifest = read_json(manifest_path)
        expected_resume_signature = canonical_hash(signature)
        if existing_manifest.get("resume_signature") != expected_resume_signature:
            previous_signature = existing_manifest.get("signature") or {}
            if not legacy_root_signature_is_migratable(
                previous_signature, signature
            ):
                store.close()
                raise RuntimeError("Resume signature mismatch; use a new RUN_DIR")

            # Every root assigned to this shard must already be represented by
            # at least one successful trajectory, and every stored trajectory
            # must point to the exact current root checkpoint.  This prevents a
            # gzip-metadata migration from masking a genuinely changed root.
            expected_root_hashes = {
                str(root["sample_id"]): str(root["root_checkpoint_sha256"])
                for root_index, root in enumerate(roots)
                if root_index % args.num_shards == args.shard_id
            }
            represented_roots = set()
            mismatches = []
            verified_records = 0
            for record in store.iter_records():
                sample_id = str(record.get("sample_id"))
                supplied = str(record.get("root_checkpoint_sha256"))
                expected = expected_root_hashes.get(sample_id)
                if expected is None or supplied != expected:
                    mismatches.append(sample_id)
                else:
                    represented_roots.add(sample_id)
                verified_records += 1
            missing_roots = sorted(set(expected_root_hashes) - represented_roots)
            if mismatches or missing_roots:
                store.close()
                raise RuntimeError(
                    "Unsafe legacy root-signature migration refused: "
                    f"checkpoint_mismatches={len(mismatches)}, "
                    f"roots_without_verified_records={len(missing_roots)}"
                )
            migrated = dict(existing_manifest)
            migrated["signature"] = signature
            migrated["resume_signature"] = expected_resume_signature
            migrated["root_signature_migration"] = {
                "migrated_at": utc_stamp(),
                "from": "compressed_gzip_artifact_sha256",
                "to": "ordered_sample_and_root_checkpoint_semantic_sha256",
                "verified_existing_records": verified_records,
                "verified_shard_roots": len(expected_root_hashes),
                "checkpoint_mismatches": 0,
            }
            write_json(manifest_path, migrated)
            print(
                "[resume-migration] verified unchanged root checkpoints; "
                "replaced unstable gzip artifact hash with semantic root hash",
                flush=True,
            )
    else:
        write_json(manifest_path, {
            "created_at": utc_stamp(), "signature": signature,
            "resume_signature": canonical_hash(signature),
            "expected_records": len(jobs), "model_identity": identity,
        })

    # v4.3.1 reconstructed truncated evidence with decode(prefix_ids).  A
    # tokenizer cleanup edge case made one nominal 900-token record expose only
    # four tokens.  Invalidate only positive-backoff records produced by that
    # deprecated policy; v4.3.3 will regenerate them from original-text offsets.
    deprecated_boundary_keys = []
    for stored_record in list(store.iter_records()):
        deprecated = any(
            evidence.get("prefix_boundary_policy")
            == "largest_exact_roundtrippable_token_prefix_not_exceeding_cap"
            and int(evidence.get("roundtrip_boundary_backoff_tokens", 0)) > 0
            for evidence in (
                item.get("evidence") or {}
                for item in stored_record.get("search_rounds") or []
            )
        )
        if not deprecated:
            continue
        key = (
            f"{stored_record['sample_id']}|{stored_record['condition_id']}"
        )
        store.invalidate_record(key, {
            "created_at": utc_stamp(),
            "sample_id": stored_record["sample_id"],
            "condition_id": stored_record["condition_id"],
            "error_type": "DeprecatedEvidenceBoundaryPolicy",
            "error": (
                "v4.3.1 decode-derived evidence boundary invalidated for "
                "exact original-text-offset regeneration"
            ),
        })
        deprecated_boundary_keys.append(key)
    if deprecated_boundary_keys:
        print(
            "[boundary-policy-upgrade] invalidated for exact retry: "
            + ", ".join(deprecated_boundary_keys),
            flush=True,
        )
    completed = store.completed_keys()
    failures = store.failed_keys()
    succeeded = failed = skipped = 0
    started = time.time()
    for root, condition in jobs:
        key = f"{root['sample_id']}|{condition['condition_id']}"
        if key in completed or (key in failures and not args.retry_failures):
            skipped += 1
            continue
        try:
            record = run_one(root, condition, args, tokenizer, model, retriever)
            store.put_record(key, record)
            succeeded += 1
            elapsed = time.time() - started
            rate = succeeded / elapsed if elapsed else 0.0
            remaining = max(0, len(jobs) - skipped - succeeded - failed)
            eta = remaining / rate if rate else 0.0
            print(
                f"[ok] {root['sample_id']} {condition['condition_id']} "
                f"searches={record['search_calls']} status={record['answer_status']} "
                f"repairs={record['usage']['format_repair_calls']} eta={eta/3600:.2f}h",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            store.put_failure(key, {
                "created_at": utc_stamp(), "sample_id": root["sample_id"],
                "condition": condition, "error_type": type(exc).__name__,
                "error": str(exc), "traceback": traceback.format_exc(),
            })
            print(f"[failed] {key}: {exc}", flush=True)
    records_total, failures_total = store.export(output)
    status = {
        "finished_at": utc_stamp(), "stage": "trajectories",
        "condition_mode": args.condition_mode,
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
