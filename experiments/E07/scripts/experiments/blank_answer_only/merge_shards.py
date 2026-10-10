#!/usr/bin/env python3
"""Merge four Answer-only shards and verify the frozen Phase-A protocol."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    PROTOCOL_VERSION, atomic_write_json, atomic_write_jsonl,
    canonical_json_hash, condition_key, read_json, read_jsonl, utc_stamp,
)
from evidence_views import CONDITIONS, requested_tokens_by_rank  # noqa: E402


def validate_record(record: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    condition_id = str(record.get("condition_id", ""))
    if record.get("schema_version") != PROTOCOL_VERSION:
        errors.append("schema_version")
    if condition_id not in CONDITIONS:
        return errors + ["unknown_condition"]
    budget = CONDITIONS[condition_id].evidence_budget
    if int(record.get("evidence_budget", -1)) != budget:
        errors.append("budget_condition_mismatch")
    if record.get("requested_tokens_by_rank") != requested_tokens_by_rank(budget):
        errors.append("allocation_mismatch")
    if record.get("mode") != "blank_answer_only" or record.get("stage") != "answer_generation":
        errors.append("mode_or_stage")
    if any(record.get(key) is not False for key in (
        "action_controller", "neutral_prompt", "action_tree"
    )):
        errors.append("forbidden_experiment_mode")
    if int(record.get("judge_calls", -1)) != 0:
        errors.append("judge_calls")
    if record.get("lexical_evidence_hit_computed") is not False:
        errors.append("lexical_hit_must_be_deferred")
    if "answer_judge" in record or "evidence_judge" in record or "evidence_hit" in record:
        errors.append("offline_field_leakage")
    if int(record.get("model_request_count", -1)) != 1:
        errors.append("model_request_count")
    attempts = (record.get("generation_attempts") or {}).get("answer") or []
    if len(attempts) != 1:
        errors.append("answer_attempt_count")
    if int((record.get("usage") or {}).get("generation_calls", -1)) != 1:
        errors.append("usage_generation_calls")
    if int((record.get("usage") or {}).get("format_repair_calls", -1)) != 0:
        errors.append("format_repair_calls")
    if int(record.get("retrieval_calls_during_condition", -1)) != 0:
        errors.append("retrieval_during_condition")
    status = record.get("answer_status")
    metrics = record.get("metrics") or {}
    if status == "valid":
        if not record.get("protocol_compliant"):
            errors.append("valid_not_compliant")
        if record.get("format_errors"):
            errors.append("valid_has_format_errors")
        answer = str(record.get("final_answer", ""))
        if not answer or "<" in answer or ">" in answer:
            errors.append("invalid_final_answer_payload")
    elif status == "invalid_format":
        if record.get("protocol_compliant"):
            errors.append("invalid_marked_compliant")
        if not record.get("format_errors"):
            errors.append("invalid_missing_format_errors")
        if str(record.get("final_answer", "")):
            errors.append("invalid_nonempty_scored_answer")
        if any(float(metrics.get(key, -1)) != 0.0 for key in (
            "em", "f1", "acc", "normalized_exact", "normalized_contains"
        )):
            errors.append("invalid_nonzero_metrics")
    else:
        errors.append("unknown_answer_status")
    for key in ("em", "f1", "acc", "normalized_exact", "normalized_contains"):
        if record.get(key) != metrics.get(key):
            errors.append(f"metric_mirror:{key}")
    supplied_hash = record.get("record_sha256")
    without_hash = dict(record)
    without_hash.pop("record_sha256", None)
    if supplied_hash != canonical_json_hash(without_hash):
        errors.append("record_sha256")
    return errors


def validate_nestedness(records: List[Dict[str, Any]]) -> List[str]:
    errors: List[str] = []
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(str(record["sample_id"]), str(record["mode"]),
                int(record.get("rollout_id", 0)))].append(record)
    for key, rows in groups.items():
        rows.sort(key=lambda row: int(row["evidence_budget"]))
        previous: Dict[int, str] = {}
        for row in rows:
            current = {
                int(item["rank"]): str(item.get("visible_text", ""))
                for item in row.get("visible_passages", [])
            }
            for rank, text in previous.items():
                candidate = current.get(rank, "")
                shared = min(len(text), len(candidate))
                if shared and text[:max(0, shared - 8)] != candidate[:max(0, shared - 8)]:
                    errors.append(
                        f"non_nested:{key}:rank{rank}:b{row['evidence_budget']}"
                    )
            previous = current
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    records: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    manifests: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []
    errors: List[str] = []
    for raw_dir in args.shard_dirs:
        shard_dir = Path(raw_dir)
        required = ("manifest.json", "run_status.json", "records.jsonl", "failures.jsonl")
        missing = [str(shard_dir / name) for name in required
                   if not (shard_dir / name).exists()]
        if missing:
            errors.extend(f"missing:{name}" for name in missing)
            continue
        manifests.append(read_json(shard_dir / "manifest.json"))
        statuses.append(read_json(shard_dir / "run_status.json"))
        records.extend(read_jsonl(shard_dir / "records.jsonl"))
        failures.extend(read_jsonl(shard_dir / "failures.jsonl"))
    keys = [condition_key(record) for record in records]
    duplicates = [key for key, count in Counter(keys).items() if count > 1]
    if duplicates:
        errors.append(f"duplicate_keys:{len(duplicates)}")
    expected = sum(int(item.get("expected_records", 0)) for item in manifests)
    if len(records) != expected:
        errors.append(f"record_count:{len(records)}!={expected}")
    if failures:
        errors.append(f"unresolved_transport_failures:{len(failures)}")
    for status in statuses:
        if not status.get("complete"):
            errors.append(f"incomplete_shard:{status.get('shard_id')}")
    record_errors = []
    for record in records:
        found = validate_record(record)
        if found:
            record_errors.append({"key": condition_key(record), "errors": found})
    if record_errors:
        errors.append(f"structurally_invalid_records:{len(record_errors)}")
    nested_errors = validate_nestedness(records)
    if nested_errors:
        errors.append(f"nestedness_errors:{len(nested_errors)}")
    records.sort(key=condition_key)
    failures.sort(key=lambda row: (
        str(row.get("sample_id")), str(row.get("condition_id")),
        str(row.get("mode")), int(row.get("rollout_id", 0)),
    ))
    atomic_write_jsonl(output_dir / "records.jsonl", records)
    atomic_write_jsonl(output_dir / "failures.jsonl", failures)
    invalid = sum(record.get("answer_status") != "valid" for record in records)
    report = {
        "created_at": utc_stamp(),
        "schema_version": PROTOCOL_VERSION,
        "answer_only": True,
        "shards_seen": len(manifests),
        "records": len(records),
        "expected_records": expected,
        "unique_keys": len(set(keys)),
        "model_calls": sum(int(record.get("model_request_count", 0)) for record in records),
        "expected_model_calls": expected,
        "judge_calls": sum(int(record.get("judge_calls", 0)) for record in records),
        "valid_answer_records": len(records) - invalid,
        "invalid_format_records": invalid,
        "invalid_format_rate": invalid / len(records) if records else 0.0,
        "transport_failures": len(failures),
        "sample_ids": len({str(record.get("sample_id")) for record in records}),
        "condition_ids": sorted({str(record.get("condition_id")) for record in records}),
        "record_error_examples": record_errors[:20],
        "nested_error_examples": nested_errors[:20],
        "errors": errors,
        "passed": not errors,
    }
    atomic_write_json(output_dir / "verify_report.json", report)
    atomic_write_json(output_dir / "parallel_manifest.json", {
        "created_at": utc_stamp(),
        "schema_version": PROTOCOL_VERSION,
        "shard_manifests": manifests,
        "merge_report": report,
    })
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
