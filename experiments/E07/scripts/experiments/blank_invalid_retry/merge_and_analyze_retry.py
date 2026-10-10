#!/usr/bin/env python3
"""Merge exact-prompt retry shards and report recovery without replacing primary data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "blank_answer_only"
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from common import (  # noqa: E402
    atomic_write_json, atomic_write_jsonl, canonical_json_hash, condition_key,
    read_json, read_jsonl, utc_stamp,
)

EXPECTED_SOURCE_SHA256 = (
    "68c5be70bed953053cc9cdcddba8cc08ab6922f1f7aed09319d9b41dbdfc6042"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mean(rows: Iterable[Dict[str, Any]], key: str) -> float:
    rows = list(rows)
    return sum(float(row.get(key, 0.0)) for row in rows) / len(rows) if rows else 0.0


def finish_type(record: Dict[str, Any]) -> str:
    value = (record.get("generation_attempt") or {}).get("finish_reason")
    if isinstance(value, dict):
        return str(value.get("type", "")).lower()
    return str(value or "").lower()


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original_records", required=True)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--expected_invalid", type=int, default=271)
    parser.add_argument("--expected_source_sha256", default=EXPECTED_SOURCE_SHA256)
    args = parser.parse_args()

    source = Path(args.original_records)
    source_sha = file_sha256(source)
    if args.expected_source_sha256 and source_sha != args.expected_source_sha256:
        raise ValueError("Original records SHA256 mismatch")
    original = read_jsonl(source)
    original_by_key = {condition_key(row): row for row in original}
    if len(original_by_key) != len(original):
        raise ValueError("Duplicate keys in original records")
    invalid_by_key = {
        key: row for key, row in original_by_key.items()
        if row.get("answer_status") == "invalid_format"
    }
    if len(invalid_by_key) != args.expected_invalid:
        raise ValueError(
            f"Expected {args.expected_invalid} original invalid records; "
            f"found {len(invalid_by_key)}"
        )

    retries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    manifests = []
    statuses = []
    errors: List[str] = []
    for raw_dir in args.shard_dirs:
        shard = Path(raw_dir)
        for required in ("manifest.json", "run_status.json", "records.jsonl", "failures.jsonl"):
            if not (shard / required).exists():
                errors.append(f"missing:{shard / required}")
        if errors:
            continue
        manifests.append(read_json(shard / "manifest.json"))
        statuses.append(read_json(shard / "run_status.json"))
        retries.extend(read_jsonl(shard / "records.jsonl"))
        failures.extend(read_jsonl(shard / "failures.jsonl"))

    retry_keys = [condition_key(row) for row in retries]
    if len(retry_keys) != len(set(retry_keys)):
        errors.append("duplicate_retry_keys")
    if set(retry_keys) != set(invalid_by_key):
        errors.append(
            f"retry_key_set_mismatch:missing={len(set(invalid_by_key)-set(retry_keys))}:"
            f"extra={len(set(retry_keys)-set(invalid_by_key))}"
        )
    if failures:
        errors.append(f"unresolved_transport_failures:{len(failures)}")
    if any(not status.get("complete") for status in statuses):
        errors.append("incomplete_shard")

    comparison_rows: List[Dict[str, Any]] = []
    retry_by_key: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    for retry in retries:
        key = condition_key(retry)
        parent = invalid_by_key.get(key)
        if parent is None:
            continue
        supplied = retry.get("record_sha256")
        unhashed = dict(retry)
        unhashed.pop("record_sha256", None)
        if supplied != canonical_json_hash(unhashed):
            errors.append(f"retry_record_hash:{key}")
        if retry.get("retry_of_record_sha256") != parent.get("record_sha256"):
            errors.append(f"parent_hash:{key}")
        parent_prompt_hash = (parent.get("generation_attempt") or {}).get("prompt_sha256")
        retry_prompt_hash = (retry.get("generation_attempt") or {}).get("prompt_sha256")
        if parent_prompt_hash != retry_prompt_hash or not retry.get("same_prompt_verified"):
            errors.append(f"prompt_changed:{key}")
        if retry.get("visible_evidence_hash") != parent.get("visible_evidence_hash"):
            errors.append(f"evidence_changed:{key}")
        if int(retry.get("model_request_count_this_stage", -1)) != 1:
            errors.append(f"retry_request_count:{key}")
        retry_by_key[key] = retry
        comparison_rows.append({
            "sample_id": key[0], "condition_id": key[1],
            "budget": int(parent["evidence_budget"]),
            "root_type": parent.get("root_type"),
            "original_finish": finish_type(parent),
            "original_format_errors": "|".join(parent.get("format_errors") or []),
            "original_output_tokens": int((parent.get("usage") or {}).get("answer_output_tokens", 0)),
            "retry_finish": finish_type(retry),
            "retry_valid": retry.get("answer_status") == "valid",
            "retry_format_errors": "|".join(retry.get("format_errors") or []),
            "retry_output_tokens": int((retry.get("retry_usage") or {}).get("answer_output_tokens", 0)),
            "identical_output": bool(retry.get("retry_output_identical_to_original")),
            "retry_em": float(retry.get("em", 0.0)),
            "retry_f1": float(retry.get("f1", 0.0)),
            "retry_acc": float(retry.get("acc", 0.0)),
            "retry_final_answer": retry.get("final_answer", ""),
        })

    retries.sort(key=condition_key)
    comparison_rows.sort(key=lambda row: (row["sample_id"], row["condition_id"]))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(output / "retry_records.jsonl", retries)
    atomic_write_jsonl(output / "retry_failures.jsonl", failures)
    write_csv(output / "retry_comparison.csv", comparison_rows)
    atomic_write_jsonl(
        output / "remaining_invalid_retry_records.jsonl",
        [row for row in retries if row.get("answer_status") != "valid"],
    )

    # A selective-retry policy spends the original call for every record and
    # one additional call only when the primary output is format-invalid.
    policy_rows: List[Dict[str, Any]] = []
    for parent in original:
        key = condition_key(parent)
        retry = retry_by_key.get(key)
        selected = retry if retry is not None else parent
        additional = (retry.get("retry_usage") or {}) if retry else {}
        policy_rows.append({
            "sample_id": key[0], "condition_id": key[1],
            "budget": int(parent["evidence_budget"]),
            "selected_attempt": "retry" if retry else "primary",
            "answer_status": selected.get("answer_status"),
            "em": float(selected.get("em", 0.0)),
            "f1": float(selected.get("f1", 0.0)),
            "acc": float(selected.get("acc", 0.0)),
            "additional_model_calls": 1 if retry else 0,
            "additional_input_tokens": int(additional.get("answer_input_tokens", 0)),
            "additional_output_tokens": int(additional.get("answer_output_tokens", 0)),
            "primary_record_sha256": parent.get("record_sha256"),
            "retry_record_sha256": retry.get("record_sha256") if retry else "",
        })
    atomic_write_jsonl(output / "selective_retry_policy_records.jsonl", policy_rows)

    by_budget: List[Dict[str, Any]] = []
    for budget in range(50, 751, 50):
        original_budget = [row for row in original if int(row["evidence_budget"]) == budget]
        retry_budget = [row for row in retries if int(row["evidence_budget"]) == budget]
        policy_budget = [row for row in policy_rows if row["budget"] == budget]
        by_budget.append({
            "budget": budget,
            "original_records": len(original_budget),
            "original_invalid": sum(row.get("answer_status") != "valid" for row in original_budget),
            "retry_records": len(retry_budget),
            "retry_valid": sum(row.get("answer_status") == "valid" for row in retry_budget),
            "retry_recovery_rate": (
                sum(row.get("answer_status") == "valid" for row in retry_budget) / len(retry_budget)
                if retry_budget else 0.0
            ),
            "retry_identical_output_rate": mean(retry_budget, "retry_output_identical_to_original"),
            "primary_em": mean(original_budget, "em"),
            "primary_f1": mean(original_budget, "f1"),
            "selective_retry_policy_em": mean(policy_budget, "em"),
            "selective_retry_policy_f1": mean(policy_budget, "f1"),
            "selective_retry_policy_valid_rate": mean(
                [{"value": row.get("answer_status") == "valid"} for row in policy_budget], "value"
            ),
        })
    write_csv(output / "by_budget.csv", by_budget)

    retry_valid = sum(row.get("answer_status") == "valid" for row in retries)
    original_total_tokens = sum(int((row.get("usage") or {}).get("total_tokens", 0)) for row in original)
    retry_additional_tokens = sum(int((row.get("retry_usage") or {}).get("total_tokens", 0)) for row in retries)
    summary = {
        "created_at": utc_stamp(),
        "source_records_sha256": source_sha,
        "integrity": {
            "original_records": len(original),
            "original_invalid_records": len(invalid_by_key),
            "retry_records": len(retries),
            "retry_unique_keys": len(set(retry_keys)),
            "same_prompt_all": all(row.get("same_prompt_verified") for row in retries),
            "same_evidence_all": all(row.get("same_visible_evidence_verified") for row in retries),
            "new_model_calls": sum(int(row.get("model_request_count_this_stage", 0)) for row in retries),
            "judge_calls": sum(int(row.get("judge_calls", 0)) for row in retries),
            "transport_failures": len(failures),
            "errors": errors,
            "passed": not errors,
        },
        "retry_outcome": {
            "valid_after_exact_retry": retry_valid,
            "remaining_invalid": len(retries) - retry_valid,
            "format_recovery_rate": retry_valid / len(retries) if retries else 0.0,
            "identical_output_count": sum(row.get("retry_output_identical_to_original") for row in retries),
            "identical_output_rate": mean(retries, "retry_output_identical_to_original"),
            "retry_em_over_271": mean(retries, "em"),
            "retry_f1_over_271": mean(retries, "f1"),
        },
        "primary_one_call": {
            "em": mean(original, "em"), "f1": mean(original, "f1"),
            "valid_rate": mean(
                [{"value": row.get("answer_status") == "valid"} for row in original], "value"
            ),
            "model_calls": len(original), "total_tokens": original_total_tokens,
        },
        "selective_retry_policy_supplementary": {
            "em": mean(policy_rows, "em"), "f1": mean(policy_rows, "f1"),
            "valid_rate": mean(
                [{"value": row.get("answer_status") == "valid"} for row in policy_rows], "value"
            ),
            "total_model_calls": len(original) + len(retries),
            "additional_model_calls": len(retries),
            "additional_tokens": retry_additional_tokens,
            "total_tokens_including_primary_and_retries": original_total_tokens + retry_additional_tokens,
            "warning": (
                "Post-hoc selective-retry policy; do not replace or relabel the "
                "pre-registered one-call primary result."
            ),
        },
        "original_error_type_counts": dict(Counter(
            error for row in invalid_by_key.values()
            for error in row.get("format_errors") or []
        )),
    }
    atomic_write_json(output / "retry_summary.json", summary)

    report = f"""# Bamboogle v5：271 条 invalid 的严格同 Prompt 重跑

## 完整性

- 原始记录：{len(original)}；原始 invalid：{len(invalid_by_key)}。
- 新增模型调用：{summary['integrity']['new_model_calls']}；Judge：0。
- Prompt 完全相同：{summary['integrity']['same_prompt_all']}；证据 hash 完全相同：{summary['integrity']['same_evidence_all']}。
- 合并审计：{'PASSED' if summary['integrity']['passed'] else 'FAILED'}。

## 重跑结果

- 恢复为 valid：{retry_valid}/{len(retries)}（{summary['retry_outcome']['format_recovery_rate']:.2%}）。
- 仍为 invalid：{len(retries)-retry_valid}/{len(retries)}。
- 与第一次输出逐字相同：{summary['retry_outcome']['identical_output_count']}/{len(retries)}（{summary['retry_outcome']['identical_output_rate']:.2%}）。
- 271 条重跑自身：EM={summary['retry_outcome']['retry_em_over_271']:.4f}，F1={summary['retry_outcome']['retry_f1_over_271']:.4f}。

## 选择性重试策略（仅补充分析）

- 原始单调用主结果：EM={summary['primary_one_call']['em']:.4f}，F1={summary['primary_one_call']['f1']:.4f}，valid={summary['primary_one_call']['valid_rate']:.2%}。
- invalid 后自动再试一次：EM={summary['selective_retry_policy_supplementary']['em']:.4f}，F1={summary['selective_retry_policy_supplementary']['f1']:.4f}，valid={summary['selective_retry_policy_supplementary']['valid_rate']:.2%}。
- 额外调用：{len(retries)}；额外 tokens：{retry_additional_tokens}。

该策略是事后只对 invalid 重试，不能覆盖或冒充一次调用的主实验结果。它应作为“格式失败检测 + 一次重试”的独立系统策略报告。
"""
    (output / "RETRY_REPORT_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
