#!/usr/bin/env python3
"""Shared data loading, identity checks, and accounting for Qwen3-only runs."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from common import as_string_list, first_nonempty, read_json, read_jsonl, stable_hash


EXPECTED_MODEL_BASENAME = "qwen3-8b"


def load_bamboogle_questions(
    intermediate_path: str | Path,
    limit: int = 0,
    sample_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = read_json(intermediate_path)
    if isinstance(raw, dict):
        for key in ("data", "items", "samples", "records"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {intermediate_path}")
    wanted = {str(item) for item in (sample_ids or [])}
    samples: List[Dict[str, Any]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        output = item.get("output") if isinstance(item.get("output"), dict) else {}
        sample_id = str(first_nonempty(item, ("id", "sample_id", "qid"), f"sample_{index}"))
        if wanted and sample_id not in wanted:
            continue
        question = str(first_nonempty(item, ("question", "query"), "")).strip()
        if not question:
            continue
        gold = as_string_list(first_nonempty(item, ("golden_answers", "answers", "answer"), []))
        samples.append(
            {
                "sample_id": sample_id,
                "question": question,
                "gold_answers": gold,
                "paper_prediction": str(first_nonempty(output, ("pred", "prediction", "answer"), "")),
                "paper_metric_score": output.get("metric_score", {}),
            }
        )
        if limit and len(samples) >= limit:
            break
    return samples, {
        "intermediate_path": str(intermediate_path),
        "raw_items": len(raw),
        "loaded_samples": len(samples),
        "selected_sample_ids": [sample["sample_id"] for sample in samples],
    }


def load_qwen_checkpoints(
    checkpoint_path: str | Path,
    _trace_path: Optional[str | Path] = None,
    limit: int = 0,
    sample_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows = read_jsonl(checkpoint_path)
    wanted = {str(item) for item in (sample_ids or [])}
    samples: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for row in rows:
        sample_id = str(row.get("sample_id", ""))
        if wanted and sample_id not in wanted:
            continue
        required = ("question", "checkpoint_prefix", "initial_query", "initial_result")
        missing = [key for key in required if not str(row.get(key, "")).strip()]
        if missing:
            warnings.append(f"{sample_id}: missing {missing}")
            continue
        if str(row.get("model_name", "")).lower() != EXPECTED_MODEL_BASENAME:
            warnings.append(f"{sample_id}: rejected non-Qwen checkpoint model={row.get('model_name')!r}")
            continue
        samples.append(
            {
                "sample_id": sample_id,
                "question": str(row["question"]),
                "gold_answers": list(row.get("gold_answers") or []),
                "checkpoint_prefix": str(row["checkpoint_prefix"]),
                "initial_query": str(row["initial_query"]),
                "initial_result": str(row["initial_result"]),
                "initial_documents": list(row.get("initial_documents") or []),
                "checkpoint_source": str(row.get("checkpoint_source", "qwen3_blank.unknown")),
                "root_origin": str(row.get("root_origin", "unknown")),
                "blank_record_sha1": str(row.get("blank_record_sha1", "")),
                "root_checkpoint_sha1": str(
                    row.get("root_checkpoint_sha1") or stable_hash(str(row["checkpoint_prefix"]), 40)
                ),
                "original_prediction": str(row.get("blank_prediction", "")),
                "original_metric_score": dict(row.get("blank_metrics") or {}),
                "original_search_count": int(row.get("blank_search_calls", 0)),
                "original_search_count_from_final_response": int(row.get("blank_search_calls", 0)),
                "original_search_count_from_trace": 0,
            }
        )
        if limit and len(samples) >= limit:
            break
    return samples, {
        "checkpoint_path": str(checkpoint_path),
        "raw_items": len(rows),
        "loaded_samples": len(samples),
        "warnings": warnings,
        "checkpoint_sources": dict(Counter(sample["checkpoint_source"] for sample in samples)),
        "root_origins": dict(Counter(sample["root_origin"] for sample in samples)),
    }


def assert_qwen_configuration(model_path: str, model_url: str) -> None:
    normalized = str(model_path).replace("\\", "/").rstrip("/").lower()
    if normalized.split("/")[-1] != EXPECTED_MODEL_BASENAME:
        raise RuntimeError(
            "Formal Qwen3-only protocol requires MODEL_PATH ending in Qwen3-8B; "
            f"received {model_path!r}"
        )
    if re.search(r"smartsearch[-_]?3b", normalized, flags=re.I):
        raise RuntimeError(f"SmartSearch-3B model path is forbidden: {model_path}")
    if re.search(r":9000(?:/|$)", model_url):
        raise RuntimeError(f"Port 9000 is reserved for the old SmartSearch model: {model_url}")


def probe_model_identity(model_url: str, timeout: float = 15.0) -> Dict[str, Any]:
    """Ask SGLang/OpenAI endpoints which model is actually loaded."""
    attempts: List[Dict[str, Any]] = []
    for suffix in ("/get_model_info", "/v1/models"):
        url = model_url.rstrip("/") + suffix
        try:
            response = requests.get(url, timeout=timeout)
            body = response.text[:10000]
            attempt = {"url": url, "status_code": response.status_code, "body": body}
            attempts.append(attempt)
            if response.ok:
                lower = body.lower()
                verified = EXPECTED_MODEL_BASENAME in lower and "smartsearch-3b" not in lower
                if EXPECTED_MODEL_BASENAME in lower or "smartsearch-3b" in lower:
                    return {"verified_qwen3_8b": verified, "evidence": attempt, "attempts": attempts}
        except Exception as exc:
            attempts.append({"url": url, "error": str(exc)})
    return {"verified_qwen3_8b": False, "evidence": None, "attempts": attempts}


def enforce_server_identity(model_url: str, strict: bool = True) -> Dict[str, Any]:
    report = probe_model_identity(model_url)
    if strict and not report["verified_qwen3_8b"]:
        raise RuntimeError(
            "Could not verify that the server actually hosts Qwen3-8B. "
            "Inspect /get_model_info or /v1/models; use --allow_unverified_server only for diagnostics. "
            + json.dumps(report, ensure_ascii=False)
        )
    return report


def sum_event_costs(events: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(events)
    output = {
        "search_calls": sum(event.get("action_code") == "S" for event in rows),
        "read_calls": sum(event.get("action_code") in {"R", "N"} for event in rows),
        "retrieved_docs_total": sum(int(event.get("retrieved_docs", 0)) for event in rows),
        "retrieved_docs_unique": len(
            {
                str(document.get("key") or document.get("id") or document.get("contents", ""))
                for event in rows
                for document in event.get("documents", event.get("tool", {}).get("documents", []))
            }
        ),
        "retrieved_tokens_total": sum(int(event.get("retrieved_tokens_total", 0)) for event in rows),
        "new_evidence_tokens": sum(int(event.get("new_evidence_tokens", 0)) for event in rows),
        "read_input_tokens": sum(int(event.get("read_input_tokens", 0)) for event in rows),
        "read_output_tokens": sum(int(event.get("read_output_tokens", 0)) for event in rows),
        "reasoning_tokens": sum(int(event.get("reasoning_tokens", 0)) for event in rows),
        "answer_tokens": sum(int(event.get("answer_tokens", 0)) for event in rows),
        "total_llm_input_tokens": sum(int(event.get("llm_input_tokens", 0)) for event in rows),
        "total_llm_output_tokens": sum(int(event.get("llm_output_tokens", 0)) for event in rows),
        "latency_seconds": sum(float(event.get("latency_seconds", 0.0)) for event in rows),
    }
    output["llm_tokens_total"] = output["total_llm_input_tokens"] + output["total_llm_output_tokens"]
    output["total_cost_tokens"] = output["llm_tokens_total"] + output["retrieved_tokens_total"]
    output["total_cost"] = float(output["total_cost_tokens"])
    return output

