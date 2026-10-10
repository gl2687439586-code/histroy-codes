#!/usr/bin/env python3
"""Identity, checkpoint loading, and accounting for the OpenRouter v6 run."""

from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

from common import as_string_list, first_nonempty, read_json, read_jsonl, stable_hash


EXPECTED_MODEL_ID = "openai/gpt-5.6-sol"
PROTOCOL_VERSION = "openrouter-gpt-5.6-sol-src-v6"


def openrouter_runtime_config() -> Dict[str, Any]:
    return {
        "provider_order": [os.environ.get("OPENROUTER_PROVIDER", "openai")],
        "allow_fallbacks": os.environ.get("OPENROUTER_ALLOW_FALLBACKS", "0").lower()
        in {"1", "true", "yes", "on"},
        "reasoning_effort": os.environ.get("OPENROUTER_REASONING_EFFORT", "medium"),
        "response_cache": False,
        "router_metadata": True,
    }


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
        samples.append(
            {
                "sample_id": sample_id,
                "question": question,
                "gold_answers": as_string_list(
                    first_nonempty(item, ("golden_answers", "answers", "answer"), [])
                ),
                "paper_prediction": str(
                    first_nonempty(output, ("pred", "prediction", "answer"), "")
                ),
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


def load_openrouter_checkpoints(
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
        missing = [
            key
            for key in ("question", "checkpoint_prefix", "initial_query", "initial_result")
            if not str(row.get(key, "")).strip()
        ]
        if missing:
            warnings.append(f"{sample_id}: missing {missing}")
            continue
        if str(row.get("model_name", "")).lower() != EXPECTED_MODEL_ID:
            warnings.append(
                f"{sample_id}: rejected checkpoint model={row.get('model_name')!r}"
            )
            continue
        prefix = str(row["checkpoint_prefix"])
        samples.append(
            {
                "sample_id": sample_id,
                "question": str(row["question"]),
                "gold_answers": list(row.get("gold_answers") or []),
                "checkpoint_prefix": prefix,
                "initial_query": str(row["initial_query"]),
                "initial_result": str(row["initial_result"]),
                "initial_documents": list(row.get("initial_documents") or []),
                "checkpoint_source": str(
                    row.get("checkpoint_source", "openrouter_blank.unknown")
                ),
                "root_origin": str(row.get("root_origin", "unknown")),
                "blank_record_sha1": str(row.get("blank_record_sha1", "")),
                "root_checkpoint_sha1": str(
                    row.get("root_checkpoint_sha1") or stable_hash(prefix, 40)
                ),
                "original_prediction": str(row.get("blank_prediction", "")),
                "original_metric_score": dict(row.get("blank_metrics") or {}),
                "original_search_count": int(row.get("blank_search_calls", 0)),
                "original_search_count_from_final_response": int(
                    row.get("blank_search_calls", 0)
                ),
                "original_search_count_from_trace": 0,
                "root_generation_event": dict(row.get("root_generation_event") or {}),
            }
        )
        if limit and len(samples) >= limit:
            break
    return samples, {
        "checkpoint_path": str(checkpoint_path),
        "raw_items": len(rows),
        "loaded_samples": len(samples),
        "warnings": warnings,
        "checkpoint_sources": dict(
            Counter(sample["checkpoint_source"] for sample in samples)
        ),
        "root_origins": dict(Counter(sample["root_origin"] for sample in samples)),
    }


def assert_openrouter_configuration(
    model_id: str, model_url: str, require_key: bool = True
) -> None:
    if model_id.strip().lower() != EXPECTED_MODEL_ID:
        raise RuntimeError(
            f"Formal protocol requires model {EXPECTED_MODEL_ID!r}; received {model_id!r}"
        )
    normalized_url = model_url.rstrip("/").lower()
    if normalized_url != "https://openrouter.ai/api/v1":
        raise RuntimeError(
            "Formal protocol requires MODEL_URL=https://openrouter.ai/api/v1; "
            f"received {model_url!r}"
        )
    if require_key and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise RuntimeError("OPENROUTER_API_KEY is not set")


def probe_model_identity(
    model_url: str,
    model_id: str = EXPECTED_MODEL_ID,
    timeout: float = 30.0,
    retries: int = 3,
) -> Dict[str, Any]:
    """Verify that the exact immutable model slug is listed by OpenRouter."""

    url = model_url.rstrip("/") + "/models"
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    errors: List[str] = []
    attempts = max(1, int(retries))
    for attempt in range(attempts):
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {key}"},
                timeout=timeout,
            )
            response.raise_for_status()
            body = response.json()
            ids = {
                str(item.get("id", ""))
                for item in (body.get("data") or [])
                if isinstance(item, dict)
            }
            return {
                "verified_openrouter_model": model_id in ids,
                "requested_model": model_id,
                "endpoint": url,
                "status_code": response.status_code,
                "catalog_model_count": len(ids),
                "attempts_used": attempt + 1,
            }
        except Exception as exc:
            errors.append(f"attempt {attempt + 1}: {type(exc).__name__}: {exc}")
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))
    return {
        "verified_openrouter_model": False,
        "requested_model": model_id,
        "endpoint": url,
        "attempts_used": attempts,
        "errors": errors,
        "error": errors[-1] if errors else "unknown model-catalog failure",
    }


def enforce_model_identity(model_url: str, model_id: str = EXPECTED_MODEL_ID) -> Dict[str, Any]:
    report = probe_model_identity(model_url, model_id)
    if not report["verified_openrouter_model"]:
        raise RuntimeError("OpenRouter model identity check failed: " + json.dumps(report))
    return report


def response_usage(response: Any, fallback_input: int, fallback_output: int) -> Dict[str, Any]:
    """Use provider-native accounting, with explicit local fallbacks."""

    raw_usage = response.raw.get("usage") if isinstance(response.raw, dict) else None
    has_provider_usage = isinstance(raw_usage, dict)
    return {
        "llm_input_tokens": int(response.input_tokens if has_provider_usage else fallback_input),
        "llm_output_tokens": int(response.output_tokens if has_provider_usage else fallback_output),
        "reasoning_tokens": int(response.reasoning_tokens or 0),
        "api_cost_usd": float(response.cost_usd or 0.0),
        "api_model_id": str(response.model_id or EXPECTED_MODEL_ID),
        "api_provider_id": str(response.provider_id or ""),
        "api_reasoning": str(response.reasoning_text or ""),
        "api_reasoning_details": response.reasoning_details,
        "api_router_metadata": (
            response.raw.get("openrouter_metadata")
            if isinstance(response.raw, dict)
            else None
        ),
        "api_request_messages": response.request_messages,
        "token_accounting_source": "openrouter_usage" if has_provider_usage else "local_fallback",
    }


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
        "api_cost_usd": sum(float(event.get("api_cost_usd", 0.0)) for event in rows),
        "latency_seconds": sum(float(event.get("latency_seconds", 0.0)) for event in rows),
    }
    output["llm_tokens_total"] = output["total_llm_input_tokens"] + output["total_llm_output_tokens"]
    output["total_cost_tokens"] = output["llm_tokens_total"] + output["retrieved_tokens_total"]
    output["total_cost"] = float(output["total_cost_tokens"])
    return output
