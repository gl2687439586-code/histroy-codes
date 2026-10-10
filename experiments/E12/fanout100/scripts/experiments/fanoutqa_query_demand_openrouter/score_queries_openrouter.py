#!/usr/bin/env python3
"""Score FanOutQA query-only READ/SEARCH demand with OpenRouter.

Each (question, dimension) pair is one independent chat-completions request.
The default one-shot policy sends at most 200 generation requests for 100
questions. State is stored atomically so an interrupted run can resume without
repeating completed requests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
import re
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - urllib fallback keeps the script portable
    requests = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "openai/gpt-5.6-sol"
DIMENSIONS = ("read", "search")
FORBIDDEN_QUESTION_FIELDS = {
    "answer", "answers", "gold", "gold_answer", "gold_answers", "label",
    "labels", "evidence", "necessary_evidence", "supporting_facts",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_input(path: Path, expected_samples: int) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    payload = read_json(path)
    if not isinstance(payload, dict) or set(payload) != {"metadata", "questions"}:
        raise ValueError("Input must contain exactly top-level metadata and questions fields")
    metadata = payload["metadata"]
    questions = payload["questions"]
    if not isinstance(metadata, dict) or not isinstance(questions, list):
        raise ValueError("metadata must be an object and questions must be an array")
    if metadata.get("benchmark") != "FanOutQA":
        raise ValueError("metadata.benchmark must be FanOutQA")
    if metadata.get("contains_gold_answers") is not False:
        raise ValueError("metadata.contains_gold_answers must be false")
    if metadata.get("sample_seed") != 2026:
        raise ValueError("metadata.sample_seed must be 2026")
    if metadata.get("sample_size") != len(questions):
        raise ValueError("metadata.sample_size does not match questions length")
    if len(questions) != expected_samples:
        raise ValueError(f"Expected {expected_samples} questions, found {len(questions)}")

    clean: List[Dict[str, str]] = []
    ids: List[str] = []
    for index, row in enumerate(questions):
        if not isinstance(row, dict):
            raise ValueError(f"questions[{index}] is not an object")
        forbidden = sorted(set(row) & FORBIDDEN_QUESTION_FIELDS)
        if forbidden:
            raise ValueError(f"questions[{index}] contains forbidden gold fields: {forbidden}")
        if set(row) != {"question_id", "query"}:
            raise ValueError(
                f"questions[{index}] must contain exactly question_id and query; got {sorted(row)}"
            )
        question_id = row["question_id"]
        query = row["query"]
        if not isinstance(question_id, str) or not question_id.strip():
            raise ValueError(f"questions[{index}].question_id must be a non-empty string")
        if query is None:
            query = ""
        if not isinstance(query, str):
            raise ValueError(f"questions[{index}].query must be a string or null")
        ids.append(question_id)
        clean.append({"question_id": question_id, "query": query})
    if len(ids) != len(set(ids)):
        duplicates = [key for key, count in Counter(ids).items() if count > 1]
        raise ValueError(f"Duplicate question IDs: {duplicates[:10]}")
    return metadata, clean


def prompt_path(dimension: str) -> Path:
    return SCRIPT_DIR / "prompts" / f"{dimension}_system_prompt.txt"


def load_prompt(dimension: str) -> str:
    text = prompt_path(dimension).read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Empty {dimension} prompt")
    return text


def result_schema(dimension: str, question: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    score_field = f"{dimension}_score"
    question_id_schema: Dict[str, Any] = {"type": "string"}
    query_schema: Dict[str, Any] = {"type": "string"}
    if question is not None:
        question_id_schema["enum"] = [question["question_id"]]
        # Do not place arbitrary query text in a strict-schema enum. OpenAI's
        # structured-output schema rejects escaped quote characters in string
        # literals. Exact query identity is enforced after generation by
        # validate_rating(), which compares the returned value byte-for-byte.
    return {
        "name": f"fanoutqa_query_{dimension}_demand",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "question_id": question_id_schema,
                "query": query_schema,
                score_field: {"type": ["integer", "null"], "minimum": 0, "maximum": 4},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "basis": {"type": "string", "minLength": 1, "maxLength": 200},
            },
            "required": ["question_id", "query", score_field, "confidence", "basis"],
        },
    }


def request_payload(
    model: str,
    dimension: str,
    question: Dict[str, str],
    prompt: str,
    max_tokens: int,
    reasoning_effort: str,
    seed: int,
) -> Dict[str, Any]:
    record = {"question_id": question["question_id"], "query": question["query"]}
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": "Evaluate this single record and return only the required JSON object:\n"
                + json.dumps(record, ensure_ascii=False),
            },
        ],
        "max_tokens": max_tokens,
        "seed": seed,
        "reasoning": {"effort": reasoning_effort, "exclude": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": result_schema(dimension, question),
        },
        "provider": {"require_parameters": True, "allow_fallbacks": False},
    }


def http_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    body: Optional[Dict[str, Any]],
    timeout: float,
) -> Tuple[int, Dict[str, Any]]:
    if requests is not None:
        try:
            response = requests.request(
                method,
                url,
                headers=headers,
                json=body,
                timeout=timeout,
            )
        except requests.exceptions.SSLError as exc:
            raise RuntimeError(
                f"TLS request to {url} failed through the current network/proxy: {exc}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"Request to {url} failed: {exc}") from exc
        raw = response.content
        if response.status_code >= 400:
            detail = raw.decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {response.status_code} from {url}: {detail[:2000]}"
            )
        try:
            parsed = response.json()
        except ValueError as exc:
            raise RuntimeError(
                f"HTTP {response.status_code} returned non-JSON: "
                f"{raw[:1000].decode('utf-8', errors='replace')}"
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(
                f"HTTP {response.status_code} returned a non-object JSON response"
            )
        return response.status_code, parsed

    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        detail = raw.decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail[:2000]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Request to {url} failed: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"HTTP {status} returned non-JSON: {raw[:1000].decode('utf-8', errors='replace')}"
        ) from exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"HTTP {status} returned a non-object JSON response")
    return status, parsed


def model_preflight(base_url: str, api_key: str, model: str, timeout: float) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}
    _, payload = http_json("GET", f"{base_url.rstrip('/')}/models", headers, None, timeout)
    models = payload.get("data")
    if not isinstance(models, list):
        raise RuntimeError("OpenRouter model catalogue response lacks data[]")
    matches = [item for item in models if item.get("id") == model]
    if len(matches) != 1:
        raise RuntimeError(f"Exact OpenRouter model ID is unavailable: {model}")
    item = matches[0]
    supported = set(item.get("supported_parameters") or [])
    required = {"max_tokens", "reasoning", "response_format", "seed", "structured_outputs"}
    missing = sorted(required - supported)
    if missing:
        raise RuntimeError(f"Model {model} lacks required parameters: {missing}")
    return {
        "id": item.get("id"),
        "canonical_slug": item.get("canonical_slug"),
        "name": item.get("name"),
        "context_length": item.get("context_length"),
        "max_completion_tokens": (item.get("top_provider") or {}).get("max_completion_tokens"),
        "supported_parameters": sorted(supported),
        "pricing": item.get("pricing") or {},
        "reasoning": item.get("reasoning") or {},
    }


def content_from_response(response: Dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise ValueError("Response must contain exactly one choice")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        pieces = []
        for part in content:
            if isinstance(part, dict) and isinstance(part.get("text"), str):
                pieces.append(part["text"])
        return "".join(pieces).strip()
    raise ValueError("Response message has no string content")


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE))


def validate_rating(
    value: Any, dimension: str, question: Dict[str, str]
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("Rating is not a JSON object")
    score_field = f"{dimension}_score"
    expected_keys = {"question_id", "query", score_field, "confidence", "basis"}
    if set(value) != expected_keys:
        raise ValueError(f"Unexpected result keys: {sorted(set(value) ^ expected_keys)}")
    if value["question_id"] != question["question_id"]:
        raise ValueError("Model changed question_id")
    if value["query"] != question["query"]:
        raise ValueError("Model changed query text")
    score = value[score_field]
    blank = not question["query"].strip()
    if blank:
        if score is not None or value["confidence"] != 0:
            raise ValueError("Blank query must receive null score and confidence 0")
    else:
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
            raise ValueError(f"{score_field} must be an integer 0..4")
    confidence = value["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("confidence must be numeric")
    if not 0 <= float(confidence) <= 1:
        raise ValueError("confidence must be in [0,1]")
    basis = value["basis"]
    if not isinstance(basis, str) or not basis.strip():
        raise ValueError("basis must be a non-empty string")
    words = count_words(basis)
    if words >= 25:
        raise ValueError(f"basis must be under 25 words; found {words}")
    return {
        "question_id": value["question_id"],
        "query": value["query"],
        score_field: score,
        "confidence": float(confidence),
        "basis": basis,
    }


def state_path(output_dir: Path, index: int, question_id: str, dimension: str) -> Path:
    key = sha256_bytes(question_id.encode("utf-8"))[:12]
    return output_dir / "state" / dimension / f"{index:04d}_{key}.json"


def task_key(index: int, dimension: str) -> str:
    return f"{index:04d}:{dimension}"


def response_model_allowed(response_model: Any, requested: str, catalogue: Dict[str, Any]) -> bool:
    return response_model in {requested, catalogue.get("canonical_slug")}


def call_task(
    *,
    index: int,
    dimension: str,
    question: Dict[str, str],
    args: argparse.Namespace,
    api_key: str,
    prompt: str,
    catalogue: Dict[str, Any],
) -> Dict[str, Any]:
    payload = request_payload(
        args.model, dimension, question, prompt, args.max_tokens,
        args.reasoning_effort, args.seed,
    )
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "HTTP-Referer": args.http_referer,
        "X-OpenRouter-Title": args.app_title,
    }
    attempts = []
    started = time.time()
    last_error = None
    for attempt_number in range(1, args.max_attempts + 1):
        attempt_started = time.time()
        try:
            status, response = http_json(
                "POST",
                f"{args.base_url.rstrip('/')}/chat/completions",
                headers,
                payload,
                args.timeout,
            )
            content = content_from_response(response)
            parsed = json.loads(content)
            validated = validate_rating(parsed, dimension, question)
            response_model = response.get("model")
            if not response_model_allowed(response_model, args.model, catalogue):
                raise ValueError(
                    f"Response model {response_model!r} differs from locked model {args.model!r}"
                )
            attempts.append({
                "attempt": attempt_number,
                "ok": True,
                "http_status": status,
                "elapsed_seconds": time.time() - attempt_started,
            })
            return {
                "schema_version": "fanoutqa-query-demand-openrouter-v1",
                "task_key": task_key(index, dimension),
                "question_index": index,
                "dimension": dimension,
                "question_id": question["question_id"],
                "query": question["query"],
                "requested_model": args.model,
                "response_model": response_model,
                "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
                "request_sha256": sha256_bytes(canonical_json(payload).encode("utf-8")),
                "created_at": utc_now(),
                "elapsed_seconds": time.time() - started,
                "physical_generation_attempts": len(attempts),
                "attempts": attempts,
                "rating": validated,
                "usage": response.get("usage") or {},
                "provider": response.get("provider"),
                "response_id": response.get("id"),
                "raw_request": payload,
                "raw_response": response,
            }
        except Exception as exc:  # keep state for a resumable scientific run
            last_error = f"{type(exc).__name__}: {exc}"
            attempts.append({
                "attempt": attempt_number,
                "ok": False,
                "elapsed_seconds": time.time() - attempt_started,
                "error": last_error,
            })
            if attempt_number < args.max_attempts:
                time.sleep(args.retry_backoff * (2 ** (attempt_number - 1)))
    return {
        "schema_version": "fanoutqa-query-demand-openrouter-v1-failure",
        "task_key": task_key(index, dimension),
        "question_index": index,
        "dimension": dimension,
        "question_id": question["question_id"],
        "query": question["query"],
        "requested_model": args.model,
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "request_sha256": sha256_bytes(canonical_json(payload).encode("utf-8")),
        "created_at": utc_now(),
        "elapsed_seconds": time.time() - started,
        "physical_generation_attempts": len(attempts),
        "attempts": attempts,
        "error": last_error,
        "raw_request": payload,
    }


def load_completed_state(path: Path, index: int, dimension: str, question: Dict[str, str]) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        state = read_json(path)
        if state.get("task_key") != task_key(index, dimension):
            return None
        rating = validate_rating(state.get("rating"), dimension, question)
        state["rating"] = rating
        return state
    except Exception:
        return None


def collect_states(
    output_dir: Path, questions: Sequence[Dict[str, str]]
) -> Tuple[List[Dict[str, Any]], List[Tuple[int, str, Dict[str, str]]]]:
    states = []
    missing = []
    for index, question in enumerate(questions):
        for dimension in DIMENSIONS:
            path = state_path(output_dir, index, question["question_id"], dimension)
            state = load_completed_state(path, index, dimension, question)
            if state is None:
                missing.append((index, dimension, question))
            else:
                states.append(state)
    states.sort(key=lambda row: (row["question_index"], DIMENSIONS.index(row["dimension"])))
    return states, missing


def usage_totals(states: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    totals: Dict[str, float] = defaultdict(float)
    by_dimension: Dict[str, Dict[str, float]] = {
        dimension: defaultdict(float) for dimension in DIMENSIONS
    }
    numeric_fields = (
        "prompt_tokens", "completion_tokens", "total_tokens", "cost",
        "cost_details.upstream_inference_cost",
    )
    for state in states:
        usage = state.get("usage") or {}
        flat = dict(usage)
        cost_details = usage.get("cost_details") or {}
        flat["cost_details.upstream_inference_cost"] = cost_details.get("upstream_inference_cost")
        for field in numeric_fields:
            value = flat.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[field] += value
                by_dimension[state["dimension"]][field] += value
    return {
        "all": dict(totals),
        "by_dimension": {key: dict(value) for key, value in by_dimension.items()},
    }


def materialize_outputs(
    output_dir: Path,
    metadata: Dict[str, Any],
    questions: Sequence[Dict[str, str]],
    catalogue: Dict[str, Any],
    args: argparse.Namespace,
    input_sha256: str,
    current_failures: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    states, missing = collect_states(output_dir, questions)
    by_key = {(row["question_index"], row["dimension"]): row for row in states}

    for dimension in DIMENSIONS:
        rows = []
        for index, question in enumerate(questions):
            state = by_key.get((index, dimension))
            if state:
                rows.append(state["rating"])
        write_jsonl(output_dir / "results" / f"{dimension}_scores.jsonl", rows)

    combined = []
    for index, question in enumerate(questions):
        read_state = by_key.get((index, "read"))
        search_state = by_key.get((index, "search"))
        if not read_state or not search_state:
            continue
        read_rating = read_state["rating"]
        search_rating = search_state["rating"]
        combined.append({
            "question_id": question["question_id"],
            "query": question["query"],
            "read_score": read_rating["read_score"],
            "read_confidence": read_rating["confidence"],
            "read_basis": read_rating["basis"],
            "search_score": search_rating["search_score"],
            "search_confidence": search_rating["confidence"],
            "search_basis": search_rating["basis"],
        })
    write_jsonl(output_dir / "results" / "combined_scores.jsonl", combined)

    raw_rows = []
    for state in states:
        raw_rows.append({
            key: value for key, value in state.items()
            if key != "rating"
        })
    write_jsonl(output_dir / "raw" / "successful_responses.jsonl", raw_rows)
    failure_path = output_dir / "failures.jsonl"
    # Validation-only and offline materialization must never erase diagnostics
    # produced by a prior paid invocation. A real invocation passes a list
    # (possibly empty) and intentionally refreshes the current failure file.
    if current_failures is not None:
        write_jsonl(failure_path, current_failures)
    elif not failure_path.exists():
        write_jsonl(failure_path, [])

    returned_models = sorted({row.get("response_model") for row in states})
    validation_errors = []
    expected = len(questions) * len(DIMENSIONS)
    if len(states) != expected:
        validation_errors.append(f"Completed {len(states)} of {expected} logical ratings")
    if len(combined) != len(questions):
        validation_errors.append(f"Combined output has {len(combined)} of {len(questions)} questions")
    allowed_models = {args.model, catalogue.get("canonical_slug")}
    if any(model not in allowed_models for model in returned_models):
        validation_errors.append(f"Unexpected returned models: {returned_models}")
    null_errors = []
    for state in states:
        score = state["rating"][f"{state['dimension']}_score"]
        if (not state["query"].strip()) != (score is None):
            null_errors.append(state["task_key"])
    if null_errors:
        validation_errors.append(f"Invalid null-score policy in {len(null_errors)} tasks")

    prior_attempts = 0
    attempt_log = output_dir / "api_attempts.jsonl"
    if attempt_log.is_file():
        with attempt_log.open("r", encoding="utf-8") as handle:
            prior_attempts = sum(1 for line in handle if line.strip())
    strict_request_count_passed = len(states) == expected and prior_attempts == expected
    if len(states) == expected and prior_attempts != expected:
        validation_errors.append(
            f"Completed all ratings but logged {prior_attempts} physical generation requests; "
            f"the strict target is {expected}"
        )

    report = {
        "created_at": utc_now(),
        "passed": not validation_errors,
        "validation_errors": validation_errors,
        "input": str(args.input.resolve()),
        "input_sha256": input_sha256,
        "metadata": metadata,
        "question_count": len(questions),
        "expected_logical_ratings": expected,
        "completed_logical_ratings": len(states),
        "complete_question_pairs": len(combined),
        "missing_tasks": [task_key(index, dimension) for index, dimension, _ in missing],
        "generation_requests_logged_across_runs": prior_attempts,
        "strict_physical_generation_request_target": expected,
        "strict_physical_generation_request_count_passed": strict_request_count_passed,
        "requested_model": args.model,
        "returned_models": returned_models,
        "model_catalogue": catalogue,
        "one_query_per_request": True,
        "dimensions_per_question": list(DIMENSIONS),
        "max_attempts_per_task": args.max_attempts,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "usage": usage_totals(states),
        "output_files": {
            "read": str(output_dir / "results" / "read_scores.jsonl"),
            "search": str(output_dir / "results" / "search_scores.jsonl"),
            "combined": str(output_dir / "results" / "combined_scores.jsonl"),
            "raw": str(output_dir / "raw" / "successful_responses.jsonl"),
        },
    }
    atomic_json(output_dir / "validation_report.json", report)
    atomic_json(output_dir / "manifest.json", report)
    return report


def safe_append_jsonl(path: Path, row: Dict[str, Any], lock: threading.Lock) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with lock:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def estimated_plan(
    questions: Sequence[Dict[str, str]], prompts: Dict[str, str], catalogue: Dict[str, Any]
) -> Dict[str, Any]:
    prompt_chars = 0
    for question in questions:
        user = json.dumps(question, ensure_ascii=False)
        for dimension in DIMENSIONS:
            prompt_chars += len(prompts[dimension]) + len(user) + 80
    approximate_prompt_tokens = int(round(prompt_chars / 4.0))
    pricing = catalogue.get("pricing") or {}
    try:
        prompt_price = float(pricing.get("prompt", 0))
        completion_price = float(pricing.get("completion", 0))
    except (TypeError, ValueError):
        prompt_price = completion_price = 0.0
    completion_low = len(questions) * len(DIMENSIONS) * 50
    completion_high = len(questions) * len(DIMENSIONS) * 300
    return {
        "questions": len(questions),
        "logical_generation_requests": len(questions) * len(DIMENSIONS),
        "requests_per_question": 2,
        "read_requests": len(questions),
        "search_requests": len(questions),
        "approximate_prompt_tokens": approximate_prompt_tokens,
        "completion_token_scenario": {"low": completion_low, "high": completion_high},
        "estimated_cost_usd": {
            "low": approximate_prompt_tokens * prompt_price + completion_low * completion_price,
            "high": approximate_prompt_tokens * prompt_price + completion_high * completion_price,
            "note": "Approximation only; final manifest sums OpenRouter-reported usage/cost.",
        },
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--input", required=True, type=Path)
    value.add_argument("--output-dir", required=True, type=Path)
    value.add_argument("--model", default=DEFAULT_MODEL)
    value.add_argument("--base-url", default=os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    value.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    value.add_argument("--workers", type=int, default=4)
    value.add_argument("--timeout", type=float, default=180.0)
    value.add_argument("--max-attempts", type=int, default=1)
    value.add_argument("--retry-backoff", type=float, default=2.0)
    value.add_argument("--max-tokens", type=int, default=2048)
    value.add_argument("--reasoning-effort", choices=("none", "low", "medium", "high", "xhigh", "max"), default="medium")
    value.add_argument("--seed", type=int, default=2026)
    value.add_argument("--expected-samples", type=int, default=100)
    value.add_argument("--http-referer", default="https://github.com/SmartSearch")
    value.add_argument("--app-title", default="FanOutQA Query READ-SEARCH Demand Scoring")
    value.add_argument("--plan", action="store_true")
    value.add_argument("--preflight-only", action="store_true")
    value.add_argument("--validate-only", action="store_true")
    return value


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parser().parse_args(argv)
    if args.workers < 1 or args.max_attempts < 1 or args.max_tokens < 1:
        raise ValueError("workers, max-attempts, and max-tokens must be positive")
    metadata, questions = load_input(args.input, args.expected_samples)
    input_hash = sha256_file(args.input)
    prompts = {dimension: load_prompt(dimension) for dimension in DIMENSIONS}

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key and not args.validate_only:
        raise RuntimeError(f"Missing API key environment variable: {args.api_key_env}")

    existing_config_path = args.output_dir / "run_config.json"
    config_signature = {
        "schema_version": "fanoutqa-query-demand-openrouter-v1",
        "input_sha256": input_hash,
        "expected_samples": args.expected_samples,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "seed": args.seed,
        "max_tokens": args.max_tokens,
        "http_transport": "requests" if requests is not None else "urllib",
        "prompt_sha256": {
            dimension: sha256_bytes(prompts[dimension].encode("utf-8"))
            for dimension in DIMENSIONS
        },
    }
    if existing_config_path.is_file():
        existing = read_json(existing_config_path)
        if existing != config_signature:
            raise RuntimeError(
                "Output directory contains a different run_config.json; use a new directory"
            )
    else:
        atomic_json(existing_config_path, config_signature)

    if args.validate_only:
        catalogue_path = args.output_dir / "model_catalogue.json"
        if not catalogue_path.is_file():
            raise RuntimeError("Missing model_catalogue.json; this run has not passed preflight")
        catalogue = read_json(catalogue_path)
        report = materialize_outputs(
            args.output_dir, metadata, questions, catalogue, args, input_hash
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["passed"] else 1

    catalogue = model_preflight(args.base_url, api_key, args.model, args.timeout)
    atomic_json(args.output_dir / "model_catalogue.json", catalogue)
    plan = estimated_plan(questions, prompts, catalogue)
    atomic_json(args.output_dir / "plan.json", plan)
    print(json.dumps({"model_preflight": catalogue, "plan": plan}, ensure_ascii=False, indent=2))
    if args.plan or args.preflight_only:
        return 0

    states, pending = collect_states(args.output_dir, questions)
    expected = len(questions) * len(DIMENSIONS)
    print(
        f"Loaded {len(questions)} FanOutQA queries; complete={len(states)}, "
        f"pending={len(pending)}, expected={expected}."
    )
    print(
        f"Locked model={args.model}; one-shot max_attempts={args.max_attempts}; "
        f"workers={args.workers}."
    )
    if args.max_attempts != 1:
        print("WARNING: max-attempts > 1 can make physical generation requests exceed 200.")

    log_lock = threading.Lock()
    failure_rows: List[Dict[str, Any]] = []
    completed_this_run = 0
    request_attempts_this_run = 0
    started = time.time()

    def submit(executor: concurrent.futures.Executor, item: Tuple[int, str, Dict[str, str]]):
        index, dimension, question = item
        return executor.submit(
            call_task,
            index=index,
            dimension=dimension,
            question=question,
            args=args,
            api_key=api_key,
            prompt=prompts[dimension],
            catalogue=catalogue,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {submit(executor, item): item for item in pending}
        for future in concurrent.futures.as_completed(future_map):
            index, dimension, question = future_map[future]
            try:
                state = future.result()
            except Exception as exc:
                state = {
                    "task_key": task_key(index, dimension),
                    "question_index": index,
                    "dimension": dimension,
                    "question_id": question["question_id"],
                    "query": question["query"],
                    "physical_generation_attempts": 0,
                    "error": f"WorkerCrash: {type(exc).__name__}: {exc}",
                    "created_at": utc_now(),
                }
            attempts = int(state.get("physical_generation_attempts", 0))
            request_attempts_this_run += attempts
            for attempt in state.get("attempts", []):
                safe_append_jsonl(
                    args.output_dir / "api_attempts.jsonl",
                    {
                        "created_at": state.get("created_at"),
                        "task_key": state.get("task_key"),
                        "attempt": attempt.get("attempt"),
                        "ok": attempt.get("ok"),
                        "http_status": attempt.get("http_status"),
                        "elapsed_seconds": attempt.get("elapsed_seconds"),
                        "error": attempt.get("error"),
                    },
                    log_lock,
                )
            if "rating" in state:
                atomic_json(
                    state_path(args.output_dir, index, question["question_id"], dimension),
                    state,
                )
                completed_this_run += 1
                score = state["rating"][f"{dimension}_score"]
                usage = state.get("usage") or {}
                status = f"score={score} tokens={usage.get('total_tokens', '?')}"
            else:
                failure_rows.append(state)
                status = f"FAILED {state.get('error')}"
            done_now = completed_this_run + len(failure_rows)
            elapsed = max(time.time() - started, 1e-9)
            eta = elapsed / done_now * max(len(pending) - done_now, 0)
            print(
                f"[{done_now}/{len(pending)}] {dimension.upper()} "
                f"{question['question_id']} {status} eta={eta / 60:.1f}min",
                flush=True,
            )

    report = materialize_outputs(
        args.output_dir, metadata, questions, catalogue, args, input_hash, failure_rows
    )
    report["this_invocation"] = {
        "started_with_pending_tasks": len(pending),
        "successful_tasks": completed_this_run,
        "failed_tasks": len(failure_rows),
        "physical_generation_attempts": request_attempts_this_run,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(args.output_dir / "manifest.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
