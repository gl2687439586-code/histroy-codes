"""Shared utilities for the TALE-EP HotpotQA reproduction."""

from __future__ import annotations

import collections
import datetime
import hashlib
import json
import os
import re
import string
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import requests


SCHEMA_VERSION = "tale-ep-hotpot500-qwen3-original-prompt-agent-v3.0"
METHOD = "tale_ep"
PROTOCOL_VARIANT = "original_tale_ep_prompts_agent_interface_v3"
TEMPERATURE = 0.1
TOP_P = 1.0
ROLLOUTS = int(os.environ.get("TALE_EP_ROLLOUTS", "1"))
MAX_SEARCHES = 4
TOP_K = 5

BUDGET_RE = re.compile(r"\[\[\s*([0-9][0-9,]*)\s*\]\]")
SEARCH_RE = re.compile(r"<Search>\s*(.*?)\s*</Search>", re.I | re.S)
ANSWER_RE = re.compile(r"<Answer>\s*(.*?)\s*</Answer>", re.I | re.S)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return sha256_text(payload)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def append_jsonl(path: str | Path, row: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def new_request_id() -> str:
    return uuid.uuid4().hex


def parse_budget(text: str) -> int:
    matches = BUDGET_RE.findall(str(text or ""))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one TALE budget in [[integer]] format; found {len(matches)}"
        )
    value = int(matches[0].replace(",", ""))
    if value <= 0:
        raise ValueError(f"TALE budget must be positive, got {value}")
    return value


def parse_action(text: str) -> Dict[str, Any]:
    search = SEARCH_RE.search(str(text or ""))
    answer = ANSWER_RE.search(str(text or ""))
    if search and not answer and search.group(1).strip():
        return {
            "type": "search",
            "query": search.group(1).strip(),
            "format_status": "valid",
        }
    if answer and not search:
        return {
            "type": "answer",
            "answer": answer.group(1).strip(),
            "format_status": "valid",
        }
    return {"type": "invalid", "format_status": "invalid_action_format"}


def _usage(data: Dict[str, Any]) -> Dict[str, int]:
    usage = data.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def chat_completion(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
    timeout: int = 1200,
    request_ledger_path: str | Path | None = None,
    request_context: Dict[str, Any] | None = None,
    request_id: str | None = None,
) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    request_id = request_id or new_request_id()
    context = dict(request_context or {})
    if request_ledger_path is not None:
        append_jsonl(
            request_ledger_path,
            {
                "event": "request_started",
                "created_at": utc_now(),
                "request_id": request_id,
                "endpoint": endpoint,
                "model": model,
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_tokens": int(max_tokens),
                "message_count": len(messages),
                "messages_sha256": canonical_hash(messages),
                "context": context,
            },
        )
    session = requests.Session()
    session.trust_env = False
    started = time.time()
    response = None
    try:
        response = session.post(endpoint, json=payload, timeout=timeout)
        latency = time.time() - started
        if response.status_code >= 400:
            if request_ledger_path is not None:
                append_jsonl(
                    request_ledger_path,
                    {
                        "event": "request_finished",
                        "created_at": utc_now(),
                        "request_id": request_id,
                        "outcome": "http_error",
                        "http_status": response.status_code,
                        "latency_seconds": latency,
                        "usage_available": False,
                        "response_preview": response.text[:4000],
                        "context": context,
                    },
                )
            raise RuntimeError(
                f"HTTP {response.status_code} from {endpoint}: {response.text[:1000]}"
            )
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"No choices returned: {str(data)[:1000]}")
        message = choices[0].get("message") or {}
        text = message.get("content")
        if text is None:
            text = choices[0].get("text", "")
        text = str(text or "")
        usage = _usage(data)
        result = {
            "request_id": request_id,
            "text": text,
            "finish_reason": choices[0].get("finish_reason"),
            "usage": usage,
            "model": data.get("model"),
            "latency_seconds": latency,
            "request": payload,
        }
        if request_ledger_path is not None:
            append_jsonl(
                request_ledger_path,
                {
                    "event": "request_finished",
                    "created_at": utc_now(),
                    "request_id": request_id,
                    "outcome": "success",
                    "http_status": response.status_code,
                    "latency_seconds": latency,
                    "usage_available": True,
                    "usage": usage,
                    "finish_reason": result["finish_reason"],
                    "returned_model": result["model"],
                    "raw_output": text,
                    "raw_output_sha256": sha256_text(text),
                    "context": context,
                },
            )
        return result
    except Exception as exc:
        # HTTP errors are recorded in their branch above. All other failures
        # receive a terminal ledger event so a crashed/failed request is never
        # silently omitted from the physical-call denominator.
        if request_ledger_path is not None and not (
            response is not None and response.status_code >= 400
        ):
            append_jsonl(
                request_ledger_path,
                {
                    "event": "request_finished",
                    "created_at": utc_now(),
                    "request_id": request_id,
                    "outcome": (
                        "transport_error" if response is None else "response_error"
                    ),
                    "http_status": (
                        response.status_code if response is not None else None
                    ),
                    "latency_seconds": time.time() - started,
                    "usage_available": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[:4000],
                    "context": context,
                },
            )
        raise


def request_attempts(paths: Sequence[str | Path]) -> List[Dict[str, Any]]:
    """Reconstruct one physical attempt per request_id from event ledgers."""
    attempts: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        source = Path(path)
        if not source.is_file():
            continue
        for event in read_jsonl(source):
            request_id = str(event.get("request_id") or "")
            if not request_id:
                raise RuntimeError(f"Ledger event without request_id: {source}")
            row = attempts.setdefault(
                request_id,
                {"request_id": request_id, "started": None, "finished": None, "consumer": []},
            )
            kind = event.get("event")
            if kind == "request_started":
                if row["started"] is not None:
                    raise RuntimeError(f"Duplicate request_started: {request_id}")
                row["started"] = event
            elif kind == "request_finished":
                if row["finished"] is not None:
                    raise RuntimeError(f"Duplicate request_finished: {request_id}")
                row["finished"] = event
            elif kind == "consumer_result":
                row["consumer"].append(event)
            else:
                raise RuntimeError(f"Unknown ledger event {kind!r}: {source}")
    return sorted(
        attempts.values(),
        key=lambda row: (
            str((row.get("started") or row.get("finished") or {}).get("created_at") or ""),
            row["request_id"],
        ),
    )


def summarize_request_attempts(attempts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    outcome_counts: Dict[str, int] = collections.Counter()
    consumer_outcomes: Dict[str, int] = collections.Counter()
    orphan_started = orphan_finished = usage_unknown = 0
    for attempt in attempts:
        started = attempt.get("started")
        finished = attempt.get("finished")
        if started is None:
            orphan_finished += 1
        if finished is None:
            orphan_started += 1
            usage_unknown += 1
        else:
            outcome_counts[str(finished.get("outcome") or "unknown")] += 1
            if finished.get("usage_available"):
                row_usage = finished.get("usage") or {}
                for key in usage:
                    usage[key] += int(row_usage.get(key) or 0)
            else:
                usage_unknown += 1
        for event in attempt.get("consumer") or []:
            consumer_outcomes[str(event.get("outcome") or "unknown")] += 1
    return {
        "physical_requests": len(attempts),
        "finished_requests": sum(1 for row in attempts if row.get("finished") is not None),
        "orphan_started_requests": orphan_started,
        "orphan_finished_requests": orphan_finished,
        "usage_unknown_requests": usage_unknown,
        "usage_lower_bound": usage,
        "request_outcomes": dict(sorted(outcome_counts.items())),
        "consumer_outcomes": dict(sorted(consumer_outcomes.items())),
        "cost_ledger_complete": orphan_started == 0 and orphan_finished == 0,
    }


def model_info(base_url: str) -> Dict[str, Any]:
    session = requests.Session()
    session.trust_env = False
    response = session.get(base_url.rstrip("/") + "/get_model_info", timeout=30)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("Model info endpoint did not return an object")
    return value


def verify_qwen3(base_url: str, expected_path: str) -> Dict[str, Any]:
    info = model_info(base_url)
    actual = str(info.get("model_path") or "")
    if "Qwen3-8B" not in actual or Path(actual).name != Path(expected_path).name:
        raise RuntimeError(f"Endpoint is not expected Qwen3-8B: {info}")
    return info


def retriever_search(
    base_url: str, query: str, top_k: int = TOP_K, timeout: int = 600
) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/search"
    payload = {"query": query, "top_n": int(top_k), "return_score": False}
    session = requests.Session()
    session.trust_env = False
    started = time.time()
    response = session.post(endpoint, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(
            f"HTTP {response.status_code} from {endpoint}: {response.text[:1000]}"
        )
    documents = response.json()
    if not isinstance(documents, list) or len(documents) != top_k:
        raise RuntimeError(
            f"Retriever returned {len(documents) if isinstance(documents, list) else type(documents)}; "
            f"expected Top-{top_k}"
        )
    normalized = []
    for rank, document in enumerate(documents, 1):
        if not isinstance(document, dict) or not str(document.get("contents") or "").strip():
            raise RuntimeError(f"Malformed document at rank {rank}: {document!r}")
        contents = str(document["contents"])
        normalized.append(
            {
                "rank": rank,
                "id": str(document.get("id") or ""),
                "contents": contents,
                "contents_sha256": sha256_text(contents),
            }
        )
    return {
        "query": query,
        "top_k": top_k,
        "documents": normalized,
        "latency_seconds": time.time() - started,
    }


def normalize_answer(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def score_prediction(prediction: str, gold_answers: Sequence[str]) -> Dict[str, float]:
    predicted = normalize_answer(prediction)
    golds = [normalize_answer(item) for item in gold_answers] or [""]
    em = float(any(predicted == gold for gold in golds))
    acc = float(any(gold and gold in predicted for gold in golds))
    f1_values = []
    for gold in golds:
        pred_tokens, gold_tokens = predicted.split(), gold.split()
        if not pred_tokens or not gold_tokens:
            f1_values.append(float(pred_tokens == gold_tokens))
            continue
        common = sum((collections.Counter(pred_tokens) & collections.Counter(gold_tokens)).values())
        if common == 0:
            f1_values.append(0.0)
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1_values.append(2 * precision * recall / (precision + recall))
    return {"em": em, "f1": max(f1_values), "acc": acc}


def sum_usage(rows: Iterable[Dict[str, Any]]) -> Dict[str, int]:
    result = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for row in rows:
        usage = row.get("usage") or {}
        for key in result:
            result[key] += int(usage.get(key) or 0)
    return result
