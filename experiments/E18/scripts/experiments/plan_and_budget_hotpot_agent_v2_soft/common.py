from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import re
import string
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import requests


LEVEL_BUDGETS = {1: 200, 2: 250, 3: 350, 4: 450, 5: 600}
OFFICIAL_TEMPERATURE = 0.1
OFFICIAL_TOP_P = 0.9

SEARCH_ACTION_RE = re.compile(r"<Search>\s*(.*?)\s*</Search>", re.I | re.S)
SUBANSWER_ACTION_RE = re.compile(
    r"<Subanswer>\s*(.*?)\s*</Subanswer>", re.I | re.S
)
FINAL_ACTION_RE = re.compile(
    r"<FinalAnswer>\s*(.*?)\s*</FinalAnswer>", re.I | re.S
)
CITE_ACTION_RE = re.compile(r"<Cite>\s*(.*?)\s*</Cite>", re.I | re.S)


def utc_now() -> str:
    import datetime

    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def parse_action(text: str) -> Dict[str, Any]:
    search = SEARCH_ACTION_RE.search(text)
    subanswer = SUBANSWER_ACTION_RE.search(text)
    final = FINAL_ACTION_RE.search(text)
    cite = CITE_ACTION_RE.search(text)
    if search and not subanswer:
        query = search.group(1).strip()
        if query:
            return {"type": "search", "query": query, "format_status": "valid"}
    if subanswer and not search:
        citations = []
        if cite:
            citations = [
                item.strip()
                for item in cite.group(1).split(",")
                if item.strip() and item.strip().upper() != "NONE"
            ]
        return {
            "type": "finish",
            "subanswer": subanswer.group(1).strip(),
            "final_answer": final.group(1).strip() if final else None,
            "citations": citations,
            "format_status": "valid",
        }
    return {"type": "invalid", "format_status": "invalid_action_format"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            rows.append(row)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(tmp, path)


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _schedule_weight(scheduler: str, i: int, n: int) -> float:
    if scheduler == "polynomial":
        return float((n - i) ** 2)
    if scheduler == "cosine":
        if n <= 1:
            return 1.0
        return 0.5 * (1.0 + math.cos(math.pi * i / (n - 1))) + 0.1
    raise ValueError(f"Unsupported scheduler: {scheduler}")


def allocate_tokens(
    credits: Sequence[int], total_tokens: int, scheduler: str
) -> List[int]:
    """Official largest-remainder allocation for Polynomial/Cosine scheduling."""
    if not credits or any(int(x) <= 0 for x in credits):
        raise ValueError("Credits must be non-empty positive integers")
    if sum(int(x) for x in credits) != 100:
        raise ValueError(f"Credits must sum to 100, got {sum(credits)}")
    n = len(credits)
    combined = [
        float(credits[i]) * _schedule_weight(scheduler, i, n) for i in range(n)
    ]
    denominator = sum(combined)
    exact = [total_tokens * x / denominator for x in combined]
    allocated = [int(math.floor(x)) for x in exact]
    remainder = total_tokens - sum(allocated)
    order = sorted(range(n), key=lambda i: (-combined[i], i))
    for i in order[:remainder]:
        allocated[i] += 1
    if sum(allocated) != total_tokens:
        raise AssertionError("Allocation does not conserve the total budget")
    return allocated


def extract_json_object(text: str) -> Dict[str, Any]:
    candidates = [text.strip()]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I))
    starts = [i for i, char in enumerate(text) if char == "{"]
    for start in starts:
        depth = 0
        in_string = False
        escaped = False
        for end in range(start, len(text)):
            char = text[end]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[start : end + 1])
                        break
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("No valid JSON object found in model output")


def _usage_dict(data: Dict[str, Any]) -> Dict[str, int]:
    usage = data.get("usage") or {}
    return {
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "total_tokens": int(usage.get("total_tokens") or 0),
    }


def chat_completion(
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    *,
    temperature: float = OFFICIAL_TEMPERATURE,
    top_p: float = OFFICIAL_TOP_P,
    timeout: int = 900,
) -> Dict[str, Any]:
    """One physical generation request. Retries are deliberately handled by callers."""
    endpoint = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": int(max_tokens),
        "stream": False,
    }
    started = time.time()
    response = requests.post(endpoint, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(
            f"HTTP {response.status_code} from {endpoint}: {response.text[:1000]}"
        )
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(f"No choices returned by {endpoint}: {str(data)[:1000]}")
    message = choices[0].get("message") or {}
    text = message.get("content")
    if text is None:
        text = choices[0].get("text", "")
    return {
        "text": str(text or ""),
        "finish_reason": choices[0].get("finish_reason"),
        "usage": _usage_dict(data),
        "model": data.get("model"),
        "latency_seconds": time.time() - started,
        "request": payload,
    }


def get_model_info(base_url: str, timeout: int = 20) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/get_model_info"
    response = requests.get(endpoint, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected model info: {data!r}")
    return data


def verify_model_path(base_url: str, required_fragment: str) -> Dict[str, Any]:
    info = get_model_info(base_url)
    path = str(info.get("model_path") or "")
    if required_fragment.lower() not in path.lower():
        raise RuntimeError(
            f"Wrong model at {base_url}: expected {required_fragment!r}, got {path!r}"
        )
    return info


def retriever_search(
    base_url: str, query: str, top_k: int = 5, timeout: int = 300
) -> Dict[str, Any]:
    endpoint = base_url.rstrip("/") + "/search"
    # FlashRAG's serving endpoint returns a two-element outer structure
    # [documents, scores] when return_score=true. This experiment does not use
    # retrieval scores, so request the stable documents-only list explicitly.
    payload = {"query": query, "top_n": int(top_k), "return_score": False}
    started = time.time()
    response = requests.post(endpoint, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise RuntimeError(
            f"HTTP {response.status_code} from {endpoint}: {response.text[:1000]}"
        )
    docs = response.json()
    if not isinstance(docs, list):
        raise RuntimeError(f"Retriever did not return a list: {str(docs)[:1000]}")
    if len(docs) != top_k:
        raise RuntimeError(f"Retriever returned {len(docs)} docs, expected {top_k}")
    normalized = []
    for rank, doc in enumerate(docs, 1):
        if not isinstance(doc, dict) or not str(doc.get("contents") or "").strip():
            raise RuntimeError(f"Malformed retriever result at rank {rank}: {doc!r}")
        contents = str(doc["contents"])
        normalized.append(
            {
                "rank": rank,
                "id": str(doc.get("id") or ""),
                "contents": contents,
                "score": doc.get("score"),
                "content_sha256": sha256_text(contents),
            }
        )
    return {
        "query": query,
        "top_k": top_k,
        "documents": normalized,
        "latency_seconds": time.time() - started,
    }


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        return "".join(char for char in value if char not in set(string.punctuation))

    return white_space_fix(remove_articles(remove_punc(text.lower())))


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    common = collections.Counter(pred_tokens) & collections.Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, gold: Any) -> Tuple[float, float]:
    answers = gold if isinstance(gold, list) else [gold]
    answers = [str(x) for x in answers if str(x).strip()]
    if not answers:
        raise ValueError("Gold answer is empty")
    return (
        max(exact_match(prediction, x) for x in answers),
        max(token_f1(prediction, x) for x in answers),
    )


def pick_id(row: Dict[str, Any]) -> str:
    for key in ("query_id", "sample_id", "question_id", "_id", "id"):
        if row.get(key) is not None:
            return str(row[key])
    raise KeyError(f"No id field in {row.keys()}")


def pick_question(row: Dict[str, Any]) -> str:
    for key in ("query", "question", "problem"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    raise KeyError(f"No question field in {row.keys()}")


def pick_answer(row: Dict[str, Any]) -> Any:
    for key in ("answer", "gold_answer", "answers", "gold_answers"):
        value = row.get(key)
        if value is not None and value != "":
            return value
    raise KeyError(f"No answer field in {row.keys()}")


def truncate_for_log(text: str, length: int = 500) -> str:
    return text if len(text) <= length else text[:length] + "..."
