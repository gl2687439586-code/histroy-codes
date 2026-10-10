#!/usr/bin/env python3
"""Shared infrastructure for the Qwen3 Answer-only evidence experiment."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROTOCOL_VERSION = "qwen3-blank-answer-only-v5"
EXPECTED_MODEL_BASENAME = "qwen3-8b"


def normalize_answer(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    best = 0.0
    for gold in gold_answers:
        gold_tokens = normalize_answer(gold).split()
        if not prediction_tokens or not gold_tokens:
            continue
        overlap = Counter(prediction_tokens) & Counter(gold_tokens)
        common = sum(overlap.values())
        if not common:
            continue
        precision = common / len(prediction_tokens)
        recall = common / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def answer_metrics(prediction: str, gold_answers: Sequence[str]) -> Dict[str, float]:
    golds = [str(item) for item in gold_answers] or [""]
    prediction_normalized = normalize_answer(prediction)
    normalized_golds = [normalize_answer(item) for item in golds]
    exact = float(any(prediction_normalized == gold for gold in normalized_golds))
    contains = float(any(gold and gold in prediction_normalized for gold in normalized_golds))
    bidirectional = float(any(
        gold and prediction_normalized
        and (gold in prediction_normalized or prediction_normalized in gold)
        for gold in normalized_golds
    ))
    return {
        "em": exact, "f1": token_f1(prediction, golds), "acc": contains,
        "normalized_exact": exact, "normalized_contains": bidirectional,
    }


def extract_last_tag(text: str, tag: str) -> str:
    matches = re.findall(
        rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.I | re.S
    )
    return matches[-1].strip() if matches else ""


def extract_boxed(text: str) -> str:
    source = text or ""
    start = source.rfind("\\boxed{")
    if start < 0:
        return ""
    position, depth, characters = start + len("\\boxed{"), 1, []
    while position < len(source):
        character = source[position]
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return "".join(characters).strip()
        characters.append(character)
        position += 1
    return ""


def extract_final_answer(text: str) -> str:
    block = extract_last_tag(text, "answer")
    boxed = extract_boxed(block or text)
    value = boxed or block or str(text or "")
    value = re.sub(
        r"^\s*(?:the\s+final\s+answer\s+is|final\s+answer\s*:|answer\s*:?)\s*",
        "", value, flags=re.I,
    )
    return value.strip().strip("$ ")


def extract_think_text(text: str) -> str:
    return "\n".join(
        part.strip() for part in re.findall(r"<think>(.*?)</think>", text or "", re.I | re.S)
    )


def public_model_meta(raw: Any) -> Any:
    item = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(item, dict):
        return item
    return {key: value for key, value in item.items() if key != "text"}


@dataclass
class ModelResponse:
    text: str
    raw: Any
    endpoint: str
    latency_seconds: float
    finish_reason: Any


class ModelClient:
    """SGLang client compatible with version 0.4.6 on the new host.

    That native /generate endpoint rejects both seed field spellings. Formal
    runs use temperature=0 by default, so omitting the unsupported field
    remains deterministic. The planned seed is retained in result metadata.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_mode: str = "sglang",
        timeout: float = 300.0,
        retries: int = 3,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_mode = api_mode
        self.timeout = timeout
        self.retries = retries

    def _post(self, url: str, payload: Dict[str, Any]) -> Tuple[Any, float]:
        import requests

        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            started = time.time()
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                if not response.ok:
                    body = response.text.strip()[:2000]
                    raise RuntimeError(
                        f"HTTP {response.status_code} {response.reason}"
                        + (f": {body}" if body else "")
                    )
                return response.json(), time.time() - started
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Request failed after {self.retries} tries: {url}: {last_error}")

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        stop: Sequence[str],
        seed: int,
    ) -> ModelResponse:
        del seed  # unsupported by this SGLang native endpoint
        if self.api_mode != "sglang":
            raise ValueError("This experiment intentionally supports only local SGLang")
        endpoint = self.base_url + "/generate"
        raw, latency = self._post(
            endpoint,
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_new_tokens,
                    "stop": list(stop),
                },
            },
        )
        item = raw[0] if isinstance(raw, list) else raw
        text = str(item.get("text", ""))
        finish = (item.get("meta_info") or {}).get("finish_reason", {})
        matched = finish.get("matched") if isinstance(finish, dict) else None
        if isinstance(matched, str) and matched in stop and not text.endswith(matched):
            text += matched
        return ModelResponse(text, raw, endpoint, latency, finish)


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(text: Any) -> str:
    return hashlib.sha256(str(text or "").encode("utf-8")).hexdigest()


def stable_hash(text: Any, length: int = 16) -> str:
    return sha256_text(text)[:length]


def canonical_json_hash(value: Any) -> str:
    return sha256_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")))


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    path = Path(path)
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: str | Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def atomic_write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    atomic_write_text(path, text)


def first_nonempty(mapping: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = mapping.get(key)
        if value not in (None, "", []):
            return value
    return default


def as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
        return [value]
    return [str(value)]


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
        sample_id = str(first_nonempty(item, ("id", "sample_id", "qid"), f"sample_{index}"))
        if wanted and sample_id not in wanted:
            continue
        question = str(first_nonempty(item, ("question", "query"), "")).strip()
        if not question:
            continue
        samples.append({
            "sample_id": sample_id,
            "question": question,
            "gold_answers": as_string_list(first_nonempty(item, ("golden_answers", "answers", "answer"), [])),
        })
        if limit and len(samples) >= limit:
            break
    return samples, {
        "benchmark": "bamboogle",
        "intermediate_path": str(intermediate_path),
        "raw_items": len(raw),
        "loaded_samples": len(samples),
        "selected_sample_ids": [sample["sample_id"] for sample in samples],
    }


def assert_qwen_configuration(model_path: str, model_url: str) -> None:
    normalized = str(model_path).replace("\\", "/").rstrip("/").lower()
    if normalized.split("/")[-1] != EXPECTED_MODEL_BASENAME:
        raise RuntimeError(f"MODEL_PATH must end in Qwen3-8B; received {model_path!r}")
    if re.search(r"smartsearch[-_]?3b", normalized, flags=re.I):
        raise RuntimeError(f"SmartSearch-3B is forbidden: {model_path}")
    if re.search(r":9000(?:/|$)", str(model_url)):
        raise RuntimeError(f"Port 9000 belongs to the old SmartSearch model: {model_url}")


def probe_model_identity(model_url: str, timeout: float = 15.0) -> Dict[str, Any]:
    import requests

    attempts: List[Dict[str, Any]] = []
    for suffix in ("/get_model_info", "/v1/models"):
        url = model_url.rstrip("/") + suffix
        try:
            response = requests.get(url, timeout=timeout)
            body = response.text[:10000]
            attempt = {"url": url, "status_code": response.status_code, "body": body}
            attempts.append(attempt)
            lower = body.lower()
            if response.ok and (EXPECTED_MODEL_BASENAME in lower or "smartsearch-3b" in lower):
                return {
                    "verified_qwen3_8b": EXPECTED_MODEL_BASENAME in lower and "smartsearch-3b" not in lower,
                    "evidence": attempt,
                    "attempts": attempts,
                }
        except Exception as exc:
            attempts.append({"url": url, "error": str(exc)})
    return {"verified_qwen3_8b": False, "evidence": None, "attempts": attempts}


def probe_server_context(model_url: str, timeout: float = 15.0) -> Dict[str, Any]:
    import requests

    attempts: List[Dict[str, Any]] = []
    for suffix in ("/get_server_args", "/server_info"):
        url = model_url.rstrip("/") + suffix
        try:
            response = requests.get(url, timeout=timeout)
            attempt: Dict[str, Any] = {"url": url, "status_code": response.status_code, "body": response.text[:10000]}
            attempts.append(attempt)
            if not response.ok:
                continue
            value = response.json()
            candidates = [value]
            if isinstance(value, dict):
                candidates.extend(item for item in value.values() if isinstance(item, dict))
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                for key in ("context_length", "context_len", "max_context_len", "max_model_len"):
                    if candidate.get(key) is not None:
                        return {"verified": True, "context_length": int(candidate[key]), "evidence": attempt, "attempts": attempts}
        except Exception as exc:
            attempts.append({"url": url, "error": str(exc)})
    return {"verified": False, "context_length": None, "evidence": None, "attempts": attempts}


class ExactTokenizer:
    """Exact local Qwen tokenizer. Approximation is forbidden in formal runs."""

    def __init__(self, model_path: str = "", tokenizer: Any = None):
        self.model_path = model_path
        if tokenizer is not None:
            self.tokenizer = tokenizer
            return
        try:
            from transformers import AutoTokenizer  # type: ignore

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, trust_remote_code=True, local_files_only=True
            )
        except Exception as exc:
            raise RuntimeError(
                f"Exact local Qwen tokenizer is required at {model_path!r}: {exc}"
            ) from exc

    def encode(self, text: Any) -> List[int]:
        return list(self.tokenizer.encode(str(text or ""), add_special_tokens=False))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self.tokenizer.decode(
            list(token_ids), skip_special_tokens=True, clean_up_tokenization_spaces=False
        ))

    def count(self, text: Any) -> int:
        return len(self.encode(text)) if str(text or "") else 0


def extract_short_answer(text: str) -> Tuple[str, str]:
    raw = str(text or "").strip()
    tagged = extract_last_tag(raw, "answer").strip()
    if tagged:
        return extract_final_answer(raw).strip(), "answer_tag"
    opened = re.findall(r"<answer>\s*([^<\r\n]+?)\s*$", raw, flags=re.I)
    if opened:
        return opened[-1].strip(), "open_answer_tag"
    boxed = extract_final_answer(raw).strip() if "\\boxed{" in raw else ""
    if boxed and boxed != raw:
        return boxed, "boxed"
    cues = re.findall(r"(?:final\s+answer|answer)\s*:\s*([^\n<>]+)", raw, flags=re.I)
    if cues:
        return cues[-1].strip(), "answer_cue"
    tail = re.split(r"</think>", raw, maxsplit=1, flags=re.I)[-1]
    cleaned = re.sub(r"</?(?:answer|think)>\s*", "", tail, flags=re.I).strip()
    return cleaned, "natural_language" if cleaned else "empty"


def extract_read_notes(text: str) -> Tuple[str, str]:
    raw = str(text or "").strip()
    tagged = extract_last_tag(raw, "read").strip()
    if tagged:
        return tagged, "read_tag"
    opened = re.findall(r"<read>\s*(.*)$", raw, flags=re.I | re.S)
    if opened:
        return opened[-1].strip(), "open_read_tag"
    return raw, "natural_language" if raw else "empty"


def condition_key(record: Dict[str, Any]) -> Tuple[str, str, str, int]:
    return (
        str(record.get("sample_id", "")),
        str(record.get("condition_id", "")),
        str(record.get("mode", "")),
        int(record.get("rollout_id", 0)),
    )


class RecordStore:
    """Crash-safe per-record parts plus an atomically consolidated JSONL."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.parts_dir = self.output_dir / "record_parts"
        self.failure_parts_dir = self.output_dir / "failure_parts"
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.failure_parts_dir.mkdir(parents=True, exist_ok=True)
        self._import_legacy_jsonl()

    @staticmethod
    def _name(key: Tuple[str, str, str, int]) -> str:
        return stable_hash("|".join(map(str, key)), 40) + ".json"

    def _import_legacy_jsonl(self) -> None:
        records_path = self.output_dir / "records.jsonl"
        if records_path.exists() and not any(self.parts_dir.glob("*.json")):
            for row in read_jsonl(records_path):
                atomic_write_json(self.parts_dir / self._name(condition_key(row)), row)

    def completed_keys(self) -> set[Tuple[str, str, str, int]]:
        keys = set()
        for path in self.parts_dir.glob("*.json"):
            keys.add(condition_key(read_json(path)))
        return keys

    def failed_keys(self) -> set[Tuple[str, str, str, int]]:
        keys = set()
        for path in self.failure_parts_dir.glob("*.json"):
            row = read_json(path)
            keys.add((str(row["sample_id"]), str(row["condition_id"]), str(row["mode"]), int(row["rollout_id"])))
        return keys

    def put_record(self, record: Dict[str, Any]) -> None:
        key = condition_key(record)
        atomic_write_json(self.parts_dir / self._name(key), record)
        failure = self.failure_parts_dir / self._name(key)
        if failure.exists():
            failure.unlink()

    def put_failure(self, failure: Dict[str, Any]) -> None:
        key = (str(failure["sample_id"]), str(failure["condition_id"]), str(failure["mode"]), int(failure["rollout_id"]))
        atomic_write_json(self.failure_parts_dir / self._name(key), failure)

    def consolidate(self) -> Tuple[int, int]:
        records = [read_json(path) for path in self.parts_dir.glob("*.json")]
        records.sort(key=condition_key)
        failures = [read_json(path) for path in self.failure_parts_dir.glob("*.json")]
        failures.sort(key=lambda row: (str(row.get("sample_id")), str(row.get("condition_id")), str(row.get("mode")), int(row.get("rollout_id", 0))))
        atomic_write_jsonl(self.output_dir / "records.jsonl", records)
        atomic_write_jsonl(self.output_dir / "failures.jsonl", failures)
        return len(records), len(failures)


def model_usage(tokenizer: ExactTokenizer, prompt: str, output: str) -> Dict[str, int]:
    return {
        "input_tokens": tokenizer.count(prompt),
        "output_tokens": tokenizer.count(output),
        "reasoning_tokens": tokenizer.count(extract_think_text(output)),
    }


__all__ = [
    "PROTOCOL_VERSION", "ModelClient", "answer_metrics",
    "public_model_meta", "utc_stamp", "sha256_text", "stable_hash",
    "canonical_json_hash", "read_json", "read_jsonl", "atomic_write_json",
    "atomic_write_jsonl", "load_bamboogle_questions", "assert_qwen_configuration",
    "probe_model_identity", "ExactTokenizer", "extract_short_answer",
    "probe_server_context",
    "extract_read_notes", "RecordStore", "condition_key", "model_usage",
]
