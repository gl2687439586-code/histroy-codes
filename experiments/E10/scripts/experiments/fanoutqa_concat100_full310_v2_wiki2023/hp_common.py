#!/usr/bin/env python3
"""Shared infrastructure for FanOutQA natural multi-search budget experiments."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import sqlite3
import string
import time
import unicodedata
import zlib
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence, Tuple

SCHEMA_VERSION = (
    "fanoutqa-dev310-qwen3-natural-thinking-up-to5-concat100-"
    "wiki2023-09-kiwix-2gpu-g2g3-v3.0"
)
EXPOSURE_STEP_TOKENS = 100
MAX_BOUNDARY_BACKOFF_TOKENS = 16
TOP_K = int(os.environ.get("TOP_K", "5"))
if TOP_K <= 0:
    raise ValueError(f"TOP_K must be positive, got {TOP_K}")
RETRIEVER_BACKEND = os.environ.get(
    "RETRIEVER_BACKEND",
    "fanoutqa_wikipedia_en_all_nopic_2023-09_kiwix_passage100_upto5",
)


class ExactTokenizer:
    """Exact local Qwen tokenizer; approximations are forbidden in formal runs."""

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
            list(token_ids), skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ))

    def count(self, text: Any) -> int:
        return len(self.encode(text)) if str(text or "") else 0

    def offsets(self, text: Any) -> List[Tuple[int, int]]:
        """Return exact character offsets for each Qwen token.

        The formal server tokenizer is fast and supports offset mappings.  A
        deterministic prefix-count fallback keeps the offline protocol tests
        usable with deliberately tiny fake tokenizers.
        """
        value = str(text or "")
        try:
            encoded = self.tokenizer(
                value, add_special_tokens=False, return_offsets_mapping=True
            )
            offsets = encoded.get("offset_mapping")
            if offsets is not None:
                return [(int(start), int(end)) for start, end in offsets]
        except (TypeError, AttributeError, NotImplementedError):
            pass
        token_ids = self.encode(value)
        result: List[Tuple[int, int]] = []
        previous = 0
        for index in range(1, len(token_ids) + 1):
            current = len(self.decode(token_ids[:index]))
            current = max(previous, min(len(value), current))
            result.append((previous, current))
            previous = current
        return result


def _normalize_answer(text: Any) -> str:
    value = str(text or "").lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def answer_metrics(prediction: str, gold_answers: Sequence[str]) -> Dict[str, float]:
    """Hotpot/SQuAD-style normalized EM and token F1 plus compatibility fields."""
    golds = [str(item) for item in gold_answers] or [""]
    prediction_normalized = _normalize_answer(prediction)
    normalized_golds = [_normalize_answer(item) for item in golds]
    exact = float(any(prediction_normalized == gold for gold in normalized_golds))
    best_f1 = 0.0
    for gold in normalized_golds:
        predicted_tokens = prediction_normalized.split()
        gold_tokens = gold.split()
        if not predicted_tokens or not gold_tokens:
            continue
        common = sum((Counter(predicted_tokens) & Counter(gold_tokens)).values())
        if common:
            precision = common / len(predicted_tokens)
            recall = common / len(gold_tokens)
            best_f1 = max(best_f1, 2 * precision * recall / (precision + recall))
    contains = float(any(gold and gold in prediction_normalized for gold in normalized_golds))
    bidirectional = float(any(
        gold and prediction_normalized
        and (gold in prediction_normalized or prediction_normalized in gold)
        for gold in normalized_golds
    ))
    return {
        "em": exact, "f1": best_f1, "acc": contains,
        "normalized_exact": exact, "normalized_contains": bidirectional,
    }


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_text(value: Any) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return sha256_text(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ))


def atomic_write(path: str | Path, text: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path: str | Path, value: Any) -> None:
    atomic_write(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    atomic_write(path, "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ))


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            yield row


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    return list(iter_jsonl(path))


def write_jsonl_gz(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(temporary, path)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def extract_title(contents: str) -> str:
    first = str(contents or "").splitlines()[0].strip() if contents else ""
    return first.strip().strip('"').strip()


def finish_type(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type", "")).lower()
    return str(value or "").lower()


@dataclass
class Generation:
    text: str
    raw_text: str
    raw: Any
    finish_reason: Any
    latency_seconds: float
    endpoint: str
    stop_repair: str


class SafeSGLangClient:
    """SGLang 0.4.6 client with prefix-aware stop-delimiter completion."""

    def __init__(self, base_url: str, timeout: float = 300.0, retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries

    @staticmethod
    def _complete_stop(text: str, finish: Any, stops: Sequence[str]) -> Tuple[str, str]:
        matched = finish.get("matched") if isinstance(finish, dict) else None
        target = matched if isinstance(matched, str) and matched in stops else None
        if target is None and isinstance(matched, str):
            target = next((stop for stop in stops if stop.startswith(matched)), None)
        if not target or text.endswith(target):
            return text, "none"
        # Complete the longest delimiter prefix already present at the end.
        for size in range(len(target) - 1, 0, -1):
            if text.endswith(target[:size]):
                return text + target[size:], f"completed_prefix_{size}"
        return text + target, "appended_full_stop"

    def generate(self, prompt: str, max_new_tokens: int,
                 stops: Sequence[str], temperature: float = 0.0) -> Generation:
        import requests

        url = self.base_url + "/generate"
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop": list(stops),
            },
        }
        last_error: Exception | None = None
        for attempt in range(self.retries):
            started = time.time()
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                if not response.ok:
                    raise RuntimeError(
                        f"HTTP {response.status_code}: {response.text[:1000]}"
                    )
                raw = response.json()
                item = raw[0] if isinstance(raw, list) else raw
                raw_text = str(item.get("text", ""))
                finish = (item.get("meta_info") or {}).get("finish_reason", {})
                text, repair = self._complete_stop(raw_text, finish, stops)
                return Generation(
                    text=text, raw_text=raw_text, raw=raw,
                    finish_reason=finish,
                    latency_seconds=time.time() - started,
                    endpoint=url, stop_repair=repair,
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Generation failed after {self.retries} tries: {last_error}")


def probe_qwen(base_url: str, timeout: float = 30.0) -> Dict[str, Any]:
    import requests

    base = base_url.rstrip("/")
    health = requests.get(base + "/health", timeout=timeout)
    info = requests.get(base + "/get_model_info", timeout=timeout)
    health.raise_for_status()
    info.raise_for_status()
    body = info.text
    verified = "qwen3-8b" in body.casefold() and "smartsearch-3b" not in body.casefold()
    return {
        "verified_qwen3_8b": verified,
        "health_status": health.status_code,
        "model_info": info.json() if "json" in info.headers.get("content-type", "") else body,
    }


class SQLiteRecordStore:
    """Crash-safe, compressed per-shard store suitable for hundreds of thousands of rows."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), timeout=60)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS records "
            "(key TEXT PRIMARY KEY, payload BLOB NOT NULL)"
        )
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS failures "
            "(key TEXT PRIMARY KEY, payload BLOB NOT NULL)"
        )
        self.db.commit()

    @staticmethod
    def _pack(value: Dict[str, Any]) -> bytes:
        raw = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return zlib.compress(raw, level=6)

    @staticmethod
    def _unpack(value: bytes) -> Dict[str, Any]:
        return json.loads(zlib.decompress(value).decode("utf-8"))

    def completed_keys(self) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT key FROM records")}

    def failed_keys(self) -> set[str]:
        return {row[0] for row in self.db.execute("SELECT key FROM failures")}

    def put_record(self, key: str, value: Dict[str, Any]) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO records(key,payload) VALUES (?,?)",
                (key, self._pack(value)),
            )
            self.db.execute("DELETE FROM failures WHERE key=?", (key,))

    def put_failure(self, key: str, value: Dict[str, Any]) -> None:
        with self.db:
            self.db.execute(
                "INSERT OR REPLACE INTO failures(key,payload) VALUES (?,?)",
                (key, self._pack(value)),
            )

    def invalidate_record(self, key: str, failure: Dict[str, Any]) -> None:
        """Atomically move one completed key back to the retryable set."""
        with self.db:
            self.db.execute("DELETE FROM records WHERE key=?", (key,))
            self.db.execute(
                "INSERT OR REPLACE INTO failures(key,payload) VALUES (?,?)",
                (key, self._pack(failure)),
            )

    def counts(self) -> Tuple[int, int]:
        records = self.db.execute("SELECT COUNT(*) FROM records").fetchone()[0]
        failures = self.db.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
        return int(records), int(failures)

    def iter_records(self) -> Iterator[Dict[str, Any]]:
        for _, payload in self.db.execute("SELECT key,payload FROM records ORDER BY key"):
            yield self._unpack(payload)

    def iter_failures(self) -> Iterator[Dict[str, Any]]:
        for _, payload in self.db.execute("SELECT key,payload FROM failures ORDER BY key"):
            yield self._unpack(payload)

    def export(self, output_dir: str | Path) -> Tuple[int, int]:
        output = Path(output_dir)
        records, failures = self.counts()
        write_jsonl_gz(output / "records.jsonl.gz", self.iter_records())
        write_jsonl_gz(output / "failures.jsonl.gz", self.iter_failures())
        return records, failures

    def close(self) -> None:
        self.db.close()


__all__ = [
    "SCHEMA_VERSION", "EXPOSURE_STEP_TOKENS", "MAX_BOUNDARY_BACKOFF_TOKENS", "TOP_K",
    "ExactTokenizer", "answer_metrics", "SafeSGLangClient", "SQLiteRecordStore",
    "atomic_write", "canonical_hash", "extract_title", "file_sha256",
    "finish_type", "iter_jsonl", "normalize_text", "probe_qwen", "read_json",
    "read_jsonl", "sha256_text", "utc_stamp",
    "write_json", "write_jsonl", "write_jsonl_gz",
]
