#!/usr/bin/env python3
"""Shared I/O, scoring, hashing, prompting, and SGLang helpers."""

from __future__ import annotations

import hashlib
import json
import os
import re
import string
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence


SCHEMA_VERSION = "adaptive-k-hotpot500-qwen3-e5-single-pass-v1"


def utc_stamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def iter_jsonl(path: str | Path) -> Iterator[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def append_jsonl(path: str | Path, value: Dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", dir=str(destination.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def extract_title(contents: str) -> str:
    first = str(contents).splitlines()[0].strip() if contents else ""
    return first.strip().strip('"').strip()


def normalize_title(value: Any) -> str:
    text = str(value or "").replace("_", " ").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_answer(value: Any) -> str:
    text = str(value or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def answer_metrics(prediction: str, gold_answers: Sequence[str]) -> Dict[str, float]:
    predicted = normalize_answer(prediction)
    golds = [normalize_answer(item) for item in gold_answers] or [""]
    exact = float(any(predicted == gold for gold in golds))
    f1_values: List[float] = []
    for gold in golds:
        pred_tokens, gold_tokens = predicted.split(), gold.split()
        if not pred_tokens or not gold_tokens:
            f1_values.append(float(pred_tokens == gold_tokens))
            continue
        common = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
        if not common:
            f1_values.append(0.0)
            continue
        precision = common / len(pred_tokens)
        recall = common / len(gold_tokens)
        f1_values.append(2 * precision * recall / (precision + recall))
    subem = float(any(gold and gold in predicted for gold in golds))
    return {"em": exact, "f1": max(f1_values), "subem": subem}


OFFICIAL_QA_TEMPLATE = """Your task is to answer the question provided. To help you answer accurately, some relevant context documents have been retrieved. After reviewing them, you’ll be asked the same question again. Please respond succinctly.

**Input:**
- **Question:**
```
{question}
```

- **Context:**
```
{context}
```

- **Question:**
```
{question}
```


**Response:**
- **Answer:**
"""


def paper_qa_prompt(question: str, context: str) -> str:
    return OFFICIAL_QA_TEMPLATE.format(question=question.strip(), context=context)


def qwen_chat_prompt(tokenizer: Any, user_prompt: str) -> str:
    messages = [{"role": "user", "content": user_prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


def finish_type(meta: Dict[str, Any]) -> str:
    value = meta.get("finish_reason")
    if isinstance(value, dict):
        return str(value.get("type") or value.get("reason") or "unknown")
    return str(value or "unknown")


def extract_qwen_answer(raw: str, meta: Dict[str, Any]) -> Dict[str, Any]:
    text = str(raw or "").strip()
    had_think_open = "<think>" in text
    had_think_close = "</think>" in text
    if had_think_close:
        answer = text.rsplit("</think>", 1)[-1].strip()
    elif had_think_open:
        answer = ""
    else:
        answer = text
    answer = re.sub(r"^\s*(?:[-*]\s*)?(?:\*\*)?Answer(?:\*\*)?\s*:\s*", "", answer, flags=re.I)
    answer = answer.strip()
    termination = finish_type(meta)
    if not answer:
        status = "truncated_thinking" if termination == "length" or had_think_open else "empty_answer"
    elif termination == "length":
        status = "length_terminated_with_answer"
    else:
        status = "valid"
    return {
        "final_answer": answer,
        "answer_status": status,
        "finish_type": termination,
        "had_think_open": had_think_open,
        "had_think_close": had_think_close,
    }


@dataclass
class GenerationResult:
    text: str
    meta: Dict[str, Any]
    elapsed_seconds: float
    response_model: str | None


class SGLangClient:
    def __init__(self, base_url: str, timeout: float = 900.0, retries: int = 3):
        import requests
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.trust_env = False

    def model_info(self) -> Dict[str, Any]:
        response = self.session.get(self.base_url + "/get_model_info", timeout=30)
        response.raise_for_status()
        value = response.json()
        if not isinstance(value, dict):
            raise RuntimeError("Model info endpoint did not return an object")
        return value

    def generate(
        self,
        prompt: str,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> GenerationResult:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_new_tokens": int(max_new_tokens),
                "stop": ["<|im_end|>"],
            },
        }
        last_error: Exception | None = None
        for attempt in range(self.retries):
            started = time.time()
            try:
                response = self.session.post(
                    self.base_url + "/generate", json=payload, timeout=self.timeout
                )
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
                value = response.json()
                if not isinstance(value, dict):
                    raise RuntimeError("Generate endpoint did not return an object")
                return GenerationResult(
                    text=str(value.get("text") or ""),
                    meta=dict(value.get("meta_info") or {}),
                    elapsed_seconds=time.time() - started,
                    response_model=value.get("model"),
                )
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Generate failed after {self.retries} attempts: {last_error}")


__all__ = [
    "SCHEMA_VERSION", "SGLangClient", "answer_metrics", "append_jsonl",
    "canonical_hash", "canonical_json", "extract_qwen_answer", "extract_title",
    "file_sha256", "iter_jsonl", "normalize_answer", "normalize_title",
    "paper_qa_prompt", "qwen_chat_prompt", "read_jsonl", "text_sha256",
    "utc_stamp", "write_json", "write_jsonl",
]
