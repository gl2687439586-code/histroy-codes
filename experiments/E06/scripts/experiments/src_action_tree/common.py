#!/usr/bin/env python3
"""Shared parsing, API, token-counting, and evaluation helpers.

This module intentionally depends only on the Python standard library plus
``requests``.  ``transformers`` is used when available for exact model-token
counts; otherwise a clearly marked approximation is used.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import string
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests


RECALL_SYSTEM_PROMPT = (
    "You are a helpful assistant that can solve the given question step by step with the help of the wikipedia search tool. "
    "Given a question, you need to first think about the reasoning process in the mind and then provide the answer. "
    "During thinking, you can invoke the wikipedia search tool to search for fact information about specific topics if needed. "
    "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags respectively, "
    "and the search query and result are enclosed within <search> </search> and <result> </result> tags respectively. "
    "For example, <think> This is the reasoning process. </think> <search> search query here </search> <result> search result here </result> "
    "<think> This is the reasoning process. </think> <answer> The final answer is \\[ \\boxed{answer here} \\] </answer>. "
    "In the last part of the answer, the final exact answer is enclosed within \\boxed{} with latex format."
)


def utc_stamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def stable_hash(text: str, length: int = 16) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:length]


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(value, dict):
                rows.append(value)
    return rows


def append_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()


def as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item) for item in parsed if str(item).strip()]
        except Exception:
            pass
        return [stripped]
    return [str(value)]


def first_nonempty(mapping: Dict[str, Any], keys: Sequence[str], default: Any = "") -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None and value != "" and value != []:
            return value
    return default


def strip_latex_answer_wrappers(text: Any) -> str:
    """Remove presentation-only LaTeX wrappers without changing answer words.

    Qwen frequently emits ``\\text{Titan IIIE/Centaur}`` while the gold answer
    is plain text.  Treating the command name ``text`` as an answer token makes
    the score depend on formatting rather than factual content.
    """
    source = str(text or "").strip()
    previous = None
    while previous != source:
        previous = source
        source = re.sub(
            r"^\s*\\(?:boxed|text|textrm|mathrm|operatorname)\s*\{(.*)\}\s*$",
            r"\1",
            source,
            flags=re.I | re.S,
        ).strip()
        source = re.sub(r"^\s*\\?\[\s*(.*?)\s*\\?\]\s*$", r"\1", source, flags=re.S).strip()
        source = re.sub(r"^\s*\$+\s*(.*?)\s*\$+\s*$", r"\1", source, flags=re.S).strip()
    source = re.sub(r"\\(?:text|textrm|mathrm|operatorname)\s*\{([^{}]*)\}", r"\1", source)
    source = source.replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    return source.strip()


def normalize_answer(text: Any) -> str:
    text = strip_latex_answer_wrappers(text).lower()
    # Replace separators with spaces so ``Titan IIIE/Centaur`` does not become
    # the artificial token ``iiiecentaur``.
    text = "".join(" " if ch in set(string.punctuation) else ch for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def normalized_query(text: Any) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(text or "").lower()))


def token_f1(prediction: str, gold_answers: Sequence[str]) -> float:
    normalized_prediction = normalize_answer(prediction)
    best = 0.0
    for gold in gold_answers:
        normalized_gold = normalize_answer(gold)
        if normalized_prediction in {"yes", "no", "noanswer"} and normalized_prediction != normalized_gold:
            continue
        if normalized_gold in {"yes", "no", "noanswer"} and normalized_prediction != normalized_gold:
            continue
        pred_tokens = normalized_prediction.split()
        gold_tokens = normalized_gold.split()
        if not pred_tokens or not gold_tokens:
            continue
        overlap = Counter(pred_tokens) & Counter(gold_tokens)
        same = sum(overlap.values())
        if same == 0:
            continue
        precision = same / len(pred_tokens)
        recall = same / len(gold_tokens)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def answer_metrics(prediction: str, gold_answers: Sequence[str]) -> Dict[str, float]:
    golds = [str(item) for item in gold_answers] or [""]
    pred_norm = normalize_answer(prediction)
    gold_norms = [normalize_answer(item) for item in golds]
    exact = float(any(pred_norm == gold for gold in gold_norms))
    contains = float(any(gold and gold in pred_norm for gold in gold_norms))
    bidirectional_contains = float(
        any(gold and pred_norm and (gold in pred_norm or pred_norm in gold) for gold in gold_norms)
    )
    return {
        "em": exact,
        "f1": token_f1(prediction, golds),
        "acc": contains,
        "normalized_exact": exact,
        "normalized_contains": bidirectional_contains,
    }


def extract_last_tag(text: str, tag: str) -> str:
    matches = re.findall(rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.DOTALL | re.IGNORECASE)
    return matches[-1].strip() if matches else ""


def extract_boxed(text: str) -> str:
    """Extract the final balanced ``\boxed{...}``, including nested braces."""
    source = text or ""
    start = source.rfind("\\boxed{")
    if start < 0:
        return ""
    pos = start + len("\\boxed{")
    depth = 1
    chars: List[str] = []
    while pos < len(source):
        char = source[pos]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(char)
        else:
            chars.append(char)
        pos += 1
    return ""


def extract_final_answer(text: str) -> str:
    answer_block = extract_last_tag(text, "answer")
    boxed = extract_boxed(answer_block or text)
    if boxed:
        return strip_latex_answer_wrappers(boxed)
    if answer_block:
        cleaned = re.sub(
            r"^\s*(?:the\s+final\s+answer\s+is|final\s+answer\s*:|answer\s*:?)\s*",
            "",
            answer_block,
            flags=re.I,
        )
        return strip_latex_answer_wrappers(cleaned)
    return strip_latex_answer_wrappers(text)


def extract_think_text(text: str) -> str:
    return "\n".join(part.strip() for part in re.findall(r"<think>(.*?)</think>", text or "", re.S | re.I))


class TokenCounter:
    """Use the local SmartSearch tokenizer, with an explicit approximation fallback."""

    def __init__(self, model_path: Optional[str] = None, force_approximate: bool = False):
        self.model_path = model_path
        self.tokenizer = None
        self.approximate = True
        self.error = ""
        if model_path and not force_approximate:
            try:
                from transformers import AutoTokenizer  # type: ignore

                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path, trust_remote_code=True, local_files_only=True
                )
                self.approximate = False
            except Exception as exc:  # pragma: no cover - depends on server environment
                self.error = str(exc)

    def count(self, text: Any) -> int:
        source = str(text or "")
        if not source:
            return 0
        if self.tokenizer is not None:
            return len(self.tokenizer.encode(source, add_special_tokens=False))
        # Better than whitespace-only for tags, punctuation, and numbers.
        return len(re.findall(r"[A-Za-z0-9_]+|[^\w\s]", source, flags=re.UNICODE))


def make_document(doc_id: Any, contents: Any, title: Any = "") -> Dict[str, Any]:
    contents_s = str(contents or "").strip()
    doc_id_s = str(doc_id or "").strip()
    title_s = str(title or "").strip()
    key = doc_id_s or title_s or ("sha1:" + stable_hash(contents_s, 40))
    return {"id": doc_id_s, "title": title_s, "contents": contents_s, "key": key}


def documents_from_trace(row: Dict[str, Any], result_text: str = "") -> List[Dict[str, Any]]:
    for key in ("docs", "documents", "retrieved_docs", "retrieval_result"):
        value = row.get(key)
        if isinstance(value, list) and value:
            documents = []
            for index, item in enumerate(value):
                if isinstance(item, dict):
                    documents.append(
                        make_document(
                            first_nonempty(item, ("id", "doc_id", "key"), ""),
                            first_nonempty(item, ("contents", "content", "text", "passage"), ""),
                            first_nonempty(item, ("title", "doc_title"), ""),
                        )
                    )
                else:
                    documents.append(make_document("", item, f"document_{index}"))
            if documents:
                return documents

    ids = as_string_list(first_nonempty(row, ("doc_ids", "doc_keys"), []))
    titles = as_string_list(row.get("doc_titles"))
    contents = as_string_list(first_nonempty(row, ("doc_contents", "passages"), []))
    if contents:
        return [
            make_document(ids[i] if i < len(ids) else "", text, titles[i] if i < len(titles) else "")
            for i, text in enumerate(contents)
        ]

    clean = str(result_text or "").strip()
    clean = re.sub(r"^\s*result\s*:\s*", "", clean, flags=re.I)
    chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", clean) if chunk.strip()]
    docs = []
    for index, chunk in enumerate(chunks):
        docs.append(
            make_document(
                ids[index] if index < len(ids) else "",
                chunk,
                titles[index] if index < len(titles) else "",
            )
        )
    # When only keys/titles are available, preserve the expected document count.
    expected = max(len(ids), len(titles))
    while len(docs) < expected:
        index = len(docs)
        docs.append(
            make_document(
                ids[index] if index < len(ids) else "",
                "",
                titles[index] if index < len(titles) else "",
            )
        )
    return docs


def split_first_checkpoint(full_response: str) -> Optional[Dict[str, str]]:
    """Return the exact prefix through the first tool observation."""
    if not full_response:
        return None

    # ReCall stores the complete prompt in ``final_response``.  Its system
    # instruction itself contains illustrative <search>/<result> tags (and
    # some local variants use empty tags), so a search from character zero can
    # mistake the instruction example for the model's first real tool call.
    # The final assistant marker begins the actual rollout while keeping the
    # returned checkpoint prefix byte-for-byte identical to the saved trace.
    assistant_marker = "<|im_start|>assistant"
    assistant_start = full_response.rfind(assistant_marker)
    search_start = assistant_start + len(assistant_marker) if assistant_start >= 0 else 0

    for search_match in re.finditer(
        r"<search>(.*?)</search>", full_response[search_start:], re.S | re.I
    ):
        absolute_search_start = search_start + search_match.start()
        absolute_search_end = search_start + search_match.end()
        query = search_match.group(1).strip()
        if not query:
            continue
        result_match = re.search(
            r"<result>(.*?)</result>", full_response[absolute_search_end:], re.S | re.I
        )
        if not result_match:
            continue
        result_start = absolute_search_end + result_match.start()
        result_end = absolute_search_end + result_match.end()
        # Preserve a following newline because ReCall.cat_tool_results adds one.
        if result_end < len(full_response) and full_response[result_end] == "\r":
            result_end += 1
        if result_end < len(full_response) and full_response[result_end] == "\n":
            result_end += 1
        return {
            "prefix": full_response[:result_end],
            "query": query,
            "result": result_match.group(1).strip(),
            "search_fragment": full_response[absolute_search_start:absolute_search_end],
            "result_fragment": full_response[result_start:result_end],
        }
    return None


def count_rollout_searches(full_response: str) -> int:
    """Count real Search actions while ignoring tag examples in the system prompt."""
    if not full_response:
        return 0
    assistant_marker = "<|im_start|>assistant"
    assistant_start = full_response.rfind(assistant_marker)
    search_start = assistant_start + len(assistant_marker) if assistant_start >= 0 else 0
    return sum(
        1
        for match in re.finditer(r"<search>(.*?)</search>", full_response[search_start:], re.S | re.I)
        if match.group(1).strip()
    )


def canonical_initial_prefix(question: str, trace_row: Dict[str, Any]) -> Dict[str, str]:
    query = str(first_nonempty(trace_row, ("query", "search_query", "tool_query"), "")).strip()
    result = str(first_nonempty(trace_row, ("result_raw", "result", "tool_result", "observation"), "")).strip()
    thought = str(
        first_nonempty(
            trace_row,
            ("think_before_query", "thought_before_query", "thinking_before_query", "think_before_search"),
            "",
        )
    ).strip()
    system = f"<|im_start|>system\n{RECALL_SYSTEM_PROMPT}<|im_end|>"
    user = f"<|im_start|>user\n{question}<|im_end|>"
    assistant = f"<|im_start|>assistant\n<think>{thought}</think>\n<search>{query}</search>\n"
    if not result.lower().startswith("result:"):
        result = "result: \n" + result
    prefix = system + "\n" + user + "\n" + assistant + f"<result>{result}\n</result>\n"
    return {"prefix": prefix, "query": query, "result": result, "search_fragment": "", "result_fragment": ""}


def _step_number(row: Dict[str, Any]) -> int:
    for key in ("valid_step_idx", "step_idx", "raw_step_idx", "step"):
        try:
            return int(row.get(key))
        except Exception:
            continue
    return 10**9


def load_experiment_samples(
    intermediate_path: str | Path,
    trace_path: Optional[str | Path] = None,
    limit: int = 0,
    sample_ids: Optional[Sequence[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load and reconcile FlashRAG intermediate data with the human trace export."""
    raw = read_json(intermediate_path)
    if isinstance(raw, dict):
        for key in ("data", "items", "samples", "records"):
            if isinstance(raw.get(key), list):
                raw = raw[key]
                break
    if not isinstance(raw, list):
        raise ValueError(f"Expected a JSON list in {intermediate_path}")

    trace_rows = read_jsonl(trace_path) if trace_path and Path(trace_path).exists() else []
    traces: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in trace_rows:
        sid = str(first_nonempty(row, ("sample_id", "id", "qid"), ""))
        if sid:
            traces[sid].append(row)
    for rows in traces.values():
        rows.sort(key=_step_number)

    wanted = set(str(item) for item in (sample_ids or []))
    samples: List[Dict[str, Any]] = []
    warnings: List[str] = []

    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        output = item.get("output") if isinstance(item.get("output"), dict) else {}
        sid = str(first_nonempty(item, ("id", "sample_id", "qid"), f"sample_{index}"))
        if wanted and sid not in wanted:
            continue
        question = str(first_nonempty(item, ("question", "query"), ""))
        gold = as_string_list(first_nonempty(item, ("golden_answers", "answers", "answer"), []))
        full_response = str(first_nonempty(output, ("final_response", "response", "trace"), ""))
        if not full_response:
            full_response = str(first_nonempty(item, ("final_response", "response", "trace"), ""))

        trace_candidates = traces.get(sid, [])
        if not trace_candidates:
            # Some old exports omit IDs; exact question matching is safe here.
            trace_candidates = [row for row in trace_rows if str(row.get("question", "")) == question]
            trace_candidates.sort(key=_step_number)
        first_trace = trace_candidates[0] if trace_candidates else {}
        if not gold and first_trace:
            gold = as_string_list(first_nonempty(first_trace, ("golden_answers", "answers", "answer"), []))

        checkpoint = split_first_checkpoint(full_response)
        source = "intermediate_data.output.final_response"
        if checkpoint is None:
            for key in ("checkpoint_prompt", "prompt_after_result", "raw_prompt", "full_prompt"):
                checkpoint = split_first_checkpoint(str(first_trace.get(key, "")))
                if checkpoint:
                    source = f"trace.{key}"
                    break
        if checkpoint is None and first_trace:
            checkpoint = canonical_initial_prefix(question, first_trace)
            source = "trace_reconstructed"
            warnings.append(f"{sid}: exact final_response checkpoint missing; reconstructed from trace fields")
        if checkpoint is None:
            warnings.append(f"{sid}: skipped because no complete first <search>/<result> checkpoint was found")
            continue
        if not checkpoint["query"]:
            warnings.append(f"{sid}: skipped because initial search query is empty")
            continue

        docs = documents_from_trace(first_trace, checkpoint["result"])
        full_response_search_count = count_rollout_searches(full_response)
        trace_search_count = sum(
            1
            for row in trace_candidates
            if str(first_nonempty(row, ("query", "search_query", "tool_query"), "")).strip()
        )
        original_search_count = max(1, full_response_search_count, trace_search_count)
        samples.append(
            {
                "sample_id": sid,
                "question": question,
                "gold_answers": gold,
                "checkpoint_prefix": checkpoint["prefix"],
                "initial_query": checkpoint["query"],
                "initial_result": checkpoint["result"],
                "initial_documents": docs,
                "checkpoint_source": source,
                "original_prediction": str(output.get("pred", "")),
                "original_metric_score": output.get("metric_score", {}),
                "original_search_count": original_search_count,
                "original_search_count_from_final_response": full_response_search_count,
                "original_search_count_from_trace": trace_search_count,
            }
        )
        if limit and len(samples) >= limit:
            break

    report = {
        "intermediate_path": str(intermediate_path),
        "trace_path": str(trace_path or ""),
        "raw_items": len(raw),
        "trace_rows": len(trace_rows),
        "loaded_samples": len(samples),
        "warnings": warnings,
        "checkpoint_sources": dict(Counter(sample["checkpoint_source"] for sample in samples)),
    }
    return samples, report


@dataclass
class ModelResponse:
    text: str
    raw: Any
    endpoint: str
    latency_seconds: float
    finish_reason: Any = None
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    model_id: str = ""
    provider_id: str = ""
    reasoning_text: str = ""
    reasoning_details: Any = None
    request_messages: Optional[List[Dict[str, str]]] = None


def qwen_serialized_prompt_to_messages(prompt: str) -> List[Dict[str, str]]:
    """Convert the v6 Qwen chat serialization to provider-neutral messages.

    Controlled/neutral v6 prompts contain one system and one user block.  The
    Blank ReCall trajectory additionally appends assistant generations and
    ``<result>`` observations.  OpenRouter has no raw assistant-prefill API, so
    an initial bare ``<think>`` prefix is dropped and retrieved observations
    are represented as user messages.  The semantic prompt text and all
    evidence remain unchanged.
    """

    block = re.compile(
        r"<\|im_start\|>(system|user|assistant)\n(.*?)(?=<\|im_start\|>|\Z)",
        flags=re.S,
    )
    matches = list(block.finditer(prompt or ""))
    if not matches:
        return [{"role": "user", "content": str(prompt)}]

    messages: List[Dict[str, str]] = []
    for match in matches:
        role = match.group(1)
        content = re.sub(r"<\|im_end\|>\s*$", "", match.group(2), flags=re.S).strip()
        if role != "assistant":
            if content:
                messages.append({"role": role, "content": content})
            continue

        # Chat Completions cannot continue a partially emitted assistant
        # message.  The initial Qwen-only prefill is redundant because the
        # shared ReCall system instruction already specifies <think> tags.
        if content in {"", "<think>"}:
            continue

        cursor = 0
        for result in re.finditer(r"<result>.*?</result>", content, flags=re.I | re.S):
            assistant_text = content[cursor : result.start()].strip()
            if assistant_text and assistant_text != "<think>":
                messages.append({"role": "assistant", "content": assistant_text})
            messages.append({"role": "user", "content": result.group(0).strip()})
            cursor = result.end()
        assistant_tail = content[cursor:].strip()
        if assistant_tail and assistant_tail != "<think>":
            messages.append({"role": "assistant", "content": assistant_tail})

    if not messages:
        return [{"role": "user", "content": str(prompt)}]
    return messages


class ModelClient:
    """Minimal SGLang, OpenAI-completion, or OpenRouter-chat client."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_mode: str = "sglang",
        timeout: float = 300.0,
        retries: int = 3,
        api_key_env: str = "OPENROUTER_API_KEY",
        max_usd: float = 0.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_mode = api_mode
        self.timeout = timeout
        self.retries = retries
        self.api_key_env = api_key_env
        self.max_usd = float(max_usd or 0.0)
        self.billing_ledger_path = os.environ.get("OPENROUTER_BILLING_LEDGER", "").strip()
        self.historical_cost_usd = 0.0
        if self.billing_ledger_path and Path(self.billing_ledger_path).exists():
            seen_response_ids = set()
            for ledger_index, ledger_row in enumerate(read_jsonl(self.billing_ledger_path)):
                response_id = str(ledger_row.get("response_id") or f"missing-{ledger_index}")
                if response_id in seen_response_ids:
                    continue
                seen_response_ids.add(response_id)
                self.historical_cost_usd += float(ledger_row.get("cost_usd", 0.0))
        self.process_cost_usd = 0.0
        self.cumulative_cost_usd = self.historical_cost_usd
        self.openrouter_provider = os.environ.get("OPENROUTER_PROVIDER", "openai").strip()
        self.openrouter_allow_fallbacks = os.environ.get(
            "OPENROUTER_ALLOW_FALLBACKS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.openrouter_reasoning_effort = os.environ.get(
            "OPENROUTER_REASONING_EFFORT", "medium"
        ).strip().lower()

    def _post(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[Any, float]:
        error: Optional[Exception] = None
        for attempt in range(self.retries):
            started = time.time()
            try:
                response = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
                if not response.ok:
                    body = response.text.strip()
                    if len(body) > 2000:
                        body = body[:2000] + "..."
                    raise RuntimeError(
                        f"HTTP {response.status_code} {response.reason}"
                        + (f": {body}" if body else "")
                    )
                return response.json(), time.time() - started
            except Exception as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Request failed after {self.retries} tries: {url}: {error}")

    def generate(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        stop: Sequence[str],
        seed: int,
    ) -> ModelResponse:
        if self.api_mode == "sglang":
            endpoint = self.base_url + "/generate"
            raw, latency = self._post(
                endpoint,
                {
                    "text": prompt,
                    "sampling_params": {
                        "temperature": temperature,
                        "max_new_tokens": max_new_tokens,
                        "stop": list(stop),
                        # SGLang 0.4.6.post4's native SamplingParams accepts
                        # neither `seed` nor `sampling_seed`. At temperature=0
                        # this request uses deterministic greedy decoding.
                    },
                },
            )
            item = raw[0] if isinstance(raw, list) else raw
            text = str(item.get("text", ""))
            finish = (item.get("meta_info") or {}).get("finish_reason", {})
            matched = finish.get("matched") if isinstance(finish, dict) else None
            if isinstance(matched, str) and matched in stop and not text.endswith(matched):
                text += matched
            return ModelResponse(text=text, raw=raw, endpoint=endpoint, latency_seconds=latency, finish_reason=finish)

        if self.api_mode == "openrouter":
            if self.max_usd > 0 and self.cumulative_cost_usd >= self.max_usd:
                raise RuntimeError(
                    f"OpenRouter stage cost guard already reached: "
                    f"${self.cumulative_cost_usd:.6f} >= ${self.max_usd:.6f}"
                )
            api_key = os.environ.get(self.api_key_env, "").strip()
            if not api_key:
                raise RuntimeError(
                    f"Missing {self.api_key_env}; export the OpenRouter API key before running"
                )
            endpoint = self.base_url + "/chat/completions"
            messages = qwen_serialized_prompt_to_messages(prompt)
            payload: Dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "stop": list(stop),
                "seed": seed,
                "stream": False,
            }
            if self.openrouter_provider:
                payload["provider"] = {
                    "order": [self.openrouter_provider],
                    "only": [self.openrouter_provider],
                    "allow_fallbacks": self.openrouter_allow_fallbacks,
                }
            if self.openrouter_reasoning_effort:
                payload["reasoning"] = {
                    "effort": self.openrouter_reasoning_effort,
                    "exclude": False,
                }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Title": "SmartSearch SRC Action Tree v6",
                "X-OpenRouter-Metadata": "enabled",
                "X-OpenRouter-Cache": "false",
            }
            referer = os.environ.get("OPENROUTER_HTTP_REFERER", "").strip()
            if referer:
                headers["HTTP-Referer"] = referer
            raw, latency = self._post(endpoint, payload, headers=headers)
            choice = (raw.get("choices") or [{}])[0]
            message = choice.get("message") or {}
            router_metadata = raw.get("openrouter_metadata") or {}
            selected_provider = str(raw.get("provider") or "")
            if not selected_provider and isinstance(router_metadata, dict):
                endpoints = router_metadata.get("endpoints") or {}
                available = endpoints.get("available") or [] if isinstance(endpoints, dict) else []
                for endpoint_info in available:
                    if isinstance(endpoint_info, dict) and endpoint_info.get("selected") is True:
                        selected_provider = str(endpoint_info.get("provider") or "")
                        break
            text = str(message.get("content") or "")
            usage = raw.get("usage") or {}
            details = usage.get("completion_tokens_details") or {}
            cost = float(usage.get("cost") or 0.0)
            self.process_cost_usd += cost
            self.cumulative_cost_usd += cost
            ledger_path = self.billing_ledger_path
            if ledger_path:
                append_jsonl(
                    ledger_path,
                    [
                        {
                            "created_at": utc_stamp(),
                            "stage": os.environ.get("OPENROUTER_STAGE", "unknown"),
                            "response_id": str(raw.get("id") or ""),
                            "model": str(raw.get("model") or self.model),
                            "provider": selected_provider,
                            "prompt_tokens": int(
                                usage.get("prompt_tokens") or usage.get("input_tokens") or 0
                            ),
                            "completion_tokens": int(
                                usage.get("completion_tokens") or usage.get("output_tokens") or 0
                            ),
                            "reasoning_tokens": int(details.get("reasoning_tokens") or 0),
                            "cost_usd": cost,
                            "request_message_sha1": stable_hash(
                                json.dumps(messages, ensure_ascii=False, sort_keys=True), 40
                            ),
                        }
                    ],
                )
            if self.max_usd > 0 and self.cumulative_cost_usd > self.max_usd:
                raise RuntimeError(
                    f"OpenRouter stage cost guard exceeded: "
                    f"${self.cumulative_cost_usd:.6f} > ${self.max_usd:.6f}"
                )
            embedded_error = raw.get("error") or choice.get("error")
            if embedded_error or choice.get("finish_reason") == "error":
                encoded_error = json.dumps(
                    embedded_error or {"finish_reason": choice.get("finish_reason")},
                    ensure_ascii=False,
                )
                raise RuntimeError(f"OpenRouter returned an embedded provider error: {encoded_error[:2000]}")
            return ModelResponse(
                text=text,
                raw=raw,
                endpoint=endpoint,
                latency_seconds=latency,
                finish_reason=choice.get("finish_reason"),
                input_tokens=int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or usage.get("output_tokens") or 0),
                reasoning_tokens=int(details.get("reasoning_tokens") or 0),
                cost_usd=cost,
                model_id=str(raw.get("model") or self.model),
                provider_id=selected_provider,
                reasoning_text=str(
                    message.get("reasoning") or message.get("reasoning_content") or ""
                ),
                reasoning_details=message.get("reasoning_details"),
                request_messages=messages,
            )

        endpoint = self.base_url + "/v1/completions"
        raw, latency = self._post(
            endpoint,
            {
                "model": self.model,
                "prompt": prompt,
                "temperature": temperature,
                "max_tokens": max_new_tokens,
                "stop": list(stop),
                "seed": seed,
            },
        )
        choice = (raw.get("choices") or [{}])[0]
        return ModelResponse(
            text=str(choice.get("text", "")),
            raw=raw,
            endpoint=endpoint,
            latency_seconds=latency,
            finish_reason=choice.get("finish_reason"),
        )


@dataclass
class RetrievalResponse:
    query: str
    documents: List[Dict[str, Any]]
    result_text: str
    raw: Any
    endpoint: str
    latency_seconds: float


class RetrieverClient:
    """Call the same Wikipedia retriever directly or through SmartSearch sandbox."""

    def __init__(
        self,
        retriever_url: str,
        sandbox_url: str = "",
        mode: str = "direct",
        timeout: float = 300.0,
        retries: int = 3,
    ):
        self.retriever_url = retriever_url.rstrip("/")
        self.sandbox_url = sandbox_url.rstrip("/")
        self.mode = mode
        self.timeout = timeout
        self.retries = retries

    def _post(self, url: str, payload: Dict[str, Any]) -> Tuple[Any, float]:
        last_error: Optional[Exception] = None
        for attempt in range(self.retries):
            started = time.time()
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json(), time.time() - started
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(f"Retriever request failed after {self.retries} tries: {url}: {last_error}")

    def search(self, query: str, top_k: int = 5) -> RetrievalResponse:
        if self.mode == "sandbox":
            if not self.sandbox_url:
                raise ValueError("sandbox mode requires --sandbox_url")
            endpoint = self.sandbox_url + "/execute"
            env = (
                "import requests\n\n"
                "def wikipedia_search(query: str, top_n: int = 5):\n"
                f"    url = {repr(self.retriever_url + '/search')}\n"
                "    if query == '':\n"
                "        return 'invalid query'\n"
                "    response = requests.post(url, json={'query': query, 'top_n': top_n})\n"
                "    response.raise_for_status()\n"
                "    retrieval_text = ''\n"
                "    for line in response.json():\n"
                "        retrieval_text += f\"{line['contents']}\\n\\n\"\n"
                "    return retrieval_text.strip()\n"
            )
            call = f"wikipedia_search(query={query!r}, top_n={int(top_k)})"
            raw, latency = self._post(endpoint, {"env": env, "call": call, "timeout": self.timeout})
            result_value = str(raw.get("result", ""))
            result_text = "result: \n" + result_value.strip()
            documents = documents_from_trace({}, result_text)
            return RetrievalResponse(query, documents, result_text, raw, endpoint, latency)

        endpoint = self.retriever_url + "/search"
        raw, latency = self._post(endpoint, {"query": query, "top_n": int(top_k), "return_score": False})
        doc_values = raw
        if (
            isinstance(raw, list)
            and len(raw) == 2
            and isinstance(raw[0], list)
            and isinstance(raw[1], list)
            and (not raw[1] or isinstance(raw[1][0], (int, float)))
        ):
            doc_values = raw[0]
        if not isinstance(doc_values, list):
            raise ValueError(f"Unexpected retriever response: {type(raw).__name__}")
        documents = []
        for index, item in enumerate(doc_values):
            if isinstance(item, dict):
                documents.append(
                    make_document(
                        first_nonempty(item, ("id", "doc_id", "key"), ""),
                        first_nonempty(item, ("contents", "content", "text"), ""),
                        first_nonempty(item, ("title", "doc_title"), ""),
                    )
                )
            else:
                documents.append(make_document("", item, f"document_{index}"))
        retrieval_text = "\n\n".join(doc["contents"] for doc in documents).strip()
        result_text = "result: \n" + retrieval_text
        return RetrievalResponse(query, documents, result_text, raw, endpoint, latency)


def public_model_meta(raw: Any) -> Any:
    """Keep useful model metadata without duplicating potentially huge text fields."""
    item = raw[0] if isinstance(raw, list) and raw else raw
    if not isinstance(item, dict):
        return item
    output = {key: value for key, value in item.items() if key not in {"text", "choices"}}
    if isinstance(item.get("choices"), list):
        output["choices"] = []
        for choice in item["choices"]:
            if not isinstance(choice, dict):
                continue
            summary = {
                key: value
                for key, value in choice.items()
                if key not in {"message", "text", "delta"}
            }
            message = choice.get("message")
            if isinstance(message, dict):
                summary["message_metadata"] = {
                    key: value
                    for key, value in message.items()
                    if key not in {"content", "reasoning", "reasoning_content", "reasoning_details"}
                }
            output["choices"].append(summary)
    return output


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def pstdev(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    center = mean(values)
    return math.sqrt(sum((value - center) ** 2 for value in values) / len(values))
