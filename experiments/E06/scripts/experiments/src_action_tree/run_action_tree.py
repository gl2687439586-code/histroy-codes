#!/usr/bin/env python3
"""Run variable-horizon SmartSearch Search-Read-Commit action trees.

The immutable first Search checkpoint is copied from the original SmartSearch
run. Each sample receives a total action horizon in [3, 6], normally derived
from its original Search count. Search and Read states are generated once per
prefix and shared by all descendants, preserving byte-identical pre-fork state.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    ModelClient,
    RetrieverClient,
    TokenCounter,
    answer_metrics,
    append_jsonl,
    extract_final_answer,
    extract_last_tag,
    extract_think_text,
    load_experiment_samples,
    normalized_query,
    public_model_meta,
    read_jsonl,
    stable_hash,
    strip_latex_answer_wrappers,
    utc_stamp,
)
from qwen3_common import (  # noqa: E402
    assert_qwen_configuration,
    enforce_server_identity,
    load_qwen_checkpoints,
)
from openrouter_common import (  # noqa: E402
    EXPECTED_MODEL_ID as OPENROUTER_MODEL_ID,
    PROTOCOL_VERSION as OPENROUTER_PROTOCOL_VERSION,
    assert_openrouter_configuration,
    enforce_model_identity as enforce_openrouter_identity,
    load_openrouter_checkpoints,
    openrouter_runtime_config,
    response_usage as openrouter_response_usage,
)
from openrouter_prompts import (  # noqa: E402
    write_prompt_manifest as write_openrouter_prompt_manifest,
)
from qwen3_prompts import (  # noqa: E402
    COMMIT_POLICY,
    FORCED_ANSWER_POLICY,
    PROTOCOL_VERSION as QWEN3_PROTOCOL_VERSION,
    READ_POLICY,
    SEARCH_POLICY,
    SHORT_ANSWER_REPAIR_POLICY,
    build_prompt as build_qwen3_prompt,
    read_notes_from_events,
    write_prompt_manifest,
)


def expected_paths(horizon: int) -> Tuple[str, ...]:
    """All terminal paths for a fixed initial S and an S/R/C horizon."""
    if horizon < 2:
        raise ValueError("horizon must be >= 2")
    output: List[str] = []

    def visit(prefix: str) -> None:
        if len(prefix) >= horizon:
            output.append(prefix)
            return
        visit(prefix + "S")
        visit(prefix + "R")
        output.append(prefix + "C")

    visit("S")
    return tuple(output)


# Backward-compatible export used by old scripts and v3 result directories.
EXPECTED_PATHS = expected_paths(3)


def sample_horizon(sample: Dict[str, Any], args: argparse.Namespace) -> int:
    if args.horizon_mode == "fixed":
        raw = args.fixed_horizon
    else:
        observed = int(sample.get("original_search_count", 0) or 0)
        if observed == 0 and sample.get("root_origin") in {
            "qwen_forced_root",
            "openrouter_forced_root",
        }:
            raw = getattr(args, "zero_search_horizon", args.min_horizon)
        else:
            raw = observed or args.min_horizon
    return max(args.min_horizon, min(args.max_horizon, int(raw)))


def select_samples(
    samples: Sequence[Dict[str, Any]],
    limit: int,
    strategy: str,
    seed: int,
) -> List[Dict[str, Any]]:
    if not limit or len(samples) <= limit:
        return list(samples)
    if strategy == "first":
        return list(samples[:limit])
    if strategy == "random":
        # Stable, order-independent sampling guards the 20-question probe
        # against accidental dataset ordering effects.
        ordered = sorted(samples, key=lambda sample: str(sample["sample_id"]))
        return random.Random(seed).sample(ordered, limit)

    # Deterministic round-robin coverage of each observed horizon. This is a
    # stress/coverage sample, not a population-proportion estimate.
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for sample in samples:
        groups.setdefault(int(sample["horizon"]), []).append(sample)
    selected: List[Dict[str, Any]] = []
    offsets = {horizon: 0 for horizon in groups}
    while len(selected) < limit:
        progressed = False
        for horizon in sorted(groups):
            index = offsets[horizon]
            if index < len(groups[horizon]):
                selected.append(groups[horizon][index])
                offsets[horizon] += 1
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
    return selected


class ActionGenerationError(RuntimeError):
    """An assigned action could not be formatted after controlled retries."""

    def __init__(
        self,
        message: str,
        *,
        action: str,
        decision_index: int,
        parent_actions: Sequence[str],
        attempts: Sequence[Dict[str, Any]],
    ):
        super().__init__(message)
        self.action = action
        self.decision_index = decision_index
        self.parent_actions = list(parent_actions)
        self.attempts = list(attempts)


SEARCH_INSTRUCTION = """[COUNTERFACTUAL ACTION OVERRIDE: SEARCH]
This is a controlled rollout. Continue searching for NEW external evidence.

Requirements:
1. Do not answer the question yet.
2. Generate exactly one new Wikipedia search query.
3. The query must not repeat or merely copy any previously used query listed below.
4. Base the query on the current counterfactual state, including any Evidence Deepening already performed.
5. Target information that is not yet sufficiently established.

Previously used queries:
{queries}

Return exactly this structure and nothing after it:
<think>brief planning for a genuinely new query</think>
<search>new query</search>

The assistant response is already prefixed with <think>. Continue inside that
block, close it with </think>, and then emit exactly one <search> block."""


READ_INSTRUCTION = """[COUNTERFACTUAL ACTION OVERRIDE: EVIDENCE DEEPENING / READ]
Do not call Search and do not give the final answer. Re-read and deeply process ONLY the evidence already present in the conversation. No new Observation may be acquired.

Explicitly organize the evidence into:
- candidate entities or answers;
- satisfied constraints from the original question;
- unsatisfied constraints;
- relations that connect evidence across hops;
- conflicts or contradictions;
- remaining unknowns and the most useful next information need.

This is an evidence-processing action, not a final-answer action. The assistant
response is already prefixed with <think>; continue inside it and close it with
</think>. Do not emit another opening <think> tag."""


COMMIT_INSTRUCTION = """[COUNTERFACTUAL ACTION OVERRIDE: COMMIT]
Stop information acquisition now. Do not call Search. Using only the evidence already available, integrate the evidence and give the best final answer.

Return exactly:
<think>concise final integration</think>
<answer>The final answer is \\[ \\boxed{short exact answer} \\]</answer>

The assistant response is already prefixed with <think>. Continue inside that
block, close it, and then emit one complete <answer> block. The Search tool is
disabled. Even if evidence is incomplete, commit to the best supported short
answer instead of searching or refusing."""


FORCED_ANSWER_INSTRUCTION = """[HORIZON REACHED: FORCED ANSWER]
The controlled action horizon has been exhausted. Do not call Search again. Using only the current counterfactual state, integrate the evidence and give the best final answer now.

Return exactly:
<think>concise final integration</think>
<answer>The final answer is \\[ \\boxed{short exact answer} \\]</answer>

The assistant response is already prefixed with <think>. Continue inside that
block, close it, and then emit one complete <answer> block. The Search tool is
disabled. Even if evidence is incomplete, commit to the best supported short
answer instead of searching or refusing."""


FLEXIBLE_SEARCH_INSTRUCTION = """[ASSIGNED BEHAVIOR: SEARCH]
Spend this action on external retrieval. State one useful Wikipedia retrieval
request for the original question. It may introduce a new angle, verify an
existing fact, or repeat a useful information need. Do not give the final
answer in place of the retrieval request.

Formatting is flexible: a <search> block, JSON query field, `Search query:`
line, or a short plain-text request are all acceptable. The experiment
controller will execute exactly one retriever call even if no explicit query
can be extracted."""


FLEXIBLE_READ_INSTRUCTION = """[ASSIGNED BEHAVIOR: EVIDENCE DEEPENING / READ]
Spend this action processing only the evidence already available. No retriever
or other external tool is available. Reconsider candidates, constraints,
cross-document relations, conflicts, and unknowns. You may write freely and
may identify a likely answer, but do not request a new Observation. The
controller will preserve the evidence set exactly."""


FLEXIBLE_COMMIT_INSTRUCTION = """[ASSIGNED BEHAVIOR: COMMIT]
Stop acquiring information and give the best final answer supported by the
fixed evidence. Search is unavailable. Tags are optional: <answer>, boxed
notation, `Final answer:`, or an ordinary concise answer are all accepted."""


FLEXIBLE_FORCED_ANSWER_INSTRUCTION = """[HORIZON REACHED: FORCED ANSWER]
The action horizon is exhausted. Using only the fixed evidence, give the best
final answer now. Search is unavailable. Tags are optional: <answer>, boxed
notation, `Final answer:`, or an ordinary concise answer are all accepted."""


@dataclass
class RolloutState:
    prompt: str
    actions: List[str]
    events: List[Dict[str, Any]]
    queries: List[str]
    documents: List[Dict[str, Any]]
    document_keys: Set[str]

    def clone(self) -> "RolloutState":
        return RolloutState(
            prompt=self.prompt,
            actions=list(self.actions),
            events=copy.deepcopy(self.events),
            queries=list(self.queries),
            documents=copy.deepcopy(self.documents),
            document_keys=set(self.document_keys),
        )


@dataclass
class Runtime:
    args: argparse.Namespace
    tokenizer: TokenCounter
    model: ModelClient
    retriever: RetrieverClient
    call_counter: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def seed_for(self, sample_id: str, rollout_id: int, label: str) -> int:
        with self.lock:
            self.call_counter += 1
            counter = self.call_counter
        digest = int(stable_hash(f"{sample_id}|{rollout_id}|{label}", 8), 16)
        return int(self.args.base_seed + rollout_id * 100003 + digest + counter) % (2**31 - 1)


def control_prompt(parent_prompt: str, instruction: str, *, flexible: bool = False) -> str:
    """Close the current assistant turn and add a prompt-level intervention."""
    return (
        parent_prompt.rstrip()
        + "\n<|im_end|>\n<|im_start|>user\n"
        + instruction.strip()
        + "\n<|im_end|>\n<|im_start|>assistant\n"
        + ("" if flexible else "<think>")
    )


def qwen3_control_prompt(question: str, parent: "RolloutState", policy: str) -> str:
    """Matched prompt used by every controlled action in the Qwen3 protocol."""
    return build_qwen3_prompt(
        question,
        parent.queries,
        parent.documents,
        read_notes_from_events(parent.events),
        policy,
    )


def uses_matched_prompt(args: argparse.Namespace) -> bool:
    return getattr(args, "prompt_protocol", "legacy") in {
        "qwen3_matched",
        "openrouter_matched",
    }


def model_usage_fields(
    runtime: "Runtime",
    response: Any,
    fallback_input: int,
    fallback_output: int,
    fallback_reasoning: int = 0,
) -> Dict[str, Any]:
    if getattr(runtime.args, "api_mode", "sglang") == "openrouter":
        return openrouter_response_usage(response, fallback_input, fallback_output)
    return {
        "llm_input_tokens": int(fallback_input),
        "llm_output_tokens": int(fallback_output),
        "reasoning_tokens": int(fallback_reasoning),
        "api_cost_usd": 0.0,
        "token_accounting_source": "local_tokenizer",
    }


def terminal_control_prompt(
    question: str,
    parent: RolloutState,
    instruction: str,
    *,
    flexible: bool = False,
) -> str:
    """Build an answer-only view of a fixed counterfactual parent state.

    Continuing the original ReCall conversation leaves its tool-seeking system
    instruction active, which can make a model emit Search even under a Commit
    intervention.  A terminal action instead receives the same already-owned
    evidence and Evidence Deepening notes in a tool-free executor.  No document
    is added and the immutable parent state remains separately hash-audited.
    """
    unique_documents: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for document in parent.documents:
        key = str(document.get("key", "")) or stable_hash(str(document.get("contents", "")), 40)
        if key in seen:
            continue
        seen.add(key)
        unique_documents.append(document)

    evidence_parts: List[str] = []
    for index, document in enumerate(unique_documents, 1):
        title = str(document.get("title", "")).strip()
        contents = str(document.get("contents", "")).strip()
        label = f"Document {index}" + (f" — {title}" if title else "")
        evidence_parts.append(f"[{label}]\n{contents}")
    evidence_text = "\n\n".join(evidence_parts) or "[No non-empty retrieved document text was recorded.]"

    deepening_parts = [
        str(event.get("evidence_deepening", "")).strip()
        for event in parent.events
        if event.get("action_code") == "R" and str(event.get("evidence_deepening", "")).strip()
    ]
    deepening_text = "\n\n".join(deepening_parts) or "[No Evidence Deepening action in this state.]"
    query_text = "\n".join(f"- {query}" for query in parent.queries) or "[None]"

    if flexible:
        system = (
            "You are the terminal answer executor in a controlled information-behavior experiment. "
            "No tools are available. Use only the supplied fixed state and give the best supported "
            "short answer. Output format is flexible; answer tags are optional."
        )
    else:
        system = (
            "You are the terminal answer executor in a controlled information-behavior experiment. "
            "No tools are available in this executor. Never emit <search> or request another observation. "
            "Use the fixed state supplied by the user and commit to the best supported short answer, even "
            "when the state is incomplete. Return one <think> block followed by one <answer> block."
        )
    user = (
        f"{instruction.strip()}\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"SEARCH QUERIES ALREADY EXECUTED:\n{query_text}\n\n"
        f"FIXED RETRIEVED EVIDENCE (NO NEW RETRIEVAL):\n{evidence_text}\n\n"
        f"EVIDENCE DEEPENING NOTES FROM THIS STATE:\n{deepening_text}"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
        + ("" if flexible else "<think>")
    )


def search_control_prompt(
    question: str,
    parent: RolloutState,
    instruction: str,
    *,
    flexible: bool = False,
) -> str:
    """Build a query-only view of the same fixed counterfactual state."""
    unique_documents: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for document in parent.documents:
        key = str(document.get("key", "")) or stable_hash(str(document.get("contents", "")), 40)
        if key in seen:
            continue
        seen.add(key)
        unique_documents.append(document)

    evidence_parts: List[str] = []
    for index, document in enumerate(unique_documents, 1):
        title = str(document.get("title", "")).strip()
        contents = str(document.get("contents", "")).strip()
        label = f"Document {index}" + (f" — {title}" if title else "")
        evidence_parts.append(f"[{label}]\n{contents}")
    evidence_text = "\n\n".join(evidence_parts) or "[No non-empty retrieved document text was recorded.]"

    deepening_parts = [
        str(event.get("evidence_deepening", "")).strip()
        for event in parent.events
        if event.get("action_code") == "R" and str(event.get("evidence_deepening", "")).strip()
    ]
    deepening_text = "\n\n".join(deepening_parts) or "[No Evidence Deepening action in this state.]"
    query_text = "\n".join(f"- {query}" for query in parent.queries) or "[None]"

    if flexible:
        system = (
            "You are the Search-action executor in a controlled information-behavior experiment. "
            "This action always spends one external retrieval call. Suggest a useful Wikipedia request. "
            "Novelty and special tags are optional; the controller, not the model, guarantees Search."
        )
    else:
        system = (
            "You are the Search-action query executor in a controlled information-behavior experiment. "
            "The assigned action is mandatory: produce exactly one NEW Wikipedia search query. Never answer "
            "the original question and never emit <answer>. Even when the current evidence appears sufficient, "
            "search for a distinct corroborating source, a verification angle, or an unresolved related constraint. "
            "The new query must not repeat any previously executed query. Return one <think> block followed by "
            "one <search> block."
        )
    query_history_label = (
        "PREVIOUSLY EXECUTED QUERIES — REUSE IS ALLOWED:"
        if flexible
        else "PREVIOUSLY EXECUTED QUERIES — ALL ARE BANNED:"
    )
    user = (
        f"{instruction.strip()}\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\n"
        f"{query_history_label}\n{query_text}\n\n"
        f"CURRENT FIXED EVIDENCE:\n{evidence_text}\n\n"
        f"EVIDENCE DEEPENING NOTES FROM THIS STATE:\n{deepening_text}"
    )
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
        + ("" if flexible else "<think>")
    )


def check_context(runtime: Runtime, prompt: str, max_new_tokens: int, label: str) -> int:
    input_tokens = runtime.tokenizer.count(prompt)
    if input_tokens + max_new_tokens > runtime.args.context_length:
        raise RuntimeError(
            f"{label}: estimated input ({input_tokens}) + max_new_tokens ({max_new_tokens}) "
            f"exceeds context_length ({runtime.args.context_length}); state was not truncated"
        )
    return input_tokens


def document_token_fields(runtime: Runtime, documents: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int, int]:
    enriched: List[Dict[str, Any]] = []
    actual_total = 0
    cost_total = 0
    for document in documents:
        item = copy.deepcopy(document)
        actual = runtime.tokenizer.count(item.get("contents", ""))
        cost = runtime.args.fixed_doc_tokens if runtime.args.fixed_doc_tokens > 0 else actual
        item["token_count_actual"] = actual
        item["token_count_cost"] = cost
        actual_total += actual
        cost_total += cost
        enriched.append(item)
    return enriched, actual_total, cost_total


def initial_llm_parts(prefix: str) -> Tuple[str, str]:
    """Approximate the original first call's exact input/output boundary.

    ReCall sends the opening ``<think>`` in the prompt, then the server generates
    through ``</search>``. The tool result is not an LLM output.
    """
    assistant = prefix.find("<|im_start|>assistant")
    think = prefix.find("<think>", assistant if assistant >= 0 else 0)
    search_end = prefix.find("</search>", think if think >= 0 else 0)
    if think >= 0:
        split = think + len("<think>")
    elif assistant >= 0:
        split = assistant + len("<|im_start|>assistant")
    else:
        split = 0
    if search_end >= 0:
        generated_end = search_end + len("</search>")
    else:
        generated_end = split
    return prefix[:split], prefix[split:generated_end]


def make_initial_state(runtime: Runtime, sample: Dict[str, Any]) -> RolloutState:
    prefix = sample["checkpoint_prefix"]
    initial_input, initial_output = initial_llm_parts(prefix)
    documents, actual_tokens, cost_tokens = document_token_fields(runtime, sample["initial_documents"])
    unique_keys: Set[str] = set()
    new_actual = 0
    new_cost = 0
    for document in documents:
        if document["key"] not in unique_keys:
            unique_keys.add(document["key"])
            new_actual += document["token_count_actual"]
            new_cost += document["token_count_cost"]

    event = {
        "event_index": 0,
        "decision_index": 0,
        "action": "Search",
        "action_code": "S",
        "fixed_initial_checkpoint": True,
        "checkpoint_source": sample["checkpoint_source"],
        "parent_state_hash": stable_hash(initial_input, 40),
        "post_state_hash": stable_hash(prefix, 40),
        "raw_prompt": initial_input,
        "raw_model_response": initial_output,
        "query": sample["initial_query"],
        "tool": {
            "name": "wikipedia_search",
            "top_k": runtime.args.top_k,
            "raw_result_text": sample["initial_result"],
            "documents": documents,
            "source": "reused_original_checkpoint",
        },
        "llm_input_tokens": runtime.tokenizer.count(initial_input),
        "llm_output_tokens": runtime.tokenizer.count(initial_output),
        "reasoning_tokens": runtime.tokenizer.count(extract_think_text("<think>" + initial_output)),
        "answer_tokens": 0,
        "retrieved_docs": len(documents),
        "retrieved_docs_nominal": runtime.args.top_k,
        "retrieved_tokens_actual": actual_tokens,
        "retrieved_tokens_total": cost_tokens,
        "new_evidence_tokens_actual": new_actual,
        "new_evidence_tokens": new_cost,
        "read_input_tokens": 0,
        "read_output_tokens": 0,
        "latency_seconds": 0.0,
    }
    if getattr(runtime.args, "checkpoint_format", "legacy") == "openrouter":
        root_usage = sample.get("root_generation_event") or {}
        for key in (
            "llm_input_tokens",
            "llm_output_tokens",
            "reasoning_tokens",
            "api_cost_usd",
            "api_model_id",
            "api_provider_id",
            "api_reasoning",
            "api_reasoning_details",
            "api_router_metadata",
            "api_request_messages",
            "token_accounting_source",
            "model_meta",
        ):
            if key in root_usage:
                event[key] = root_usage[key]
    return RolloutState(
        prompt=prefix,
        actions=["S"],
        events=[event],
        queries=[sample["initial_query"]],
        documents=documents,
        document_keys=unique_keys,
    )


def extract_flexible_query(text: str, fallback: str) -> Tuple[str, str]:
    """Extract a useful query without making formatting a branch failure.

    The final fallback is the original question. This deliberately defines a
    Search action by the external retriever call, not by query novelty or tag
    compliance.
    """
    raw = (text or "").strip()
    tagged = extract_last_tag(raw, "search")
    if tagged:
        return re.sub(r"\s+", " ", tagged).strip(), "search_tag"

    # Some serving backends omit the matched stop string from returned text.
    # In that case a valid response ends with an open <search> block even
    # though generation did stop on </search> server-side.
    open_tagged = re.findall(r"<search>\s*([^<\r\n]+?)\s*$", raw, flags=re.I)
    if open_tagged:
        return re.sub(r"\s+", " ", open_tagged[-1]).strip(), "open_search_tag"

    json_match = re.search(r'["\']query["\']\s*:\s*["\']([^"\']+)["\']', raw, flags=re.I)
    if json_match:
        return re.sub(r"\s+", " ", json_match.group(1)).strip(), "json_query"

    label_matches = re.findall(
        r"(?:search\s+query|query|检索词|搜索词)\s*[:：]\s*([^\n<>]+)",
        raw,
        flags=re.I,
    )
    if label_matches:
        return re.sub(r"\s+", " ", label_matches[-1]).strip(" `\"'"), "labeled_query"

    # Qwen-style thinking may be followed by one untagged final line.
    think_parts = re.split(r"</think>", raw, maxsplit=1, flags=re.I)
    after_think = think_parts[-1].strip() if len(think_parts) == 2 else ""
    candidate = after_think or raw
    candidate = re.sub(r"^```(?:json|text)?\s*|\s*```$", "", candidate, flags=re.I | re.S).strip()
    lines = [line.strip(" -*`\"'") for line in candidate.splitlines() if line.strip()]
    meta_prefix = re.compile(r"^(?:okay\b|i\s+(?:need|should|will)\b|we\s+(?:need|should)\b|let(?:'|’)s\b)", re.I)
    if (
        len(lines) == 1
        and 1 <= len(lines[0]) <= 300
        and "<think>" not in lines[0].lower()
        and not meta_prefix.match(lines[0])
    ):
        return re.sub(r"\s+", " ", lines[0]).strip(), "plain_text"

    # Thinking-mode Qwen often describes a perfectly usable query but reaches
    # the generation cap before emitting the requested tag. Recover only from
    # explicit query-intent language; keep the untouched raw response for audit.
    query_intent_patterns = (
        r"(?:retrieval request|search query|query)\s+(?:would be|should be|is)\s*[:\-]?\s*[\"']?([^\n\"']{3,220})",
        r"(?:search|look up|retrieve information)\s+(?:for|about|on)\s+[\"']?([^\n\"']{3,220})",
        r"(?:perform|run|do)\s+(?:a\s+)?search\s+(?:for|on)\s+[\"']?([^\n\"']{3,220})",
    )
    inferred: List[str] = []
    for pattern in query_intent_patterns:
        inferred.extend(re.findall(pattern, raw, flags=re.I))
    for value in reversed(inferred):
        candidate_query = re.split(
            r"(?:\.\s+(?:that|this|the|however|but|since|so|let|i|we)\b|</think>|<answer>|<search>)",
            str(value),
            maxsplit=1,
            flags=re.I,
        )[0]
        candidate_query = re.sub(r"\s+", " ", candidate_query).strip(" `\"'.,:;-_")
        candidate_query = re.sub(
            r"^(?:for|about|on|to\s+find|information\s+(?:about|on))\s+",
            "",
            candidate_query,
            flags=re.I,
        ).strip()
        candidate_query = re.sub(r"\s+to\s+(?:confirm|verify|check|solidify)(?:\s+it|\s+this)?\s*$", "", candidate_query, flags=re.I)
        unsafe_meta = re.search(
            r"(?:\bdocuments?\b.*\b(?:don|doesn|already|mention)\b|\bmaybe\b|\bperhaps\b|\bwait\b|\bthe user\b|\bthe policy\b|\bthe answer\b|\bthat\s*,\s*but\b)",
            candidate_query,
            flags=re.I,
        )
        if (
            3 <= len(candidate_query) <= 220
            and len(candidate_query.split()) <= 30
            and not unsafe_meta
            and not candidate_query.lower().endswith((" don", " doesn", " isn", " aren"))
        ):
            return candidate_query, "reasoning_query_recovery"

    return re.sub(r"\s+", " ", fallback).strip(), "fallback_original_question"


def sanitize_read_deepening(text: str) -> str:
    """Convert a tool-free Read response into inert state notes.

    The raw response remains in the event.  Action tags are removed from the
    state append so a model-proposed Search or Answer cannot masquerade as an
    executed action at the next decision.
    """
    raw = str(text or "").strip()
    think = extract_think_text(raw).strip()
    tail_parts = re.split(r"</think>", raw, maxsplit=1, flags=re.I)
    tail = tail_parts[-1].strip() if len(tail_parts) == 2 else ""
    source = "\n\n".join(part for part in (think, tail) if part).strip() or raw
    source = re.sub(r"</?(?:think|search|answer)>\s*", "", source, flags=re.I)
    source = re.sub(r"\n{3,}", "\n\n", source).strip()
    return source


def extract_flexible_answer(text: str) -> Tuple[str, str]:
    """Extract tags, boxed answers, answer cues, or a natural-language reply."""
    raw = (text or "").strip()
    answer_block = extract_last_tag(raw, "answer")
    if answer_block:
        return extract_final_answer(raw).strip(), "answer_tag"

    # Accept an answer whose closing tag was consumed as the server stop
    # sequence. A length-truncated open tag is still rejected later by the
    # terminal-completeness check.
    open_answer = re.findall(r"<answer>\s*([^<\r\n]+?)\s*$", raw, flags=re.I)
    if open_answer:
        return strip_latex_answer_wrappers(open_answer[-1]).strip(), "open_answer_tag"

    boxed = re.search(r"\\boxed\s*\{", raw)
    if boxed:
        return extract_final_answer(raw).strip(), "boxed"

    think_parts = re.split(r"</think>", raw, maxsplit=1, flags=re.I)
    tail = think_parts[-1].strip() if len(think_parts) == 2 else raw
    cue_matches = re.findall(
        r"(?:the\s+final\s+answer\s+is|final\s+answer|answer)\s*[:：]?\s*(.+)",
        tail,
        flags=re.I,
    )
    if cue_matches:
        return cue_matches[-1].strip(), "answer_cue"

    cleaned = re.sub(r"</?(?:think|answer)>\s*", "", tail, flags=re.I).strip()
    if cleaned:
        return cleaned, "natural_language"
    return "", "empty"


def finish_reason_is_length(value: Any) -> bool:
    if isinstance(value, dict):
        return str(value.get("type", "")).lower() == "length"
    return str(value or "").lower() == "length"


def terminal_answer_is_complete(candidate: str, source: str, finish_reason: Any, generated: str) -> bool:
    if not str(candidate or "").strip():
        return False
    if not finish_reason_is_length(finish_reason):
        return True
    if source in {"answer_tag", "boxed"}:
        return True
    return bool(re.search(r"</think>", generated or "", flags=re.I))


def generate_search(
    runtime: Runtime,
    sample_id: str,
    question: str,
    rollout_id: int,
    parent: RolloutState,
    decision_index: int,
) -> RolloutState:
    flexible = runtime.args.action_validation == "flexible"
    previous_norm = {normalized_query(query) for query in parent.queries}
    attempts: List[Dict[str, Any]] = []
    accepted_prompt = ""
    accepted_response = ""
    accepted_query = ""
    accepted_meta: Dict[str, Any] = {}
    accepted_latency = 0.0
    accepted_query_source = ""
    accepted_repeated = False
    accepted_usage: Dict[str, Any] = {}

    for attempt in range(runtime.args.action_attempts):
        query_lines = "\n".join(f"- {query}" for query in parent.queries)
        instruction = FLEXIBLE_SEARCH_INSTRUCTION if flexible else SEARCH_INSTRUCTION.format(queries=query_lines)
        if attempt and not flexible:
            previous = attempts[-1]
            previous_output = str(previous.get("raw_model_response", ""))[-1200:]
            instruction += (
                f"\n\nFORMAT RECOVERY ATTEMPT {attempt + 1}: The previous output failed because "
                f"`{previous.get('reason', 'invalid')}`. Do not reproduce it. "
                "Choose a different information need and produce one syntactically valid new <search> query. "
                "Remember that the response is already prefixed by <think>; close that block and emit one "
                f"<search>...</search> block.\nPREVIOUS INVALID OUTPUT:\n{previous_output}"
            )
        prompt = (
            qwen3_control_prompt(question, parent, SEARCH_POLICY)
            if uses_matched_prompt(runtime.args)
            else search_control_prompt(question, parent, instruction, flexible=flexible)
        )
        input_tokens = check_context(runtime, prompt, runtime.args.search_max_new_tokens, "Search")
        seed = runtime.seed_for(sample_id, rollout_id, f"search-d{decision_index}-a{attempt}")
        response = runtime.model.generate(
            prompt=prompt,
            temperature=runtime.args.temperature,
            max_new_tokens=runtime.args.search_max_new_tokens,
            stop=("</search>",) if flexible else ("</search>", "</answer>"),
            seed=seed,
        )
        full_generated = response.text if flexible else "<think>" + response.text
        if flexible:
            query, query_source = extract_flexible_query(full_generated, question)
        else:
            query = extract_last_tag(full_generated, "search")
            query_source = "search_tag" if query else "missing"
        normalized = normalized_query(query)
        repeated = bool(normalized and normalized in previous_norm)
        valid = bool(query and normalized) if flexible else bool(
            query and normalized and not repeated and "<answer>" not in response.text.lower()
        )
        reason = "accepted" if valid else (
            "missing_search" if not query else "reused_query" if repeated else "unexpected_answer"
        )
        attempt_usage = model_usage_fields(
            runtime,
            response,
            input_tokens,
            runtime.tokenizer.count(response.text),
            runtime.tokenizer.count(
                extract_think_text(
                    response.text if uses_matched_prompt(runtime.args) else "<think>" + response.text
                )
            ),
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "seed": seed,
                "valid": valid,
                "reason": reason,
                "query_candidate": query,
                "query_source": query_source,
                "repeated_query": repeated,
                "raw_prompt": prompt,
                "raw_model_response": response.text,
                "model_meta": public_model_meta(response.raw),
                "finish_reason": response.finish_reason,
                "input_tokens": input_tokens,
                "output_tokens": runtime.tokenizer.count(response.text),
                "latency_seconds": response.latency_seconds,
                "usage": attempt_usage,
            }
        )
        if valid:
            accepted_prompt = prompt
            accepted_response = response.text
            accepted_query = query.strip()
            accepted_meta = public_model_meta(response.raw)
            accepted_latency = response.latency_seconds
            accepted_query_source = query_source
            accepted_repeated = repeated
            accepted_usage = attempt_usage
            break
    if not accepted_query:
        raise ActionGenerationError(
            f"Search action failed after {runtime.args.action_attempts} attempts; "
            f"previous queries={parent.queries!r}; reasons={[item['reason'] for item in attempts]}",
            action="Search",
            decision_index=decision_index,
            parent_actions=parent.actions,
            attempts=attempts,
        )

    retrieval = runtime.retriever.search(accepted_query, runtime.args.top_k)
    documents, actual_tokens, cost_tokens = document_token_fields(runtime, retrieval.documents)
    new_actual = 0
    new_cost = 0
    next_keys = set(parent.document_keys)
    for document in documents:
        if document["key"] not in next_keys:
            next_keys.add(document["key"])
            new_actual += document["token_count_actual"]
            new_cost += document["token_count_cost"]

    # Ensure serialized state contains one closed action block. Raw output is
    # preserved separately; this repairs a closing tag consumed by the server
    # stop sequence without duplicating an already complete block.
    if accepted_query_source == "search_tag":
        canonical_search = ""
    elif accepted_query_source == "open_search_tag":
        canonical_search = "</search>"
    else:
        canonical_search = f"\n<search>{accepted_query}</search>"
    tool_append = f"{canonical_search}\n<result>{retrieval.result_text}\n</result>\n"
    next_prompt = accepted_prompt + accepted_response + tool_append
    event = {
        "event_index": len(parent.events),
        "decision_index": decision_index,
        "action": "Search",
        "action_code": "S",
        "fixed_initial_checkpoint": False,
        "search_executor": "query_only_fixed_state",
        "action_validation": runtime.args.action_validation,
        "search_input_evidence_document_count": len(parent.document_keys),
        "parent_state_hash": stable_hash(parent.prompt, 40),
        "action_prompt_hash": stable_hash(accepted_prompt, 40),
        "post_state_hash": stable_hash(next_prompt, 40),
        "raw_prompt": accepted_prompt,
        "raw_model_response": accepted_response,
        "accepted_trace_append": accepted_response + tool_append,
        "query": accepted_query,
        "query_source": accepted_query_source,
        "query_was_repeated": accepted_repeated,
        "query_fallback_used": accepted_query_source == "fallback_original_question",
        "model_proposed_answer_during_search": "<answer>" in accepted_response.lower(),
        "attempts": attempts,
        "model_meta": accepted_meta,
        "tool": {
            "name": "wikipedia_search",
            "top_k": runtime.args.top_k,
            "endpoint": retrieval.endpoint,
            "raw_response": retrieval.raw,
            "raw_result_text": retrieval.result_text,
            "documents": documents,
            "latency_seconds": retrieval.latency_seconds,
        },
        **accepted_usage,
        "answer_tokens": 0,
        "retrieved_docs": len(documents),
        "retrieved_docs_nominal": runtime.args.top_k,
        "retrieved_tokens_actual": actual_tokens,
        "retrieved_tokens_total": cost_tokens,
        "new_evidence_tokens_actual": new_actual,
        "new_evidence_tokens": new_cost,
        "read_input_tokens": 0,
        "read_output_tokens": 0,
        "latency_seconds": accepted_latency + retrieval.latency_seconds,
    }
    state = parent.clone()
    state.prompt = next_prompt
    state.actions.append("S")
    state.events.append(event)
    state.queries.append(accepted_query)
    state.documents.extend(documents)
    state.document_keys = next_keys
    return state


def generate_read(
    runtime: Runtime,
    sample_id: str,
    question: str,
    rollout_id: int,
    parent: RolloutState,
    decision_index: int,
) -> RolloutState:
    flexible = runtime.args.action_validation == "flexible"
    attempts: List[Dict[str, Any]] = []
    accepted_prompt = ""
    accepted_response = ""
    accepted_meta: Dict[str, Any] = {}
    accepted_latency = 0.0
    accepted_deepening = ""
    accepted = False
    accepted_usage: Dict[str, Any] = {}
    for attempt in range(runtime.args.action_attempts):
        instruction = FLEXIBLE_READ_INSTRUCTION if flexible else READ_INSTRUCTION
        if attempt and not flexible:
            previous = attempts[-1]
            previous_output = str(previous.get("raw_model_response", ""))[-1200:]
            instruction += (
                f"\n\nFORMAT RECOVERY ATTEMPT {attempt + 1}: The previous output failed because "
                f"`{previous.get('reason', 'invalid')}`. Output only evidence analysis in the already-open "
                f"<think> block, then close it; no Search and no answer.\nPREVIOUS INVALID OUTPUT:\n{previous_output}"
            )
        prompt = (
            qwen3_control_prompt(question, parent, READ_POLICY)
            if uses_matched_prompt(runtime.args)
            else control_prompt(parent.prompt, instruction, flexible=flexible)
        )
        input_tokens = check_context(runtime, prompt, runtime.args.read_max_new_tokens, "Read")
        seed = runtime.seed_for(sample_id, rollout_id, f"read-d{decision_index}-a{attempt}")
        response = runtime.model.generate(
            prompt=prompt,
            temperature=runtime.args.temperature,
            max_new_tokens=runtime.args.read_max_new_tokens,
            stop=() if flexible else ("</think>", "</search>", "</answer>"),
            seed=seed,
        )
        generated = response.text if flexible else "<think>" + response.text
        deepening = sanitize_read_deepening(generated)
        lower = response.text.lower()
        # In flexible mode the controller guarantees Read by withholding all
        # tools. Format or a model-written candidate answer cannot change the
        # assigned external behavior.
        valid = True if flexible else bool(deepening and "<search>" not in lower and "<answer>" not in lower)
        reason = "accepted" if valid else "forbidden_action_or_missing_think"
        attempt_usage = model_usage_fields(
            runtime,
            response,
            input_tokens,
            runtime.tokenizer.count(response.text),
            runtime.tokenizer.count(
                extract_think_text(
                    response.text if uses_matched_prompt(runtime.args) else "<think>" + response.text
                )
            ),
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "seed": seed,
                "valid": valid,
                "reason": reason,
                "raw_prompt": prompt,
                "raw_model_response": response.text,
                "model_meta": public_model_meta(response.raw),
                "finish_reason": response.finish_reason,
                "input_tokens": input_tokens,
                "output_tokens": runtime.tokenizer.count(response.text),
                "latency_seconds": response.latency_seconds,
                "usage": attempt_usage,
            }
        )
        if valid:
            accepted_prompt = prompt
            accepted_response = response.text
            accepted_meta = public_model_meta(response.raw)
            accepted_latency = response.latency_seconds
            accepted_deepening = deepening
            accepted_usage = attempt_usage
            accepted = True
            break
    if not accepted:
        raise ActionGenerationError(
            f"Read action failed after {runtime.args.action_attempts} attempts",
            action="EvidenceDeepening",
            decision_index=decision_index,
            parent_actions=parent.actions,
            attempts=attempts,
        )

    read_input = int(accepted_usage.get("llm_input_tokens", runtime.tokenizer.count(accepted_prompt)))
    read_output = int(accepted_usage.get("llm_output_tokens", runtime.tokenizer.count(accepted_response)))
    event = {
        "event_index": len(parent.events),
        "decision_index": decision_index,
        "action": "EvidenceDeepening",
        "action_code": "R",
        "action_validation": runtime.args.action_validation,
        "parent_state_hash": stable_hash(parent.prompt, 40),
        "action_prompt_hash": stable_hash(accepted_prompt, 40),
        "post_state_hash": stable_hash(accepted_prompt + accepted_deepening, 40),
        "raw_prompt": accepted_prompt,
        "raw_model_response": accepted_response,
        "accepted_trace_append": accepted_deepening,
        "evidence_deepening": accepted_deepening,
        "model_requested_search_during_read": "<search>" in accepted_response.lower(),
        "model_proposed_answer_during_read": "<answer>" in accepted_response.lower(),
        "attempts": attempts,
        "model_meta": accepted_meta,
        **accepted_usage,
        "answer_tokens": 0,
        "retrieved_docs": 0,
        "retrieved_docs_nominal": 0,
        "retrieved_tokens_actual": 0,
        "retrieved_tokens_total": 0,
        "new_evidence_tokens_actual": 0,
        "new_evidence_tokens": 0,
        "read_input_tokens": read_input,
        "read_output_tokens": read_output,
        "latency_seconds": accepted_latency,
        "evidence_keys_before": sorted(parent.document_keys),
        "evidence_keys_after": sorted(parent.document_keys),
        "evidence_set_unchanged": True,
    }
    state = parent.clone()
    state.prompt = accepted_prompt + accepted_deepening
    state.actions.append("R")
    state.events.append(event)
    return state


def generate_answer(
    runtime: Runtime,
    sample: Dict[str, Any],
    rollout_id: int,
    parent: RolloutState,
    commit: bool,
    decision_index: int,
) -> Dict[str, Any]:
    flexible = runtime.args.action_validation == "flexible"
    if flexible:
        instruction = FLEXIBLE_COMMIT_INSTRUCTION if commit else FLEXIBLE_FORCED_ANSWER_INSTRUCTION
    else:
        instruction = COMMIT_INSTRUCTION if commit else FORCED_ANSWER_INSTRUCTION
    attempts: List[Dict[str, Any]] = []
    accepted_prompt = ""
    accepted_response = ""
    accepted_meta: Dict[str, Any] = {}
    accepted_latency = 0.0
    accepted_final_answer = ""
    accepted_answer_source = ""
    accepted_finish_reason: Any = None
    accepted = False
    accepted_usage: Dict[str, Any] = {}
    for attempt in range(runtime.args.action_attempts):
        current_instruction = instruction
        if attempt and not flexible:
            previous = attempts[-1]
            previous_output = str(previous.get("raw_model_response", ""))[-1200:]
            current_instruction += (
                f"\n\nFORMAT RECOVERY ATTEMPT {attempt + 1}: The previous output failed because "
                f"`{previous.get('reason', 'invalid')}`. Search is disabled. Continue after the already-open "
                "<think>, close it, and include one complete <answer>...</answer> block now. "
                f"Commit to the best short answer.\nPREVIOUS INVALID OUTPUT:\n{previous_output}"
            )
        prompt = (
            qwen3_control_prompt(
                sample["question"],
                parent,
                COMMIT_POLICY if commit else FORCED_ANSWER_POLICY,
            )
            if uses_matched_prompt(runtime.args)
            else terminal_control_prompt(sample["question"], parent, current_instruction, flexible=flexible)
        )
        input_tokens = check_context(runtime, prompt, runtime.args.answer_max_new_tokens, "Answer")
        seed = runtime.seed_for(sample["sample_id"], rollout_id, f"answer-d{decision_index}-c{int(commit)}-a{attempt}")
        response = runtime.model.generate(
            prompt=prompt,
            temperature=runtime.args.temperature,
            max_new_tokens=runtime.args.answer_max_new_tokens,
            stop=() if flexible else ("</answer>",),
            seed=seed,
        )
        generated = response.text if flexible else "<think>" + response.text
        answer_block = extract_last_tag(generated, "answer")
        if flexible:
            final_candidate, answer_source = extract_flexible_answer(generated)
            # A successful model call always completes the assigned Commit.
            # Empty text is retained as an empty prediction (quality 0) rather
            # than deleting all seven counterfactual paths for the sample.
            valid = True
        else:
            final_candidate = extract_final_answer(generated)
            answer_source = "answer_tag" if answer_block else "missing"
            valid = bool(answer_block and "<search>" not in response.text.lower())
        attempt_usage = model_usage_fields(
            runtime,
            response,
            input_tokens,
            runtime.tokenizer.count(response.text),
            runtime.tokenizer.count(extract_think_text(generated)),
        )
        attempts.append(
            {
                "attempt": attempt + 1,
                "seed": seed,
                "valid": valid,
                "reason": "accepted" if valid else "missing_answer_or_forbidden_search",
                "answer_candidate": final_candidate,
                "answer_source": answer_source,
                "raw_prompt": prompt,
                "raw_model_response": response.text,
                "model_meta": public_model_meta(response.raw),
                "finish_reason": response.finish_reason,
                "input_tokens": input_tokens,
                "output_tokens": runtime.tokenizer.count(response.text),
                "latency_seconds": response.latency_seconds,
                "usage": attempt_usage,
            }
        )
        if valid:
            accepted_prompt = prompt
            accepted_response = response.text
            accepted_meta = public_model_meta(response.raw)
            accepted_latency = response.latency_seconds
            accepted_final_answer = final_candidate
            accepted_answer_source = answer_source
            accepted_finish_reason = response.finish_reason
            accepted_usage = attempt_usage
            accepted = True
            break
    if not accepted:
        raise ActionGenerationError(
            f"Answer generation failed after {runtime.args.action_attempts} attempts",
            action="Commit" if commit else "ForcedAnswer",
            decision_index=decision_index,
            parent_actions=parent.actions,
            attempts=attempts,
        )

    generated = accepted_response if flexible else "<think>" + accepted_response
    answer_candidate = accepted_final_answer if flexible else extract_final_answer(generated)
    answer_block = extract_last_tag(generated, "answer")
    finish_was_length = finish_reason_is_length(accepted_finish_reason)
    closed_think = bool(re.search(r"</think>", generated, flags=re.I))
    terminal_completed = terminal_answer_is_complete(
        answer_candidate,
        accepted_answer_source,
        accepted_finish_reason,
        generated,
    )
    repair_event: Dict[str, Any] = {}
    if not terminal_completed and uses_matched_prompt(runtime.args):
        # Preserve the original truncated generation, then retry from the exact
        # same immutable parent state with a tiny answer-only policy.  This is a
        # format repair, not another Search/Read decision.
        repair_prompt = qwen3_control_prompt(sample["question"], parent, SHORT_ANSWER_REPAIR_POLICY)
        check_context(runtime, repair_prompt, runtime.args.answer_repair_max_new_tokens, "AnswerRepair")
        repair_seed = runtime.seed_for(
            sample["sample_id"], rollout_id, f"answer-repair-d{decision_index}-c{int(commit)}"
        )
        repair_response = runtime.model.generate(
            prompt=repair_prompt,
            temperature=runtime.args.temperature,
            max_new_tokens=runtime.args.answer_repair_max_new_tokens,
            stop=("</answer>",),
            seed=repair_seed,
        )
        repair_generated = repair_response.text
        repair_candidate, repair_source = extract_flexible_answer(repair_generated)
        repair_completed = terminal_answer_is_complete(
            repair_candidate,
            repair_source,
            repair_response.finish_reason,
            repair_generated,
        )
        repair_usage = model_usage_fields(
            runtime,
            repair_response,
            runtime.tokenizer.count(repair_prompt),
            runtime.tokenizer.count(repair_response.text),
            runtime.tokenizer.count(extract_think_text(repair_response.text)),
        )
        repair_event = {
            "attempted": True,
            "raw_prompt": repair_prompt,
            "raw_model_response": repair_response.text,
            "answer_candidate": repair_candidate,
            "answer_source": repair_source,
            "terminal_completed": repair_completed,
            "finish_reason": repair_response.finish_reason,
            "model_meta": public_model_meta(repair_response.raw),
            **repair_usage,
            "latency_seconds": repair_response.latency_seconds,
        }
        if repair_completed:
            answer_candidate = repair_candidate
            accepted_answer_source = f"repair:{repair_source}"
            terminal_completed = True
    # Keep the terminal path even when generation is truncated, but score an
    # incomplete thought as an empty answer instead of leaking gold mentions
    # from chain-of-thought into ACC/contains metrics.
    final_answer = answer_candidate if terminal_completed else ""
    controller_answer_block = "" if answer_block and not repair_event else f"\n<answer>{final_answer}</answer>"
    combined_input = int(accepted_usage.get("llm_input_tokens", 0)) + int(
        repair_event.get("llm_input_tokens", 0)
    )
    combined_output = int(accepted_usage.get("llm_output_tokens", 0)) + int(
        repair_event.get("llm_output_tokens", 0)
    )
    combined_reasoning = int(accepted_usage.get("reasoning_tokens", 0)) + int(
        repair_event.get("reasoning_tokens", 0)
    )
    combined_api_cost = float(accepted_usage.get("api_cost_usd", 0.0)) + float(
        repair_event.get("api_cost_usd", 0.0)
    )
    event = {
        "event_index": len(parent.events),
        "decision_index": decision_index,
        "action": "Commit" if commit else "ForcedAnswer",
        "action_code": "C" if commit else "A",
        "action_validation": runtime.args.action_validation,
        "terminal": True,
        "parent_state_hash": stable_hash(parent.prompt, 40),
        "action_prompt_hash": stable_hash(accepted_prompt, 40),
        "post_state_hash": stable_hash(accepted_prompt + accepted_response + controller_answer_block, 40),
        "forced_by_horizon": not commit,
        "terminal_executor": "tool_free_fixed_evidence",
        "terminal_evidence_document_count": len(parent.document_keys),
        "raw_prompt": accepted_prompt,
        "raw_model_response": accepted_response,
        "accepted_trace_append": accepted_response + controller_answer_block,
        "answer_extraction_source": accepted_answer_source,
        "answer_candidate_before_validity_filter": answer_candidate,
        "terminal_completed": terminal_completed,
        "terminal_finish_was_length": finish_was_length,
        "terminal_closed_think": closed_think,
        "terminal_answer_repair": repair_event,
        "terminal_answer_repair_attempted": bool(repair_event),
        "terminal_answer_repair_succeeded": bool(repair_event.get("terminal_completed", False)),
        "controller_answer_block_added": bool(controller_answer_block),
        "model_requested_search_during_answer": "<search>" in accepted_response.lower(),
        "attempts": attempts,
        "model_meta": accepted_meta,
        "llm_input_tokens": combined_input,
        "llm_output_tokens": combined_output,
        "reasoning_tokens": combined_reasoning,
        "api_cost_usd": combined_api_cost,
        "api_model_id": accepted_usage.get("api_model_id", ""),
        "api_provider_id": accepted_usage.get("api_provider_id", ""),
        "api_reasoning": accepted_usage.get("api_reasoning", ""),
        "api_reasoning_details": accepted_usage.get("api_reasoning_details"),
        "api_router_metadata": accepted_usage.get("api_router_metadata"),
        "api_request_messages": accepted_usage.get("api_request_messages"),
        "token_accounting_source": accepted_usage.get("token_accounting_source", "local_tokenizer"),
        "answer_tokens": runtime.tokenizer.count(final_answer),
        "retrieved_docs": 0,
        "retrieved_docs_nominal": 0,
        "retrieved_tokens_actual": 0,
        "retrieved_tokens_total": 0,
        "new_evidence_tokens_actual": 0,
        "new_evidence_tokens": 0,
        "read_input_tokens": 0,
        "read_output_tokens": 0,
        "latency_seconds": accepted_latency + float(repair_event.get("latency_seconds", 0.0)),
        "format_has_answer_tag": bool(answer_block),
        "format_has_boxed_answer": bool(re.search(r"\\boxed\s*\{", accepted_response)),
    }
    terminal = parent.clone()
    if commit:
        terminal.actions.append("C")
    terminal.events.append(event)
    terminal.prompt = accepted_prompt + accepted_response + controller_answer_block
    return finalize_trajectory(runtime, sample, rollout_id, terminal, final_answer, generated, commit)


def finalize_trajectory(
    runtime: Runtime,
    sample: Dict[str, Any],
    rollout_id: int,
    state: RolloutState,
    final_answer: str,
    raw_answer_generation: str,
    committed: bool,
) -> Dict[str, Any]:
    path = "".join(state.actions)
    metrics = answer_metrics(final_answer, sample["gold_answers"])
    sums = {
        "search_calls": sum(1 for event in state.events if event["action_code"] == "S"),
        "read_calls": sum(1 for event in state.events if event["action_code"] == "R"),
        "search_query_fallback_calls": sum(
            1 for event in state.events if event.get("action_code") == "S" and event.get("query_fallback_used")
        ),
        "repeated_search_query_calls": sum(
            1 for event in state.events if event.get("action_code") == "S" and event.get("query_was_repeated")
        ),
        "retrieved_docs_total": sum(int(event.get("retrieved_docs", 0)) for event in state.events),
        "retrieved_docs_nominal": sum(int(event.get("retrieved_docs_nominal", 0)) for event in state.events),
        "retrieved_docs_unique": len(state.document_keys),
        "retrieved_tokens_total": sum(int(event.get("retrieved_tokens_total", 0)) for event in state.events),
        "retrieved_tokens_actual": sum(int(event.get("retrieved_tokens_actual", 0)) for event in state.events),
        "new_evidence_tokens": sum(int(event.get("new_evidence_tokens", 0)) for event in state.events),
        "new_evidence_tokens_actual": sum(int(event.get("new_evidence_tokens_actual", 0)) for event in state.events),
        "read_input_tokens": sum(int(event.get("read_input_tokens", 0)) for event in state.events),
        "read_output_tokens": sum(int(event.get("read_output_tokens", 0)) for event in state.events),
        "reasoning_tokens": sum(int(event.get("reasoning_tokens", 0)) for event in state.events),
        "answer_tokens": sum(int(event.get("answer_tokens", 0)) for event in state.events),
        "total_llm_input_tokens": sum(int(event.get("llm_input_tokens", 0)) for event in state.events),
        "total_llm_output_tokens": sum(int(event.get("llm_output_tokens", 0)) for event in state.events),
        "api_cost_usd": sum(float(event.get("api_cost_usd", 0.0)) for event in state.events),
        "latency_seconds": sum(float(event.get("latency_seconds", 0.0)) for event in state.events),
    }
    sums["llm_tokens_total"] = sums["total_llm_input_tokens"] + sums["total_llm_output_tokens"]
    # Unified proxy: model input + model output + external evidence acquisition.
    # All constituent fields remain available for alternative cost definitions.
    sums["total_cost_tokens"] = sums["llm_tokens_total"] + sums["retrieved_tokens_total"]
    sums["total_cost"] = float(sums["total_cost_tokens"])
    terminal_event = state.events[-1] if state.events else {}

    return {
        "schema_version": (
            OPENROUTER_PROTOCOL_VERSION
            if getattr(runtime.args, "checkpoint_format", "legacy") == "openrouter"
            else QWEN3_PROTOCOL_VERSION
            if getattr(runtime.args, "checkpoint_format", "legacy") == "qwen3"
            else "src-action-tree-v4-variable-horizon"
        ),
        "created_at": utc_stamp(),
        "sample_id": sample["sample_id"],
        "rollout_id": rollout_id,
        "question": sample["question"],
        "gold_answers": sample["gold_answers"],
        "horizon": int(sample["horizon"]),
        "original_search_count": int(sample.get("original_search_count", 1)),
        "expected_terminal_paths": len(expected_paths(int(sample["horizon"]))),
        "path": path,
        "action_sequence": list(state.actions),
        "committed": committed,
        "forced_answer": not committed,
        "final_answer": final_answer,
        "raw_answer_generation": raw_answer_generation,
        "terminal_completed": bool(terminal_event.get("terminal_completed", False)),
        "terminal_finish_was_length": bool(terminal_event.get("terminal_finish_was_length", False)),
        "answer_extraction_source": terminal_event.get("answer_extraction_source", ""),
        "metrics": metrics,
        **metrics,
        **sums,
        "queries": list(state.queries),
        "document_keys_unique": sorted(state.document_keys),
        "events": state.events,
        "checkpoint_source": sample["checkpoint_source"],
        "checkpoint_prefix": sample["checkpoint_prefix"],
        "checkpoint_sha1": stable_hash(sample["checkpoint_prefix"], 40),
        "raw_final_prompt_trace": state.prompt,
        "token_count_is_approximate": runtime.tokenizer.approximate,
        "fixed_doc_tokens": runtime.args.fixed_doc_tokens,
        "action_validation": runtime.args.action_validation,
        "prompt_protocol": getattr(runtime.args, "prompt_protocol", "legacy"),
        "model_name": (
            OPENROUTER_MODEL_ID
            if getattr(runtime.args, "checkpoint_format", "legacy") == "openrouter"
            else "qwen3-8b"
            if getattr(runtime.args, "checkpoint_format", "legacy") == "qwen3"
            else "legacy"
        ),
        "model_path": getattr(runtime.args, "model_path", "legacy"),
        "model_url": getattr(runtime.args, "model_url", "legacy"),
        "old_checkpoint_used": getattr(runtime.args, "checkpoint_format", "legacy") == "legacy",
        "openrouter_runtime": (
            openrouter_runtime_config()
            if getattr(runtime.args, "checkpoint_format", "legacy") == "openrouter"
            else None
        ),
        "root_origin": sample.get("root_origin", "legacy"),
        "blank_record_sha1": sample.get("blank_record_sha1", ""),
        "root_checkpoint_sha1": sample.get("root_checkpoint_sha1", stable_hash(sample["checkpoint_prefix"], 40)),
    }


def run_one_tree(runtime: Runtime, sample: Dict[str, Any], rollout_id: int) -> List[Dict[str, Any]]:
    root = make_initial_state(runtime, sample)
    horizon = int(sample["horizon"])
    wanted = expected_paths(horizon)
    results: Dict[str, Dict[str, Any]] = {}

    def expand(parent: RolloutState) -> None:
        decision_index = len(parent.actions)
        if decision_index >= horizon:
            raise AssertionError(f"Cannot expand terminal prefix {''.join(parent.actions)} at H={horizon}")

        # Each S/R child is generated exactly once and recursively shared by
        # every terminal descendant beneath that child.
        search_state = generate_search(
            runtime,
            sample["sample_id"],
            sample["question"],
            rollout_id,
            parent,
            decision_index=decision_index,
        )
        read_state = generate_read(
            runtime,
            sample["sample_id"],
            sample["question"],
            rollout_id,
            parent,
            decision_index=decision_index,
        )

        for child in (search_state, read_state):
            child_path = "".join(child.actions)
            if len(child.actions) >= horizon:
                results[child_path] = generate_answer(
                    runtime,
                    sample,
                    rollout_id,
                    child,
                    commit=False,
                    decision_index=horizon,
                )
            else:
                expand(child)

        commit_record = generate_answer(
            runtime,
            sample,
            rollout_id,
            parent,
            commit=True,
            decision_index=decision_index,
        )
        results[commit_record["path"]] = commit_record

    expand(root)
    if set(results) != set(wanted):
        raise AssertionError(
            f"Internal path mismatch at H={horizon}: missing={sorted(set(wanted)-set(results))}, "
            f"extra={sorted(set(results)-set(wanted))}"
        )
    return [results[path] for path in wanted]


def completed_tree_keys(path: Path) -> Set[Tuple[str, int]]:
    by_tree: Dict[Tuple[str, int], Set[str]] = {}
    horizons: Dict[Tuple[str, int], int] = {}
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                key = (str(row["sample_id"]), int(row["rollout_id"]))
                by_tree.setdefault(key, set()).add(str(row["path"]))
                horizons[key] = int(row.get("horizon", 3))
            except Exception:
                continue
    return {
        key
        for key, paths in by_tree.items()
        if paths == set(expected_paths(horizons.get(key, 3)))
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intermediate_data", required=True)
    parser.add_argument("--trace_jsonl", default="")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_url", default="http://127.0.0.1:9000")
    parser.add_argument("--retriever_url", default="http://127.0.0.1:6008")
    parser.add_argument("--sandbox_url", default="http://127.0.0.1:6006")
    parser.add_argument("--model_path", default="/home/researcher/smartsearch_workspace/models/SmartSearch-3B")
    parser.add_argument("--api_mode", choices=("sglang", "openai", "openrouter"), default="sglang")
    parser.add_argument(
        "--tokenizer_path",
        default="",
        help="Local tokenizer for document-cost compatibility; defaults to model_path",
    )
    parser.add_argument("--retriever_mode", choices=("direct", "sandbox"), default="direct")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--n_rollouts", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_ids", default="", help="Comma-separated IDs; empty means all")
    parser.add_argument("--horizon_mode", choices=("original_searches", "fixed"), default="original_searches")
    parser.add_argument("--min_horizon", type=int, default=3)
    parser.add_argument("--max_horizon", type=int, default=6)
    parser.add_argument("--fixed_horizon", type=int, default=3)
    parser.add_argument(
        "--zero_search_horizon",
        type=int,
        default=3,
        help="Tree horizon for forced-root samples whose Blank Search count is zero",
    )
    parser.add_argument(
        "--sample_strategy",
        choices=("random", "first", "stratified_horizon"),
        default="random",
        help="random is the primary unbiased probe; stratified_horizon is coverage-only, not prevalence",
    )
    parser.add_argument("--sample_seed", type=int, default=2026)
    parser.add_argument("--max_workers", type=int, default=1)
    parser.add_argument("--base_seed", type=int, default=2026)
    parser.add_argument("--context_length", type=int, default=32768)
    parser.add_argument("--search_max_new_tokens", type=int, default=512)
    parser.add_argument("--read_max_new_tokens", type=int, default=768)
    parser.add_argument("--answer_max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--answer_repair_max_new_tokens",
        type=int,
        default=96,
        help="Answer-only retry budget used only when a terminal generation is incomplete",
    )
    parser.add_argument("--action_attempts", type=int, default=4)
    parser.add_argument(
        "--action_validation",
        choices=("flexible", "strict"),
        default="flexible",
        help=(
            "flexible defines actions by controller-side tool access and accepts natural model output; "
            "strict requires SmartSearch tags and novel Search queries"
        ),
    )
    parser.add_argument("--model_timeout", type=float, default=300.0)
    parser.add_argument("--retriever_timeout", type=float, default=300.0)
    parser.add_argument("--request_retries", type=int, default=3)
    parser.add_argument("--max_usd", type=float, default=0.0, help="Fail after this tree-process spend; 0 disables")
    parser.add_argument("--fixed_doc_tokens", type=int, default=0, help="0=count actual tokenizer tokens")
    parser.add_argument("--approximate_tokens", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry_failures",
        action="store_true",
        help="Retry incomplete tree keys already recorded in failures.jsonl",
    )
    parser.add_argument(
        "--allow_reconstructed_checkpoint",
        action="store_true",
        help="Permit trace-field reconstruction when the exact original prompt is absent (not recommended for formal runs)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Validate and inspect inputs without network calls")
    parser.add_argument(
        "--checkpoint_format",
        choices=("legacy", "qwen3", "openrouter"),
        default="legacy",
        help="Read first-Search checkpoints exported by the selected model's Blank condition",
    )
    parser.add_argument(
        "--prompt_protocol",
        choices=("legacy", "qwen3_matched", "openrouter_matched"),
        default="legacy",
        help="qwen3_matched uses the audited common template and action-specific policy field",
    )
    parser.add_argument("--allow_unverified_server", action="store_true")
    parser.add_argument(
        "--skip_previous_failures",
        action="store_true",
        help="On resume, skip sample/rollout keys already recorded in failures.jsonl",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.checkpoint_format == "qwen3":
        assert_qwen_configuration(args.model_path, args.model_url)
        if args.prompt_protocol != "qwen3_matched":
            raise ValueError("Qwen3-only formal trees require --prompt_protocol qwen3_matched")
    if args.checkpoint_format == "openrouter":
        assert_openrouter_configuration(
            args.model_path, args.model_url, require_key=not args.dry_run
        )
        if args.api_mode != "openrouter":
            raise ValueError("OpenRouter formal trees require --api_mode openrouter")
        if args.prompt_protocol != "openrouter_matched":
            raise ValueError("OpenRouter formal trees require --prompt_protocol openrouter_matched")
    if args.n_rollouts < 1:
        raise ValueError("--n_rollouts must be >= 1")
    if not (3 <= args.min_horizon <= args.max_horizon <= 6):
        raise ValueError("Require 3 <= min_horizon <= max_horizon <= 6")
    if not (args.min_horizon <= args.fixed_horizon <= args.max_horizon):
        raise ValueError("--fixed_horizon must lie within [min_horizon, max_horizon]")
    if not (args.min_horizon <= args.zero_search_horizon <= args.max_horizon):
        raise ValueError("--zero_search_horizon must lie within [min_horizon, max_horizon]")
    if args.top_k != 5:
        print(f"WARNING: experimental specification uses top_k=5; received top_k={args.top_k}", file=sys.stderr)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    terminal_path = output_dir / "terminal_trajectories.jsonl"
    failure_path = output_dir / "failures.jsonl"

    requested_sample_ids = [part.strip() for part in args.sample_ids.split(",") if part.strip()]
    # Formal runs require exact checkpoints. Load the candidate pool first so
    # that rejecting reconstructed rows does not silently turn probe20 into a
    # smaller experiment.
    loader_limit = 0
    sample_loader = (
        load_openrouter_checkpoints
        if args.checkpoint_format == "openrouter"
        else load_qwen_checkpoints
        if args.checkpoint_format == "qwen3"
        else load_experiment_samples
    )
    samples, load_report = sample_loader(
        args.intermediate_data,
        args.trace_jsonl or None,
        limit=loader_limit,
        sample_ids=requested_sample_ids,
    )
    if args.checkpoint_format == "legacy" and not args.allow_reconstructed_checkpoint:
        reconstructed = [sample["sample_id"] for sample in samples if sample["checkpoint_source"] == "trace_reconstructed"]
        samples = [sample for sample in samples if sample["checkpoint_source"] != "trace_reconstructed"]
        load_report["reconstructed_checkpoints_rejected"] = reconstructed
        load_report["loaded_samples_after_exact_checkpoint_requirement"] = len(samples)
    for sample in samples:
        sample["horizon"] = sample_horizon(sample, args)
    load_report["loaded_samples_before_final_limit"] = len(samples)
    samples = select_samples(samples, args.limit, args.sample_strategy, args.sample_seed)
    load_report["loaded_samples_selected"] = len(samples)
    load_report["selected_sample_ids"] = [sample["sample_id"] for sample in samples]
    load_report["selected_horizon_distribution"] = {
        str(horizon): sum(int(sample["horizon"]) == horizon for sample in samples)
        for horizon in range(args.min_horizon, args.max_horizon + 1)
    }
    tokenizer = TokenCounter(
        args.tokenizer_path or args.model_path,
        force_approximate=args.approximate_tokens,
    )
    load_report["token_count_is_approximate"] = tokenizer.approximate
    load_report["tokenizer_error"] = tokenizer.error
    (output_dir / "load_report.json").write_text(
        json.dumps(load_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.prompt_protocol == "openrouter_matched":
        write_openrouter_prompt_manifest(output_dir / "prompt_manifest.json")
    elif args.prompt_protocol == "qwen3_matched":
        write_prompt_manifest(output_dir / "prompt_manifest.json")
    if not samples:
        raise RuntimeError("No runnable samples were loaded; inspect load_report.json")

    path_counts_by_sample = {
        sample["sample_id"]: len(expected_paths(int(sample["horizon"]))) for sample in samples
    }
    horizon_distribution = {
        str(horizon): sum(int(sample["horizon"]) == horizon for sample in samples)
        for horizon in range(args.min_horizon, args.max_horizon + 1)
    }
    resume_signature = {
        "schema_version": (
            OPENROUTER_PROTOCOL_VERSION
            if args.checkpoint_format == "openrouter"
            else QWEN3_PROTOCOL_VERSION
            if args.checkpoint_format == "qwen3"
            else "src-action-tree-v4-variable-horizon"
        ),
        "selected_sample_ids": [sample["sample_id"] for sample in samples],
        "sample_horizons": {
            sample["sample_id"]: int(sample["horizon"]) for sample in samples
        },
        "n_rollouts": args.n_rollouts,
        "model_url": args.model_url,
        "model_path": args.model_path,
        "api_mode": args.api_mode,
        "tokenizer_path": args.tokenizer_path,
        "retriever_url": args.retriever_url,
        "retriever_mode": args.retriever_mode,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "action_validation": args.action_validation,
        "search_max_new_tokens": args.search_max_new_tokens,
        "read_max_new_tokens": args.read_max_new_tokens,
        "answer_max_new_tokens": args.answer_max_new_tokens,
        "answer_repair_max_new_tokens": args.answer_repair_max_new_tokens,
        "context_length": args.context_length,
        "fixed_doc_tokens": args.fixed_doc_tokens,
        "checkpoint_format": args.checkpoint_format,
        "prompt_protocol": args.prompt_protocol,
        "openrouter_runtime": (
            openrouter_runtime_config()
            if args.checkpoint_format == "openrouter"
            else None
        ),
    }
    manifest = {
        "schema_version": (
            OPENROUTER_PROTOCOL_VERSION
            if args.checkpoint_format == "openrouter"
            else QWEN3_PROTOCOL_VERSION
            if args.checkpoint_format == "qwen3"
            else "src-action-tree-v4-variable-horizon"
        ),
        "created_at": utc_stamp(),
        "arguments": vars(args),
        "resume_signature": resume_signature,
        "horizon_definition": (
            "fixed_horizon" if args.horizon_mode == "fixed"
            else "clamp(selected model Blank natural Search count, min_horizon, max_horizon)"
            if args.checkpoint_format in {"qwen3", "openrouter"}
            else "clamp(original SmartSearch Search count, min_horizon, max_horizon)"
        ),
        "horizon_assignment_is_pre_rollout": True,
        "horizon_assignment_note": (
            "Original Search count is measured before counterfactual generation; "
            "adaptive-horizon strata may differ in original difficulty"
        ),
        "horizon_distribution": horizon_distribution,
        "sample_horizon_assignments": {
            sample["sample_id"]: {
                "horizon": int(sample["horizon"]),
                "original_search_count": int(sample.get("original_search_count", 1)),
                "from_final_response": int(
                    sample.get("original_search_count_from_final_response", 0)
                ),
                "from_trace": int(sample.get("original_search_count_from_trace", 0)),
            }
            for sample in samples
        },
        "terminal_paths_per_horizon": {
            str(horizon): len(expected_paths(horizon))
            for horizon in range(args.min_horizon, args.max_horizon + 1)
        },
        "model_calls_per_tree_by_horizon": {
            str(horizon): 2 ** (horizon + 1) - 3
            for horizon in range(args.min_horizon, args.max_horizon + 1)
        },
        "new_retriever_calls_per_tree_by_horizon": {
            str(horizon): 2 ** (horizon - 1) - 1
            for horizon in range(args.min_horizon, args.max_horizon + 1)
        },
        "expected_paths_by_sample": {
            sample["sample_id"]: list(expected_paths(int(sample["horizon"]))) for sample in samples
        },
        "loaded_samples": len(samples),
        "selected_sample_ids": [sample["sample_id"] for sample in samples],
        "expected_terminal_trajectories": sum(path_counts_by_sample.values()) * args.n_rollouts,
        "expected_model_calls": sum(2 ** (int(sample["horizon"]) + 1) - 3 for sample in samples)
        * args.n_rollouts,
        "expected_new_retriever_calls": sum(
            2 ** (int(sample["horizon"]) - 1) - 1 for sample in samples
        )
        * args.n_rollouts,
        "token_count_is_approximate": tokenizer.approximate,
        "cost_definition": "total_cost_tokens = total_llm_input_tokens + total_llm_output_tokens + retrieved_tokens_total",
        "openrouter_runtime": (
            openrouter_runtime_config()
            if args.checkpoint_format == "openrouter"
            else None
        ),
    }
    manifest_path = output_dir / "manifest.json"
    if terminal_path.exists():
        if not args.resume:
            raise FileExistsError(
                f"{terminal_path} already exists; choose a new output_dir or pass --resume"
            )
        if not manifest_path.exists():
            raise RuntimeError(
                "Cannot safely resume: terminal output exists but manifest.json is missing; "
                "use a new RUN_DIR"
            )
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("resume_signature") != resume_signature:
            raise RuntimeError(
                "Cannot safely resume because samples, horizons, model, or generation settings "
                "changed; use a new RUN_DIR"
            )
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.dry_run:
        dry = {
            **manifest,
            "network_calls_made": False,
            "sample_preview": [
                {
                    "sample_id": sample["sample_id"],
                    "question": sample["question"],
                    "gold_answers": sample["gold_answers"],
                    "initial_query": sample["initial_query"],
                    "initial_docs": len(sample["initial_documents"]),
                    "original_search_count": sample.get("original_search_count", 1),
                    "original_search_count_from_final_response": sample.get(
                        "original_search_count_from_final_response", 0
                    ),
                    "original_search_count_from_trace": sample.get(
                        "original_search_count_from_trace", 0
                    ),
                    "horizon": sample["horizon"],
                    "terminal_paths": len(expected_paths(int(sample["horizon"]))),
                    "checkpoint_source": sample["checkpoint_source"],
                    "checkpoint_tokens": tokenizer.count(sample["checkpoint_prefix"]),
                }
                for sample in samples[:20]
            ],
        }
        (output_dir / "dry_run_report.json").write_text(json.dumps(dry, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(dry, ensure_ascii=False, indent=2))
        return 0

    if args.checkpoint_format == "qwen3":
        identity = enforce_server_identity(args.model_url, strict=not args.allow_unverified_server)
        (output_dir / "model_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    elif args.checkpoint_format == "openrouter":
        identity = enforce_openrouter_identity(args.model_url, args.model_path)
        (output_dir / "model_identity.json").write_text(
            json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    model = ModelClient(
        args.model_url,
        args.model_path,
        args.api_mode,
        args.model_timeout,
        args.request_retries,
        max_usd=args.max_usd,
    )
    retriever = RetrieverClient(
        args.retriever_url,
        sandbox_url=args.sandbox_url,
        mode=args.retriever_mode,
        timeout=args.retriever_timeout,
        retries=args.request_retries,
    )
    runtime = Runtime(args=args, tokenizer=tokenizer, model=model, retriever=retriever)
    complete = completed_tree_keys(terminal_path) if args.resume else set()
    previous_failures: Set[Tuple[str, int]] = set()
    if args.resume and args.skip_previous_failures and failure_path.exists():
        for row in read_jsonl(failure_path):
            try:
                previous_failures.add((str(row["sample_id"]), int(row["rollout_id"])))
            except Exception:
                continue
    jobs = [
        (sample, rollout_id)
        for sample in samples
        for rollout_id in range(args.n_rollouts)
        if (sample["sample_id"], rollout_id) not in complete
        and (
            args.retry_failures
            or (sample["sample_id"], rollout_id) not in previous_failures
        )
    ]
    print(
        f"Loaded {len(samples)} samples; pending trees={len(jobs)}; "
        f"horizons={horizon_distribution}; expected terminal records="
        f"{sum(path_counts_by_sample.values()) * args.n_rollouts}."
    )

    succeeded = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_map = {
            executor.submit(run_one_tree, runtime, sample, rollout_id): (sample, rollout_id)
            for sample, rollout_id in jobs
        }
        for future in as_completed(future_map):
            sample, rollout_id = future_map[future]
            try:
                records = future.result()
                append_jsonl(terminal_path, records)
                succeeded += 1
                path_preview = ",".join(row["path"] for row in records[:8])
                if len(records) > 8:
                    path_preview += f",...(+{len(records) - 8})"
                print(
                    f"[ok] {sample['sample_id']} H={sample['horizon']} rollout={rollout_id}: "
                    f"{len(records)} paths [{path_preview}]"
                )
            except Exception as exc:
                failed += 1
                failure_record: Dict[str, Any] = {
                    "created_at": utc_stamp(),
                    "sample_id": sample["sample_id"],
                    "rollout_id": rollout_id,
                    "horizon": sample.get("horizon"),
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
                if isinstance(exc, ActionGenerationError):
                    failure_record.update(
                        {
                            "failed_action": exc.action,
                            "decision_index": exc.decision_index,
                            "parent_actions": exc.parent_actions,
                            "attempts": exc.attempts,
                        }
                    )
                append_jsonl(
                    failure_path,
                    [failure_record],
                )
                print(f"[failed] {sample['sample_id']} rollout={rollout_id}: {exc}", file=sys.stderr)

    complete_total = len(completed_tree_keys(terminal_path))
    expected_total = len(samples) * args.n_rollouts
    stage_complete = complete_total == expected_total
    final_status = {
        "finished_at": utc_stamp(),
        "pending_trees_attempted": len(jobs),
        "succeeded_trees": succeeded,
        "failed_trees": failed,
        "complete_trees_total": complete_total,
        "expected_trees_total": expected_total,
        "stage_complete": stage_complete,
        "exit_code": 0 if stage_complete else 1,
        "terminal_trajectory_file": str(terminal_path),
        "failure_file": str(failure_path),
        "api_cost_usd_this_process": model.process_cost_usd,
        "api_cost_usd_stage_cumulative_ledger": model.cumulative_cost_usd,
        "api_cost_note": (
            "Actual OpenRouter charges observed in this process, including failed/retried generations. "
            "Do not sum per-trajectory API costs to obtain the bill because tree prefixes are shared."
        ),
    }
    (output_dir / "run_status.json").write_text(
        json.dumps(final_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(final_status, ensure_ascii=False, indent=2))
    if not stage_complete:
        print(
            "Tree stage incomplete: refusing downstream analysis. "
            f"complete={complete_total}/{expected_total}. Resume with --retry_failures.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
