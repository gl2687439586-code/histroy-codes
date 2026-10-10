#!/usr/bin/env python3
"""Offline self-test for the Qwen3-only three-condition protocol."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from common import ModelResponse, RetrievalResponse, TokenCounter, answer_metrics, read_jsonl, stable_hash
from prepare_qwen3_roots import enrich_documents
from qwen3_common import assert_qwen_configuration, load_qwen_checkpoints
from qwen3_prompts import (
    COMMON_SYSTEM,
    COMMIT_POLICY,
    NEUTRAL_POLICY,
    READ_POLICY,
    SEARCH_POLICY,
    build_prompt,
    prompt_manifest,
)
from run_action_tree import (
    Runtime,
    extract_flexible_answer,
    extract_flexible_query,
    generate_answer,
    generate_read,
    generate_search,
    make_initial_state,
)
from run_qwen3_natural import run_sample


class TinyCounter:
    approximate = False
    error = ""

    @staticmethod
    def count(text: str) -> int:
        return len(str(text).split())


class FakeModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, temperature, max_new_tokens, stop, seed):
        self.prompts.append(prompt)
        value = self.responses.pop(0)
        if isinstance(value, tuple):
            text, finish_reason = value
        else:
            text, finish_reason = value, {"type": "stop"}
        return ModelResponse(text=text, raw={"fake": True}, endpoint="fake://model", latency_seconds=0.01, finish_reason=finish_reason)


class FakeRetriever:
    def __init__(self):
        self.queries = []

    def search(self, query, top_k=5):
        self.queries.append(query)
        documents = [
            {"id": f"d{i}", "title": f"Doc {i}", "contents": f"evidence {i}", "key": f"d{i}"}
            for i in range(top_k)
        ]
        return RetrievalResponse(query, documents, "result: evidence", documents, "fake://retriever", 0.01)


def args(**overrides):
    values = {
        "fixed_doc_tokens": 0,
        "top_k": 5,
        "action_validation": "flexible",
        "action_attempts": 1,
        "prompt_protocol": "qwen3_matched",
        "search_max_new_tokens": 64,
        "read_max_new_tokens": 64,
        "answer_max_new_tokens": 64,
        "answer_repair_max_new_tokens": 32,
        "context_length": 4096,
        "temperature": 0.0,
        "base_seed": 2026,
        "checkpoint_format": "qwen3",
        "model_path": "/home/researcher/models/Qwen3-8B",
        "model_url": "http://127.0.0.1:9001",
        "max_steps": 3,
        "max_new_tokens": 64,
        "seed": 2026,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def sample_and_root():
    prefix = (
        "<|im_start|>system\nbase<|im_end|>\n<|im_start|>user\nQ?<|im_end|>\n"
        "<|im_start|>assistant\n<think>plan</think><search>q1</search>\n"
        "<result>result: evidence\n</result>\n"
    )
    docs = [{"id": "d0", "title": "Doc", "contents": "evidence", "key": "d0"}]
    sample = {
        "sample_id": "test_0",
        "question": "Q?",
        "gold_answers": ["A"],
        "checkpoint_prefix": prefix,
        "initial_query": "q1",
        "initial_result": "result: evidence",
        "initial_documents": docs,
        "checkpoint_source": "qwen3_blank.first_natural_search",
        "root_origin": "qwen_natural_root",
        "root_checkpoint_sha1": stable_hash(prefix, 40),
        "blank_record_sha1": "blanksha",
        "horizon": 3,
        "original_search_count": 1,
    }
    root = {
        **sample,
        "model_name": "qwen3-8b",
        "model_path": "/home/researcher/models/Qwen3-8B",
        "model_url": "http://127.0.0.1:9001",
        "old_checkpoint_used": False,
        "blank_prediction": "A",
        "blank_metrics": {"f1": 1.0},
        "blank_search_calls": 1,
        "root_generation_event": {
            "action": "Search",
            "action_code": "S",
            "documents": docs,
            "llm_input_tokens": 3,
            "llm_output_tokens": 2,
            "retrieved_docs": 1,
            "retrieved_tokens_total": 1,
            "new_evidence_tokens": 1,
        },
    }
    return sample, root


def main() -> int:
    # Configuration fail-closed checks.
    assert_qwen_configuration("/home/researcher/models/Qwen3-8B", "http://127.0.0.1:9001")
    for model_path, model_url in (
        ("/home/researcher/models/SmartSearch-3B", "http://127.0.0.1:9001"),
        ("/home/researcher/models/Qwen3-8B", "http://127.0.0.1:9000"),
    ):
        try:
            assert_qwen_configuration(model_path, model_url)
            raise AssertionError("contaminated configuration was accepted")
        except RuntimeError:
            pass

    # Prompt match: common text and state rendering are identical; only policy differs.
    documents = [{"id": "1", "title": "T", "contents": "E", "key": "1"}]
    search = build_prompt("Q", ["q"], documents, ["n"], SEARCH_POLICY)
    neutral = build_prompt("Q", ["q"], documents, ["n"], NEUTRAL_POLICY)
    assert COMMON_SYSTEM in search and COMMON_SYSTEM in neutral
    assert search.replace(SEARCH_POLICY, "<POLICY>") == neutral.replace(NEUTRAL_POLICY, "<POLICY>")
    assert prompt_manifest()["policies"]["EvidenceDeepening"] == READ_POLICY

    sample, root_row = sample_and_root()
    runtime = Runtime(args=args(), tokenizer=TinyCounter(), model=FakeModel([]), retriever=FakeRetriever())
    parent = make_initial_state(runtime, sample)
    original_keys = set(parent.document_keys)

    runtime.model = FakeModel(["Candidates: A; no new observation."])
    read = generate_read(runtime, sample["sample_id"], sample["question"], 0, parent, 1)
    assert runtime.retriever.queries == []
    assert read.document_keys == original_keys
    assert read.events[-1]["evidence_set_unchanged"] is True
    assert READ_POLICY in read.events[-1]["raw_prompt"]

    # A model-written Search/Answer inside Read is preserved raw but converted
    # to inert notes before it becomes state for the next decision.
    runtime.model = FakeModel(["<think>Candidate A.</think><search>q2</search><answer>A</answer>"])
    sanitized_read = generate_read(runtime, sample["sample_id"], sample["question"], 0, parent, 1)
    assert "<search>" not in sanitized_read.events[-1]["accepted_trace_append"].lower()
    assert "<answer>" not in sanitized_read.events[-1]["accepted_trace_append"].lower()
    assert sanitized_read.document_keys == original_keys

    runtime.model = FakeModel(["<search>q2</search>"])
    search_state = generate_search(runtime, sample["sample_id"], sample["question"], 0, parent, 1)
    assert runtime.retriever.queries == ["q2"]
    assert SEARCH_POLICY in search_state.events[-1]["raw_prompt"]
    recovered_query, recovered_source = extract_flexible_query(
        "<think>The retrieval request should be U.S. president in 1812.", sample["question"]
    )
    assert recovered_query == "U.S. president in 1812"
    assert recovered_source == "reasoning_query_recovery"
    assert extract_flexible_query("<search>Voyager 2 launch vehicle", "fallback") == (
        "Voyager 2 launch vehicle",
        "open_search_tag",
    )
    assert extract_flexible_answer("<answer>Titan IIIE/Centaur") == (
        "Titan IIIE/Centaur",
        "open_answer_tag",
    )

    runtime.model = FakeModel(["<answer>A</answer>"])
    answer = generate_answer(runtime, sample, 0, parent, True, 1)
    assert answer["final_answer"] == "A"
    assert COMMIT_POLICY in answer["events"][-1]["raw_prompt"]
    assert answer["old_checkpoint_used"] is False
    assert answer_metrics(r"\text{Titan IIIE/Centaur}", ["Titan IIIE"])["f1"] == 0.8

    runtime.model = FakeModel(
        [
            ("<think>unfinished reasoning without an answer", {"type": "length", "length": 64}),
            "<answer>A</answer>",
        ]
    )
    repaired = generate_answer(runtime, sample, 0, parent, True, 1)
    repair = repaired["events"][-1]
    assert repaired["final_answer"] == "A"
    assert repair["terminal_completed"] is True
    assert repair["terminal_answer_repair_attempted"] is True
    assert repair["terminal_answer_repair_succeeded"] is True

    # Neutral begins at the byte-identical exported root and performs one
    # naturally selected final answer without retrieval.
    neutral_model = FakeModel(["<answer>A</answer>"])
    neutral_retriever = FakeRetriever()
    neutral_record = run_sample(
        {"sample_id": "test_0", "question": "Q?", "gold_answers": ["A"]},
        "neutral",
        neutral_model,
        neutral_retriever,
        TinyCounter(),
        args(),
        root_row,
    )
    assert neutral_record["root_checkpoint_sha1"] == root_row["root_checkpoint_sha1"]
    assert neutral_record["search_calls"] == 1  # shared root only
    assert neutral_retriever.queries == []
    assert NEUTRAL_POLICY in neutral_model.prompts[0]

    # Loader rejects a row that claims a different model.
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "roots.jsonl"
        good = dict(root_row)
        bad = dict(root_row, sample_id="bad", model_name="smartsearch-3b")
        path.write_text(
            json.dumps(good, ensure_ascii=False) + "\n" + json.dumps(bad, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        loaded, report = load_qwen_checkpoints(path)
        assert [row["sample_id"] for row in loaded] == ["test_0"]
        assert any("rejected non-Qwen" in warning for warning in report["warnings"])

    print("QWEN3 PROTOCOL SELF-TEST PASSED: model guard, matched prompts, root identity, Search/Read/Commit invariants.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
