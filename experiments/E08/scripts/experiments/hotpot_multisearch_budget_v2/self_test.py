#!/usr/bin/env python3
"""Offline protocol tests; no model, retriever, network, or gold leakage."""

from __future__ import annotations

import tempfile
from pathlib import Path

from actions import generate_action, parse_action
from analyze_hotpotqa import score_record
from hp_common import (
    ExactTokenizer, Generation, SQLiteRecordStore, SafeSGLangClient,
    requested_tokens_by_rank,
)
from prompts import decision_prompt, initial_search_prompt
from retrieval import render_evidence


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in str(text)]

    def decode(self, ids, **kwargs):
        return "".join(chr(item) for item in ids)


class FakeClient:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def generate(self, prompt, maximum, stops, temperature=0.0):
        self.calls += 1
        return Generation(
            text=self.text, raw_text=self.text, raw={"text": self.text},
            finish_reason={"type": "length", "length": maximum},
            latency_seconds=0.0, endpoint="fake", stop_repair="none",
        )


def main() -> int:
    assert requested_tokens_by_rank(50) == [50] + [0] * 9
    assert requested_tokens_by_rank(150) == [150] + [0] * 9
    assert requested_tokens_by_rank(200) == [150, 50] + [0] * 8
    assert requested_tokens_by_rank(1500) == [150] * 10
    assert parse_action("<search>alpha beta</search>", ["search"])["payload"] == "alpha beta"
    assert parse_action("<answer>Paris</answer>", ["search", "answer"])["action"] == "answer"
    assert parse_action("<answer>Paris", ["answer"])["payload"] == "Paris"
    repaired, mode = SafeSGLangClient._complete_stop(
        "<answer>Paris</answer", {"matched": "</answer>"}, ["</answer>"]
    )
    assert repaired == "<answer>Paris</answer>" and mode.startswith("completed_prefix")
    repaired, _ = SafeSGLangClient._complete_stop(
        "<answer>Paris</", {"matched": "</answer>"}, ["</answer>"]
    )
    assert repaired == "<answer>Paris</answer>"
    tokenizer = ExactTokenizer(tokenizer=CharacterTokenizer())
    fake = FakeClient("<think>still considering</think>")
    failed = generate_action(fake, tokenizer, "prompt", ["search", "answer"], 8192)
    assert fake.calls == 1 and failed["parse_mode"] == "failed"
    documents = [
        {"id": str(i), "contents": f'"Title {i}"\n' + (f"sentence {i} " * 30),
         "title": f"Title {i}", "contents_sha256": str(i)}
        for i in range(1, 11)
    ]
    view = render_evidence(tokenizer, documents, 200, 1)
    assert len(view["visible_passages"]) == 2
    assert view["actual_tokens_by_rank"][:2] == [150, 50]
    assert sum(view["actual_tokens_by_rank"]) == 200
    assert "/no_think" not in initial_search_prompt("question")
    assert "/no_think" not in decision_prompt(
        "question", [{"search_round": 1, "query": "q", "evidence": view}], 3
    )
    trajectory = {
        "sample_id": "x", "condition_id": "b0200", "evidence_budget_per_search": 200,
        "final_answer": "Paris", "search_calls": 1,
        "visible_evidence_tokens_total": 200,
        "usage": {"standalone_model_calls": 2, "standalone_llm_input_tokens": 10,
                  "standalone_llm_output_tokens": 4,
                  "continuation_llm_input_tokens": 8,
                  "continuation_llm_output_tokens": 3,
                  "standalone_format_repair_calls": 0},
        "answer_status": "valid", "context_compactions": 0,
        "search_rounds": [{"search_round": 1, "query": "Paris France", "evidence": {
            "retrieved_docs": [{"title": "France"}, {"title": "Paris"}],
            "visible_passages": [{"title": "France", "visible_text": "Paris is the capital of France."},
                                 {"title": "Paris", "visible_text": "Paris is a city."}],
        }}],
    }
    label = {
        "gold_answers": ["Paris"], "type": "bridge", "level": "hard",
        "supporting_titles": ["France", "Paris"],
        "supporting_facts": [
            {"title": "France", "sent_id": 0, "sentence": "Paris is the capital of France."},
            {"title": "Paris", "sent_id": 0, "sentence": "Paris is a city."},
        ],
    }
    score = score_record(trajectory, label)
    assert score["answer_em"] == 1 and score["support_sentence_strict_all"]
    assert score["cumulative_retrieved_all_titles"]
    with tempfile.TemporaryDirectory() as temporary:
        store = SQLiteRecordStore(Path(temporary) / "state.sqlite")
        store.put_record("a", {"sample_id": "a", "value": 1})
        assert store.completed_keys() == {"a"}
        assert list(store.iter_records())[0]["value"] == 1
        store.export(temporary)
        store.close()
    print("SELF-TEST PASSED: budgets, Top-10 slicing, one-call action, stop repair, native thinking, "
          "Hotpot evidence labels, compressed resume store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
