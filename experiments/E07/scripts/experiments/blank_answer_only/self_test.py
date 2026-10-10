#!/usr/bin/env python3
"""Offline protocol tests; no model server, GPU, or retriever is required."""

from __future__ import annotations

from types import SimpleNamespace

from common import ModelResponse, canonical_json_hash
from evidence_views import ALL_CONDITIONS, build_evidence_view
from merge_shards import validate_nestedness, validate_record
from offline_lexical_hit import label_record, token_sequence_contains
from run_answer_only import run_job, strict_answer_parse


class FakeTokenizer:
    def encode(self, text):
        return str(text).split()

    def decode(self, ids):
        return " ".join(ids)

    def count(self, text):
        return len(self.encode(text))


class FakeModel:
    def __init__(self, output):
        self.output = output
        self.calls = 0

    def generate(self, prompt, temperature, maximum, stop, seed):
        self.calls += 1
        return ModelResponse(
            self.output,
            {"text": self.output, "meta_info": {
                "finish_reason": {"type": "stop", "matched": "</answer>"},
                "prompt_tokens": len(prompt.split()), "completion_tokens": 8,
            }},
            "http://fake/generate", 0.25,
            {"type": "stop", "matched": "</answer>"},
        )


def main() -> int:
    tokenizer = FakeTokenizer()
    valid = strict_answer_parse(
        "<think>reason</think>\n<answer>James Madison</answer>",
        {"type": "stop", "matched": "</answer>"}, tokenizer,
    )
    assert valid["valid"] and valid["final_answer"] == "James Madison"
    malformed = (
        "<think>x</think><answer>Voyager 2</</answer>",
        "<think>x</think><answer>1824</</answer>",
        "<think>x</think><answer>x</answer</answer>",
        "<think>x</think>free text <answer>x</answer>",
        "<think>x</think><answer></answer>",
    )
    for output in malformed:
        assert not strict_answer_parse(
            output, {"type": "stop", "matched": "</answer>"}, tokenizer
        )["valid"], output
    assert not strict_answer_parse(
        "<think>x</think><answer>x</answer>", {"type": "length"}, tokenizer
    )["valid"]

    documents = [
        {"id": str(rank), "contents": " ".join(
            f"d{rank}_{index}" for index in range(220)
        ), "raw_token_count": 220}
        for rank in range(1, 6)
    ]
    previous = {}
    for condition in ALL_CONDITIONS:
        view = build_evidence_view(tokenizer, documents, condition)
        assert view["requested_tokens_by_rank"] == [
            min(150, max(0, condition.evidence_budget - 150 * index))
            for index in range(5)
        ]
        current = {item["rank"]: item["visible_text"] for item in view["visible_passages"]}
        for rank, text in previous.items():
            assert current[rank].startswith(text)
        previous = current

    root = {
        "sample_id": "test_0", "question": "Who was president in 1812?",
        "gold_answers": ["James Madison"], "root_type": "natural_search",
        "root_origin_detail": "natural_search", "root_query": "president 1812",
        "root_prompt_hash": "p", "root_checkpoint_sha256": "c",
        "retrieval_hash": "r", "root_cache_record_hash": "rc",
        "documents": documents,
    }
    model = FakeModel("<think>reason</think>\n<answer>James Madison</answer>")
    args = SimpleNamespace(
        max_new_tokens=6144, context_length=16384,
        server_max_total_tokens=10086, seed=2026, temperature=0.0,
        model_path="/home/researcher/smartsearch_rebuild/models/Qwen3-8B",
        model_url="http://127.0.0.1:9101",
    )
    record = run_job(root, "b750", 0, model, tokenizer, args)
    assert model.calls == 1
    assert record["model_request_count"] == 1
    assert record["judge_calls"] == 0
    assert record["lexical_evidence_hit_computed"] is False
    assert record["answer_status"] == "valid"
    assert record["em"] == 1.0 and record["f1"] == 1.0
    assert "answer_judge" not in record and "evidence_judge" not in record
    assert validate_record(record) == []
    assert validate_nestedness([record]) == []
    supplied = record["record_sha256"]
    without_hash = dict(record)
    without_hash.pop("record_sha256")
    assert supplied == canonical_json_hash(without_hash)

    invalid_model = FakeModel("<think>reason</think><answer>James Madison</</answer>")
    invalid = run_job(root, "b750", 0, invalid_model, tokenizer, args)
    assert invalid_model.calls == 1
    assert invalid["answer_status"] == "invalid_format"
    assert invalid["final_answer"] == ""
    assert invalid["candidate_answer"] == "James Madison</"
    assert invalid["em"] == 0.0 and invalid["f1"] == 0.0
    assert validate_record(invalid) == []

    lexical = label_record(record)
    assert lexical["model_calls"] == 0
    assert lexical["any_gold_answer_lexical_hit"] is False
    record_with_answer_evidence = dict(record)
    record_with_answer_evidence["visible_passages"] = [
        {"rank": 1, "visible_text": "The president was James Madison."}
    ]
    lexical = label_record(record_with_answer_evidence)
    assert lexical["any_gold_answer_lexical_hit"] is True
    assert lexical["hit_ranks"] == [1]
    assert token_sequence_contains(["pilot"], ["l"]) is False

    print(
        "SELF-TEST PASSED: 15 nested budgets, strict malformed-tag rejection, "
        "one model call, local EM/F1, no judges or evidence-hit computation."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
