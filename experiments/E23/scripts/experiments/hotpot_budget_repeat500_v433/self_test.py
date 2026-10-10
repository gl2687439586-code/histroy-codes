#!/usr/bin/env python3
"""Offline protocol tests; no model, retriever, network, or gold leakage."""

from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

from actions import generate_action, parse_action
from analyze_hotpotqa import main as analyze_main, score_record
from analyze_answer_flips import metric_audit, observed_stable_budget
from hp_common import (
    MAX_BOUNDARY_BACKOFF_TOKENS, TOP_K, ExactTokenizer, Generation,
    SQLiteRecordStore, SafeSGLangClient,
    read_json, write_jsonl,
)
from locate_supporting_evidence import locate_record
from prompts import decision_prompt, initial_search_prompt
from run_trajectories import legacy_root_signature_is_migratable
from retrieval import (
    concatenate_documents, exposure_conditions, largest_roundtrippable_prefix,
    render_evidence,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(char) for char in str(text)]

    def decode(self, ids, **kwargs):
        return "".join(chr(item) for item in ids)


class UnstableBoundaryTokenizer(CharacterTokenizer):
    def decode(self, ids, **kwargs):
        value = super().decode(ids, **kwargs)
        return value + "?" if len(ids) >= 5 else value

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False):
        value = str(text)
        return {"offset_mapping": [(i, i + 1) for i in range(len(value))]}


class ContextMergeTokenizer(CharacterTokenizer):
    def encode(self, text, add_special_tokens=False):
        value = str(text)
        ids = super().encode(value, add_special_tokens=add_special_tokens)
        if value == "abcde":
            ids[-1] = 999999
        return ids

    def __call__(self, text, add_special_tokens=False,
                 return_offsets_mapping=False):
        value = str(text)
        return {"offset_mapping": [(i, i + 1) for i in range(len(value))]}


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
    assert TOP_K == 5, f"Formal v4.3 protocol requires TOP_K=5, got {TOP_K}"
    assert MAX_BOUNDARY_BACKOFF_TOKENS == 16
    script_dir = Path(__file__).resolve().parent
    launcher = (script_dir / "tmux_2gpu.sh").read_text(encoding="utf-8")
    orchestrator = (script_dir / "orchestrate.sh").read_text(encoding="utf-8")
    assert 'GPU_IDS="${GPU_IDS:-2,3}"' in launcher
    assert 'BASE_PORT="${BASE_PORT:-9103}"' in launcher
    assert 'NUM_GPUS="${NUM_GPUS:-2}"' in orchestrator
    assert "NUM_GPUS='$NUM_GPUS'" in launcher
    assert parse_action("<search>alpha beta</search>", ["search"])["payload"] == "alpha beta"
    assert parse_action("<answer>Paris</answer>", ["search", "answer"])["action"] == "answer"
    assert parse_action("<answer>Paris", ["answer"])["payload"] == "Paris"
    long_answer = "A" * 512
    parsed_long_answer = parse_action(
        f"<answer>{long_answer}</answer>", ["search", "answer"]
    )
    assert parsed_long_answer == {"action": "answer", "payload": long_answer}
    assert parse_action(
        f"<search>{'q' * 257}</search>", ["search"]
    ) is None
    legacy_signature = {
        "schema_version": "v", "stage": "trajectories",
        "roots_sha256": "gzip-artifact-a", "top_k": 5,
    }
    semantic_signature = {
        "schema_version": "v", "stage": "trajectories",
        "roots_semantic_sha256": "semantic-b", "top_k": 5,
    }
    assert legacy_root_signature_is_migratable(
        legacy_signature, semantic_signature
    )
    changed_protocol = dict(semantic_signature, top_k=10)
    assert not legacy_root_signature_is_migratable(
        legacy_signature, changed_protocol
    )
    flip_source = (Path(__file__).parent / "analyze_answer_flips.py").read_text(
        encoding="utf-8"
    )
    assert "到1500" not in flip_source
    assert "29个相邻比较" not in flip_source
    repaired, mode = SafeSGLangClient._complete_stop(
        "<answer>Paris</answer", {"matched": "</answer>"}, ["</answer>"]
    )
    assert repaired == "<answer>Paris</answer>" and mode.startswith("completed_prefix")
    repaired, _ = SafeSGLangClient._complete_stop(
        "<answer>Paris</", {"matched": "</answer>"}, ["</answer>"]
    )
    assert repaired == "<answer>Paris</answer>"
    tokenizer = ExactTokenizer(tokenizer=CharacterTokenizer())
    unstable = ExactTokenizer(tokenizer=UnstableBoundaryTokenizer())
    text, ids, actual, backoff = largest_roundtrippable_prefix(
        unstable, unstable.encode("abcdefghij"), "abcdefghij", 5
    )
    assert text == "abcde" and ids == unstable.encode("abcde")
    assert actual == 5 and backoff == 0
    context_merge = ExactTokenizer(tokenizer=ContextMergeTokenizer())
    full_ids = context_merge.encode("abcdefghij")
    text, ids, actual, backoff = largest_roundtrippable_prefix(
        context_merge, full_ids, "abcdefghij", 5
    )
    assert text == "abcd" and ids == full_ids[:4]
    assert actual == 4 and backoff == 1
    fake = FakeClient("<think>still considering</think>")
    failed = generate_action(fake, tokenizer, "prompt", ["search", "answer"], 8192)
    assert fake.calls == 1 and failed["parse_mode"] == "failed"
    documents = [
        {"id": str(i), "contents": f'"Title {i}"\n' + (f"sentence {i} " * 30),
         "title": f"Title {i}", "contents_sha256": str(i)}
        for i in range(1, TOP_K + 1)
    ]
    concatenated = concatenate_documents(tokenizer, documents)
    conditions = exposure_conditions(tokenizer, documents, 100)
    assert [item["root_exposure_tokens"] for item in conditions[:-1]][:2] == [100, 200]
    assert conditions[-1]["exposure_mode"] == "full_per_search"
    assert conditions[-1]["root_exposure_tokens"] == concatenated["full_context_token_count"]
    view = render_evidence(tokenizer, documents, conditions[1], 1)
    assert view["actual_exposed_token_count"] == 200
    assert view["full_context_token_count"] == concatenated["full_context_token_count"]
    assert view["full_context_sha256"] == concatenated["full_context_sha256"]
    assert view["roundtrip_boundary_backoff_tokens"] == 0
    assert "/no_think" not in initial_search_prompt("question")
    assert "/no_think" not in decision_prompt(
        "question", [{"search_round": 1, "query": "q", "evidence": view}], 3
    )
    trajectory = {
        "sample_id": "x", "condition_id": conditions[1]["condition_id"],
        "sample_index": 0, "question": "Which sentence is supporting?",
        "condition_label": "200", "exposure_mode": "token_cap",
        "root_exposure_tokens": 200,
        "root_full_context_tokens": concatenated["full_context_token_count"],
        "evidence_budget_per_search": 200,
        "final_answer": "Paris", "search_calls": 1,
        "top_k_per_search": TOP_K, "retrieved_docs_total": TOP_K,
        "visible_evidence_tokens_total": 200,
        "usage": {"standalone_model_calls": 2, "standalone_llm_input_tokens": 10,
                  "standalone_llm_output_tokens": 4,
                  "continuation_llm_input_tokens": 8,
                  "continuation_llm_output_tokens": 3,
                  "standalone_format_repair_calls": 0},
        "answer_status": "valid", "context_compactions": 0,
        "search_rounds": [{"search_round": 1, "query": "Paris France", "evidence": view}],
    }
    label = {
        "gold_answers": ["Paris"], "type": "bridge", "level": "hard",
        "supporting_titles": ["Title 1"],
        "supporting_facts": [
            {"title": "Title 1", "sent_id": 0, "sentence": "sentence 1"},
        ],
    }
    score = score_record(trajectory, label)
    assert score["answer_em"] == 1 and score["support_sentence_strict_all"]
    assert score["root_actual_exposure_tokens"] == 200
    assert score["roundtrip_boundary_backoff_tokens_total"] == 0
    located = locate_record(tokenizer, trajectory, label)
    occurrence = located["supporting_facts"][0]["occurrences"][0]
    assert occurrence["global_document_number_1based"] == 1
    assert occurrence["document_token_span"]["token_start_1based"] > 0
    assert occurrence["search_context_token_span"]["token_end_inclusive_1based"] <= 200
    fallback_trajectory = dict(trajectory)
    fallback_trajectory["answer_status"] = "local_fallback"
    fallback_score = score_record(fallback_trajectory, label)
    assert fallback_score["answer_em"] == 0
    assert fallback_score["recovered_answer_em"] == 1
    assert observed_stable_budget([50, 100, 150, 200], [0, 1, 0, 1]) == 200
    synthetic_grid = {
        "a": {
            50: {"answer_em": 0, "answer_acc": 0, "answer_f1": 0, "local_fallback": False},
            100: {"answer_em": 1, "answer_acc": 1, "answer_f1": 1, "local_fallback": False},
            150: {"answer_em": 0, "answer_acc": 0, "answer_f1": 0, "local_fallback": False},
        },
        "b": {
            50: {"answer_em": 1, "answer_acc": 1, "answer_f1": 1, "local_fallback": False},
            100: {"answer_em": 1, "answer_acc": 1, "answer_f1": 1, "local_fallback": False},
            150: {"answer_em": 1, "answer_acc": 1, "answer_f1": 1, "local_fallback": False},
        },
    }
    flip = metric_audit(synthetic_grid, [50, 100, 150], "answer_em", 200, 2026)
    assert flip["transitions"][0]["wrong_to_correct_count"] == 1
    assert flip["transitions"][1]["correct_to_wrong_count"] == 1
    sample_a = next(row for row in flip["per_sample"] if row["sample_id"] == "a")
    assert sample_a["minimum_success_budget_msb"] == 100
    assert sample_a["stable_sufficient_budget_ssb_observed"] is None
    assert sample_a["answer_flip_count"] == 2
    with tempfile.TemporaryDirectory() as temporary:
        store = SQLiteRecordStore(Path(temporary) / "state.sqlite")
        store.put_record("a", {"sample_id": "a", "value": 1})
        assert store.completed_keys() == {"a"}
        assert list(store.iter_records())[0]["value"] == 1
        store.invalidate_record("a", {
            "sample_id": "a", "error_type": "BoundaryPolicyUpgrade",
        })
        assert store.completed_keys() == set()
        assert store.failed_keys() == {"a"}
        store.put_record("a", {"sample_id": "a", "value": 2})
        assert store.completed_keys() == {"a"}
        assert store.failed_keys() == set()
        store.export(temporary)
        store.close()
        trajectory_path = Path(temporary) / "trajectory.jsonl"
        label_path = Path(temporary) / "labels.jsonl"
        analysis_path = Path(temporary) / "analysis"
        write_jsonl(trajectory_path, [trajectory])
        write_jsonl(label_path, [{"sample_id": "x", **label}])
        old_argv = sys.argv
        try:
            sys.argv = [
                "analyze_hotpotqa.py", "--trajectories", str(trajectory_path),
                "--labels", str(label_path), "--output_dir", str(analysis_path),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                assert analyze_main() == 0
        finally:
            sys.argv = old_argv
        summary = read_json(analysis_path / "summary.json")
        assert summary["records"] == 1
        assert summary["overall"]["answer_em"] == 1
    print("SELF-TEST PASSED: exact concatenation, stable-boundary token prefixes, "
          "explicit boundary backoff, exact FULL endpoint, "
          "document/global evidence token positions, "
          "one-call action, stop repair, native thinking, "
          "Hotpot evidence labels, strict fallback scoring, paired answer flips, MSB/SSB, "
          "original-text offset boundaries, deprecated-record invalidation, "
          "safe semantic-root resume migration, compressed resume store.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
