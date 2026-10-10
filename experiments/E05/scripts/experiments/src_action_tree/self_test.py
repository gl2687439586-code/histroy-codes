#!/usr/bin/env python3
"""Offline smoke test; no model, retriever, or internet service is required."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    ModelResponse,
    RetrievalResponse,
    TokenCounter,
    append_jsonl,
    load_experiment_samples,
    make_document,
)
from run_action_tree import Runtime, expected_paths, run_one_tree, terminal_answer_is_complete  # noqa: E402


class MockModel:
    def generate(
        self,
        prompt: str,
        temperature: float,
        max_new_tokens: int,
        stop: Sequence[str],
        seed: int,
    ) -> ModelResponse:
        if "BEHAVIOR: SEARCH" in prompt.rsplit("<|im_start|>user", 1)[-1]:
            # Qwen-like length-truncated thinking: no <search> tag. Flexible
            # mode must still execute Search through a deterministic fallback.
            text = "I should verify the key entity and date from another source, but I am still planning."
        elif "EVIDENCE DEEPENING / READ" in prompt.rsplit("<|im_start|>user", 1)[-1]:
            text = (
                "Candidates: Paris. Satisfied: the evidence states the capital. "
                "Conflicts: none. Unknowns: none.</think>"
            )
        else:
            # Qwen-like terminal output: correct boxed answer but no <answer> tag.
            text = "The evidence is sufficient.</think>\nThe final answer is \\[ \\boxed{Paris} \\]"
        return ModelResponse(text=text, raw={"text": text, "meta_info": {}}, endpoint="mock://model", latency_seconds=0.001)


class MockRetriever:
    def search(self, query: str, top_k: int = 5) -> RetrievalResponse:
        docs = [make_document(f"{query}-{index}", f"Mock evidence {index}: Paris is supported.") for index in range(top_k)]
        result_text = "result: \n" + "\n\n".join(doc["contents"] for doc in docs)
        return RetrievalResponse(query, docs, result_text, docs, "mock://retriever", 0.001)


def runtime_args() -> argparse.Namespace:
    return argparse.Namespace(
        base_seed=2026,
        context_length=32768,
        fixed_doc_tokens=0,
        top_k=5,
        action_attempts=2,
        action_validation="flexible",
        temperature=0.0,
        search_max_new_tokens=128,
        read_max_new_tokens=128,
        answer_max_new_tokens=128,
    )


def main() -> int:
    assert terminal_answer_is_complete("Paris", "natural_language", {"type": "stop"}, "Paris")
    assert not terminal_answer_is_complete(
        "I am still considering Paris", "natural_language", {"type": "length"}, "I am still considering Paris"
    )
    assert terminal_answer_is_complete(
        "Paris", "boxed", {"type": "length"}, "The answer is \\boxed{Paris}"
    )
    prefix = (
        "<|im_start|>system\nMock format: <search></search> "
        "<result></result>. Ignore these instruction tags.<|im_end|>\n"
        "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n"
        "<|im_start|>assistant\n<think>Find the country fact.</think>\n"
        "<search>capital of France</search>\n"
        "<result>result: \nParis is the capital of France.\n</result>\n"
    )
    sample: Dict[str, Any] = {
        "sample_id": "test_mock",
        "question": "What is the capital of France?",
        "gold_answers": ["Paris"],
        "checkpoint_prefix": prefix,
        "initial_query": "capital of France",
        "initial_result": "result: \nParis is the capital of France.",
        "initial_documents": [make_document("initial-paris", "Paris is the capital of France.")],
        "checkpoint_source": "self_test_exact",
        "original_search_count": 3,
        "horizon": 3,
    }
    runtime = Runtime(
        args=runtime_args(),
        tokenizer=TokenCounter(force_approximate=True),
        model=MockModel(),  # type: ignore[arg-type]
        retriever=MockRetriever(),  # type: ignore[arg-type]
    )
    first_rollout = run_one_tree(runtime, sample, rollout_id=0)
    second_rollout = run_one_tree(runtime, sample, rollout_id=1)
    sample_h4 = dict(sample)
    sample_h4.update(sample_id="test_mock_h4", original_search_count=4, horizon=4)
    h4_rollout = run_one_tree(runtime, sample_h4, rollout_id=0)
    records = first_rollout + second_rollout + h4_rollout
    assert [record["path"] for record in first_rollout] == list(expected_paths(3))
    assert [record["path"] for record in second_rollout] == list(expected_paths(3))
    assert [record["path"] for record in h4_rollout] == list(expected_paths(4))
    assert len(expected_paths(3)) == 7
    assert len(expected_paths(4)) == 15
    assert len(expected_paths(5)) == 31
    assert len(expected_paths(6)) == 63
    assert all(record["final_answer"] == "Paris" for record in records)
    assert all(record["f1"] == 1.0 for record in records)
    assert all(record["terminal_completed"] is True for record in records)
    assert all(record["retrieved_docs_nominal"] == record["search_calls"] * 5 for record in records)
    assert any(record["search_query_fallback_calls"] > 0 for record in records)
    for record in records:
        assert record["search_calls"] == record["action_sequence"].count("S")
        for event in record["events"]:
            if event["action_code"] == "R":
                assert event["evidence_set_unchanged"] is True

    with tempfile.TemporaryDirectory(prefix="src_action_tree_test_") as temp:
        temp_path = Path(temp)
        intermediate = temp_path / "intermediate_data.json"
        intermediate.write_text(
            json.dumps(
                [
                    {
                        "id": "test_mock",
                        "question": sample["question"],
                        "golden_answers": sample["gold_answers"],
                        "output": {"final_response": prefix, "pred": "Paris"},
                    }
                ]
            ),
            encoding="utf-8",
        )
        loaded, load_report = load_experiment_samples(intermediate)
        assert load_report["loaded_samples"] == 1
        assert loaded[0]["checkpoint_prefix"] == prefix
        assert loaded[0]["initial_query"] == "capital of France"

        dry_dir = temp_path / "dry_run"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "run_action_tree.py"),
                "--intermediate_data",
                str(intermediate),
                "--output_dir",
                str(dry_dir),
                "--model_path",
                "mock-model",
                "--approximate_tokens",
                "--dry_run",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert (dry_dir / "dry_run_report.json").exists()

        terminal = temp_path / "terminal_trajectories.jsonl"
        # Simulate an interrupted append followed by a complete tree retry.
        append_jsonl(terminal, [records[0]])
        append_jsonl(terminal, records)
        verification = temp_path / "verification.json"
        subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "verify_outputs.py"), "--input", str(terminal), "--report", str(verification)],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        analysis_dir = temp_path / "analysis"
        subprocess.run(
            [
                sys.executable,
                str(SCRIPT_DIR / "analyze_action_tree.py"),
                "--input",
                str(terminal),
                "--output_dir",
                str(analysis_dir),
                "--quality_metric",
                "f1",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        verification_data = json.loads(verification.read_text(encoding="utf-8"))
        assert verification_data["passed"] is True
        assert verification_data["duplicate_records_ignored"] == 1
        findings = json.loads((analysis_dir / "findings.json").read_text(encoding="utf-8"))
        assert findings["complete_samples"] == 2
        assert findings["oracle_value_field"] == "qmean"
        assert findings["duplicate_terminal_records_ignored"] == 1
        assert (analysis_dir / "state_action_values.csv").exists()
        assert (analysis_dir / "budget_action_distributions.csv").exists()
        assert (analysis_dir / "paired_read_vs_search.csv").exists()
        assert (analysis_dir / "metric_sensitivity_summary.csv").exists()
        assert (analysis_dir / "horizon_summary.csv").exists()
        assert "p_read_gt_search_root_ci95_wilson" in findings
        assert findings["horizon_sample_counts"] == {"3": 1, "4": 1}
        assert {row["horizon"] for row in findings["horizon_summary"]} == {3, 4}

    print("SELF-TEST PASSED: mixed H=3/H=4 trees, H=3..6 path counts, flexible control, invariants, variable-horizon DP, Pareto, budgets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
