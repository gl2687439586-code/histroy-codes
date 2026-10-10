#!/usr/bin/env python3
"""Offline protocol and entry-point import tests; no service calls."""

import importlib
import tempfile
from pathlib import Path

from common import (
    MAX_SEARCHES,
    METHOD,
    ROLLOUTS,
    TEMPERATURE,
    TOP_K,
    TOP_P,
    parse_action,
    parse_budget,
    request_attempts,
    score_prediction,
    summarize_request_attempts,
    write_jsonl,
)
from prompts import TALE_EP_BUDGET_SENTENCE, TALE_EP_ESTIMATOR_INSTRUCTION, estimator_prompt


def main() -> int:
    # Import every Python entry point so stale symbols left by protocol edits fail
    # before a tmux worker is launched.
    for module_name in (
        "analyze",
        "audit",
        "estimate_budgets",
        "merge_records",
        "prepare_dataset",
        "run_trajectories",
        "write_retriever_config",
    ):
        importlib.import_module(module_name)
    run_script = (Path(__file__).resolve().parent / "run.sh").read_text(encoding="utf-8")
    assert 'ESTIMATOR_FORMAT_RETRY_PASSES="${ESTIMATOR_FORMAT_RETRY_PASSES:-10}"' in run_script
    assert "estimate_until_complete" in run_script
    assert "--allow-partial" in run_script
    assert "Freezing persistent estimator failures" in run_script
    assert METHOD == "tale_ep"
    assert (ROLLOUTS, TEMPERATURE, TOP_P, MAX_SEARCHES, TOP_K) == (1, 0.1, 1.0, 4, 5)
    assert parse_budget("Budget: [[512]]") == 512
    assert parse_action("<Search>Albert Einstein birthplace</Search>")["type"] == "search"
    assert parse_action("<Answer>Ulm</Answer>")["type"] == "answer"
    assert parse_action("<Search>x</Search><Answer>y</Answer>")["type"] == "invalid"
    prompt = estimator_prompt("Question?")
    assert TALE_EP_ESTIMATOR_INSTRUCTION in prompt
    assert prompt == f"Question: Question?\n\n{TALE_EP_ESTIMATOR_INSTRUCTION}"
    assert "COMPLETE MULTI-TURN TRAJECTORY" not in prompt
    assert "Do NOT estimate only the length of the short final answer" not in prompt
    assert "less than 512 tokens" in TALE_EP_BUDGET_SENTENCE.format(budget=512)
    metrics = score_prediction("Ulm, Germany", ["Ulm"])
    assert metrics["em"] == 0.0 and metrics["acc"] == 1.0 and metrics["f1"] > 0
    with tempfile.TemporaryDirectory() as temporary:
        ledger = Path(temporary) / "ledger.jsonl"
        write_jsonl(
            ledger,
            [
                {"event": "request_started", "request_id": "failed-parse"},
                {
                    "event": "request_finished",
                    "request_id": "failed-parse",
                    "outcome": "success",
                    "usage_available": True,
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 20,
                        "total_tokens": 30,
                    },
                },
                {
                    "event": "consumer_result",
                    "request_id": "failed-parse",
                    "outcome": "downstream_parse_error",
                },
                {"event": "request_started", "request_id": "retry-success"},
                {
                    "event": "request_finished",
                    "request_id": "retry-success",
                    "outcome": "success",
                    "usage_available": True,
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 21,
                        "total_tokens": 32,
                    },
                },
                {
                    "event": "consumer_result",
                    "request_id": "retry-success",
                    "outcome": "accepted_estimate",
                },
            ],
        )
        summary = summarize_request_attempts(request_attempts([ledger]))
        assert summary["physical_requests"] == 2
        assert summary["usage_lower_bound"]["total_tokens"] == 62
        assert summary["consumer_outcomes"]["downstream_parse_error"] == 1
        assert summary["cost_ledger_complete"] is True
    print(
        "TALE-EP V3 ORIGINAL-PROMPT SELF-TEST PASSED: verbatim paper prompts, "
        "retry-complete request ledger, temperature=0.1, one rollout, "
        "max four searches, Top-5, metrics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
