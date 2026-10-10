#!/usr/bin/env python3
"""Offline end-to-end test for parse-failure and retry cost accounting."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from common import (
    METHOD,
    PROTOCOL_VARIANT,
    ROLLOUTS,
    canonical_hash,
    write_json,
    write_jsonl,
)
from prompts import TALE_EP_BUDGET_SENTENCE, estimator_prompt


def events(request_id, context, usage=None, consumer=None, outcome="success"):
    rows = [
        {
            "event": "request_started",
            "request_id": request_id,
            "created_at": request_id,
            "context": context,
        },
        {
            "event": "request_finished",
            "request_id": request_id,
            "created_at": request_id,
            "outcome": outcome,
            "usage_available": usage is not None,
            "usage": usage,
            "context": context,
        },
    ]
    if consumer:
        rows.append(
            {
                "event": "consumer_result",
                "request_id": request_id,
                "outcome": consumer,
            }
        )
    return rows


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        estimator_dir = root / "estimator"
        solver_dir = root / "solver"
        estimator_dir.mkdir()
        solver_dir.mkdir()
        question = {"sample_index": 0, "sample_id": "q1", "question": "Who?"}
        label = {"sample_id": "q1", "answer": "Alice", "gold_answers": ["Alice"]}
        estimate = {
            "method": METHOD,
            "protocol_variant": PROTOCOL_VARIANT,
            "sample_index": 0,
            "sample_id": "q1",
            "question": "Who?",
            "estimator_prompt": estimator_prompt("Who?"),
            "estimated_budget": 500,
            "successful_estimator_request_id": "est-success",
            "gold_labels_loaded": False,
        }
        estimate["record_hash"] = canonical_hash(estimate)
        write_jsonl(root / "questions.jsonl", [question])
        write_jsonl(root / "labels.jsonl", [label])
        write_jsonl(root / "estimates.jsonl", [estimate])
        estimator_events = []
        estimator_events += events(
            "est-parse-fail",
            {"sample_id": "q1", "stage": "query_only_budget_estimation"},
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            "downstream_parse_error",
        )
        estimator_events += events(
            "est-success",
            {"sample_id": "q1", "stage": "query_only_budget_estimation"},
            {"prompt_tokens": 11, "completion_tokens": 29, "total_tokens": 40},
            "accepted_estimate",
        )
        write_jsonl(
            estimator_dir / "shard_0.physical_requests.jsonl", estimator_events
        )
        trajectories = []
        solver_events = []
        for rollout in range(ROLLOUTS):
            request_id = f"solve-{rollout}"
            trajectories.append(
                {
                    "method": METHOD,
                    "protocol_variant": PROTOCOL_VARIANT,
                    "sample_index": 0,
                    "sample_id": "q1",
                    "question": "Who?",
                    "rollout_id": rollout,
                    "temperature": 0.1,
                    "top_p": 1.0,
                    "estimated_budget": 500,
                    "estimate_record_hash": estimate["record_hash"],
                    "maximum_allowed_searches": 4,
                    "top_k_per_search": 5,
                    "search_rounds": [],
                    "model_turns": [{"physical_request_id": request_id}],
                    "prediction": "Alice",
                    "answer_status": "valid",
                    "solver_usage": {
                        "prompt_tokens": 50,
                        "completion_tokens": 100,
                        "total_tokens": 150,
                        "model_calls": 1,
                        "search_calls": 0,
                    },
                    "budget_compliant": True,
                    "budget_overrun_tokens": 0,
                    "tale_budget_sentence": TALE_EP_BUDGET_SENTENCE.format(budget=500),
                    "gold_labels_loaded": False,
                }
            )
            solver_events += events(
                request_id,
                {"sample_id": "q1", "rollout_id": rollout, "stage": "agent_trajectories"},
                {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150},
                "parsed_answer",
            )
        # A failed physical rerun attempt has no usage. It must remain in the
        # request count and force cost_is_exact=False instead of disappearing.
        solver_events += events(
            "solve-0-transport-fail",
            {"sample_id": "q1", "rollout_id": 0, "stage": "agent_trajectories"},
            None,
            None,
            "transport_error",
        )
        write_jsonl(solver_dir / "shard_0.physical_requests.jsonl", solver_events)
        write_jsonl(root / "trajectories.jsonl", trajectories)
        analysis_dir = root / "analysis"
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "analyze.py"),
                "--questions",
                str(root / "questions.jsonl"),
                "--labels",
                str(root / "labels.jsonl"),
                "--estimates",
                str(root / "estimates.jsonl"),
                "--trajectories",
                str(root / "trajectories.jsonl"),
                "--estimator-ledger-dir",
                str(estimator_dir),
                "--solver-ledger-dir",
                str(solver_dir),
                "--output-dir",
                str(analysis_dir),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        audit_path = analysis_dir / "AUDIT.json"
        subprocess.run(
            [
                sys.executable,
                str(script_dir / "audit.py"),
                "--questions",
                str(root / "questions.jsonl"),
                "--labels",
                str(root / "labels.jsonl"),
                "--estimates",
                str(root / "estimates.jsonl"),
                "--trajectories",
                str(root / "trajectories.jsonl"),
                "--estimator-ledger-dir",
                str(estimator_dir),
                "--solver-ledger-dir",
                str(solver_dir),
                "--analysis",
                str(analysis_dir / "summary.json"),
                "--expected-questions",
                "1",
                "--output",
                str(audit_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        summary = json.loads((analysis_dir / "summary.json").read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        assert summary["physical_cost_entire_run"]["estimator"]["physical_requests"] == 2
        assert summary["physical_cost_entire_run"]["estimator"]["usage_lower_bound"]["total_tokens"] == 70
        assert summary["physical_cost_entire_run"]["solver"]["physical_requests"] == ROLLOUTS + 1
        assert summary["physical_cost_entire_run"]["end_to_end"]["cost_is_exact"] is False
        assert audit["passed"] is True
        assert audit["physical_request_accounting"]["successful_response_parse_failures_counted"] == 1
    print("TALE-EP V3 ACCOUNTING INTEGRATION TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
