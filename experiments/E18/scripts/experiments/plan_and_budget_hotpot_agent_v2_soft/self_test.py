from __future__ import annotations

import json
import inspect
import tempfile
from pathlib import Path

from .common import LEVEL_BUDGETS, allocate_tokens, extract_json_object, parse_action
from . import common
from .plan_questions import validate_evaluation
from .prompts import REASONER_SYSTEM, reasoner_user
from . import run_trajectories


def main() -> int:
    assert LEVEL_BUDGETS == {1: 200, 2: 250, 3: 350, 4: 450, 5: 600}
    for scheduler in ("polynomial", "cosine"):
        for credits in ([50, 50], [20, 30, 50], [10, 20, 30, 40], [10, 15, 20, 25, 30]):
            for total in LEVEL_BUDGETS.values():
                allocation = allocate_tokens(credits, total, scheduler)
                assert len(allocation) == len(credits)
                assert sum(allocation) == total
                assert all(x > 0 for x in allocation)

    # Polynomial front-loads equally credited steps.
    assert allocate_tokens([25, 25, 25, 25], 600, "polynomial") == [320, 180, 80, 20]
    cosine = allocate_tokens([25, 25, 25, 25], 600, "cosine")
    assert sum(cosine) == 600 and cosine[0] > cosine[-1]

    assert parse_action("<Search>Albert Einstein birthplace</Search>")["type"] == "search"
    parsed = parse_action(
        "<Subanswer>Ulm</Subanswer><FinalAnswer>Ulm</FinalAnswer><Cite>8874316</Cite>"
    )
    assert parsed["type"] == "finish" and parsed["final_answer"] == "Ulm"
    assert parse_action("free form")["type"] == "invalid"
    assert extract_json_object('prefix ```json\n{"x": 1}\n``` suffix') == {"x": 1}

    try:
        validate_evaluation(
            {
                "problem": {"reason": "at most 50 words", "evaluated_level": 2},
                "subquestions": [
                    {
                        "index": 1,
                        "reason": "at most 50 words",
                        "evaluated_level": 2,
                        "credit": 50,
                    },
                    {
                        "index": 2,
                        "reason": "at most 50 words",
                        "evaluated_level": 2,
                        "credit": 50,
                    },
                ],
            },
            [
                {"index": 1, "subquestion": "One?", "hint": "First."},
                {"index": 2, "subquestion": "Two?", "hint": "Second."},
            ],
        )
        raise AssertionError("Planner placeholder reason was accepted")
    except ValueError as exc:
        assert "placeholder" in str(exc)

    prompt = reasoner_user(
        original_question="Question?",
        level=3,
        total_budget=350,
        scheduler="polynomial",
        subquestion_index=2,
        subquestion_count=2,
        subquestion="Second?",
        hint="Combine.",
        subquestion_budget=100,
        actual_output_tokens_so_far=120,
        previous_summaries=[
            {"step_index": 1, "subanswer": "summary", "cited_document_ids": ["d1"]}
        ],
        search_rounds=[
            {
                "search_round": 1,
                "query": "query",
                "documents": [
                    {"id": f"d{i}", "rank": i, "contents": f"doc {i}"}
                    for i in range(1, 6)
                ],
            }
        ],
        search_limit_reached=False,
    )
    assert "FinalAnswer" in prompt and "doc 5" in prompt
    assert "soft target" in REASONER_SYSTEM
    assert "not a hard generation cutoff" in REASONER_SYSTEM
    assert "Actual output tokens used" in prompt
    assert "120 tokens" in prompt
    assert "advisory" in prompt
    assert "50 tokens" not in REASONER_SYSTEM
    assert "Read" not in REASONER_SYSTEM
    assert '"return_score": False' in inspect.getsource(common.retriever_search)

    # A soft-budget overrun must not stop the normal Search -> Finish trajectory.
    class FakeTokenizer:
        @staticmethod
        def encode(text, add_special_tokens=False):
            return text.split()

    responses = iter(
        [
            {
                "text": "<Search>target entity</Search>",
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 140,
                    "total_tokens": 160,
                },
                "latency_seconds": 0.01,
            },
            {
                "text": (
                    "<Subanswer>resolved fact</Subanswer>"
                    "<FinalAnswer>resolved fact</FinalAnswer><Cite>d1</Cite>"
                ),
                "finish_reason": "stop",
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 60,
                    "total_tokens": 100,
                },
                "latency_seconds": 0.01,
            },
        ]
    )

    original_chat = run_trajectories.chat_completion
    original_search = run_trajectories.retriever_search
    try:
        run_trajectories.chat_completion = lambda *args, **kwargs: next(responses)
        run_trajectories.retriever_search = lambda *args, **kwargs: {
            "query": "target entity",
            "top_k": 5,
            "documents": [
                {"id": f"d{i}", "rank": i, "contents": f"document {i}"}
                for i in range(1, 6)
            ],
            "latency_seconds": 0.01,
        }
        trajectory = run_trajectories.solve_subquestion(
            tokenizer=FakeTokenizer(),
            model_url="http://unused",
            model_name="Qwen3-8B",
            retriever_url="http://unused",
            question="Question?",
            level=1,
            total_budget=200,
            scheduler="polynomial",
            step={
                "subquestion": "Resolve it?",
                "hint": "Find the fact.",
                "credit": 100,
                "evaluated_level": 1,
            },
            step_index=1,
            step_count=1,
            step_budget=100,
            previous_summaries=[],
            max_searches=4,
            top_k=5,
            generation_max_tokens=12288,
        )
    finally:
        run_trajectories.chat_completion = original_chat
        run_trajectories.retriever_search = original_search

    assert trajectory["answer_status"] == "valid"
    assert len(trajectory["search_rounds"]) == 1
    assert trajectory["consumed_output_tokens"] == 200
    assert trajectory["soft_budget_overrun_tokens"] == 100
    assert trajectory["soft_budget_compliant"] is False
    assert all(
        turn["technical_generation_max_tokens"] == 12288
        for turn in trajectory["model_turns"]
    )

    print(
        "PLAN-AND-BUDGET HOTPOT AGENT SOFT-BUDGET SELF-TEST PASSED: official levels, "
        "Polynomial/Cosine conservation, action grammar, Top-5 full evidence, "
        "last-step synthesis, no Read action, prompt-only budget control."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
