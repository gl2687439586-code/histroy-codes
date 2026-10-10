#!/usr/bin/env python3
"""Offline protocol tests; performs no model or retriever request."""

from __future__ import annotations

from adaptive_k import parse_flashrag_scored_response, select_adaptive_k
from common import OFFICIAL_QA_TEMPLATE, extract_qwen_answer, paper_qa_prompt


def main() -> int:
    scores = [1.00, 0.99, 0.98, 0.50, 0.49, 0.48, 0.47, 0.46, 0.45, 0.44]
    selected = select_adaptive_k(scores, ignore_extreme_tail=0.1, retrieve_more=5)
    assert selected["largest_gap_after_rank_1based"] == 3
    assert selected["selected_k"] == 8
    assert selected["retrieve_more_buffer"] == 5

    # The very last gap is excluded by the exact official tail formula.
    tail = [1.0, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.80, 0.79, 0.78, 0.10]
    tail_selected = select_adaptive_k(tail, ignore_extreme_tail=0.1, retrieve_more=5)
    assert tail_selected["largest_gap_after_rank_1based"] == 7
    assert tail_selected["tail_gap_count_excluded"] == 1

    docs, parsed_scores = parse_flashrag_scored_response([
        [{"id": "a", "contents": "A\ntext"}, {"id": "b", "contents": "B\ntext"}],
        [0.9, 0.8],
    ])
    assert len(docs) == 2 and parsed_scores == [0.9, 0.8]
    prompt = paper_qa_prompt("Q?", "D")
    assert prompt.count("Q?") == 2 and prompt.endswith("- **Answer:**\n")
    assert prompt == OFFICIAL_QA_TEMPLATE.format(question="Q?", context="D")
    assert "<search>" not in prompt and "Read" not in prompt
    answer = extract_qwen_answer("<think>reason</think>\nParis", {"finish_reason": {"type": "stop"}})
    assert answer["final_answer"] == "Paris" and answer["answer_status"] == "valid"
    truncated = extract_qwen_answer("<think>unfinished", {"finish_reason": {"type": "length"}})
    assert truncated["answer_status"] == "truncated_thinking"
    print("ADAPTIVE-K PROTOCOL SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
