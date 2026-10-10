#!/usr/bin/env python3
"""Minimal Qwen3 ReAct protocol prompts with native thinking enabled."""

from __future__ import annotations

from typing import Any, Dict, List


SYSTEM = (
    "You are Qwen3-8B solving an open-domain multi-hop question. "
    "Use Wikipedia search when evidence is missing."
)


def chat(user: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
        f"<|im_start|>user\n{user.rstrip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def initial_search_prompt(question: str) -> str:
    return chat(f"""Question:
{question.strip()}

Generate the single best Wikipedia search query needed to start solving this
question. Think naturally, then call the search tool using:
<search>concise search query</search>""")


def render_history(rounds: List[Dict[str, Any]]) -> str:
    parts = []
    for item in rounds:
        parts.append(
            f"Search {item['search_round']} query:\n{item['query']}\n\n"
            f"Search {item['search_round']} visible retrieved evidence:\n"
            f"{item['evidence']['evidence_text']}"
        )
    return "\n\n=====\n\n".join(parts)


def decision_prompt(question: str, rounds: List[Dict[str, Any]],
                    remaining_searches: int) -> str:
    queries = "; ".join(item["query"] for item in rounds)
    return chat(f"""Question:
{question.strip()}

Search history and currently visible evidence:
{render_history(rounds)}

Queries already used: {queries}
Additional searches allowed: {remaining_searches}

Continue solving naturally from the current evidence. If essential evidence is
still missing, issue a new query:
<search>concise new query</search>

If the evidence is sufficient, commit now:
<answer>short answer only</answer>

Use the tags above so the external tool controller can execute the action.""")


def forced_answer_prompt(question: str, rounds: List[Dict[str, Any]]) -> str:
    return chat(f"""Question:
{question.strip()}

Search history and currently visible evidence:
{render_history(rounds)}

The search limit has been reached. Think using the evidence already retrieved,
then give the best short answer using:
<answer>short answer only</answer>""")
