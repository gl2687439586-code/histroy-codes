#!/usr/bin/env python3
"""Prompt used by the frozen Answer-only phase.

The phase deliberately keeps native Qwen3 thinking enabled.  No judge, tool,
controller, Read, Search, or post-hoc repair prompt is allowed here.
"""

from __future__ import annotations


def answer_prompt(question: str, evidence_text: str) -> str:
    user = f"""Question:
{question.strip()}

Retrieved passages (fixed rank order and fixed visible prefixes):
{evidence_text}

Answer the question using the retrieved passages and your reasoning. Do not
search, request a tool, or ask for more information. Your response must end
with exactly one final tag in this form:
<answer>short answer only</answer>

Inside the tag, write only the short answer: no explanation, no XML/HTML
markup, and no additional angle brackets."""
    return (
        "<|im_start|>system\n"
        "You are Qwen3-8B answering a knowledge question. The retrieved "
        "passages are the only external evidence available in this run."
        "<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
