#!/usr/bin/env python3
"""Reliable action parsing and bounded generation."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from hp_common import ExactTokenizer, SafeSGLangClient, finish_type, sha256_text


MAX_SEARCH_QUERY_CHARS = 256


def parse_action(text: str, allowed: List[str]) -> Optional[Dict[str, str]]:
    candidates = []
    for name in allowed:
        for match in re.finditer(
            rf"<{name}>\s*(.*?)\s*</{name}>", str(text or ""), flags=re.I | re.S
        ):
            payload = match.group(1).strip()
            if payload and "<" not in payload and ">" not in payload:
                candidates.append((match.start(), name, " ".join(payload.split())))
    if not candidates:
        for name in allowed:
            matches = list(re.finditer(
                rf"<{name}>\s*([^<>]+?)\s*$", str(text or ""), flags=re.I | re.S
            ))
            if matches:
                payload = " ".join(matches[-1].group(1).split())
                if payload:
                    candidates.append((matches[-1].start(), name, payload))
    if not candidates:
        return None
    _, name, payload = max(candidates, key=lambda item: item[0])
    # Search queries are controller inputs and remain deliberately bounded.
    # Answers are already bounded by max_new_tokens.  Applying the same
    # 256-character query limit to a canonical <answer>...</answer> caused
    # valid, closed answers to be mislabeled as local fallbacks.
    if name == "search" and len(payload) > MAX_SEARCH_QUERY_CHARS:
        return None
    return {"action": name, "payload": payload}


def attempt_dict(prompt: str, generation: Any, tokenizer: ExactTokenizer,
                 maximum: int, parse_mode: str) -> Dict[str, Any]:
    thinking_parts = re.findall(
        r"<think>(.*?)</think>", generation.text, flags=re.I | re.S
    )
    lower_output = generation.text.lower()
    last_open = lower_output.rfind("<think>")
    last_close = lower_output.rfind("</think>")
    if last_open > last_close:
        thinking_parts.append(generation.text[last_open + len("<think>"):])
    thinking = "\n".join(thinking_parts)
    return {
        "prompt": prompt,
        "prompt_sha256": sha256_text(prompt),
        "raw_output": generation.raw_text,
        "output": generation.text,
        "output_sha256": sha256_text(generation.text),
        "max_new_tokens": maximum,
        "finish_reason": generation.finish_reason,
        "finish_type": finish_type(generation.finish_reason),
        "latency_seconds": generation.latency_seconds,
        "input_tokens": tokenizer.count(prompt),
        "output_tokens": tokenizer.count(generation.text),
        "thinking_tokens": tokenizer.count(thinking),
        "stop_boundary_repair": generation.stop_repair,
        "parse_mode": parse_mode,
    }


def generate_action(client: SafeSGLangClient, tokenizer: ExactTokenizer,
                    prompt: str, allowed: List[str], maximum: int,
                    repair_maximum: int = 0) -> Dict[str, Any]:
    """Generate exactly once. `repair_maximum` is retained only for CLI compatibility."""
    del repair_maximum
    stops = [f"</{name}>" for name in allowed]
    generation = client.generate(prompt, maximum, stops, temperature=0.0)
    parsed = parse_action(generation.text, allowed)
    attempts = [attempt_dict(
        prompt, generation, tokenizer, maximum,
        "canonical_or_boundary" if parsed else "unparsed",
    )]
    if parsed:
        return {**parsed, "parse_mode": "canonical_or_boundary", "attempts": attempts}
    return {
        "action": "", "payload": "", "parse_mode": "failed",
        "attempts": attempts,
    }


def plain_answer_fallback(text: str) -> str:
    value = re.sub(r"<think>.*?</think>", " ", str(text or ""), flags=re.I | re.S)
    value = re.sub(r"</?\w+[^>]*>", " ", value)
    cues = re.findall(r"(?:final\s+answer|answer)\s*:\s*([^\n]+)", value, flags=re.I)
    if cues:
        value = cues[-1]
    else:
        lines = [line.strip() for line in value.splitlines() if line.strip()]
        value = lines[-1] if lines else ""
    return " ".join(value.split())[:256]
