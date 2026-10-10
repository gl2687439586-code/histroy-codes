#!/usr/bin/env python3
"""OpenRouter manifest wrapper around the byte-identical v6 action prompts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from openrouter_common import EXPECTED_MODEL_ID, PROTOCOL_VERSION, openrouter_runtime_config
from qwen3_prompts import (  # re-export the frozen v6 prompt implementation
    COMMIT_POLICY,
    COMMON_SYSTEM,
    COMMON_USER_TEMPLATE,
    FORCED_ANSWER_POLICY,
    NEUTRAL_POLICY,
    READ_POLICY,
    SEARCH_POLICY,
    SHORT_ANSWER_REPAIR_POLICY,
    build_prompt,
    prompt_manifest as qwen3_prompt_manifest,
    read_notes_from_events,
    state_fields,
)


def prompt_manifest() -> Dict[str, Any]:
    manifest = qwen3_prompt_manifest()
    manifest["protocol_version"] = PROTOCOL_VERSION
    manifest["model"] = EXPECTED_MODEL_ID
    manifest["prompt_text_relation_to_qwen3_v6"] = (
        "COMMON_SYSTEM, COMMON_USER_TEMPLATE, and every NEXT-STEP POLICY are "
        "byte-identical to qwen3-only-src-v6; only chat-role serialization changes."
    )
    manifest["serialization_adapter"] = (
        "Qwen <|im_start|> blocks are converted to OpenRouter Chat Completions "
        "messages; retrieved <result> blocks become user observations."
    )
    manifest["openrouter_routing"] = openrouter_runtime_config()
    return manifest


def write_prompt_manifest(path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(prompt_manifest(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
