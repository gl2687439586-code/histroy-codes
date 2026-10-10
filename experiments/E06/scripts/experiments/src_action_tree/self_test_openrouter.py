#!/usr/bin/env python3
"""Offline protocol/client tests; performs no OpenRouter or Retriever calls."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ModelClient, qwen_serialized_prompt_to_messages  # noqa: E402
from openrouter_common import EXPECTED_MODEL_ID, assert_openrouter_configuration  # noqa: E402
from openrouter_prompts import READ_POLICY, build_prompt  # noqa: E402
from run_action_tree import expected_paths  # noqa: E402


def main() -> int:
    old_key = os.environ.get("OPENROUTER_API_KEY")
    old_ledger = os.environ.get("OPENROUTER_BILLING_LEDGER")
    os.environ["OPENROUTER_API_KEY"] = "test-key-never-sent"
    with tempfile.TemporaryDirectory() as temporary_directory:
      ledger_path = Path(temporary_directory) / "billing.jsonl"
      os.environ["OPENROUTER_BILLING_LEDGER"] = str(ledger_path)
      try:
        assert_openrouter_configuration(
            EXPECTED_MODEL_ID, "https://openrouter.ai/api/v1"
        )
        prompt = build_prompt("Question?", ["query"], [], [], READ_POLICY)
        messages = qwen_serialized_prompt_to_messages(prompt)
        assert [message["role"] for message in messages] == ["system", "user"]
        assert "NEXT-STEP POLICY" in messages[1]["content"]

        blank = (
            "<|im_start|>system\nSystem<|im_end|>\n"
            "<|im_start|>user\nQuestion?<|im_end|>\n"
            "<|im_start|>assistant\n<think>Need evidence.</think>"
            "<search>query</search>\n<result>result: evidence</result>\n"
        )
        blank_messages = qwen_serialized_prompt_to_messages(blank)
        assert [message["role"] for message in blank_messages] == [
            "system",
            "user",
            "assistant",
            "user",
        ]

        captured = {}
        client = ModelClient(
            "https://openrouter.ai/api/v1", EXPECTED_MODEL_ID, "openrouter", 30, 1
        )

        def fake_post(url, payload, headers=None):
            captured.update({"url": url, "payload": payload, "headers": headers})
            return (
                {
                    "id": "gen-self-test",
                    "model": EXPECTED_MODEL_ID,
                    "provider": "OpenAI",
                    "openrouter_metadata": {
                        "endpoints": {
                            "available": [
                                {"provider": "OpenAI", "selected": True}
                            ]
                        }
                    },
                    "choices": [
                        {
                            "message": {"content": "<answer>OK</answer>"},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 101,
                        "completion_tokens": 7,
                        "completion_tokens_details": {"reasoning_tokens": 2},
                        "cost": 0.00123,
                    },
                },
                0.25,
            )

        client._post = fake_post  # type: ignore[method-assign]
        response = client.generate(prompt, 0.0, 32, ("</answer>",), 2026)
        assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
        assert captured["payload"]["model"] == EXPECTED_MODEL_ID
        assert captured["payload"]["seed"] == 2026
        assert captured["payload"]["provider"] == {
            "order": ["openai"],
            "only": ["openai"],
            "allow_fallbacks": False,
        }
        assert captured["payload"]["reasoning"]["effort"] == "medium"
        assert captured["headers"]["Authorization"].startswith("Bearer ")
        assert captured["headers"]["X-OpenRouter-Metadata"] == "enabled"
        assert captured["headers"]["X-OpenRouter-Cache"] == "false"
        assert response.input_tokens == 101
        assert response.output_tokens == 7
        assert response.reasoning_tokens == 2
        assert response.provider_id == "OpenAI"
        assert abs(response.cost_usd - 0.00123) < 1e-12
        ledger_row = __import__("json").loads(ledger_path.read_text(encoding="utf-8"))
        assert ledger_row["response_id"] == "gen-self-test"
        assert abs(ledger_row["cost_usd"] - 0.00123) < 1e-12
        resumed_client = ModelClient(
            "https://openrouter.ai/api/v1", EXPECTED_MODEL_ID, "openrouter", 30, 1
        )
        assert abs(resumed_client.historical_cost_usd - 0.00123) < 1e-12
        assert expected_paths(3) == ("SSS", "SSR", "SSC", "SRS", "SRR", "SRC", "SC")
      finally:
        if old_key is None:
            os.environ.pop("OPENROUTER_API_KEY", None)
        else:
            os.environ["OPENROUTER_API_KEY"] = old_key
        if old_ledger is None:
            os.environ.pop("OPENROUTER_BILLING_LEDGER", None)
        else:
            os.environ["OPENROUTER_BILLING_LEDGER"] = old_ledger

    print("OPENROUTER SELF-TEST PASSED: chat serialization, exact model, usage, cost, seven paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
