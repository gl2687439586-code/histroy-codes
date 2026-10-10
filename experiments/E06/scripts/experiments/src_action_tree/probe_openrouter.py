#!/usr/bin/env python3
"""Fail-closed OpenRouter GPT-5.6 Sol and local retrieval service probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ModelClient, RetrieverClient, utc_stamp  # noqa: E402
from openrouter_common import (  # noqa: E402
    EXPECTED_MODEL_ID,
    assert_openrouter_configuration,
    enforce_model_identity,
)
from openrouter_prompts import COMMIT_POLICY, build_prompt  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model", default=EXPECTED_MODEL_ID)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:6008")
    parser.add_argument("--sandbox_url", default="http://127.0.0.1:6006")
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()

    report = {"created_at": utc_stamp(), "model": args.model}
    try:
        assert_openrouter_configuration(args.model, args.model_url)
        report["model_catalog_identity"] = enforce_model_identity(args.model_url, args.model)
        # Exactly one successful bounded 256-token request. Transient network
        # failures may be retried, but only completed responses are billed and
        # recorded in the append-only ledger.
        model = ModelClient(args.model_url, args.model, "openrouter", args.timeout, 3, max_usd=0.0)
        prompt = build_prompt(
            "Reply with the single word OK.", [], [], [], COMMIT_POLICY
        )
        response = model.generate(prompt, 0.0, 256, ("</answer>",), 2026)
        report["model_generate"] = {
            "ok": bool(response.text.strip()),
            "returned_model": response.model_id,
            "returned_provider": response.provider_id,
            "text": response.text,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "reasoning_tokens": response.reasoning_tokens,
            "cost_usd": response.cost_usd,
            "latency_seconds": response.latency_seconds,
        }
    except Exception as exc:
        report["model_generate"] = {"ok": False, "error": str(exc)}

    try:
        retriever = RetrieverClient(
            args.retriever_url, args.sandbox_url, "direct", args.timeout, 1
        )
        response = retriever.search("Albert Einstein birthplace", 5)
        report["retriever_search"] = {
            "ok": len(response.documents) == 5,
            "documents": len(response.documents),
            "latency_seconds": response.latency_seconds,
        }
    except Exception as exc:
        report["retriever_search"] = {"ok": False, "error": str(exc)}

    returned_model = str(report.get("model_generate", {}).get("returned_model", ""))
    returned_provider = str(report.get("model_generate", {}).get("returned_provider", ""))
    report["passed"] = bool(
        report.get("model_catalog_identity", {}).get("verified_openrouter_model")
        and report.get("model_generate", {}).get("ok")
        and returned_model.startswith(EXPECTED_MODEL_ID)
        and returned_provider.lower().startswith("openai")
        and report.get("retriever_search", {}).get("ok")
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
