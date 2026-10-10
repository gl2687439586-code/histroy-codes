#!/usr/bin/env python3
"""Offline protocol tests; never contacts OpenRouter."""

import importlib.util
import json
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "hotpot_read", HERE / "score_hotpot_read_openrouter.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def main():
    with tempfile.TemporaryDirectory() as temporary:
        source = Path(temporary) / "queries.jsonl"
        rows = [
            {"query_id": "q1", "query": "Who wrote the novel?"},
            {"query_id": "q2", "query": "Which city is older?"},
        ]
        source.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        metadata, questions = MODULE.load_input(source, 2)
        assert metadata["benchmark"] == "HotpotQA"
        assert questions[0] == {"question_id": "q1", "query": rows[0]["query"]}
        assert MODULE.DIMENSIONS == ("read",)

        prompt = MODULE.load_prompt("read")
        payload = MODULE.request_payload(
            "openai/gpt-5.6-sol", "read", questions[0], prompt, 2048, "medium", 2026
        )
        assert len(payload["messages"]) == 2
        assert payload["reasoning"] == {"effort": "medium", "exclude": True}
        assert payload["seed"] == 2026
        assert payload["provider"] == {
            "require_parameters": True,
            "allow_fallbacks": False,
        }
        schema = payload["response_format"]["json_schema"]
        assert "read_score" in schema["schema"]["properties"]
        assert "search_score" not in schema["schema"]["properties"]

        rating = MODULE.validate_rating(
            {
                "question_id": "q1",
                "query": rows[0]["query"],
                "read_score": 2,
                "confidence": 0.9,
                "basis": "Requires integrating two localized facts.",
            },
            "read",
            questions[0],
        )
        assert rating["read_score"] == 2

    print(
        "HOTPOTQA READ-ONLY SELF-TEST PASSED: JSONL input, exact READ prompt, "
        "one request per query, schema, model parameters, and validation."
    )


if __name__ == "__main__":
    main()
