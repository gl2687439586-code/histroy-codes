#!/usr/bin/env python3
"""Summarize append-only OpenRouter request ledgers without prompt contents."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from common import read_jsonl, utc_stamp


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: List[Dict[str, Any]] = []
    missing: List[str] = []
    for value in args.input:
        path = Path(value)
        if path.exists():
            rows.extend(read_jsonl(path))
        else:
            missing.append(str(path))

    # Response IDs are globally unique.  The hash fallback only protects
    # against an unusual response that omits an ID; it does not expose text.
    unique: Dict[str, Dict[str, Any]] = {}
    duplicate_rows = 0
    for index, row in enumerate(rows):
        key = str(row.get("response_id") or "")
        if not key:
            key = f"missing-id:{index}"
        if key in unique:
            duplicate_rows += 1
            continue
        unique[key] = row

    by_stage: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "requests": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "cost_usd": 0.0,
        }
    )
    for row in unique.values():
        stage = str(row.get("stage") or "unknown")
        target = by_stage[stage]
        target["requests"] += 1
        target["prompt_tokens"] += int(row.get("prompt_tokens", 0))
        target["completion_tokens"] += int(row.get("completion_tokens", 0))
        target["reasoning_tokens"] += int(row.get("reasoning_tokens", 0))
        target["cost_usd"] += float(row.get("cost_usd", 0.0))

    report = {
        "created_at": utc_stamp(),
        "input_ledgers": args.input,
        "missing_ledgers": missing,
        "ledger_rows": len(rows),
        "unique_billed_responses": len(unique),
        "duplicate_rows_ignored": duplicate_rows,
        "providers": dict(Counter(str(row.get("provider") or "unknown") for row in unique.values())),
        "models": dict(Counter(str(row.get("model") or "unknown") for row in unique.values())),
        "by_stage": dict(by_stage),
        "total": {
            "requests": len(unique),
            "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in unique.values()),
            "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in unique.values()),
            "reasoning_tokens": sum(int(row.get("reasoning_tokens", 0)) for row in unique.values()),
            "cost_usd": sum(float(row.get("cost_usd", 0.0)) for row in unique.values()),
        },
        "interpretation": (
            "This is the actual API-request bill reconstructed from unique response IDs. "
            "Per-trajectory costs are counterfactual path costs and must not be summed because tree prefixes are shared."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
