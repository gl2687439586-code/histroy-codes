#!/usr/bin/env python3
"""Small connectivity probe for model, retriever, and optional sandbox."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import ModelClient, RetrieverClient, utc_stamp  # noqa: E402


def get_health(url: str, timeout: float) -> Dict[str, Any]:
    started = time.time()
    try:
        response = requests.get(url.rstrip("/") + "/health", timeout=timeout)
        return {
            "ok": response.ok,
            "status_code": response.status_code,
            "latency_seconds": time.time() - started,
            "body": response.text[:2000],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "latency_seconds": time.time() - started}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_url", default="http://127.0.0.1:9000")
    parser.add_argument("--retriever_url", default="http://127.0.0.1:6008")
    parser.add_argument("--sandbox_url", default="http://127.0.0.1:6006")
    parser.add_argument("--model_path", default="/home/researcher/smartsearch_workspace/models/SmartSearch-3B")
    parser.add_argument("--api_mode", choices=("sglang", "openai"), default="sglang")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--skip_sandbox", action="store_true")
    args = parser.parse_args()

    report: Dict[str, Any] = {"created_at": utc_stamp()}
    report["retriever_health"] = get_health(args.retriever_url, min(args.timeout, 30.0))
    try:
        retriever = RetrieverClient(args.retriever_url, timeout=args.timeout, retries=1)
        result = retriever.search("Citibank founding year", top_k=5)
        report["retriever_search"] = {
            "ok": len(result.documents) > 0,
            "documents": len(result.documents),
            "ids": [doc["id"] for doc in result.documents],
            "latency_seconds": result.latency_seconds,
            "result_preview": result.result_text[:500],
        }
    except Exception as exc:
        report["retriever_search"] = {"ok": False, "error": str(exc)}

    try:
        model = ModelClient(args.model_url, args.model_path, args.api_mode, args.timeout, retries=1)
        response = model.generate(
            "<|im_start|>user\nReply with OK.<|im_end|>\n<|im_start|>assistant\n",
            temperature=0.0,
            max_new_tokens=8,
            stop=("<|im_end|>",),
            seed=2026,
        )
        report["model_generate"] = {
            "ok": bool(response.text),
            "endpoint": response.endpoint,
            "latency_seconds": response.latency_seconds,
            "text": response.text,
            "finish_reason": response.finish_reason,
        }
    except Exception as exc:
        report["model_generate"] = {"ok": False, "error": str(exc)}

    if not args.skip_sandbox:
        try:
            sandbox = RetrieverClient(
                args.retriever_url,
                sandbox_url=args.sandbox_url,
                mode="sandbox",
                timeout=args.timeout,
                retries=1,
            )
            result = sandbox.search("Citibank founding year", top_k=5)
            report["sandbox_execute"] = {
                "ok": bool(result.result_text),
                "endpoint": result.endpoint,
                "latency_seconds": result.latency_seconds,
                "result_preview": result.result_text[:500],
            }
        except Exception as exc:
            report["sandbox_execute"] = {"ok": False, "error": str(exc)}

    ok = bool(report.get("retriever_search", {}).get("ok") and report.get("model_generate", {}).get("ok"))
    if not args.skip_sandbox:
        ok = ok and bool(report.get("sandbox_execute", {}).get("ok"))
    report["passed"] = ok
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
