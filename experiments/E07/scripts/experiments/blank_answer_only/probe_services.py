#!/usr/bin/env python3
"""Read-only health, identity, and generation probe for a Qwen3 model replica."""

import argparse
import json
import time

import requests


def request(method, url, timeout, **kwargs):
    started = time.time()
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        return {
            "ok": True, "status_code": response.status_code,
            "latency_seconds": time.time() - started, "body": response.text[:1000],
        }
    except Exception as exc:
        return {"ok": False, "latency_seconds": time.time() - started, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_url", default="http://127.0.0.1:9101")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()
    base = args.model_url.rstrip("/")
    report = {
        "model_health": request("GET", base + "/health", args.timeout),
        "model_identity": request("GET", base + "/get_model_info", args.timeout),
        "model_generate": request("POST", base + "/generate", args.timeout, json={
            "text": "<|im_start|>user\nReply with OK.\n/no_think<|im_end|>\n<|im_start|>assistant\n",
            "sampling_params": {"temperature": 0.0, "max_new_tokens": 16},
        }),
    }
    identity = report["model_identity"].get("body", "").lower()
    report["passed"] = all(x.get("ok") for x in report.values()) \
        and "qwen3-8b" in identity and "smartsearch-3b" not in identity
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
