#!/usr/bin/env python3
"""Export GPT-5.6 Sol first-Search checkpoints from the Blank condition."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    ModelClient,
    RetrieverClient,
    TokenCounter,
    append_jsonl,
    extract_last_tag,
    public_model_meta,
    read_jsonl,
    stable_hash,
    utc_stamp,
)
from openrouter_common import (  # noqa: E402
    EXPECTED_MODEL_ID,
    PROTOCOL_VERSION,
    assert_openrouter_configuration,
    enforce_model_identity,
    openrouter_runtime_config,
    response_usage,
)
from openrouter_prompts import SEARCH_POLICY, build_prompt, write_prompt_manifest  # noqa: E402


def extract_query(text: str, fallback: str) -> Tuple[str, str]:
    tagged = extract_last_tag(text, "search").strip()
    if tagged:
        return re.sub(r"\s+", " ", tagged), "search_tag"
    open_tagged = re.findall(r"<search>\s*([^<\r\n]+?)\s*$", text or "", flags=re.I)
    if open_tagged:
        return re.sub(r"\s+", " ", open_tagged[-1]).strip(), "open_search_tag"
    json_match = re.search(r'["\']query["\']\s*:\s*["\']([^"\']+)', text, flags=re.I)
    if json_match:
        return re.sub(r"\s+", " ", json_match.group(1)).strip(), "json_query"
    labeled = re.findall(r"(?:search\s+query|query)\s*:\s*([^\n<>]+)", text, flags=re.I)
    if labeled:
        return re.sub(r"\s+", " ", labeled[-1]).strip(), "labeled_query"
    return re.sub(r"\s+", " ", fallback).strip(), "fallback_original_question"


def enrich_documents(tokenizer: TokenCounter, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output = []
    for document in documents:
        item = dict(document)
        count = tokenizer.count(str(item.get("contents", "")))
        item["token_count_actual"] = count
        item["token_count_cost"] = count
        output.append(item)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank", required=True, help="Blank trajectories.jsonl")
    parser.add_argument("--output", required=True, help="GPT-5.6 Sol root checkpoint JSONL")
    parser.add_argument("--model_url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--model_path", default=EXPECTED_MODEL_ID, help="OpenRouter model slug")
    parser.add_argument(
        "--tokenizer_path",
        default="/home/researcher/smartsearch_rebuild/models/Qwen3-8B",
        help="Local tokenizer used only for fixed document-cost compatibility",
    )
    parser.add_argument("--retriever_url", default="http://127.0.0.1:6008")
    parser.add_argument("--sandbox_url", default="http://127.0.0.1:6006")
    parser.add_argument("--retriever_mode", choices=("direct", "sandbox"), default="direct")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max_usd", type=float, default=0.0, help="Fail after this stage spend; 0 disables")
    parser.add_argument("--natural_only", action="store_true", help="Exclude Blank samples that never searched")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    assert_openrouter_configuration(args.model_path, args.model_url)
    identity = enforce_model_identity(args.model_url, args.model_path)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_prompt_manifest(output_path.parent / "prompt_manifest.json")
    blank_rows = read_jsonl(args.blank)
    resume_signature = {
        "schema_version": PROTOCOL_VERSION,
        "blank_path": str(Path(args.blank).resolve()),
        "blank_record_hashes": {
            str(row.get("sample_id", "")): str(row.get("record_sha1", ""))
            for row in blank_rows
        },
        "model_path": args.model_path,
        "model_url": args.model_url,
        "tokenizer_path": args.tokenizer_path,
        "retriever_url": args.retriever_url,
        "retriever_mode": args.retriever_mode,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "natural_only": args.natural_only,
        "openrouter_runtime": openrouter_runtime_config(),
    }
    manifest_path = output_path.parent / "root_export_manifest.json"
    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "created_at": utc_stamp(),
        "arguments": vars(args),
        "resume_signature": resume_signature,
    }
    if output_path.exists():
        if not args.resume:
            raise FileExistsError(f"{output_path} exists; pass --resume or use a new output")
        if not manifest_path.exists():
            raise RuntimeError("Cannot safely resume root export: root_export_manifest.json is missing")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("resume_signature") != resume_signature:
            raise RuntimeError("Cannot safely resume root export after changing data or API settings")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    existing = {
        str(row.get("sample_id", "")) for row in read_jsonl(output_path)
    } if args.resume and output_path.exists() else set()
    tokenizer = TokenCounter(args.tokenizer_path)
    model = ModelClient(
        args.model_url,
        args.model_path,
        "openrouter",
        args.timeout,
        3,
        max_usd=args.max_usd,
    )
    retriever = RetrieverClient(args.retriever_url, args.sandbox_url, args.retriever_mode, args.timeout, 3)
    natural = forced = excluded = 0
    for blank in blank_rows:
        sample_id = str(blank.get("sample_id", ""))
        if not sample_id or sample_id in existing:
            continue
        if blank.get("condition") != "blank" or str(blank.get("model_name", "")).lower() != EXPECTED_MODEL_ID:
            raise RuntimeError(f"Non-GPT-5.6-Sol Blank record rejected for {sample_id}")
        if blank.get("first_search_observed"):
            checkpoint_prefix = str(blank["checkpoint_prefix"])
            row = {
                "schema_version": PROTOCOL_VERSION,
                "created_at": utc_stamp(),
                "sample_id": sample_id,
                "question": blank["question"],
                "gold_answers": blank.get("gold_answers", []),
                "model_name": EXPECTED_MODEL_ID,
                "model_path": args.model_path,
                "model_url": args.model_url,
                "old_checkpoint_used": False,
                "openrouter_runtime": openrouter_runtime_config(),
                "root_origin": "openrouter_natural_root",
                "checkpoint_source": "openrouter_blank.first_natural_search",
                "checkpoint_prefix": checkpoint_prefix,
                "root_checkpoint_sha1": stable_hash(checkpoint_prefix, 40),
                "initial_query": blank["initial_query"],
                "initial_result": blank["initial_result"],
                "initial_documents": blank.get("initial_documents", []),
                "blank_prediction": blank.get("final_answer", ""),
                "blank_metrics": blank.get("metrics", {}),
                "blank_search_calls": int(blank.get("search_calls", 0)),
                "blank_record_sha1": blank.get("record_sha1", ""),
                "root_generation_event": blank.get("events", [None])[0],
            }
            natural += 1
        else:
            if args.natural_only:
                excluded += 1
                continue
            prompt = build_prompt(blank["question"], [], [], [], SEARCH_POLICY)
            response = model.generate(
                prompt=prompt,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                stop=("</search>",),
                seed=args.seed + int(stable_hash(f"{sample_id}|forced-root", 8), 16) % 100000000,
            )
            query, query_source = extract_query(response.text, blank["question"])
            retrieval = retriever.search(query, args.top_k)
            documents = enrich_documents(tokenizer, retrieval.documents)
            if query_source == "search_tag":
                canonical_search = ""
            elif query_source == "open_search_tag":
                canonical_search = "</search>"
            else:
                canonical_search = f"\n<search>{query}</search>"
            append = f"{canonical_search}\n<result>{retrieval.result_text}\n</result>\n"
            checkpoint_prefix = prompt + response.text + append
            usage = response_usage(
                response, tokenizer.count(prompt), tokenizer.count(response.text)
            )
            row = {
                "schema_version": PROTOCOL_VERSION,
                "created_at": utc_stamp(),
                "sample_id": sample_id,
                "question": blank["question"],
                "gold_answers": blank.get("gold_answers", []),
                "model_name": EXPECTED_MODEL_ID,
                "model_path": args.model_path,
                "model_url": args.model_url,
                "old_checkpoint_used": False,
                "openrouter_runtime": openrouter_runtime_config(),
                "root_origin": "openrouter_forced_root",
                "checkpoint_source": "openrouter_forced_root.after_blank_no_search",
                "checkpoint_prefix": checkpoint_prefix,
                "root_checkpoint_sha1": stable_hash(checkpoint_prefix, 40),
                "initial_query": query,
                "initial_result": retrieval.result_text,
                "initial_documents": documents,
                "blank_prediction": blank.get("final_answer", ""),
                "blank_metrics": blank.get("metrics", {}),
                "blank_search_calls": int(blank.get("search_calls", 0)),
                "blank_record_sha1": blank.get("record_sha1", ""),
                "root_generation_event": {
                    "action": "Search",
                    "action_code": "S",
                    "forced_root": True,
                    "raw_prompt": prompt,
                    "raw_model_response": response.text,
                    "query": query,
                    "query_source": query_source,
                    "documents": documents,
                    "raw_result_text": retrieval.result_text,
                    "model_meta": public_model_meta(response.raw),
                    **usage,
                    "answer_tokens": 0,
                    "retrieved_docs": len(documents),
                    "retrieved_tokens_total": sum(int(doc.get("token_count_cost", 0)) for doc in documents),
                    "new_evidence_tokens": sum(int(doc.get("token_count_cost", 0)) for doc in documents),
                    "read_input_tokens": 0,
                    "read_output_tokens": 0,
                    "latency_seconds": response.latency_seconds + retrieval.latency_seconds,
                },
            }
            forced += 1
        append_jsonl(output_path, [row])
        print(f"[ok] {sample_id} root_origin={row['root_origin']}")
    report = {
        "created_at": utc_stamp(),
        "model_identity": identity,
        "natural_roots_exported": natural,
        "forced_roots_exported": forced,
        "no_search_samples_excluded": excluded,
        "output": str(output_path),
        "api_cost_usd_this_process": model.process_cost_usd,
        "api_cost_usd_stage_cumulative_ledger": model.cumulative_cost_usd,
        "api_cost_note": "Actual OpenRouter charges observed in this process, including failed/retried generations.",
    }
    (output_path.parent / "root_export_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
