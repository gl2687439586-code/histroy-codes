#!/usr/bin/env python3
"""Run GPT-5.6 Sol natural ReCall or the v6 matched-neutral condition."""

from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import (  # noqa: E402
    ModelClient,
    RECALL_SYSTEM_PROMPT,
    RetrieverClient,
    TokenCounter,
    answer_metrics,
    append_jsonl,
    extract_final_answer,
    extract_last_tag,
    extract_think_text,
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
    load_bamboogle_questions,
    openrouter_runtime_config,
    response_usage,
    sum_event_costs,
)
from openrouter_prompts import (  # noqa: E402
    NEUTRAL_POLICY,
    build_prompt,
    write_prompt_manifest,
)


def recall_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{RECALL_SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n<think>"
    )


def flexible_query(text: str) -> Tuple[str, str]:
    tagged = extract_last_tag(text, "search").strip()
    if tagged:
        return re.sub(r"\s+", " ", tagged), "search_tag"
    open_tagged = re.findall(r"<search>\s*([^<\r\n]+?)\s*$", text or "", flags=re.I)
    if open_tagged:
        return re.sub(r"\s+", " ", open_tagged[-1]).strip(), "open_search_tag"
    json_match = re.search(r'["\']query["\']\s*:\s*["\']([^"\']+)', text, flags=re.I)
    if json_match:
        return re.sub(r"\s+", " ", json_match.group(1)).strip(), "json_query"
    label = re.findall(r"(?:search\s+query|query)\s*:\s*([^\n<>]+)", text, flags=re.I)
    if label:
        return re.sub(r"\s+", " ", label[-1]).strip(), "labeled_query"
    return "", "missing"


def flexible_answer(text: str) -> Tuple[str, str]:
    raw = (text or "").strip()
    answer_block = extract_last_tag(raw, "answer")
    if answer_block:
        return extract_final_answer(raw).strip(), "answer_tag"
    open_answer = re.findall(r"<answer>\s*([^<\r\n]+?)\s*$", raw, flags=re.I)
    if open_answer:
        return open_answer[-1].strip(), "open_answer_tag"
    if re.search(r"\\boxed\s*\{", raw):
        return extract_final_answer(raw).strip(), "boxed"
    parts = re.split(r"</think>", raw, maxsplit=1, flags=re.I)
    tail = parts[-1].strip() if len(parts) == 2 else raw
    cues = re.findall(
        r"(?:the\s+final\s+answer\s+is|final\s+answer|answer)\s*:\s*(.+)",
        tail,
        flags=re.I,
    )
    if cues:
        return cues[-1].strip(), "answer_cue"
    cleaned = re.sub(r"</?(?:think|answer)>\s*", "", tail, flags=re.I).strip()
    return (cleaned, "natural_language") if cleaned else ("", "empty")


def query_fallback(text: str, question: str) -> Tuple[str, str]:
    query, source = flexible_query(text)
    if query:
        return query, source
    # A natural run is not assigned Search. The fallback is only used when the
    # model explicitly emits an opening Search marker but fails to close it.
    opening = re.search(r"<search>\s*([^<\n]{1,300})", text, flags=re.I)
    if opening:
        return re.sub(r"\s+", " ", opening.group(1)).strip(), "unclosed_search"
    return "", "missing"


def document_fields(tokenizer: TokenCounter, documents: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    output: List[Dict[str, Any]] = []
    total = 0
    for document in documents:
        item = dict(document)
        count = tokenizer.count(str(item.get("contents", "")))
        item["token_count_actual"] = count
        item["token_count_cost"] = count
        total += count
        output.append(item)
    return output, total


def run_sample(
    sample: Dict[str, Any],
    condition: str,
    model: ModelClient,
    retriever: RetrieverClient,
    tokenizer: TokenCounter,
    args: argparse.Namespace,
    root: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if condition == "neutral" and root is None:
        raise RuntimeError("Neutral condition requires the matched GPT-5.6-Sol first-Search root")
    prompt = str(root["checkpoint_prefix"]) if root else recall_prompt(sample["question"])
    events: List[Dict[str, Any]] = []
    queries: List[str] = [str(root["initial_query"])] if root else []
    documents: List[Dict[str, Any]] = list(root.get("initial_documents") or []) if root else []
    document_keys = {
        str(document.get("key") or document.get("id") or document.get("contents", ""))
        for document in documents
    }
    first_checkpoint: Optional[Dict[str, Any]] = (
        {
            "checkpoint_prefix": str(root["checkpoint_prefix"]),
            "initial_query": str(root["initial_query"]),
            "initial_result": str(root["initial_result"]),
            "initial_documents": documents,
            "root_checkpoint_sha1": str(root["root_checkpoint_sha1"]),
        }
        if root
        else None
    )
    if root:
        root_event = dict(root.get("root_generation_event") or {})
        root_event.setdefault("event_index", 0)
        root_event.setdefault("decision_index", 0)
        root_event.setdefault("action", "Search")
        root_event.setdefault("action_code", "S")
        root_event["shared_openrouter_root"] = True
        root_event["prompt_condition"] = "shared_root_before_neutral_intervention"
        events.append(root_event)
    final_answer = ""
    final_generation = ""
    terminated = False

    start_step = 1 if root else 0
    for step in range(start_step, args.max_steps):
        if condition == "blank":
            call_prompt = prompt
        else:
            call_prompt = build_prompt(sample["question"], queries, documents, [], NEUTRAL_POLICY)
        input_tokens = tokenizer.count(call_prompt)
        if input_tokens + args.max_new_tokens > args.context_length:
            raise RuntimeError(
                f"Context overflow at step {step}: input={input_tokens}, max_new={args.max_new_tokens}"
            )
        response = model.generate(
            prompt=call_prompt,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            stop=("</search>", "</answer>"),
            seed=args.seed + int(stable_hash(f"{sample['sample_id']}|{condition}|{step}", 8), 16) % 100000000,
        )
        generated = response.text
        # ReCall starts Blank inside <think>; matched-neutral starts with no
        # prefilled tag and may naturally use Qwen's own thinking wrapper.
        parsed = ("<think>" + generated) if condition == "blank" and step == 0 else generated
        query, query_source = query_fallback(parsed, sample["question"])
        answer_tag = extract_last_tag(parsed, "answer")
        answer_candidate, answer_source = flexible_answer(parsed)

        if query:
            retrieval = retriever.search(query, args.top_k)
            enriched, retrieved_tokens = document_fields(tokenizer, retrieval.documents)
            new_evidence = 0
            for document in enriched:
                key = str(document.get("key", ""))
                if key not in document_keys:
                    document_keys.add(key)
                    new_evidence += int(document.get("token_count_cost", 0))
            if query_source == "search_tag":
                canonical_search = ""
            elif query_source == "open_search_tag":
                canonical_search = "</search>"
            else:
                canonical_search = f"\n<search>{query}</search>"
            tool_append = f"{canonical_search}\n<result>{retrieval.result_text}\n</result>\n"
            if condition == "blank":
                prompt = call_prompt + generated + tool_append
            else:
                # Neutral calls reserialize the state at every decision, exactly
                # like controlled calls; prose is not mislabeled as a Read.
                prompt = call_prompt + generated + tool_append
            usage = response_usage(response, input_tokens, tokenizer.count(generated))
            event = {
                "event_index": step,
                "decision_index": step,
                "action": "Search",
                "action_code": "S",
                "prompt_condition": condition,
                "raw_prompt": call_prompt,
                "raw_model_response": generated,
                "query": query,
                "query_source": query_source,
                "documents": enriched,
                "tool": {
                    "name": "wikipedia_search",
                    "top_k": args.top_k,
                    "endpoint": retrieval.endpoint,
                    "raw_response": retrieval.raw,
                    "raw_result_text": retrieval.result_text,
                    "documents": enriched,
                },
                **usage,
                "answer_tokens": 0,
                "retrieved_docs": len(enriched),
                "retrieved_tokens_total": retrieved_tokens,
                "new_evidence_tokens": new_evidence,
                "read_input_tokens": 0,
                "read_output_tokens": 0,
                "latency_seconds": response.latency_seconds + retrieval.latency_seconds,
                "model_meta": public_model_meta(response.raw),
            }
            events.append(event)
            queries.append(query)
            documents.extend(enriched)
            if first_checkpoint is None:
                checkpoint_prefix = call_prompt + generated + tool_append
                first_checkpoint = {
                    "checkpoint_prefix": checkpoint_prefix,
                    "initial_query": query,
                    "initial_result": retrieval.result_text,
                    "initial_documents": enriched,
                    "root_checkpoint_sha1": stable_hash(checkpoint_prefix, 40),
                }
            continue

        # No routed Search means the natural policy terminates. If tags are
        # missing we retain a flexible natural-language answer rather than
        # deleting the question from the baseline.
        final_answer = answer_candidate
        final_generation = parsed
        usage = response_usage(response, input_tokens, tokenizer.count(generated))
        event = {
            "event_index": step,
            "decision_index": step,
            "action": "Answer",
            "action_code": "A",
            "terminal": True,
            "prompt_condition": condition,
            "raw_prompt": call_prompt,
            "raw_model_response": generated,
            "answer_extraction_source": "answer_tag" if answer_tag else answer_source,
            **usage,
            "answer_tokens": tokenizer.count(final_answer),
            "retrieved_docs": 0,
            "retrieved_tokens_total": 0,
            "new_evidence_tokens": 0,
            "read_input_tokens": 0,
            "read_output_tokens": 0,
            "latency_seconds": response.latency_seconds,
            "model_meta": public_model_meta(response.raw),
        }
        events.append(event)
        terminated = True
        break

    if not terminated:
        # Natural runs have a finite safety cap. Generate a tool-free final
        # answer from the already acquired state without changing search cost.
        from openrouter_prompts import FORCED_ANSWER_POLICY

        call_prompt = build_prompt(sample["question"], queries, documents, [], FORCED_ANSWER_POLICY)
        response = model.generate(
            prompt=call_prompt,
            temperature=args.temperature,
            max_new_tokens=args.answer_max_new_tokens,
            stop=("</answer>",),
            seed=args.seed + int(stable_hash(f"{sample['sample_id']}|{condition}|forced", 8), 16) % 100000000,
        )
        final_generation = response.text
        final_answer, answer_source = flexible_answer(final_generation)
        usage = response_usage(
            response, tokenizer.count(call_prompt), tokenizer.count(response.text)
        )
        events.append(
            {
                "event_index": len(events),
                "decision_index": len(events),
                "action": "ForcedAnswer",
                "action_code": "A",
                "terminal": True,
                "forced_by_max_steps": True,
                "prompt_condition": condition,
                "raw_prompt": call_prompt,
                "raw_model_response": response.text,
                "answer_extraction_source": answer_source,
                **usage,
                "answer_tokens": tokenizer.count(final_answer),
                "retrieved_docs": 0,
                "retrieved_tokens_total": 0,
                "new_evidence_tokens": 0,
                "read_input_tokens": 0,
                "read_output_tokens": 0,
                "latency_seconds": response.latency_seconds,
                "model_meta": public_model_meta(response.raw),
            }
        )

    metrics = answer_metrics(final_answer, sample["gold_answers"])
    record: Dict[str, Any] = {
        "schema_version": PROTOCOL_VERSION,
        "created_at": utc_stamp(),
        "condition": condition,
        "prompt_condition": condition,
        "sample_id": sample["sample_id"],
        "question": sample["question"],
        "gold_answers": sample["gold_answers"],
        "model_name": EXPECTED_MODEL_ID,
        "model_path": args.model_path,
        "model_url": args.model_url,
        "model_backend": "openrouter_chat_completions",
        "tokenizer_path": args.tokenizer_path,
        "old_checkpoint_used": False,
        "openrouter_runtime": openrouter_runtime_config(),
        "first_search_observed": first_checkpoint is not None,
        "root_origin": "openrouter_natural_root" if first_checkpoint is not None else "openrouter_no_search",
        "final_answer": final_answer,
        "raw_answer_generation": final_generation,
        "metrics": metrics,
        **metrics,
        **sum_event_costs(events),
        "queries": queries,
        "document_keys_unique": sorted(document_keys),
        "events": events,
        "paper_prediction": sample.get("paper_prediction", ""),
        "paper_metric_score": sample.get("paper_metric_score", {}),
        "token_count_is_approximate": tokenizer.approximate,
    }
    if first_checkpoint:
        record.update(first_checkpoint)
    if root:
        record["root_origin"] = root.get("root_origin", "unknown")
        record["blank_record_sha1"] = root.get("blank_record_sha1", "")
    record["record_sha1"] = stable_hash(json.dumps(record, ensure_ascii=False, sort_keys=True), 40)
    return record


def completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("sample_id", "")) for row in read_jsonl(path) if row.get("sample_id")}


def failed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {str(row.get("sample_id", "")) for row in read_jsonl(path) if row.get("sample_id")}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", choices=("blank", "neutral"), required=True)
    parser.add_argument("--intermediate_data", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--roots", default="", help="Required for neutral: exported GPT-5.6 root JSONL")
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
    parser.add_argument("--max_steps", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--answer_max_new_tokens", type=int, default=2048)
    parser.add_argument("--context_length", type=int, default=32768)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--sample_ids", default="")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max_usd", type=float, default=0.0, help="Fail after this stage spend; 0 disables")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry_failures",
        action="store_true",
        help="Retry IDs already present in failures.jsonl (default skips them to avoid an endless full pipeline)",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    assert_openrouter_configuration(args.model_path, args.model_url, require_key=not args.dry_run)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    requested = [part.strip() for part in args.sample_ids.split(",") if part.strip()]
    samples, load_report = load_bamboogle_questions(args.intermediate_data, args.limit, requested)
    roots_by_id: Dict[str, Dict[str, Any]] = {}
    if args.condition == "neutral":
        if not args.roots:
            raise ValueError("Neutral condition requires --roots")
        roots_by_id = {str(row["sample_id"]): row for row in read_jsonl(args.roots)}
        missing = [sample["sample_id"] for sample in samples if sample["sample_id"] not in roots_by_id]
        load_report["missing_matched_roots"] = missing
        samples = [sample for sample in samples if sample["sample_id"] in roots_by_id]
        load_report["loaded_samples_after_root_match"] = len(samples)
    tokenizer = TokenCounter(args.tokenizer_path)
    load_report["token_count_is_approximate"] = tokenizer.approximate
    (output_dir / "load_report.json").write_text(json.dumps(load_report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_prompt_manifest(output_dir / "prompt_manifest.json")
    manifest = {
        "schema_version": PROTOCOL_VERSION,
        "created_at": utc_stamp(),
        "condition": args.condition,
        "arguments": vars(args),
        "model_name": EXPECTED_MODEL_ID,
        "model_backend": "openrouter_chat_completions",
        "openrouter_runtime": openrouter_runtime_config(),
        "document_tokenizer_path": args.tokenizer_path,
        "old_checkpoint_used": False,
        "loaded_samples": len(samples),
    }
    manifest["resume_signature"] = {
        "schema_version": PROTOCOL_VERSION,
        "condition": args.condition,
        "selected_sample_ids": [sample["sample_id"] for sample in samples],
        "model_path": args.model_path,
        "model_url": args.model_url,
        "retriever_url": args.retriever_url,
        "retriever_mode": args.retriever_mode,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "max_steps": args.max_steps,
        "max_new_tokens": args.max_new_tokens,
        "answer_max_new_tokens": args.answer_max_new_tokens,
        "context_length": args.context_length,
        "openrouter_runtime": openrouter_runtime_config(),
        "root_hashes": {
            sid: str(root.get("root_checkpoint_sha1", "")) for sid, root in roots_by_id.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    trajectory_path = output_dir / "trajectories.jsonl"
    if trajectory_path.exists():
        if not args.resume:
            raise FileExistsError(f"{trajectory_path} exists; pass --resume or use a new output directory")
        if not manifest_path.exists():
            raise RuntimeError("Cannot safely resume: manifest.json is missing")
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("resume_signature") != manifest["resume_signature"]:
            raise RuntimeError("Cannot safely resume after changing samples, roots, model, or generation settings")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({**manifest, "network_calls_made": False}, ensure_ascii=False, indent=2))
        return 0

    identity = enforce_model_identity(args.model_url, args.model_path)
    (output_dir / "model_identity.json").write_text(json.dumps(identity, ensure_ascii=False, indent=2), encoding="utf-8")
    output_path = trajectory_path
    failure_path = output_dir / "failures.jsonl"
    done = completed_ids(output_path) if args.resume else set()
    failed_before = failed_ids(failure_path) if args.resume and not args.retry_failures else set()
    model = ModelClient(
        args.model_url,
        args.model_path,
        "openrouter",
        args.timeout,
        3,
        max_usd=args.max_usd,
    )
    retriever = RetrieverClient(args.retriever_url, args.sandbox_url, args.retriever_mode, args.timeout, 3)
    succeeded = failed = 0
    for sample in samples:
        if sample["sample_id"] in done or sample["sample_id"] in failed_before:
            continue
        try:
            record = run_sample(
                sample,
                args.condition,
                model,
                retriever,
                tokenizer,
                args,
                roots_by_id.get(sample["sample_id"]),
            )
            append_jsonl(output_path, [record])
            succeeded += 1
            print(f"[ok] {sample['sample_id']} condition={args.condition} searches={record['search_calls']} f1={record['f1']:.3f}")
        except Exception as exc:
            failed += 1
            append_jsonl(failure_path, [{
                "created_at": utc_stamp(),
                "sample_id": sample["sample_id"],
                "condition": args.condition,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "traceback": traceback.format_exc(),
            }])
            print(f"[failed] {sample['sample_id']}: {exc}", file=sys.stderr)
    final_complete = len(completed_ids(output_path))
    final_failed_ids = failed_ids(failure_path) - completed_ids(output_path)
    status = {
        "finished_at": utc_stamp(),
        "condition": args.condition,
        "succeeded_this_attempt": succeeded,
        "failed_this_attempt": failed,
        "complete_total": final_complete,
        "unresolved_failed_total": len(final_failed_ids),
        "unresolved_failed_ids": sorted(final_failed_ids),
        "api_cost_usd_this_process": model.process_cost_usd,
        "api_cost_usd_stage_cumulative_ledger": model.cumulative_cost_usd,
        "api_cost_note": "Actual OpenRouter charges observed in this process, including failed/retried generations.",
    }
    stage_complete = final_complete == len(samples) and not final_failed_ids
    status["expected_total"] = len(samples)
    status["stage_complete"] = stage_complete
    status["exit_code"] = 0 if stage_complete else 1
    (output_dir / "run_status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(status, ensure_ascii=False, indent=2))
    if not stage_complete:
        print(
            "Stage incomplete: refusing to advance to the next experimental stage. "
            f"complete={final_complete}/{len(samples)}, unresolved_failures={len(final_failed_ids)}. "
            f"Inspect {failure_path} before resuming.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
