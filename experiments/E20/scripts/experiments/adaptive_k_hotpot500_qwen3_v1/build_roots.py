#!/usr/bin/env python3
"""Perform exactly one scored Top-N retrieval and freeze Adaptive-k roots."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

from adaptive_k import parse_flashrag_scored_response, select_adaptive_k
from common import (
    SCHEMA_VERSION, append_jsonl, canonical_hash, extract_title, file_sha256,
    paper_qa_prompt, qwen_chat_prompt, read_jsonl, text_sha256, utc_stamp,
    write_json,
)


class ScoredRetrieverClient:
    def __init__(self, base_url: str, timeout: float, retries: int):
        import requests
        self.url = base_url.rstrip("/") + "/search"
        self.timeout = timeout
        self.retries = retries
        self.session = requests.Session()
        self.session.trust_env = False

    def search(self, query: str, candidate_count: int) -> Tuple[List[Dict[str, Any]], List[float], float]:
        last_error: Exception | None = None
        for attempt in range(self.retries):
            started = time.time()
            try:
                response = self.session.post(
                    self.url,
                    json={"query": query, "top_n": candidate_count, "return_score": True},
                    timeout=self.timeout,
                )
                if not response.ok:
                    raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1000]}")
                documents, scores = parse_flashrag_scored_response(response.json())
                if len(documents) != candidate_count:
                    raise RuntimeError(
                        f"Retriever returned {len(documents)} candidates; expected {candidate_count}"
                    )
                return documents, scores, time.time() - started
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.retries:
                    time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(f"Retriever failed after {self.retries} attempts: {last_error}")


def signature(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "adaptive-k-noclass",
        "paper": "Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-k",
        "retrieval_calls_per_question": 1,
        "maximum_allowed_searches": 4,
        "candidate_count": args.candidate_count,
        "strategy": "largest_gap",
        "ignore_extreme_tail": args.ignore_extreme_tail,
        "retrieve_more": args.retrieve_more,
        "retriever_embedding": "intfloat/e5-base-v2 (existing environment adaptation)",
        "corpus": "wiki18_100w.jsonl",
        "questions_sha256": file_sha256(args.questions),
        "model_path": str(Path(args.model_path).resolve()),
        "prompt_template_sha256": text_sha256(paper_qa_prompt("{question}", "{context}")),
        "limit": args.limit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--retriever_url", default="http://127.0.0.1:6008")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--candidate_count", type=int, default=1000)
    parser.add_argument("--ignore_extreme_tail", type=float, default=0.1)
    parser.add_argument("--retrieve_more", type=int, default=5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    questions = read_jsonl(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        raise RuntimeError("No questions selected")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "roots.jsonl"
    failures_path = output / "failures.jsonl"
    manifest_path = output / "manifest.json"
    run_signature = signature(args)
    signature_hash = canonical_hash(run_signature)
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("signature_hash") != signature_hash:
            raise RuntimeError("Root output directory contains a different run signature")

    existing_rows = read_jsonl(records_path) if records_path.exists() else []
    for row in existing_rows:
        if row.get("signature_hash") != signature_hash:
            raise RuntimeError("Existing root record has a different run signature")
    existing = {str(row["sample_id"]): row for row in existing_rows}
    if len(existing) != len(existing_rows):
        raise RuntimeError("Duplicate sample_id in existing roots")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True
    )
    retriever = ScoredRetrieverClient(args.retriever_url, args.timeout, args.retries)
    succeeded = failed = 0
    started_all = time.time()
    for offset, question in enumerate(questions):
        sample_id = str(question["sample_id"])
        if sample_id in existing:
            continue
        try:
            documents, scores, elapsed = retriever.search(
                str(question["question"]), args.candidate_count
            )
            selection = select_adaptive_k(
                scores,
                ignore_extreme_tail=args.ignore_extreme_tail,
                retrieve_more=args.retrieve_more,
            )
            selected_k = int(selection["selected_k"])
            selected_documents = []
            candidates = []
            for rank, (document, score) in enumerate(zip(documents, scores), 1):
                title = extract_title(document["contents"])
                metadata = {
                    "rank": rank,
                    "id": document["id"],
                    "title": title,
                    "score": score,
                }
                candidates.append(metadata)
                if rank <= selected_k:
                    selected_documents.append({
                        **metadata,
                        "contents": document["contents"],
                        "contents_sha256": text_sha256(document["contents"]),
                    })
            context = "\n\n".join(item["contents"] for item in selected_documents)
            user_prompt = paper_qa_prompt(str(question["question"]), context)
            chat_prompt = qwen_chat_prompt(tokenizer, user_prompt)
            record = {
                "schema_version": SCHEMA_VERSION,
                "signature_hash": signature_hash,
                "sample_index": int(question.get("sample_index", offset)),
                "sample_id": sample_id,
                "question": str(question["question"]),
                "method": "adaptive-k-noclass",
                "retrieval_query": str(question["question"]),
                "retrieval_calls": 1,
                "maximum_allowed_searches": 4,
                "candidate_count": len(candidates),
                "selection": selection,
                "candidates": candidates,
                "selected_documents": selected_documents,
                "selected_context_sha256": text_sha256(context),
                "selected_context_token_count": len(tokenizer.encode(context, add_special_tokens=False)),
                "reader_user_prompt_token_count": len(tokenizer.encode(user_prompt, add_special_tokens=False)),
                "reader_chat_prompt_token_count": len(tokenizer.encode(chat_prompt, add_special_tokens=False)),
                "retrieval_latency_seconds": elapsed,
            }
            record["root_hash"] = canonical_hash(record)
            append_jsonl(records_path, record)
            existing[sample_id] = record
            succeeded += 1
            elapsed_all = max(time.time() - started_all, 1e-9)
            remaining = len(questions) - len(existing)
            eta_hours = (elapsed_all / max(succeeded + failed, 1)) * remaining / 3600
            print(
                f"[root-ok] {sample_id} k={selected_k} candidates={len(candidates)} "
                f"prompt_tokens={record['reader_chat_prompt_token_count']} eta={eta_hours:.2f}h",
                flush=True,
            )
        except Exception as exc:
            failed += 1
            append_jsonl(failures_path, {
                "created_at": utc_stamp(), "sample_id": sample_id,
                "error_type": type(exc).__name__, "error": str(exc),
                "signature_hash": signature_hash,
            })
            print(f"[root-failed] {sample_id}: {exc}", flush=True)

    ordered_ids = [str(row["sample_id"]) for row in questions]
    missing = [sample_id for sample_id in ordered_ids if sample_id not in existing]
    manifest = {
        "created_at": utc_stamp(),
        "signature": run_signature,
        "signature_hash": signature_hash,
        "expected_roots": len(questions),
        "roots": len(existing),
        "succeeded_this_attempt": succeeded,
        "failed_this_attempt": failed,
        "unresolved_failures": len(missing),
        "unresolved_sample_ids": missing,
        "complete": not missing,
        "single_pass_retrieval_verified": all(
            int(row.get("retrieval_calls", -1)) == 1 for row in existing.values()
        ),
        "output": str(records_path.resolve()),
    }
    write_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
