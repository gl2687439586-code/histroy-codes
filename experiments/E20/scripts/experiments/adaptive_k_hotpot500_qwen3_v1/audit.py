#!/usr/bin/env python3
"""Final protocol and integrity audit for the Adaptive-k-only run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adaptive_k import select_adaptive_k
from common import canonical_hash, read_jsonl, utc_stamp, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--generations", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    roots = read_jsonl(args.roots)
    generations = read_jsonl(args.generations)
    summary = json.loads(Path(args.analysis).read_text(encoding="utf-8"))
    errors, warnings = [], []
    qids = [str(row["sample_id"]) for row in questions]
    root_by_id = {str(row["sample_id"]): row for row in roots}
    generation_by_key = {
        (str(row["sample_id"]), int(row["rollout_id"])): row for row in generations
    }
    if len(qids) != len(set(qids)):
        errors.append("Duplicate question IDs")
    if len(root_by_id) != len(qids):
        errors.append(f"Expected {len(qids)} roots; found {len(root_by_id)}")
    if len(roots) != len(root_by_id):
        errors.append("Duplicate root sample IDs")
    if len(generation_by_key) != len(qids) * 2:
        errors.append(f"Expected {len(qids) * 2} unique generations; found {len(generation_by_key)}")
    if len(generations) != len(generation_by_key):
        errors.append("Duplicate generation sample/rollout keys")

    oversized = []
    for sample_id in qids:
        root = root_by_id.get(sample_id)
        if root is None:
            errors.append(f"Missing root: {sample_id}")
            continue
        if root.get("method") != "adaptive-k-noclass" or int(root.get("retrieval_calls", -1)) != 1:
            errors.append(f"Non-single-pass root: {sample_id}")
        if int(root.get("maximum_allowed_searches", -1)) != 4:
            errors.append(f"Maximum-search declaration mismatch: {sample_id}")
        if int(root.get("candidate_count", -1)) != 1000:
            errors.append(f"Candidate count is not 1000: {sample_id}")
        candidates = root.get("candidates") or []
        if len(candidates) != 1000:
            errors.append(f"Candidate payload length is not 1000: {sample_id}")
        if any(int(item.get("rank", -1)) != rank for rank, item in enumerate(candidates, 1)):
            errors.append(f"Candidate ranks are not contiguous: {sample_id}")
        scores = [item["score"] for item in candidates]
        try:
            recomputed = select_adaptive_k(scores, ignore_extreme_tail=0.1, retrieve_more=5)
            if recomputed != root.get("selection"):
                errors.append(f"Adaptive-k recomputation mismatch: {sample_id}")
        except Exception as exc:
            errors.append(f"Adaptive-k recomputation failed for {sample_id}: {exc}")
            continue
        selected_k = int(root["selection"]["selected_k"])
        selected_documents = root.get("selected_documents") or []
        if len(selected_documents) != selected_k:
            errors.append(f"Selected document count mismatch: {sample_id}")
        selected_projection = [
            {key: item.get(key) for key in ("rank", "id", "title", "score")}
            for item in selected_documents
        ]
        candidate_prefix = [
            {key: item.get(key) for key in ("rank", "id", "title", "score")}
            for item in candidates[:selected_k]
        ]
        if selected_projection != candidate_prefix:
            errors.append(f"Selected documents are not the exact candidate prefix: {sample_id}")
        for rollout in (0, 1):
            row = generation_by_key.get((sample_id, rollout))
            if row is None:
                errors.append(f"Missing generation: {(sample_id, rollout)}")
                continue
            if row.get("root_hash") != root.get("root_hash"):
                errors.append(f"Root hash mismatch: {(sample_id, rollout)}")
            if int(row.get("retrieval_calls", -1)) != 1:
                errors.append(f"Generation reports !=1 retrieval: {(sample_id, rollout)}")
            if int(row.get("adaptive_k", -1)) != selected_k:
                errors.append(f"Generation k mismatch: {(sample_id, rollout)}")
            if row.get("selected_document_ids") != [item.get("id") for item in selected_documents]:
                errors.append(f"Generation document IDs mismatch: {(sample_id, rollout)}")
            if float(row.get("temperature", -1)) != 0.6 or float(row.get("top_p", -1)) != 0.95:
                errors.append(f"Sampling mismatch: {(sample_id, rollout)}")
            if row.get("answer_status") != "valid":
                warnings.append(f"Non-valid answer status {(sample_id, rollout)}: {row.get('answer_status')}")
            if int(row.get("effective_max_new_tokens", 0)) < int(row.get("requested_max_new_tokens", 0)):
                oversized.append((sample_id, rollout))

    if int(summary.get("questions", -1)) != len(qids):
        errors.append("Analysis question count mismatch")
    if int(summary.get("records", -1)) != len(qids) * 2:
        errors.append("Analysis record count mismatch")
    report = {
        "created_at": utc_stamp(), "passed": not errors,
        "questions": len(qids), "roots": len(roots), "generation_records": len(generations),
        "unique_generation_keys": len(generation_by_key),
        "retrieval_calls_expected": len(qids),
        "retrieval_calls_recorded": sum(int(row.get("retrieval_calls", 0)) for row in roots),
        "model_calls_expected": len(qids) * 2,
        "adaptive_k_recomputed_for_all_roots": not any("recomputation" in item for item in errors),
        "no_iterative_search": all(int(row.get("retrieval_calls", -1)) == 1 for row in roots),
        "rollouts_share_frozen_root": not any("Root hash mismatch" in item for item in errors),
        "capacity_adjusted_generation_count": len(oversized),
        "capacity_adjusted_examples": oversized[:20],
        "warnings": warnings, "warning_count": len(warnings),
        "errors": errors, "error_count": len(errors),
        "content_hashes": {
            "roots": canonical_hash(roots), "generations": canonical_hash(generations),
        },
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
