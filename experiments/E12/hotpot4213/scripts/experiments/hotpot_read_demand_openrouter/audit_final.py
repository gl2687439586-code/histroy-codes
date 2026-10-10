#!/usr/bin/env python3
"""Independent final integrity audit for the recovered READ-only dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--expected-samples", type=int, required=True)
    p.add_argument("--expected-model", default="openai/gpt-5.6-sol")
    args = p.parse_args()

    source = read_jsonl(args.input)
    errors = []
    warnings = []
    if len(source) != args.expected_samples:
        errors.append(f"input rows={len(source)}, expected={args.expected_samples}")
    bad_keys = [i for i, row in enumerate(source) if set(row) != {"query_id", "query"}]
    if bad_keys:
        errors.append(f"input rows with invalid keys: {bad_keys[:10]}")
    source_ids = [str(row.get("query_id", "")) for row in source]
    if len(source_ids) != len(set(source_ids)):
        errors.append("input contains duplicate query_id values")
    source_map = {str(row["query_id"]): row["query"] for row in source if "query_id" in row}

    result_path = args.run_dir / "results/read_scores_combined.jsonl"
    results = read_jsonl(result_path) if result_path.is_file() else []
    result_ids = [str(row.get("question_id", "")) for row in results]
    if len(results) != args.expected_samples:
        errors.append(f"accepted result rows={len(results)}, expected={args.expected_samples}")
    if len(result_ids) != len(set(result_ids)):
        errors.append("accepted results contain duplicate question_id values")
    if set(result_ids) != set(source_ids):
        errors.append(f"accepted/source ID mismatch: missing={len(set(source_ids)-set(result_ids))}, extra={len(set(result_ids)-set(source_ids))}")
    query_mismatches = [row.get("question_id") for row in results if source_map.get(str(row.get("question_id"))) != row.get("query")]
    if query_mismatches:
        errors.append(f"accepted query text mismatch for {len(query_mismatches)} IDs")
    invalid_scores = [row.get("question_id") for row in results if row.get("read_score") not in (0, 1, 2, 3, 4)]
    if invalid_scores:
        errors.append(f"invalid/null scores for {len(invalid_scores)} IDs")

    config_path = args.run_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    prompt_hash = (config.get("prompt_sha256") or {}).get("read")
    if config.get("model") != args.expected_model:
        errors.append(f"run_config model={config.get('model')!r}")

    state_files = sorted((args.run_dir / "state/read").glob("*.json"))
    state_models = Counter()
    state_prompt_hashes = Counter()
    state_ids = []
    state_mismatches = []
    for path in state_files:
        state = json.loads(path.read_text(encoding="utf-8"))
        state_ids.append(str(state.get("question_id", "")))
        state_models[str(state.get("response_model", ""))] += 1
        state_prompt_hashes[str(state.get("prompt_sha256", ""))] += 1
        qid = str(state.get("question_id", ""))
        if source_map.get(qid) != state.get("query"):
            state_mismatches.append(qid)
    if len(state_files) != args.expected_samples:
        errors.append(f"state files={len(state_files)}, expected={args.expected_samples}")
    if len(state_ids) != len(set(state_ids)):
        errors.append("state files contain duplicate question IDs")
    if state_mismatches:
        errors.append(f"state/source query mismatch for {len(state_mismatches)} IDs")
    if set(state_models) != {args.expected_model}:
        errors.append(f"unexpected response models: {dict(state_models)}")
    if not prompt_hash or set(state_prompt_hashes) != {prompt_hash}:
        errors.append(f"prompt hash mismatch: config={prompt_hash}, states={dict(state_prompt_hashes)}")

    attempt_path = args.run_dir / "api_attempts.jsonl"
    attempts = read_jsonl(attempt_path) if attempt_path.is_file() else []
    successful_attempts = sum(bool(row.get("ok")) for row in attempts)
    failed_attempts = len(attempts) - successful_attempts
    if len(attempts) != args.expected_samples:
        warnings.append(
            f"physical requests={len(attempts)} rather than strict one-shot target {args.expected_samples}; "
            "accepted outputs remain usable if all integrity checks pass"
        )

    report = {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "input": str(args.input),
        "input_sha256": sha256(args.input),
        "expected_samples": args.expected_samples,
        "source_records": len(source),
        "accepted_records": len(results),
        "unique_accepted_ids": len(set(result_ids)),
        "state_records": len(state_files),
        "model_counts": dict(state_models),
        "prompt_hash_counts": dict(state_prompt_hashes),
        "physical_requests": len(attempts),
        "successful_physical_requests": successful_attempts,
        "failed_physical_requests": failed_attempts,
        "strict_one_shot": len(attempts) == args.expected_samples and failed_attempts == 0,
        "score_distribution": {
            str(key): value
            for key, value in sorted(
                Counter(row.get("read_score") for row in results).items(),
                key=lambda item: str(item[0]),
            )
        },
    }
    audit_dir = args.run_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    (audit_dir / "final_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
