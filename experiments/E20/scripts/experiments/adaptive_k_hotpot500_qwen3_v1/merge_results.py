#!/usr/bin/env python3
"""Merge two generation shards and enforce exact 500 x 2 completeness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import canonical_hash, file_sha256, read_jsonl, utc_stamp, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--shards_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rollouts", type=int, default=2)
    parser.add_argument("--num_shards", type=int, default=2)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    roots = read_jsonl(args.roots)
    root_by_id = {str(row["sample_id"]): row for row in roots}
    errors = []
    if len(root_by_id) != len(questions):
        errors.append(f"roots={len(root_by_id)} questions={len(questions)}")
    records = []
    shard_manifests = []
    for shard_id in range(args.num_shards):
        record_path = Path(args.shards_dir) / f"shard_{shard_id}.jsonl"
        manifest_path = Path(args.shards_dir) / f"shard_{shard_id}.manifest.json"
        if not record_path.is_file() or not manifest_path.is_file():
            errors.append(f"Missing shard {shard_id} outputs")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        shard_manifests.append(manifest)
        if not manifest.get("complete") or manifest.get("unresolved_failures"):
            errors.append(f"Shard {shard_id} is incomplete")
        records.extend(read_jsonl(record_path))

    by_key = {}
    for row in records:
        key = (str(row.get("sample_id")), int(row.get("rollout_id", -1)))
        if key in by_key:
            errors.append(f"Duplicate generation key {key}")
        by_key[key] = row
        root = root_by_id.get(key[0])
        if root is None:
            errors.append(f"Unknown root for {key}")
        elif row.get("root_hash") != root.get("root_hash"):
            errors.append(f"Root hash mismatch for {key}")

    expected_keys = [
        (str(question["sample_id"]), rollout_id)
        for question in questions for rollout_id in range(args.rollouts)
    ]
    missing = [key for key in expected_keys if key not in by_key]
    if missing:
        errors.append(f"Missing {len(missing)} generation records: {missing[:10]}")
    ordered = [by_key[key] for key in expected_keys if key in by_key]
    output = Path(args.output)
    if not errors:
        write_jsonl(output, ordered)
    manifest = {
        "created_at": utc_stamp(),
        "questions": len(questions),
        "roots": len(roots),
        "rollouts": args.rollouts,
        "expected_records": len(expected_keys),
        "records": len(ordered),
        "unique_keys": len(by_key),
        "errors": errors,
        "error_count": len(errors),
        "passed": not errors,
        "questions_sha256": file_sha256(args.questions),
        "roots_sha256": file_sha256(args.roots),
        "shard_signature_hashes": [item.get("signature_hash") for item in shard_manifests],
        "content_hash": canonical_hash(ordered),
        "output": str(output.resolve()),
    }
    write_json(str(output) + ".manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
