#!/usr/bin/env python3
"""Strictly merge sharded JSONL files with a unique-key integrity manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl, sha256_file, utc_now, write_json, write_jsonl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", choices=["estimates", "trajectories"], required=True)
    parser.add_argument("--expected", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=2)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write an explicitly coverage-limited merge instead of inventing missing records.",
    )
    args = parser.parse_args()

    source = Path(args.input_dir).resolve()
    rows = []
    errors = []
    for shard in range(args.num_shards):
        manifest_path = source / f"shard_{shard}.manifest.json"
        records_path = source / f"shard_{shard}.jsonl"
        if not manifest_path.is_file() or not records_path.is_file():
            errors.append(f"Missing shard {shard} manifest or records")
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest.get("complete") and not args.allow_partial:
            errors.append(f"Shard {shard} is incomplete")
        rows.extend(read_jsonl(records_path))
    if args.stage == "estimates":
        key = lambda row: str(row["sample_id"])
        rows.sort(key=lambda row: int(row["sample_index"]))
    else:
        key = lambda row: (str(row["sample_id"]), int(row["rollout_id"]))
        rows.sort(key=lambda row: (int(row["sample_index"]), int(row["rollout_id"])))
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        errors.append("Duplicate unique key across shards")
    if len(rows) != args.expected and not args.allow_partial:
        errors.append(f"Expected {args.expected} rows, found {len(rows)}")
    output = Path(args.output).resolve()
    if not errors:
        write_jsonl(output, rows)
    manifest = {
        "created_at": utc_now(),
        "stage": args.stage,
        "expected_records": args.expected,
        "records": len(rows),
        "coverage": len(rows) / args.expected if args.expected else 0.0,
        "allow_partial": bool(args.allow_partial),
        "missing_records": max(args.expected - len(rows), 0),
        "unique_keys": len(set(keys)),
        "errors": errors,
        "error_count": len(errors),
        "passed": not errors,
        "output": str(output),
        "output_sha256": sha256_file(output) if output.is_file() and not errors else None,
    }
    write_json(str(output) + ".manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
