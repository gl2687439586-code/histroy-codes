#!/usr/bin/env python3
"""Merge and verify four shared-root shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hp_common import canonical_hash, read_json, read_jsonl, utc_stamp, write_json, write_jsonl_gz


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    questions = read_jsonl(args.questions)
    questions.sort(key=lambda row: int(row.get("sample_index", 0)))
    if args.limit:
        questions = questions[:args.limit]
    expected_ids = {str(row["sample_id"]) for row in questions}
    rows = []
    failures = []
    errors = []
    for raw in args.shard_dirs:
        shard = Path(raw)
        if not (shard / "records.jsonl.gz").exists():
            errors.append(f"missing:{shard / 'records.jsonl.gz'}")
            continue
        rows.extend(read_jsonl(shard / "records.jsonl.gz"))
        failures.extend(read_jsonl(shard / "failures.jsonl.gz"))
        status = read_json(shard / "run_status.json")
        if not status.get("complete"):
            errors.append(f"incomplete:{raw}")
    ids = [str(row["sample_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("duplicate_root_ids")
    if set(ids) != expected_ids:
        errors.append(
            f"root_id_set_mismatch:missing={len(expected_ids-set(ids))}:"
            f"extra={len(set(ids)-expected_ids)}"
        )
    if failures:
        errors.append(f"unresolved_failures:{len(failures)}")
    for row in rows:
        supplied = row.get("root_checkpoint_sha256")
        value = dict(row)
        value.pop("root_checkpoint_sha256", None)
        if supplied != canonical_hash(value):
            errors.append(f"root_hash:{row.get('sample_id')}")
        if row.get("gold_labels_loaded") is not False:
            errors.append(f"label_leakage:{row.get('sample_id')}")
        if len(row.get("documents") or []) != 10:
            errors.append(f"top10:{row.get('sample_id')}")
    rows.sort(key=lambda row: int(row.get("sample_index", 0)))
    output = Path(args.output)
    write_jsonl_gz(output, rows)
    report = {
        "created_at": utc_stamp(), "expected_roots": len(expected_ids),
        "roots": len(rows), "unique_ids": len(set(ids)),
        "unresolved_failures": len(failures), "errors": errors,
        "passed": not errors,
        "output": str(output),
    }
    write_json(output.with_suffix(output.suffix + ".manifest.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
