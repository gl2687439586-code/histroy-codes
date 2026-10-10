#!/usr/bin/env python3
"""Stream-merge and verify four compressed trajectory shards."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

from hp_common import BUDGETS, canonical_hash, iter_jsonl, read_json, read_jsonl, utc_stamp, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--budgets", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    roots = read_jsonl(args.roots)
    roots.sort(key=lambda row: int(row.get("sample_index", 0)))
    if args.limit:
        roots = roots[:args.limit]
    budgets = sorted({int(item) for item in args.budgets.split(",") if item.strip()}) \
        if args.budgets else BUDGETS
    expected = {f"{root['sample_id']}|b{budget:04d}" for root in roots for budget in budgets}
    seen = set()
    errors = []
    failures = 0
    records = 0
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as sink:
        for raw in args.shard_dirs:
            shard = Path(raw)
            required = [shard / "records.jsonl.gz", shard / "failures.jsonl.gz",
                        shard / "run_status.json"]
            missing = [str(path) for path in required if not path.exists()]
            if missing:
                errors.extend(f"missing:{path}" for path in missing)
                continue
            status = read_json(shard / "run_status.json")
            if not status.get("complete"):
                errors.append(f"incomplete:{raw}")
            failures += sum(1 for _ in iter_jsonl(shard / "failures.jsonl.gz"))
            for row in iter_jsonl(shard / "records.jsonl.gz"):
                key = f"{row.get('sample_id')}|b{int(row.get('evidence_budget_per_search',-1)):04d}"
                if key in seen:
                    errors.append(f"duplicate:{key}")
                    continue
                seen.add(key)
                supplied = row.get("record_sha256")
                value = dict(row)
                value.pop("record_sha256", None)
                if supplied != canonical_hash(value):
                    errors.append(f"record_hash:{key}")
                if row.get("gold_labels_loaded") is not False:
                    errors.append(f"label_leakage:{key}")
                if int(row.get("top_k_per_search", -1)) != 10:
                    errors.append(f"top_k:{key}")
                if not 1 <= int(row.get("search_calls", 0)) <= int(row.get("max_searches", 0)):
                    errors.append(f"search_count:{key}")
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                records += 1
    os.replace(temporary, output)
    if failures:
        errors.append(f"unresolved_failures:{failures}")
    if seen != expected:
        errors.append(
            f"key_set:missing={len(expected-seen)}:extra={len(seen-expected)}"
        )
    report = {
        "created_at": utc_stamp(), "expected_records": len(expected),
        "records": records, "unique_keys": len(seen),
        "unresolved_failures": failures,
        "errors": errors[:100], "error_count": len(errors),
        "passed": not errors, "output": str(output),
    }
    write_json(output.with_suffix(output.suffix + ".manifest.json"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
