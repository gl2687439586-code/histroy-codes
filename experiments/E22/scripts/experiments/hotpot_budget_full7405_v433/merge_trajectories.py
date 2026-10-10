#!/usr/bin/env python3
"""Stream-merge and verify four compressed trajectory shards."""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path

from hp_common import (
    MAX_BOUNDARY_BACKOFF_TOKENS, TOP_K, canonical_hash, iter_jsonl,
    read_json, read_jsonl, utc_stamp, write_json,
)
from run_trajectories import conditions_for_root


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--shard_dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exposures", default="")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    roots = read_jsonl(args.roots)
    roots.sort(key=lambda row: int(row.get("sample_index", 0)))
    if args.limit:
        roots = roots[:args.limit]
    expected = {
        f"{root['sample_id']}|{condition['condition_id']}"
        for root in roots for condition in conditions_for_root(root, args.exposures)
    }
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
                key = f"{row.get('sample_id')}|{row.get('condition_id')}"
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
                if int(row.get("top_k_per_search", -1)) != TOP_K:
                    errors.append(f"top_k:{key}")
                if row.get("doc_slot_tokens") is not None:
                    errors.append(f"legacy_doc_slot:{key}")
                for search_round in row.get("search_rounds") or []:
                    evidence = search_round.get("evidence") or {}
                    actual = int(evidence.get("actual_exposed_token_count", -1))
                    total = int(evidence.get("full_context_token_count", -1))
                    if not 0 < actual <= total:
                        errors.append(f"exposure_range:{key}:round{search_round.get('search_round')}")
                    if int(evidence.get("visible_evidence_tokens", -1)) != actual:
                        errors.append(f"exposure_accounting:{key}:round{search_round.get('search_round')}")
                    if evidence.get("exact_token_roundtrip") is not True:
                        errors.append(f"token_roundtrip:{key}:round{search_round.get('search_round')}")
                    target = int(evidence.get(
                        "target_token_cap_before_roundtrip", actual
                    ))
                    backoff = int(evidence.get(
                        "roundtrip_boundary_backoff_tokens", target - actual
                    ))
                    if not 0 <= backoff == target - actual:
                        errors.append(
                            f"roundtrip_backoff:{key}:round{search_round.get('search_round')}"
                        )
                    if backoff > MAX_BOUNDARY_BACKOFF_TOKENS:
                        errors.append(
                            f"excessive_roundtrip_backoff:{key}:"
                            f"round{search_round.get('search_round')}:"
                            f"backoff={backoff}"
                        )
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
