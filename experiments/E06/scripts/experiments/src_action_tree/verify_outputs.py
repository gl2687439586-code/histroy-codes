#!/usr/bin/env python3
"""Structural and causal-comparability checks for generated action trees."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import normalized_query, read_jsonl, utc_stamp  # noqa: E402
from run_action_tree import expected_paths  # noqa: E402


def comparable_event(event: Dict[str, Any]) -> str:
    """Stable serialization for events expected to be identical across siblings."""
    return json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    raw_rows = read_jsonl(args.input)
    rows: List[Dict[str, Any]] = []
    latest_by_path: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    duplicate_counts: Dict[Tuple[str, int, str], int] = defaultdict(int)
    for row in raw_rows:
        path_key = (
            str(row.get("sample_id")),
            int(row.get("rollout_id", 0)),
            str(row.get("path")),
        )
        duplicate_counts[path_key] += 1
        latest_by_path[path_key] = row
    rows = list(latest_by_path.values())
    trees: Dict[Tuple[str, int], Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("sample_id")), int(row.get("rollout_id", 0)))
        trees[key][str(row.get("path"))] = row

    errors: List[str] = []
    warnings: List[str] = []
    for (sample_id, rollout_id, path), count in sorted(duplicate_counts.items()):
        if count > 1:
            warnings.append(
                f"{sample_id}/r{rollout_id}/{path}: {count} records found; "
                "latest retained (likely interrupted resume)"
            )
    checked = 0
    horizon_counts: Dict[int, int] = defaultdict(int)
    for (sample_id, rollout_id), paths in sorted(trees.items()):
        horizons = {int(row.get("horizon", 3)) for row in paths.values()}
        if len(horizons) != 1:
            errors.append(f"{sample_id}/r{rollout_id}: inconsistent horizons={sorted(horizons)}")
            continue
        horizon = next(iter(horizons))
        wanted = set(expected_paths(horizon))
        missing = wanted - set(paths)
        extra = set(paths) - wanted
        if missing or extra:
            errors.append(
                f"{sample_id}/r{rollout_id}/H{horizon}: missing={sorted(missing)}, extra={sorted(extra)}"
            )
            continue
        checked += 1
        horizon_counts[horizon] += 1

        checkpoints = {row.get("checkpoint_prefix", "") for row in paths.values()}
        if len(checkpoints) != 1 or not next(iter(checkpoints), ""):
            errors.append(f"{sample_id}/r{rollout_id}: branches do not share one non-empty checkpoint prefix")

        initial_events = {comparable_event(row["events"][0]) for row in paths.values() if row.get("events")}
        if len(initial_events) != 1:
            errors.append(f"{sample_id}/r{rollout_id}: initial Search event differs across branches")

        # Audit every reachable S/R prefix. All children must see one identical
        # parent hash; every shared S/R edge must contain one identical event.
        prefixes = {
            path[:depth]
            for path in wanted
            for depth in range(1, min(len(path), horizon))
            if "C" not in path[:depth]
        }
        for prefix in sorted(prefixes, key=lambda item: (len(item), item)):
            depth = len(prefix)
            descendants = [row for path, row in paths.items() if path.startswith(prefix) and len(path) > depth]
            parent_hashes = {
                row["events"][depth].get("parent_state_hash")
                for row in descendants
                if len(row.get("events", [])) > depth
            }
            if len(parent_hashes) != 1 or not next(iter(parent_hashes), ""):
                errors.append(
                    f"{sample_id}/r{rollout_id}/{prefix}: children do not share a byte-identical parent"
                )
            for action in ("S", "R"):
                child_rows = [
                    row
                    for path, row in paths.items()
                    if path.startswith(prefix + action) and len(row.get("events", [])) > depth
                ]
                child_events = {comparable_event(row["events"][depth]) for row in child_rows}
                if len(child_events) > 1:
                    errors.append(
                        f"{sample_id}/r{rollout_id}/{prefix + action}: shared edge event differs across descendants"
                    )

        for path, row in paths.items():
            expected_search, expected_read = path.count("S"), path.count("R")
            if int(row.get("search_calls", -1)) != expected_search:
                errors.append(f"{sample_id}/r{rollout_id}/{path}: search_calls={row.get('search_calls')} expected={expected_search}")
            if int(row.get("read_calls", -1)) != expected_read:
                errors.append(f"{sample_id}/r{rollout_id}/{path}: read_calls={row.get('read_calls')} expected={expected_read}")
            if not row.get("final_answer"):
                warnings.append(f"{sample_id}/r{rollout_id}/{path}: extracted final answer is empty")
            if not row.get("terminal_completed", True):
                warnings.append(f"{sample_id}/r{rollout_id}/{path}: terminal generation was incomplete and scored empty")
            queries = [normalized_query(query) for query in row.get("queries", [])]
            if len(queries) != len(set(queries)):
                message = f"{sample_id}/r{rollout_id}/{path}: repeated normalized Search query"
                if row.get("action_validation") == "strict":
                    errors.append(message)
                else:
                    warnings.append(message + " (allowed in flexible mode)")
            for event in row.get("events", []):
                if event.get("action_code") == "R":
                    if not event.get("evidence_set_unchanged"):
                        errors.append(f"{sample_id}/r{rollout_id}/{path}: Read changed the evidence set")
                    if event.get("retrieved_docs", 0) or event.get("retrieved_tokens_total", 0):
                        errors.append(f"{sample_id}/r{rollout_id}/{path}: Read acquired a new Observation")
                if not event.get("raw_prompt"):
                    errors.append(f"{sample_id}/r{rollout_id}/{path}: event missing raw_prompt")

    report = {
        "schema_version": "src-action-tree-verification-v4-variable-horizon",
        "created_at": utc_stamp(),
        "input": args.input,
        "raw_records": len(raw_rows),
        "records_after_deduplication": len(rows),
        "duplicate_records_ignored": len(raw_rows) - len(rows),
        "trees_seen": len(trees),
        "complete_trees_checked": checked,
        "horizon_tree_counts": dict(sorted(horizon_counts.items())),
        "errors": errors,
        "warnings": warnings,
        "passed": not errors,
    }
    report_path = Path(args.report) if args.report else Path(args.input).with_name("verification_report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
