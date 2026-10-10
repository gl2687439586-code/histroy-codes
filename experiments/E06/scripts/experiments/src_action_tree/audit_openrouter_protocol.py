#!/usr/bin/env python3
"""Fail-closed audit for GPT-5.6-Sol Blank, controlled-tree, and neutral runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import read_jsonl, stable_hash, utc_stamp  # noqa: E402
from openrouter_common import EXPECTED_MODEL_ID, PROTOCOL_VERSION  # noqa: E402
from openrouter_prompts import prompt_manifest  # noqa: E402
from run_action_tree import expected_paths  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank", required=True)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--blank_load_report",
        default="",
        help="Blank load_report.json; used to enforce full selected-sample coverage.",
    )
    parser.add_argument(
        "--require_complete",
        action="store_true",
        help="Fail if any Blank/root/tree/Neutral sample or terminal answer is missing.",
    )
    args = parser.parse_args()

    blank = read_jsonl(args.blank)
    roots = read_jsonl(args.roots)
    tree = read_jsonl(args.tree)
    neutral = read_jsonl(args.neutral)
    errors: List[str] = []
    warnings: List[str] = []
    expected_samples = 0
    if args.blank_load_report:
        load_report_path = Path(args.blank_load_report)
        if not load_report_path.exists():
            errors.append(f"Blank load report does not exist: {load_report_path}")
        else:
            load_report = json.loads(load_report_path.read_text(encoding="utf-8"))
            expected_samples = int(load_report.get("loaded_samples", 0) or 0)

    def check_openrouter(rows: List[Dict[str, Any]], label: str) -> None:
        for row in rows:
            sid = str(row.get("sample_id", "?"))
            if str(row.get("model_name", "")).lower() != EXPECTED_MODEL_ID:
                errors.append(f"{label}/{sid}: model_name is not {EXPECTED_MODEL_ID}")
            if row.get("old_checkpoint_used") is not False:
                errors.append(f"{label}/{sid}: old_checkpoint_used must be false")
            if str(row.get("model_path", "")).lower() != EXPECTED_MODEL_ID:
                errors.append(f"{label}/{sid}: wrong OpenRouter model slug")
            if str(row.get("model_url", "")).rstrip("/").lower() != "https://openrouter.ai/api/v1":
                errors.append(f"{label}/{sid}: wrong OpenRouter API base URL")

    check_openrouter(blank, "blank")
    check_openrouter(roots, "roots")
    check_openrouter(tree, "tree")
    check_openrouter(neutral, "neutral")

    observed_runtimes = []
    for label, rows in (("blank", blank), ("roots", roots), ("tree", tree), ("neutral", neutral)):
        for row in rows:
            runtime = row.get("openrouter_runtime")
            if not isinstance(runtime, dict):
                errors.append(f"{label}/{row.get('sample_id', '?')}: missing OpenRouter runtime config")
                continue
            observed_runtimes.append(json.dumps(runtime, sort_keys=True))
            if runtime.get("provider_order") != ["openai"]:
                errors.append(f"{label}/{row.get('sample_id', '?')}: provider is not pinned to openai")
            if runtime.get("allow_fallbacks") is not False:
                errors.append(f"{label}/{row.get('sample_id', '?')}: provider fallbacks must be disabled")
            if runtime.get("response_cache") is not False:
                errors.append(f"{label}/{row.get('sample_id', '?')}: response cache must be disabled")
    if len(set(observed_runtimes)) > 1:
        errors.append("OpenRouter runtime configuration differs across conditions or trajectories")

    def check_api_events(rows: List[Dict[str, Any]], label: str) -> None:
        for row in rows:
            sid = str(row.get("sample_id", "?"))
            events = list(row.get("events") or [])
            if label == "roots":
                events = [dict(row.get("root_generation_event") or {})]
            for index, event in enumerate(events):
                if not event:
                    continue
                api_model = str(event.get("api_model_id", ""))
                if api_model and not api_model.startswith(EXPECTED_MODEL_ID):
                    errors.append(
                        f"{label}/{sid}/event{index}: routed model {api_model!r} is not GPT-5.6 Sol"
                    )
                if event.get("action_code") in {"S", "R", "C", "A"} and not api_model:
                    warnings.append(f"{label}/{sid}/event{index}: missing returned API model id")
                api_provider = str(event.get("api_provider_id", ""))
                if api_provider and not api_provider.lower().startswith("openai"):
                    errors.append(
                        f"{label}/{sid}/event{index}: provider {api_provider!r} is not the pinned OpenAI provider"
                    )
                if event.get("action_code") in {"S", "R", "C", "A"} and not api_provider:
                    warnings.append(f"{label}/{sid}/event{index}: missing returned API provider id")
                router_metadata = event.get("api_router_metadata") or {}
                pipeline = router_metadata.get("pipeline") or [] if isinstance(router_metadata, dict) else []
                if any(
                    isinstance(stage, dict) and stage.get("type") == "context_compression"
                    for stage in pipeline
                ):
                    errors.append(
                        f"{label}/{sid}/event{index}: OpenRouter context compression changed the supplied state"
                    )

    check_api_events(blank, "blank")
    check_api_events(roots, "roots")
    check_api_events(tree, "tree")
    check_api_events(neutral, "neutral")

    blank_by_id = {str(row["sample_id"]): row for row in blank}
    root_by_id = {str(row["sample_id"]): row for row in roots}
    neutral_by_id = {str(row["sample_id"]): row for row in neutral}
    tree_by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in tree:
        tree_by_id[str(row["sample_id"])].append(row)

    if args.require_complete and expected_samples:
        observed_counts = {
            "Blank": len(blank_by_id),
            "Roots": len(root_by_id),
            "Tree": len(tree_by_id),
            "Neutral": len(neutral_by_id),
        }
        for label, observed in observed_counts.items():
            if observed != expected_samples:
                errors.append(
                    f"{label} sample coverage is {observed}/{expected_samples}; formal analysis requires all samples"
                )

    for sid, root in root_by_id.items():
        source = str(root.get("checkpoint_source", ""))
        if not source.startswith("openrouter_"):
            errors.append(f"roots/{sid}: non-OpenRouter checkpoint source {source!r}")
        actual_hash = stable_hash(str(root.get("checkpoint_prefix", "")), 40)
        if actual_hash != root.get("root_checkpoint_sha1"):
            errors.append(f"roots/{sid}: checkpoint hash mismatch")
        blank_row = blank_by_id.get(sid)
        if not blank_row:
            errors.append(f"roots/{sid}: no matching Blank record")
        elif root.get("root_origin") == "openrouter_natural_root":
            if root.get("root_checkpoint_sha1") != blank_row.get("root_checkpoint_sha1"):
                errors.append(f"roots/{sid}: natural root is not the Blank first Search checkpoint")

    for sid, rows in tree_by_id.items():
        root = root_by_id.get(sid)
        if not root:
            errors.append(f"tree/{sid}: no matching root checkpoint")
            continue
        for row in rows:
            if row.get("prompt_protocol") != "openrouter_matched":
                errors.append(f"tree/{sid}/{row.get('path')}: wrong prompt protocol")
            if row.get("root_checkpoint_sha1") != root.get("root_checkpoint_sha1"):
                errors.append(f"tree/{sid}/{row.get('path')}: root hash differs from exported root")
            for event in row.get("events", []):
                if event.get("action_code") == "R":
                    if event.get("evidence_set_unchanged") is not True:
                        errors.append(f"tree/{sid}/{row.get('path')}: Read changed the evidence set")
                    accepted_append = str(event.get("accepted_trace_append", "")).lower()
                    if "<search>" in accepted_append or "<answer>" in accepted_append:
                        errors.append(
                            f"tree/{sid}/{row.get('path')}: executable action tag leaked through Read state"
                        )
            if row.get("terminal_completed") is not True:
                message = f"tree/{sid}/{row.get('path')}: terminal answer is incomplete"
                (errors if args.require_complete else warnings).append(message)
        horizons = {int(row.get("horizon", 3)) for row in rows}
        if len(horizons) != 1:
            errors.append(f"tree/{sid}: multiple horizons {sorted(horizons)}")
        elif {str(row.get("path", "")) for row in rows} != set(expected_paths(next(iter(horizons)))):
            errors.append(f"tree/{sid}: incomplete terminal path set")

    common = set(blank_by_id) & set(neutral_by_id)
    missing_neutral = sorted(set(blank_by_id) - set(neutral_by_id))
    if missing_neutral:
        message = f"Neutral missing {len(missing_neutral)} Blank IDs"
        (errors if args.require_complete else warnings).append(message)
    if not common:
        errors.append("Blank and Neutral have no matched sample IDs")
    complete_tree_ids = set(tree_by_id)
    expected_tree_ids = set(root_by_id)
    missing_tree = sorted(expected_tree_ids - complete_tree_ids)
    extra_tree = sorted(complete_tree_ids - expected_tree_ids)
    if missing_tree:
        message = f"Tree missing {len(missing_tree)} exported root IDs"
        (errors if args.require_complete else warnings).append(message)
    if extra_tree:
        errors.append(f"Tree has {len(extra_tree)} IDs with no exported root")
    unmatched_joint = sorted((set(blank_by_id) | set(neutral_by_id) | expected_tree_ids) - common)
    if unmatched_joint:
        warnings.append(f"Three-condition sample coverage is incomplete for {len(unmatched_joint)} IDs")

    missing_roots = sorted(set(blank_by_id) - set(root_by_id))
    if missing_roots:
        message = f"Roots missing {len(missing_roots)} Blank IDs"
        (errors if args.require_complete else warnings).append(message)

    natural_ids = {sid for sid, row in root_by_id.items() if row.get("root_origin") == "openrouter_natural_root"}
    forced_ids = {sid for sid, row in root_by_id.items() if row.get("root_origin") == "openrouter_forced_root"}
    if not natural_ids:
        warnings.append(
            "No natural GPT-5.6 Search roots were observed; the primary natural-root "
            "estimand is unavailable. Forced-root action-tree results remain sensitivity analysis."
        )
    report = {
        "schema_version": PROTOCOL_VERSION,
        "created_at": utc_stamp(),
        "passed": not errors,
        "require_complete": args.require_complete,
        "expected_samples_from_blank_load_report": expected_samples,
        "errors": errors,
        "warnings": warnings,
        "counts": {
            "blank": len(blank),
            "roots": len(roots),
            "tree_terminal_records": len(tree),
            "tree_samples": len(tree_by_id),
            "neutral": len(neutral),
            "matched_blank_neutral": len(common),
            "natural_roots": len(natural_ids),
            "forced_roots": len(forced_ids),
            "missing_tree_samples": len(missing_tree),
            "joint_complete_case_samples": len(common & complete_tree_ids),
        },
        "root_origin_distribution": dict(Counter(row.get("root_origin", "unknown") for row in roots)),
        "primary_natural_root_estimand_available": bool(natural_ids),
        "forced_root_action_tree_sensitivity_available": bool(forced_ids),
        "primary_tree_subset": "openrouter_natural_root",
        "sensitivity_tree_subset": "all roots including openrouter_forced_root",
        "prompt_manifest": prompt_manifest(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
