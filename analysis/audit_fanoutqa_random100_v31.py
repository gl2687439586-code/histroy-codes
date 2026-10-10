import gzip
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parent
EXTRACTED = WORKSPACE / "audit_fanoutqa_random100_v31_20260910"
RUN = (
    EXTRACTED
    / "outputs"
    / "fanoutqa_concat100_full310_v2_wiki2023"
    / "random100_v31_upto5_fallback_marked_seed2026"
)
SAMPLE_DIR = (
    EXTRACTED / "data" / "fanoutqa" / "prepared" / "random100_seed2026"
)


def read_json(path):
    path = Path(path)
    if path.is_absolute() and not str(path).startswith("\\\\?\\"):
        path = Path("\\\\?\\" + str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def read_gzip_jsonl(path):
    path = Path(path)
    if path.is_absolute() and not str(path).startswith("\\\\?\\"):
        path = Path("\\\\?\\" + str(path))
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical_hash(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mean(values):
    return sum(values) / len(values) if values else None


def bootstrap_mean_ci(values, repetitions=20000, seed=2026):
    if not values:
        return [None, None]
    generator = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(repetitions):
        estimates.append(sum(values[generator.randrange(n)] for _ in range(n)) / n)
    estimates.sort()
    return [
        estimates[int(0.025 * repetitions)],
        estimates[min(repetitions - 1, int(0.975 * repetitions))],
    ]


def exact_mcnemar_p(left_wins, right_wins):
    discordant = left_wins + right_wins
    if discordant == 0:
        return 1.0
    smaller = min(left_wins, right_wins)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1)) / (2**discordant)
    return min(1.0, 2.0 * tail)


errors = []
warnings = []


def require(condition, message):
    if not condition:
        errors.append(message)


root_path = RUN / "roots" / "roots.merged.jsonl.gz"
budget_path = RUN / "budget" / "trajectories.merged.jsonl.gz"
blank_path = RUN / "blank" / "trajectories.merged.jsonl.gz"
roots = read_gzip_jsonl(root_path)
budget = read_gzip_jsonl(budget_path)
blank = read_gzip_jsonl(blank_path)
budget_scored = read_gzip_jsonl(RUN / "analysis" / "budget" / "scored_records.jsonl.gz")
blank_scored = read_gzip_jsonl(RUN / "analysis" / "blank" / "scored_records.jsonl.gz")

root_manifest = read_json(Path(str(root_path) + ".manifest.json"))
budget_manifest = read_json(Path(str(budget_path) + ".manifest.json"))
blank_manifest = read_json(Path(str(blank_path) + ".manifest.json"))
sample_manifest = read_json(SAMPLE_DIR / "sample_manifest.json")
budget_summary = read_json(RUN / "analysis" / "budget" / "summary.json")
blank_summary = read_json(RUN / "analysis" / "blank" / "summary.json")
comparison = read_json(RUN / "analysis" / "comparison" / "summary.json")
flip_summary = read_json(RUN / "analysis" / "budget" / "answer_flip" / "summary.json")

require(len(roots) == 100, f"root count is {len(roots)}")
require(len(budget) == 1126, f"budget count is {len(budget)}")
require(len(blank) == 100, f"blank count is {len(blank)}")
require(len(budget_scored) == len(budget), "budget scored-record mismatch")
require(len(blank_scored) == len(blank), "blank scored-record mismatch")
for name, manifest in (
    ("roots", root_manifest),
    ("budget", budget_manifest),
    ("blank", blank_manifest),
):
    require(manifest.get("passed") is True, f"{name} merged manifest not passed")
    require(not manifest.get("errors"), f"{name} merged manifest has errors")
    require(manifest.get("unresolved_failures") == 0, f"{name} unresolved failures")

root_ids = [str(row["sample_id"]) for row in roots]
budget_keys = [(str(row["sample_id"]), str(row["condition_id"])) for row in budget]
blank_ids = [str(row["sample_id"]) for row in blank]
selected_ids = [str(item) for item in sample_manifest["selected_sample_ids"]]
require(len(set(root_ids)) == 100, "duplicate root IDs")
require(len(set(budget_keys)) == 1126, "duplicate budget keys")
require(len(set(blank_ids)) == 100, "duplicate blank IDs")
require(set(root_ids) == set(blank_ids) == set(selected_ids), "sample sets differ")
require(sample_manifest["seed"] == 2026, "sample seed differs")
require(sample_manifest["sample_size"] == 100, "sample size differs")
require(sample_manifest["source_question_count"] == 310, "source count differs")

root_by_id = {str(row["sample_id"]): row for row in roots}

# Generation must never consume the sealed labels.
require(all(row.get("gold_labels_loaded") is False for row in roots + budget + blank), "label leakage flag")
require(all(row.get("dataset_name") == "fanoutqa" for row in roots + budget + blank), "dataset provenance mismatch")
require(all(row.get("wikipedia_snapshot") == "2023-09" for row in roots + budget + blank), "snapshot mismatch")
require(
    all(
        row.get("retriever_backend")
        == "fanoutqa_wikipedia_en_all_nopic_2023-09_kiwix_passage100_upto5"
        for row in roots + budget + blank
    ),
    "retriever backend mismatch",
)

# Shard statuses and signatures.
shard_statuses = []
signatures = []
for stage in ("roots", "budget", "blank"):
    for shard_id in range(4):
        status_path = RUN / stage / "shards" / f"shard_{shard_id}" / "run_status.json"
        status = read_json(status_path)
        shard_statuses.append({"group": stage, "shard": status_path.parent.name, **status})
        require(status.get("complete") is True, f"incomplete {stage}/{status_path.parent.name}")
        require(status.get("unresolved_failures") == 0, f"unresolved {stage}/{status_path.parent.name}")
    for shard_id in range(4):
        manifest_path = RUN / stage / "shards" / f"shard_{shard_id}" / "manifest.json"
        signature = read_json(manifest_path).get("signature", {})
        signatures.append(signature)

for signature in signatures:
    if signature.get("stage") == "trajectories":
        require(signature.get("model_path", "").endswith("/models/Qwen3-8B"), "model mismatch")
        require(signature.get("context_length") == 40960, "context length mismatch")
        require(signature.get("server_max_total_tokens") == 30384, "token pool mismatch")
        require(signature.get("answer_max_new_tokens") == 8192, "answer cap mismatch")
        require(signature.get("temperature") == 0.0, "temperature mismatch")
        require(signature.get("requested_max_k") == 5, "requested K mismatch")
        require(signature.get("gold_labels_loaded") is False, "manifest label leakage")

# Record integrity, shared root, token exposure, and retrieval invariants.
record_hash_errors = []
root_identity_errors = []
grid_errors = []
query_mutations = []
query_fallbacks = []
roundtrip_errors = []
cap_violations = []
boundary_backoffs = []
retrieval_k = Counter()
retrieval_shortfalls = []
zero_rounds = []
status_counts = {"budget": Counter(), "blank": Counter()}
finish_types = Counter()
parse_modes = Counter()
fallback_details = []
truncation_details = []
physical_continuation_calls = 0
physical_additional_retriever_calls = 0

for stage, rows in (("budget", budget), ("blank", blank)):
    for row in rows:
        sample_id = str(row["sample_id"])
        status_counts[stage][str(row.get("answer_status"))] += 1
        require(row.get("context_compactions") == 0, f"context compaction: {stage}/{sample_id}")
        require(row.get("search_calls") == len(row.get("search_rounds") or []), f"search count mismatch: {stage}/{sample_id}")
        require(row.get("retrieved_docs_total") == sum(row.get("retrieved_docs_by_round") or []), f"doc total mismatch: {stage}/{sample_id}")

        supplied = row.get("record_sha256")
        value = dict(row)
        value.pop("record_sha256", None)
        if supplied != canonical_hash(value):
            record_hash_errors.append([stage, sample_id, row.get("condition_id")])

        root = root_by_id[sample_id]
        rounds = row["search_rounds"]
        first = rounds[0]
        first_evidence = first["evidence"]
        root_ok = (
            row["root_checkpoint_sha256"] == root["root_checkpoint_sha256"]
            and row["root_query"] == root["query"] == first["query"]
            and row["root_retrieval_hash"] == root["retrieval_hash"] == first_evidence["retrieval_hash"]
            and [doc["contents_sha256"] for doc in root["documents"]]
            == [doc["contents_sha256"] for doc in first_evidence["retrieved_docs"]]
        )
        if not root_ok:
            root_identity_errors.append([stage, sample_id, row["condition_id"]])

        physical_continuation_calls += len(row.get("decision_events") or [])
        physical_additional_retriever_calls += int(row.get("additional_search_calls_after_root", 0))
        decision_events = row.get("decision_events") or []
        for event_index, event in enumerate(decision_events):
            parse_modes[str(event.get("parse_mode"))] += 1
            for attempt in event.get("attempts") or []:
                finish_types[str(attempt.get("finish_type"))] += 1
                if attempt.get("finish_type") == "length":
                    truncation_details.append({
                        "stage": stage,
                        "sample_id": sample_id,
                        "condition": str(row.get("condition_label")),
                        "answer_status": row.get("answer_status"),
                        "event_action": event.get("action"),
                        "event_parse_mode": event.get("parse_mode"),
                        "is_final_event": event_index == len(decision_events) - 1,
                        "output_tokens": attempt.get("output_tokens"),
                        "max_new_tokens": attempt.get("max_new_tokens"),
                        "final_answer_characters": len(row.get("final_answer") or ""),
                    })

        if row.get("answer_status") != "valid":
            last = (row.get("decision_events") or [{}])[-1]
            attempts = last.get("attempts") or []
            fallback_details.append({
                "stage": stage,
                "sample_id": sample_id,
                "condition": str(row.get("condition_label")),
                "answer_status": row.get("answer_status"),
                "answer_parse_mode": row.get("answer_parse_mode"),
                "last_event_action": last.get("action"),
                "last_event_parse_mode": last.get("parse_mode"),
                "finish_types": [item.get("finish_type") for item in attempts],
                "output_tokens": [item.get("output_tokens") for item in attempts],
                "final_answer": row.get("final_answer"),
            })

        for search_round in rounds:
            evidence = search_round["evidence"]
            documents = evidence["retrieved_docs"]
            actual_k = int(evidence["actual_retrieved_k"])
            retrieval_k[actual_k] += 1
            require(actual_k == len(documents), f"actual K mismatch: {stage}/{sample_id}")
            require(0 <= actual_k <= 5, f"actual K outside 0..5: {stage}/{sample_id}")
            require(evidence["requested_max_k"] == 5, f"requested K != 5: {stage}/{sample_id}")
            require(evidence["retrieval_shortfall"] == (actual_k < 5), f"shortfall flag mismatch: {stage}/{sample_id}")
            require(evidence["zero_result"] == (actual_k == 0), f"zero flag mismatch: {stage}/{sample_id}")
            require(evidence["underfilled_tokens"] == evidence["requested_token_cap"] - evidence["actual_exposed_token_count"], f"underfill mismatch: {stage}/{sample_id}")
            if actual_k < 5:
                item = {
                    "stage": stage,
                    "sample_id": sample_id,
                    "condition": str(row["condition_label"]),
                    "round": search_round["search_round"],
                    "query": search_round["query"],
                    "actual_k": actual_k,
                }
                retrieval_shortfalls.append(item)
                if actual_k == 0:
                    zero_rounds.append(item)
            if evidence.get("exact_token_roundtrip") is not True:
                roundtrip_errors.append([stage, sample_id, row["condition_id"], search_round["search_round"]])
            if evidence["actual_exposed_token_count"] > evidence["target_token_cap_before_roundtrip"]:
                cap_violations.append([stage, sample_id, row["condition_id"], search_round["search_round"]])
            if evidence["roundtrip_boundary_backoff_tokens"]:
                boundary_backoffs.append({
                    "stage": stage,
                    "sample_id": sample_id,
                    "condition": row["condition_label"],
                    "round": search_round["search_round"],
                    "target": evidence["target_token_cap_before_roundtrip"],
                    "actual": evidence["actual_exposed_token_count"],
                    "backoff": evidence["roundtrip_boundary_backoff_tokens"],
                })
            if row["condition_label"] == "FULL" or stage == "blank":
                require(evidence["actual_exposed_token_count"] == evidence["full_context_token_count"], f"full context truncated: {stage}/{sample_id}")
            else:
                require(evidence["requested_token_cap"] == row["token_cap_per_search"], f"cap differs across rounds: {stage}/{sample_id}")
            for document in documents:
                if document.get("requested_query") != search_round["query"] or document.get("retrieval_query") != search_round["query"]:
                    query_mutations.append([stage, sample_id, search_round["query"], document.get("requested_query"), document.get("retrieval_query")])
                if document.get("retrieval_query_is_fallback"):
                    query_fallbacks.append([stage, sample_id, search_round["query"]])

for sample_id, root in root_by_id.items():
    expected = [str(value) for value in range(100, root["root_full_context_tokens"] + 1, 100)] + ["FULL"]
    actual = [row["condition_label"] for row in budget if str(row["sample_id"]) == sample_id]
    if actual != expected:
        grid_errors.append([sample_id, root["root_full_context_tokens"], expected, actual])

require(not record_hash_errors, "record hash mismatches")
require(not root_identity_errors, "shared-root mismatches")
require(not grid_errors, "root-specific budget-grid mismatches")
require(not query_mutations, "query mutations detected")
require(not query_fallbacks, "fallback query detected")
require(not roundtrip_errors, "token round-trip errors")
require(not cap_violations, "token cap violations")
require(not boundary_backoffs, "unexpected boundary backoff")

# Scoring policy and fallback behavior.
policy = budget_summary["metric_policy"]["invalid_outputs"]
require("score the stored recovered final_answer normally" in policy, "old fallback-zero scoring policy")
budget_score_by_key = {(str(row["sample_id"]), str(row["condition_id"])): row for row in budget_scored}
blank_score_by_id = {str(row["sample_id"]): row for row in blank_scored}

def subset_metrics(rows):
    fields = (
        "official_loose_accuracy",
        "official_strict_accuracy",
        "rougeL_f1",
        "compat_f1",
    )
    return {field: mean([float(row[field]) for row in rows]) for field in fields} | {"records": len(rows)}

budget_valid = [row for row in budget_scored if row["answer_valid"] == 1.0]
budget_fallback = [row for row in budget_scored if row["answer_valid"] == 0.0]
blank_valid = [row for row in blank_scored if row["answer_valid"] == 1.0]
blank_fallback = [row for row in blank_scored if row["answer_valid"] == 0.0]
fallback_condition_counts = Counter(row["condition_label"] for row in budget_fallback)
budget_fallback_sample_ids = {str(row["sample_id"]) for row in budget_fallback}
blank_fallback_sample_ids = {str(row["sample_id"]) for row in blank_fallback}

fallback_finish_classes = Counter()
for item in fallback_details:
    finishes = item["finish_types"]
    if "length" in finishes:
        fallback_finish_classes["length"] += 1
    elif "stop" in finishes:
        fallback_finish_classes["stopped_but_unparseable"] += 1
    else:
        fallback_finish_classes["other"] += 1

truncated_but_parsed = [
    item for item in truncation_details
    if item["answer_status"] == "valid"
]
if truncated_but_parsed:
    warnings.append(
        f"{len(truncated_but_parsed)} length-capped generations were parsed as valid; "
        "retain their scores but label them truncated_but_parsed in reliability reporting"
    )

# Paired FULL-vs-Blank estimates with question-level bootstrap intervals.
full_by_id = {
    str(row["sample_id"]): row
    for row in budget_scored
    if row["condition_label"] == "FULL"
}
paired = {}
for field in (
    "official_loose_accuracy",
    "official_strict_accuracy",
    "rouge1_f1",
    "rouge2_f1",
    "rougeL_f1",
):
    differences = [
        float(full_by_id[sample_id][field]) - float(blank_score_by_id[sample_id][field])
        for sample_id in sorted(root_by_id)
    ]
    left_wins = sum(value > 0 for value in differences)
    right_wins = sum(value < 0 for value in differences)
    paired[field] = {
        "full_mean": mean([float(full_by_id[sample_id][field]) for sample_id in root_by_id]),
        "blank_mean": mean([float(blank_score_by_id[sample_id][field]) for sample_id in root_by_id]),
        "mean_difference": mean(differences),
        "bootstrap_95_ci": bootstrap_mean_ci(differences),
        "full_wins": left_wins,
        "ties": sum(value == 0 for value in differences),
        "full_losses": right_wins,
    }
    if field == "official_strict_accuracy":
        paired[field]["exact_mcnemar_p"] = exact_mcnemar_p(left_wins, right_wins)

# Common 100..700 grid and paired adjacent changes.
common_budgets = list(range(100, 701, 100))
budget_by_sample_and_label = defaultdict(dict)
for row in budget_scored:
    budget_by_sample_and_label[str(row["sample_id"])][str(row["condition_label"])] = row
common_curve = []
for value in common_budgets:
    label = str(value)
    rows = [budget_by_sample_and_label[sample_id][label] for sample_id in sorted(root_by_id)]
    common_curve.append({
        "budget": value,
        **subset_metrics(rows),
        "answer_valid": mean([float(row["answer_valid"]) for row in rows]),
        "visible_tokens": mean([float(row["visible_evidence_tokens_total"]) for row in rows]),
        "evidence_visibility_recall": mean([float(row["necessary_title_visibility_recall"]) for row in rows]),
    })

adjacent_loose = []
for left, right in zip(common_budgets, common_budgets[1:]):
    differences = [
        float(budget_by_sample_and_label[sample_id][str(right)]["official_loose_accuracy"])
        - float(budget_by_sample_and_label[sample_id][str(left)]["official_loose_accuracy"])
        for sample_id in sorted(root_by_id)
    ]
    adjacent_loose.append({
        "transition": f"{left}->{right}",
        "mean_change": mean(differences),
        "bootstrap_95_ci": bootstrap_mean_ci(differences),
        "gains": sum(value > 0 for value in differences),
        "ties": sum(value == 0 for value in differences),
        "harms": sum(value < 0 for value in differences),
    })

# Retrieval visibility diagnostic on the independent per-question Blank group.
blank_visible = [row for row in blank_scored if row["necessary_title_any_visible"] == 1.0]
blank_not_visible = [row for row in blank_scored if row["necessary_title_any_visible"] == 0.0]
blank_visibility_diagnostic = {
    "any_visible": subset_metrics(blank_visible),
    "none_visible": subset_metrics(blank_not_visible),
}

root_full_tokens = [int(row["root_full_context_tokens"]) for row in roots]
grid_count_distribution = Counter(len(row["exposure_conditions"]) for row in roots)

# Actual physical calls: shared roots once, plus each counterfactual continuation.
physical_model_calls = len(roots) + physical_continuation_calls
physical_retriever_calls = len(roots) + physical_additional_retriever_calls
budget_elapsed = max(
    status["elapsed_seconds_this_attempt"]
    for status in shard_statuses
    if status["group"] == "budget"
)
blank_elapsed = max(
    status["elapsed_seconds_this_attempt"]
    for status in shard_statuses
    if status["group"] == "blank"
)
root_elapsed = max(
    status["elapsed_seconds_this_attempt"]
    for status in shard_statuses
    if status["group"] == "roots"
)

report = {
    "passed": not errors,
    "errors": errors,
    "warnings": warnings,
    "archive_sha256": "76e0f0a9e56a1684340a3b6dc25605744402bfd82be88fae243a1b9b867d5987f",
    "integrity": {
        "roots": len(roots),
        "budget_records": len(budget),
        "blank_records": len(blank),
        "unique_budget_keys": len(set(budget_keys)),
        "sample_seed": sample_manifest["seed"],
        "selected_ids_sha256": sample_manifest["selected_ids_sha256"],
        "record_hash_errors": len(record_hash_errors),
        "root_identity_errors": len(root_identity_errors),
        "grid_errors": len(grid_errors),
    },
    "configuration": {
        "model": "/home/researcher/smartsearch_rebuild/models/Qwen3-8B",
        "context_length": 40960,
        "server_max_total_tokens": 30384,
        "answer_max_new_tokens": 8192,
        "temperature": 0.0,
        "max_searches": 4,
        "requested_max_k": 5,
        "snapshot": "2023-09",
        "retriever": "fanoutqa_wikipedia_en_all_nopic_2023-09_kiwix_passage100_upto5",
        "fallback_scoring_policy": policy,
    },
    "budget_grid": {
        "records_per_sample_mean": len(budget) / len(roots),
        "conditions_per_sample_distribution": dict(sorted(grid_count_distribution.items())),
        "root_full_tokens_min": min(root_full_tokens),
        "root_full_tokens_median": statistics.median(root_full_tokens),
        "root_full_tokens_max": max(root_full_tokens),
        "common_grid": common_budgets,
    },
    "protocol": {
        "retrieval_rounds": sum(retrieval_k.values()),
        "retrieved_k_distribution": dict(sorted(retrieval_k.items())),
        "shortfall_rounds": len(retrieval_shortfalls),
        "zero_result_rounds": len(zero_rounds),
        "shortfall_examples": retrieval_shortfalls,
        "query_mutations": len(query_mutations),
        "query_fallbacks": len(query_fallbacks),
        "roundtrip_errors": len(roundtrip_errors),
        "cap_violations": len(cap_violations),
        "boundary_backoffs": len(boundary_backoffs),
        "context_compactions": sum(int(row.get("context_compactions", 0)) for row in budget + blank),
    },
    "format_reliability": {
        "budget_status": dict(status_counts["budget"]),
        "blank_status": dict(status_counts["blank"]),
        "budget_valid_rate": len(budget_valid) / len(budget_scored),
        "blank_valid_rate": len(blank_valid) / len(blank_scored),
        "fallback_finish_classes": dict(fallback_finish_classes),
        "all_generation_finish_types": dict(finish_types),
        "truncated_generation_count": len(truncation_details),
        "truncated_but_parsed_count": len(truncated_but_parsed),
        "truncated_but_parsed_details": truncated_but_parsed,
        "budget_unique_samples_with_fallback": len(budget_fallback_sample_ids),
        "blank_unique_samples_with_fallback": len(blank_fallback_sample_ids),
        "samples_with_fallback_in_both": len(budget_fallback_sample_ids & blank_fallback_sample_ids),
        "budget_fallback_condition_counts": dict(sorted(fallback_condition_counts.items(), key=lambda item: (item[0] == "FULL", int(item[0]) if item[0] != "FULL" else 10**9))),
        "fallback_details": fallback_details,
    },
    "quality": {
        "budget_all_records": subset_metrics(budget_scored),
        "budget_valid_only": subset_metrics(budget_valid),
        "budget_fallback_only": subset_metrics(budget_fallback),
        "blank_all_records": subset_metrics(blank_scored),
        "blank_valid_only": subset_metrics(blank_valid),
        "blank_fallback_only": subset_metrics(blank_fallback),
        "full_vs_blank_paired": paired,
        "common_budget_curve": common_curve,
        "adjacent_loose_changes": adjacent_loose,
        "quality_then_cost_oracle": comparison["quality_then_cost_oracle"],
        "answer_flip_summary": flip_summary,
        "blank_visibility_diagnostic": blank_visibility_diagnostic,
    },
    "compute": {
        "physical_model_calls": physical_model_calls,
        "physical_retriever_calls": physical_retriever_calls,
        "root_wall_hours": root_elapsed / 3600,
        "budget_wall_hours": budget_elapsed / 3600,
        "blank_wall_hours": blank_elapsed / 3600,
        "approx_sequential_stage_wall_hours": (root_elapsed + budget_elapsed + blank_elapsed) / 3600,
    },
}

output = WORKSPACE / "fanoutqa_random100_v31_independent_audit.json"
output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({
    "passed": report["passed"],
    "errors": errors,
    "integrity": report["integrity"],
    "budget_grid": report["budget_grid"],
    "protocol": {key: value for key, value in report["protocol"].items() if key != "shortfall_examples"},
    "format_reliability": {key: value for key, value in report["format_reliability"].items() if key not in {"fallback_details", "budget_fallback_condition_counts"}},
    "quality": {key: value for key, value in report["quality"].items() if key in {"budget_all_records", "budget_valid_only", "budget_fallback_only", "blank_all_records", "blank_valid_only", "blank_fallback_only", "full_vs_blank_paired", "blank_visibility_diagnostic"}},
    "compute": report["compute"],
    "output": str(output),
}, ensure_ascii=False, indent=2))

if errors:
    raise SystemExit(1)
