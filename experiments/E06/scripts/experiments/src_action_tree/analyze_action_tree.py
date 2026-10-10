#!/usr/bin/env python3
"""Analyze SmartSearch Search-Read-Commit action-tree rollouts.

Outputs sequence quality/cost, backward-DP Quality-Oracle values, all-state and
oracle-path action distributions, pairwise action findings, Pareto dominance,
and token/doc-budget constrained oracles.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import mean, pstdev, read_jsonl, utc_stamp  # noqa: E402
from run_action_tree import expected_paths  # noqa: E402


ACTION_NAME = {"S": "Search", "R": "Read", "C": "Commit"}
ACTION_PRIORITY = {"S": 1, "R": 2, "C": 3}  # required tie-break: C > R > S


def state_definitions(horizon: int) -> Dict[str, Dict[str, Any]]:
    paths = expected_paths(horizon)
    states = sorted(
        {
            path[:depth]
            for path in paths
            for depth in range(1, min(len(path), horizon))
            if "C" not in path[:depth]
        },
        key=lambda state: (len(state), state),
    )
    output: Dict[str, Dict[str, Any]] = {}
    for state in states:
        output[state] = {
            "decision": f"Decision{len(state)}",
            "actions": {
                action: tuple(path for path in paths if path.startswith(state + action))
                for action in ("S", "R", "C")
            },
        }
    return output


def has_read_to_search_transition(path: str) -> bool:
    body = path[1:]
    return any(action == "S" and "R" in body[:index] for index, action in enumerate(body))


NUMERIC_FIELDS = (
    "em",
    "f1",
    "acc",
    "normalized_exact",
    "normalized_contains",
    "search_calls",
    "read_calls",
    "search_query_fallback_calls",
    "repeated_search_query_calls",
    "retrieved_docs_total",
    "retrieved_docs_nominal",
    "retrieved_docs_unique",
    "retrieved_tokens_total",
    "retrieved_tokens_actual",
    "new_evidence_tokens",
    "new_evidence_tokens_actual",
    "read_input_tokens",
    "read_output_tokens",
    "reasoning_tokens",
    "answer_tokens",
    "total_llm_input_tokens",
    "total_llm_output_tokens",
    "llm_tokens_total",
    "total_cost_tokens",
    "total_cost",
    "latency_seconds",
    "terminal_completed",
    "terminal_finish_was_length",
)


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields_list: List[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields_list.append(key)
        fields = fields_list
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", quoting=csv.QUOTE_ALL)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def numeric(row: Dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0))
    except Exception:
        return 0.0


def path_tie_key(path: str) -> Tuple[int, ...]:
    # Fixed initialization S is ignored. Higher tuple is preferred.
    return tuple(ACTION_PRIORITY.get(action, 0) for action in path[1:])


def choose_best_path(path_values: Dict[str, float], eligible: Optional[Iterable[str]] = None) -> str:
    paths = list(eligible if eligible is not None else path_values.keys())
    if not paths:
        return ""
    return max(paths, key=lambda path: (path_values[path], path_tie_key(path), path))


def choose_quality_then_cost_path(
    path_values: Dict[str, float],
    path_costs: Dict[str, float],
    eligible: Optional[Iterable[str]] = None,
) -> str:
    """Maximize quality, then minimize cost, with the fixed tie-break tertiary."""
    paths = list(eligible if eligible is not None else path_values.keys())
    if not paths:
        return ""
    best_quality = max(path_values[path] for path in paths)
    quality_ties = [path for path in paths if abs(path_values[path] - best_quality) <= 1e-12]
    best_cost = min(path_costs[path] for path in quality_ties)
    cost_ties = [path for path in quality_ties if abs(path_costs[path] - best_cost) <= 1e-12]
    return max(cost_ties, key=lambda path: (path_tie_key(path), path))


def quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    p = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = (p + z2 / (2.0 * total)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * total)) / total) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_mean_interval(
    values: Sequence[float],
    *,
    samples: int,
    seed: int,
) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1 or samples <= 0:
        value = float(values[0])
        return value, value
    rng = random.Random(seed)
    n = len(values)
    estimates = [mean([float(values[rng.randrange(n)]) for _ in range(n)]) for _ in range(samples)]
    return quantile(estimates, 0.025), quantile(estimates, 0.975)


def exact_sign_test(positive: int, negative: int) -> float:
    """Two-sided exact sign test, excluding ties."""
    total = positive + negative
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(positive, negative) + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def parse_budget_values(spec: str, auto_source: Sequence[float]) -> List[float]:
    if not spec:
        return []
    if spec.strip().lower() == "auto":
        return sorted(set(round(quantile(auto_source, q), 2) for q in (0.25, 0.5, 0.75, 1.0)))
    return sorted(set(float(part.strip()) for part in spec.split(",") if part.strip()))


def aggregate_sample_paths(records: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[(str(row.get("sample_id")), str(row.get("path")))].append(row)

    rows: List[Dict[str, Any]] = []
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for (sample_id, path), items in sorted(grouped.items()):
        first = items[0]
        aggregate: Dict[str, Any] = {
            "sample_id": sample_id,
            "path": path,
            "horizon": int(first.get("horizon", 3)),
            "original_search_count": int(first.get("original_search_count", 0)),
            "n_rollouts": len(items),
            "question": first.get("question", ""),
            "gold_answers": json.dumps(first.get("gold_answers", []), ensure_ascii=False),
            "final_answers": json.dumps([item.get("final_answer", "") for item in items], ensure_ascii=False),
        }
        for field in NUMERIC_FIELDS:
            values = [numeric(item, field) for item in items]
            aggregate[f"{field}_mean"] = mean(values)
            aggregate[f"{field}_std"] = pstdev(values)
            aggregate[f"{field}_max"] = max(values) if values else 0.0
            aggregate[f"{field}_min"] = min(values) if values else 0.0
        rows.append(aggregate)
        lookup[sample_id][path] = aggregate
    return rows, lookup


def aggregate_sequences(sample_path_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # The same early-Commit path (for example SC) can occur in every horizon.
    # Keep horizon strata separate so that adaptive-horizon difficulty is not
    # silently mixed into a single sequence mean.
    grouped: Dict[Tuple[int, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in sample_path_rows:
        grouped[(int(row.get("horizon", 3)), str(row["path"]))].append(row)
    output: List[Dict[str, Any]] = []
    for horizon, path in sorted(grouped, key=lambda item: (item[0], len(item[1]), item[1])):
        items = grouped.get((horizon, path), [])
        if not items:
            continue
        row: Dict[str, Any] = {"horizon": horizon, "path": path, "samples": len(items)}
        for field in NUMERIC_FIELDS:
            values = [float(item[f"{field}_mean"]) for item in items]
            row[f"{field}_mean"] = mean(values)
            row[f"{field}_std_across_samples"] = pstdev(values)
        row["search_count_expected"] = path.count("S")
        row["read_count_expected"] = path.count("R")
        row["early_commit"] = path.endswith("C")
        output.append(row)
    return output


def paired_sequence_comparisons(
    sample_lookup: Dict[str, Dict[str, Dict[str, Any]]],
    quality_metric: str,
    cost_metric: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> List[Dict[str, Any]]:
    """Matched R-vs-S swaps with identical parent state and future suffix.

    To avoid treating many nodes from one question as independent samples, all
    matched suffix effects are first averaged within sample and decision depth,
    then uncertainty is computed across questions.
    """
    per_sample_decision_quality: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    per_sample_decision_cost: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    matched_pairs: Counter = Counter()
    for sample_id, paths in sample_lookup.items():
        horizon = int(next(iter(paths.values())).get("horizon", 3))
        definitions = state_definitions(horizon)
        for state in definitions:
            decision = len(state)
            for read_path in paths:
                prefix = state + "R"
                if not read_path.startswith(prefix):
                    continue
                suffix = read_path[len(prefix):]
                search_path = state + "S" + suffix
                if search_path not in paths:
                    continue
                per_sample_decision_quality[(sample_id, horizon, decision)].append(
                    float(paths[read_path][f"{quality_metric}_mean"])
                    - float(paths[search_path][f"{quality_metric}_mean"])
                )
                per_sample_decision_cost[(sample_id, horizon, decision)].append(
                    float(paths[read_path][f"{cost_metric}_mean"])
                    - float(paths[search_path][f"{cost_metric}_mean"])
                )
                matched_pairs[(horizon, decision)] += 1

    rows: List[Dict[str, Any]] = []
    observed_horizons = sorted({key[1] for key in per_sample_decision_quality})
    strata: List[Optional[int]] = [None] + observed_horizons
    pair_index = 0
    for horizon_filter in strata:
        decisions = sorted(
            {
                key[2]
                for key in per_sample_decision_quality
                if horizon_filter is None or key[1] == horizon_filter
            }
        )
        for decision in decisions:
            keys = sorted(
                key
                for key in per_sample_decision_quality
                if key[2] == decision and (horizon_filter is None or key[1] == horizon_filter)
            )
            quality_diffs = [mean(per_sample_decision_quality[key]) for key in keys]
            cost_diffs = [mean(per_sample_decision_cost[key]) for key in keys]
            positive = sum(value > 1e-12 for value in quality_diffs)
            negative = sum(value < -1e-12 for value in quality_diffs)
            ties = len(quality_diffs) - positive - negative
            q_low, q_high = bootstrap_mean_interval(
                quality_diffs,
                samples=bootstrap_samples,
                seed=bootstrap_seed + pair_index * 2,
            )
            c_low, c_high = bootstrap_mean_interval(
                cost_diffs,
                samples=bootstrap_samples,
                seed=bootstrap_seed + pair_index * 2 + 1,
            )
            rows.append(
                {
                    "horizon": horizon_filter if horizon_filter is not None else "all",
                    "decision": f"Decision{decision}",
                    "n_samples": len(quality_diffs),
                    "matched_terminal_path_pairs": sum(
                        count
                        for (horizon, pair_decision), count in matched_pairs.items()
                        if pair_decision == decision
                        and (horizon_filter is None or horizon == horizon_filter)
                    ),
                    "quality_metric": quality_metric,
                    "mean_quality_difference_read_minus_search": mean(quality_diffs),
                    "quality_difference_ci95_low": q_low,
                    "quality_difference_ci95_high": q_high,
                    "read_wins": positive,
                    "ties": ties,
                    "search_wins": negative,
                    "two_sided_exact_sign_test_p": exact_sign_test(positive, negative),
                    "cost_metric": cost_metric,
                    "mean_cost_difference_read_minus_search": mean(cost_diffs),
                    "cost_difference_ci95_low": c_low,
                    "cost_difference_ci95_high": c_high,
                }
            )
            pair_index += 1
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index]["two_sided_exact_sign_test_p"]))
    running = 0.0
    total_tests = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (total_tests - rank) * float(rows[index]["two_sided_exact_sign_test_p"]))
        running = max(running, adjusted)
        rows[index]["holm_adjusted_sign_test_p"] = running
    return rows


def metric_sensitivity(
    sample_lookup: Dict[str, Dict[str, Dict[str, Any]]],
    metrics: Sequence[str],
    value_field: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
    path_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    paths_by_sample: Dict[str, Set[str]] = defaultdict(set)
    for metric in metrics:
        counts: Counter = Counter()
        qualities: List[float] = []
        costs: List[float] = []
        completed_rates: List[float] = []
        for sample_id, paths in sorted(sample_lookup.items()):
            _, lookup = action_values_for_sample(sample_id, paths, metric)
            oracle_path, _ = oracle_for_sample(sample_id, lookup, value_field)
            row = paths[oracle_path]
            counts[oracle_path] += 1
            paths_by_sample[sample_id].add(oracle_path)
            qualities.append(float(row[f"{metric}_mean"]))
            costs.append(float(row["total_cost_tokens_mean"]))
            completed_rates.append(float(row.get("terminal_completed_mean", 0.0)))
            path_rows.append(
                {
                    "sample_id": sample_id,
                    "horizon": int(row.get("horizon", 3)),
                    "metric": metric,
                    "oracle_path": oracle_path,
                    "oracle_quality": qualities[-1],
                    "total_cost_tokens": costs[-1],
                    "terminal_completed": completed_rates[-1],
                }
            )
        n = len(sample_lookup)
        summaries.append(
            {
                "metric": metric,
                "samples": n,
                "oracle_path_counts": json.dumps(dict(counts), ensure_ascii=False, sort_keys=True),
                "mean_oracle_quality": mean(qualities),
                "mean_oracle_cost_tokens": mean(costs),
                "oracle_terminal_completion_rate": mean(completed_rates),
                "read_first_oracle_ratio": (
                    sum(count for path, count in counts.items() if len(path) > 1 and path[1] == "R") / n
                    if n else 0.0
                ),
                "read_to_search_oracle_ratio": (
                    sum(count for path, count in counts.items() if has_read_to_search_transition(path)) / n
                    if n else 0.0
                ),
            }
        )
    agreement = mean([float(len(paths) == 1) for paths in paths_by_sample.values()])
    return path_rows, summaries, agreement


def action_values_for_sample(
    sample_id: str,
    paths: Dict[str, Dict[str, Any]],
    quality_metric: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Dict[str, Any]]]]:
    path_mean = {path: float(value[f"{quality_metric}_mean"]) for path, value in paths.items()}
    path_max = {path: float(value[f"{quality_metric}_max"]) for path, value in paths.items()}
    rows: List[Dict[str, Any]] = []
    lookup: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    horizon = int(next(iter(paths.values())).get("horizon", 3))

    for state, definition in state_definitions(horizon).items():
        for action, descendant_paths in definition["actions"].items():
            available = [path for path in descendant_paths if path in paths]
            if not available:
                continue
            best_mean_path = choose_best_path(path_mean, available)
            best_max_path = choose_best_path(path_max, available)
            descendant_rollout_values: List[float] = []
            for path in available:
                # We only need mean/max in the flat row; per-rollout values are
                # represented by max and the path rollout count.
                descendant_rollout_values.append(path_mean[path])
            row = {
                "sample_id": sample_id,
                "horizon": horizon,
                "state": state,
                "decision": definition["decision"],
                "action": action,
                "action_name": ACTION_NAME[action],
                "descendant_paths": ",".join(available),
                # Best-case oracle over all descendant terminal rollouts.
                "qmax": max(path_max[path] for path in available),
                "best_path_qmax": best_max_path,
                # Expected-return oracle: first mean repeated rollouts for each
                # terminal sequence, then take the best descendant sequence.
                "qmean": path_mean[best_mean_path],
                "best_path_qmean": best_mean_path,
                # Descriptive robustness, not used for DP action selection.
                "q_descendant_path_mean": mean(descendant_rollout_values),
            }
            rows.append(row)
            lookup[state][action] = row
    return rows, lookup


def select_action(action_rows: Dict[str, Dict[str, Any]], value_field: str) -> str:
    return max(
        action_rows,
        key=lambda action: (float(action_rows[action][value_field]), ACTION_PRIORITY[action]),
    )


def oracle_for_sample(
    sample_id: str,
    action_lookup: Dict[str, Dict[str, Dict[str, Any]]],
    value_field: str,
) -> Tuple[str, List[Dict[str, Any]]]:
    state = "S"
    decisions: List[Dict[str, Any]] = []
    while state in action_lookup:
        action = select_action(action_lookup[state], value_field)
        decisions.append(
            {
                "sample_id": sample_id,
                "decision": f"Decision{len(state)}",
                "state": state,
                "action": action,
                "action_name": ACTION_NAME[action],
                "value": action_lookup[state][action][value_field],
            }
        )
        next_path = state + action
        if action == "C" or next_path not in action_lookup:
            return next_path, decisions
        state = next_path
    raise RuntimeError(f"No oracle state available for {sample_id}: {state}")


def distribution_rows(
    decisions: Sequence[Dict[str, Any]],
    scope: str,
    horizon: Optional[int] = None,
) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for row in decisions:
        if horizon is not None and int(row.get("horizon", 3)) != horizon:
            continue
        grouped[(str(row["decision"]), str(row.get("state", "all")))][str(row["action"])] += 1
    output: List[Dict[str, Any]] = []
    for (decision, state), counts in sorted(grouped.items()):
        denominator = sum(counts.values())
        for action in ("S", "R", "C"):
            count = counts[action]
            output.append(
                {
                    "scope": scope,
                    "horizon": horizon if horizon is not None else "all",
                    "decision": decision,
                    "state": state,
                    "action": action,
                    "action_name": ACTION_NAME[action],
                    "count": count,
                    "denominator_active_states": denominator,
                    "percentage": 100.0 * count / denominator if denominator else 0.0,
                }
            )
    return output


def pareto_rows(
    sample_lookup: Dict[str, Dict[str, Dict[str, Any]]],
    quality_metric: str,
    cost_metric: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    frontier: List[Dict[str, Any]] = []
    dominated: List[Dict[str, Any]] = []
    for sample_id, paths in sorted(sample_lookup.items()):
        for path, row in paths.items():
            quality = float(row[f"{quality_metric}_mean"])
            cost = float(row[f"{cost_metric}_mean"])
            dominators: List[str] = []
            for other_path, other in paths.items():
                if other_path == path:
                    continue
                other_quality = float(other[f"{quality_metric}_mean"])
                other_cost = float(other[f"{cost_metric}_mean"])
                if other_quality >= quality and other_cost <= cost and (other_quality > quality or other_cost < cost):
                    dominators.append(other_path)
            target = dominated if dominators else frontier
            target.append(
                {
                    "sample_id": sample_id,
                    "horizon": int(row.get("horizon", 3)),
                    "path": path,
                    "quality_metric": quality_metric,
                    "quality": quality,
                    "cost_metric": cost_metric,
                    "cost": cost,
                    "dominated": bool(dominators),
                    "dominated_by": ",".join(sorted(dominators)),
                }
            )
    return frontier, dominated


def aggregate_pareto(sequence_rows: Sequence[Dict[str, Any]], quality_metric: str, cost_metric: str) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in sequence_rows:
        path = str(row["path"])
        quality = float(row[f"{quality_metric}_mean"])
        cost = float(row[f"{cost_metric}_mean"])
        dominators = []
        for other in sequence_rows:
            # Only compare policies observed under the same allowed horizon.
            if int(other.get("horizon", 3)) != int(row.get("horizon", 3)):
                continue
            if other["path"] == path:
                continue
            oq = float(other[f"{quality_metric}_mean"])
            oc = float(other[f"{cost_metric}_mean"])
            if oq >= quality and oc <= cost and (oq > quality or oc < cost):
                dominators.append(str(other["path"]))
        output.append(
            {
                "horizon": int(row.get("horizon", 3)),
                "path": path,
                "quality": quality,
                "cost": cost,
                "pareto_optimal": not dominators,
                "dominated_by": ",".join(sorted(dominators)),
            }
        )
    return output


def budget_oracle(
    sample_lookup: Dict[str, Dict[str, Dict[str, Any]]],
    quality_metric: str,
    cost_metric: str,
    budgets: Sequence[float],
    budget_kind: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    for budget in budgets:
        for sample_id, paths in sorted(sample_lookup.items()):
            sample_horizon = int(next(iter(paths.values())).get("horizon", 3))
            eligible = [
                path for path, row in paths.items() if float(row[f"{cost_metric}_mean"]) <= budget + 1e-12
            ]
            if not eligible:
                selected_rows.append(
                    {
                        "budget_kind": budget_kind,
                        "cost_metric": cost_metric,
                        "budget": budget,
                        "sample_id": sample_id,
                        "horizon": sample_horizon,
                        "feasible": False,
                        "selected_path": "",
                    }
                )
                continue
            path_values = {path: float(paths[path][f"{quality_metric}_mean"]) for path in eligible}
            selected = choose_best_path(path_values)
            chosen = paths[selected]
            selected_rows.append(
                {
                    "budget_kind": budget_kind,
                    "cost_metric": cost_metric,
                    "budget": budget,
                    "sample_id": sample_id,
                    "horizon": int(chosen.get("horizon", 3)),
                    "feasible": True,
                    "selected_path": selected,
                    "quality": path_values[selected],
                    "cost": float(chosen[f"{cost_metric}_mean"]),
                }
            )
            # Emit every active decision on a variable-length selected path.
            # The leading S is the fixed initialization, not a controller choice.
            for index, action in enumerate(selected[1:], start=1):
                decision_rows.append(
                    {
                        "budget_kind": budget_kind,
                        "cost_metric": cost_metric,
                        "budget": budget,
                        "sample_id": sample_id,
                        "horizon": int(chosen.get("horizon", 3)),
                        "decision": f"Decision{index}",
                        "state": selected[:index],
                        "action": action,
                    }
                )

    distributions: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, str, float, str], Counter] = defaultdict(Counter)
    grouped_horizon: Dict[Tuple[str, str, float, int, str], Counter] = defaultdict(Counter)
    for row in decision_rows:
        key = (row["budget_kind"], row["cost_metric"], float(row["budget"]), row["decision"])
        grouped[key][row["action"]] += 1
        horizon_key = (
            row["budget_kind"],
            row["cost_metric"],
            float(row["budget"]),
            int(row.get("horizon", 3)),
            row["decision"],
        )
        grouped_horizon[horizon_key][row["action"]] += 1
    for (kind, metric, budget, decision), counts in sorted(grouped.items()):
        denominator = sum(counts.values())
        feasible_samples = sum(
            1
            for row in selected_rows
            if row["budget_kind"] == kind and row["cost_metric"] == metric and float(row["budget"]) == budget and row["feasible"]
        )
        for action in ("S", "R", "C"):
            distributions.append(
                {
                    "budget_kind": kind,
                    "cost_metric": metric,
                    "budget": budget,
                    "horizon": "all",
                    "decision": decision,
                    "action": action,
                    "action_name": ACTION_NAME[action],
                    "count": counts[action],
                    "denominator_active_states": denominator,
                    "feasible_samples": feasible_samples,
                    "percentage": 100.0 * counts[action] / denominator if denominator else 0.0,
                }
            )
    for (kind, metric, budget, horizon, decision), counts in sorted(grouped_horizon.items()):
        denominator = sum(counts.values())
        feasible_samples = sum(
            1
            for row in selected_rows
            if row["budget_kind"] == kind
            and row["cost_metric"] == metric
            and float(row["budget"]) == budget
            and row["feasible"]
            and int(row.get("horizon", 3)) == horizon
        )
        for action in ("S", "R", "C"):
            distributions.append(
                {
                    "budget_kind": kind,
                    "cost_metric": metric,
                    "budget": budget,
                    "horizon": horizon,
                    "decision": decision,
                    "action": action,
                    "action_name": ACTION_NAME[action],
                    "count": counts[action],
                    "denominator_active_states": denominator,
                    "feasible_samples": feasible_samples,
                    "percentage": 100.0 * counts[action] / denominator if denominator else 0.0,
                }
            )
    return selected_rows, distributions


def markdown_table(rows: Sequence[Dict[str, Any]], columns: Sequence[Tuple[str, str]], digits: int = 4) -> List[str]:
    output = ["| " + " | ".join(label for _, label in columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                value = f"{value:.{digits}f}"
            values.append(str(value).replace("|", "&#124;"))
        output.append("| " + " | ".join(values) + " |")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="terminal_trajectories.jsonl")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--quality_metric", choices=("em", "f1", "acc", "normalized_exact", "normalized_contains"), default="f1")
    parser.add_argument("--oracle_value", choices=("auto", "qmax", "qmean"), default="auto")
    parser.add_argument("--pareto_cost_metric", default="total_cost_tokens")
    parser.add_argument("--token_budget_metric", default="total_cost_tokens")
    parser.add_argument("--token_budgets", default="auto", help="Comma-separated values or auto")
    parser.add_argument("--doc_budget_metric", default="retrieved_docs_nominal")
    parser.add_argument("--doc_budgets", default="5,10,15")
    parser.add_argument(
        "--sensitivity_metrics",
        default="f1,em,acc,normalized_contains",
        help="Comma-separated answer metrics for oracle-path robustness checks",
    )
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--bootstrap_seed", type=int, default=2026)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_records = read_jsonl(args.input)
    if not raw_records:
        raise RuntimeError(f"No trajectories found in {args.input}")

    # A process can be interrupted while append_jsonl is writing a completed
    # tree. On resume, keep the newest record for each tree/path so a harmless
    # partial append cannot inflate rollout counts or exclude the sample.
    latest_records: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for row in raw_records:
        key = (
            str(row.get("sample_id")),
            int(row.get("rollout_id", 0)),
            str(row.get("path")),
        )
        latest_records[key] = row
    duplicate_records_ignored = len(raw_records) - len(latest_records)
    records = list(latest_records.values())
    records = [
        row
        for row in records
        if str(row.get("path")) in set(expected_paths(int(row.get("horizon", 3))))
    ]
    sample_path_rows, sample_lookup = aggregate_sample_paths(records)

    completeness_rows: List[Dict[str, Any]] = []
    complete_samples: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for sample_id, paths in sorted(sample_lookup.items()):
        horizons = {int(row.get("horizon", 3)) for row in paths.values()}
        horizon = next(iter(horizons)) if len(horizons) == 1 else -1
        valid_paths = set(expected_paths(horizon)) if horizon >= 2 else set()
        missing = sorted(valid_paths - set(paths))
        extra = sorted(set(paths) - valid_paths)
        rollout_counts = {path: int(paths[path]["n_rollouts"]) for path in paths}
        complete = len(horizons) == 1 and not missing and not extra and len(set(rollout_counts.values())) == 1
        completeness_rows.append(
            {
                "sample_id": sample_id,
                "complete": complete,
                "horizon": horizon,
                "missing_paths": ",".join(missing),
                "extra_paths": ",".join(extra),
                "rollout_counts": json.dumps(rollout_counts, sort_keys=True),
            }
        )
        if complete:
            complete_samples[sample_id] = paths
    if not complete_samples:
        raise RuntimeError("No sample has a complete variable-horizon tree with balanced rollout counts")

    # Exclude incomplete samples from every comparative/oracle statistic.
    sample_path_rows = [row for row in sample_path_rows if row["sample_id"] in complete_samples]
    sequence_rows = aggregate_sequences(sample_path_rows)
    write_csv(output_dir / "completeness.csv", completeness_rows)
    write_csv(output_dir / "sample_sequence_values.csv", sample_path_rows)
    write_csv(output_dir / "sequence_level_quality_cost.csv", sequence_rows)
    metric_disagreements = [
        {
            "sample_id": row["sample_id"],
            "path": row["path"],
            "final_answers": row["final_answers"],
            "em": row["em_mean"],
            "f1": row["f1_mean"],
            "acc": row["acc_mean"],
            "normalized_contains": row["normalized_contains_mean"],
            "terminal_completed": row.get("terminal_completed_mean", 0.0),
        }
        for row in sample_path_rows
        if (
            abs(float(row["em_mean"]) - float(row["acc_mean"])) > 1e-12
            or (1e-12 < float(row["f1_mean"]) < 1.0 - 1e-12)
            or float(row.get("terminal_completed_mean", 0.0)) < 1.0
        )
    ]
    write_csv(output_dir / "metric_disagreements_for_audit.csv", metric_disagreements)

    paired_rows = paired_sequence_comparisons(
        complete_samples,
        args.quality_metric,
        args.pareto_cost_metric,
        args.bootstrap_samples,
        args.bootstrap_seed,
    )
    write_csv(output_dir / "paired_read_vs_search.csv", paired_rows)

    has_repeats = any(int(row["n_rollouts"]) > 1 for row in sample_path_rows)
    value_field = args.oracle_value
    if value_field == "auto":
        value_field = "qmean" if has_repeats else "qmax"

    all_action_rows: List[Dict[str, Any]] = []
    action_lookup_by_sample: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {}
    all_state_decisions: List[Dict[str, Any]] = []
    oracle_path_rows: List[Dict[str, Any]] = []
    quality_then_cost_oracle_rows: List[Dict[str, Any]] = []
    oracle_decisions: List[Dict[str, Any]] = []
    comparisons: List[Dict[str, Any]] = []

    for sample_id, paths in sorted(complete_samples.items()):
        sample_horizon = int(next(iter(paths.values())).get("horizon", 3))
        action_rows, action_lookup = action_values_for_sample(sample_id, paths, args.quality_metric)
        action_lookup_by_sample[sample_id] = action_lookup
        for state, actions in action_lookup.items():
            selected = select_action(actions, value_field)
            for action, row in actions.items():
                row["oracle_value_field"] = value_field
                row["selected_action_at_state"] = action == selected
                all_action_rows.append(row)
            all_state_decisions.append(
                {
                    "sample_id": sample_id,
                    "horizon": sample_horizon,
                    "decision": f"Decision{len(state)}",
                    "state": state,
                    "action": selected,
                }
            )
            comparisons.append(
                {
                    "sample_id": sample_id,
                    "horizon": sample_horizon,
                    "state": state,
                    "decision": f"Decision{len(state)}",
                    "value_field": value_field,
                    "search_value": actions["S"][value_field],
                    "read_value": actions["R"][value_field],
                    "commit_value": actions["C"][value_field],
                    "read_gt_search": float(actions["R"][value_field]) > float(actions["S"][value_field]),
                    "commit_gt_search": float(actions["C"][value_field]) > float(actions["S"][value_field]),
                    "selected_action": selected,
                }
            )
        oracle_path, decisions = oracle_for_sample(sample_id, action_lookup, value_field)
        path_row = paths[oracle_path]
        oracle_path_rows.append(
            {
                "sample_id": sample_id,
                "horizon": sample_horizon,
                "original_search_count": int(path_row.get("original_search_count", 0)),
                "oracle_path": oracle_path,
                "quality_metric": args.quality_metric,
                "quality": path_row[f"{args.quality_metric}_mean"],
                "cost_total_tokens": path_row["total_cost_tokens_mean"],
                "search_calls": path_row["search_calls_mean"],
                "read_calls": path_row["read_calls_mean"],
                "retrieved_docs_total": path_row["retrieved_docs_total_mean"],
            }
        )
        for decision in decisions:
            decision["horizon"] = sample_horizon
        oracle_decisions.extend(decisions)
        path_values = {
            path: float(row[f"{args.quality_metric}_mean"])
            for path, row in paths.items()
        }
        path_costs = {
            path: float(row["total_cost_tokens_mean"])
            for path, row in paths.items()
        }
        quality_then_cost_path = choose_quality_then_cost_path(path_values, path_costs)
        quality_then_cost_row = paths[quality_then_cost_path]
        quality_then_cost_oracle_rows.append(
            {
                "sample_id": sample_id,
                "horizon": sample_horizon,
                "original_search_count": int(quality_then_cost_row.get("original_search_count", 0)),
                "oracle_path": quality_then_cost_path,
                "quality_metric": args.quality_metric,
                "quality": path_values[quality_then_cost_path],
                "cost_total_tokens": path_costs[quality_then_cost_path],
                "search_calls": quality_then_cost_row["search_calls_mean"],
                "read_calls": quality_then_cost_row["read_calls_mean"],
                "retrieved_docs_total": quality_then_cost_row["retrieved_docs_total_mean"],
                "selection_rule": "maximize quality, then minimize total_cost_tokens",
            }
        )

    write_csv(output_dir / "state_action_values.csv", all_action_rows)
    write_csv(output_dir / "action_value_comparisons.csv", comparisons)
    write_csv(output_dir / "quality_oracle_paths.csv", oracle_path_rows)
    write_csv(output_dir / "quality_then_cost_oracle_paths.csv", quality_then_cost_oracle_rows)

    requested_sensitivity_metrics = [
        metric.strip()
        for metric in args.sensitivity_metrics.split(",")
        if metric.strip() in {"em", "f1", "acc", "normalized_exact", "normalized_contains"}
    ]
    sensitivity_paths, sensitivity_summary, metric_agreement = metric_sensitivity(
        complete_samples,
        requested_sensitivity_metrics,
        value_field,
    )
    write_csv(output_dir / "metric_sensitivity_oracle_paths.csv", sensitivity_paths)
    write_csv(output_dir / "metric_sensitivity_summary.csv", sensitivity_summary)

    distribution = []
    distribution.extend(distribution_rows(all_state_decisions, "all_counterfactual_states"))
    distribution.extend(distribution_rows(oracle_decisions, "oracle_path"))
    observed_horizons = sorted(
        {int(next(iter(paths.values())).get("horizon", 3)) for paths in complete_samples.values()}
    )
    for horizon in observed_horizons:
        distribution.extend(
            distribution_rows(all_state_decisions, "all_counterfactual_states_by_horizon", horizon)
        )
        distribution.extend(distribution_rows(oracle_decisions, "oracle_path_by_horizon", horizon))
    # Add the fixed initialization explicitly, but keep it out of Decision1/2 denominators.
    distribution.append(
        {
            "scope": "fixed_initialization",
            "horizon": "all",
            "decision": "Initialization",
            "state": "root",
            "action": "S",
            "action_name": "Search",
            "count": len(complete_samples),
            "denominator_active_states": len(complete_samples),
            "percentage": 100.0,
        }
    )
    write_csv(output_dir / "optimal_action_distributions.csv", distribution)

    frontier, dominated = pareto_rows(complete_samples, args.quality_metric, args.pareto_cost_metric)
    write_csv(output_dir / "pareto_frontier.csv", frontier)
    write_csv(output_dir / "dominated_trajectories.csv", dominated)
    aggregate_frontier = aggregate_pareto(sequence_rows, args.quality_metric, args.pareto_cost_metric)
    write_csv(output_dir / "sequence_level_pareto.csv", aggregate_frontier)

    token_source = [float(row[f"{args.token_budget_metric}_mean"]) for row in sample_path_rows]
    doc_source = [float(row[f"{args.doc_budget_metric}_mean"]) for row in sample_path_rows]
    token_budgets = parse_budget_values(args.token_budgets, token_source)
    doc_budgets = parse_budget_values(args.doc_budgets, doc_source)
    budget_selected: List[Dict[str, Any]] = []
    budget_distribution: List[Dict[str, Any]] = []
    selected, dist = budget_oracle(
        complete_samples, args.quality_metric, args.token_budget_metric, token_budgets, "token"
    )
    budget_selected.extend(selected)
    budget_distribution.extend(dist)
    selected, dist = budget_oracle(
        complete_samples, args.quality_metric, args.doc_budget_metric, doc_budgets, "document"
    )
    budget_selected.extend(selected)
    budget_distribution.extend(dist)
    write_csv(output_dir / "budget_constrained_oracle_paths.csv", budget_selected)
    write_csv(output_dir / "budget_action_distributions.csv", budget_distribution)

    unique_events: Dict[Tuple[str, int, str, str], Dict[str, Any]] = {}
    for trajectory in records:
        sample_id = str(trajectory.get("sample_id"))
        if sample_id not in complete_samples:
            continue
        rollout_id = int(trajectory.get("rollout_id", 0))
        for event in trajectory.get("events", []):
            key = (
                sample_id,
                rollout_id,
                str(event.get("action_code", "")),
                str(event.get("post_state_hash", "")),
            )
            unique_events.setdefault(key, event)
    behavior_rows: List[Dict[str, Any]] = []
    for (sample_id, rollout_id, action_code, post_state_hash), event in sorted(unique_events.items()):
        behavior_rows.append(
            {
                "sample_id": sample_id,
                "rollout_id": rollout_id,
                "action_code": action_code,
                "action": event.get("action", ""),
                "decision_index": event.get("decision_index", ""),
                "fixed_initial_checkpoint": bool(event.get("fixed_initial_checkpoint", False)),
                "post_state_hash": post_state_hash,
                "query": event.get("query", ""),
                "query_source": event.get("query_source", ""),
                "query_was_repeated": bool(event.get("query_was_repeated", False)),
                "query_fallback_used": bool(event.get("query_fallback_used", False)),
                "retrieved_docs": event.get("retrieved_docs", 0),
                "retrieved_docs_nominal": event.get("retrieved_docs_nominal", 0),
                "new_evidence_tokens": event.get("new_evidence_tokens", 0),
                "terminal_completed": event.get("terminal_completed", ""),
                "terminal_finish_was_length": event.get("terminal_finish_was_length", ""),
            }
        )
    write_csv(output_dir / "unique_behavior_events.csv", behavior_rows)
    counterfactual_search_events = [
        row for row in behavior_rows if row["action_code"] == "S" and not row["fixed_initial_checkpoint"]
    ]

    root_comparisons = [row for row in comparisons if row["state"] == "S"]
    all_comparisons = comparisons
    path_counts = Counter(row["oracle_path"] for row in oracle_path_rows)
    quality_then_cost_path_counts = Counter(
        row["oracle_path"] for row in quality_then_cost_oracle_rows
    )
    n_samples = len(complete_samples)
    read_first_count = sum(
        count for path, count in path_counts.items() if len(path) > 1 and path[1] == "R"
    )
    root_read_successes = sum(bool(row["read_gt_search"]) for row in root_comparisons)
    root_commit_successes = sum(bool(row["commit_gt_search"]) for row in root_comparisons)
    comparisons_by_sample: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in all_comparisons:
        comparisons_by_sample[str(row["sample_id"])].append(row)
    sample_read_gt_search_rates = [
        mean([float(row["read_gt_search"]) for row in comparisons_by_sample[sample_id]])
        for sample_id in sorted(comparisons_by_sample)
    ]
    sample_commit_gt_search_rates = [
        mean([float(row["commit_gt_search"]) for row in comparisons_by_sample[sample_id]])
        for sample_id in sorted(comparisons_by_sample)
    ]
    all_read_cluster_ci = bootstrap_mean_interval(
        sample_read_gt_search_rates,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 7001,
    )
    all_commit_cluster_ci = bootstrap_mean_interval(
        sample_commit_gt_search_rates,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 7002,
    )
    non_monotonic_count = sum(
        count for path, count in path_counts.items() if has_read_to_search_transition(path)
    )
    oracle_mean_quality = mean([float(row["quality"]) for row in oracle_path_rows])
    oracle_mean_cost = mean([float(row["cost_total_tokens"]) for row in oracle_path_rows])
    always_search_rows = [
        paths["S" * int(next(iter(paths.values())).get("horizon", 3))]
        for paths in complete_samples.values()
    ]
    always_search_qualities = [float(row[f"{args.quality_metric}_mean"]) for row in always_search_rows]
    always_search_costs = [float(row["total_cost_tokens_mean"]) for row in always_search_rows]
    always_search_mean_quality = mean(always_search_qualities)
    always_search_mean_cost = mean(always_search_costs)
    oracle_by_sample = {str(row["sample_id"]): row for row in oracle_path_rows}
    oracle_vs_always_rows: List[Dict[str, Any]] = []
    for sample_id, paths in sorted(complete_samples.items()):
        horizon = int(next(iter(paths.values())).get("horizon", 3))
        oracle_row = oracle_by_sample[sample_id]
        baseline_row = paths["S" * horizon]
        oracle_quality = float(oracle_row["quality"])
        baseline_quality = float(baseline_row[f"{args.quality_metric}_mean"])
        oracle_cost = float(oracle_row["cost_total_tokens"])
        baseline_cost = float(baseline_row["total_cost_tokens_mean"])
        oracle_vs_always_rows.append(
            {
                "sample_id": sample_id,
                "horizon": horizon,
                "oracle_path": oracle_row["oracle_path"],
                "always_search_path": "S" * horizon,
                "oracle_quality": oracle_quality,
                "always_search_quality": baseline_quality,
                "quality_difference_oracle_minus_always_search": oracle_quality - baseline_quality,
                "oracle_cost_tokens": oracle_cost,
                "always_search_cost_tokens": baseline_cost,
                "cost_difference_oracle_minus_always_search": oracle_cost - baseline_cost,
            }
        )
    write_csv(output_dir / "oracle_vs_always_search_paired.csv", oracle_vs_always_rows)
    oracle_quality_differences = [
        float(row["quality_difference_oracle_minus_always_search"])
        for row in oracle_vs_always_rows
    ]
    oracle_cost_differences = [
        float(row["cost_difference_oracle_minus_always_search"])
        for row in oracle_vs_always_rows
    ]
    oracle_quality_ci = bootstrap_mean_interval(
        oracle_quality_differences,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 9001,
    )
    oracle_cost_ci = bootstrap_mean_interval(
        oracle_cost_differences,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed + 9002,
    )
    oracle_quality_wins = sum(value > 1e-12 for value in oracle_quality_differences)
    oracle_quality_losses = sum(value < -1e-12 for value in oracle_quality_differences)
    oracle_quality_ties = n_samples - oracle_quality_wins - oracle_quality_losses
    oracle_vs_always_summary = {
        "n_samples": n_samples,
        "mean_quality_difference_oracle_minus_always_search": mean(oracle_quality_differences),
        "quality_difference_ci95_bootstrap": oracle_quality_ci,
        "oracle_quality_wins": oracle_quality_wins,
        "quality_ties": oracle_quality_ties,
        "oracle_quality_losses": oracle_quality_losses,
        "quality_two_sided_exact_sign_test_p": exact_sign_test(
            oracle_quality_wins, oracle_quality_losses
        ),
        "mean_cost_difference_oracle_minus_always_search": mean(oracle_cost_differences),
        "cost_difference_ci95_bootstrap": oracle_cost_ci,
    }
    horizon_counts = Counter(
        int(next(iter(paths.values())).get("horizon", 3)) for paths in complete_samples.values()
    )
    completed_terminal_records = [
        row for row in records if row["sample_id"] in complete_samples and bool(row.get("terminal_completed", True))
    ]
    used_terminal_records = [row for row in records if row["sample_id"] in complete_samples]
    horizon_summary_rows: List[Dict[str, Any]] = []
    for horizon in sorted(horizon_counts):
        horizon_ids = {
            sample_id
            for sample_id, paths in complete_samples.items()
            if int(next(iter(paths.values())).get("horizon", 3)) == horizon
        }
        oracle_h = [row for row in oracle_path_rows if int(row["horizon"]) == horizon]
        always_h = [
            complete_samples[sample_id]["S" * horizon]
            for sample_id in sorted(horizon_ids)
        ]
        root_h = [row for row in root_comparisons if int(row.get("horizon", 3)) == horizon]
        read_first_h = sum(
            len(str(row["oracle_path"])) > 1 and str(row["oracle_path"])[1] == "R"
            for row in oracle_h
        )
        non_monotonic_h = sum(
            has_read_to_search_transition(str(row["oracle_path"])) for row in oracle_h
        )
        root_read_h = sum(bool(row["read_gt_search"]) for row in root_h)
        root_commit_h = sum(bool(row["commit_gt_search"]) for row in root_h)
        frontier_h = [row for row in frontier if int(row.get("horizon", 3)) == horizon]
        dominated_h = [row for row in dominated if int(row.get("horizon", 3)) == horizon]
        terminal_h = [row for row in used_terminal_records if int(row.get("horizon", 3)) == horizon]
        completed_h = [row for row in terminal_h if bool(row.get("terminal_completed", True))]
        oracle_quality_h = mean([float(row["quality"]) for row in oracle_h])
        oracle_cost_h = mean([float(row["cost_total_tokens"]) for row in oracle_h])
        always_quality_h = mean(
            [float(row[f"{args.quality_metric}_mean"]) for row in always_h]
        )
        always_cost_h = mean([float(row["total_cost_tokens_mean"]) for row in always_h])
        n_h = len(horizon_ids)
        horizon_summary_rows.append(
            {
                "horizon": horizon,
                "samples": n_h,
                "terminal_paths_per_sample": len(expected_paths(horizon)),
                "model_calls_per_tree": 2 ** (horizon + 1) - 3,
                "original_search_count_mean": mean(
                    [float(row.get("original_search_count", 0)) for row in oracle_h]
                ),
                "oracle_mean_quality": oracle_quality_h,
                "always_search_mean_quality": always_quality_h,
                "oracle_quality_gain_vs_always_search": oracle_quality_h - always_quality_h,
                "oracle_mean_cost_tokens": oracle_cost_h,
                "always_search_mean_cost_tokens": always_cost_h,
                "oracle_cost_reduction_vs_always_search": (
                    (always_cost_h - oracle_cost_h) / always_cost_h if always_cost_h else 0.0
                ),
                "read_first_count": read_first_h,
                "read_first_ratio": read_first_h / n_h if n_h else 0.0,
                "read_first_ci95_wilson": json.dumps(wilson_interval(read_first_h, n_h)),
                "non_monotonic_count": non_monotonic_h,
                "non_monotonic_ratio": non_monotonic_h / n_h if n_h else 0.0,
                "non_monotonic_ci95_wilson": json.dumps(
                    wilson_interval(non_monotonic_h, n_h)
                ),
                "p_read_gt_search_root": mean(
                    [float(row["read_gt_search"]) for row in root_h]
                ),
                "p_read_gt_search_root_ci95_wilson": json.dumps(
                    wilson_interval(root_read_h, len(root_h))
                ),
                "p_commit_gt_search_root": mean(
                    [float(row["commit_gt_search"]) for row in root_h]
                ),
                "p_commit_gt_search_root_ci95_wilson": json.dumps(
                    wilson_interval(root_commit_h, len(root_h))
                ),
                "dominated_trajectory_rate": (
                    len(dominated_h) / (len(frontier_h) + len(dominated_h))
                    if frontier_h or dominated_h
                    else 0.0
                ),
                "terminal_completion_rate": len(completed_h) / len(terminal_h) if terminal_h else 0.0,
            }
        )
    write_csv(output_dir / "horizon_summary.csv", horizon_summary_rows)
    findings = {
        "schema_version": "src-action-tree-analysis-v4-variable-horizon",
        "created_at": utc_stamp(),
        "input": args.input,
        "complete_samples": n_samples,
        "terminal_records_used": len([row for row in records if row["sample_id"] in complete_samples]),
        "raw_terminal_records": len(raw_records),
        "duplicate_terminal_records_ignored": duplicate_records_ignored,
        "quality_metric": args.quality_metric,
        "oracle_value_field": value_field,
        "tie_break": "Commit > Read > Search",
        "quality_oracle_definition": "maximize quality with fixed Commit > Read > Search tie-break",
        "quality_then_cost_oracle_definition": "maximize quality, then minimize total_cost_tokens",
        "qmax_definition": "maximum individual terminal-rollout reward among descendants",
        "qmean_definition": "maximum, over descendant terminal sequences, of that sequence's mean reward across repeated rollouts",
        "p_read_gt_search_root": mean([float(row["read_gt_search"]) for row in root_comparisons]),
        "p_read_gt_search_root_ci95_wilson": wilson_interval(root_read_successes, len(root_comparisons)),
        "p_commit_gt_search_root": mean([float(row["commit_gt_search"]) for row in root_comparisons]),
        "p_commit_gt_search_root_ci95_wilson": wilson_interval(root_commit_successes, len(root_comparisons)),
        "p_read_gt_search_all_counterfactual_states": mean(sample_read_gt_search_rates),
        "p_read_gt_search_all_states_ci95_question_cluster_bootstrap": all_read_cluster_ci,
        "p_read_gt_search_all_states_node_weighted_descriptive": mean(
            [float(row["read_gt_search"]) for row in all_comparisons]
        ),
        "p_commit_gt_search_all_counterfactual_states": mean(sample_commit_gt_search_rates),
        "p_commit_gt_search_all_states_ci95_question_cluster_bootstrap": all_commit_cluster_ci,
        "p_commit_gt_search_all_states_node_weighted_descriptive": mean(
            [float(row["commit_gt_search"]) for row in all_comparisons]
        ),
        "quality_oracle_path_counts": dict(path_counts),
        "quality_then_cost_oracle_path_counts": dict(quality_then_cost_path_counts),
        "quality_then_cost_oracle_mean_quality": mean(
            [float(row["quality"]) for row in quality_then_cost_oracle_rows]
        ),
        "quality_then_cost_oracle_mean_cost_tokens": mean(
            [float(row["cost_total_tokens"]) for row in quality_then_cost_oracle_rows]
        ),
        "read_first_oracle_ratio": read_first_count / n_samples,
        "non_monotonic_oracle_policy_frequency": non_monotonic_count / n_samples,
        "non_monotonic_frequency_ci95_wilson": wilson_interval(non_monotonic_count, n_samples),
        "non_monotonic_definition": "the oracle performs Search after at least one earlier Read",
        "horizon_sample_counts": dict(sorted(horizon_counts.items())),
        "horizon_summary": horizon_summary_rows,
        "pareto_frontier_trajectory_count": len(frontier),
        "dominated_trajectory_count": len(dominated),
        "dominated_trajectory_rate": len(dominated) / (len(frontier) + len(dominated)),
        "terminal_completion_rate": len(completed_terminal_records) / len(used_terminal_records),
        "oracle_mean_quality": oracle_mean_quality,
        "oracle_mean_cost_tokens": oracle_mean_cost,
        "always_search_mean_quality": always_search_mean_quality,
        "always_search_mean_cost_tokens": always_search_mean_cost,
        "oracle_cost_reduction_vs_always_search": (
            (always_search_mean_cost - oracle_mean_cost) / always_search_mean_cost
            if always_search_mean_cost else 0.0
        ),
        "oracle_vs_always_search_paired": oracle_vs_always_summary,
        "metric_oracle_path_agreement_rate": metric_agreement,
        "counterfactual_search_nodes": len(counterfactual_search_events),
        "counterfactual_search_repeat_rate": mean(
            [float(bool(row["query_was_repeated"])) for row in counterfactual_search_events]
        ),
        "counterfactual_search_fallback_rate": mean(
            [float(bool(row["query_fallback_used"])) for row in counterfactual_search_events]
        ),
        "counterfactual_search_zero_new_evidence_rate": mean(
            [float(float(row["new_evidence_tokens"]) <= 0.0) for row in counterfactual_search_events]
        ),
        "metric_sensitivity": sensitivity_summary,
        "paired_read_vs_search": paired_rows,
        "token_budgets": token_budgets,
        "doc_budgets": doc_budgets,
    }
    (output_dir / "findings.json").write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")

    report: List[str] = []
    report.append("# SmartSearch Search–Read–Commit Action Tree Results")
    report.append("")
    report.append(f"- Complete samples: **{n_samples}**")
    report.append(f"- Quality metric: **{args.quality_metric}**")
    report.append(f"- Oracle value: **{value_field}**")
    report.append("- Tie-break: **Commit > Read > Search**")
    report.append("- Secondary Quality-then-Cost Oracle: **maximize quality, then minimize total tokens**")
    report.append("")
    report.append("## Sequence-level quality and cost")
    report.append("")
    report.extend(
        markdown_table(
            sequence_rows,
            (
                ("horizon", "Horizon"),
                ("path", "Path"),
                (f"{args.quality_metric}_mean", args.quality_metric.upper()),
                ("total_cost_tokens_mean", "Total cost tokens"),
                ("retrieved_docs_nominal_mean", "Nominal docs"),
                ("search_calls_mean", "Search"),
                ("read_calls_mean", "Read"),
            ),
            digits=3,
        )
    )
    report.append("")
    report.append("## Horizon-stratified summary")
    report.append("")
    report.extend(
        markdown_table(
            horizon_summary_rows,
            (
                ("horizon", "H"),
                ("samples", "N"),
                ("terminal_paths_per_sample", "Paths/tree"),
                ("model_calls_per_tree", "Model calls/tree"),
                ("oracle_mean_quality", "Oracle quality"),
                ("always_search_mean_quality", "Always-S quality"),
                ("oracle_mean_cost_tokens", "Oracle cost"),
                ("always_search_mean_cost_tokens", "Always-S cost"),
                ("read_first_ratio", "Read-first"),
                ("non_monotonic_ratio", "Read-to-Search"),
            ),
            digits=3,
        )
    )
    report.append("")
    report.append("## Core findings")
    report.append("")
    report.append(
        f"- P(Read > Search), Decision1/root: **{findings['p_read_gt_search_root']:.3f}** "
        f"(Wilson 95% CI {findings['p_read_gt_search_root_ci95_wilson'][0]:.3f}–{findings['p_read_gt_search_root_ci95_wilson'][1]:.3f})"
    )
    report.append(f"- P(Commit > Search), Decision1/root: **{findings['p_commit_gt_search_root']:.3f}**")
    report.append(f"- Read-first oracle ratio: **{findings['read_first_oracle_ratio']:.3f}**")
    report.append(
        f"- Non-monotonic oracle frequency (Search after an earlier Read): "
        f"**{findings['non_monotonic_oracle_policy_frequency']:.3f}**"
    )
    report.append(f"- Dominated trajectory rate: **{findings['dominated_trajectory_rate']:.3f}**")
    report.append(f"- Terminal completion rate: **{findings['terminal_completion_rate']:.3f}**")
    report.append(
        f"- Quality oracle vs horizon-matched always-Search: quality "
        f"**{oracle_mean_quality:.3f} vs {always_search_mean_quality:.3f}**, "
        f"cost **{oracle_mean_cost:.1f} vs {always_search_mean_cost:.1f}**, reduction "
        f"**{findings['oracle_cost_reduction_vs_always_search']:.3f}**"
    )
    report.append(
        f"- Paired oracle-minus-always-Search quality difference: "
        f"**{oracle_vs_always_summary['mean_quality_difference_oracle_minus_always_search']:.3f}** "
        f"(bootstrap 95% CI {oracle_quality_ci[0]:.3f}-{oracle_quality_ci[1]:.3f}; "
        f"win/tie/loss {oracle_quality_wins}/{oracle_quality_ties}/{oracle_quality_losses}; "
        f"sign-test p={oracle_vs_always_summary['quality_two_sided_exact_sign_test_p']:.4f})"
    )
    report.append(f"- Oracle-path agreement across answer metrics: **{metric_agreement:.3f}**")
    report.append(
        f"- Counterfactual Search repeat/fallback/zero-new-evidence rates: "
        f"**{findings['counterfactual_search_repeat_rate']:.3f} / "
        f"{findings['counterfactual_search_fallback_rate']:.3f} / "
        f"{findings['counterfactual_search_zero_new_evidence_rate']:.3f}** "
        f"over {findings['counterfactual_search_nodes']} unique Search nodes"
    )
    report.append("")
    report.append("## Matched Read-first vs Search-first comparisons")
    report.append("")
    report.extend(
        markdown_table(
            paired_rows,
            (
                ("horizon", "Horizon"),
                ("decision", "Decision"),
                ("matched_terminal_path_pairs", "Matched pairs"),
                ("mean_quality_difference_read_minus_search", "Δ quality"),
                ("quality_difference_ci95_low", "CI low"),
                ("quality_difference_ci95_high", "CI high"),
                ("read_wins", "Read wins"),
                ("ties", "Ties"),
                ("search_wins", "Search wins"),
                ("two_sided_exact_sign_test_p", "Sign-test p"),
                ("holm_adjusted_sign_test_p", "Holm p"),
                ("mean_cost_difference_read_minus_search", "Δ cost"),
            ),
            digits=4,
        )
    )
    report.append("")
    report.append("## Oracle path counts")
    report.append("")
    report.extend(
        markdown_table(
            [{"path": path, "count": path_counts[path]} for path in sorted(path_counts, key=lambda item: (len(item), item))],
            (("path", "Path"), ("count", "Count")),
        )
    )
    report.append("")
    report.append("Detailed CSV/JSON files in this directory contain state-level Qmax/Qmean, both requested action distributions, Pareto domination witnesses, and token/document-budget constrained policies.")
    (output_dir / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    config = {
        "created_at": utc_stamp(),
        "arguments": vars(args),
        "raw_records_read": len(raw_records),
        "records_after_deduplication": len(records),
        "duplicate_records_ignored": duplicate_records_ignored,
        "complete_samples": n_samples,
        "excluded_incomplete_samples": len(sample_lookup) - n_samples,
        "outputs": sorted(path.name for path in output_dir.iterdir() if path.is_file()),
    }
    (output_dir / "analysis_manifest.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(findings, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
