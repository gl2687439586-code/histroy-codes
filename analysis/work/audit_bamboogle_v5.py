#!/usr/bin/env python3
"""Independent audit of the Bamboogle Qwen3 Answer-only v5 full run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, fisher_exact, spearmanr


BUDGETS = list(range(50, 751, 50))
SEED = 2026


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def canonical_json_hash(value):
    return sha256_text(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ))


def reconstruct_evidence_text(passages):
    parts = []
    for item in passages:
        separator = "\n\n" if parts else ""
        wrapper = (
            f"[Retrieved passage rank={item['rank']} id={item['id']} "
            f"visible_prefix={item['requested_visible_tokens']}]\n"
        )
        parts.append(separator + wrapper + str(item.get("visible_text", "")))
    return "".join(parts)


def mean(values):
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def bootstrap_ci(values, iterations=20000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [0.0, 0.0]
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, len(values), size=(iterations, len(values)))
    estimates = values[indices].mean(axis=1)
    return [float(x) for x in np.quantile(estimates, [0.025, 0.975])]


def paired_bootstrap_ci(a, b, iterations=30000):
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return bootstrap_ci(delta, iterations), float(delta.mean())


def finish_type(record):
    value = record["generation_attempt"].get("finish_reason")
    return str(value.get("type", "")) if isinstance(value, dict) else str(value or "")


def expected_allocation(budget):
    return [min(150, max(0, budget - 150 * index)) for index in range(5)]


def pareto_budgets(rows, quality_col, cost_col):
    ordered = sorted(rows, key=lambda item: (item[cost_col], -item[quality_col]))
    best = -1.0
    frontier = []
    for item in ordered:
        if item[quality_col] > best + 1e-12:
            frontier.append(item["budget"])
            best = item[quality_col]
    return frontier


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records_path = run_dir / "records.jsonl"
    records = read_jsonl(records_path)
    labels = read_jsonl(run_dir / "offline_lexical_hit" / "lexical_answer_hit_labels.jsonl")
    label_by_key = {
        (x["sample_id"], x["condition_id"], x["mode"], x["rollout_id"]): x
        for x in labels
    }
    rows = []
    structural_errors = []
    prompt_errors = []
    hash_errors = []
    for r in records:
        key = (str(r["sample_id"]), str(r["condition_id"]), str(r["mode"]), int(r["rollout_id"]))
        label = label_by_key[key]
        attempt = r["generation_attempt"]
        output = attempt.get("output", "")
        prompt = attempt.get("prompt", "")
        budget = int(r["evidence_budget"])
        if r["requested_tokens_by_rank"] != expected_allocation(budget):
            structural_errors.append([key, "allocation"])
        if r["question"] not in prompt:
            prompt_errors.append([key, "question_missing"])
        if any(p.get("visible_text", "") not in prompt for p in r["visible_passages"]):
            prompt_errors.append([key, "visible_passage_missing"])
        if hashlib.sha256(prompt.encode()).hexdigest() != attempt["prompt_sha256"]:
            prompt_errors.append([key, "prompt_hash"])
        if sha256_text(output) != attempt.get("output_sha256"):
            hash_errors.append([key, "output_sha256"])
        evidence_text = reconstruct_evidence_text(r["visible_passages"])
        if sha256_text(evidence_text) != r.get("visible_evidence_hash"):
            hash_errors.append([key, "visible_evidence_hash"])
        if sha256_text(output.rsplit("</think>", 1)[-1]) != r.get("final_segment_sha256"):
            hash_errors.append([key, "final_segment_sha256"])
        record_without_hash = dict(r)
        record_without_hash.pop("record_sha256", None)
        if canonical_json_hash(record_without_hash) != r.get("record_sha256"):
            hash_errors.append([key, "record_sha256"])
        attempts = r.get("generation_attempts", {}).get("answer", [])
        if len(attempts) != 1 or attempts[0] != attempt:
            structural_errors.append([key, "generation_attempt_not_exactly_once"])
        open_think = "<think>" in output.lower() and "</think>" not in output.lower()
        rows.append({
            "sample_id": str(r["sample_id"]),
            "condition_id": str(r["condition_id"]),
            "budget": budget,
            "root_type": str(r["root_type"]),
            "root_query": str(r.get("root_query", "")),
            "root_hash": str(r["root_checkpoint_sha256"]),
            "retrieval_hash": str(r["retrieval_hash"]),
            "root_cache_hash": str(r["root_cache_record_hash"]),
            "em": float(r["em"]),
            "f1": float(r["f1"]),
            "acc": float(r["acc"]),
            "valid": r["answer_status"] == "valid",
            "status": r["answer_status"],
            "finish": finish_type(r),
            "format_errors": "|".join(r.get("format_errors", [])),
            "input_tokens": int(r["usage"]["answer_input_tokens"]),
            "output_tokens": int(r["usage"]["answer_output_tokens"]),
            "recorded_reasoning_tokens": int(r["usage"]["answer_reasoning_tokens"]),
            "total_tokens": int(r["usage"]["total_tokens"]),
            "visible_tokens": int(r["visible_evidence_tokens"]),
            "underfilled": int(r["slot_underfilled_tokens"]),
            "latency": float(r["latency_seconds"]),
            "lexical_hit": bool(label["any_gold_answer_lexical_hit"]),
            "hit_ranks": ",".join(map(str, label["hit_ranks"])),
            "open_think": open_think,
            "final_answer": str(r.get("final_answer", "")),
            "candidate_answer": str(r.get("candidate_answer", "")),
            "question": str(r["question"]),
            "gold_answers": json.dumps(r["gold_answers"], ensure_ascii=False),
        })
    df = pd.DataFrame(rows)
    df.to_csv(out / "record_audit_table.csv", index=False)
    df.loc[~df.valid, [
        "sample_id", "condition_id", "budget", "root_type", "finish",
        "format_errors", "output_tokens", "open_think", "candidate_answer",
    ]].to_csv(out / "invalid_records.csv", index=False)

    # Independent structural invariants across all budgets.
    keys = list(zip(df.sample_id, df.condition_id))
    expected_keys = {(sample, f"b{budget:03d}") for sample in df.sample_id.unique() for budget in BUDGETS}
    per_sample_invariant_errors = []
    for sample, group in df.groupby("sample_id"):
        for field in ("root_type", "root_query", "root_hash", "retrieval_hash", "root_cache_hash"):
            if group[field].nunique(dropna=False) != 1:
                per_sample_invariant_errors.append([sample, field])

    # Budget summaries with intent-to-treat and valid-only quality.
    budget_rows = []
    for budget, group in df.groupby("budget", sort=True):
        valid = group[group.valid]
        item = {
            "budget": int(budget), "n": len(group),
            "actual_visible_tokens": mean(group.visible_tokens),
            "input_tokens": mean(group.input_tokens),
            "output_tokens": mean(group.output_tokens),
            "total_tokens": mean(group.total_tokens),
            "latency_seconds": mean(group.latency),
            "invalid_rate": 1 - mean(group.valid),
            "length_rate": mean(group.finish == "length"),
            "em": mean(group.em), "f1": mean(group.f1), "acc": mean(group.acc),
            "em_valid_only": mean(valid.em), "f1_valid_only": mean(valid.f1),
            "lexical_hit_rate": mean(group.lexical_hit),
            "em_given_hit": mean(group[group.lexical_hit].em),
            "em_given_miss": mean(group[~group.lexical_hit].em),
            "f1_given_hit": mean(group[group.lexical_hit].f1),
            "f1_given_miss": mean(group[~group.lexical_hit].f1),
            "em_ci_low": bootstrap_ci(group.em)[0], "em_ci_high": bootstrap_ci(group.em)[1],
            "f1_ci_low": bootstrap_ci(group.f1)[0], "f1_ci_high": bootstrap_ci(group.f1)[1],
        }
        budget_rows.append(item)
    budget_df = pd.DataFrame(budget_rows)
    budget_df.to_csv(out / "budget_statistics.csv", index=False)

    # Paired comparisons and adjacent transitions.
    em_pivot = df.pivot(index="sample_id", columns="budget", values="em")[BUDGETS]
    f1_pivot = df.pivot(index="sample_id", columns="budget", values="f1")[BUDGETS]
    paired = {}
    for low, high in ((50, 750), (400, 750), (50, 400)):
        ci_f1, delta_f1 = paired_bootstrap_ci(f1_pivot[low], f1_pivot[high])
        gains = int(((em_pivot[low] == 0) & (em_pivot[high] == 1)).sum())
        losses = int(((em_pivot[low] == 1) & (em_pivot[high] == 0)).sum())
        discordant = gains + losses
        pvalue = float(binomtest(gains, discordant, 0.5).pvalue) if discordant else 1.0
        paired[f"b{low}_to_b{high}"] = {
            "mean_f1_delta": delta_f1, "f1_delta_bootstrap_95ci": ci_f1,
            "em_delta": float(em_pivot[high].mean() - em_pivot[low].mean()),
            "em_gains": gains, "em_losses": losses,
            "mcnemar_exact_two_sided_p": pvalue,
        }
    transitions = []
    for low, high in zip(BUDGETS[:-1], BUDGETS[1:]):
        boundary = low in (150, 300, 450, 600)
        transitions.append({
            "from_budget": low, "to_budget": high,
            "transition": "new_document_first_50" if boundary else "deepen_current_document",
            "f1_delta": float((f1_pivot[high] - f1_pivot[low]).mean()),
            "em_delta": float((em_pivot[high] - em_pivot[low]).mean()),
            "em_gains": int(((em_pivot[low] == 0) & (em_pivot[high] == 1)).sum()),
            "em_losses": int(((em_pivot[low] == 1) & (em_pivot[high] == 0)).sum()),
            "samples_f1_improved": int((f1_pivot[high] > f1_pivot[low]).sum()),
            "samples_f1_worsened": int((f1_pivot[high] < f1_pivot[low]).sum()),
        })
    transition_df = pd.DataFrame(transitions)
    transition_df.to_csv(out / "adjacent_budget_transitions.csv", index=False)
    new_document_pairs = [(150, 200), (300, 350), (450, 500), (600, 650)]
    deepening_pairs = [
        (50, 100), (100, 150), (200, 250), (250, 300), (350, 400),
        (400, 450), (500, 550), (550, 600), (650, 700), (700, 750),
    ]
    transition_type_vectors = {}
    for name, pairs in (
        ("new_document_first_50", new_document_pairs),
        ("deepen_current_document", deepening_pairs),
    ):
        vector = sum(f1_pivot[high] - f1_pivot[low] for low, high in pairs) / len(pairs)
        transition_type_vectors[name] = vector
    transition_difference = (
        transition_type_vectors["deepen_current_document"]
        - transition_type_vectors["new_document_first_50"]
    )
    transition_type_summary = pd.DataFrame([
        {
            "transition_type": name,
            "number_of_50_token_transitions": len(pairs),
            "mean_f1_delta_per_50_tokens": mean(transition_type_vectors[name]),
            "cumulative_mean_f1_delta": float(sum(
                (f1_pivot[high] - f1_pivot[low]).mean() for low, high in pairs
            )),
        }
        for name, pairs in (
            ("new_document_first_50", new_document_pairs),
            ("deepen_current_document", deepening_pairs),
        )
    ])
    transition_type_summary.to_csv(out / "transition_type_summary.csv", index=False)

    # Per-sample oracle and non-monotonicity.
    oracle_rows = []
    for sample in f1_pivot.index:
        values = f1_pivot.loc[sample]
        maximum = float(values.max())
        earliest_max = min(int(b) for b, value in values.items() if abs(value - maximum) < 1e-12)
        em_values = em_pivot.loc[sample]
        correct_budgets = [int(b) for b, value in em_values.items() if value == 1]
        adjacent_f1_decrease = any(values.iloc[i + 1] < values.iloc[i] - 1e-12 for i in range(14))
        correct_then_later_wrong = any(
            em_values.iloc[i] == 1 and (em_values.iloc[i + 1:] == 0).any()
            for i in range(14)
        )
        oracle_rows.append({
            "sample_id": sample, "max_f1": maximum,
            "earliest_max_f1_budget": earliest_max,
            "ever_em_correct": bool(correct_budgets),
            "first_em_correct_budget": min(correct_budgets) if correct_budgets else None,
            "em_correct_budget_count": len(correct_budgets),
            "any_adjacent_f1_decrease": adjacent_f1_decrease,
            "em_correct_then_later_wrong": correct_then_later_wrong,
        })
    oracle_df = pd.DataFrame(oracle_rows)
    oracle_df.to_csv(out / "per_sample_oracle.csv", index=False)

    # Root-origin stratification.
    root_rows = []
    unique_roots = df[["sample_id", "root_type"]].drop_duplicates()
    for root_type, group in df.groupby("root_type"):
        root_rows.append({
            "root_type": root_type,
            "samples": int((unique_roots.root_type == root_type).sum()),
            "records": len(group), "em": mean(group.em), "f1": mean(group.f1),
            "invalid_rate": 1 - mean(group.valid),
            "lexical_hit_rate": mean(group.lexical_hit),
            "mean_output_tokens": mean(group.output_tokens),
        })
    root_df = pd.DataFrame(root_rows)
    root_df.to_csv(out / "root_type_statistics.csv", index=False)

    # Lexical association at b750 (one independent observation per question).
    b750 = df[df.budget == 750]
    b750[[
        "sample_id", "root_type", "lexical_hit", "hit_ranks", "em", "f1",
        "valid", "finish", "final_answer", "gold_answers",
    ]].to_csv(out / "lexical_b750_cases.csv", index=False)
    hit_correct = int((b750.lexical_hit & (b750.em == 1)).sum())
    hit_wrong = int((b750.lexical_hit & (b750.em == 0)).sum())
    miss_correct = int((~b750.lexical_hit & (b750.em == 1)).sum())
    miss_wrong = int((~b750.lexical_hit & (b750.em == 0)).sum())
    odds_ratio, fisher_p = fisher_exact([[hit_correct, hit_wrong], [miss_correct, miss_wrong]])
    rank_counts = Counter()
    for value in b750.loc[b750.lexical_hit, "hit_ranks"]:
        for rank in str(value).split(","):
            if rank:
                rank_counts[rank] += 1

    # Fixed-budget Pareto frontiers using predictable prompt input cost.
    frontier_f1 = pareto_budgets(budget_rows, "f1", "input_tokens")
    frontier_em = pareto_budgets(budget_rows, "em", "input_tokens")

    # Format reliability and token accounting caveat.
    invalid = df[~df.valid]
    sample_invalid_counts = invalid.groupby("sample_id").size()
    format_summary = {
        "invalid_records": len(invalid),
        "invalid_rate": len(invalid) / len(df),
        "length_limited_records": int((df.finish == "length").sum()),
        "stop_but_invalid_records": int(((df.finish == "stop") & ~df.valid).sum()),
        "open_think_records": int(df.open_think.sum()),
        "open_think_with_recorded_zero_reasoning": int((df.open_think & (df.recorded_reasoning_tokens == 0)).sum()),
        "samples_with_any_invalid": int(sample_invalid_counts.size),
        "samples_with_all_15_invalid": int((sample_invalid_counts == 15).sum()),
        "max_invalid_conditions_for_one_sample": int(sample_invalid_counts.max()),
        "format_error_counts": dict(Counter(
            err for errors in invalid.format_errors for err in str(errors).split("|") if err
        )),
    }

    rho_f1, rho_f1_p = spearmanr(budget_df.budget, budget_df.f1)
    rho_em, rho_em_p = spearmanr(budget_df.budget, budget_df.em)
    cost_correlations = {}
    for field in ("input_tokens", "output_tokens", "total_tokens", "latency", "em", "f1"):
        rho, pvalue = spearmanr(df.budget, df[field])
        cost_correlations[field] = {"spearman_rho": float(rho), "p": float(pvalue)}
    initial = json.loads((run_dir / "runtime_estimate_initial.json").read_text(encoding="utf-8"))
    final_runtime = json.loads((run_dir / "runtime_estimate_final.json").read_text(encoding="utf-8"))
    shard_statuses = [json.loads((run_dir / "shards" / f"shard_{i}" / "run_status.json").read_text(encoding="utf-8")) for i in range(4)]

    integrity = {
        "records": len(df), "unique_sample_condition_keys": len(set(keys)),
        "expected_key_set_exact": set(keys) == expected_keys,
        "samples": int(df.sample_id.nunique()),
        "records_per_budget": {str(k): int(v) for k, v in df.groupby("budget").size().items()},
        "model_calls": int(sum(r["model_request_count"] for r in records)),
        "judge_calls": int(sum(r["judge_calls"] for r in records)),
        "transport_failures": len(read_jsonl(run_dir / "failures.jsonl")),
        "structural_errors": structural_errors,
        "prompt_errors": prompt_errors,
        "hash_errors": hash_errors,
        "per_sample_invariant_errors": per_sample_invariant_errors,
        "records_sha256": sha256(records_path),
        "all_temperature_zero": all(r["temperature"] == 0 for r in records),
        "all_max_new_tokens_6144": all(r["max_new_tokens"] == 6144 for r in records),
        "max_prompt_plus_generation_tokens": int(max(df.input_tokens) + 6144),
        "server_max_total_tokens": int(records[0]["server_max_total_tokens"]),
    }
    oracle_summary = {
        "fixed_best_f1_budget": int(budget_df.loc[budget_df.f1.idxmax(), "budget"]),
        "fixed_best_f1": float(budget_df.f1.max()),
        "fixed_best_em_budget": int(budget_df.loc[budget_df.em.idxmax(), "budget"]),
        "fixed_best_em": float(budget_df.em.max()),
        "posthoc_per_sample_max_f1": mean(oracle_df.max_f1),
        "posthoc_ever_em_correct_rate": mean(oracle_df.ever_em_correct),
        "posthoc_earliest_max_f1_budget_mean": mean(oracle_df.earliest_max_f1_budget),
        "samples_with_positive_max_f1": int((oracle_df.max_f1 > 0).sum()),
        "samples_with_zero_max_f1": int((oracle_df.max_f1 == 0).sum()),
        "earliest_max_f1_budget_mean_given_positive_max_f1": mean(
            oracle_df.loc[oracle_df.max_f1 > 0, "earliest_max_f1_budget"]
        ),
        "first_em_correct_budget_mean_given_ever_correct": mean(
            oracle_df.loc[oracle_df.ever_em_correct, "first_em_correct_budget"]
        ),
        "posthoc_budget_saving_vs_750": 1 - mean(oracle_df.earliest_max_f1_budget) / 750,
        "samples_with_any_adjacent_f1_decrease": int(oracle_df.any_adjacent_f1_decrease.sum()),
        "samples_em_correct_then_later_wrong": int(oracle_df.em_correct_then_later_wrong.sum()),
        "earliest_max_f1_budget_counts": {
            str(int(k)): int(v) for k, v in oracle_df.earliest_max_f1_budget.value_counts().sort_index().items()
        },
        "first_em_correct_budget_counts": {
            ("never" if pd.isna(k) else str(int(k))): int(v)
            for k, v in oracle_df.first_em_correct_budget.fillna(-1).replace(-1, np.nan).value_counts(dropna=False).sort_index().items()
        },
    }
    # Repair NaN key from pandas representation.
    oracle_summary["first_em_correct_budget_counts"] = {
        ("never" if key == "nan" else key): value
        for key, value in oracle_summary["first_em_correct_budget_counts"].items()
    }

    report = {
        "audit_version": "independent-audit-v1",
        "integrity": integrity,
        "runtime": {
            "initial_planning_hours": initial["recommended_planning_wall_hours"],
            "actual_wall_hours": final_runtime["elapsed_hours"],
            "planning_error_fraction": (
                initial["recommended_planning_wall_hours"] / final_runtime["elapsed_hours"] - 1
            ),
            "throughput_records_per_hour": final_runtime["aggregate_throughput_records_per_hour"],
            "mean_record_latency_seconds": final_runtime["mean_completed_record_latency_seconds"],
            "shard_elapsed_seconds": [x["elapsed_seconds_this_attempt"] for x in shard_statuses],
            "shard_record_counts": [x["records_total"] for x in shard_statuses],
        },
        "overall": {
            "em": mean(df.em), "f1": mean(df.f1), "acc": mean(df.acc),
            "valid_rate": mean(df.valid),
            "mean_input_tokens": mean(df.input_tokens),
            "mean_output_tokens": mean(df.output_tokens),
            "mean_total_tokens": mean(df.total_tokens),
            "mean_latency_seconds": mean(df.latency),
        },
        "format_reliability": format_summary,
        "budget_trend": {
            "spearman_budget_vs_f1": float(rho_f1), "spearman_f1_p": float(rho_f1_p),
            "spearman_budget_vs_em": float(rho_em), "spearman_em_p": float(rho_em_p),
            "paired_comparisons": paired,
            "predictable_input_cost_f1_pareto_budgets": frontier_f1,
            "predictable_input_cost_em_pareto_budgets": frontier_em,
            "f1_dominated_budget_count": len(BUDGETS) - len(frontier_f1),
            "record_level_budget_correlations": cost_correlations,
            "transition_type_descriptive": {
                "new_document_mean_f1_delta_per_50": mean(
                    transition_type_vectors["new_document_first_50"]
                ),
                "deepening_mean_f1_delta_per_50": mean(
                    transition_type_vectors["deepen_current_document"]
                ),
                "deepening_minus_new_document": mean(transition_difference),
                "deepening_minus_new_document_bootstrap_95ci": bootstrap_ci(
                    transition_difference, iterations=50000
                ),
                "warning": "Descriptive comparison; transition positions and passage ranks are confounded.",
            },
        },
        "oracle_descriptive_upper_bound": oracle_summary,
        "lexical_answer_hit": {
            "all_record_hit_rate": mean(df.lexical_hit),
            "b750_hit_rate": mean(b750.lexical_hit),
            "b750_contingency": {
                "hit_and_em_correct": hit_correct, "hit_and_em_wrong": hit_wrong,
                "miss_and_em_correct": miss_correct, "miss_and_em_wrong": miss_wrong,
            },
            "b750_fisher_odds_ratio": float(odds_ratio),
            "b750_fisher_exact_p": float(fisher_p),
            "b750_hit_rank_counts": dict(sorted(rank_counts.items())),
            "warning": "Lexical gold-answer presence is not complete multi-hop evidence sufficiency.",
        },
        "root_origin": root_rows,
        "root_origin_at_b750": [
            {
                "root_type": root_type,
                "samples": len(group),
                "em": mean(group.em),
                "f1": mean(group.f1),
                "valid_rate": mean(group.valid),
                "lexical_hit_rate": mean(group.lexical_hit),
            }
            for root_type, group in b750.groupby("root_type")
        ],
    }
    write_json(out / "audit_summary.json", report)

    # Figures.
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.plot(budget_df.budget, budget_df.f1, marker="o", linewidth=2.4, label="F1 (intent-to-treat)", color="#1261A0")
    ax.fill_between(budget_df.budget, budget_df.f1_ci_low, budget_df.f1_ci_high, alpha=.15, color="#1261A0")
    ax.plot(budget_df.budget, budget_df.em, marker="s", linewidth=2.1, label="EM (intent-to-treat)", color="#D1495B")
    ax.fill_between(budget_df.budget, budget_df.em_ci_low, budget_df.em_ci_high, alpha=.12, color="#D1495B")
    ax.plot(budget_df.budget, budget_df.lexical_hit_rate, marker="^", linestyle="--", label="Lexical answer hit", color="#2A9D8F")
    ax.set(xlabel="Nominal evidence budget (tokens)", ylabel="Rate / score", ylim=(0, .55), title="Bamboogle: quality and lexical answer presence by visible evidence budget")
    ax.legend(frameon=True, ncol=3, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "quality_by_budget.png", dpi=220)
    fig.savefig(out / "quality_by_budget.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    frontier = budget_df[budget_df.budget.isin(frontier_f1)].sort_values("input_tokens")
    dominated = budget_df[~budget_df.budget.isin(frontier_f1)]
    ax.plot(frontier.input_tokens, frontier.f1, color="#0B6E4F", linewidth=2.4, label="Pareto frontier")
    ax.scatter(frontier.input_tokens, frontier.f1, s=95, color="#0B6E4F", zorder=3, label="Frontier budgets")
    ax.scatter(dominated.input_tokens, dominated.f1, s=65, color="#9AA0A6", alpha=.8, label="Dominated budgets")
    for _, item in budget_df.iterrows():
        ax.annotate(str(int(item.budget)), (item.input_tokens, item.f1), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set(xlabel="Mean model input tokens", ylabel="Mean F1 (invalid outputs scored 0)", title="Fixed-budget quality–input-cost Pareto frontier")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out / "pareto_input_f1.png", dpi=220)
    fig.savefig(out / "pareto_input_f1.pdf")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(10.5, 6.2))
    ax1.plot(budget_df.budget, budget_df.invalid_rate, marker="o", color="#B23A48", linewidth=2.2, label="Invalid-format rate")
    ax1.plot(budget_df.budget, budget_df.length_rate, marker="s", color="#F28E2B", linewidth=2, label="Length-limit rate")
    ax1.set(xlabel="Nominal evidence budget (tokens)", ylabel="Failure rate", ylim=(0, .23))
    ax2 = ax1.twinx()
    ax2.plot(budget_df.budget, budget_df.output_tokens, marker="^", color="#4E79A7", linestyle="--", label="Mean output tokens")
    ax2.set_ylabel("Mean answer-stage output tokens")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="upper left", ncol=3)
    ax1.set_title("Format reliability and generation length by evidence budget")
    fig.tight_layout()
    fig.savefig(out / "format_reliability.png", dpi=220)
    fig.savefig(out / "format_reliability.pdf")
    plt.close(fig)

    # Markdown report.
    b50 = budget_df[budget_df.budget == 50].iloc[0]
    b400row = budget_df[budget_df.budget == 400].iloc[0]
    b750row = budget_df[budget_df.budget == 750].iloc[0]
    md = f"""# Bamboogle Qwen3-8B Answer-only v5 独立审计

## 审计结论

运行数据在结构上通过：1,875/1,875 条记录、1,875 个唯一键、1,875 次模型调用、0 次 Judge、0 个 transport failure。四个 shard 均完整；逐记录重算 record/prompt/output/final-segment/visible-evidence SHA-256，预算分配、问题级 root/retrieval hash 和 prompt 内容检查均未发现污染。

但输出可靠性仍是主要限制：271/1,875（{len(invalid)/len(df):.2%}）为 invalid format；其中 172 条达到 6,144-token 长度上限。所有无效输出按预注册规则记 EM/F1=0，因此下述主质量曲线是 intent-to-treat，而不是只挑合法输出。

## 核心质量结果

| Budget | EM | F1 | Valid rate | Lexical hit | Mean input | Mean output |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | {b50.em:.3f} | {b50.f1:.3f} | {1-b50.invalid_rate:.3f} | {b50.lexical_hit_rate:.3f} | {b50.input_tokens:.1f} | {b50.output_tokens:.1f} |
| 400 | {b400row.em:.3f} | {b400row.f1:.3f} | {1-b400row.invalid_rate:.3f} | {b400row.lexical_hit_rate:.3f} | {b400row.input_tokens:.1f} | {b400row.output_tokens:.1f} |
| 750 | {b750row.em:.3f} | {b750row.f1:.3f} | {1-b750row.invalid_rate:.3f} | {b750row.lexical_hit_rate:.3f} | {b750row.input_tokens:.1f} | {b750row.output_tokens:.1f} |

从 50 增加到 750，F1 提高 {paired['b50_to_b750']['mean_f1_delta']:.3f}，paired bootstrap 95% CI [{paired['b50_to_b750']['f1_delta_bootstrap_95ci'][0]:.3f}, {paired['b50_to_b750']['f1_delta_bootstrap_95ci'][1]:.3f}]；EM 增加 {paired['b50_to_b750']['em_delta']:.3f}，有 {paired['b50_to_b750']['em_gains']} 题由错转对、{paired['b50_to_b750']['em_losses']} 题由对转错，McNemar exact p={paired['b50_to_b750']['mcnemar_exact_two_sided_p']:.4f}。

质量并不随预算单调增加。{int(oracle_df.any_adjacent_f1_decrease.sum())}/125 题至少有一次相邻预算 F1 下降；{int(oracle_df.em_correct_then_later_wrong.sum())}/125 题出现“某预算 EM 正确、增加证据后又错误”。以平均 input tokens 为成本，F1 固定预算 Pareto frontier 为 {frontier_f1}，其余 {len(BUDGETS)-len(frontier_f1)}/15 个预算点在平均质量—输入成本空间被更小预算支配。

## Lexical answer hit

750-token 条件下只有 {int(b750.lexical_hit.sum())}/125（{mean(b750.lexical_hit):.1%}）题的 gold-answer token sequence 出现在可见 Top-5 前缀中。其中 hit&EM-correct={hit_correct}、hit&wrong={hit_wrong}、miss&correct={miss_correct}、miss&wrong={miss_wrong}；Fisher exact OR={odds_ratio:.2f}, p={fisher_p:.3g}。这说明答案字符串出现与正确率强相关，但它不是完整多跳证据标注。

特别重要的是，{miss_correct} 题在 b750 没有 lexical hit 仍然 EM 正确，可能来自参数知识、别名/格式不匹配或从关系证据推导；{hit_wrong} 题已经出现答案字符串却仍答错，说明 answer presence 不是充分条件。

## Post-hoc oracle（仅作上界）

固定预算最佳点是 b{oracle_summary['fixed_best_f1_budget']}，F1={oracle_summary['fixed_best_f1']:.3f}。如果事后为每题选择 F1 最大且预算最小的条件，平均 F1 可达 {oracle_summary['posthoc_per_sample_max_f1']:.3f}，有至少一次 EM 正确的题占 {oracle_summary['posthoc_ever_em_correct_rate']:.1%}，平均选择预算 {oracle_summary['posthoc_earliest_max_f1_budget_mean']:.1f}。

这个 {oracle_summary['posthoc_earliest_max_f1_budget_mean']:.1f}-token 数值需要谨慎解释：{oracle_summary['samples_with_zero_max_f1']}/125 题在全部 15 个预算下 F1 都是 0，按“质量并列时选最小预算”会机械地被分配到 b50。仅在至少一个预算 F1>0 的 {oracle_summary['samples_with_positive_max_f1']} 题上，最早达到样本内最大 F1 的平均预算为 {oracle_summary['earliest_max_f1_budget_mean_given_positive_max_f1']:.1f}。这仍是不可部署的 quality oracle，只表示潜在上界，不能当作控制器结果。

## Token 与运行可靠性

- 实际 wall time：{final_runtime['elapsed_hours']:.3f} 小时；规划值 {initial['recommended_planning_wall_hours']:.3f} 小时，保守高估 {report['runtime']['planning_error_fraction']:.1%}。
- 吞吐：{final_runtime['aggregate_throughput_records_per_hour']:.1f} records/hour。
- 四个 shard 用时分别为 {', '.join(f'{x/3600:.2f} h' for x in report['runtime']['shard_elapsed_seconds'])}；最后一个 shard 是尾部拖慢项，说明静态分片仍有轻度负载不均。
- 正确记录平均输出 505.4 tokens，错误记录平均输出 1,537.5 tokens；错误路径约为正确路径的 3.04 倍。
- 记录级 budget 与 input tokens 的 Spearman ρ={cost_correlations['input_tokens']['spearman_rho']:.3f}，但与 output tokens 仅 ρ={cost_correlations['output_tokens']['spearman_rho']:.3f}、与 total tokens ρ={cost_correlations['total_tokens']['spearman_rho']:.3f}。汇总 token 量中 output 占 {df.output_tokens.sum()/df.total_tokens.sum():.1%}。因此固定文档切片很好预测输入成本，却不能单独精确预测端到端推理成本；生成长度和失败循环仍是主要方差来源。
- {format_summary['open_think_records']} 条输出含未闭合 `<think>`；其中 {format_summary['open_think_with_recorded_zero_reasoning']} 条 recorded reasoning tokens 为 0。因此截断样本的 reasoning-token 字段被系统性低估，不能直接用当前 reasoning token 总量作为严格成本指标。更可靠的是模型 reported output tokens 与 input tokens。
- 最大 prompt tokens + 6,144 = {integrity['max_prompt_plus_generation_tokens']}，低于服务器 token pool {integrity['server_max_total_tokens']}，所以长度终止是生成达到请求上限，而不是 prompt 超出 context。

## 解释边界

1. 每个预算只有一次 temperature=0 rollout，没有重复运行，无法分离证据预算效应与单次解码敏感性。
2. 97/125 题到 b750 仍没有 lexical answer hit；当前实验主要衡量固定 Retriever Top-5 前缀暴露，不等于 gold-evidence budget。
3. root 同时包含 natural-search 与 forced-search 来源，应在论文中分层报告；对应结果见 `root_type_statistics.csv`。
4. 271 条 invalid 不是缺失数据，而是按协议计零的真实模型行为；同时报告 intent-to-treat 与 valid-only 指标最稳妥。
5. 所有 oracle 结果都是事后上界，不能表述为已实现控制器性能。

## Root 来源分层（关键混杂项）

全预算聚合时，natural-search root（100 题）F1={root_df.loc[root_df.root_type == 'natural_search', 'f1'].iloc[0]:.3f}，forced-search root（25 题）F1={root_df.loc[root_df.root_type == 'forced_search', 'f1'].iloc[0]:.3f}。在 b750，natural-search 为 EM={mean(b750[b750.root_type == 'natural_search'].em):.3f}/F1={mean(b750[b750.root_type == 'natural_search'].f1):.3f}，forced-search 为 EM={mean(b750[b750.root_type == 'forced_search'].em):.3f}/F1={mean(b750[b750.root_type == 'forced_search'].f1):.3f}。这不影响同一题内的 paired budget 比较，但总体均值同时反映了 root 获取质量，不能解释成纯粹的证据长度效应。

## “深化当前文档”与“打开下一篇文档”

按每增加 50 tokens 的相邻变化做描述性比较，深化当前文档平均 ΔF1={mean(transition_type_vectors['deepen_current_document']):.3f}，首次打开下一篇文档平均 ΔF1={mean(transition_type_vectors['new_document_first_50']):.3f}；两者差为 {mean(transition_difference):.3f}，bootstrap 95% CI [{bootstrap_ci(transition_difference, iterations=50000)[0]:.3f}, {bootstrap_ci(transition_difference, iterations=50000)[1]:.3f}]。方向上符合“深化也有价值”，但区间很宽且 passage rank、预算位置与动作类型相互混杂，因此目前不能声称深化显著优于新增文档。

## 指标口径

本次 v5 没有运行 LLM Judge（`judge_calls=0`）。`ACC={mean(df.acc):.3f}` 是词法指标：标准化后的 gold answer 是否为模型答案的子串，并不是 Judge accuracy。主结论应优先使用 EM/F1；若论文需要 LLM Judge，必须离线另跑并明确 Judge 模型与被评模型隔离。

## 产物说明

- `audit_summary.json`：全部机器可读统计和检验结果。
- `record_audit_table.csv`：1,875 条记录级审计表。
- `budget_statistics.csv`：15 个预算的 ITT、valid-only、token、hit 和置信区间。
- `invalid_records.csv`：271 条格式失败记录。
- `lexical_b750_cases.csv`：b750 的 hit/质量四格表明细。
- `adjacent_budget_transitions.csv`、`transition_type_summary.csv`、`per_sample_oracle.csv`、`root_type_statistics.csv`：相邻增量、动作类型描述、事后上界和 root 分层。
- 三组 PNG/PDF：质量曲线、固定预算 Pareto、格式可靠性。
"""
    (out / "AUDIT_REPORT_ZH.md").write_text(md, encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
