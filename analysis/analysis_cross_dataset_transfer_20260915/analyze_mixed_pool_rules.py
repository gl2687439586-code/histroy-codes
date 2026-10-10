from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyze_transfer as base


HERE = Path(__file__).resolve().parent
OUT = HERE / "mixed_pool_rules"
BUDGETS = base.COMMON
SEED = base.SEED + 431
ALPHA = 5.0
SOFT_TEMPERATURE = 0.05


def softmax(values: np.ndarray, temperature: float) -> np.ndarray:
    logits = (np.asarray(values, dtype=float) - np.max(values)) / temperature
    weights = np.exp(logits)
    return weights / weights.sum()


def score_midranks(scores: pd.DataFrame) -> dict[int, float]:
    counts = scores.read_score.astype(int).value_counts().sort_index()
    total = counts.sum()
    cumulative = 0
    result = {}
    for score, count in counts.items():
        result[int(score)] = float((cumulative + count / 2) / total)
        cumulative += count
    return result


def quantile_bin(value: float) -> str:
    if value <= 1 / 3:
        return "low"
    if value <= 2 / 3:
        return "mid"
    return "high"


def attach_metadata(data: pd.DataFrame, scores: pd.DataFrame, dataset: str) -> pd.DataFrame:
    ranks = score_midranks(scores)
    meta = scores[["sample_id", "read_score"]].copy()
    meta["sample_id"] = meta.sample_id.astype(str)
    meta["read_score"] = meta.read_score.astype(int)
    meta["score_midrank"] = meta.read_score.map(ranks)
    meta["quantile_bin"] = meta.score_midrank.map(quantile_bin)
    out = data.copy()
    out["sample_id"] = out.sample_id.astype(str)
    out["dataset"] = dataset
    return out.merge(meta, on="sample_id", validate="many_to_one")


def dataset_curve(rows: pd.DataFrame, metric: str, column: str | None, value) -> tuple[np.ndarray, np.ndarray, int]:
    subset = rows if column is None else rows[rows[column] == value]
    means = subset.groupby("budget")[metric].mean().reindex(BUDGETS)
    costs = subset.groupby("budget").total_llm_tokens.mean().reindex(BUDGETS)
    if means.isna().any() or costs.isna().any():
        raise RuntimeError(f"Incomplete curve for {column}={value}")
    return means.to_numpy(float), costs.to_numpy(float), int(subset.sample_id.nunique())


def curve(
    training: pd.DataFrame,
    metric: str,
    group_column: str,
    group_value,
    balanced: bool,
) -> tuple[np.ndarray, np.ndarray, dict]:
    group_rows = training[training[group_column] == group_value]
    if group_rows.empty:
        group_rows = training

    if balanced:
        global_q, global_c, group_q, group_c = [], [], [], []
        group_counts = {}
        for dataset, dataset_rows in training.groupby("dataset"):
            q, c, _ = dataset_curve(dataset_rows, metric, None, None)
            global_q.append(q)
            global_c.append(c)
            local = dataset_rows[dataset_rows[group_column] == group_value]
            group_counts[dataset] = int(local.sample_id.nunique())
            if not local.empty:
                q, c, _ = dataset_curve(local, metric, None, None)
                group_q.append(q)
                group_c.append(c)
        gq = np.mean(group_q, axis=0)
        gc = np.mean(group_c, axis=0)
        bq = np.mean(global_q, axis=0)
        bc = np.mean(global_c, axis=0)
        effective_n = int(group_rows.sample_id.nunique())
    else:
        gq, gc, effective_n = dataset_curve(group_rows, metric, None, None)
        bq, bc, _ = dataset_curve(training, metric, None, None)
        group_counts = {
            dataset: int(rows.sample_id.nunique())
            for dataset, rows in group_rows.groupby("dataset")
        }

    quality = (effective_n * gq + ALPHA * bq) / (effective_n + ALPHA)
    cost = (effective_n * gc + ALPHA * bc) / (effective_n + ALPHA)
    audit = {
        "effective_group_n": effective_n,
        "group_counts_by_dataset": group_counts,
        "quality_curve": {str(b): float(v) for b, v in zip(BUDGETS, quality)},
        "cost_curve": {str(b): float(v) for b, v in zip(BUDGETS, cost)},
    }
    return quality, cost, audit


def hierarchical_curve(
    training: pd.DataFrame,
    held_dataset: str,
    metric: str,
    group_column: str,
    group_value,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Target-conditioned curve with the mixed pool used only as a prior.

    This avoids treating a per-search token cap as identical across datasets
    while still letting the other dataset stabilize small score groups.
    """
    local_all = training[training.dataset == held_dataset]
    local = local_all[local_all[group_column] == group_value]
    pooled = training[training[group_column] == group_value]
    if local.empty:
        local = local_all
    if pooled.empty:
        pooled = training
    lq, lc, local_n = dataset_curve(local, metric, None, None)
    pq, pc, pooled_n = dataset_curve(pooled, metric, None, None)
    quality = (local_n * lq + ALPHA * pq) / (local_n + ALPHA)
    cost = (local_n * lc + ALPHA * pc) / (local_n + ALPHA)
    audit = {
        "effective_group_n": local_n,
        "pooled_prior_n": pooled_n,
        "group_counts_by_dataset": {
            dataset: int(rows.sample_id.nunique())
            for dataset, rows in pooled.groupby("dataset")
        },
        "quality_curve": {str(b): float(v) for b, v in zip(BUDGETS, quality)},
        "cost_curve": {str(b): float(v) for b, v in zip(BUDGETS, cost)},
    }
    return quality, cost, audit


def hard_choice(quality: np.ndarray, cost: np.ndarray) -> int:
    order = sorted(
        range(len(BUDGETS)),
        key=lambda j: (-quality[j], cost[j], BUDGETS[j]),
    )
    return int(BUDGETS[order[0]])


def metric_columns(data: pd.DataFrame) -> list[str]:
    return [
        column for column in [
            "official_loose_accuracy", "official_strict_accuracy", "rougeL_f1",
            "answer_f1", "answer_em", "answer_acc", "total_llm_tokens",
            "visible_evidence_tokens_total", "search_calls",
        ] if column in data.columns
    ]


def loo_policy(
    mixed: pd.DataFrame,
    grouping: str,
    balanced: bool,
    soft: bool,
    name: str,
    target_conditioned: bool = False,
) -> tuple[pd.DataFrame, list[dict]]:
    rows, audits = [], []
    for (dataset, sample_id), held_group in mixed.groupby(["dataset", "sample_id"], sort=False):
        training = mixed[~((mixed.dataset == dataset) & (mixed.sample_id == sample_id))]
        held = held_group.set_index("budget").loc[BUDGETS]
        group_value = held.iloc[0][grouping]
        if target_conditioned:
            quality, cost, audit = hierarchical_curve(
                training, dataset, "answer_f1", grouping, group_value
            )
        else:
            quality, cost, audit = curve(training, "answer_f1", grouping, group_value, balanced)
        base_record = {
            "sample_id": sample_id,
            "dataset": dataset,
            "read_score": int(held.iloc[0].read_score),
            "score_midrank": float(held.iloc[0].score_midrank),
            "quantile_bin": str(held.iloc[0].quantile_bin),
            "policy": name,
            "training_metric": "answer_f1",
            "grouping": grouping,
            "pooling": (
                "target_conditioned_hierarchical"
                if target_conditioned else
                ("dataset_balanced" if balanced else "raw_question_weighted")
            ),
            "own_outcomes_excluded": True,
        }
        if soft:
            probability = softmax(quality, SOFT_TEMPERATURE)
            record = {
                **base_record,
                "selected_budget": float(np.dot(probability, BUDGETS)),
                "budget_probability": json.dumps(
                    {str(b): float(p) for b, p in zip(BUDGETS, probability)},
                    sort_keys=True,
                ),
            }
            for column in metric_columns(held):
                record[column] = float(np.dot(probability, held[column].to_numpy(float)))
        else:
            budget = hard_choice(quality, cost)
            source = held.loc[budget]
            record = {
                **source.to_dict(),
                **base_record,
                "selected_budget": budget,
                "budget_probability": json.dumps({str(budget): 1.0}),
            }
        rows.append(record)
        audits.append({
            "sample_id": sample_id,
            "dataset": dataset,
            "policy": name,
            "group_value": str(group_value),
            **audit,
        })
    return pd.DataFrame(rows), audits


def baseline(rows: pd.DataFrame, name: str, dataset: str) -> pd.DataFrame:
    out = rows.copy()
    out["sample_id"] = out.sample_id.astype(str)
    out["dataset"] = dataset
    out["policy"] = name
    out["selected_budget"] = np.nan
    return out


def fixed(data: pd.DataFrame, budget: int, dataset: str) -> pd.DataFrame:
    out = data[data.budget == budget].copy()
    out["policy"] = f"fixed_{budget}"
    out["selected_budget"] = budget
    return out


def oracle(data: pd.DataFrame, dataset: str) -> pd.DataFrame:
    selected = []
    for _, group in data.groupby("sample_id"):
        selected.append(
            group.sort_values(
                ["answer_f1", "total_llm_tokens", "budget"],
                ascending=[False, True, True],
            ).iloc[0]
        )
    out = pd.DataFrame(selected)
    out["policy"] = "common_grid_answer_f1_oracle"
    out["selected_budget"] = out.budget.astype(int)
    out["dataset"] = dataset
    return out


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = metric_columns(rows)
    return rows.groupby(["dataset", "policy"], dropna=False).agg(
        n=("sample_id", "nunique"),
        mean_selected_budget=("selected_budget", "mean"),
        **{metric: (metric, "mean") for metric in metrics},
    ).reset_index()


def comparisons(candidates: list[pd.DataFrame], references: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    records = []
    for candidate in candidates:
        dataset = str(candidate.dataset.iloc[0])
        for reference in references[dataset]:
            for metric in metric_columns(candidate):
                if metric not in reference.columns:
                    continue
                records.append({
                    "dataset": dataset,
                    "candidate": candidate.policy.iloc[0],
                    "reference": reference.policy.iloc[0],
                    "metric": metric,
                    **base.paired_delta(candidate, reference, metric, SEED + len(records)),
                })
    return pd.DataFrame(records)


def plot(summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    for ax, dataset in zip(axes, ["HotpotQA", "FanOutQA"]):
        sub = summary[summary.dataset == dataset]
        baseline_name = "FULL" if dataset == "HotpotQA" else "BLANK"
        keep = sub.policy.str.startswith("mixed_") | sub.policy.eq(baseline_name) | sub.policy.eq("common_grid_answer_f1_oracle")
        sub = sub[keep]
        for _, row in sub.iterrows():
            color = "#4C78A8"
            if row.policy.startswith("mixed_"):
                color = "#E45756"
            if row.policy.endswith("oracle"):
                color = "#54A24B"
            ax.scatter(row.total_llm_tokens, row.answer_f1, s=85, color=color)
            ax.annotate(row.policy.replace("mixed_", ""), (row.total_llm_tokens, row.answer_f1),
                        xytext=(5, 5), textcoords="offset points", fontsize=7)
        ax.set_title(dataset)
        ax.set_xlabel("Mean total LLM tokens")
        ax.set_ylabel("Answer F1")
    fig.suptitle("Mixed HotpotQA + FanOutQA leave-one-out budget rules", fontsize=16, weight="bold")
    fig.savefig(OUT / "mixed_pool_quality_cost.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "mixed_pool_quality_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    hs, h, h_full, fs, f, f_blank, _ = base.load_data()
    hotpot = attach_metadata(h, hs, "HotpotQA")
    fanout = attach_metadata(f, fs, "FanOutQA")
    mixed = pd.concat([hotpot, fanout], ignore_index=True, sort=False)

    policies, curve_audits = [], []
    specifications = [
        ("read_score", False, False, False, "mixed_exact_score_hard_raw_image_rule"),
        ("read_score", True, False, False, "mixed_exact_score_hard_balanced"),
        ("quantile_bin", True, False, False, "mixed_quantile_hard_balanced"),
        ("read_score", True, True, False, "mixed_exact_score_soft_balanced"),
        ("quantile_bin", True, True, False, "mixed_quantile_soft_balanced"),
        ("read_score", False, False, True, "mixed_hierarchical_exact_hard"),
        ("read_score", False, True, True, "mixed_hierarchical_exact_soft"),
    ]
    for grouping, balanced, soft, target_conditioned, name in specifications:
        result, audit = loo_policy(
            mixed, grouping, balanced, soft, name,
            target_conditioned=target_conditioned,
        )
        policies.append(result)
        curve_audits.extend(audit)

    h_full = baseline(h_full, "FULL", "HotpotQA")
    f_blank = baseline(f_blank, "BLANK", "FanOutQA")
    fixed_h = [fixed(hotpot, budget, "HotpotQA") for budget in BUDGETS]
    fixed_f = [fixed(fanout, budget, "FanOutQA") for budget in BUDGETS]
    oracle_h = oracle(hotpot, "HotpotQA")
    oracle_f = oracle(fanout, "FanOutQA")
    all_rows = pd.concat(
        [*policies, h_full, f_blank, *fixed_h, *fixed_f, oracle_h, oracle_f],
        ignore_index=True,
        sort=False,
    )
    summaries = summarize(all_rows)

    best_fixed = {}
    references = {"HotpotQA": [h_full], "FanOutQA": [f_blank]}
    for dataset, candidates in [("HotpotQA", fixed_h), ("FanOutQA", fixed_f)]:
        fixed_summary = summaries[(summaries.dataset == dataset) & summaries.policy.str.startswith("fixed_")]
        best_name = fixed_summary.sort_values(
            ["answer_f1", "total_llm_tokens", "mean_selected_budget"],
            ascending=[False, True, True],
        ).iloc[0].policy
        best = next(item for item in candidates if item.policy.iloc[0] == best_name)
        references[dataset].append(best)
        best_fixed[dataset] = best_name

    policy_by_dataset = []
    for policy in policies:
        for dataset in ["HotpotQA", "FanOutQA"]:
            policy_by_dataset.append(policy[policy.dataset == dataset].copy())
    paired = comparisons(policy_by_dataset, references)

    all_rows.to_csv(OUT / "per_question_policies.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(OUT / "policy_summary.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUT / "paired_comparisons.csv", index=False, encoding="utf-8-sig")
    (OUT / "loo_curve_audit.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in curve_audits),
        encoding="utf-8",
    )

    audit = {
        "protocol": {
            "questions": {"HotpotQA": 500, "FanOutQA": 100, "mixed": 600},
            "common_budgets": BUDGETS,
            "selection_metric": "answer_f1",
            "own_question_outcomes_excluded": True,
            "other_target_dataset_labels_allowed": True,
            "interpretation": "pooled supervised leave-one-out evaluation, not zero-shot transfer",
            "global_smoothing_pseudocount": ALPHA,
            "soft_temperature_fixed_a_priori": SOFT_TEMPERATURE,
        },
        "integrity": {
            "mixed_rows": len(mixed),
            "mixed_questions": int(mixed.groupby(["dataset", "sample_id"]).ngroups),
            "expected_rows": 600 * len(BUDGETS),
            "policy_rows": int(sum(len(x) for x in policies)),
            "expected_policy_rows": len(specifications) * 600,
            "passed": bool(
                len(mixed) == 3600
                and sum(len(x) for x in policies) == len(specifications) * 600
            ),
        },
        "best_fixed_answer_f1": best_fixed,
        "summary": summaries.to_dict("records"),
        "comparisons": paired.to_dict("records"),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    plot(summaries)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
