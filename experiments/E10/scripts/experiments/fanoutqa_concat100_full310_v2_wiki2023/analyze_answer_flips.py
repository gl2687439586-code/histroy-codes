#!/usr/bin/env python3
"""Paired answer-flip audit over an already scored evidence-budget grid.

The unit of resampling is the FanOutQA question.  No model or retriever is
called.  Local fallbacks retain their explicit format-status marker, while the
stored recovered final answer is evaluated using the same answer metric as all
other outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from hp_common import file_sha256, iter_jsonl, utc_stamp, write_json


METRICS = ("official_strict_accuracy",)
METRIC_LABELS = {
    "official_strict_accuracy": "FanOutQA 官方 Strict Accuracy（二元完全覆盖）",
}


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    location = (len(ordered) - 1) * probability
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def descriptive(values: Sequence[float], prefix: str) -> Dict[str, Any]:
    clean = [float(value) for value in values]
    if not clean:
        return {
            f"{prefix}_n": 0, f"{prefix}_mean": None,
            f"{prefix}_median": None, f"{prefix}_p25": None,
            f"{prefix}_p75": None,
        }
    return {
        f"{prefix}_n": len(clean), f"{prefix}_mean": mean(clean),
        f"{prefix}_median": median(clean),
        f"{prefix}_p25": quantile(clean, 0.25),
        f"{prefix}_p75": quantile(clean, 0.75),
    }


def logsumexp(log_values: Sequence[float]) -> float:
    maximum = max(log_values)
    return maximum + math.log(sum(math.exp(value - maximum) for value in log_values))


def mcnemar_exact_p(wrong_to_correct: int, correct_to_wrong: int) -> float:
    """Two-sided exact McNemar p-value (binomial test with p=0.5)."""
    discordant = int(wrong_to_correct) + int(correct_to_wrong)
    if discordant == 0:
        return 1.0
    lower = min(int(wrong_to_correct), int(correct_to_wrong))
    log_probabilities = [
        math.lgamma(discordant + 1) - math.lgamma(k + 1)
        - math.lgamma(discordant - k + 1) - discordant * math.log(2.0)
        for k in range(lower + 1)
    ]
    return min(1.0, 2.0 * math.exp(logsumexp(log_probabilities)))


def mcnemar_continuity_p(wrong_to_correct: int, correct_to_wrong: int) -> float:
    discordant = int(wrong_to_correct) + int(correct_to_wrong)
    if discordant == 0:
        return 1.0
    chi_square = (max(0.0, abs(wrong_to_correct - correct_to_wrong) - 1.0) ** 2) / discordant
    # A chi-square random variable with one degree of freedom has this survival function.
    return math.erfc(math.sqrt(chi_square / 2.0))


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    count = len(p_values)
    if not count:
        return []
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running = 1.0
    for reverse_rank in range(count - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, float(p_values[index]) * count / rank)
        adjusted[index] = min(1.0, running)
    return adjusted


def bootstrap_transition(
    counts: Sequence[int], repetitions: int, rng: np.random.Generator,
) -> Dict[str, float]:
    total = int(sum(counts))
    if total <= 0:
        raise ValueError("Cannot bootstrap an empty paired transition")
    probabilities = np.asarray(counts, dtype=np.float64) / total
    draws = rng.multinomial(total, probabilities, size=int(repetitions))
    # Category order: WW, WC, CW, CC.
    gain = draws[:, 1] / total
    harm = draws[:, 2] / total
    net = gain - harm
    low, high = 2.5, 97.5
    return {
        "gain_ci_low": float(np.percentile(gain, low)),
        "gain_ci_high": float(np.percentile(gain, high)),
        "harm_ci_low": float(np.percentile(harm, low)),
        "harm_ci_high": float(np.percentile(harm, high)),
        "net_gain_ci_low": float(np.percentile(net, low)),
        "net_gain_ci_high": float(np.percentile(net, high)),
    }


def bootstrap_binary_mean(
    successes: int, total: int, repetitions: int, rng: np.random.Generator,
) -> Tuple[float, float]:
    if total <= 0:
        return float("nan"), float("nan")
    draws = rng.binomial(total, successes / total, size=int(repetitions)) / total
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def observed_stable_budget(budgets: Sequence[int], correctness: Sequence[int]) -> int | None:
    """Finite-grid SSB: first budget whose entire observed suffix is correct."""
    for index, budget in enumerate(budgets):
        if all(correctness[index:]):
            return int(budget)
    return None


def strict_value(row: Mapping[str, Any], field: str) -> float:
    return float(row.get(field, 0.0))


def load_complete_grid(
    scored_records: str | Path,
) -> Tuple[Dict[str, Dict[int, Dict[str, Any]]], List[int]]:
    grid: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in iter_jsonl(scored_records):
        # The exact FULL endpoint has a different token count for every root
        # and remains in the primary analysis.  Paired adjacent transitions
        # use only the 100-token caps shared by every sampled question.
        if row.get("exposure_mode") != "token_cap":
            continue
        sample_id = str(row["sample_id"])
        budget = int(row["budget"])
        if budget in grid[sample_id]:
            raise ValueError(f"Duplicate scored key: {sample_id}/b{budget:04d}")
        grid[sample_id][budget] = row
    if not grid:
        raise ValueError("No scored records were loaded")
    common = set.intersection(*(set(rows) for rows in grid.values()))
    budgets = sorted(common)
    if len(budgets) < 2:
        raise ValueError(f"Fewer than two common 100-token caps: {budgets}")
    if any(right - left != 100 for left, right in zip(budgets, budgets[1:])):
        raise ValueError(f"Common budget grid is not adjacent by 100 tokens: {budgets}")
    trimmed = {
        sample_id: {budget: rows[budget] for budget in budgets}
        for sample_id, rows in grid.items()
    }
    return trimmed, budgets


def metric_audit(
    grid: Mapping[str, Mapping[int, Mapping[str, Any]]], budgets: Sequence[int],
    metric: str, bootstrap_repetitions: int, seed: int,
) -> Dict[str, Any]:
    sample_ids = sorted(grid)
    n = len(sample_ids)
    rng = np.random.default_rng(seed)
    correctness: Dict[str, List[int]] = {}
    f1_values: Dict[str, List[float]] = {}
    for sample_id in sample_ids:
        correctness[sample_id] = [
            int(strict_value(grid[sample_id][budget], metric) >= 0.5)
            for budget in budgets
        ]
        f1_values[sample_id] = [
            strict_value(grid[sample_id][budget], "official_loose_accuracy")
            for budget in budgets
        ]

    transitions: List[Dict[str, Any]] = []
    for index, (left, right) in enumerate(zip(budgets, budgets[1:])):
        pairs = [(correctness[sample_id][index], correctness[sample_id][index + 1])
                 for sample_id in sample_ids]
        counts = Counter(pairs)
        ww = counts[(0, 0)]
        wc = counts[(0, 1)]
        cw = counts[(1, 0)]
        cc = counts[(1, 1)]
        row = {
            "metric": metric,
            "budget_from": int(left), "budget_to": int(right), "n_pairs": n,
            "wrong_to_wrong_count": ww,
            "wrong_to_correct_count": wc,
            "correct_to_wrong_count": cw,
            "correct_to_correct_count": cc,
            "wrong_to_wrong_rate": ww / n,
            "gain": wc / n,
            "harm": cw / n,
            "correct_to_correct_rate": cc / n,
            "net_gain": (wc - cw) / n,
            "accuracy_from": (cw + cc) / n,
            "accuracy_to": (wc + cc) / n,
            "discordant_pairs": wc + cw,
            "mcnemar_exact_p": mcnemar_exact_p(wc, cw),
            "mcnemar_continuity_corrected_p": mcnemar_continuity_p(wc, cw),
        }
        row.update(bootstrap_transition([ww, wc, cw, cc], bootstrap_repetitions, rng))
        transitions.append(row)
    adjusted = benjamini_hochberg([row["mcnemar_exact_p"] for row in transitions])
    for row, q_value in zip(transitions, adjusted):
        row["mcnemar_exact_bh_q"] = q_value
        row["mcnemar_significant_bh_0_05"] = q_value < 0.05

    per_sample: List[Dict[str, Any]] = []
    oracle_correct = {}
    oracle_f1 = {}
    for sample_id in sample_ids:
        values = correctness[sample_id]
        f1s = f1_values[sample_id]
        msb = next((int(budget) for budget, value in zip(budgets, values) if value), None)
        ssb = observed_stable_budget(budgets, values)
        positive = sum(left == 0 and right == 1 for left, right in zip(values, values[1:]))
        negative = sum(left == 1 and right == 0 for left, right in zip(values, values[1:]))
        oracle_correct[sample_id] = max(values)
        oracle_f1[sample_id] = max(f1s)
        best_f1_index = min(
            (index for index, value in enumerate(f1s) if value == oracle_f1[sample_id]),
            default=0,
        )
        per_sample.append({
            "sample_id": sample_id, "metric": metric,
            "correctness_pattern": "".join(str(value) for value in values),
            "correct_budget_count": sum(values),
            "ever_correct": bool(max(values)),
            "correct_at_max_budget": bool(values[-1]),
            "minimum_success_budget_msb": msb,
            "stable_sufficient_budget_ssb_observed": ssb,
            "ssb_minus_msb": (ssb - msb) if msb is not None and ssb is not None else None,
            "answer_flip_count": positive + negative,
            "wrong_to_correct_flip_count": positive,
            "correct_to_wrong_flip_count": negative,
            "unstable_after_first_correct": bool(
                msb is not None and any(
                    not value for budget, value in zip(budgets, values) if budget > msb
                )
            ),
            "oracle_binary_correct": oracle_correct[sample_id],
            "oracle_best_f1": oracle_f1[sample_id],
            "oracle_min_budget_at_best_f1": int(budgets[best_f1_index]),
        })

    oracle_accuracy = mean(oracle_correct.values())
    oracle_mean_f1 = mean(oracle_f1.values())
    regret_rows: List[Dict[str, Any]] = []
    for index, budget in enumerate(budgets):
        current = [correctness[sample_id][index] for sample_id in sample_ids]
        avoidable = [oracle_correct[sample_id] - value
                     for sample_id, value in zip(sample_ids, current)]
        f1_at_budget = [f1_values[sample_id][index] for sample_id in sample_ids]
        f1_regrets = [oracle_f1[sample_id] - value
                      for sample_id, value in zip(sample_ids, f1_at_budget)]
        avoidable_count = int(sum(avoidable))
        low, high = bootstrap_binary_mean(
            avoidable_count, n, bootstrap_repetitions, rng
        )
        regret_rows.append({
            "metric": metric, "budget": int(budget), "n": n,
            "fixed_budget_accuracy": mean(current),
            "per_question_oracle_accuracy": oracle_accuracy,
            "binary_oracle_regret": mean(avoidable),
            "binary_oracle_regret_ci_low": low,
            "binary_oracle_regret_ci_high": high,
            "avoidable_wrong_count": avoidable_count,
            "fixed_budget_mean_f1_strict": mean(f1_at_budget),
            "per_question_oracle_mean_f1": oracle_mean_f1,
            "f1_oracle_regret": mean(f1_regrets),
        })

    msbs = [row["minimum_success_budget_msb"] for row in per_sample
            if row["minimum_success_budget_msb"] is not None]
    ssbs = [row["stable_sufficient_budget_ssb_observed"] for row in per_sample
            if row["stable_sufficient_budget_ssb_observed"] is not None]
    gaps = [row["ssb_minus_msb"] for row in per_sample if row["ssb_minus_msb"] is not None]
    ever_correct_count = sum(row["ever_correct"] for row in per_sample)
    summary = {
        "metric": metric, "metric_label": METRIC_LABELS[metric],
        "questions": n, "budgets": list(map(int, budgets)),
        "oracle_binary_accuracy": oracle_accuracy,
        "never_correct_count": sum(not row["ever_correct"] for row in per_sample),
        "never_correct_rate": sum(not row["ever_correct"] for row in per_sample) / n,
        "always_correct_count": sum(row["correct_budget_count"] == len(budgets) for row in per_sample),
        "always_correct_rate": sum(row["correct_budget_count"] == len(budgets) for row in per_sample) / n,
        "any_answer_flip_count": sum(row["answer_flip_count"] > 0 for row in per_sample),
        "any_answer_flip_rate": sum(row["answer_flip_count"] > 0 for row in per_sample) / n,
        "any_harmful_flip_count": sum(row["correct_to_wrong_flip_count"] > 0 for row in per_sample),
        "any_harmful_flip_rate": sum(row["correct_to_wrong_flip_count"] > 0 for row in per_sample) / n,
        "unstable_after_first_correct_count": sum(row["unstable_after_first_correct"] for row in per_sample),
        "unstable_after_first_correct_rate_among_ever_correct": (
            sum(row["unstable_after_first_correct"] for row in per_sample) / ever_correct_count
            if ever_correct_count else 0.0
        ),
        "mean_answer_flips_per_question": mean(row["answer_flip_count"] for row in per_sample),
        "mean_wrong_to_correct_flips_per_question": mean(
            row["wrong_to_correct_flip_count"] for row in per_sample
        ),
        "mean_correct_to_wrong_flips_per_question": mean(
            row["correct_to_wrong_flip_count"] for row in per_sample
        ),
        **descriptive(msbs, "msb"),
        **descriptive(ssbs, "ssb_observed"),
        **descriptive(gaps, "ssb_minus_msb"),
        "most_positive_net_gain_transition": max(
            transitions, key=lambda row: (row["net_gain"], -row["budget_from"])
        )["budget_from"] if transitions else None,
        "most_negative_net_gain_transition": min(
            transitions, key=lambda row: (row["net_gain"], row["budget_from"])
        )["budget_from"] if transitions else None,
        "mcnemar_bh_significant_transition_count": sum(
            row["mcnemar_significant_bh_0_05"] for row in transitions
        ),
    }
    return {
        "summary": summary, "transitions": transitions,
        "per_sample": per_sample, "regret": regret_rows,
    }


def run_audit(
    scored_records: str | Path, output_dir: str | Path,
    metrics: Sequence[str] = METRICS,
    primary_metric: str = "official_strict_accuracy",
    bootstrap_repetitions: int = 10000, seed: int = 2026,
) -> Dict[str, Any]:
    for metric in metrics:
        if metric not in METRICS:
            raise ValueError(f"Unsupported correctness metric: {metric}")
    if primary_metric not in metrics:
        raise ValueError("primary_metric must be included in metrics")
    if bootstrap_repetitions < 100:
        raise ValueError("At least 100 bootstrap repetitions are required")
    grid, budgets = load_complete_grid(scored_records)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = {}
    for offset, metric in enumerate(metrics):
        result = metric_audit(
            grid, budgets, metric, bootstrap_repetitions, seed + offset * 100003
        )
        results[metric] = result
        write_csv(output / f"adjacent_transitions_{metric}.csv", result["transitions"])
        write_csv(output / f"per_question_msb_ssb_{metric}.csv", result["per_sample"])
        write_csv(output / f"fixed_budget_oracle_regret_{metric}.csv", result["regret"])

    primary = results[primary_metric]
    summary = {
        "created_at": utc_stamp(),
        "scored_records": str(Path(scored_records).resolve()),
        "scored_records_sha256": file_sha256(scored_records),
        "primary_metric": primary_metric,
        "correctness_rule": "official_strict_accuracy==1; local_fallback is reported separately",
        "paired_unit": "question",
        "bootstrap_method": "paired nonparametric question bootstrap (multinomial-equivalent)",
        "bootstrap_repetitions": int(bootstrap_repetitions),
        "bootstrap_seed": int(seed),
        "confidence_level": 0.95,
        "mcnemar_method": "two-sided exact binomial; BH-FDR across adjacent-budget tests",
        "ssb_scope_warning": (
            "SSB is stable only on the finite observed budget grid and does not imply "
            "stability beyond the maximum evaluated budget"
        ),
        "questions": len(grid), "budgets": budgets,
        "metric_summaries": {metric: result["summary"] for metric, result in results.items()},
    }
    write_json(output / "summary.json", summary)

    transition_lines = "\n".join(
        f"| {row['budget_from']}→{row['budget_to']} | "
        f"{row['wrong_to_correct_count']} ({row['gain']:.3f}) | "
        f"{row['correct_to_wrong_count']} ({row['harm']:.3f}) | "
        f"{row['net_gain']:.3f} [{row['net_gain_ci_low']:.3f}, {row['net_gain_ci_high']:.3f}] | "
        f"{row['mcnemar_exact_p']:.4g} | {row['mcnemar_exact_bh_q']:.4g} |"
        for row in primary["transitions"]
    )
    main = primary["summary"]
    report = f"""# FanOutQA 相邻证据预算答案翻转审计

主正确性口径为 **{METRIC_LABELS[primary_metric]}**。`local_fallback` 保留为独立格式状态，
其恢复的 `final_answer` 使用相同答案指标正常评分。配对、bootstrap 与 McNemar 的统计单位
均为问题，而不是把同一道题的不同预算轨迹当成独立样本。

## 核心稳定性

- 问题数：{main['questions']}；所有题共有的预算网格：{budgets[0]}–{budgets[-1]}，步长100。
- 每题长度不同的 `FULL` 终点由主分析单独报告，不混入矩形配对检验。
- 至少一次答对（逐题二元 Oracle）：{main['oracle_binary_accuracy']:.2%}。
- 从未答对：{main['never_correct_rate']:.2%}。
- 至少发生一次答案翻转：{main['any_answer_flip_rate']:.2%}。
- 至少发生一次 Correct→Wrong：{main['any_harmful_flip_rate']:.2%}。
- 在曾答对的问题中，首次答对后再次出错：{main['unstable_after_first_correct_rate_among_ever_correct']:.2%}。
- 平均每题翻转次数：{main['mean_answer_flips_per_question']:.3f}。
- MSB 中位数：{main['msb_median']}；有限观测网格 SSB 中位数：{main['ssb_observed_median']}。
- SSB−MSB 中位数：{main['ssb_minus_msb_median']} tokens。

## 相邻预算翻转

| 预算 | Wrong→Correct（Gain） | Correct→Wrong（Harm） | NetGain及95%配对bootstrap CI | McNemar p | BH-FDR q |
|---:|---:|---:|---:|---:|---:|
{transition_lines}

## 定义与解释限制

- `MSB_i`：该题在观测网格上第一次答对的最小预算。
- `SSB_i`：从该预算到最大公共观测预算 {budgets[-1]} 的所有后续已观测预算均答对的最小预算。
- `SSB−MSB` 衡量首次成功后被额外证据破坏、再恢复所需的预算跨度。
- 固定预算 regret = 逐题 Oracle 是否曾答对 − 该固定预算是否答对；另附连续 F1 regret。
- SSB 只表示所有样本共有的 **100-token 有限网格内** 的经验稳定，不外推到更大预算或每题 `FULL` 终点。
- {len(primary['transitions'])} 个相邻比较同时报告原始精确 McNemar p 和 Benjamini–Hochberg q。

连续的 Loose Accuracy 与 ROUGE 由主分析报告；McNemar 只用于二元 Strict Accuracy。
"""
    (output / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scored_records", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--metrics", default=",".join(METRICS))
    parser.add_argument(
        "--primary_metric", default="official_strict_accuracy", choices=METRICS
    )
    parser.add_argument("--bootstrap_repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    summary = run_audit(
        args.scored_records, args.output_dir, metrics=metrics,
        primary_metric=args.primary_metric,
        bootstrap_repetitions=args.bootstrap_repetitions, seed=args.seed,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
