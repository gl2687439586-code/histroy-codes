"""Summarize three greedy fresh runs on identical questions and common budgets."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics

from hp_common import iter_jsonl, write_json

METRICS = {"em": "answer_em", "f1": "answer_f1", "lexical_acc": "answer_acc"}


def total(row):
    return float(row["standalone_llm_input_tokens"]) + float(row["standalone_llm_output_tokens"])


def order(group):
    return 10**9 if group == "FULL" else int(group)


def means(rows):
    rows = list(rows)
    result = {"questions": len(rows), **{name: statistics.mean(float(row[key]) for row in rows) for name, key in METRICS.items()}}
    result.update({"llm_total_tokens_per_question": statistics.mean(total(row) for row in rows),
                   "evidence_tokens_per_question": statistics.mean(float(row["visible_evidence_tokens_total"]) for row in rows),
                   "searches_per_question": statistics.mean(float(row["search_calls"]) for row in rows),
                   "fallback_rate": statistics.mean(float(row.get("local_fallback", False)) for row in rows)})
    return result


def oracle(by_key, ids, conditions, metric):
    selected = {}
    key = METRICS[metric]
    for sid in sorted(ids):
        selected[sid] = min(conditions, key=lambda group: (
            -float(by_key[sid, group][key]), total(by_key[sid, group]),
            float(by_key[sid, group]["visible_evidence_tokens_total"]),
            float(by_key[sid, group]["search_calls"]), order(group)))
    return selected


def analyze(batch, expected=500):
    datasets, rootsets = [], []
    expected_ids = None
    for repetition in range(1, 4):
        run = batch / f"repeat_{repetition}"
        status = json.loads((run / "PIPELINE_STATUS.json").read_text(encoding="utf-8"))
        if status.get("stage") != "complete":
            raise RuntimeError(f"repeat_{repetition} has not completed: {status}")
        wanted = {str(row["sample_id"]) for row in iter_jsonl(run / "dataset/questions.no_labels.jsonl")}
        if len(wanted) != expected or (expected_ids is not None and wanted != expected_ids):
            raise RuntimeError("Question membership differs across repetitions")
        expected_ids = wanted
        by_key = {}
        for row in iter_jsonl(run / "analysis/scored_records.jsonl.gz"):
            key = (str(row["sample_id"]), str(row["condition_group"]))
            if key in by_key or key[0] not in wanted:
                raise RuntimeError(f"Duplicate or unexpected scored key: {key}")
            if row.get("local_fallback") and any(float(row[field]) != 0 for field in METRICS.values()):
                raise RuntimeError(f"Nonzero primary score for fallback: {key}")
            by_key[key] = row
        if {sid for sid, _ in by_key} != wanted or any((sid, "FULL") not in by_key for sid in wanted):
            raise RuntimeError("Missing samples or FULL records")
        roots = {str(row["sample_id"]): row for row in iter_jsonl(run / "roots/roots.merged.jsonl.gz")}
        if set(roots) != wanted:
            raise RuntimeError("Root membership differs from questions")
        datasets.append(by_key)
        rootsets.append(roots)
    common = set.intersection(*(set(group for sid, group in data if sid == min(expected_ids)) for data in datasets))
    common = {group for group in common if all((sid, group) in data for data in datasets for sid in expected_ids)}
    numeric = common - {"FULL"}
    if not numeric:
        raise RuntimeError("No common numeric budgets covering every question in all three runs")
    conditions = sorted(common, key=order)
    summaries, winners, per_run_oracles, transfers = [], {}, [], []
    for repetition, data in enumerate(datasets, 1):
        fixed = {group: means(data[sid, group] for sid in expected_ids) for group in conditions}
        summaries.append({"repetition": repetition, "terminal_records": len(data), "fixed_conditions": fixed})
        for metric in METRICS:
            uniform = min(numeric, key=lambda group: (-fixed[group][metric], fixed[group]["llm_total_tokens_per_question"], order(group)))
            for kind, candidates in (("numeric_only", numeric), ("numeric_plus_FULL", common)):
                selected = oracle(data, expected_ids, candidates, metric)
                winners[repetition, metric, kind] = selected
                selected_stats = means(data[sid, group] for sid, group in selected.items())
                baseline = fixed[uniform]
                per_run_oracles.append({"repetition": repetition, "selection_and_evaluation_metric": metric,
                    "candidates": kind, "best_uniform_numeric_budget": uniform, "oracle": selected_stats,
                    "best_uniform": baseline,
                    "quality_delta_vs_best_uniform": selected_stats[metric] - baseline[metric],
                    "llm_total_tokens_saved_fraction_vs_best_uniform": 1 - selected_stats["llm_total_tokens_per_question"] / baseline["llm_total_tokens_per_question"]})
    aggregate = {}
    for group in conditions:
        aggregate[group] = {}
        for field in ("em", "f1", "lexical_acc", "llm_total_tokens_per_question", "evidence_tokens_per_question", "searches_per_question", "fallback_rate"):
            values = [item["fixed_conditions"][group][field] for item in summaries]
            aggregate[group][field] = {"mean": statistics.mean(values), "sample_std": statistics.stdev(values), "per_repeat": values}
    roots = []
    for source, target in itertools.permutations(range(1, 4), 2):
        source_data, target_data = datasets[source - 1], datasets[target - 1]
        if source < target:
            roots.append({"source_repeat": source, "target_repeat": target, "questions": expected,
                          "same_initial_query": sum(rootsets[source - 1][sid]["query"] == rootsets[target - 1][sid]["query"] for sid in expected_ids),
                          "same_first_evidence_sha256": sum(rootsets[source - 1][sid]["root_full_context_sha256"] == rootsets[target - 1][sid]["root_full_context_sha256"] for sid in expected_ids)})
        for metric in METRICS:
            chosen = winners[source, metric, "numeric_plus_FULL"]
            transferred = means(target_data[sid, group] for sid, group in chosen.items())
            full = means(target_data[sid, "FULL"] for sid in expected_ids)
            target_fixed = summaries[target - 1]["fixed_conditions"]
            uniform = min(numeric, key=lambda group: (-target_fixed[group][metric], target_fixed[group]["llm_total_tokens_per_question"], order(group)))
            baseline = target_fixed[uniform]
            transfers.append({"source_repeat": source, "target_repeat": target, "metric": metric,
                              "selection": "source-run hindsight winner, evaluated on target run",
                              "transferred": transferred, "target_FULL": full,
                              "quality_delta_vs_target_FULL": transferred[metric] - full[metric],
                              "llm_total_tokens_saved_fraction_vs_target_FULL": 1 - transferred["llm_total_tokens_per_question"] / full["llm_total_tokens_per_question"],
                              "target_best_uniform_numeric_budget": uniform,
                              "quality_delta_vs_target_best_uniform": transferred[metric] - baseline[metric],
                              "llm_total_tokens_saved_fraction_vs_target_best_uniform": 1 - transferred["llm_total_tokens_per_question"] / baseline["llm_total_tokens_per_question"]})
    report = {"questions_per_repeat": expected, "repetitions": 3, "common_conditions": conditions,
              "condition_comparability": "Every reported condition covers the same 500 questions in all three repetitions",
              "cost_definition": "Per-question standalone input+output, including that repetition's root; no second addition of evidence",
              "repetition_note": "Fresh greedy executions; no best-answer selection across runs, no random-sample independence assumed",
              "by_repeat": summaries, "mean_and_sample_std": aggregate,
              "oracle_vs_best_uniform": per_run_oracles, "crossrun_oracle_transfer": transfers,
              "root_pair_consistency": roots}
    output = batch / "analysis_repeats"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "SUMMARY.json", report)
    lines = ["# HotpotQA同题500道：三次独立全流程复现", "",
             "三次均temperature=0，分别重新生成root和后续轨迹；不从三次中挑最好答案。",
             "每题平均LLM总tokens=所选策略的root+continuation输入与输出；实验枚举成本另列于每次token_budget目录。",
             "词法ACC为normalized contains，沿用原协议；不是语义Judge。", "",
             "共同候选：" + ", ".join(conditions), "",
             "| 条件 | EM均值±样本标准差 | F1均值±标准差 | 词法ACC均值±标准差 | 总LLM tokens/题均值±标准差 |",
             "|---|---:|---:|---:|---:|"]
    for group in conditions:
        values = aggregate[group]
        lines.append(f"| {group} | {values['em']['mean']:.4f} ± {values['em']['sample_std']:.4f} | {values['f1']['mean']:.4f} ± {values['f1']['sample_std']:.4f} | {values['lexical_acc']['mean']:.4f} ± {values['lexical_acc']['sample_std']:.4f} | {values['llm_total_tokens_per_question']['mean']:.1f} ± {values['llm_total_tokens_per_question']['sample_std']:.1f} |")
    lines += ["", "## 同指标Oracle相对最优统一数值预算", "",
              "候选含FULL的逐题Oracle看过gold，属于上界；统一数值预算也是本样本事后最优参考。质量差和成本节省分列。", "",
              "| 重复 | 选择和评估指标 | 最优统一B | Oracle质量 | 统一预算质量 | 质量差 | LLM总tokens节省 |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for row in per_run_oracles:
        if row["candidates"] != "numeric_plus_FULL":
            continue
        metric = row["selection_and_evaluation_metric"]
        lines.append(f"| {row['repetition']} | {metric} | {row['best_uniform_numeric_budget']} | {row['oracle'][metric]:.4f} | {row['best_uniform'][metric]:.4f} | {row['quality_delta_vs_best_uniform']:+.4f} | {row['llm_total_tokens_saved_fraction_vs_best_uniform']:.2%} |")
    lines += ["", "SUMMARY.json包含每次原始均值、六个方向的跨轮Oracle预算迁移，以及三对root一致性。",
              "平均值和标准差仅描述这三次执行；500题在三次中重复出现，不能当1500个独立样本。"]
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "questions_per_repeat": expected, "common_conditions": conditions}))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.batch_dir)


if __name__ == "__main__":
    main()
