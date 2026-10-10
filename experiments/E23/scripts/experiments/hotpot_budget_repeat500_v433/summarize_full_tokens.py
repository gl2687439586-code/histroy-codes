#!/usr/bin/env python3
"""Offline per-question token accounting, without model or retriever calls."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from hp_common import iter_jsonl, utc_stamp, write_json

FIELDS = {
    "em": "answer_em", "f1": "answer_f1", "lexical_acc": "answer_acc",
    "search_calls": "search_calls", "evidence_tokens": "visible_evidence_tokens_total",
    "llm_input_tokens": "standalone_llm_input_tokens",
    "llm_output_tokens": "standalone_llm_output_tokens",
    "thinking_tokens": "standalone_thinking_tokens", "model_calls": "standalone_model_calls",
    "fallback_rate": "local_fallback", "context_compactions": "context_compactions",
}


def aggregate(rows, expected_questions):
    buckets = defaultdict(lambda: {"ids": set(), "sums": Counter(), "n": 0})
    ids, keys = set(), set()
    continuation = Counter()
    total_records = 0
    for row in rows:
        sid = str(row["sample_id"])
        group = str(row["condition_group"])
        key = (sid, str(row["condition_id"]))
        if key in keys:
            raise RuntimeError(f"Duplicate scored key: {key}")
        keys.add(key)
        ids.add(sid)
        total_records += 1
        if row.get("local_fallback") and any(float(row[x]) != 0 for x in ("answer_em", "answer_f1", "answer_acc")):
            raise RuntimeError(f"Fallback has nonzero primary quality score: {key}")
        for dimension in (("budget", group), ("budget_search", group, int(row["search_calls"])),
                          ("budget_em", group, int(float(row["answer_em"])) )):
            bucket = buckets[dimension]
            bucket["n"] += 1
            bucket["ids"].add(sid)
            for name, source in FIELDS.items():
                bucket["sums"][name] += float(row.get(source, 0))
            bucket["sums"]["llm_total_tokens"] += float(row["standalone_llm_input_tokens"]) + float(row["standalone_llm_output_tokens"])
        continuation["input_tokens"] += int(row["continuation_llm_input_tokens"])
        continuation["output_tokens"] += int(row["continuation_llm_output_tokens"])
    if len(ids) != expected_questions:
        raise RuntimeError(f"Expected {expected_questions} questions, found {len(ids)}")
    outputs = defaultdict(list)
    for key, bucket in sorted(buckets.items(), key=lambda x: (x[0][0], 10**9 if x[0][1] == "FULL" else int(x[0][1]), x[0][2:])):
        n = bucket["n"]
        if n != len(bucket["ids"]):
            raise RuntimeError(f"Duplicate question inside group {key}")
        result = {"condition": key[1], "questions": n, "coverage": n / expected_questions,
                  "contains_all_questions": n == expected_questions}
        if key[0] == "budget_search":
            result["search_rounds"] = key[2]
        elif key[0] == "budget_em":
            result["em_correct"] = key[2]
        result.update({"mean_" + name + "_per_question": value / n for name, value in bucket["sums"].items()})
        outputs[key[0]].append(result)
    return dict(outputs), {"questions": len(ids), "records": total_records, "continuation_successful_recorded_tokens": dict(continuation)}


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def frontiers(rows):
    comparable = [x for x in rows if x["contains_all_questions"]]
    output = []
    for cost in ("mean_evidence_tokens_per_question", "mean_llm_total_tokens_per_question"):
        for row in comparable:
            dominated = any(
                other[cost] <= row[cost]
                and other["mean_f1_per_question"] >= row["mean_f1_per_question"]
                and (other[cost] < row[cost] or other["mean_f1_per_question"] > row["mean_f1_per_question"])
                for other in comparable
            )
            output.append({"cost_axis": cost, "on_frontier": not dominated, **row})
    return output


def oracle_accounting(scored_path, common_conditions):
    definitions = {
        "f1_then_input_common_numeric": (common_conditions - {"FULL"}, "standalone_llm_input_tokens"),
        "f1_then_total_common_numeric_plus_FULL": (common_conditions, "total"),
    }
    winners = {name: {} for name, (groups, _) in definitions.items() if groups}
    # Keep only a small selected row per question, not raw prompts/traces.
    for row in iter_jsonl(scored_path):
        sid = str(row["sample_id"])
        group = str(row["condition_group"])
        for name in winners:
            groups, cost_name = definitions[name]
            if group not in groups:
                continue
            total = int(row["standalone_llm_input_tokens"]) + int(row["standalone_llm_output_tokens"])
            cost = total if cost_name == "total" else int(row[cost_name])
            key = (float(row["answer_f1"]), -cost, -int(row["visible_evidence_tokens_total"]), -int(row["search_calls"]))
            previous = winners[name].get(sid)
            if previous is None or key > previous[0]:
                compact = {"sample_id": sid, "selected_condition": group,
                           "em": row["answer_em"], "f1": row["answer_f1"], "lexical_acc": row["answer_acc"],
                           "search_calls": row["search_calls"], "evidence_tokens": row["visible_evidence_tokens_total"],
                           "llm_input_tokens": row["standalone_llm_input_tokens"],
                           "llm_output_tokens": row["standalone_llm_output_tokens"], "llm_total_tokens": total}
                winners[name][sid] = (key, compact)
    summaries, selected = [], []
    for name, by_id in winners.items():
        chosen = [item[1] for _, item in sorted(by_id.items())]
        selected.extend({"oracle": name, **x} for x in chosen)
        n = len(chosen)
        stats = {"oracle": name, "questions": n,
                 "candidate_conditions": ",".join(sorted(definitions[name][0]))}
        for field in ("em", "f1", "lexical_acc", "search_calls", "evidence_tokens", "llm_input_tokens", "llm_output_tokens", "llm_total_tokens"):
            stats["mean_" + field + "_per_question"] = sum(float(x[field]) for x in chosen) / n
        summaries.append(stats)
    return summaries, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-questions", type=int, default=7405)
    args = parser.parse_args()
    run = args.run_dir
    tables, summary = aggregate(iter_jsonl(run / "analysis/scored_records.jsonl.gz"), args.expected_questions)
    root_usage = Counter()
    root_ids = set()
    for root in iter_jsonl(run / "roots/roots.merged.jsonl.gz"):
        sid = root["sample_id"]
        if sid in root_ids:
            raise RuntimeError(f"Duplicate root: {sid}")
        root_ids.add(sid)
        for name in ("llm_input_tokens", "llm_output_tokens", "model_calls"):
            root_usage[name] += int(root.get("usage", {}).get(name, 0))
    if len(root_ids) != args.expected_questions:
        raise RuntimeError("Root count differs from full question count")
    summary.update({"created_at": utc_stamp(), "root_successful_recorded_usage": dict(root_usage),
                    "cost_definition": "Each policy row includes root plus continuation. Input+output excludes any double addition of evidence tokens.",
                    "physical_accounting_scope": "Successfully recorded root calls once plus continuations. Unrecorded HTTP failures and service probes are not included."})
    continuation = summary["continuation_successful_recorded_tokens"]
    summary["physical_successful_recorded_input_output_tokens"] = root_usage["llm_input_tokens"] + root_usage["llm_output_tokens"] + continuation["input_tokens"] + continuation["output_tokens"]
    output = run / "analysis/token_budget"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output / f"by_{name}.csv", rows)
    write_csv(output / "pareto_by_cost_axis.csv", frontiers(tables["budget"]))
    common = {x["condition"] for x in tables["budget"] if x["contains_all_questions"]}
    oracle_means, oracle_rows = oracle_accounting(run / "analysis/scored_records.jsonl.gz", common)
    write_csv(output / "oracle_mean_tokens.csv", oracle_means)
    write_csv(output / "oracle_per_question.csv", oracle_rows)
    summary["common_conditions"] = sorted(common)
    summary["oracle_mean_tokens"] = oracle_means
    write_json(output / "summary.json", summary)
    lines = ["# 每题平均 token budget 与质量", "",
             f"题目：{summary['questions']}；轨迹：{summary['records']}。", "",
             "每行是一种预算条件下的每题平均值；B是每轮证据上限。LLM总tokens=输入+输出，证据tokens已包含在输入中。",
             "输入含重复送入上下文的历史证据；累计暴露证据tokens则按每次搜索的暴露量求和，并非跨轮去重后的证据量。", "",
             "| 条件 | 题数 | 覆盖率 | EM | F1 | 词法Acc | 搜索/题 | 证据/题 | 输入/题 | 输出/题 | 总LLM/题 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in tables["budget"]:
        metrics = [row["mean_" + name + "_per_question"] for name in ("em", "f1", "lexical_acc", "search_calls", "evidence_tokens", "llm_input_tokens", "llm_output_tokens", "llm_total_tokens")]
        lines.append(f"| {row['condition']} | {row['questions']} | {row['coverage']:.1%} | " + " | ".join(f"{x:.3f}" if j < 3 else f"{x:.1f}" for j, x in enumerate(metrics)) + " |")
    lines += ["", "各预算条件覆盖的题目集合可能不同。公平横向比较只使用覆盖率100%的条件，或另行构造同题配对集合。",
              "FULL按每轮真实检索长度暴露，不能视为统一固定token预算。词法Acc不是LLM Judge。",
              "by_budget_search.csv联合统计预算与实际搜索轮数；by_budget_em.csv按EM正确与错误分层。",
              "上述两种分层是结果后的描述性统计，不能据此认定搜索次数/长推理造成正确或错误。",
              "全量的系统开销与选定策略每题的独立执行开销分开统计：跨预算共享root节省实验计算，不降低独立部署该策略的root成本。",
              "", "## 后验Oracle与成本前沿", "",
              "pareto_by_cost_axis.csv分别以每题实际证据tokens、LLM输入+输出tokens计算前沿，仅比较题集完整的条件。",
              "oracle_mean_tokens.csv同时报告每题最高F1的两种选择：公共数值条件内以输入成本破平局，以及公共数值条件加FULL内以总LLM成本破平局。",
              "Oracle是看过真实答案后选出来的上界，不能将其质量和所选路径成本宣称为已部署控制器的表现。"]
    (output / "REPORT_ZH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
