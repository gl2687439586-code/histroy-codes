#!/usr/bin/env python3
"""Offline answer and annotated-evidence evaluation for HotpotQA trajectories."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set

from hp_common import (
    BUDGETS, answer_metrics, canonical_hash, file_sha256, iter_jsonl,
    normalize_text, read_jsonl, utc_stamp, write_json,
)


def title_set(items: Iterable[Dict[str, Any]]) -> Set[str]:
    return {normalize_text(item.get("title")) for item in items if normalize_text(item.get("title"))}


def sentence_hits(facts: List[Dict[str, Any]], visible_texts: List[str]):
    normalized_passages = [normalize_text(text) for text in visible_texts]
    passage_token_sets = [set(text.split()) for text in normalized_passages]
    strict = []
    fuzzy = []
    fuzzy_scores = []
    for fact in facts:
        sentence = normalize_text(fact.get("sentence"))
        if not sentence:
            strict.append(False); fuzzy.append(False); fuzzy_scores.append(0.0)
            continue
        strict_hit = any(sentence in passage for passage in normalized_passages)
        tokens = sentence.split()
        score = max((len(set(tokens) & passage) / len(set(tokens))
                     for passage in passage_token_sets), default=0.0)
        strict.append(strict_hit)
        fuzzy.append(strict_hit or (len(tokens) >= 5 and score >= 0.8))
        fuzzy_scores.append(score)
    return strict, fuzzy, fuzzy_scores


def score_record(row: Dict[str, Any], label: Dict[str, Any]) -> Dict[str, Any]:
    answer = answer_metrics(row.get("final_answer", ""), label.get("gold_answers") or [])
    facts = label.get("supporting_facts") or []
    gold_titles = {normalize_text(item) for item in label.get("supporting_titles") or []}
    rounds = row.get("search_rounds") or []
    all_retrieved = [doc for rnd in rounds for doc in rnd["evidence"]["retrieved_docs"]]
    all_visible = [doc for rnd in rounds for doc in rnd["evidence"]["visible_passages"]]
    root_retrieved = rounds[0]["evidence"]["retrieved_docs"] if rounds else []
    root_visible = rounds[0]["evidence"]["visible_passages"] if rounds else []
    retrieved_titles = title_set(all_retrieved)
    visible_titles = title_set(all_visible)
    root_retrieved_titles = title_set(root_retrieved)
    root_visible_titles = title_set(root_visible)
    visible_texts = [item.get("visible_text", "") for item in all_visible]
    root_visible_texts = [item.get("visible_text", "") for item in root_visible]
    strict, fuzzy, fuzzy_scores = sentence_hits(facts, visible_texts)
    root_strict, root_fuzzy, _ = sentence_hits(facts, root_visible_texts)
    title_denominator = max(1, len(gold_titles))
    fact_denominator = max(1, len(facts))
    round_metrics = []
    cumulative_title_hits: Set[str] = set()
    cumulative_strict_hits: Set[int] = set()
    cumulative_fuzzy_hits: Set[int] = set()
    for round_item in rounds:
        round_retrieved = title_set(round_item["evidence"]["retrieved_docs"])
        round_visible_items = round_item["evidence"]["visible_passages"]
        round_visible_titles = title_set(round_visible_items)
        round_strict, round_fuzzy, _ = sentence_hits(
            facts, [item.get("visible_text", "") for item in round_visible_items]
        )
        title_hits = gold_titles & round_retrieved
        strict_hits = {index for index, hit in enumerate(round_strict) if hit}
        fuzzy_hits = {index for index, hit in enumerate(round_fuzzy) if hit}
        new_title_hits = title_hits - cumulative_title_hits
        new_strict_hits = strict_hits - cumulative_strict_hits
        new_fuzzy_hits = fuzzy_hits - cumulative_fuzzy_hits
        cumulative_title_hits |= title_hits
        cumulative_strict_hits |= strict_hits
        cumulative_fuzzy_hits |= fuzzy_hits
        round_metrics.append({
            "search_round": int(round_item["search_round"]),
            "query": round_item["query"],
            "retrieved_support_title_recall_this_round": len(title_hits) / title_denominator,
            "new_retrieved_support_title_recall": len(new_title_hits) / title_denominator,
            "cumulative_retrieved_support_title_recall": len(cumulative_title_hits) / title_denominator,
            "visible_support_title_recall_this_round": len(gold_titles & round_visible_titles) / title_denominator,
            "strict_support_sentence_recall_this_round": len(strict_hits) / fact_denominator,
            "new_strict_support_sentence_recall": len(new_strict_hits) / fact_denominator,
            "cumulative_strict_support_sentence_recall": len(cumulative_strict_hits) / fact_denominator,
            "new_fuzzy_support_sentence_recall": len(new_fuzzy_hits) / fact_denominator,
            "cumulative_fuzzy_support_sentence_recall": len(cumulative_fuzzy_hits) / fact_denominator,
        })
    answer_norms = [normalize_text(item) for item in label.get("gold_answers") or []]
    evidence_norm = " ".join(normalize_text(text) for text in visible_texts)
    return {
        "sample_id": row["sample_id"],
        "condition_id": row["condition_id"],
        "budget": int(row["evidence_budget_per_search"]),
        "type": label.get("type", row.get("type")),
        "level": label.get("level", row.get("level")),
        "answer_em": answer["em"], "answer_f1": answer["f1"],
        "answer_acc": answer["acc"],
        "search_calls": int(row["search_calls"]),
        "multiple_search": int(row["search_calls"]) > 1,
        "visible_evidence_tokens_total": int(row["visible_evidence_tokens_total"]),
        "standalone_model_calls": int(row["usage"]["standalone_model_calls"]),
        "standalone_llm_input_tokens": int(row["usage"]["standalone_llm_input_tokens"]),
        "standalone_llm_output_tokens": int(row["usage"]["standalone_llm_output_tokens"]),
        "standalone_thinking_tokens": int(row["usage"].get("standalone_thinking_tokens", 0)),
        "continuation_llm_input_tokens": int(row["usage"]["continuation_llm_input_tokens"]),
        "continuation_llm_output_tokens": int(row["usage"]["continuation_llm_output_tokens"]),
        "format_repair_calls": int(row["usage"]["standalone_format_repair_calls"]),
        "local_fallback": row.get("answer_status") == "local_fallback",
        "context_compactions": int(row.get("context_compactions", 0)),
        "root_retrieved_title_recall": len(gold_titles & root_retrieved_titles) / title_denominator,
        "root_retrieved_all_titles": gold_titles <= root_retrieved_titles,
        "root_visible_title_recall": len(gold_titles & root_visible_titles) / title_denominator,
        "root_visible_all_titles": gold_titles <= root_visible_titles,
        "cumulative_retrieved_title_recall": len(gold_titles & retrieved_titles) / title_denominator,
        "cumulative_retrieved_any_title": bool(gold_titles & retrieved_titles),
        "cumulative_retrieved_all_titles": gold_titles <= retrieved_titles,
        "cumulative_visible_title_recall": len(gold_titles & visible_titles) / title_denominator,
        "cumulative_visible_any_title": bool(gold_titles & visible_titles),
        "cumulative_visible_all_titles": gold_titles <= visible_titles,
        "support_sentence_strict_recall": sum(strict) / fact_denominator,
        "support_sentence_strict_any": any(strict),
        "support_sentence_strict_all": bool(strict) and all(strict),
        "support_sentence_fuzzy_recall": sum(fuzzy) / fact_denominator,
        "support_sentence_fuzzy_any": any(fuzzy),
        "support_sentence_fuzzy_all": bool(fuzzy) and all(fuzzy),
        "root_support_sentence_strict_recall": sum(root_strict) / fact_denominator,
        "root_support_sentence_strict_all": bool(root_strict) and all(root_strict),
        "root_support_sentence_fuzzy_recall": sum(root_fuzzy) / fact_denominator,
        "answer_lexical_hit_visible": any(value and value in evidence_norm for value in answer_norms),
        "gold_supporting_title_count": len(gold_titles),
        "gold_supporting_fact_count": len(facts),
        "mean_support_sentence_token_recall": sum(fuzzy_scores) / fact_denominator,
        "per_search_round_evidence": round_metrics,
    }


NUMERIC_FIELDS = [
    "answer_em", "answer_f1", "answer_acc", "search_calls", "multiple_search",
    "visible_evidence_tokens_total", "standalone_model_calls",
    "standalone_llm_input_tokens", "standalone_llm_output_tokens",
    "standalone_thinking_tokens",
    "continuation_llm_input_tokens", "continuation_llm_output_tokens",
    "format_repair_calls", "local_fallback", "context_compactions",
    "root_retrieved_title_recall", "root_retrieved_all_titles",
    "root_visible_title_recall", "root_visible_all_titles",
    "cumulative_retrieved_title_recall", "cumulative_retrieved_any_title",
    "cumulative_retrieved_all_titles", "cumulative_visible_title_recall",
    "cumulative_visible_any_title", "cumulative_visible_all_titles",
    "support_sentence_strict_recall", "support_sentence_strict_any",
    "support_sentence_strict_all", "support_sentence_fuzzy_recall",
    "support_sentence_fuzzy_any", "support_sentence_fuzzy_all",
    "root_support_sentence_strict_recall", "root_support_sentence_strict_all",
    "root_support_sentence_fuzzy_recall", "answer_lexical_hit_visible",
    "mean_support_sentence_token_recall",
]


def new_accumulator():
    return {"n": 0, "sums": Counter(), "search_distribution": Counter()}


def add(acc, row):
    acc["n"] += 1
    for field in NUMERIC_FIELDS:
        acc["sums"][field] += float(row[field])
    acc["search_distribution"][str(int(row["search_calls"]))] += 1


def finish(acc, extra=None):
    result = {**(extra or {}), "n": acc["n"]}
    for field in NUMERIC_FIELDS:
        result[field] = acc["sums"][field] / acc["n"] if acc["n"] else 0.0
    result["search_count_distribution"] = json.dumps(
        dict(sorted(acc["search_distribution"].items())), ensure_ascii=False
    )
    return result


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def pareto(rows, quality="answer_f1", cost="standalone_llm_input_tokens"):
    ordered = sorted(rows, key=lambda row: (row[cost], -row[quality]))
    best = -1.0; frontier = []
    for row in ordered:
        if row[quality] > best + 1e-12:
            frontier.append(int(row["budget"])); best = row[quality]
    return frontier


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    labels = {str(row["sample_id"]): row for row in read_jsonl(args.labels)}
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    scored_path = output / "scored_records.jsonl.gz"
    temporary = scored_path.with_name(scored_path.name + f".tmp.{os.getpid()}")
    by_budget_acc = defaultdict(new_accumulator)
    by_group_acc = defaultdict(new_accumulator)
    by_hit_acc = defaultdict(new_accumulator)
    overall_acc = new_accumulator()
    support_all_acc = {True: new_accumulator(), False: new_accumulator()}
    by_round_acc = defaultdict(lambda: {"n": 0, "sums": Counter()})
    seen = set(); missing_labels = []
    with gzip.open(temporary, "wt", encoding="utf-8", compresslevel=6) as sink:
        for trajectory in iter_jsonl(args.trajectories):
            sample_id = str(trajectory["sample_id"])
            label = labels.get(sample_id)
            if label is None:
                missing_labels.append(sample_id); continue
            score = score_record(trajectory, label)
            key = (sample_id, score["budget"])
            if key in seen:
                raise ValueError(f"Duplicate trajectory key: {key}")
            seen.add(key)
            sink.write(json.dumps(score, ensure_ascii=False) + "\n")
            add(overall_acc, score)
            add(by_budget_acc[score["budget"]], score)
            add(by_group_acc[(score["budget"], score["type"], score["level"])], score)
            title_class = (
                "all" if score["cumulative_retrieved_all_titles"]
                else "partial" if score["cumulative_retrieved_any_title"] else "none"
            )
            sentence_class = (
                "all" if score["support_sentence_strict_all"]
                else "partial" if score["support_sentence_strict_any"] else "none"
            )
            add(by_hit_acc[(score["budget"], title_class, sentence_class)], score)
            add(support_all_acc[bool(score["support_sentence_strict_all"])], score)
            for round_score in score["per_search_round_evidence"]:
                round_acc = by_round_acc[(score["budget"], round_score["search_round"])]
                round_acc["n"] += 1
                for field, value in round_score.items():
                    if field not in {"search_round", "query"}:
                        round_acc["sums"][field] += float(value)
    os.replace(temporary, scored_path)

    budget_rows = [finish(by_budget_acc[budget], {"budget": budget})
                   for budget in sorted(by_budget_acc)]
    group_rows = [finish(acc, {"budget": key[0], "type": key[1], "level": key[2]})
                  for key, acc in sorted(by_group_acc.items())]
    hit_rows = [finish(acc, {
        "budget": key[0], "retrieved_support_title_class": key[1],
        "visible_strict_support_sentence_class": key[2],
    }) for key, acc in sorted(by_hit_acc.items())]
    write_csv(output / "by_budget.csv", budget_rows)
    write_csv(output / "by_budget_type_level.csv", group_rows)
    write_csv(output / "by_budget_evidence_hit.csv", hit_rows)
    round_rows = []
    for (budget, search_round), acc in sorted(by_round_acc.items()):
        row = {
            "budget": budget, "search_round": search_round, "n": acc["n"],
            "trajectory_fraction_reaching_round": (
                acc["n"] / by_budget_acc[budget]["n"] if by_budget_acc[budget]["n"] else 0.0
            ),
        }
        for field, total in acc["sums"].items():
            row[field] = total / acc["n"] if acc["n"] else 0.0
        round_rows.append(row)
    write_csv(output / "by_budget_search_round.csv", round_rows)
    frontier = pareto(budget_rows)
    evidence_frontier = pareto(
        budget_rows, quality="answer_f1", cost="visible_evidence_tokens_total"
    )
    summary = {
        "created_at": utc_stamp(),
        "trajectories_sha256": file_sha256(args.trajectories),
        "labels_sha256": file_sha256(args.labels),
        "records": overall_acc["n"], "unique_sample_budget_keys": len(seen),
        "missing_label_records": len(missing_labels),
        "gold_join_stage": "offline_after_generation",
        "overall": finish(overall_acc),
        "quality_conditioned_on_all_strict_support_visible": {
            "all_support_visible": finish(support_all_acc[True]),
            "not_all_support_visible": finish(support_all_acc[False]),
        },
        "input_token_f1_pareto_budgets": frontier,
        "visible_evidence_token_f1_pareto_budgets": evidence_frontier,
        "format_reliability": {
            "format_repair_call_rate": finish(overall_acc)["format_repair_calls"],
            "local_fallback_rate": finish(overall_acc)["local_fallback"],
            "context_compaction_rate": finish(overall_acc)["context_compactions"],
        },
    }
    write_json(output / "summary.json", summary)
    key_rows = {row["budget"]: row for row in budget_rows}
    selected = sorted(key_rows)
    table = "\n".join(
        f"| {budget} | {key_rows[budget]['answer_em']:.3f} | "
        f"{key_rows[budget]['answer_f1']:.3f} | {key_rows[budget]['search_calls']:.2f} | "
        f"{key_rows[budget]['cumulative_retrieved_all_titles']:.3f} | "
        f"{key_rows[budget]['support_sentence_strict_all']:.3f} | "
        f"{key_rows[budget]['support_sentence_fuzzy_all']:.3f} |"
        for budget in selected
    )
    report = f"""# HotpotQA Qwen3-8B 自然多轮搜索证据预算报告

生成阶段不读取答案或 supporting facts；标签在轨迹冻结后离线连接。

| 每次搜索预算 | EM | F1 | 平均 Search 次数 | 累计检索命中全部支持标题 | 可见严格支持句全部命中 | 可见模糊支持句全部命中 |
|---:|---:|---:|---:|---:|---:|---:|
{table}

- 轨迹数：{overall_acc['n']}。
- 多轮搜索比例：{finish(overall_acc)['multiple_search']:.2%}。
- 格式修复调用率：{finish(overall_acc)['format_repair_calls']:.2%}。
- 本地文本 fallback 率：{finish(overall_acc)['local_fallback']:.2%}。
- F1—模型输入 token Pareto budgets：{frontier}。
- F1—实际可见证据 token Pareto budgets：{evidence_frontier}。

`retrieved title` 衡量 Top-10 检索是否命中标注页面；`visible sentence`
衡量相应 supporting-fact 句子是否真的落入当前 50-token 增量可见前缀。
严格句命中是完整标准化句子的连续包含；模糊句命中要求至少 80% 的支持句
token 出现在同一可见 passage 中。两者均应报告，严格值作为主口径。
"""
    (output / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if not missing_labels else 1


if __name__ == "__main__":
    raise SystemExit(main())
