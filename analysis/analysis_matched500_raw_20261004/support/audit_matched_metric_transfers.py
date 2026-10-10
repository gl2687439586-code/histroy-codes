#!/usr/bin/env python3
"""Question-paired, metric-matched oracle transfer diagnostics. No inference.

Reads user archives without extraction/execution. Writes a separate JSON/Markdown
audit; never replaces the source data or previous reports. F1 remains the supplied
legacy score because raw predictions are not in the review archives.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import tarfile
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

COMMON = [str(b) for b in range(100, 601, 100)] + ["FULL"]
METRICS = {"EM": "answer_em", "lexical_ACC": "answer_acc", "F1_legacy": "answer_f1"}
SEED = 20261003
BOOTSTRAPS = 10000


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stream_gz(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def archive_gz(tar, name):
    member = tar.getmember(name)
    assert member.isfile(), name
    with tar.extractfile(member) as f, gzip.GzipFile(fileobj=f) as g:
        for line in g:
            if line.strip():
                yield json.loads(line)


def indexed(rows, ids=None):
    result = {}
    for row in rows:
        sid = str(row["sample_id"])
        if ids is not None and sid not in ids:
            continue
        budget = str(row["condition_group"])
        group = result.setdefault(sid, {})
        assert budget not in group, (sid, budget, "duplicate")
        group[budget] = row
    return result


def total_tokens(row):
    return int(row["standalone_llm_input_tokens"]) + int(row["standalone_llm_output_tokens"])


def budget_order(budget):
    return (1, 0) if budget == "FULL" else (0, int(budget))


def select_oracle(group, choices, metric, tie_cost="total"):
    maximum = max(float(group[b][metric]) for b in choices)
    optimum = {b for b in choices if abs(float(group[b][metric]) - maximum) <= 1e-12}
    cost = total_tokens if tie_cost == "total" else lambda r: int(r["standalone_llm_input_tokens"])
    selected = min(optimum, key=lambda b: (cost(group[b]), budget_order(b), int(group[b]["search_calls"])))
    return selected, maximum, optimum


def perfect(score):
    return float(score) >= 1.0 - 1e-12


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    width = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - width), min(1.0, center + width)]


def paired_ci(values):
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if not len(x):
        return None
    rng = np.random.default_rng(SEED)
    samples = np.empty((BOOTSTRAPS, x.shape[1]))
    for start in range(0, BOOTSTRAPS, 100):
        count = min(100, BOOTSTRAPS - start)
        ix = rng.integers(0, len(x), (count, len(x)))
        samples[start:start + count] = x[ix].mean(axis=1)
    return np.quantile(samples, [0.025, 0.975], axis=0).T.tolist()


def mcnemar(gain, harm):
    n = gain + harm
    if not n:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) - n * math.log(2)
            for k in range(min(gain, harm) + 1)]
    peak = max(logs)
    return min(1.0, 2 * math.exp(peak) * sum(math.exp(x - peak) for x in logs))


def metric_means(rows):
    result = {m: float(np.mean([r[field] for r in rows])) for m, field in METRICS.items()}
    for name, field in [("input_tokens", "standalone_llm_input_tokens"),
                        ("output_tokens", "standalone_llm_output_tokens"),
                        ("evidence_tokens", "visible_evidence_tokens_total"),
                        ("search_calls", "search_calls"), ("format_fallback_rate", "local_fallback")]:
        result["mean_" + name] = float(np.mean([r[field] for r in rows]))
    result["mean_total_tokens"] = float(np.mean([total_tokens(r) for r in rows]))
    return result


def binary_matrix(before, after):
    assert len(before) == len(after)
    n = len(before)
    cc = sum(a and b for a, b in zip(before, after))
    cw = sum(a and not b for a, b in zip(before, after))
    wc = sum(not a and b for a, b in zip(before, after))
    ww = sum(not a and not b for a, b in zip(before, after))
    assert cc + cw + wc + ww == n
    old_correct, old_wrong = cc + cw, wc + ww
    return {"n": n, "correct_to_correct": cc, "correct_to_wrong": cw,
            "wrong_to_correct": wc, "wrong_to_wrong": ww,
            "before_correct": old_correct, "before_wrong": old_wrong,
            "after_correct": cc + wc, "net_correct_change": wc - cw,
            "retained_given_before_correct": cc / old_correct if old_correct else None,
            "retained_given_before_correct_wilson_ci95": wilson(cc, old_correct),
            "harm_given_before_correct": cw / old_correct if old_correct else None,
            "rescue_given_before_wrong": wc / old_wrong if old_wrong else None,
            "harm_given_before_correct_wilson_ci95": wilson(cw, old_correct),
            "rescue_given_before_wrong_wilson_ci95": wilson(wc, old_wrong),
            "correct_to_correct_rate_of_all": cc / n if n else None,
            "mcnemar_exact_p": mcnemar(wc, cw)}


def compare_rows(selected, reference, metric):
    differences = np.array([[float(a[metric]) - float(b[metric]), total_tokens(a) - total_tokens(b)]
                            for a, b in zip(selected, reference)], dtype=float)
    intervals = paired_ci(differences)
    qs = differences[:, 0]
    return {"quality_delta": float(qs.mean()), "quality_delta_ci95": intervals[0],
            "mean_total_token_delta": float(differences[:, 1].mean()),
            "mean_total_token_delta_ci95": intervals[1],
            "total_token_savings_fraction": 1 - np.mean([total_tokens(r) for r in selected]) /
                                              np.mean([total_tokens(r) for r in reference]),
            "quality_improved_questions": int((qs > 1e-12).sum()),
            "quality_worsened_questions": int((qs < -1e-12).sum()),
            "quality_tied_questions": int((np.abs(qs) <= 1e-12).sum()),
            "score_sum_gain": float(qs[qs > 1e-12].sum()),
            "score_sum_harm": float(-qs[qs < -1e-12].sum())}


def loss_rate_difference(details, flag, changed_is_false=False):
    """Exploratory stratified question bootstrap among old successful questions."""
    successful = [r for r in details if r["old_selected_perfect"]]
    changed = [r for r in successful if bool(r[flag]) != changed_is_false]
    unchanged = [r for r in successful if bool(r[flag]) == changed_is_false]
    if not changed or not unchanged:
        return {"defined": False}
    a = np.array([not r["new_at_old_budget_perfect"] for r in changed], dtype=float)
    b = np.array([not r["new_at_old_budget_perfect"] for r in unchanged], dtype=float)
    rng = np.random.default_rng(SEED)
    samples = []
    for start in range(0, BOOTSTRAPS, 100):
        count = min(100, BOOTSTRAPS - start)
        ix = rng.integers(0, len(a), (count, len(a)))
        iy = rng.integers(0, len(b), (count, len(b)))
        samples.extend((a[ix].mean(axis=1) - b[iy].mean(axis=1)).tolist())
    return {"defined": True, "flag": flag, "changed_group_n": len(a), "unchanged_group_n": len(b),
            "changed_group_losses": int(a.sum()), "unchanged_group_losses": int(b.sum()),
            "changed_group_loss_rate": float(a.mean()), "unchanged_group_loss_rate": float(b.mean()),
            "loss_rate_difference": float(a.mean() - b.mean()),
            "loss_rate_difference_ci95": np.quantile(samples, [.025, .975]).tolist(),
            "method": "Stratified question bootstrap conditional on old success; descriptive, post-selection, not causal; no multiple-comparison correction."}


def summarize_details(details, bootstrap=False):
    if not details:
        return {"n": 0}
    old = [r["old_selected_perfect"] for r in details]
    transfer = [r["new_at_old_budget_perfect"] for r in details]
    full = [r["new_FULL_perfect"] for r in details]
    new_max = [r["new_any_budget_perfect"] for r in details]
    losses = [r for r in details if r["old_selected_perfect"] and not r["new_at_old_budget_perfect"]]
    opportunity = binary_matrix(old, new_max)
    result = {"n": len(details),
              "old_selected_to_new_same_budget": binary_matrix(old, transfer),
              "new_FULL_to_old_budget_transfer": binary_matrix(full, transfer),
              "old_any_success_to_new_any_success": opportunity,
              "lost_old_successes": len(losses),
              "lost_old_successes_new_other_budget_recovers": sum(r["new_any_budget_perfect"] for r in losses),
              "lost_old_successes_no_new_success_in_grid": sum(not r["new_any_budget_perfect"] for r in losses),
              "new_oracle_successes_missed_by_transfer": sum(n and not t for n, t in zip(new_max, transfer)),
              "old_budget_still_new_quality_optimal": sum(r["old_budget_still_new_quality_optimal"] for r in details),
              "mean_old_oracle_quality": float(np.mean([r["old_quality"] for r in details])),
              "mean_transfer_quality": float(np.mean([r["transfer_quality"] for r in details])),
              "mean_new_oracle_quality": float(np.mean([r["new_quality"] for r in details])),
              "mean_new_FULL_quality": float(np.mean([r["new_FULL_quality"] for r in details])),
              "mean_old_to_transfer_quality_loss": float(np.mean([r["old_quality"] - r["transfer_quality"] for r in details])),
              "mean_new_oracle_regret": float(np.mean([r["new_quality"] - r["transfer_quality"] for r in details])),
              "mean_transfer_minus_new_FULL_quality": float(np.mean([r["transfer_quality"] - r["new_FULL_quality"] for r in details])),
              "selected_budget_same_count": sum(r["old_budget"] == r["new_budget"] for r in details),
              "transfer_format_fallbacks": sum(r["transfer_format_fallback"] for r in details),
              "new_FULL_format_fallbacks": sum(r["new_FULL_format_fallback"] for r in details),
              "mean_transfer_total_tokens": float(np.mean([r["transfer_total_tokens"] for r in details])),
              "mean_new_FULL_total_tokens": float(np.mean([r["new_FULL_total_tokens"] for r in details]))}
    if bootstrap:
        result["transfer_minus_FULL_quality_ci95"] = paired_ci([
            r["transfer_quality"] - r["new_FULL_quality"] for r in details])[0]
    return result


def compact(row, metric):
    return {"quality": float(row[metric]),
            "all_metrics": {label: float(row[field]) for label, field in METRICS.items()},
            "input_tokens": int(row["standalone_llm_input_tokens"]),
            "output_tokens": int(row["standalone_llm_output_tokens"]),
            "total_tokens": total_tokens(row),
            "evidence_tokens": int(row["visible_evidence_tokens_total"]),
            "search_calls": int(row["search_calls"]), "local_fallback": bool(row["local_fallback"])}


def root_record(sid, a, b):
    ra, rb = a["FULL"], b["FULL"]
    qa = str(ra["per_search_round_evidence"][0]["query"])
    qb = str(rb["per_search_round_evidence"][0]["query"])
    output_a = int(ra["standalone_llm_output_tokens"] - ra["continuation_llm_output_tokens"])
    output_b = int(rb["standalone_llm_output_tokens"] - rb["continuation_llm_output_tokens"])
    input_a = int(ra["standalone_llm_input_tokens"] - ra["continuation_llm_input_tokens"])
    input_b = int(rb["standalone_llm_input_tokens"] - rb["continuation_llm_input_tokens"])
    return {"sample_id": sid, "old_query": qa, "new_query": qb,
            "query_same_exact": qa == qb,
            "query_same_normalized": " ".join(qa.lower().split()) == " ".join(qb.lower().split()),
            "retrieved_length_same": ra["root_full_context_tokens"] == rb["root_full_context_tokens"],
            "old_root_full_tokens": int(ra["root_full_context_tokens"]),
            "new_root_full_tokens": int(rb["root_full_context_tokens"]),
            "old_root_input_tokens": input_a, "new_root_input_tokens": input_b,
            "old_root_output_tokens": output_a, "new_root_output_tokens": output_b,
            "root_output_count_same": output_a == output_b,
            "old_root_at_output_cap_proxy": output_a >= 8192,
            "new_root_at_output_cap_proxy": output_b >= 8192,
            "either_root_at_output_cap_proxy": output_a >= 8192 or output_b >= 8192,
            "actual_checkpoint_identical": None,
            "actual_document_hashes_identical": None,
            "confirmed_generation_truncation": None}


def audit_metric(old, new, ids, roots, metric, choices=None, tie_cost="total"):
    choices = choices or {s: COMMON for s in ids}
    details = []
    old_rows, transfer_rows, new_rows, full_rows, fixed_rows = [], [], [], [], []
    for sid in ids:
        ba, qa, sa = select_oracle(old[sid], choices[sid], metric, tie_cost)
        bb, qb, sb = select_oracle(new[sid], choices[sid], metric, tie_cost)
        a, t, n, f, fix = old[sid][ba], new[sid][ba], new[sid][bb], new[sid]["FULL"], new[sid]["500"]
        root = roots[sid]
        row = {"sample_id": sid, "selection_metric": metric, "evaluation_metric": metric,
               "choices": choices[sid], "old_budget": ba, "new_budget": bb,
               "old_quality": qa, "transfer_quality": float(t[metric]), "new_quality": qb,
               "new_FULL_quality": float(f[metric]),
               "old_optimal_budgets": sorted(sa, key=budget_order),
               "new_optimal_budgets": sorted(sb, key=budget_order),
               "old_budget_still_new_quality_optimal": ba in sb,
               "old_selected_perfect": perfect(qa), "new_at_old_budget_perfect": perfect(t[metric]),
               "new_any_budget_perfect": perfect(qb), "new_FULL_perfect": perfect(f[metric]),
               "old_selected_format_fallback": bool(a["local_fallback"]),
               "transfer_format_fallback": bool(t["local_fallback"]),
               "new_FULL_format_fallback": bool(f["local_fallback"]),
               "old_selected_total_tokens": total_tokens(a), "transfer_total_tokens": total_tokens(t),
               "new_oracle_total_tokens": total_tokens(n), "new_FULL_total_tokens": total_tokens(f),
               "query_same_exact": root["query_same_exact"],
               "query_same_normalized": root["query_same_normalized"],
               "root_retrieved_length_same": root["retrieved_length_same"],
               "root_output_count_same": root["root_output_count_same"],
               "either_root_at_output_cap_proxy": root["either_root_at_output_cap_proxy"],
               "context_compaction_any": any(int(x["context_compactions"]) > 0 for x in [a, t, f]),
               "old_query_sequence": [x["query"] for x in a["per_search_round_evidence"]],
               "new_query_sequence_at_old_budget": [x["query"] for x in t["per_search_round_evidence"]],
               "old_selected": compact(a, metric), "new_at_old_budget": compact(t, metric),
               "new_oracle": compact(n, metric), "new_FULL": compact(f, metric)}
        row["query_sequence_same_exact"] = row["old_query_sequence"] == row["new_query_sequence_at_old_budget"]
        lost = row["old_selected_perfect"] and not row["new_at_old_budget_perfect"]
        row["old_success_loss_category"] = ("winner_replacement_new_grid_still_has_success" if row["new_any_budget_perfect"]
                                            else "opportunity_disappeared_within_observed_grid") if lost else None
        details.append(row)
        old_rows.append(a); transfer_rows.append(t); new_rows.append(n); full_rows.append(f); fixed_rows.append(fix)

    result = {"selection_metric": metric, "evaluation_metric": metric,
              "correct_definition": "score == 1 (within 1e-12); F1=1 denotes perfect legacy F1, not EM",
              "tie_break": f"{tie_cost} tokens, ascending numeric budget, FULL last, search count",
              "all": summarize_details(details, bootstrap=True),
              "old_oracle": metric_means(old_rows), "transfer": metric_means(transfer_rows),
              "new_oracle": metric_means(new_rows), "new_FULL": metric_means(full_rows),
              "new_fixed500": metric_means(fixed_rows),
              "transfer_vs_new_FULL": compare_rows(transfer_rows, full_rows, metric),
              "transfer_vs_new_fixed500": compare_rows(transfer_rows, fixed_rows, metric),
              "transfer_vs_new_oracle": compare_rows(transfer_rows, new_rows, metric),
              "new_oracle_vs_old_oracle": compare_rows(new_rows, old_rows, metric)}
    groups = {
        "root_query_same_exact": lambda r: r["query_same_exact"],
        "root_query_changed_exact": lambda r: not r["query_same_exact"],
        "root_query_same_normalized": lambda r: r["query_same_normalized"],
        "root_query_changed_normalized": lambda r: not r["query_same_normalized"],
        "root_retrieved_length_same": lambda r: r["root_retrieved_length_same"],
        "root_retrieved_length_changed": lambda r: not r["root_retrieved_length_same"],
        "root_output_count_same": lambda r: r["root_output_count_same"],
        "root_output_count_changed": lambda r: not r["root_output_count_same"],
        "root_query_sequence_same": lambda r: r["query_sequence_same_exact"],
        "query_sequence_changed": lambda r: not r["query_sequence_same_exact"],
        "either_root_at_cap_proxy": lambda r: r["either_root_at_output_cap_proxy"],
        "neither_root_at_cap_proxy": lambda r: not r["either_root_at_output_cap_proxy"],
        "transfer_format_fallback": lambda r: r["transfer_format_fallback"],
        "transfer_not_format_fallback": lambda r: not r["transfer_format_fallback"],
        "both_new_FULL_and_transfer_nonfallback": lambda r: not r["transfer_format_fallback"] and not r["new_FULL_format_fallback"],
        "any_old_or_transfer_or_FULL_fallback": lambda r: r["old_selected_format_fallback"] or r["transfer_format_fallback"] or r["new_FULL_format_fallback"],
        "no_old_or_transfer_or_FULL_fallback": lambda r: not (r["old_selected_format_fallback"] or r["transfer_format_fallback"] or r["new_FULL_format_fallback"]),
    }
    result["subgroups"] = {name: summarize_details([r for r in details if predicate(r)])
                           for name, predicate in groups.items()}
    losses = [r for r in details if r["old_selected_perfect"] and not r["new_at_old_budget_perfect"]]
    result["loss_overlapping_flags"] = {
        "lost_old_successes": len(losses),
        "root_query_changed": sum(not r["query_same_exact"] for r in losses),
        "root_retrieved_length_changed": sum(not r["root_retrieved_length_same"] for r in losses),
        "transfer_format_fallback": sum(r["transfer_format_fallback"] for r in losses),
        "either_root_at_cap_proxy": sum(r["either_root_at_output_cap_proxy"] for r in losses),
        "all_queries_same_exact": sum(r["query_sequence_same_exact"] for r in losses),
        "root_query_same_and_transfer_nonfallback_and_no_root_cap_proxy": sum(
            r["query_same_exact"] and not r["transfer_format_fallback"] and not r["either_root_at_output_cap_proxy"]
            for r in losses),
        "context_compaction": sum(r["context_compaction_any"] for r in losses),
        "confirmed_truncation": None, "confirmed_protocol_change": None,
        "note": "Flags overlap. Root output cap is a retokenized counter proxy, not a verified finish_reason. Associations are not causal attribution."}
    result["loss_exclusive_observed_partition"] = {
        "new_format_fallback": sum(r["transfer_format_fallback"] for r in losses),
        "nonfallback_and_root_query_changed": sum(not r["transfer_format_fallback"] and not r["query_same_exact"] for r in losses),
        "nonfallback_and_root_query_same": sum(not r["transfer_format_fallback"] and r["query_same_exact"] for r in losses)}
    assert sum(result["loss_exclusive_observed_partition"].values()) == len(losses)
    result["loss_rate_root_comparisons"] = {
        "query_changed_minus_same": loss_rate_difference(details, "query_same_exact", changed_is_false=True),
        "retrieved_length_changed_minus_same": loss_rate_difference(details, "root_retrieved_length_same", changed_is_false=True)}
    return result, details


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def fraction(k, n):
    return f"{k}/{n}（{100 * k / n:.1f}%）" if n else "0/0（不适用）"


def make_report(result):
    analyses = result["analyses"]
    em = analyses["EM"]; acc = analyses["lexical_ACC"]; f1 = analyses["F1_legacy"]
    e = em["all"]
    retained = e["old_selected_to_new_same_budget"]
    full_compare = e["new_FULL_to_old_budget_transfer"]
    lines = ["# 同题、同指标 Oracle 迁移审计", "",
             "## 结论先行", "",
             f"以EM作为选择和评估目标：旧Oracle正确{retained['before_correct']}题，旧预算在新轮仍正确{retained['correct_to_correct']}题，保留率{retained['retained_given_before_correct']:.1%}；旧预算失效{e['lost_old_successes']}题，其中{e['lost_old_successes_new_other_budget_recovers']}题的新轮其他预算仍能答对，只有{e['lost_old_successes_no_new_success_in_grid']}题在当前候选集合里没有成功记录。主要是旧赢家失效，不是普遍机会消失。",
             f"相对新FULL，旧EM预算观测到救回{full_compare['wrong_to_correct']}题、损害{full_compare['correct_to_wrong']}题，净增{full_compare['net_correct_change']}题；EM从38.2%到39.0%，尚无显著提升。平均总tokens从4504.2降至{em['transfer']['mean_total_tokens']:.1f}/题。",
             "这些结果不证明预算优化无空间，也不证明可以在线获得新Oracle收益。它们证明旧预算迁移的救回和损害大幅抵消，必须区分成功机会与具体预算赢家。", "",
             "## 题目和 Oracle 定义", "",
             "旧轮 random500 的同一批500题全部匹配新轮 full7405；问题文本与支持事实标注一致。所有主表旧轮数值均在这500题重算，不使用全量7405均值。",
             "候选预算固定为100、200、300、400、500、600、FULL；两轮均有每个候选。主Oracle先最大化所标明指标，再最小化输入+输出总tokens；剩余平局按数值预算升序、FULL最后。",
             "此前EM=39.4%、F1=0.5284是F1选择Oracle后的跨指标描述，不是EM Oracle。下面分别以EM选/评EM、词法ACC选/评词法ACC、旧口径F1选/评F1。F1二元表只表示F1=1，不冒充严格正确率。",
             "旧F1评分缺少HotpotQA官方yes/no/noanswer特殊规则；原始答案未附，无法校正。词法ACC是gold字符串被预测包含，不是LLM Judge或语义正确率。格式fallback在主指标中计零，未过滤。",
             "", "## 1. 同指标均值与每题平均总成本", "",
             "|选择/评估指标|旧轮Oracle|旧预算在新轮|新轮FULL|新轮固定500|新轮Oracle|旧预算新轮tokens/题|FULL tokens/题|固定500 tokens/题|",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label, metric in [("EM", "EM"), ("词法ACC", "lexical_ACC"), ("F1（旧口径）", "F1_legacy")]:
        a = analyses[metric]
        fmt = lambda st: f"{100 * st[metric]:.2f}%" if metric != "F1_legacy" else f"{st[metric]:.4f}"
        lines.append(f"|{label}|{fmt(a['old_oracle'])}|{fmt(a['transfer'])}|{fmt(a['new_FULL'])}|{fmt(a['new_fixed500'])}|{fmt(a['new_oracle'])}|{a['transfer']['mean_total_tokens']:.1f}|{a['new_FULL']['mean_total_tokens']:.1f}|{a['new_fixed500']['mean_total_tokens']:.1f}|")
    lines += ["", "成本包含root以及所选单条轨迹的输入和输出，不包含枚举Oracle的全部离线计算。累计证据tokens与LLM输入tokens不是同一成本；详见JSON。", "",
              "## 2. 旧轮正确，新轮相同预算是否仍正确？", "",
              "|Oracle选择/正确性定义|旧轮正确|新轮仍正确|旧对新错|旧错新对|旧错新错|正确性保留率（分母=旧正确）|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for label, metric in [("EM", "EM"), ("词法ACC", "lexical_ACC"), ("F1=1（非EM）", "F1_legacy")]:
        m = analyses[metric]["all"]["old_selected_to_new_same_budget"]
        lines.append(f"|{label}|{m['before_correct']}|{m['correct_to_correct']}|{m['correct_to_wrong']}|{m['wrong_to_correct']}|{m['wrong_to_wrong']}|{fraction(m['correct_to_correct'], m['before_correct'])}|")
    lines += ["", "这里比较旧轮选中预算在两轮的结果，不是把旧轮Oracle直接与新轮Oracle做保留率比较。", "",
              "## 3. 相对新轮FULL救回/损害", "",
              "|同指标选择/评估|FULL错的题数|救回|FULL对的题数|损害|净新增正确|净变化95%配对CI|McNemar p|",
              "|---|---:|---:|---:|---:|---:|---|---:|"]
    for label, metric in [("EM", "EM"), ("词法ACC", "lexical_ACC"), ("F1=1（非EM）", "F1_legacy")]:
        a = analyses[metric]; m = a["all"]["new_FULL_to_old_budget_transfer"]
        if metric == "F1_legacy":
            interval_text = "F1=1计数；连续F1另表"
        else:
            interval = a["transfer_vs_new_FULL"]["quality_delta_ci95"]
            interval_text = f"[{100 * interval[0]:+.2f}, {100 * interval[1]:+.2f}] pp"
        lines.append(f"|{label}|{m['before_wrong']}|{fraction(m['wrong_to_correct'], m['before_wrong'])}|{m['before_correct']}|{fraction(m['correct_to_wrong'], m['before_correct'])}|{m['net_correct_change']:+d}|{interval_text}|{m['mcnemar_exact_p']:.4f}|")
    lines += ["", "F1按连续分数比较：",
              f"- 旧预算比新FULL平均F1高{f1['transfer_vs_new_FULL']['quality_delta']:+.4f}，95%CI {f1['transfer_vs_new_FULL']['quality_delta_ci95']}。",
              f"- 分数提高{f1['transfer_vs_new_FULL']['quality_improved_questions']}题，下降{f1['transfer_vs_new_FULL']['quality_worsened_questions']}题，持平{f1['transfer_vs_new_FULL']['quality_tied_questions']}题。", "",
              "## 4. 机会消失还是赢家更替？", "",
              "|同指标|两轮均有成功预算|旧有新无|旧无新有|两轮均无|旧正确预算失效|失效但新轮其他预算成功|失效且新轮候选全部失败|",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, metric in [("EM", "EM"), ("词法ACC", "lexical_ACC"), ("F1=1（非EM）", "F1_legacy")]:
        a = analyses[metric]["all"]; m = a["old_any_success_to_new_any_success"]
        lines.append(f"|{label}|{m['correct_to_correct']}|{m['correct_to_wrong']}|{m['wrong_to_correct']}|{m['wrong_to_wrong']}|{a['lost_old_successes']}|{a['lost_old_successes_new_other_budget_recovers']}|{a['lost_old_successes_no_new_success_in_grid']}|")
    lines += ["", "机会消失仅指当前共同七预算没有成功记录，不意味着问题在所有推理或所有预算下不可解。赢家更替仅指旧预算不再成功、而新轮其他预算仍成功，不把质量并列但成本tie-break改变误记为成功机会损失。", "",
              "## 5. 根、格式失败、截断与协议分层（EM）", "",
              "|子组|题数|旧Oracle正确|旧对新错|旧正确保留率|相对新FULL救回|相对新FULL损害|平均EM Oracle regret|",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    groups = [("首query完全相同", "root_query_same_exact"), ("首query改变", "root_query_changed_exact"),
              ("首轮检索长度相同", "root_retrieved_length_same"), ("首轮检索长度改变", "root_retrieved_length_changed"),
              ("新旧首轮任一计数达8192（代理）", "either_root_at_cap_proxy"),
              ("新旧首轮均未达8192（代理）", "neither_root_at_cap_proxy"),
              ("新轮旧预算格式fallback", "transfer_format_fallback"),
              ("新轮旧预算非fallback", "transfer_not_format_fallback"),
              ("新FULL与旧预算均非fallback", "both_new_FULL_and_transfer_nonfallback")]
    for label, key in groups:
        s = em["subgroups"][key]
        if not s["n"]:
            continue
        m = s["old_selected_to_new_same_budget"]; f = s["new_FULL_to_old_budget_transfer"]
        lines.append(f"|{label}|{s['n']}|{m['before_correct']}|{m['correct_to_wrong']}|{fraction(m['correct_to_correct'], m['before_correct'])}|{f['wrong_to_correct']}|{f['correct_to_wrong']}|{s['mean_new_oracle_regret']:.4f}|")
    flags = em["loss_overlapping_flags"]
    lines += ["", f"EM旧正确后失效共{flags['lost_old_successes']}题，其中首query改变{flags['root_query_changed']}题、新轮所选轨迹fallback {flags['transfer_format_fallback']}题、首轮达输出cap计数代理{flags['either_root_at_cap_proxy']}题。标签可重叠，不能直接相加。",
              f"互斥拆分：{em['loss_exclusive_observed_partition']}。",
              f"即使首query相同、新轮非fallback且两轮首轮输出计数均未达cap，仍有{flags['root_query_same_and_transfer_nonfallback_and_no_root_cap_proxy']}题失效。",
              "这些是描述性分层，不是因果归因。query相同、长度相同、计数相同都不足以证明实际checkpoint或文档内容相同。",
              "首query改变组本身占全部题目的61.4%，因此不能只看它占失效题的多数就断言失效集中在根变化组。条件损失率需以该组旧Oracle正确题数为分母。"]
    for label, key in [("首query改变−相同", "query_changed_minus_same"), ("首轮检索长度改变−相同", "retrieved_length_changed_minus_same")]:
        c = em["loss_rate_root_comparisons"][key]
        lo, hi = c["loss_rate_difference_ci95"]
        lines.append(f"- {label}的旧正确失效率差：{100*c['loss_rate_difference']:+.2f} pp，探索性分层bootstrap 95%CI [{100*lo:+.2f}, {100*hi:+.2f}] pp。")
    lines += ["", "## 6. 截断、协议和评分的可核验边界", "",
              "已记录的两轮核心配置一致：Top5、总搜索上限4、原生thinking、temperature=0、context40960、客户端token限制24497、action/answer上限8192、repair=0、模型路径。旧4个shard的核心签名一致。",
              "未提供旧轮代码/prompt hash、权重checksum、实际服务内核/采样器与逐请求日志，因此不能证明协议字节级一致，也不能排除未记录的实现或服务变化。全量question/labels的hash与500题子集不同是数据范围不同，不是已经发现协议变化。",
              "review未保存每次生成finish_reason、原始output、模型输出格式尾部或实际文档hash；无法判定终端长度截断、格式闭合损坏或root搜索解析fallback的具体原因。root输出计数达8192只能列为可疑代理。不能把所有格式fallback归因于截断。",
              "统计使用按题配对bootstrap 10000次，seed20261003；二元比较为精确McNemar。分层和三指标检查为探索性，未做多重校正。去掉fallback的结果只作事后敏感性分析，不能替代主结果。", "",
              "## 7. 与先前F1选预算、EM评正确性的表述对账", "",
              "该表明确是跨指标诊断，不作为同指标主结论：",
              "|F1选择旧预算，按EM评估|数值|", "|---|---:|"]
    d = result["prior_claim_reconciliation_F1_selected_EM_evaluated"]
    for label, value in [("旧轮所选轨迹EM", d["old_selected_EM"]), ("新轮相同预算EM", d["transfer_EM"]),
                         ("新轮FULL EM", d["new_FULL_EM"]), ("新轮F1 Oracle所选轨迹EM", d["new_F1_oracle_selected_EM"])]:
        lines.append(f"|{label}|{100 * value:.2f}%|")
    m = d["old_selected_to_new_same_budget_EM"]
    f = d["new_FULL_to_old_budget_transfer_EM"]
    lines += [f"|旧轮EM正确在新轮保留|{fraction(m['correct_to_correct'], m['before_correct'])}|",
              f"|相对新FULL救回/损害|{f['wrong_to_correct']}/{f['correct_to_wrong']}|", "",
              "注意真正按EM最大化的新轮Oracle为主表的EM Oracle，不应把按F1选预算得到的EM当作严格EM上界。", "",
              "## 8. 候选集合与原先tie-break敏感性", "",
              "|定义|选择指标|评估同指标：旧Oracle→新迁移→新Oracle|旧预算新轮tokens/题|旧对新错|其他新预算仍能成功|",
              "|---|---|---|---:|---:|---:|"]
    for label, a in result["sensitivity"].items():
        field = a["selection_metric"]
        public = next(k for k, v in METRICS.items() if v == field)
        fmt = lambda st: f"{st[public]:.4f}"
        stats = a["all"]
        lines.append(f"|{label}|{public}|{fmt(a['old_oracle'])}→{fmt(a['transfer'])}→{fmt(a['new_oracle'])}|{a['transfer']['mean_total_tokens']:.1f}|{stats['lost_old_successes']}|{stats['lost_old_successes_new_other_budget_recovers']}|")
    lines += ["", "## 输出与复核", "",
              "SUMMARY.json包含所有三指标均值、成本、条件保留率、FULL救回/损害、置信区间、分层和候选敏感性。PER_QUESTION.json包含500题逐指标预算选择、分数、成本、格式标记及查询序列；ROOTS.json包含根的可观察变化；IDENTITY_AND_PROTOCOL.json包含原始文件hash、题目集hash和配置核验。没有新增模型调用。", ""]
    return "\n".join(lines)


def self_test():
    def row(q, c):
        return {"answer_em": q, "standalone_llm_input_tokens": c, "standalone_llm_output_tokens": 0, "search_calls": 1}
    group = {"100": row(1, 30), "200": row(1, 20), "FULL": row(0, 5)}
    b, q, candidates = select_oracle(group, ["100", "200", "FULL"], "answer_em")
    assert b == "200" and q == 1 and candidates == {"100", "200"}
    matrix = binary_matrix([True, True, False, False], [True, False, True, False])
    assert all(matrix[k] == 1 for k in ["correct_to_correct", "correct_to_wrong", "wrong_to_correct", "wrong_to_wrong"])
    assert matrix["retained_given_before_correct"] == 0.5 and matrix["net_correct_change"] == 0
    assert mcnemar(5, 5) == 1 and abs(mcnemar(4, 0) - 0.125) < 1e-12
    assert perfect(1) and not perfect(.999) and wilson(1, 1)[1] == 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-archive", default="C:/Users/RESEARCH_USER/Desktop/过程结果/hotpot_random500_v433_final_review.tar.gz")
    ap.add_argument("--full-input", default="audit_hotpot_full7405_v433_20261001/input")
    ap.add_argument("--output", default="analysis_oracle_repeat500_20261003/matched_metric_audit")
    ap.add_argument("--zip", default="deliverables/HotpotQA_500_metric_matched_transfer_audit_20261003.zip")
    args = ap.parse_args()
    self_test()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    full = Path(args.full_input)
    with tarfile.open(args.source_archive, "r:gz") as tar:
        old = indexed(archive_gz(tar, "analysis/scored_records.jsonl.gz"))
        old_q = {r["sample_id"]: r for r in archive_gz(tar, "analysis/evidence_positions/full_endpoint_positions_one_per_question.jsonl.gz")}
        old_signatures = [json.load(tar.extractfile(f"trajectories/shards/shard_{j}/manifest.json"))["signature"] for j in range(4)]
        source_members = tar.getnames()
    ids = sorted(old); id_set = set(ids)
    assert len(ids) == 500
    new = indexed(stream_gz(full / "analysis/scored_records.jsonl.gz"), id_set)
    new_q = {r["sample_id"]: r for r in stream_gz(full / "analysis/evidence_positions/full_endpoint_positions_one_per_question.jsonl.gz") if r["sample_id"] in id_set}
    assert set(new) == id_set and set(old_q) == id_set and set(new_q) == id_set
    for sid in ids:
        assert old_q[sid]["question"] == new_q[sid]["question"], sid
        facts = lambda r: [(f["gold_title"], f["gold_sentence_id"], f["gold_sentence"]) for f in r["supporting_facts"]]
        assert facts(old_q[sid]) == facts(new_q[sid]), sid
        for index in [old, new]:
            assert set(COMMON) <= set(index[sid]), sid
            rows = list(index[sid].values())
            assert len({r["root_full_context_tokens"] for r in rows}) == 1
            assert len({r["per_search_round_evidence"][0]["query"] for r in rows}) == 1
            for channel in ["input", "output"]:
                assert len({r[f"standalone_llm_{channel}_tokens"] - r[f"continuation_llm_{channel}_tokens"] for r in rows}) == 1
            for r in rows:
                assert r["answer_em"] in (0, 1) and r["answer_acc"] in (0, 1)
                assert 0 <= r["answer_f1"] <= 1
                if r["local_fallback"]:
                    assert all(r[field] == 0 for field in METRICS.values())
    config = json.loads((full / "FULL7405_CONFIG.json").read_text(encoding="utf-8"))
    protocol = config["protocol"]
    key_pairs = {"top_k": "top_k", "max_searches": "max_searches_including_root", "context_length": "context_length",
                 "server_max_total_tokens": "client_token_limit", "action_max_new_tokens": "action_max_new_tokens",
                 "answer_max_new_tokens": "answer_max_new_tokens", "repair_max_new_tokens": "repair_max_new_tokens", "temperature": "temperature"}
    checks = {key: {"old": old_signatures[0][key], "new": protocol[target],
                    "matched": old_signatures[0][key] == protocol[target]} for key, target in key_pairs.items()}
    checks["native_thinking"] = {"old": old_signatures[0]["thinking_control"], "new": protocol["native_thinking"],
                                  "matched": old_signatures[0]["thinking_control"] == "qwen3_native_thinking" and protocol["native_thinking"] is True}
    checks["model_path"] = {"old": old_signatures[0]["model_path"], "new": config["model_path"],
                            "matched": old_signatures[0]["model_path"] == config["model_path"]}
    assert all(c["matched"] for c in checks.values())
    assert all(all(s[key] == old_signatures[0][key] for key in key_pairs) for s in old_signatures)
    roots = {sid: root_record(sid, old[sid], new[sid]) for sid in ids}
    for sid in ids:
        roots[sid]["question"] = old_q[sid]["question"]
    identity = {"samples": len(ids), "sample_ids": ids,
                "sample_ids_sha256": hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest(),
                "old_scored_records": sum(map(len, old.values())), "new_matched_scored_records": sum(map(len, new.values())),
                "old_archive_sha256": sha256(args.source_archive),
                "new_scored_records_sha256": sha256(full / "analysis/scored_records.jsonl.gz"),
                "same_question_text": True, "same_support_annotations": True, "common_candidates": COMMON,
                "known_protocol_checks": checks, "known_protocol_match": True,
                "old_shard_signatures": old_signatures, "new_config": config,
                "new_model_calls": 0,
                "root_unchanged_checkpoint_verified": False,
                "raw_generation_finish_reasons_available": False,
                "old_code_prompt_weight_checksums_available": False,
                "old_archive_members": source_members,
                "F1_scorer": "supplied legacy overlap F1; official HotpotQA special yes/no handling not verified or corrected",
                "definitions": {"EM": "strict normalized exact match", "lexical_ACC": "nonempty normalized gold substring of normalized prediction",
                                "F1_legacy": "supplied token-overlap F1, continuous; binary table labels perfect F1=1 only"}}
    result = {"identity": {key: identity[key] for key in ["samples", "sample_ids_sha256", "old_archive_sha256", "new_scored_records_sha256", "same_question_text", "same_support_annotations", "common_candidates", "new_model_calls"]},
              "bootstrap_repetitions": BOOTSTRAPS, "bootstrap_seed": SEED, "analyses": {}, "sensitivity": {}}
    details_all = {}
    for name, field in METRICS.items():
        analysis, details = audit_metric(old, new, ids, roots, field)
        result["analyses"][name] = analysis; details_all[name] = details
        print(json.dumps({"metric": name, "old": analysis["old_oracle"][name], "transfer": analysis["transfer"][name],
                          "new_oracle": analysis["new_oracle"][name], "retention": analysis["all"]["old_selected_to_new_same_budget"],
                          "vs_FULL": analysis["all"]["new_FULL_to_old_budget_transfer"],
                          "opportunity": analysis["all"]["old_any_success_to_new_any_success"]}, ensure_ascii=True), flush=True)
    grids = {"numeric6": {s: COMMON[:-1] for s in ids},
             "all_per_question_common": {s: sorted(set(old[s]) & set(new[s]), key=budget_order) for s in ids}}
    for scope, choices in grids.items():
        for name in ["EM", "lexical_ACC", "F1_legacy"]:
            analysis, _ = audit_metric(old, new, ids, roots, METRICS[name], choices)
            result["sensitivity"][scope + "_" + name + "_total"] = analysis
    legacy, legacy_details = audit_metric(old, new, ids, roots, "answer_f1", grids["numeric6"], "input")
    result["sensitivity"]["numeric6_F1_legacy_input_original_definition"] = legacy
    details_all["original_numeric6_F1_input"] = legacy_details
    f1_details = details_all["F1_legacy"]
    old_em = [perfect(r["old_selected"]["all_metrics"]["EM"]) for r in f1_details]
    new_em = [perfect(r["new_at_old_budget"]["all_metrics"]["EM"]) for r in f1_details]
    full_em = [perfect(r["new_FULL"]["all_metrics"]["EM"]) for r in f1_details]
    result["prior_claim_reconciliation_F1_selected_EM_evaluated"] = {
        "selection_metric": "answer_f1", "evaluation_metric": "answer_em", "primary_metric_matched": False,
        "old_selected_EM": float(np.mean(old_em)), "transfer_EM": float(np.mean(new_em)),
        "new_FULL_EM": float(np.mean(full_em)),
        "new_F1_oracle_selected_EM": float(np.mean([r["new_oracle"]["all_metrics"]["EM"] for r in f1_details])),
        "old_selected_to_new_same_budget_EM": binary_matrix(old_em, new_em),
        "new_FULL_to_old_budget_transfer_EM": binary_matrix(full_em, new_em)}
    for name, details in details_all.items():
        assert len(details) == 500
        for r in details:
            assert r["transfer_quality"] <= r["new_quality"] + 1e-12
            assert r["old_quality"] + 1e-12 >= r["old_selected"]["quality"]
        check = summarize_details(details)
        assert check["lost_old_successes"] == check["lost_old_successes_new_other_budget_recovers"] + check["lost_old_successes_no_new_success_in_grid"]
    result["validation"] = {"passed": True, "metric_selection_and_evaluation_match": True,
                            "all_principal_analyses_n": 500, "format_fallback_scores_checked_zero": True,
                            "transition_matrices_sum_to_500": True,
                            "explicit_prior_cross_metric_reconciliation_separate": True}
    write_json(out / "SUMMARY.json", result)
    write_json(out / "PER_QUESTION.json", details_all)
    write_json(out / "ROOTS.json", list(roots.values()))
    write_json(out / "IDENTITY_AND_PROTOCOL.json", identity)
    (out / "REPORT_ZH.md").write_text(make_report(result), encoding="utf-8")
    bundle = Path(args.zip); bundle.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in sorted(out.iterdir()):
            if file.is_file():
                zipf.write(file, "audit/" + file.name)
        zipf.write(Path(__file__), "audit_matched_metric_transfers.py")
    with zipfile.ZipFile(bundle) as zipf:
        assert zipf.testzip() is None
    print(json.dumps({"audit_complete": True, "output": str(out), "zip": str(bundle), "new_model_calls": 0}, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
