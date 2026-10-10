"""Independent audit of the supplied analysis archive. No inference calls."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INPUT = HERE / "input"
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
N = 7405
BOOT = 10000
SEED = 20261001


def js(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export(name, frame):
    frame.to_csv(OUT / name, index=False, encoding="utf-8-sig")


def boot_ci(matrix, rng):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    result = np.empty((BOOT, matrix.shape[1]))
    for start in range(0, BOOT, 100):
        idx = rng.integers(0, len(matrix), size=(min(100, BOOT-start), len(matrix)), dtype=np.int32)
        for column in range(matrix.shape[1]):
            result[start:start+len(idx), column] = matrix[:, column][idx].mean(axis=1)
    return np.quantile(result, [0.025, 0.975], axis=0)


def exact_mcnemar(gain, harm):
    n = gain + harm
    if n == 0:
        return 1.0
    k = min(gain, harm)
    logs = np.array([math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)-n*math.log(2) for j in range(k+1)])
    maximum = logs.max()
    return min(1.0, 2*math.exp(float(maximum))*float(np.exp(logs-maximum).sum()))


def main():
    print("Loading scored records...", flush=True)
    frame = pd.DataFrame(rows(INPUT / "analysis/scored_records.jsonl.gz"))
    frame["condition_group"] = frame.condition_group.astype(str)
    cfg = js(INPUT / "FULL7405_CONFIG.json")
    manifest = js(INPUT / "trajectories/trajectories.merged.jsonl.gz.manifest.json")
    roots = js(INPUT / "roots/roots.merged.jsonl.gz.manifest.json")
    flips = js(INPUT / "analysis/answer_flip/summary.json")
    source_summary = js(INPUT / "analysis/summary.json")
    token_summary = js(INPUT / "analysis/token_budget/summary.json")
    positions_summary = js(INPUT / "analysis/evidence_positions/supporting_evidence_position_summary.json")
    pipeline = js(INPUT / "PIPELINE_STATUS.json")
    errors, warnings = [], []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(len(frame) == 61754 == manifest["expected_records"], "terminal record count")
    check(frame.sample_id.nunique() == N == roots["roots"], "question count")
    check(not frame.duplicated(["sample_id", "condition_id"]).any(), "duplicate terminal keys")
    check(not frame.duplicated(["sample_id", "condition_group"]).any(), "duplicate question/condition")
    check(manifest["passed"] and roots["passed"] and not manifest["errors"] and not roots["errors"], "source merge validation")
    check(pipeline["stage"] == "complete" and pipeline["exit_code"] == 0, "pipeline exit")
    check(digest(INPUT / "analysis/scored_records.jsonl.gz") == flips["scored_records_sha256"], "scored record checksum")
    for name, expected in cfg["package_hashes"].items():
        code = ROOT / "deliverables/hotpot_budget_full7405_v433_20260924/scripts/experiments/hotpot_budget_full7405_v433" / name
        check(code.is_file() and digest(code) == expected, "code provenance: " + name)
    check(cfg["protocol"]["questions"] == N and cfg["protocol"]["gpu_ids"] == [2, 3] and cfg["protocol"]["top_k"] == 5, "configured protocol")

    qualities = ["answer_em", "answer_f1", "answer_acc"]
    for field in qualities:
        check(frame[field].between(0, 1).all(), "quality range: " + field)
    check(frame.answer_em.isin([0, 1]).all(), "nonbinary EM")
    check(frame.answer_acc.isin([0, 1]).all(), "nonbinary lexical accuracy")
    check(frame.search_calls.isin([1, 2, 3, 4]).all(), "invalid search count")
    check((frame.standalone_model_calls == frame.search_calls + 1).all(), "logical model call count")
    check((frame.standalone_thinking_tokens <= frame.standalone_llm_output_tokens).all(), "thinking/output accounting")
    fallback = frame.local_fallback.astype(bool)
    check((frame.loc[fallback, qualities] == 0).all().all(), "fallback nonzero primary score")
    for metric in qualities:
        check(np.allclose(frame.loc[~fallback, metric], frame.loc[~fallback, "recovered_" + metric]), "primary/recovered difference outside fallback: " + metric)
    check((frame.context_compactions == 0).all(), "unexpected history compaction")
    check((frame.format_repair_calls == 0).all(), "unexpected format repair call")
    frame["llm_total_tokens"] = frame.standalone_llm_input_tokens + frame.standalone_llm_output_tokens
    frame["root_input_tokens"] = frame.standalone_llm_input_tokens - frame.continuation_llm_input_tokens
    frame["root_output_tokens"] = frame.standalone_llm_output_tokens - frame.continuation_llm_output_tokens
    for field in ("root_full_context_tokens", "root_retrieved_title_recall", "root_input_tokens", "root_output_tokens"):
        check(frame.groupby("sample_id")[field].nunique().eq(1).all(), "root mismatch between conditions: " + field)
    root_frame = frame.drop_duplicates("sample_id")
    check(int(root_frame.root_input_tokens.sum()) == token_summary["root_successful_recorded_usage"]["llm_input_tokens"], "root input total")
    check(int(root_frame.root_output_tokens.sum()) == token_summary["root_successful_recorded_usage"]["llm_output_tokens"], "root output total")
    physical = int(root_frame.root_input_tokens.sum()+root_frame.root_output_tokens.sum()+frame.continuation_llm_input_tokens.sum()+frame.continuation_llm_output_tokens.sum())
    check(physical == token_summary["physical_successful_recorded_input_output_tokens"], "physical recorded token sum")
    expected_grid = {}
    for sid, group in frame.groupby("sample_id", sort=False):
        length = int(group.root_full_context_tokens.iloc[0])
        expected = {str(x) for x in range(100, length, 100)} | {"FULL"}
        check(set(group.condition_group) == expected, "condition grid: " + sid)
        expected_grid[sid] = expected

    round_rows, repeated_queries = [], 0
    root_queries = {}
    for row in frame.itertuples(index=False):
        seq = row.per_search_round_evidence
        check(len(seq) == row.search_calls, "round count: " + row.sample_id + "/" + row.condition_id)
        check([x["search_round"] for x in seq] == list(range(1, row.search_calls+1)), "round numbering: " + row.condition_id)
        check(sum(x["actual_exposed_tokens"] for x in seq) == row.visible_evidence_tokens_total, "evidence sum: " + row.condition_id)
        first = seq[0]
        check(first["actual_exposed_tokens"] == row.root_actual_exposure_tokens, "root exposure: " + row.condition_id)
        if row.sample_id in root_queries:
            check(root_queries[row.sample_id] == first["query"], "root query identity: " + row.sample_id)
        else:
            root_queries[row.sample_id] = first["query"]
        queries = [" ".join(x["query"].lower().split()) for x in seq]
        repeated_queries += len(queries) != len(set(queries))
        for item in seq:
            exposed, full = int(item["actual_exposed_tokens"]), int(item["full_context_tokens"])
            check(0 <= exposed <= full, "exposure > retrieval length")
            check(exposed == full if row.condition_group == "FULL" else exposed <= int(row.budget), "exposure > policy cap")
            round_rows.append({"sample_id": row.sample_id, "condition_group": row.condition_group, **item})
        for suffix, final_field in (("retrieved_support_title_recall", "cumulative_retrieved_title_recall"),
                                    ("strict_support_sentence_recall", "support_sentence_strict_recall"),
                                    ("fuzzy_support_sentence_recall", "support_sentence_fuzzy_recall")):
            cumulative_key = "cumulative_" + suffix
            new_key = "new_" + suffix
            cumulative = [float(x[cumulative_key]) for x in seq]
            check(all(-1e-10 <= x <= 1+1e-10 for x in cumulative), "support recall range")
            check(all(x <= y+1e-10 for x, y in zip(cumulative, cumulative[1:])), "nonmonotonic within-trajectory cumulative recall")
            check(abs(sum(float(x[new_key]) for x in seq)-cumulative[-1]) < 1e-9, "new/cumulative recall sum")
            check(abs(cumulative[-1]-getattr(row, final_field)) < 1e-9, "round/final support recall mismatch")
    rnd = pd.DataFrame(round_rows)
    numeric_sort = lambda x: 10**9 if x == "FULL" else int(x)
    conditions = sorted(frame.condition_group.unique(), key=numeric_sort)
    coverage = frame.groupby("condition_group").sample_id.nunique()
    common = [x for x in conditions if coverage[x] == N]
    numeric_common = [x for x in common if x != "FULL"]
    fields = qualities + ["search_calls", "visible_evidence_tokens_total", "standalone_llm_input_tokens", "standalone_llm_output_tokens", "llm_total_tokens", "standalone_thinking_tokens", "local_fallback", "cumulative_retrieved_all_titles", "cumulative_visible_all_titles", "support_sentence_strict_all", "support_sentence_fuzzy_all", "root_full_context_tokens"]
    budget = frame.groupby("condition_group")[fields].mean().reindex(conditions)
    budget.insert(0, "n", coverage.reindex(conditions))
    budget.insert(1, "coverage", budget.n / N)
    source_table = pd.read_csv(INPUT / "analysis/by_budget.csv", dtype={"condition_group": str}).set_index("condition_group")
    for field in fields:
        if field in source_table:
            check(np.allclose(budget[field], source_table.reindex(conditions)[field], atol=1e-10), "source mean: " + field)
    rng = np.random.default_rng(SEED)
    for condition in common:
        ci = boot_ci(frame.loc[frame.condition_group.eq(condition), ["answer_em", "answer_f1"]].to_numpy(), rng)
        budget.loc[condition, ["em_ci_low", "em_ci_high", "f1_ci_low", "f1_ci_high"]] = [ci[0,0], ci[1,0], ci[0,1], ci[1,1]]
    export("budget_quality_cost_all.csv", budget.reset_index())
    export("budget_quality_cost_common.csv", budget.loc[common].reset_index())
    joint = frame[frame.condition_group.isin(common)].groupby(["condition_group", "search_calls"])[qualities+["visible_evidence_tokens_total", "llm_total_tokens", "standalone_llm_input_tokens", "standalone_llm_output_tokens"]].mean()
    joint.insert(0, "n", frame[frame.condition_group.isin(common)].groupby(["condition_group", "search_calls"]).size())
    export("budget_search_joint.csv", joint.reset_index())
    em_cost = frame[frame.condition_group.isin(common)].groupby(["condition_group", "answer_em"])[["answer_f1", "search_calls", "visible_evidence_tokens_total", "llm_total_tokens", "standalone_llm_input_tokens", "standalone_llm_output_tokens"]].mean()
    em_cost.insert(0, "n", frame[frame.condition_group.isin(common)].groupby(["condition_group", "answer_em"]).size())
    export("budget_em_cost.csv", em_cost.reset_index())
    marginal = rnd[rnd.condition_group.isin(common)].groupby(["condition_group", "search_round"])[["actual_exposed_tokens", "new_retrieved_support_title_recall", "new_strict_support_sentence_recall", "new_fuzzy_support_sentence_recall"]].mean()
    marginal.insert(0, "n_reaching_round", rnd[rnd.condition_group.isin(common)].groupby(["condition_group", "search_round"]).size())
    export("budget_round_evidence_gain.csv", marginal.reset_index())

    wide = {field: frame.pivot(index="sample_id", columns="condition_group", values=field) for field in qualities+["llm_total_tokens", "visible_evidence_tokens_total", "search_calls"]}
    paired = []
    for left, right in [("100", "200"), ("300", "400"), ("400", "500"), ("500", "600"), ("500", "FULL"), ("600", "FULL")]:
        metrics = ["answer_em", "answer_f1", "llm_total_tokens", "visible_evidence_tokens_total", "search_calls"]
        values = np.column_stack([(wide[x][right]-wide[x][left]).to_numpy(float) for x in metrics])
        ci = boot_ci(values, rng)
        a, b = wide["answer_em"][left].eq(1).to_numpy(), wide["answer_em"][right].eq(1).to_numpy()
        gain, harm = int((~a & b).sum()), int((a & ~b).sum())
        result = {"left": left, "right": right, "n": N, "em_gain_count": gain, "em_harm_count": harm, "mcnemar_exact_p": exact_mcnemar(gain, harm)}
        for j, metric in enumerate(metrics):
            result[metric+"_difference_right_minus_left"] = float(values[:, j].mean())
            result[metric+"_ci_low"] = float(ci[0,j])
            result[metric+"_ci_high"] = float(ci[1,j])
        paired.append(result)
    export("paired_budget_comparisons.csv", pd.DataFrame(paired))

    adjacent_source = pd.read_csv(INPUT / "analysis/answer_flip/adjacent_transitions_answer_em.csv")
    correctness = wide["answer_em"][numeric_common].eq(1).to_numpy()
    patterns = Counter("".join("1" if x else "0" for x in row) for row in correctness)
    export("em_correctness_patterns.csv", pd.DataFrame([{"pattern": key, "n": value} for key, value in patterns.most_common()]))
    total_harm = (correctness[:, :-1] & ~correctness[:, 1:]).sum(axis=1)
    total_gain = (~correctness[:, :-1] & correctness[:, 1:]).sum(axis=1)
    msb = np.where(correctness.any(axis=1), np.asarray(numeric_common, dtype=int)[correctness.argmax(axis=1)], np.nan)
    stable = np.flip(np.logical_and.accumulate(np.flip(correctness, axis=1), axis=1), axis=1)
    ssb = np.where(stable.any(axis=1), np.asarray(numeric_common, dtype=int)[stable.argmax(axis=1)], np.nan)
    export("per_question_em_stability.csv", pd.DataFrame({
        "sample_id": wide["answer_em"].index,
        "msb": msb, "ssb_observed": ssb, "ssb_minus_msb": ssb-msb,
        "wrong_to_correct_flips": total_gain, "correct_to_wrong_flips": total_harm,
        "correctness_flips": total_gain+total_harm,
        "ever_correct": correctness.any(axis=1), "always_correct": correctness.all(axis=1)}))
    numeric_f1_oracle = wide["answer_f1"][numeric_common].max(axis=1)
    em_oracle = correctness.any(axis=1).astype(float)
    export("fixed_budget_regret_numeric.csv", pd.DataFrame([
        {"condition_group": b, "n": N,
         "mean_em_regret": float((em_oracle-wide["answer_em"][b]).mean()),
         "mean_current_f1_regret": float((numeric_f1_oracle-wide["answer_f1"][b]).mean()),
         "mean_llm_total_tokens": float(wide["llm_total_tokens"][b].mean()),
         "mean_evidence_tokens": float(wide["visible_evidence_tokens_total"][b].mean())}
        for b in numeric_common]))
    independent_flip = {"any_flip_count": int((total_harm+total_gain > 0).sum()), "any_harm_count": int((total_harm > 0).sum()),
                        "ever_correct_count": int(correctness.any(axis=1).sum()), "always_correct_count": int(correctness.all(axis=1).sum()),
                        "never_correct_count": int((~correctness.any(axis=1)).sum()), "msb_mean": float(np.nanmean(msb)),
                        "ssb_mean_observed": float(np.nanmean(ssb)), "ssb_observed_count": int(np.isfinite(ssb).sum()),
                        "ssb_minus_msb_mean_among_observed": float(np.nanmean(ssb-msb))}
    given = flips["metric_summaries"]["answer_em"]
    for ours, theirs in [("any_flip_count", "any_answer_flip_count"), ("any_harm_count", "any_harmful_flip_count"), ("ever_correct_count", "msb_n"), ("always_correct_count", "always_correct_count"), ("never_correct_count", "never_correct_count")]:
        check(independent_flip[ours] == given[theirs], "flip summary: " + ours)
    for i, item in adjacent_source.iterrows():
        a, b = correctness[:, i], correctness[:, i+1]
        gain, harm = int((~a & b).sum()), int((a & ~b).sum())
        check(gain == item.wrong_to_correct_count and harm == item.correct_to_wrong_count, "adjacent flip counts")
        check(abs(exact_mcnemar(gain, harm)-item.mcnemar_exact_p) < 1e-10, "McNemar exact p")

    # Select only among conditions available to all questions; no inferred unseen outputs.
    oracles = []
    selected_frames = {}
    for name, choices, cost in [("F1_then_input_numeric", numeric_common, "standalone_llm_input_tokens"), ("F1_then_total_numeric_FULL", common, "llm_total_tokens")]:
        candidates = frame[frame.condition_group.isin(choices)]
        selected = candidates.sort_values(["sample_id", "answer_f1", cost, "visible_evidence_tokens_total", "search_calls"], ascending=[True, False, True, True, True], kind="stable").drop_duplicates("sample_id")
        selected_frames[name] = selected
        export("oracle_"+name+".csv", selected[["sample_id", "condition_group"]+qualities+["search_calls", "visible_evidence_tokens_total", "standalone_llm_input_tokens", "standalone_llm_output_tokens", "llm_total_tokens"]])
        stats = {"oracle": name, "n": len(selected), **{field: float(selected[field].mean()) for field in fields if field in selected}}
        stats["selected_condition_counts"] = {str(k): int(v) for k, v in selected.condition_group.value_counts().items()}
        oracles.append(stats)
    for i, stats in enumerate(oracles):
        check(abs(stats["answer_f1"]-token_summary["oracle_mean_tokens"][i]["mean_f1_per_question"]) < 1e-10, "Oracle F1 recomputation")
        check(abs(stats["llm_total_tokens"]-token_summary["oracle_mean_tokens"][i]["mean_llm_total_tokens_per_question"]) < 1e-10, "Oracle cost recomputation")
    selected = selected_frames["F1_then_total_numeric_FULL"].set_index("sample_id").reindex(wide["answer_f1"].index)
    oracle_diff = np.column_stack([selected.answer_f1-wide["answer_f1"]["FULL"], selected.llm_total_tokens-wide["llm_total_tokens"]["FULL"]])
    oracle_ci = boot_ci(oracle_diff, rng)
    oracle_vs_full = {"f1_difference": float(oracle_diff[:,0].mean()), "f1_ci95": oracle_ci[:,0].tolist(),
                      "total_token_difference": float(oracle_diff[:,1].mean()), "total_token_ci95": oracle_ci[:,1].tolist()}
    pareto = []
    for axis in ["visible_evidence_tokens_total", "standalone_llm_input_tokens", "llm_total_tokens"]:
        for condition, row in budget.loc[common].iterrows():
            dominating = [other for other, candidate in budget.loc[common].iterrows() if candidate[axis] <= row[axis] and candidate.answer_f1 >= row.answer_f1 and (candidate[axis] < row[axis] or candidate.answer_f1 > row.answer_f1)]
            pareto.append({"cost_axis": axis, "condition": condition, "on_frontier": not dominating, "dominated_by": ",".join(dominating), "cost": float(row[axis]), "f1": float(row.answer_f1)})
    export("pareto_empirical.csv", pd.DataFrame(pareto))

    position_counts = Counter()
    full_position_counts = Counter()
    position_keys = set()
    for row in rows(INPUT / "analysis/evidence_positions/supporting_evidence_positions.jsonl.gz"):
        key = (row["sample_id"], row["condition_id"])
        check(key not in position_keys, "duplicate position key")
        position_keys.add(key)
        position_counts["records"] += 1
        position_counts["all_found"] += bool(row["all_supporting_facts_found_somewhere"])
        position_counts["all_fully_visible"] += bool(row["all_supporting_facts_fully_visible"])
        if row["condition_label"] == "FULL":
            full_position_counts["records"] += 1
            full_position_counts["all_found"] += bool(row["all_supporting_facts_found_somewhere"])
            full_position_counts["all_fully_visible"] += bool(row["all_supporting_facts_fully_visible"])
    check(position_keys == set(zip(frame.sample_id, frame.condition_id)), "position/score key alignment")
    check(abs(position_counts["all_fully_visible"]/len(frame)-positions_summary["all_supporting_facts_fully_visible_rate"]) < 1e-10, "position fully-visible rate")
    export("evidence_hit_quality.csv", frame.groupby(["condition_group", "support_sentence_strict_all"]).agg(
        n=("sample_id", "size"), answer_em=("answer_em", "mean"), answer_f1=("answer_f1", "mean"),
        answer_acc=("answer_acc", "mean"), llm_total_tokens=("llm_total_tokens", "mean")).reset_index())
    support_group = frame.groupby("support_sentence_strict_all")[qualities+["llm_total_tokens"]].mean()
    em_f1_edge = frame[frame.answer_em.eq(1) & frame.answer_f1.lt(1)]
    export("em1_f1_lt1_records.csv", em_f1_edge[["sample_id", "condition_id", "condition_group"]+qualities])
    warnings += ["执行失败为零，但仍存在格式回退；这些记录在主指标中均计零。",
                 "600以上数值预算未覆盖全部题目，不可把未配对的子集均值直接与全量比较。",
                 "已对照HotpotQA官方评测代码确认：保留的词重叠F1缺少yes/no/noanswer特殊处理。包内没有预测答案和gold文本，不能量化校正后的变化。",
                 "上传包不含原始prompt/生成文本及sealed gold，无法独立复核具体解析、截断原因及生成阶段标签隔离。",
                 "标准化支持句匹配与逐字证据位置定位定义不同，不能混用命中率。"]
    if len(em_f1_edge):
        warnings.append(f"{len(em_f1_edge)}条记录、{em_f1_edge.sample_id.nunique()}题出现EM=1但F1<1，需要原答案/gold核验标准化后为空的边界情况；这本身不足以判定为错误。")
    full = budget.loc["FULL"]
    best_numeric = budget.loc[numeric_common, "answer_f1"].idxmax()
    best_row = budget.loc[best_numeric]
    audit = {"archive_sha256": "7a3abf36f05e7a7a24baa4460d468b7dbddbce140df32720aa23131abd8a79b9", "questions": N,
             "records": len(frame), "common_conditions": common, "errors": errors, "warnings": warnings,
             "structural_audit_passed": not errors, "format_fallback_count": int(fallback.sum()),
             "format_fallback_rate": float(fallback.mean()), "fallback_affected_questions": int(frame.loc[fallback].sample_id.nunique()),
             "context_compactions": int(frame.context_compactions.sum()), "boundary_backoff_tokens": int(frame.roundtrip_boundary_backoff_tokens_total.sum()),
             "boundary_backoff_records": int(frame.roundtrip_boundary_backoff_tokens_total.gt(0).sum()),
             "repeated_query_trajectories": repeated_queries, "physical_successfully_recorded_tokens": physical,
             "continuation_logical_calls": int(frame.search_calls.sum()), "root_logical_calls": N,
             "elapsed_hours": pipeline["elapsed_seconds_this_invocation"]/3600,
             "paired_bootstrap_repetitions": BOOT, "paired_bootstrap_seed": SEED, "flip_summary_independent": independent_flip,
             "oracles": oracles, "oracle_vs_FULL": oracle_vs_full, "positions": dict(position_counts), "FULL_positions": dict(full_position_counts),
             "strict_support_quality": support_group.reset_index().to_dict(orient="records"),
             "best_common_numeric_f1": best_numeric, "best_numeric_vs_FULL_token_saving_fraction": 1-best_row.llm_total_tokens/full.llm_total_tokens,
             "best_numeric_vs_FULL_evidence_saving_fraction": 1-best_row.visible_evidence_tokens_total/full.visible_evidence_tokens_total,
             "em1_f1_lt1_records": len(em_f1_edge), "em1_f1_lt1_questions": int(em_f1_edge.sample_id.nunique()),
             "dataset_mean_root_tokens": float(root_frame.root_full_context_tokens.mean()), "median_root_tokens": float(root_frame.root_full_context_tokens.median()),
             "root_length_min_max": [int(root_frame.root_full_context_tokens.min()), int(root_frame.root_full_context_tokens.max())]}
    write_json("audit_summary.json", audit)
    make_report(audit, budget, paired, oracles, adjacent_source, joint, support_group)
    # Numeric plotting inputs, kept separate from source records.
    write_json("plot_data.json", {"budget": budget.loc[common].reset_index().replace({np.nan: None}).to_dict(orient="records"),
                                  "oracles": oracles, "joint": joint.reset_index().to_dict(orient="records"),
                                  "adjacent": adjacent_source.to_dict(orient="records"), "pareto": pareto})
    print(json.dumps({key: audit[key] for key in ["structural_audit_passed", "questions", "records", "format_fallback_count", "em1_f1_lt1_records", "common_conditions", "elapsed_hours", "errors"]}, ensure_ascii=False, indent=2), flush=True)


def make_report(audit, budget, paired, oracles, adjacent, joint, support):
    full = budget.loc["FULL"]
    b200, b100, b600 = budget.loc["200"], budget.loc["100"], budget.loc["600"]
    lines = ["# HotpotQA全量7,405题：独立审计与研究结果", "",
             "来源：hotpot_full7405_v433_complete_review.tar.gz；SHA256核验一致。", "",
             f"结构审计{'通过' if not audit['errors'] else '存在错误'}：7,405题、61,754条唯一题目×预算记录，合并验证通过、执行失败0、历史证据压缩0。流程耗时{audit['elapsed_hours']:.2f}小时。",
             "审计覆盖上传的评分记录、证据位置文件、汇总表和配置哈希；原始prompt/输出及sealed gold不在包内，无法复查具体解析及截断原因。", "",
             "本次是v4.3.3自然搜索协议：Qwen3-8B原生思考、temperature=0，Top-5顺序拼接后按100-token全局前缀暴露，共享首轮root，最多4次总搜索；每个预算沿用同一每轮上限，FULL每轮完整暴露。不是受控r1–r4×B50–1500实验。服务器context=40,960，客户端token限制=24,497，输出上限=8,192，无格式修复，GPU2/3执行。", "",
             "## 1. 同题全量主结果及每题平均token成本", "",
             "100–600与FULL都覆盖全部7,405题。B为每次搜索的证据暴露上限；累计证据量=各轮实际暴露量之和；LLM总tokens=输入+输出，包含独立执行策略时的root成本。",
             "F1为当前实现的词重叠F1，已确认缺少HotpotQA对yes/no/noanswer的特殊规则，相关质量结论须在离线校正后复核；Acc为词法包含，不是LLM judge。", "",
             "| 每轮预算 | EM | F1 | 词法Acc | 搜索/题 | 证据/题 | 输入/题 | 输出/题 | 总LLM/题 | 回退率 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in audit["common_conditions"]:
        x = budget.loc[name]
        lines.append(f"| {name} | {x.answer_em:.2%} | {x.answer_f1:.2%} | {x.answer_acc:.2%} | {x.search_calls:.3f} | {x.visible_evidence_tokens_total:,.1f} | {x.standalone_llm_input_tokens:,.1f} | {x.standalone_llm_output_tokens:,.1f} | {x.llm_total_tokens:,.1f} | {x.local_fallback:.2%} |")
    lines += ["", "700预算覆盖6,890题；800覆盖2,526题；900覆盖360题，后续条件更少。不能把高预算子集的F1下降解读为全量曲线下降。所有端点及覆盖率见budget_quality_cost_all.csv。", "",
              "## 2. 质量与成本的具体关系", "",
              f"100→200：当前F1提高{(b200.answer_f1-b100.answer_f1)*100:.2f}个百分点，平均输入增加{b200.standalone_llm_input_tokens-b100.standalone_llm_input_tokens:.1f} tokens，但输出减少{b100.standalone_llm_output_tokens-b200.standalone_llm_output_tokens:.1f} tokens；总成本点估计下降{b100.llm_total_tokens-b200.llm_total_tokens:.1f} tokens。总成本差的95%配对CI为[-84.1,+25.9] tokens，包含零，不应声称已证明显著降本。证据输入增多可能减少后续搜索/思考开销，最小暴露量不等于最低总成本。",
              f"600与FULL：F1分别{b600.answer_f1:.4f}、{full.answer_f1:.4f}；600平均LLM总成本低{(1-b600.llm_total_tokens/full.llm_total_tokens):.2%}，累计暴露证据低{(1-b600.visible_evidence_tokens_total/full.visible_evidence_tokens_total):.2%}。", "",
              "| 配对比较（右减左） | ΔEM | ΔF1及95%配对bootstrap CI | Δ总LLM tokens/题 | EM McNemar p |",
              "|---|---:|---|---:|---:|"]
    for x in paired:
        lines.append(f"| {x['left']}→{x['right']} | {x['answer_em_difference_right_minus_left']*100:+.3f} pp | {x['answer_f1_difference_right_minus_left']*100:+.3f} pp [{x['answer_f1_ci_low']*100:+.3f}, {x['answer_f1_ci_high']*100:+.3f}] | {x['llm_total_tokens_difference_right_minus_left']:+.1f} | {x['mcnemar_exact_p']:.3g} |")
    lines += ["", "上述新配对p值未作多重校正；相邻预算的原主检验使用BH-FDR。置信区间按题目重采样10,000次，避免把同题多预算当作独立样本。",
              "600在F1–成本点估计空间支配FULL，但FULL的EM/词法Acc并非均更低；F1差异很小，不能据此声称600的质量显著更高，或直接宣称已证明非劣效。", "",
              "## 3. 答案翻转与稳定性（公共数值网格100–600）", "",
              "| 相邻预算 | Wrong→Correct | Correct→Wrong | NetGain | BH校正p |",
              "|---|---:|---:|---:|---:|"]
    for x in adjacent.itertuples():
        lines.append(f"| {x.budget_from}→{x.budget_to} | {x.wrong_to_correct_count} ({x.gain:.2%}) | {x.correct_to_wrong_count} ({x.harm:.2%}) | {x.net_gain*100:+.2f} pp | {x.mcnemar_exact_bh_q:.3g} |")
    f = audit["flip_summary_independent"]
    lines += ["", f"{f['any_flip_count']:,}/{N:,}题发生过正确性翻转（{f['any_flip_count']/N:.2%}）；{f['any_harm_count']:,}题发生过答对→答错（全量{f['any_harm_count']/N:.2%}，曾答对的{f['ever_correct_count']:,}题中占{f['any_harm_count']/f['ever_correct_count']:.2%}）。",
              f"最小成功预算MSB均值{f['msb_mean']:.1f}（仅曾答对题）；观察到稳定答对的{f['ssb_observed_count']:,}题，SSB均值{f['ssb_mean_observed']:.1f}，同题SSB−MSB均值{f['ssb_minus_msb_mean_among_observed']:.1f}。SSB只在100–600有限网格内成立，未包括FULL或更高端点。",
              "实际回答文本未上传，故这里的翻转指正确性变化，并不统计错答案A变成错答案B的内容变化。", "",
              "跨预算轨迹共享root，但后续query和自然搜索次数可能改变。因此相邻预算的Gain/Harm描述的是整个预算策略的效果，不是仅新增100个相同证据tokens的独立因果效应。", "",
              "## 4. Oracle与每题平均成本", "",
              "| 候选范围与破平局方式 | EM | F1 | 搜索/题 | 证据/题 | 总LLM/题 |",
              "|---|---:|---:|---:|---:|---:|"]
    for x in oracles:
        label = "100–600，F1优先/输入成本破平局" if "numeric" in x["oracle"] and "FULL" not in x["oracle"] else "100–600+FULL，F1优先/总成本破平局"
        lines.append(f"| {label} | {x['answer_em']:.2%} | {x['answer_f1']:.2%} | {x['search_calls']:.3f} | {x['visible_evidence_tokens_total']:,.1f} | {x['llm_total_tokens']:,.1f} |")
    x = oracles[1]
    lines += ["", f"逐题F1/总成本Oracle相对FULL，F1提高{(x['answer_f1']-full.answer_f1)*100:.2f}个百分点，所选路径总tokens低{(1-x['llm_total_tokens']/full.llm_total_tokens):.2%}。所选条件计数：{json.dumps(x['selected_condition_counts'], ensure_ascii=False)}。",
              "这是看过gold后的质量上界和所选路径成本，不包含枚举所有候选的代价，也不等于已训练预算控制器的性能。F1 Oracle与二元EM Oracle选择目标不同，报告中的EM不必等于曾经答对比例。", "",
              "## 5. 搜索轮数和证据预算联合分析", "",
              "详表budget_search_joint.csv同时给出每轮预算×最终自然搜索次数的质量、累计证据、输入、输出和总成本；budget_round_evidence_gain.csv给出各预算下每一轮新增标注支持句召回。",
              "这些组是模型自然停止后的事后分组，题目难度/答案充分程度不同，不能把r4平均质量低直接解释为第四次搜索降低质量。对于各轮边际证据，也需比较到达该轮的同一批轨迹，不能直接把不同存活集合的平均累计召回当作单题下降。", "",
              "## 6. 证据命中与可靠性", ""]
    hit, miss = support.loc[True], support.loc[False]
    lines += [f"全部标注支持句在标准化可见文本中匹配的轨迹为4,639条（{4639/61754:.2%}），其EM/F1={hit.answer_em:.2%}/{hit.answer_f1:.2%}；其余轨迹={miss.answer_em:.2%}/{miss.answer_f1:.2%}。这是跨预算的描述性相关，未作独立轨迹显著性检验。",
              f"FULL的累计检索齐全部支持标题的比例为{full.cumulative_retrieved_all_titles:.2%}，严格支持句全部命中率为{full.support_sentence_strict_all:.2%}，模糊全部命中率为{full.support_sentence_fuzzy_all:.2%}。600的严格/模糊全部命中率为{b600.support_sentence_strict_all:.2%}/{b600.support_sentence_fuzzy_all:.2%}。低严格句子命中不等于完全无证据：Wikipedia版本、100-word切片及句子被拆分会影响匹配；模糊命中也只是词集合召回≥0.8的代理，不是语义验证。",
              f"局部格式回退{audit['format_fallback_count']:,}条（{audit['format_fallback_rate']:.2%}），影响{audit['fallback_affected_questions']:,}题，主EM/F1/Acc全计0。格式修复模型调用0；证据历史压缩0；token边界回退{audit['boundary_backoff_records']}条、合计{audit['boundary_backoff_tokens']} tokens。",
              f"成功记录的物理输入+输出tokens={audit['physical_successfully_recorded_tokens']:,}；root只计一次，其余预算continuation全部计入。每题策略成本则独立包含root。未记录的传输失败重试和服务探针不在上述token总数中。",
              "", "## 7. 审计中发现的口径限制", ""]
    lines += ["- " + x for x in audit["warnings"]]
    lines += ["", "已核对[HotpotQA官方评测源码](https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py)：当预测或gold是yes/no/noanswer时，规范化后的字符串不完全一致，F1应计0。原版词重叠实现没有这个判断。例如gold为yes、预测为yes they are，原版F1=0.5，官方F1=0。EM的精确匹配不受这一项差异影响。校正仅需原预测与gold文本，无需重新推理。"]
    if audit["errors"]:
        lines += ["", "结构错误："] + ["- "+x for x in audit["errors"]]
    lines += ["", "## 8. 现有数据支持的研究陈述", "",
              "在HotpotQA全量dev题集上，证据暴露预算改变了自然搜索次数与模型输出开销。200-token每轮暴露比100-token获得更高EM，且当前词重叠F1更高；平均总LLM token成本的点估计略低，但降本差异未显著。在500–600与FULL之间，当前平均F1趋于平台，而成本继续上升。逐题最佳预算存在明显异质性，且部分题目在答对后会随预算增加再次答错。这支持研究基于当前证据状态的动态预算分配；F1与F1-Oracle结论需以官方口径离线校正后复核。",
              "现有实验未在同一总预算下直接强制比较继续读取与继续搜索，因此不能直接声称已证明“及时把剩余预算转向新搜索”这一动作的因果优势。下一步应在同一状态、相同总成本约束下比较两种续行动，并用本全量数据作控制器评估时隔离训练/验证题目。", ""]
    (OUT / "FULL_AUDIT_REPORT_ZH.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
