"""Compare matched 500 question IDs; descriptive cross-run, not a shared-root experiment."""
from __future__ import annotations

import gzip
import json
import math
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "outputs/hotpot_adaptive_k_full500_audit_20261002"
OLD = Path("C:/Users/RESEARCH_USER/Desktop/过程结果/hotpot_random500_v433_final_review.tar.gz")


def mean(rows, key):
    return statistics.mean(r[key] for r in rows)


def main():
    sys.path.insert(0, str(ROOT / "work/_deps"))
    import numpy as np
    current = json.loads((DEST / "per_question.json").read_text(encoding="utf-8"))
    current_by_id = {r["sample_id"]:r for r in current}
    with tarfile.open(OLD) as archive:
        with gzip.open(archive.extractfile("analysis/scored_records.jsonl.gz"),"rt",encoding="utf-8") as handle:
            old = [json.loads(line) for line in handle if line.strip()]
        old_summary = json.load(archive.extractfile("analysis/summary.json"))
    assert len(old) == 4180
    groups, by_id = defaultdict(dict), defaultdict(dict)
    for row in old:
        sid,group = row["sample_id"],row["condition_group"]
        assert sid not in groups[group]
        groups[group][sid] = row
        by_id[sid][group] = row
    assert set(groups["FULL"]) == set(current_by_id) and len(current_by_id) == 500
    common = sorted([g for g,rs in groups.items() if g != "FULL" and set(rs) == set(current_by_id)], key=int)
    def standardize(row):
        return {"sample_id":row["sample_id"],"em":row["answer_em"],"f1":row["answer_f1"],
            "acc_contains":row["answer_acc"],"total_tokens":row["standalone_llm_input_tokens"]+row["standalone_llm_output_tokens"],
            "search_calls":row["search_calls"],"evidence_tokens":row["visible_evidence_tokens_total"],
            "local_fallback":row["local_fallback"]}
    def summary(rows):
        return {"n":len(rows),"EM":mean(rows,"em"),"F1":mean(rows,"f1"),"ACC_contains":mean(rows,"acc_contains"),
            "mean_total_tokens":mean(rows,"total_tokens"),"mean_search_calls":mean(rows,"search_calls"),
            "mean_evidence_tokens":mean(rows,"evidence_tokens")}
    table = []
    for group in common + ["FULL"]:
        table.append({"condition":group,**summary([standardize(r) for r in groups[group].values()])})
    table.append({"condition":"Adaptive-k",**summary(current)})
    best_fixed = max([r for r in table if r["condition"] in common],key=lambda r:(r["F1"],-r["mean_total_tokens"]))
    # Old public Oracle chooses independently for each problem after observing answers.
    oracle = {}
    for quality in ("answer_em","answer_f1","answer_acc"):
        chosen = [min((by_id[sid][g] for g in common),key=lambda r:(-r[quality],
            r["standalone_llm_input_tokens"]+r["standalone_llm_output_tokens"],int(r["condition_group"])))
            for sid in current_by_id]
        oracle[quality] = {**summary([standardize(r) for r in chosen]),
            "selected_budget_distribution":dict(Counter(r["condition_group"] for r in chosen))}
    pairs = {}
    baseline_names = list(dict.fromkeys(["FULL",best_fixed["condition"],"500","600"]))
    rng = np.random.default_rng(2026)
    for baseline in baseline_names:
        if baseline not in common and baseline != "FULL":
            continue
        base = [standardize(groups[baseline][r["sample_id"]]) for r in current]
        deltas = np.array([[r["em"]-b["em"],r["f1"]-b["f1"],r["acc_contains"]-b["acc_contains"],
            r["local_total_tokens"]-b["total_tokens"]] for r,b in zip(current,base)])
        bootstrap = []
        for _ in range(20):
            indices = rng.integers(0,500,size=(1000,500))
            bootstrap.append(deltas[indices].mean(axis=1))
        boot = np.concatenate(bootstrap,axis=0)
        result = {"n":500,"bootstrap_resamples":20000,"deltas":{}}
        for index,key in enumerate(("em","f1","acc_contains","total_tokens")):
            info = {"mean_current_minus_baseline":float(deltas[:,index].mean()),
                "paired_bootstrap_95ci":np.quantile(boot[:,index],[.025,.975]).tolist()}
            if key in ("em","acc_contains"):
                gains = int((deltas[:,index] > 0).sum()); losses = int((deltas[:,index] < 0).sum())
                n = gains + losses
                p = min(1,2*sum(math.comb(n,k) for k in range(min(gains,losses)+1))/2**n) if n else 1
                info.update(gains=gains,losses=losses,unchanged=500-n,exact_mcnemar_p=p)
            result["deltas"][key] = info
        pairs[baseline] = result
    # First query was generated again: measure this confound rather than assume identity.
    changed, unchanged = [],[]
    with gzip.open(DEST / "full500_uploaded_r1_t0/roots/roots.merged.jsonl.gz","rt",encoding="utf-8") as handle:
        for line in handle:
            root = json.loads(line)
            old_query = groups["FULL"][root["sample_id"]]["per_search_round_evidence"][0]["query"]
            record = {"sample_id":root["sample_id"],"old_query":old_query,"new_query":root["query"]}
            (unchanged if old_query == root["query"] else changed).append(record)
    query_groups = {}
    for name,rs in (("exact_query_same",unchanged),("query_changed",changed)):
        chosen = [current_by_id[r["sample_id"]] for r in rs]
        base = [standardize(groups["FULL"][r["sample_id"]]) for r in rs]
        query_groups[name] = {"n":len(rs),"current":summary(chosen),"old_FULL":summary(base)}
    report = {"common_fixed_budget_groups":common,"all_group_counts":{g:len(v) for g,v in groups.items()},
        "table":table,"best_fixed_by_F1":best_fixed,"old_public_oracles":oracle,"paired_comparisons":pairs,
        "first_query_comparison":query_groups,"changed_query_pairs":changed,
        "old_fallback_counts_by_group":{g:sum(r["local_fallback"] for r in rs.values()) for g,rs in groups.items()},
        "new_fallback_count":sum(r["answer_status"] == "local_fallback" for r in current),
        "same_500_IDs":True,"new_total_token_local_mean":mean(current,"local_total_tokens"),
        "new_total_token_server_mean":mean(current,"total_tokens"),
        "methodological_scope":["Same 500 IDs but different runs; no shared first-search checkpoints.",
            "Only fixed caps observed for all 500 are compared; larger incomplete groups are not mixed in.",
            "Old per-question input+output uses local counting. Current local counting is used for paired cost differences.",
            "Public Oracle is hindsight on old cap trajectories, not a deployed controller and not a universal upper bound on new candidate contexts.",
            "Bootstrap intervals/McNemar p are exploratory unadjusted comparisons, not a preregistered superiority or equivalence test."]}
    (DEST / "COMPARISON_WITH_OLD_BUDGET.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines = ["# Adaptive-k全量500题与原预算暴露实验对照", "",
        "结论：有小幅质量点估计提升，但没有已证实的稳定整体质量优势或质量—成本Pareto改善；成本明显更高。相对500/600的ACC有探索性正信号，不能忽略，也不能推广为所有指标或相对FULL的显著提升。", "",
        "两批使用同500个题号。本表固定组必须每组含500题；更大但题目不齐的预算不进入比较。", "",
        "| 条件（每轮证据） | 题数 | EM | F1 | ACC_contains | 平均总token | 平均搜索次数 |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['condition']} | {r['n']} | {r['EM']:.4f} | {r['F1']:.4f} | {r['ACC_contains']:.4f} | {r['mean_total_tokens']:,.1f} | {r['mean_search_calls']:.3f} |")
    lines += ["", "FULL是原Top-5完整证据，每轮长度实际变化；不是固定1500-token上限。Adaptive-k每轮完整前k篇，不设整题累计硬预算。",
        "总token包括首轮query生成与所有后续模型输入/输出，不仅是READ evidence token。", "",
        "## 与FULL及最佳固定预算的配对差异", "",
        "| 对照 | EM差及95%CI | F1差及95%CI | ACC差及95%CI | 总token差及95%CI |", "|---|---|---|---|---|"]
    for name,result in pairs.items():
        parts = []
        for key in ("em","f1","acc_contains","total_tokens"):
            info = result["deltas"][key]
            lo,hi = info["paired_bootstrap_95ci"]
            fmt = ",.1f" if key == "total_tokens" else ".4f"
            parts.append(f"{info['mean_current_minus_baseline']:{fmt}} [{lo:{fmt}}, {hi:{fmt}}]")
        lines.append("| "+name+" | "+" | ".join(parts)+" |")
    lines += ["", "95%区间使用按题配对bootstrap、20000次重采样；区间跨0不能证明显著提升，也不能证明完全等效。比较为探索性、未经多重比较校正。", "",
        "相对600：ACC差+0.032，未校正95%CI为[0.004,0.062]，McNemar p=0.044；相对500的ACC也有名义正信号。但这是包含匹配而非语义Judge，且跨运行query漂移、事后比较和多重检验均限制其解释。相对FULL的EM/F1/ACC区间全部跨0。", "",
        "从均值看Adaptive-k不构成对FULL/600的Pareto改善：质量略高但成本更高；也不能反过来说FULL严格支配Adaptive-k，因为质量点估计并不更高。", "",
        "## 原公共预算Oracle", "",
        "各题事后在全部500题共同的固定cap条件中选择质量最高、平局最低总cost。不可拿它当已实现方法。"]
    for quality,r in oracle.items():
        lines.append(f"- 按{quality}选择：EM={r['EM']:.4f}、F1={r['F1']:.4f}、ACC_contains={r['ACC_contains']:.4f}、总token={r['mean_total_tokens']:,.1f}。")
    lines += ["", "## 比较边界", "",
        f"- 首次query完全相同{len(unchanged)}/500题；改变{len(changed)}/500题，不是仅改变证据选择的严格因果消融。",
        "- 框架Prompt与动作逻辑保持，但旧组Top-5、新组Top-1000选k、根重生成、服务并发和记账差异必须披露。",
        f"- 本次server均值{mean(current,'total_tokens'):,.3f}，按旧本地口径为{mean(current,'local_total_tokens'):,.3f}；本批记账差异极小，不能解释主要成本增幅。",
        "- k最小6来自1篇基础+5篇buffer；相较旧Top-5，算法通常给更多证据，所以本适配未必产生原论文相对大量上下文的节省效果。",
        "- 不把500题不同cap的4180条轨迹简单混合均值当公平基线。",
        "- 当前结果不推翻原论文，也不证明预算控制无效；它检验的是在本Agent、E5/wiki18候选分布上的per-search Adaptive-k适配。", ""]
    (DEST / "COMPARISON_WITH_OLD_BUDGET_ZH.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"table":table,"oracles":oracle,"pairs":pairs,
        "first_query_group_sizes":{k:v["n"] for k,v in query_groups.items()}},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
