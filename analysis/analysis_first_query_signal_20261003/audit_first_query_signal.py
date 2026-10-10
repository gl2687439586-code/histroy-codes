#!/usr/bin/env python3
"""Read-only, question-disjoint offline test of first-generation token routing.

No inference, no source modification. EM is primary because the supplied F1
scorer is not the official HotpotQA yes/no scorer. Root length is a proxy for
tokens emitted through the first query, not a validated pre-query think count.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

BUDGETS = ["100", "200", "300", "400", "500", "600", "FULL"]
LAMBDAS = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
SEEDS = [20261003, 20261004, 20261005]


def export_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load(path):
    items = {}
    duplicates = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            key = r["sample_id"]
            root_out = int(r["standalone_llm_output_tokens"] - r["continuation_llm_output_tokens"])
            root_in = int(r["standalone_llm_input_tokens"] - r["continuation_llm_input_tokens"])
            root = (root_out, root_in)
            item = items.setdefault(key, {"root": root, "rows": {}})
            assert root == item["root"], (key, "root not invariant")
            b = str(r["condition_group"])
            if b in BUDGETS:
                duplicates += int(b in item["rows"])
                item["rows"][b] = r
    assert duplicates == 0
    ids = sorted(items)
    assert all(set(items[i]["rows"]) == set(BUDGETS) for i in ids)
    roots = np.array([items[i]["root"] for i in ids], dtype=float)
    mats = {}
    fields = {
        "em": "answer_em", "f1_supplied": "answer_f1",
        "input": "standalone_llm_input_tokens", "output": "standalone_llm_output_tokens",
        "searches": "search_calls", "evidence": "visible_evidence_tokens_total",
        "fallback": "local_fallback",
    }
    for name, field in fields.items():
        mats[name] = np.array([[items[i]["rows"][b][field] for b in BUDGETS] for i in ids], dtype=float)
    mats["total"] = mats["input"] + mats["output"]
    assert np.all(roots[:, 0] > 0)
    return ids, roots, mats


def best_action(quality, cost):
    # Lexicographic tie-break: maximum utility, then lower mean total tokens.
    optimum = np.max(quality)
    candidates = np.flatnonzero(np.isclose(quality, optimum, rtol=0, atol=1e-12))
    return int(candidates[np.argmin(cost[candidates])])


def fit(x, em, cost, lam, family):
    utility = em - lam * cost / 1000.0
    global_action = best_action(utility.mean(axis=0), cost.mean(axis=0))
    if family == "global":
        return {"family": family, "action": global_action}
    if family == "stump":
        model = {"family": "global", "action": global_action}
        best = utility[:, global_action].mean()
        best_cost = cost[:, global_action].mean()
        for threshold in np.unique(np.quantile(x, np.arange(0.05, 0.951, 0.025))):
            left = x <= threshold
            if min(left.sum(), (~left).sum()) < 200:
                continue
            actions = [best_action(utility[m].mean(axis=0), cost[m].mean(axis=0)) for m in [left, ~left]]
            sel = np.where(left, actions[0], actions[1])
            value = utility[np.arange(len(x)), sel].mean()
            tc = cost[np.arange(len(x)), sel].mean()
            if value > best + 1e-12 or (abs(value - best) < 1e-12 and tc < best_cost):
                best, best_cost = value, tc
                model = {"family": family, "threshold": float(threshold), "left_action": actions[0], "right_action": actions[1]}
        return model
    if family == "bins5":
        cuts = np.unique(np.quantile(x, [0.2, 0.4, 0.6, 0.8]))
        groups = np.searchsorted(cuts, x, side="left")
        actions = []
        for j in range(len(cuts) + 1):
            mask = groups == j
            actions.append(best_action(utility[mask].mean(axis=0), cost[mask].mean(axis=0)))
        return {"family": family, "cuts": cuts.tolist(), "actions": actions}
    raise ValueError(family)


def predict(model, x):
    if model["family"] == "global":
        return np.full(len(x), model["action"], dtype=int)
    if model["family"] == "stump":
        return np.where(x <= model["threshold"], model["left_action"], model["right_action"])
    return np.array(model["actions"])[np.searchsorted(model["cuts"], x, side="left")]


def folds(n, seed):
    assignment = np.empty(n, dtype=int)
    assignment[np.random.default_rng(seed).permutation(n)] = np.arange(n) % 5
    return assignment


def crossfit(x, mats, lam, family, assignment):
    n = len(x)
    actions = np.empty(n, dtype=int)
    rules = []
    random_mix_em = np.empty(n)
    random_mix_cost = np.empty(n)
    for fold in range(5):
        train, test = assignment != fold, assignment == fold
        model = fit(x[train], mats["em"][train], mats["total"][train], lam, family)
        actions[test] = predict(model, x[test])
        rules.append({"fold": fold, "train_n": int(train.sum()), "test_n": int(test.sum()), **model})
        # Independent-of-feature baseline with the same held-out budget proportions.
        # These weights use action counts, never quality labels to choose a policy.
        weights = np.bincount(actions[test], minlength=len(BUDGETS)) / test.sum()
        random_mix_em[test] = mats["em"][test] @ weights
        random_mix_cost[test] = mats["total"][test] @ weights
    return actions, rules, random_mix_em, random_mix_cost


def summary(mats, actions):
    idx = np.arange(len(actions))
    return {name: float(mat[idx, actions].mean()) for name, mat in mats.items()}


def ci_mean(diff, seed=20261003, n_boot=5000):
    rng = np.random.default_rng(seed)
    n = len(diff)
    values = []
    for start in range(0, n_boot, 100):
        batch = min(100, n_boot-start)
        ix = rng.integers(0, n, (batch, n))
        values.extend(diff[ix].mean(axis=1).tolist())
    return [float(v) for v in np.quantile(values, [0.025, 0.975])]


def compare(mats, actions, reference, label):
    idx = np.arange(len(actions))
    chosen_em = mats["em"][idx, actions]
    chosen_cost = mats["total"][idx, actions]
    ref_em = mats["em"][idx, reference]
    ref_cost = mats["total"][idx, reference]
    gain = int(((chosen_em == 1) & (ref_em == 0)).sum())
    harm = int(((chosen_em == 0) & (ref_em == 1)).sum())
    return {
        "reference": label,
        "em_delta": float((chosen_em-ref_em).mean()),
        "em_delta_ci95": ci_mean(chosen_em-ref_em),
        "total_token_delta": float((chosen_cost-ref_cost).mean()),
        "total_token_delta_ci95": ci_mean(chosen_cost-ref_cost),
        "cost_reduction_fraction": float(1-chosen_cost.mean()/ref_cost.mean()),
        "gain_questions": gain, "harm_questions": harm,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="audit_hotpot_full7405_v433_20261001/input/analysis/scored_records.jsonl.gz")
    ap.add_argument("--output", default="analysis_first_query_signal_20261003/results")
    ap.add_argument("--root-total-only", action="store_true", help="Input+output first-stage token signal sensitivity analysis")
    args = ap.parse_args()
    source = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    ids, roots, mats = load(source)
    n = len(ids)
    x = roots[:, 0]
    if args.root_total_only:
        sensitivity = []
        comparisons = []
        for seed in SEEDS:
            actions, rules, mix_em, mix_cost = crossfit(roots.sum(axis=1), mats, 0.0, "stump", folds(n, seed))
            sensitivity.append({"split_seed": seed, **summary(mats, actions), "rules": rules})
            if seed == SEEDS[0]:
                for label,j in [("FULL",6),("fixed500",4),("fixed600",5)]:
                    comparisons.append(compare(mats,actions,np.full(n,j),label))
                diff = mats["em"][np.arange(n),actions]-mix_em
                comparisons.append({"reference":"same_proportions_feature_independent_mix", "em_delta":float(diff.mean()), "em_delta_ci95":ci_mean(diff), "total_token_delta":float((mats["total"][np.arange(n),actions]-mix_cost).mean())})
        result = {"signal":"root input+output tokens", "samples":n, "sensitivity_not_primary":True, "evaluations":sensitivity, "comparisons":comparisons}
        (out / "root_total_sensitivity.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    diagnostics = []
    cuts = np.unique(np.quantile(x[x < 8192], [0.2, 0.4, 0.6, 0.8]))
    grp = np.searchsorted(cuts, x, side="left")
    groups = [(f"Q{j+1}", (grp == j) & (x < 8192)) for j in range(5)] + [("at_output_cap_8192", x >= 8192)]
    for label, mask in groups:
        for j, b in enumerate(BUDGETS):
            diagnostics.append({"signal_group": label, "n": int(mask.sum()), "root_output_min": float(x[mask].min()), "root_output_max": float(x[mask].max()), "budget": b, **{name: float(mat[mask,j].mean()) for name, mat in mats.items()}})
    export_csv(out / "signal_quintile_budget_results.csv", diagnostics)
    fixed = [{"policy": b, **summary(mats, np.full(n,j))} for j,b in enumerate(BUDGETS)]
    export_csv(out / "fixed_baselines.csv", fixed)

    evaluations, rules, predictions = [], [], {}
    for population in ["all7405", "below_cap"]:
        mask = np.ones(n, dtype=bool) if population == "all7405" else x < 8192
        xx = x[mask]
        mm = {name: mat[mask] for name,mat in mats.items()}
        for seed in SEEDS:
            assignment = folds(len(xx), seed)
            for lam in LAMBDAS:
                for family in ["global", "stump", "bins5"]:
                    actions, rs, mix_em, mix_cost = crossfit(xx, mm, lam, family, assignment)
                    stats = summary(mm, actions)
                    idx = np.arange(len(xx))
                    ev = {"population": population, "split_seed": seed, "family": family, "lambda_em_per_1000_tokens": lam, "n": len(xx), **stats,
                          "mix_same_budget_proportions_em": float(mix_em.mean()),
                          "em_above_random_mix": float((mm["em"][idx,actions]-mix_em).mean()),
                          "mix_same_budget_proportions_total": float(mix_cost.mean()),
                          "budget_distribution": json.dumps(dict(Counter(BUDGETS[j] for j in actions)),sort_keys=True)}
                    evaluations.append(ev)
                    if seed == SEEDS[0] and population == "all7405":
                        key = f"{family}_lambda{lam}"
                        predictions[key] = actions
                        for r in rs:
                            rules.append({"policy": key, **r})
    export_csv(out / "crossfit_policy_results.csv", evaluations)
    comparisons = []
    for lam in LAMBDAS:
        for family in ["stump", "bins5"]:
            key = f"{family}_lambda{lam}"
            actions = predictions[key]
            for label, ref in [
                ("FULL", np.full(n,6)), ("fixed500", np.full(n,4)), ("fixed600", np.full(n,5)),
                ("global_same_lambda", predictions[f"global_lambda{lam}"]),
            ]:
                comparisons.append({"policy": key, **compare(mats, actions, ref, label)})
            # Summary OOF draws do not account for overlapping training-set uncertainty.
            assignment = folds(n, SEEDS[0])
            _, _, mix_em, mix_cost = crossfit(x, mats, lam, family, assignment)
            diff = mats["em"][np.arange(n), actions] - mix_em
            comparisons.append({"policy": key, "reference": "same_proportions_feature_independent_mix", "em_delta": float(diff.mean()), "em_delta_ci95": ci_mean(diff), "total_token_delta": float((mats["total"][np.arange(n),actions]-mix_cost).mean())})
    export_csv(out / "paired_policy_comparisons.csv", [{k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in comparisons])
    first_assignment = folds(n,SEEDS[0])
    export_csv(out / "out_of_fold_choices.csv", [{"sample_id": i, "root_output_tokens": int(roots[k,0]), "root_input_tokens": int(roots[k,1]), "fold": int(first_assignment[k]), **{p:BUDGETS[int(a[k])] for p,a in predictions.items()}} for k,i in enumerate(ids)])
    result = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "samples": n, "budget_choices": BUDGETS,
        "signal": "standalone_llm_output_tokens minus continuation_llm_output_tokens",
        "signal_interpretation": "retokenized root generation length through first search; exact pre-query thinking count not available in review",
        "root_output_quantiles": {str(q):float(np.quantile(x,q)) for q in [0,.1,.25,.5,.75,.9,.99,1]},
        "root_output_mean": float(x.mean()), "root_at_cap_count": int((x>=8192).sum()),
        "root_invariant_errors": 0,
        "method": "5-fold question-disjoint crossfitting, 3 predefined split seeds; all budget choices of a question kept together; EM training objective",
        "utility": "EM minus lambda times (standalone input+output tokens)/1000; root cost included",
        "new_model_calls": 0,
        "quality_limitations": "supplied F1 not official yes/no-aware HotpotQA scorer; EM primary; no raw generation/answers or root parse modes in supplied review",
        "inference_limitations": "exploratory cross-validation, not a new blind test; CIs conditional on OOF predictions do not fully propagate training uncertainty; multiple lambda/family comparisons not adjusted",
        "fixed": fixed, "comparisons": comparisons, "rules": rules,
    }
    (out / "audit_summary.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    report = ["# 首轮 query 生成长度：HotpotQA 7,405 题离线诊断", "",
        "原始数据未改动，新增模型调用为 0。主信号为首轮生成的输出 token 数（含思考、query 和标签），不是已验证的纯思考 token 数。",
        f"7,405 题全部具备共同预算 100–600 和 FULL。信号中位数 {np.median(x):.0f}，均值 {x.mean():.1f}；{(x>=8192).sum()} 题达到 8,192 上限，这些题未必真正生成有效 query，需额外区分原始 root parse mode。",
        "",
        "题目级 5 折交叉验证，按种子 20261003、20261004、20261005 重复。每折阈值和预算仅在另外四折选择。单阈值最多两个预算；bins5 为五分位分组诊断。预算上限沿用原轨迹，后续搜索自然运行，未干预首轮生成长度。成本为每题独立运行的输入+输出 tokens，包含共同 root 成本，非 GPU 时间或货币成本。",
        "",
        "## 第一组划分的结果", "", "|方法|λ|EM|均值总 tokens|与同预算比例随机分配相比 EM 增益|", "|---|---:|---:|---:|---:|"]
    for ev in evaluations:
        if ev["population"] == "all7405" and ev["split_seed"] == SEEDS[0]:
            report.append(f"|{ev['family']}|{ev['lambda_em_per_1000_tokens']:.3f}|{ev['em']*100:.3f}%|{ev['total']:.1f}|{ev['em_above_random_mix']*100:+.3f} pp|")
    report.extend(["", "## 解读限制", "",
        "不能用信号与最终正确率的相关性代替预算选择能力。相同预算比例的随机混合基线检验收益是否来自特征，而不是简单降低预算。应同时比较 FULL、固定 500/600、以及仅用训练集选择的全局固定预算。",
        "当前 F1 缺少官方 yes/no/noanswer 不匹配的零分规则，因此这里以 EM 为主，F1 仅列为原有口径。需要原始答案和 gold 才能修正 F1。",
        "所有 λ/模型结果均为探索性报告，不能事后选最好一个并称作预注册独立验证。配对 bootstrap 区间条件于已得到的 OOF 预测，未完全包含训练估计的不确定性，也未校正多重比较。",
        "并未验证修改第一轮思考长度的因果效果。首轮截断、fallback、题型或问题长度都可能影响信号，应在后续原始 root 导出中区分。",
    ])
    (out / "REPORT_ZH.md").write_text("\n".join(report)+"\n", encoding="utf-8")
    print(json.dumps({"samples":n,"root_at_cap":int((x>=8192).sum()),"output":str(out),"new_model_calls":0},ensure_ascii=False))
    for ev in evaluations:
        if ev["population"] == "all7405" and ev["split_seed"] == SEEDS[0]:
            print(json.dumps(ev,ensure_ascii=False))


if __name__ == "__main__":
    main()
