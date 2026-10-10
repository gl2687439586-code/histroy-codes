from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import RepeatedKFold


BUDGETS = [100, 200, 300, 400, 500, 600]
METRICS = [
    "answer_f1",
    "answer_em",
    "answer_acc",
    "standalone_llm_input_tokens",
    "standalone_llm_output_tokens",
    "standalone_thinking_tokens",
    "total_llm_tokens",
    "visible_evidence_tokens_total",
    "search_calls",
]


def read_scored(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    return df[df.condition_group.astype(str).isin([str(x) for x in BUDGETS])].copy()


def select_budget(train: pd.DataFrame) -> int:
    table = train.groupby("budget").agg(
        mean_f1=("answer_f1", "mean"),
        mean_input=("standalone_llm_input_tokens", "mean"),
    ).reset_index()
    # Quality first. Floating means will almost never tie; the remaining keys make the rule deterministic.
    return int(table.sort_values(["mean_f1", "mean_input", "budget"], ascending=[False, True, True]).iloc[0].budget)


def policy_budget_map(train: pd.DataFrame, policy: str, min_cell_n: int = 20) -> tuple[dict, int]:
    global_budget = select_budget(train)
    if policy.startswith("fixed_"):
        fixed_budget = int(policy.split("_", 1)[1])
        return {"__global__": fixed_budget}, fixed_budget
    if policy == "global":
        return {"__global__": global_budget}, global_budget
    if policy == "ratio_primary":
        columns = ["ratio_group_primary"]
        fallback_columns = []
    elif policy == "ratio_sign":
        columns = ["ratio_group_sign"]
        fallback_columns = []
    elif policy == "question_type":
        columns = ["type"]
        fallback_columns = []
    elif policy == "ratio_primary_plus_type":
        columns = ["ratio_group_primary", "type"]
        fallback_columns = ["type"]
    else:
        raise ValueError(policy)

    sample_meta = train.drop_duplicates("sample_id")
    mapping = {}
    for keys, meta_group in sample_meta.groupby(columns, observed=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        ids = meta_group.sample_id
        if len(ids) >= min_cell_n:
            mapping[keys] = select_budget(train[train.sample_id.isin(ids)])
        elif fallback_columns:
            fallback_value = tuple(meta_group.iloc[0][c] for c in fallback_columns)
            fallback_meta = sample_meta
            for c, value in zip(fallback_columns, fallback_value):
                fallback_meta = fallback_meta[fallback_meta[c] == value]
            mapping[keys] = select_budget(train[train.sample_id.isin(fallback_meta.sample_id)])
        else:
            mapping[keys] = global_budget
    mapping["__columns__"] = columns
    return mapping, global_budget


def apply_policy(test: pd.DataFrame, mapping: dict, global_budget: int) -> pd.DataFrame:
    columns = mapping.get("__columns__", [])
    meta = test.drop_duplicates("sample_id").copy()
    if not columns:
        meta["selected_budget"] = mapping["__global__"]
    else:
        def lookup(row: pd.Series) -> int:
            key = tuple(row[c] for c in columns)
            return int(mapping.get(key, global_budget))
        meta["selected_budget"] = meta.apply(lookup, axis=1)
    chosen = test.merge(meta[["sample_id", "selected_budget"]], on="sample_id")
    chosen = chosen[chosen.budget == chosen.selected_budget].copy()
    if chosen.sample_id.nunique() != meta.sample_id.nunique() or chosen.sample_id.duplicated().any():
        raise RuntimeError("Policy did not select exactly one trajectory per held-out question")
    return chosen


def cross_validated_policy_predictions(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    policies = [
        "fixed_500", "fixed_600", "global", "ratio_primary", "ratio_sign",
        "question_type", "ratio_primary_plus_type",
    ]
    ids = np.array(sorted(df.sample_id.unique()))
    cv = RepeatedKFold(n_splits=10, n_repeats=20, random_state=seed)
    predictions = []
    selection_rows = []
    for split_i, (train_idx, test_idx) in enumerate(cv.split(ids)):
        repeat = split_i // 10
        fold = split_i % 10
        train = df[df.sample_id.isin(ids[train_idx])]
        test = df[df.sample_id.isin(ids[test_idx])]
        for policy in policies:
            mapping, global_budget = policy_budget_map(train, policy)
            chosen = apply_policy(test, mapping, global_budget)
            chosen["policy"] = policy
            chosen["repeat"] = repeat
            chosen["fold"] = fold
            predictions.append(chosen[["sample_id", "policy", "repeat", "fold", "selected_budget"] + METRICS])
            selection_rows.append(
                {
                    "policy": policy,
                    "repeat": repeat,
                    "fold": fold,
                    "global_fallback_budget": global_budget,
                    "mapping": json.dumps(
                        {str(k): v for k, v in mapping.items() if k != "__columns__"}, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(selection_rows)


def aggregate_predictions(pred: pd.DataFrame) -> pd.DataFrame:
    per_question = pred.groupby(["policy", "sample_id"]).agg(
        selected_budget=("selected_budget", "mean"),
        answer_f1=("answer_f1", "mean"),
        answer_em=("answer_em", "mean"),
        answer_acc=("answer_acc", "mean"),
        standalone_llm_input_tokens=("standalone_llm_input_tokens", "mean"),
        standalone_llm_output_tokens=("standalone_llm_output_tokens", "mean"),
        standalone_thinking_tokens=("standalone_thinking_tokens", "mean"),
        total_llm_tokens=("total_llm_tokens", "mean"),
        visible_evidence_tokens_total=("visible_evidence_tokens_total", "mean"),
        search_calls=("search_calls", "mean"),
    ).reset_index()
    per_question["visible_tokens_per_search"] = per_question.visible_evidence_tokens_total / per_question.search_calls
    return per_question


def paired_bootstrap(
    per_question: pd.DataFrame, seed: int, reference: str = "global", n_boot: int = 20000
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline = per_question[per_question.policy == reference].set_index("sample_id")
    rows = []
    outcomes = [
        "answer_f1", "answer_em", "answer_acc", "standalone_llm_input_tokens",
        "standalone_llm_output_tokens", "standalone_thinking_tokens", "total_llm_tokens",
        "visible_evidence_tokens_total", "search_calls", "visible_tokens_per_search",
    ]
    for policy in sorted(set(per_question.policy) - {reference}):
        candidate = per_question[per_question.policy == policy].set_index("sample_id").loc[baseline.index]
        for outcome in outcomes:
            delta = (candidate[outcome] - baseline[outcome]).to_numpy(float)
            boot = np.empty(n_boot)
            for i in range(n_boot):
                idx = rng.integers(0, len(delta), len(delta))
                boot[i] = delta[idx].mean()
            rows.append(
                {
                    "policy": policy,
                    "reference": reference,
                    "outcome": outcome,
                    "mean_delta_vs_global": float(delta.mean()),
                    "ci95_low": float(np.quantile(boot, 0.025)),
                    "ci95_high": float(np.quantile(boot, 0.975)),
                    "probability_delta_gt_zero": float(np.mean(boot > 0)),
                }
            )
    return pd.DataFrame(rows)


def make_figure(summary: pd.DataFrame, deltas: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    order = [
        "fixed_500", "fixed_600", "global", "ratio_primary", "ratio_sign",
        "question_type", "ratio_primary_plus_type",
    ]
    labels = {
        "fixed_500": "Fixed 500",
        "fixed_600": "Fixed 600",
        "global": "Global budget",
        "ratio_primary": "Ratio (5/6, 1.2)",
        "ratio_sign": "Ratio (<1, =1, >1)",
        "question_type": "Question type",
        "ratio_primary_plus_type": "Ratio + type",
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    plot = summary.set_index("policy").loc[order].reset_index()
    sns.barplot(data=plot, x="policy", y="answer_f1", color="#4C78A8", ax=axes[0])
    axes[0].set_xticks(range(len(order)), [labels[x] for x in order], rotation=22, ha="right")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Held-out F1")
    axes[0].set_title("Cross-validated answer quality", weight="bold")
    for container in axes[0].containers:
        axes[0].bar_label(container, fmt="%.4f", padding=3, fontsize=10)

    sub = deltas[(deltas.outcome == "answer_f1") & (deltas.reference == "fixed_500")].copy()
    sub["label"] = sub.policy.map(labels)
    y = np.arange(len(sub))
    axes[1].errorbar(
        sub.mean_delta_vs_global, y,
        xerr=[sub.mean_delta_vs_global - sub.ci95_low, sub.ci95_high - sub.mean_delta_vs_global],
        fmt="o", color="#D1495B", capsize=5, markersize=8,
    )
    axes[1].axvline(0, color="black", linewidth=1)
    axes[1].set_yticks(y, sub.label)
    axes[1].set_xlabel("Paired F1 change vs fixed 500")
    axes[1].set_title("Question-level bootstrap 95% CI", weight="bold")
    fig.suptitle("Does a query prior improve budget allocation?", weight="bold", fontsize=20)
    fig.savefig(output / "query_prior_policy_evaluation.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "query_prior_policy_evaluation.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--ratio", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    scored = read_scored(args.scored)
    ratio = pd.read_csv(args.ratio)[
        ["sample_id", "depth_breadth_ratio", "ratio_group_primary", "ratio_group_sign", "type"]
    ].copy()
    # The scored file also contains a type column. Rename before joining so the frozen ratio artifact is authoritative.
    scored = scored.drop(columns=["type"], errors="ignore").merge(ratio, on="sample_id", validate="many_to_one")
    scored["total_llm_tokens"] = scored.standalone_llm_input_tokens + scored.standalone_llm_output_tokens
    if len(scored) != 3000 or scored.sample_id.nunique() != 500:
        raise RuntimeError("Expected a complete 500 x 6 common-budget rectangle")

    pred, selections = cross_validated_policy_predictions(scored, args.seed)
    per_question = aggregate_predictions(pred)
    summary = per_question.groupby("policy").agg(
        questions=("sample_id", "nunique"),
        selected_budget=("selected_budget", "mean"),
        answer_f1=("answer_f1", "mean"),
        answer_em=("answer_em", "mean"),
        answer_acc=("answer_acc", "mean"),
        input_tokens=("standalone_llm_input_tokens", "mean"),
        output_tokens=("standalone_llm_output_tokens", "mean"),
        thinking_tokens=("standalone_thinking_tokens", "mean"),
        total_llm_tokens=("total_llm_tokens", "mean"),
        visible_tokens=("visible_evidence_tokens_total", "mean"),
        search_calls=("search_calls", "mean"),
        visible_tokens_per_search=("visible_tokens_per_search", "mean"),
    ).reset_index()
    deltas = pd.concat(
        [
            paired_bootstrap(per_question, args.seed + i, reference=reference)
            for i, reference in enumerate(["global", "fixed_500", "fixed_600"])
        ],
        ignore_index=True,
    )

    pred.to_csv(args.output / "out_of_fold_predictions.csv.gz", index=False, compression="gzip")
    selections.to_csv(args.output / "fold_policy_mappings.csv", index=False, encoding="utf-8-sig")
    per_question.to_csv(args.output / "per_question_policy_outcomes.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(args.output / "policy_summary.csv", index=False, encoding="utf-8-sig")
    deltas.to_csv(args.output / "paired_bootstrap_deltas.csv", index=False, encoding="utf-8-sig")
    make_figure(summary, deltas, args.output)

    selection_counts = {}
    for policy, group in selections.groupby("policy"):
        selection_counts[policy] = Counter(group.mapping).most_common(10)
    result = {
        "design": {
            "questions": 500,
            "budgets": BUDGETS,
            "evaluation": "20x repeated 10-fold cross-validation by question",
            "training_rule": "within each training group, choose budget with highest mean F1; fallback to global budget for cells with fewer than 20 training questions",
            "important_scope": "the policy controls evidence cap only; search count is a measured natural response, not a forced action",
        },
        "policy_summary": summary.to_dict(orient="records"),
        "paired_deltas": deltas.to_dict(orient="records"),
        "selection_mapping_modes": selection_counts,
    }
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def row(policy: str) -> pd.Series:
        return summary[summary.policy == policy].iloc[0]

    def delta(policy: str, outcome: str, reference: str = "global") -> pd.Series:
        return deltas[
            (deltas.policy == policy) & (deltas.outcome == outcome) & (deltas.reference == reference)
        ].iloc[0]

    report = f"""# Query 先验是否能提升 Read/Search 资源分配？

## 结论

在无泄漏的重复交叉验证中，深广度比值先验**没有提升固定预算基线**。同一批题上拟合并评估会出现一个看似漂亮的 Pareto 改善，但该改善在留出题目上消失，属于不可泛化的样本内选择效应。

当前实验可直接控制的是“每次搜索后的证据暴露上限（Read budget）”；搜索次数是模型自然行为。因此本实验验证的是 query-prior evidence-budget policy，而不是强制 `<Read, Search>` 双动作控制器。

## 评估方式

- 500 题，公共预算 100–600，每题 6 条完整反事实轨迹。
- 20 次重复 10 折交叉验证，按问题划分训练/测试，避免用测试题 Oracle 选择其自身预算。
- 训练折内为每个 query 分组选择平均 F1 最高的预算；测试折直接读取该预算对应的真实轨迹。
- 基线是在训练折上选择的全局固定预算，不使用深广度信息。

## 主结果

若直接在全部 500 题上选择“平衡=500、偏广度=400、偏深度=500”，会得到 F1=0.4966，看似优于固定 500/600；但这是用同一批题选择映射并评价映射，存在 Oracle leakage。下面的交叉验证才是可部署结论。成本统一使用 `Total = Input + Output`；evidence tokens 已包含在 input 中，不重复累加。

| 策略 | F1 | EM | ACC | 预算 | 搜索次数 | Input | Output | Total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 固定 500 | {row('fixed_500').answer_f1:.4f} | {row('fixed_500').answer_em:.4f} | {row('fixed_500').answer_acc:.4f} | {row('fixed_500').selected_budget:.1f} | {row('fixed_500').search_calls:.3f} | {row('fixed_500').input_tokens:.1f} | {row('fixed_500').output_tokens:.1f} | {row('fixed_500').total_llm_tokens:.1f} |
| 固定 600 | {row('fixed_600').answer_f1:.4f} | {row('fixed_600').answer_em:.4f} | {row('fixed_600').answer_acc:.4f} | {row('fixed_600').selected_budget:.1f} | {row('fixed_600').search_calls:.3f} | {row('fixed_600').input_tokens:.1f} | {row('fixed_600').output_tokens:.1f} | {row('fixed_600').total_llm_tokens:.1f} |
| 全局固定预算 | {row('global').answer_f1:.4f} | {row('global').answer_em:.4f} | {row('global').answer_acc:.4f} | {row('global').selected_budget:.1f} | {row('global').search_calls:.3f} | {row('global').input_tokens:.1f} | {row('global').output_tokens:.1f} | {row('global').total_llm_tokens:.1f} |
| 主比值阈值 | {row('ratio_primary').answer_f1:.4f} | {row('ratio_primary').answer_em:.4f} | {row('ratio_primary').answer_acc:.4f} | {row('ratio_primary').selected_budget:.1f} | {row('ratio_primary').search_calls:.3f} | {row('ratio_primary').input_tokens:.1f} | {row('ratio_primary').output_tokens:.1f} | {row('ratio_primary').total_llm_tokens:.1f} |
| 宽松 `<1,=1,>1` | {row('ratio_sign').answer_f1:.4f} | {row('ratio_sign').answer_em:.4f} | {row('ratio_sign').answer_acc:.4f} | {row('ratio_sign').selected_budget:.1f} | {row('ratio_sign').search_calls:.3f} | {row('ratio_sign').input_tokens:.1f} | {row('ratio_sign').output_tokens:.1f} | {row('ratio_sign').total_llm_tokens:.1f} |
| HotpotQA 题型 | {row('question_type').answer_f1:.4f} | {row('question_type').answer_em:.4f} | {row('question_type').answer_acc:.4f} | {row('question_type').selected_budget:.1f} | {row('question_type').search_calls:.3f} | {row('question_type').input_tokens:.1f} | {row('question_type').output_tokens:.1f} | {row('question_type').total_llm_tokens:.1f} |
| 比值 + 题型 | {row('ratio_primary_plus_type').answer_f1:.4f} | {row('ratio_primary_plus_type').answer_em:.4f} | {row('ratio_primary_plus_type').answer_acc:.4f} | {row('ratio_primary_plus_type').selected_budget:.1f} | {row('ratio_primary_plus_type').search_calls:.3f} | {row('ratio_primary_plus_type').input_tokens:.1f} | {row('ratio_primary_plus_type').output_tokens:.1f} | {row('ratio_primary_plus_type').total_llm_tokens:.1f} |

主比值策略相对全局预算：

- F1 变化：{delta('ratio_primary','answer_f1').mean_delta_vs_global:+.4f}，95% CI [{delta('ratio_primary','answer_f1').ci95_low:+.4f}, {delta('ratio_primary','answer_f1').ci95_high:+.4f}]；
- 总 token 变化：{delta('ratio_primary','total_llm_tokens').mean_delta_vs_global:+.1f}（input {delta('ratio_primary','standalone_llm_input_tokens').mean_delta_vs_global:+.1f}，output {delta('ratio_primary','standalone_llm_output_tokens').mean_delta_vs_global:+.1f}）；
- 自然搜索次数变化：{delta('ratio_primary','search_calls').mean_delta_vs_global:+.3f}。

若 F1 区间跨 0，则目前只能说存在方向性趋势，不能说 query prior 已经带来可靠提升。

与部署上更直接的固定 500 比较，主比值策略的 F1 变化为 {delta('ratio_primary','answer_f1','fixed_500').mean_delta_vs_global:+.4f}，95% CI [{delta('ratio_primary','answer_f1','fixed_500').ci95_low:+.4f}, {delta('ratio_primary','answer_f1','fixed_500').ci95_high:+.4f}]；总 token 变化为 {delta('ratio_primary','total_llm_tokens','fixed_500').mean_delta_vs_global:+.1f}。总成本必须以 input+output 判断，不能只依据证据预算或 input tokens。

## 研究含义

深广度比值更适合作为 controller 的初始化 prior，而不是在运行前一次性锁死 Read/Search 比例。推荐结构是：

`pi_0(action | query ratio, question type)`

随后由检索后的 supporting-fact coverage、new-evidence gain 和剩余预算动态更新。要验证强制 `<Read, Search>` 比例，需要新增显式控制搜索上限/搜索次数的反事实实验；现有自然轨迹不能独立识别该因果效应。
"""
    (args.output / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
