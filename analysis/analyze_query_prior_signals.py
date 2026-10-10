#!/usr/bin/env python3
"""Find query-only prior signals in the HotpotQA random-500 budget surface.

The analysis deliberately separates controlled evidence exposure from natural
search behavior.  Total cost is always LLM input plus LLM output; visible
evidence tokens are already included in the input and are never added twice.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BUDGETS = [100, 200, 300, 400, 500, 600]
EPSILON = 0.05
READ_GAIN_THRESHOLD = 0.10


def read_jsonl(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bh_adjust(p_values: pd.Series) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def ratio_group(value: float) -> str:
    if value <= 5 / 6:
        return "breadth-biased"
    if value >= 1.2:
        return "depth-biased"
    return "balanced"


def select_min_cost(rows: pd.DataFrame) -> pd.Series:
    ordered = rows.sort_values(
        ["total_llm_tokens", "standalone_llm_input_tokens", "budget", "search_calls"],
        ascending=True,
    )
    return ordered.iloc[0]


def build_question_table(labels: pd.DataFrame, scored: pd.DataFrame) -> pd.DataFrame:
    scored = scored[scored.budget.isin(BUDGETS)].copy()
    scored["total_llm_tokens"] = (
        scored.standalone_llm_input_tokens + scored.standalone_llm_output_tokens
    )
    duplicates = scored.duplicated(["sample_id", "budget"], keep=False)
    if duplicates.any():
        raise RuntimeError("Duplicate sample/budget rows in common budget surface")

    records: list[dict] = []
    for sample_id, group in scored.groupby("sample_id", sort=False):
        group = group.sort_values("budget")
        if set(group.budget.astype(int)) != set(BUDGETS):
            continue
        by_budget = group.set_index("budget")
        max_f1 = float(group.answer_f1.max())
        epsilon_rows = group[group.answer_f1 >= max_f1 - EPSILON - 1e-12]
        epsilon_oracle = select_min_cost(epsilon_rows)
        correct_rows = group[group.answer_acc >= 0.5]
        min_correct = select_min_cost(correct_rows) if len(correct_rows) else None
        root = by_budget.loc[100]

        root_retrieved_all = bool(root.root_retrieved_all_titles)
        root_support_all_100 = bool(root.root_support_sentence_strict_all)
        answer_correct_100 = bool(root.answer_acc >= 0.5)
        if not root_retrieved_all:
            initial_gap = "search-gap"
        elif not root_support_all_100:
            initial_gap = "read-gap"
        elif not answer_correct_100:
            initial_gap = "integration-gap"
        else:
            initial_gap = "early-sufficient"

        low_mean = float(by_budget.loc[[100, 200], "answer_f1"].mean())
        high_mean = float(by_budget.loc[[400, 500, 600], "answer_f1"].mean())
        read_gain = high_mean - low_mean
        if read_gain >= READ_GAIN_THRESHOLD:
            read_class = "positive"
        elif read_gain <= -READ_GAIN_THRESHOLD:
            read_class = "negative"
        else:
            read_class = "flat"

        acc_sequence = by_budget.loc[BUDGETS, "answer_acc"].to_numpy(dtype=float)
        f1_sequence = by_budget.loc[BUDGETS, "answer_f1"].to_numpy(dtype=float)
        total_sequence = by_budget.loc[BUDGETS, "total_llm_tokens"].to_numpy(dtype=float)
        marginal_f1 = np.diff(f1_sequence)
        marginal_total = np.diff(total_sequence)

        record = {
            "sample_id": sample_id,
            "question_type": str(root["type"]),
            "any_correct": int(len(correct_rows) > 0),
            "max_f1": max_f1,
            "correct_budget_count": int((group.answer_acc >= 0.5).sum()),
            "tokens_to_solve": float(min_correct.total_llm_tokens) if min_correct is not None else np.nan,
            "tokens_to_solve_budget": int(min_correct.budget) if min_correct is not None else np.nan,
            "tokens_to_solve_search_calls": int(min_correct.search_calls) if min_correct is not None else np.nan,
            "epsilon_budget": int(epsilon_oracle.budget),
            "epsilon_search_calls": int(epsilon_oracle.search_calls),
            "epsilon_total_tokens": float(epsilon_oracle.total_llm_tokens),
            "epsilon_f1": float(epsilon_oracle.answer_f1),
            "epsilon_high_budget": int(epsilon_oracle.budget >= 400),
            "epsilon_multi_search": int(epsilon_oracle.search_calls >= 2),
            "read_gain_high_minus_low": read_gain,
            "read_gain_class": read_class,
            "read_gain_positive": int(read_class == "positive"),
            "read_gain_negative": int(read_class == "negative"),
            "answer_flip_rate": float(np.mean(np.diff(acc_sequence) != 0)),
            "root_retrieved_title_recall": float(root.root_retrieved_title_recall),
            "root_retrieved_all_titles": int(root_retrieved_all),
            "root_support_sentence_strict_recall_100": float(root.root_support_sentence_strict_recall),
            "root_visible_title_recall_100": float(root.root_visible_title_recall),
            "root_support_sentence_strict_recall_600": float(by_budget.loc[600].root_support_sentence_strict_recall),
            "initial_resource_gap": initial_gap,
            "first_search_failure": int(not root_retrieved_all),
            "initial_read_gap": int(initial_gap == "read-gap"),
            "initial_integration_gap": int(initial_gap == "integration-gap"),
            "initial_early_sufficient": int(initial_gap == "early-sufficient"),
            "mean_search_calls": float(group.search_calls.mean()),
            "search_calls_at_100": int(by_budget.loc[100].search_calls),
            "search_calls_at_600": int(by_budget.loc[600].search_calls),
            "mean_total_tokens": float(group.total_llm_tokens.mean()),
            "mean_output_tokens": float(group.standalone_llm_output_tokens.mean()),
            "mean_input_tokens": float(group.standalone_llm_input_tokens.mean()),
        }
        for idx, start in enumerate(BUDGETS[:-1]):
            end = BUDGETS[idx + 1]
            record[f"delta_f1_{start}_{end}"] = float(marginal_f1[idx])
            record[f"delta_total_{start}_{end}"] = float(marginal_total[idx])
        records.append(record)

    targets = pd.DataFrame(records)
    merged = labels.merge(targets, left_on="query_id", right_on="sample_id", validate="one_to_one")
    merged["ratio"] = merged.depth_score / merged.breadth_score
    merged["score_interaction"] = merged.depth_score * merged.breadth_score
    merged["ratio_group"] = merged.ratio.map(ratio_group)
    return merged


def build_features(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    ratio = df[["ratio"]].astype(float)
    scores = df[["breadth_score", "depth_score", "ratio", "score_interaction"]].astype(float)
    cue_names = sorted(
        {f"breadth::{cue}" for cues in df.breadth_cues for cue in cues}
        | {f"depth::{cue}" for cues in df.depth_cues for cue in cues}
    )
    cue_data = {}
    for cue in cue_names:
        family, value = cue.split("::", 1)
        source = "breadth_cues" if family == "breadth" else "depth_cues"
        cue_data[cue] = df[source].map(lambda items: int(value in items))
    cues = pd.DataFrame(cue_data, index=df.index)
    label_dummies = pd.get_dummies(df["relative_label"], prefix="label", dtype=int)
    confidence = df[["confidence"]].astype(float)
    question_type = pd.get_dummies(df["question_type"], prefix="hotpot_type", dtype=int)
    full = pd.concat([scores, confidence, label_dummies, cues], axis=1)
    return {
        "ratio_only": ratio,
        "scores": scores,
        "scores_plus_query_cues": full,
        # Diagnostic controls: official HotpotQA type is not treated as a
        # deployable feature, but it reveals how much of the score signal is
        # actually comparison-vs-bridge topology.
        "question_type_only_diagnostic": question_type,
        "ratio_plus_type_diagnostic": pd.concat([ratio, question_type], axis=1),
        "scores_cues_plus_type_diagnostic": pd.concat([full, question_type], axis=1),
    }


def type_stratified_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Compare breadth-biased versus balanced within each HotpotQA type."""
    outcomes = [
        "any_correct", "first_search_failure", "read_gain_positive",
        "epsilon_high_budget", "epsilon_multi_search",
    ]
    rows = []
    for question_type in sorted(df.question_type.unique()):
        subset = df[df.question_type == question_type]
        for outcome in outcomes:
            a = subset[subset.ratio_group == "breadth-biased"][outcome].astype(int)
            b = subset[subset.ratio_group == "balanced"][outcome].astype(int)
            if len(a) == 0 or len(b) == 0:
                continue
            table = np.array([
                [int(a.sum()), int(len(a) - a.sum())],
                [int(b.sum()), int(len(b) - b.sum())],
            ])
            odds, p = fisher_exact(table)
            rows.append({
                "question_type": question_type,
                "contrast": "breadth-biased_vs_balanced",
                "outcome": outcome,
                "n_breadth_biased": len(a),
                "n_balanced": len(b),
                "breadth_biased_rate": float(a.mean()),
                "balanced_rate": float(b.mean()),
                "odds_ratio": float(odds),
                "p_value": float(p),
            })
    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh_adjust(result.p_value)
    return result.sort_values(["q_value_bh", "p_value"])


def continuous_associations(df: pd.DataFrame) -> pd.DataFrame:
    predictors = ["breadth_score", "depth_score", "ratio", "confidence"]
    outcomes = [
        "any_correct", "max_f1", "tokens_to_solve", "epsilon_budget",
        "epsilon_search_calls", "epsilon_total_tokens", "read_gain_high_minus_low",
        "root_retrieved_title_recall", "root_support_sentence_strict_recall_100",
        "answer_flip_rate", "mean_output_tokens", "mean_total_tokens",
    ]
    rows = []
    for predictor in predictors:
        for outcome in outcomes:
            pair = df[[predictor, outcome]].dropna()
            rho, p = spearmanr(pair[predictor], pair[outcome])
            rows.append({
                "predictor": predictor, "outcome": outcome, "n": len(pair),
                "spearman_rho": float(rho), "p_value": float(p),
            })
    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh_adjust(result.p_value)
    return result.sort_values(["q_value_bh", "p_value"])


def cue_associations(df: pd.DataFrame) -> pd.DataFrame:
    cues = sorted(
        {f"breadth::{cue}" for items in df.breadth_cues for cue in items}
        | {f"depth::{cue}" for items in df.depth_cues for cue in items}
    )
    binary_outcomes = [
        "any_correct", "epsilon_high_budget", "epsilon_multi_search",
        "read_gain_positive", "first_search_failure", "initial_read_gap",
    ]
    continuous_outcomes = ["tokens_to_solve", "read_gain_high_minus_low", "epsilon_total_tokens"]
    rows = []
    for cue in cues:
        family, value = cue.split("::", 1)
        source = "breadth_cues" if family == "breadth" else "depth_cues"
        present = df[source].map(lambda items: value in items)
        for outcome in binary_outcomes:
            table = pd.crosstab(present, df[outcome]).reindex(index=[False, True], columns=[0, 1], fill_value=0)
            odds, p = fisher_exact(table.to_numpy())
            rows.append({
                "cue": cue, "outcome": outcome, "test": "fisher_exact",
                "n_present": int(present.sum()), "effect": float(odds), "p_value": float(p),
                "present_mean": float(df.loc[present, outcome].mean()),
                "absent_mean": float(df.loc[~present, outcome].mean()),
            })
        for outcome in continuous_outcomes:
            valid = df[outcome].notna()
            yes = df.loc[present & valid, outcome]
            no = df.loc[~present & valid, outcome]
            if len(yes) < 2 or len(no) < 2:
                continue
            stat, p = mannwhitneyu(yes, no, alternative="two-sided")
            rows.append({
                "cue": cue, "outcome": outcome, "test": "mann_whitney",
                "n_present": int((present & valid).sum()),
                "effect": float(yes.mean() - no.mean()), "p_value": float(p),
                "present_mean": float(yes.mean()), "absent_mean": float(no.mean()),
            })
    result = pd.DataFrame(rows)
    result["q_value_bh"] = bh_adjust(result.p_value)
    return result.sort_values(["q_value_bh", "p_value"])


def classification_cv(features: dict[str, pd.DataFrame], df: pd.DataFrame, seed: int) -> pd.DataFrame:
    targets = [
        "any_correct", "epsilon_high_budget", "epsilon_multi_search",
        "read_gain_positive", "first_search_failure", "initial_read_gap",
    ]
    rows = []
    for target in targets:
        y = df[target].astype(int).to_numpy()
        if len(np.unique(y)) < 2 or min(Counter(y).values()) < 5:
            continue
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=seed)
        for feature_name, xdf in features.items():
            x = xdf.to_numpy(dtype=float)
            for fold, (train, test) in enumerate(cv.split(x, y)):
                model = make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(),
                    LogisticRegression(C=0.2, class_weight="balanced", max_iter=3000),
                )
                model.fit(x[train], y[train])
                prob = model.predict_proba(x[test])[:, 1]
                pred = (prob >= 0.5).astype(int)
                rows.append({
                    "target": target, "features": feature_name, "fold": fold,
                    "roc_auc": roc_auc_score(y[test], prob),
                    "balanced_accuracy": balanced_accuracy_score(y[test], pred),
                    "log_loss": log_loss(y[test], prob, labels=[0, 1]),
                    "prevalence": float(y[test].mean()),
                })
    folds = pd.DataFrame(rows)
    summary = folds.groupby(["target", "features"], as_index=False).agg(
        folds=("fold", "size"), prevalence=("prevalence", "mean"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_sd=("roc_auc", "std"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_sd=("balanced_accuracy", "std"),
        log_loss_mean=("log_loss", "mean"), log_loss_sd=("log_loss", "std"),
    )
    return folds, summary


def regression_cv(features: dict[str, pd.DataFrame], df: pd.DataFrame, seed: int) -> pd.DataFrame:
    targets = ["tokens_to_solve", "read_gain_high_minus_low", "epsilon_total_tokens"]
    rows = []
    for target in targets:
        valid = df[target].notna().to_numpy()
        y = df.loc[valid, target].to_numpy(dtype=float)
        cv = RepeatedKFold(n_splits=5, n_repeats=10, random_state=seed)
        for feature_name, xdf in features.items():
            x = xdf.loc[valid].to_numpy(dtype=float)
            for fold, (train, test) in enumerate(cv.split(x)):
                model = make_pipeline(
                    SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)
                )
                model.fit(x[train], y[train])
                pred = model.predict(x[test])
                baseline = np.full(len(test), np.median(y[train]))
                rho = spearmanr(y[test], pred).statistic
                rows.append({
                    "target": target, "features": feature_name, "fold": fold,
                    "spearman": float(rho) if np.isfinite(rho) else 0.0,
                    "mae": mean_absolute_error(y[test], pred),
                    "baseline_median_mae": mean_absolute_error(y[test], baseline),
                    "mae_improvement": mean_absolute_error(y[test], baseline) - mean_absolute_error(y[test], pred),
                    "r2": r2_score(y[test], pred),
                })
    folds = pd.DataFrame(rows)
    summary = folds.groupby(["target", "features"], as_index=False).agg(
        folds=("fold", "size"), spearman_mean=("spearman", "mean"),
        spearman_sd=("spearman", "std"), mae_mean=("mae", "mean"),
        baseline_median_mae=("baseline_median_mae", "mean"),
        mae_improvement_mean=("mae_improvement", "mean"), r2_mean=("r2", "mean"),
    )
    return folds, summary


def grouped_summary(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for view in ["breadth_score", "depth_score", "ratio_group", "relative_label"]:
        part = df.groupby(view, dropna=False).agg(
            n=("sample_id", "size"), solvable_rate=("any_correct", "mean"),
            mean_max_f1=("max_f1", "mean"), mean_tokens_to_solve=("tokens_to_solve", "mean"),
            mean_read_gain=("read_gain_high_minus_low", "mean"),
            read_gain_positive_rate=("read_gain_positive", "mean"),
            first_search_failure_rate=("first_search_failure", "mean"),
            initial_read_gap_rate=("initial_read_gap", "mean"),
            high_budget_oracle_rate=("epsilon_high_budget", "mean"),
            multi_search_oracle_rate=("epsilon_multi_search", "mean"),
            mean_epsilon_total=("epsilon_total_tokens", "mean"),
        ).reset_index().rename(columns={view: "level"})
        part.insert(0, "view", view)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def plot_dashboard(df: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for score, color in [("breadth_score", "#2378B5"), ("depth_score", "#E07A2D")]:
        summary = df.groupby(score).agg(n=("sample_id", "size"), value=("any_correct", "mean")).reset_index()
        summary = summary[summary.n >= 10]
        axes[0, 0].plot(summary[score], summary.value, marker="o", label=score.replace("_score", ""), color=color)
        read = df.groupby(score).agg(n=("sample_id", "size"), value=("read_gain_high_minus_low", "mean")).reset_index()
        read = read[read.n >= 10]
        axes[0, 1].plot(read[score], read.value, marker="o", label=score.replace("_score", ""), color=color)
    axes[0, 0].set(title="Solvable rate by query score", xlabel="Score", ylabel="P(any correct)", ylim=(0, 1))
    axes[0, 1].axhline(0, color="black", lw=1)
    axes[0, 1].set(title="Mean read gain: high-budget F1 - low-budget F1", xlabel="Score", ylabel="Delta F1")
    axes[0, 0].legend(); axes[0, 1].legend()

    order = ["breadth-biased", "balanced", "depth-biased"]
    gap = pd.crosstab(df.ratio_group, df.initial_resource_gap, normalize="index").reindex(order).fillna(0)
    group_n = df.ratio_group.value_counts()
    gap.index = [f"{name}\n(n={int(group_n.get(name, 0))})" for name in gap.index]
    gap.plot(kind="bar", stacked=True, ax=axes[1, 0], colormap="Set2")
    axes[1, 0].set(title="Initial evidence-state gap by depth/breadth ratio", xlabel="Prior group", ylabel="Proportion")
    axes[1, 0].legend(title="Initial gap", fontsize=8)

    solvable = df[df.any_correct == 1].copy()
    sns.boxplot(data=solvable, x="ratio_group", y="tokens_to_solve", order=order, ax=axes[1, 1], showfliers=False)
    solved_n = solvable.ratio_group.value_counts()
    axes[1, 1].set_xticklabels([f"{name}\n(n={int(solved_n.get(name, 0))})" for name in order])
    axes[1, 1].set(title="Total tokens to first correct answer", xlabel="Prior group", ylabel="Input + output tokens")
    fig.suptitle("Query-prior signal audit on HotpotQA random-500", fontsize=19, weight="bold")
    fig.savefig(output / "query_prior_signal_dashboard.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "query_prior_signal_dashboard.pdf", bbox_inches="tight")
    plt.close(fig)


def report_text(
    df: pd.DataFrame,
    assoc: pd.DataFrame,
    cues: pd.DataFrame,
    cls: pd.DataFrame,
    reg: pd.DataFrame,
    stratified: pd.DataFrame,
) -> str:
    gap_counts = df.initial_resource_gap.value_counts()
    top_assoc = assoc.head(12)
    top_cues = cues[(cues.n_present >= 10) & (cues.q_value_bh <= 0.20)].head(12)
    best_cls = cls.sort_values(["target", "roc_auc_mean"], ascending=[True, False]).groupby("target").head(1)
    best_reg = reg.sort_values(["target", "spearman_mean"], ascending=[True, False]).groupby("target").head(1)
    lines = [
        "# HotpotQA 500题：最可能出现的 Query 先验信号",
        "",
        "## 口径",
        "",
        "- 总成本为 Input + Output；evidence 已包含在 Input，不重复相加。",
        "- Read 是受控的证据暴露预算；Search 次数是模型自然行为，只报告相关性。",
        f"- ε-optimal 使用 ε={EPSILON:.2f}：质量距本题最佳 F1 不超过 ε，再选择总 token 最低轨迹。",
        "- Tokens-to-Solve 是所有正确轨迹中最低总 token；无正确轨迹视为未解，而不是赋予一个‘最佳错误预算’。",
        "",
        "## 数据概况",
        "",
        f"- 题目：{len(df)}；可解：{int(df.any_correct.sum())}；当前网格未解：{int((1-df.any_correct).sum())}。",
        "- 初始证据状态：" + "；".join(f"{k}={v}" for k, v in gap_counts.items()) + "。",
        f"- 正向 Read 增益（高预算平均 F1 至少高 {READ_GAIN_THRESHOLD:.2f}）：{int(df.read_gain_positive.sum())}/{len(df)}。",
        f"- 负向 Read 增益：{int(df.read_gain_negative.sum())}/{len(df)}；其余为近似平坦。",
        "",
        "## 连续评分最强相关",
        "",
        top_assoc.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Query cue 候选信号（n>=10且BH q<=0.20）",
        "",
        top_cues.to_markdown(index=False, floatfmt=".4f") if len(top_cues) else "没有 cue 在该筛选标准下通过。",
        "",
        "## 无泄漏重复交叉验证：分类",
        "",
        best_cls.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 无泄漏重复交叉验证：连续目标",
        "",
        best_reg.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 控制HotpotQA题型后的检验",
        "",
        stratified.head(12).to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 判断原则",
        "",
        "- AUC 接近0.5、回归R²≤0或MAE不能优于训练折中位数，说明该query prior不能泛化。",
        "- 先看可解性、初始检索缺口和Read边际收益；不要只看精确Oracle pair。",
        "- 搜索次数未被独立干预，因此epsilon_multi_search只能作为自然行为先验，不能作Search因果收益。",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    labels = pd.DataFrame(read_jsonl(args.labels))
    scored = pd.DataFrame(read_jsonl(args.scored))
    df = build_question_table(labels, scored)
    if len(df) != 500:
        raise RuntimeError(f"Expected 500 matched questions, got {len(df)}")
    features = build_features(df)

    assoc = continuous_associations(df)
    cues = cue_associations(df)
    cls_folds, cls_summary = classification_cv(features, df, args.seed)
    reg_folds, reg_summary = regression_cv(features, df, args.seed)
    groups = grouped_summary(df)
    stratified = type_stratified_tests(df)

    df.to_csv(args.output / "per_question_prior_targets.csv", index=False, encoding="utf-8-sig")
    assoc.to_csv(args.output / "continuous_score_associations.csv", index=False, encoding="utf-8-sig")
    cues.to_csv(args.output / "query_cue_associations.csv", index=False, encoding="utf-8-sig")
    cls_folds.to_csv(args.output / "classification_cv_folds.csv", index=False, encoding="utf-8-sig")
    cls_summary.to_csv(args.output / "classification_cv_summary.csv", index=False, encoding="utf-8-sig")
    reg_folds.to_csv(args.output / "regression_cv_folds.csv", index=False, encoding="utf-8-sig")
    reg_summary.to_csv(args.output / "regression_cv_summary.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(args.output / "score_group_prior_target_summary.csv", index=False, encoding="utf-8-sig")
    stratified.to_csv(args.output / "question_type_stratified_tests.csv", index=False, encoding="utf-8-sig")
    plot_dashboard(df, args.output)

    report = report_text(df, assoc, cues, cls_summary, reg_summary, stratified)
    (args.output / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    summary = {
        "questions": len(df),
        "total_cost": "standalone_llm_input_tokens + standalone_llm_output_tokens",
        "solvable": int(df.any_correct.sum()),
        "unsolved": int((1 - df.any_correct).sum()),
        "initial_resource_gap_counts": gap_counts_to_json(df.initial_resource_gap),
        "read_gain_class_counts": gap_counts_to_json(df.read_gain_class),
        "top_continuous_associations": assoc.head(20).to_dict("records"),
        "classification_cv": cls_summary.to_dict("records"),
        "regression_cv": reg_summary.to_dict("records"),
        "question_type_stratified_tests": stratified.to_dict("records"),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def gap_counts_to_json(series: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in series.value_counts().items()}


if __name__ == "__main__":
    raise SystemExit(main())
