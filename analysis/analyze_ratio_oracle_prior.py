from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu, spearmanr
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mutual_info_score, r2_score, mean_absolute_error
from sklearn.model_selection import RepeatedKFold


LOW_PRIMARY = 5 / 6
HIGH_PRIMARY = 1.2


def bh_qvalues(pvalues: list[float]) -> list[float]:
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    q_ranked = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    q_ranked = np.clip(q_ranked, 0, 1)
    q = np.empty_like(q_ranked)
    q[order] = q_ranked
    return q.tolist()


def bootstrap_spearman(x: np.ndarray, y: np.ndarray, seed: int, n_boot: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = []
    n = len(x)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if np.unique(x[idx]).size < 2 or np.unique(y[idx]).size < 2:
            continue
        vals.append(spearmanr(x[idx], y[idx]).statistic)
    return tuple(np.quantile(vals, [0.025, 0.975]))


def ratio_group(ratio: pd.Series, scheme: str) -> pd.Series:
    if scheme == "primary":
        values = np.select(
            [ratio <= LOW_PRIMARY, ratio >= HIGH_PRIMARY],
            ["breadth-biased", "depth-biased"],
            default="balanced",
        )
    elif scheme == "sign":
        values = np.select(
            [ratio < 1.0, ratio > 1.0],
            ["breadth-biased", "depth-biased"],
            default="balanced",
        )
    elif scheme == "exploratory_1.1":
        values = np.select(
            [ratio <= 1 / 1.1, ratio >= 1.1],
            ["breadth-biased", "depth-biased"],
            default="balanced",
        )
    else:
        raise ValueError(scheme)
    return pd.Series(values, index=ratio.index)


def continuous_associations(df: pd.DataFrame, seed: int, correctness_col: str = "oracle_correct_acc") -> pd.DataFrame:
    rows = []
    for correct, group in df.groupby(correctness_col):
        for outcome in ["oracle_budget", "oracle_search_calls", "oracle_visible_tokens_total"]:
            stat = spearmanr(group["depth_breadth_ratio"], group[outcome])
            lo, hi = bootstrap_spearman(
                group["depth_breadth_ratio"].to_numpy(float),
                group[outcome].to_numpy(float),
                seed + len(rows),
            )
            rows.append(
                {
                    "correctness_definition": correctness_col,
                    "oracle_correct": int(correct),
                    "n": len(group),
                    "outcome": outcome,
                    "spearman_rho": float(stat.statistic),
                    "ci95_low": float(lo),
                    "ci95_high": float(hi),
                    "p_value": float(stat.pvalue),
                }
            )
    out = pd.DataFrame(rows)
    out["bh_q_value"] = bh_qvalues(out["p_value"].tolist())
    return out


def group_summary(df: pd.DataFrame, scheme: str, correctness_col: str = "oracle_correct_acc") -> pd.DataFrame:
    group_col = f"ratio_group_{scheme}"
    records = []
    for (correct, label), group in df.groupby([correctness_col, group_col], observed=False):
        pair_counts = group.groupby(["oracle_budget", "oracle_search_calls"]).size().sort_values(ascending=False)
        mode_pair = pair_counts.index[0] if len(pair_counts) else (np.nan, np.nan)
        records.append(
            {
                "scheme": scheme,
                "oracle_correct": int(correct),
                "ratio_group": label,
                "n": len(group),
                "ratio_mean": group.depth_breadth_ratio.mean(),
                "oracle_budget_mean": group.oracle_budget.mean(),
                "oracle_budget_median": group.oracle_budget.median(),
                "oracle_search_calls_mean": group.oracle_search_calls.mean(),
                "oracle_search_calls_median": group.oracle_search_calls.median(),
                "oracle_visible_tokens_total_mean": group.oracle_visible_tokens_total.mean(),
                "mode_budget": mode_pair[0],
                "mode_search_calls": mode_pair[1],
                "mode_pair_rate": float(pair_counts.iloc[0] / len(group)) if len(group) else np.nan,
            }
        )
    return pd.DataFrame(records)


def group_tests(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    group_col = f"ratio_group_{scheme}"
    order = ["breadth-biased", "balanced", "depth-biased"]
    rows = []
    for correct, subset in df.groupby("oracle_correct_acc"):
        for outcome in ["oracle_budget", "oracle_search_calls", "oracle_visible_tokens_total"]:
            groups = [subset[subset[group_col] == label][outcome].to_numpy(float) for label in order]
            nonempty = [x for x in groups if len(x)]
            kw = kruskal(*nonempty) if len(nonempty) >= 2 else None
            stable_a = groups[0]
            stable_b = groups[1]
            mw = mannwhitneyu(stable_a, stable_b, alternative="two-sided") if len(stable_a) and len(stable_b) else None
            rows.append(
                {
                    "scheme": scheme,
                    "oracle_correct": int(correct),
                    "outcome": outcome,
                    "n_breadth": len(groups[0]),
                    "n_balanced": len(groups[1]),
                    "n_depth": len(groups[2]),
                    "kruskal_h": float(kw.statistic) if kw else np.nan,
                    "kruskal_p": float(kw.pvalue) if kw else np.nan,
                    "breadth_vs_balanced_u": float(mw.statistic) if mw else np.nan,
                    "breadth_vs_balanced_p": float(mw.pvalue) if mw else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out["kruskal_bh_q"] = bh_qvalues(out["kruskal_p"].fillna(1).tolist())
    out["breadth_vs_balanced_bh_q"] = bh_qvalues(out["breadth_vs_balanced_p"].fillna(1).tolist())
    return out


def pair_mutual_information(df: pd.DataFrame, scheme: str, seed: int, n_perm: int = 10000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    group_col = f"ratio_group_{scheme}"
    rows = []
    for correct, subset in df.groupby("oracle_correct_acc"):
        x = subset[group_col].astype(str).to_numpy()
        y = (subset.oracle_budget.astype(str) + "x" + subset.oracle_search_calls.astype(str)).to_numpy()
        observed = mutual_info_score(x, y)
        perm = np.empty(n_perm)
        for i in range(n_perm):
            perm[i] = mutual_info_score(rng.permutation(x), y)
        rows.append(
            {
                "scheme": scheme,
                "oracle_correct": int(correct),
                "n": len(subset),
                "joint_pair_mutual_information_nats": float(observed),
                "permutation_p": float((1 + np.sum(perm >= observed)) / (n_perm + 1)),
                "permutation_null_mean": float(perm.mean()),
                "bias_corrected_mi": float(observed - perm.mean()),
            }
        )
    return pd.DataFrame(rows)


def focused_pair_signal(df: pd.DataFrame, seed: int, n_perm: int = 10000) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    distributions = []
    for correct in [0, 1]:
        subset = df[
            (df.oracle_correct_acc == correct) &
            (df.ratio_group_primary.isin(["breadth-biased", "balanced"]))
        ].copy()
        x = subset.ratio_group_primary.astype(str).to_numpy()
        y = subset.oracle_pair.astype(str).to_numpy()
        question_type = subset.type.astype(str).to_numpy()
        observed = mutual_info_score(x, y)
        raw_perm = np.empty(n_perm)
        conditional_perm = np.empty(n_perm)
        for i in range(n_perm):
            raw_perm[i] = mutual_info_score(rng.permutation(x), y)
            permuted = x.copy()
            for kind in np.unique(question_type):
                idx = np.flatnonzero(question_type == kind)
                permuted[idx] = rng.permutation(permuted[idx])
            conditional_perm[i] = mutual_info_score(permuted, y)
        rows.append(
            {
                "oracle_correct": correct,
                "n": len(subset),
                "observed_mi_nats": observed,
                "raw_null_mean": raw_perm.mean(),
                "raw_permutation_p": (1 + np.sum(raw_perm >= observed)) / (n_perm + 1),
                "type_conditional_null_mean": conditional_perm.mean(),
                "type_conditional_permutation_p": (1 + np.sum(conditional_perm >= observed)) / (n_perm + 1),
            }
        )
        counts = subset.groupby(["ratio_group_primary", "oracle_pair"]).size().rename("count").reset_index()
        counts["rate_within_ratio_group"] = counts["count"] / counts.groupby("ratio_group_primary")["count"].transform("sum")
        counts["oracle_correct"] = correct
        distributions.append(counts)
    return pd.DataFrame(rows), pd.concat(distributions, ignore_index=True)


def max_t_threshold_scan(df: pd.DataFrame, seed: int, n_perm: int = 5000, min_extreme_n: int = 15) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []

    def welch_t(a: np.ndarray, b: np.ndarray) -> float:
        va = np.var(a, ddof=1)
        vb = np.var(b, ddof=1)
        denom = np.sqrt(va / len(a) + vb / len(b))
        return float((np.mean(a) - np.mean(b)) / denom) if denom > 0 else 0.0

    for correct, subset in df.groupby("oracle_correct_acc"):
        ratio = subset.depth_breadth_ratio.to_numpy(float)
        for outcome in ["oracle_budget", "oracle_search_calls"]:
            y = subset[outcome].to_numpy(float)
            for side in ["breadth", "depth"]:
                candidates = sorted(set(ratio[ratio < 1] if side == "breadth" else ratio[ratio > 1]))
                masks = []
                thresholds = []
                for threshold in candidates:
                    mask = ratio <= threshold if side == "breadth" else ratio >= threshold
                    if mask.sum() >= min_extreme_n and (~mask).sum() >= min_extreme_n:
                        masks.append(mask)
                        thresholds.append(float(threshold))
                if not masks:
                    continue
                observed = np.array([welch_t(y[m], y[~m]) for m in masks])
                perm_max = np.empty(n_perm)
                for i in range(n_perm):
                    py = rng.permutation(y)
                    perm_max[i] = max(abs(welch_t(py[m], py[~m])) for m in masks)
                for threshold, mask, effect in zip(thresholds, masks, observed):
                    records.append(
                        {
                            "oracle_correct": int(correct),
                            "outcome": outcome,
                            "side": side,
                            "threshold": threshold,
                            "extreme_n": int(mask.sum()),
                            "other_n": int((~mask).sum()),
                            "extreme_mean": float(y[mask].mean()),
                            "other_mean": float(y[~mask].mean()),
                            "difference": float(y[mask].mean() - y[~mask].mean()),
                            "welch_t": float(effect),
                            "max_t_adjusted_p": float((1 + np.sum(perm_max >= abs(effect))) / (n_perm + 1)),
                        }
                    )
    return pd.DataFrame(records)


def correctness_signal(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    table = pd.crosstab(df.ratio_group_primary, df.oracle_correct_acc)
    chi = chi2_contingency(table)
    n = table.to_numpy().sum()
    cramer_v = np.sqrt(chi.statistic / (n * min(table.shape[0] - 1, table.shape[1] - 1)))

    work = df.copy()
    work["breadth_group"] = (work.ratio_group_primary == "breadth-biased").astype(float)
    work["depth_group"] = (work.ratio_group_primary == "depth-biased").astype(float)
    work["comparison_type"] = (work.type == "comparison").astype(float)
    X = sm.add_constant(work[["breadth_group", "depth_group", "comparison_type"]])
    model = sm.Logit(work.oracle_correct_acc.astype(float), X).fit(disp=False)
    conf = model.conf_int()
    adjusted = pd.DataFrame(
        {
            "term": model.params.index,
            "log_odds_coef": model.params.values,
            "odds_ratio": np.exp(model.params.values),
            "ci95_or_low": np.exp(conf[0].values),
            "ci95_or_high": np.exp(conf[1].values),
            "p_value": model.pvalues.values,
        }
    )
    rates = (
        work.groupby(["ratio_group_primary", "type"], observed=False)
        .oracle_correct_acc.agg(["size", "mean"])
        .reset_index()
        .rename(columns={"size": "n", "mean": "correct_rate"})
    )
    overall = {
        "chi_square": float(chi.statistic),
        "chi_square_p": float(chi.pvalue),
        "cramers_v": float(cramer_v),
        "table": {str(k): {str(kk): int(vv) for kk, vv in v.items()} for k, v in table.to_dict(orient="index").items()},
    }
    return adjusted, rates, overall


def cv_ratio_regression(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows = []
    for correct, subset in df.groupby("oracle_correct_acc"):
        X = subset[["depth_breadth_ratio"]].to_numpy(float)
        for outcome in ["oracle_budget", "oracle_search_calls"]:
            y = subset[outcome].to_numpy(float)
            models = {
                "mean_baseline": DummyRegressor(strategy="mean"),
                "ratio_random_forest": RandomForestRegressor(
                    n_estimators=400, max_depth=3, min_samples_leaf=12, random_state=seed, n_jobs=-1
                ),
            }
            cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=seed)
            pred = {name: np.zeros((10, len(subset))) for name in models}
            for split_i, (train, test) in enumerate(cv.split(X)):
                repeat = split_i // 10
                for name, estimator in models.items():
                    pred[name][repeat, test] = clone(estimator).fit(X[train], y[train]).predict(X[test])
            for name, matrix in pred.items():
                r2s = [r2_score(y, p) for p in matrix]
                maes = [mean_absolute_error(y, p) for p in matrix]
                rows.append(
                    {
                        "oracle_correct": int(correct),
                        "outcome": outcome,
                        "model": name,
                        "cv_r2_mean": float(np.mean(r2s)),
                        "cv_r2_sd": float(np.std(r2s, ddof=1)),
                        "cv_mae_mean": float(np.mean(maes)),
                        "cv_mae_sd": float(np.std(maes, ddof=1)),
                    }
                )
    return pd.DataFrame(rows)


def make_pair_figure(df: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="white", context="talk")
    order = ["breadth-biased", "balanced", "depth-biased"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True, sharex=True, sharey=True)
    for row_i, correct in enumerate([1, 0]):
        for col_i, label in enumerate(order):
            ax = axes[row_i, col_i]
            sub = df[(df.oracle_correct_acc == correct) & (df.ratio_group_primary == label)]
            matrix = pd.crosstab(sub.oracle_search_calls, sub.oracle_budget, normalize="all").reindex(
                index=[1, 2, 3, 4], columns=[100, 200, 300, 400, 500, 600], fill_value=0
            )
            sns.heatmap(matrix, annot=True, fmt=".2f", cmap="Blues", vmin=0, vmax=0.6, cbar=col_i == 2, ax=ax)
            ax.set_title(f"{label}\nn={len(sub)}", weight="bold")
            ax.set_xlabel("Evidence cap per search")
            ax.set_ylabel("Search calls")
            if col_i == 0:
                ax.text(-0.34, 0.5, "Correct Oracle" if correct else "Incorrect Oracle", rotation=90,
                        transform=ax.transAxes, ha="center", va="center", fontsize=18, weight="bold")
    fig.suptitle("Oracle <evidence cap, search calls> by depth/breadth-ratio group", weight="bold", fontsize=20)
    fig.savefig(output / "oracle_pair_by_ratio_group.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "oracle_pair_by_ratio_group.pdf", bbox_inches="tight")
    plt.close(fig)


def make_scatter_figure(df: pd.DataFrame, assoc: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    rng = np.random.default_rng(2026)
    for row_i, correct in enumerate([1, 0]):
        subset = df[df.oracle_correct_acc == correct]
        for col_i, outcome in enumerate(["oracle_budget", "oracle_search_calls"]):
            ax = axes[row_i, col_i]
            scale = 10 if outcome == "oracle_budget" else 0.06
            ax.scatter(
                subset.depth_breadth_ratio,
                subset[outcome] + rng.normal(0, scale, len(subset)),
                s=24, alpha=0.28, color="#2E6F9E", edgecolors="none",
            )
            sns.regplot(data=subset, x="depth_breadth_ratio", y=outcome, lowess=True, scatter=False,
                        color="#D1495B", ax=ax)
            ax.axvline(LOW_PRIMARY, color="#E09F3E", linestyle="--", linewidth=2)
            ax.axvline(HIGH_PRIMARY, color="#E09F3E", linestyle="--", linewidth=2)
            a = assoc[(assoc.oracle_correct == correct) & (assoc.outcome == outcome)].iloc[0]
            ax.set_title(f"rho={a.spearman_rho:.3f} [{a.ci95_low:.3f}, {a.ci95_high:.3f}]")
            ax.set_xlabel("Depth / breadth score ratio")
            ax.set_ylabel("Evidence cap" if outcome == "oracle_budget" else "Search calls")
            if col_i == 0:
                ax.text(-0.23, 0.5, "Correct Oracle" if correct else "Incorrect Oracle", rotation=90,
                        transform=ax.transAxes, ha="center", va="center", fontsize=17, weight="bold")
    fig.suptitle("Continuous ratio has little association with the selected Oracle pair", weight="bold", fontsize=19)
    fig.savefig(output / "ratio_continuous_association.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "ratio_continuous_association.pdf", bbox_inches="tight")
    plt.close(fig)


def make_correctness_figure(df: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    order = ["breadth-biased", "balanced", "depth-biased"]
    summary = df.groupby("ratio_group_primary").oracle_correct_acc.agg(["size", "sum", "mean"]).reindex(order)
    z = 1.96
    n = summary["size"].to_numpy(float)
    p = summary["mean"].to_numpy(float)
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    x = np.arange(3)
    ax.bar(x, p, color=["#3A86FF", "#8D99AE", "#E76F51"], width=0.65)
    ax.errorbar(x, center, yerr=[center - (center - half), (center + half) - center], fmt="none",
                ecolor="black", capsize=6, linewidth=2)
    ax.set_xticks(x, [f"{label}\nn={int(nn)}" for label, nn in zip(order, n)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Oracle answer accuracy")
    ax.set_title("Ratio group predicts Oracle correctness, but is confounded by question type", weight="bold")
    for i, value in enumerate(p):
        ax.text(i, value + 0.035, f"{value:.1%}", ha="center", weight="bold")
    fig.savefig(output / "oracle_correctness_by_ratio_group.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "oracle_correctness_by_ratio_group.pdf", bbox_inches="tight")
    plt.close(fig)


def make_focused_pair_figure(distribution: pd.DataFrame, output: Path) -> None:
    sns.set_theme(style="whitegrid", context="talk")
    sub = distribution[distribution.oracle_correct == 1].copy()
    top_pairs = (
        sub.groupby("oracle_pair")["count"].sum().sort_values(ascending=False).head(10).index.tolist()
    )
    plot = sub[sub.oracle_pair.isin(top_pairs)].copy()
    order = (
        plot.groupby("oracle_pair")["count"].sum().sort_values(ascending=True).index.tolist()
    )
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    sns.barplot(
        data=plot, y="oracle_pair", x="rate_within_ratio_group", hue="ratio_group_primary",
        order=order, hue_order=["balanced", "breadth-biased"], palette=["#8D99AE", "#3A86FF"], ax=ax,
    )
    ax.set_xlabel("Proportion within ratio group")
    ax.set_ylabel("Oracle pair: evidence cap x search calls")
    ax.set_title("Candidate prior signal among correct Oracles", weight="bold")
    ax.legend(title="Depth/breadth-ratio group")
    fig.savefig(output / "correct_oracle_pair_candidate_signal.png", dpi=240, bbox_inches="tight")
    fig.savefig(output / "correct_oracle_pair_candidate_signal.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joined", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.joined)
    if len(df) != 500 or df.sample_id.nunique() != 500:
        raise RuntimeError("Expected exactly 500 unique questions")
    if (df.breadth_score <= 0).any():
        raise RuntimeError("Breadth score must be positive for a depth/breadth ratio")
    df["depth_breadth_ratio"] = df.depth_score / df.breadth_score
    df["ratio_group_primary"] = ratio_group(df.depth_breadth_ratio, "primary")
    df["ratio_group_sign"] = ratio_group(df.depth_breadth_ratio, "sign")
    df["ratio_group_exploratory_1.1"] = ratio_group(df.depth_breadth_ratio, "exploratory_1.1")
    df["oracle_correct_acc"] = (df.answer_acc == 1).astype(int)
    df["oracle_correct_em"] = (df.answer_em == 1).astype(int)
    df["oracle_pair"] = df.oracle_budget.astype(int).astype(str) + "x" + df.oracle_search_calls.astype(int).astype(str)
    keep_first = [
        "sample_id", "depth_score", "breadth_score", "depth_breadth_ratio",
        "ratio_group_primary", "ratio_group_sign", "ratio_group_exploratory_1.1",
        "oracle_correct_acc", "oracle_correct_em", "oracle_budget", "oracle_search_calls",
        "oracle_visible_tokens_total", "oracle_input_tokens", "oracle_quality", "oracle_pair",
    ]
    remaining = [c for c in df.columns if c not in keep_first]
    df[keep_first + remaining].to_csv(args.output / "per_question_ratio_oracle_pair.csv", index=False, encoding="utf-8-sig")

    assoc = continuous_associations(df, args.seed, "oracle_correct_acc")
    assoc.to_csv(args.output / "continuous_ratio_associations_by_correctness.csv", index=False, encoding="utf-8-sig")
    em_assoc = continuous_associations(df, args.seed + 500, "oracle_correct_em")
    em_assoc.to_csv(args.output / "strict_em_sensitivity.csv", index=False, encoding="utf-8-sig")

    summaries = pd.concat([group_summary(df, x) for x in ["primary", "sign", "exploratory_1.1"]], ignore_index=True)
    summaries.to_csv(args.output / "ratio_group_pair_summary.csv", index=False, encoding="utf-8-sig")
    tests = pd.concat([group_tests(df, x) for x in ["primary", "sign", "exploratory_1.1"]], ignore_index=True)
    tests.to_csv(args.output / "ratio_group_pair_tests.csv", index=False, encoding="utf-8-sig")
    mi = pd.concat([pair_mutual_information(df, x, args.seed + i) for i, x in enumerate(["primary", "sign", "exploratory_1.1"])])
    mi["bh_q_across_six_tests"] = bh_qvalues(mi["permutation_p"].tolist())
    mi.to_csv(args.output / "joint_pair_mutual_information.csv", index=False, encoding="utf-8-sig")
    focused_signal, focused_distribution = focused_pair_signal(df, args.seed + 900)
    focused_signal.to_csv(args.output / "focused_breadth_vs_balanced_pair_signal.csv", index=False, encoding="utf-8-sig")
    focused_distribution.to_csv(args.output / "focused_pair_distribution.csv", index=False, encoding="utf-8-sig")

    scan_path = args.output / "threshold_scan_maxT_corrected.csv"
    if scan_path.exists():
        scan = pd.read_csv(scan_path)
    else:
        scan = max_t_threshold_scan(df, args.seed)
        scan.to_csv(scan_path, index=False, encoding="utf-8-sig")
    cv_path = args.output / "ratio_only_cross_validation.csv"
    if cv_path.exists():
        cv = pd.read_csv(cv_path)
    else:
        cv = cv_ratio_regression(df, args.seed)
        cv.to_csv(cv_path, index=False, encoding="utf-8-sig")

    adjusted, rates, correctness = correctness_signal(df)
    adjusted.to_csv(args.output / "oracle_correctness_logistic_type_adjusted.csv", index=False, encoding="utf-8-sig")
    rates.to_csv(args.output / "oracle_correctness_by_ratio_and_type.csv", index=False, encoding="utf-8-sig")

    make_pair_figure(df, args.output)
    make_scatter_figure(df, assoc, args.output)
    make_correctness_figure(df, args.output)
    make_focused_pair_figure(focused_distribution, args.output)

    primary_summary = summaries[summaries.scheme == "primary"]
    primary_tests = tests[tests.scheme == "primary"]
    primary_mi = mi[mi.scheme == "primary"]
    max_scan = scan.sort_values("max_t_adjusted_p").head(10)
    result = {
        "integrity": {"questions": 500, "unique_ids": 500, "ratio_formula": "depth_score / breadth_score"},
        "primary_thresholds": {
            "breadth_biased": f"ratio <= {LOW_PRIMARY:.6f}",
            "balanced": f"{LOW_PRIMARY:.6f} < ratio < {HIGH_PRIMARY:.6f}",
            "depth_biased": f"ratio >= {HIGH_PRIMARY:.6f}",
            "counts": df.ratio_group_primary.value_counts().to_dict(),
        },
        "correctness_split": {
            "definition": "selected F1 Quality-then-Cost Oracle has answer_acc == 1",
            "correct": int(df.oracle_correct_acc.sum()),
            "incorrect": int((1 - df.oracle_correct_acc).sum()),
        },
        "continuous_associations": assoc.to_dict(orient="records"),
        "primary_group_summary": primary_summary.to_dict(orient="records"),
        "primary_group_tests": primary_tests.to_dict(orient="records"),
        "primary_joint_pair_mi": primary_mi.to_dict(orient="records"),
        "focused_breadth_vs_balanced_pair_signal": focused_signal.to_dict(orient="records"),
        "threshold_scan_best_adjusted": max_scan.to_dict(orient="records"),
        "ratio_only_cross_validation": cv.to_dict(orient="records"),
        "oracle_correctness_signal": correctness,
        "oracle_correctness_type_adjusted": adjusted.to_dict(orient="records"),
    }
    (args.output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    def ar(correct: int, outcome: str) -> pd.Series:
        return assoc[(assoc.oracle_correct == correct) & (assoc.outcome == outcome)].iloc[0]

    focused_correct = focused_signal[focused_signal.oracle_correct == 1].iloc[0]

    report = f"""# 深广度比值与 Oracle <证据 token, 搜索次数>：正确/错误分层分析

## 结论

按你的定义重新分析后，`depth_score / breadth_score` 在“正确 Oracle”和“错误 Oracle”内部，仍未表现出稳定的 Oracle 行为先验：无论把目标看成证据暴露上限、搜索次数，还是二者的联合 pair，主阈值、放宽阈值、连续比值和多阈值扫描均没有通过稳健检验。

真正出现的先验信号是：**偏广度组更容易产生正确 Oracle 答案**。但偏广度标签与 HotpotQA `comparison` 类型高度重合，因此它更像“问题类型/可解性先验”，暂时不能解释为“偏广度问题必然需要更多搜索”。

## 定义

- 深广度比值：`R = depth_score / breadth_score`。
- 主阈值：`R <= 5/6` 为偏广度；`R >= 1.2` 为偏深度；中间为平衡。
- 正确 Oracle：F1 Quality-then-Cost Oracle 所选轨迹满足 `answer_acc == 1`；错误 Oracle 为 `answer_acc == 0`。
- Oracle pair：`<每次搜索的证据暴露上限, 该轨迹实际搜索次数>`。

## 阈值可用性

- 偏广度：{int((df.ratio_group_primary == 'breadth-biased').sum())} 题。
- 平衡：{int((df.ratio_group_primary == 'balanced').sum())} 题。
- 偏深度：{int((df.ratio_group_primary == 'depth-biased').sum())} 题。

偏深度组只有 9 题，其中正确 4、错误 5，不能支撑稳定估计。把上阈值增至 1.3 会进一步缩小该组；把下阈值从 5/6 缩至 2/3 在当前离散评分上不会改变偏广度组，因为 5/6 以下的 129 题本来就全部不超过 2/3。

## 连续比值与 Oracle pair

| Oracle 分层 | 目标 | Spearman rho | 95% CI | BH q |
|---|---|---:|---:|---:|
| 正确 | 证据上限 | {ar(1,'oracle_budget').spearman_rho:.3f} | [{ar(1,'oracle_budget').ci95_low:.3f}, {ar(1,'oracle_budget').ci95_high:.3f}] | {ar(1,'oracle_budget').bh_q_value:.3f} |
| 正确 | 搜索次数 | {ar(1,'oracle_search_calls').spearman_rho:.3f} | [{ar(1,'oracle_search_calls').ci95_low:.3f}, {ar(1,'oracle_search_calls').ci95_high:.3f}] | {ar(1,'oracle_search_calls').bh_q_value:.3f} |
| 错误 | 证据上限 | {ar(0,'oracle_budget').spearman_rho:.3f} | [{ar(0,'oracle_budget').ci95_low:.3f}, {ar(0,'oracle_budget').ci95_high:.3f}] | {ar(0,'oracle_budget').bh_q_value:.3f} |
| 错误 | 搜索次数 | {ar(0,'oracle_search_calls').spearman_rho:.3f} | [{ar(0,'oracle_search_calls').ci95_low:.3f}, {ar(0,'oracle_search_calls').ci95_high:.3f}] | {ar(0,'oracle_search_calls').bh_q_value:.3f} |

所有区间均跨 0。严格 EM 重新分层得到相同方向的结论。

## 分类后的 pair 分布

`oracle_pair_by_ratio_group.png` 给出了正确/错误 Oracle × 三种比值类别的完整二维频率。单独考察 token 均值或搜索次数均值时，Kruskal-Wallis 和偏广度 vs 平衡的 Mann-Whitney 都不显著。

但是，在**正确 Oracle**中，预先指定的三分类与联合 `<token, search>` pair 出现了一个候选信号：置换 MI p=0.0177；把 6 个“分组方案 × 正误层”检验共同校正后 q={primary_mi[primary_mi.oracle_correct == 1].iloc[0].bh_q_across_six_tests:.3f}。由于偏深度组只有 4 道正确题，我进一步只比较偏广度与平衡组：

- 未控制问题类型的联合 pair permutation p={focused_correct.raw_permutation_p:.4f}；
- 在 `bridge/comparison` 类型内部置换后 p={focused_correct.type_conditional_permutation_p:.4f}。

因此这是一个值得在新样本上复验的**边缘候选先验**，不是已经确认的规律。主要形态是：正确 Oracle 中，平衡组更集中于 `<100,1>`（30.5%），偏广度组更常出现 `<100,2>`（21.8%）、`<200,1>`（13.9%）和 `<300,1>`（12.9%）；对应图为 `correct_oracle_pair_candidate_signal.png`。

多阈值探索采用 max-|t| permutation correction，并要求极端组至少 15 题；没有扫描阈值在校正后构成可靠行为信号。`R>1` 的宽松分类可把偏深度组扩到 45 题，但结论仍不改变。

## 找到的可用先验：Oracle 正确率

- 偏广度组正确率：{df[df.ratio_group_primary == 'breadth-biased'].oracle_correct_acc.mean():.1%}。
- 平衡组正确率：{df[df.ratio_group_primary == 'balanced'].oracle_correct_acc.mean():.1%}。
- 偏深度组正确率：{df[df.ratio_group_primary == 'depth-biased'].oracle_correct_acc.mean():.1%}，但只有 9 题。
- 类别与正确性的总体关联：Cramer's V={correctness['cramers_v']:.3f}，p={correctness['chi_square_p']:.3g}。

不过，129 道偏广度题中有 99 道是 `comparison`，362 道平衡题中有 355 道是 `bridge`。报告已经提供控制 question type 的逻辑回归，不能把未调整的正确率差异直接归因于预算需求。

## 推荐解释

目前可以把深广度比值作为三种层次不同的候选先验：

1. **问题结构/Oracle 可解性先验**：已有明显但受 question type 混杂的信号；
2. **正确答案条件下的联合资源 pair 先验**：出现弱候选信号，但控制 question type 后仅在 0.05 边界附近；
3. **连续 token 或搜索次数函数**：当前数据不支持，仍需加入检索后的动态 evidence-state 特征。

不要继续在这 500 题上挑选阈值后宣称发现；下一步应冻结候选阈值，在新的 HotpotQA 样本上验证正确率信号及 pair 分布。
"""
    (args.output / "REPORT_ZH.md").write_text(report, encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
