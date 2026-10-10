from __future__ import annotations

import gzip
import json
import math
from collections import Counter, defaultdict
from itertools import combinations_with_replacement
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, kruskal, spearmanr


BASE = Path(__file__).resolve().parent
SCORES_PATH = BASE / "read/hotpot_random500_openrouter_gpt56sol_read_v2_fresh/results/read_scores_combined.jsonl"
SCORES_AUDIT = BASE / "read/hotpot_random500_openrouter_gpt56sol_read_v2_fresh/audit/final_audit.json"
SCORES_CONFIG = BASE / "read/hotpot_random500_openrouter_gpt56sol_read_v2_fresh/run_config.json"
SCORES_STATES = BASE / "read/hotpot_random500_openrouter_gpt56sol_read_v2_fresh/state/read"
TRAJ_PATH = BASE / "traj/analysis/scored_records.jsonl.gz"
TRAJ_MANIFEST = BASE / "traj/trajectories/trajectories.merged.jsonl.gz.manifest.json"
OUT = BASE / "results"

BUDGETS = [100, 200, 300, 400, 500, 600]
METRICS = ["answer_f1", "answer_em", "answer_acc"]
SEED = 2026
N_BOOT = 10000


def jsonl(path: Path, gz: bool = False):
    opener = gzip.open if gz else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return float(value)


def rank_corr(x, y):
    valid = [(a, b) for a, b in zip(x, y) if pd.notna(a) and pd.notna(b)]
    if len(valid) < 3:
        return {"n": len(valid), "rho": None, "p": None}
    stat = spearmanr([a for a, _ in valid], [b for _, b in valid])
    return {"n": len(valid), "rho": float(stat.statistic), "p": float(stat.pvalue)}


def bootstrap_delta(values, seed=SEED, n_boot=N_BOOT):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    chunks = []
    left = n_boot
    while left:
        n = min(left, 1000)
        idx = rng.integers(0, len(values), size=(n, len(values)))
        chunks.append(values[idx].mean(axis=1))
        left -= n
    means = np.concatenate(chunks)
    return {
        "mean": float(values.mean()),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
        "bootstrap_probability_gt_zero": float(np.mean(means > 0)),
    }


def select_budget(train: pd.DataFrame, metric: str, ids=None, alpha=0.0):
    if ids is not None:
        group = train[train.sample_id.isin(ids)]
    else:
        group = train
    table = group.groupby("budget").agg(
        quality=(metric, "mean"),
        cost=("total_llm_tokens", "mean"),
    )
    if alpha > 0 and ids is not None:
        global_table = train.groupby("budget").agg(
            global_quality=(metric, "mean"),
            global_cost=("total_llm_tokens", "mean"),
        )
        n_ids = len(set(ids))
        table = table.join(global_table)
        table["quality"] = (
            n_ids * table["quality"] + alpha * table["global_quality"]
        ) / (n_ids + alpha)
        table["cost"] = (
            n_ids * table["cost"] + alpha * table["global_cost"]
        ) / (n_ids + alpha)
    row = table.reset_index().sort_values(
        ["quality", "cost", "budget"], ascending=[False, True, True]
    ).iloc[0]
    return int(row.budget)


def bucket(score):
    if score <= 1:
        return "low_1"
    if score == 2:
        return "mid_2"
    return "high_3plus"


def loo_predictions(df: pd.DataFrame, scores: pd.DataFrame, metric: str):
    ids = sorted(df.sample_id.unique())
    id_to_i = {sid: i for i, sid in enumerate(ids)}
    budget_to_j = {budget: j for j, budget in enumerate(BUDGETS)}
    score_by_id = scores.set_index("sample_id").read_score.to_dict()
    bucket_by_id = {sid: bucket(score_by_id[sid]) for sid in ids}
    by_score = defaultdict(list)
    by_bucket = defaultdict(list)
    for sid in ids:
        by_score[score_by_id[sid]].append(sid)
        by_bucket[bucket_by_id[sid]].append(sid)

    pivot_quality = df.pivot(index="sample_id", columns="budget", values=metric).loc[ids, BUDGETS]
    pivot_cost = df.pivot(index="sample_id", columns="budget", values="total_llm_tokens").loc[ids, BUDGETS]
    q_matrix = pivot_quality.to_numpy(float)
    c_matrix = pivot_cost.to_numpy(float)
    global_q_sum = q_matrix.sum(axis=0)
    global_c_sum = c_matrix.sum(axis=0)
    score_q_sum = {key: q_matrix[[id_to_i[x] for x in members]].sum(axis=0) for key, members in by_score.items()}
    score_c_sum = {key: c_matrix[[id_to_i[x] for x in members]].sum(axis=0) for key, members in by_score.items()}
    bucket_q_sum = {key: q_matrix[[id_to_i[x] for x in members]].sum(axis=0) for key, members in by_bucket.items()}
    bucket_c_sum = {key: c_matrix[[id_to_i[x] for x in members]].sum(axis=0) for key, members in by_bucket.items()}
    rows = []
    monotone_maps = list(combinations_with_replacement(BUDGETS, 3))

    def choose(q_mean, c_mean):
        return int(sorted(range(len(BUDGETS)), key=lambda j: (-q_mean[j], c_mean[j], BUDGETS[j]))[0])

    for held in ids:
        i = id_to_i[held]
        score = score_by_id[held]
        coarse = bucket_by_id[held]
        global_q = (global_q_sum - q_matrix[i]) / (len(ids) - 1)
        global_c = (global_c_sum - c_matrix[i]) / (len(ids) - 1)
        score_n = len(by_score[score]) - 1
        score_q = (score_q_sum[score] - q_matrix[i]) / score_n
        score_c = (score_c_sum[score] - c_matrix[i]) / score_n
        bucket_n = len(by_bucket[coarse]) - 1
        coarse_q = (bucket_q_sum[coarse] - q_matrix[i]) / bucket_n
        coarse_c = (bucket_c_sum[coarse] - c_matrix[i]) / bucket_n
        shrunk_q = (score_n * score_q + 20.0 * global_q) / (score_n + 20.0)
        shrunk_c = (score_n * score_c + 20.0 * global_c) / (score_n + 20.0)

        choices = {
            "fixed_500": budget_to_j[500],
            "fixed_600": budget_to_j[600],
            "loo_global": choose(global_q, global_c),
            "loo_exact_score": choose(score_q, score_c),
            "loo_coarse_score": choose(coarse_q, coarse_c),
            "loo_shrunk_score_a20": choose(shrunk_q, shrunk_c),
        }

        # A train-only, interpretable monotone lookup: low <= medium <= high budget.
        coarse_keys = ["low_1", "mid_2", "high_3plus"]
        train_group_q = {}
        train_group_c = {}
        train_group_n = {}
        for key in coarse_keys:
            train_group_q[key] = bucket_q_sum[key] - (q_matrix[i] if coarse == key else 0)
            train_group_c[key] = bucket_c_sum[key] - (c_matrix[i] if coarse == key else 0)
            train_group_n[key] = len(by_bucket[key]) - int(coarse == key)
        best = None
        for mapping_tuple in monotone_maps:
            mapping = dict(zip(coarse_keys, mapping_tuple))
            q_sum = sum(train_group_q[key][budget_to_j[mapping[key]]] for key in coarse_keys)
            c_sum = sum(train_group_c[key][budget_to_j[mapping[key]]] for key in coarse_keys)
            key = (float(q_sum / (len(ids) - 1)), -float(c_sum / (len(ids) - 1)), -sum(mapping_tuple))
            if best is None or key > best[0]:
                best = (key, mapping)
        choices["loo_monotone_coarse"] = budget_to_j[best[1][coarse]]

        for policy, chosen_j in choices.items():
            chosen = BUDGETS[chosen_j]
            record = df[(df.sample_id == held) & (df.budget == chosen)].iloc[0]
            rows.append({
                "sample_id": held,
                "read_score": score,
                "score_bucket": coarse,
                "optimized_metric": metric,
                "policy": policy,
                "selected_budget": chosen,
                **{col: float(record[col]) for col in METRICS},
                "input_tokens": float(record.standalone_llm_input_tokens),
                "output_tokens": float(record.standalone_llm_output_tokens),
                "total_llm_tokens": float(record.total_llm_tokens),
                "visible_evidence_tokens": float(record.visible_evidence_tokens_total),
                "search_calls": float(record.search_calls),
            })
    return pd.DataFrame(rows)


def policy_summary(pred: pd.DataFrame):
    return pred.groupby(["optimized_metric", "policy"]).agg(
        n=("sample_id", "nunique"),
        mean_selected_budget=("selected_budget", "mean"),
        answer_em=("answer_em", "mean"),
        answer_f1=("answer_f1", "mean"),
        answer_acc=("answer_acc", "mean"),
        input_tokens=("input_tokens", "mean"),
        output_tokens=("output_tokens", "mean"),
        total_llm_tokens=("total_llm_tokens", "mean"),
        visible_evidence_tokens=("visible_evidence_tokens", "mean"),
        search_calls=("search_calls", "mean"),
    ).reset_index()


def paired_policy_deltas(pred: pd.DataFrame):
    rows = []
    for optimized_metric in METRICS:
        sub = pred[pred.optimized_metric == optimized_metric]
        base = sub[sub.policy == "loo_global"].set_index("sample_id")
        for policy in sorted(set(sub.policy) - {"loo_global"}):
            cand = sub[sub.policy == policy].set_index("sample_id").loc[base.index]
            for outcome in ["answer_f1", "answer_em", "answer_acc", "total_llm_tokens"]:
                delta = cand[outcome].to_numpy(float) - base[outcome].to_numpy(float)
                result = bootstrap_delta(delta, seed=SEED + len(rows))
                discordant_plus = int(np.sum((cand[outcome] > base[outcome]).to_numpy()))
                discordant_minus = int(np.sum((cand[outcome] < base[outcome]).to_numpy()))
                p = None
                if outcome in ("answer_em", "answer_acc") and discordant_plus + discordant_minus:
                    p = float(binomtest(discordant_plus, discordant_plus + discordant_minus, 0.5).pvalue)
                rows.append({
                    "optimized_metric": optimized_metric,
                    "policy": policy,
                    "reference": "loo_global",
                    "outcome": outcome,
                    **result,
                    "questions_better": discordant_plus,
                    "questions_worse": discordant_minus,
                    "paired_binary_exact_p": p,
                })
    return pd.DataFrame(rows)


def make_oracles(df: pd.DataFrame, scores: pd.DataFrame):
    score_map = scores.set_index("sample_id").to_dict("index")
    rows = []
    for sid, group in df.groupby("sample_id"):
        base = {
            "sample_id": sid,
            "query": score_map[sid]["query"],
            "read_score": int(score_map[sid]["read_score"]),
            "read_confidence": float(score_map[sid]["read_confidence"]),
        }
        ordered = group.sort_values("budget")
        for metric in METRICS:
            chosen = group.sort_values(
                [metric, "total_llm_tokens", "budget"], ascending=[False, True, True]
            ).iloc[0]
            base[f"oracle_budget_{metric}"] = int(chosen.budget)
            base[f"oracle_quality_{metric}"] = float(chosen[metric])
            base[f"oracle_total_tokens_{metric}"] = float(chosen.total_llm_tokens)
        for metric in ["answer_em", "answer_acc"]:
            vals = ordered[metric].to_numpy(float)
            budgets = ordered.budget.to_numpy(int)
            ones = np.where(vals == 1)[0]
            base[f"ever_correct_{metric}"] = bool(len(ones))
            base[f"msb_{metric}"] = int(budgets[ones[0]]) if len(ones) else None
            stable = [i for i in range(len(vals)) if vals[i] == 1 and np.all(vals[i:] == 1)]
            base[f"ssb_{metric}"] = int(budgets[stable[0]]) if stable else None
            base[f"any_regression_{metric}"] = bool(any(vals[i] == 1 and vals[i + 1] == 0 for i in range(len(vals) - 1)))
        rows.append(base)
    return pd.DataFrame(rows)


def association_tables(df: pd.DataFrame, scores: pd.DataFrame, oracle: pd.DataFrame):
    process = df.groupby("sample_id").agg(
        mean_search_calls=("search_calls", "mean"),
        mean_visible_evidence_tokens=("visible_evidence_tokens_total", "mean"),
        mean_input_tokens=("standalone_llm_input_tokens", "mean"),
        mean_output_tokens=("standalone_llm_output_tokens", "mean"),
        mean_total_tokens=("total_llm_tokens", "mean"),
        mean_f1=("answer_f1", "mean"),
        mean_em=("answer_em", "mean"),
        mean_acc=("answer_acc", "mean"),
    ).reset_index().merge(scores, on="sample_id", validate="one_to_one")

    correlation_rows = []
    for outcome in [
        "mean_search_calls", "mean_visible_evidence_tokens", "mean_input_tokens",
        "mean_output_tokens", "mean_total_tokens", "mean_f1", "mean_em", "mean_acc",
    ]:
        correlation_rows.append({"predictor": "read_score", "outcome": outcome, **rank_corr(process.read_score, process[outcome])})
    for metric in METRICS:
        correlation_rows.append({
            "predictor": "read_score", "outcome": f"oracle_budget_{metric}",
            **rank_corr(oracle.read_score, oracle[f"oracle_budget_{metric}"]),
        })
    high_f1 = (oracle.oracle_budget_answer_f1 >= 500).astype(int)
    correlation_rows.append({
        "predictor": "read_score", "outcome": "oracle_f1_budget_at_least_500",
        **rank_corr(oracle.read_score, high_f1),
    })
    for metric in ["answer_em", "answer_acc"]:
        solved = oracle[oracle[f"ever_correct_{metric}"]]
        correlation_rows.append({
            "predictor": "read_score", "outcome": f"msb_{metric}_among_ever_correct",
            **rank_corr(solved.read_score, solved[f"msb_{metric}"]),
        })
        stable = oracle[oracle[f"ssb_{metric}"].notna()]
        correlation_rows.append({
            "predictor": "read_score", "outcome": f"ssb_{metric}_among_stably_correct",
            **rank_corr(stable.read_score, stable[f"ssb_{metric}"]),
        })
    correlations = pd.DataFrame(correlation_rows)

    group_rows = []
    for score, group in process.groupby("read_score"):
        o = oracle[oracle.read_score == score]
        row = {
            "read_score": int(score), "n": len(group),
            **{c: float(group[c].mean()) for c in group.columns if c.startswith("mean_")},
        }
        for metric in METRICS:
            row[f"mean_oracle_budget_{metric}"] = float(o[f"oracle_budget_{metric}"].mean())
            row[f"mean_oracle_quality_{metric}"] = float(o[f"oracle_quality_{metric}"].mean())
        for metric in ["answer_em", "answer_acc"]:
            row[f"ever_correct_rate_{metric}"] = float(o[f"ever_correct_{metric}"].mean())
            row[f"mean_msb_{metric}_solved"] = safe_float(o.loc[o[f"ever_correct_{metric}"], f"msb_{metric}"].mean())
            row[f"regression_rate_{metric}"] = float(o[f"any_regression_{metric}"].mean())
        group_rows.append(row)
    return process, correlations, pd.DataFrame(group_rows)


def bootstrap_budget_selection(df: pd.DataFrame, scores: pd.DataFrame, n_boot=10000):
    merged = df.merge(scores[["sample_id", "read_score"]], on="sample_id", validate="many_to_one")
    merged["score_bucket"] = merged.read_score.map(bucket)
    rng = np.random.default_rng(SEED)
    rows = []
    for metric in METRICS:
        q = merged.pivot(index="sample_id", columns="budget", values=metric)[BUDGETS]
        c = merged.pivot(index="sample_id", columns="budget", values="total_llm_tokens")[BUDGETS]
        meta = scores.set_index("sample_id").loc[q.index]
        meta["score_bucket"] = meta.read_score.map(bucket)
        for group_name, group_ids in meta.groupby("score_bucket").groups.items():
            ids = list(group_ids)
            qg = q.loc[ids].to_numpy(float)
            cg = c.loc[ids].to_numpy(float)
            counts = Counter()
            margins = []
            for _ in range(n_boot):
                idx = rng.integers(0, len(ids), len(ids))
                qm = qg[idx].mean(axis=0)
                cm = cg[idx].mean(axis=0)
                order = sorted(range(len(BUDGETS)), key=lambda j: (-qm[j], cm[j], BUDGETS[j]))
                counts[BUDGETS[order[0]]] += 1
                margins.append(float(qm[order[0]] - qm[order[1]]))
            for budget in BUDGETS:
                rows.append({
                    "optimized_metric": metric, "score_bucket": group_name,
                    "group_n": len(ids), "selected_budget": budget,
                    "bootstrap_selection_probability": counts[budget] / n_boot,
                    "mean_winner_margin": float(np.mean(margins)),
                })
    return pd.DataFrame(rows)


def compare_em_rule_to_fixed600(predictions: pd.DataFrame, df: pd.DataFrame, scores: pd.DataFrame):
    sub = predictions[predictions.optimized_metric == "answer_em"]
    rule = sub[sub.policy == "loo_coarse_score"].set_index("sample_id")
    base = sub[sub.policy == "fixed_600"].set_index("sample_id").loc[rule.index]
    rows = []
    for outcome in ["answer_f1", "answer_em", "answer_acc", "total_llm_tokens"]:
        delta = rule[outcome].to_numpy(float) - base[outcome].to_numpy(float)
        stat = bootstrap_delta(delta, seed=SEED + 811 + len(rows))
        plus = int(np.sum(delta > 0))
        minus = int(np.sum(delta < 0))
        p = None
        if outcome in ("answer_em", "answer_acc") and plus + minus:
            p = float(binomtest(plus, plus + minus, 0.5).pvalue)
        rows.append({
            "rule": "READ=2 -> 600; READ=1 or >=3 -> 500",
            "reference": "fixed_600", "outcome": outcome, **stat,
            "questions_better": plus, "questions_worse": minus,
            "paired_binary_exact_p": p,
        })

    score_map = scores.set_index("sample_id")
    df_index = df.set_index(["sample_id", "budget"])
    cases = []
    for sid in rule.index:
        chosen = int(rule.loc[sid, "selected_budget"])
        if chosen == 600:
            continue
        r = df_index.loc[(sid, chosen)]
        b = df_index.loc[(sid, 600)]
        changes = {
            "em_delta": float(r.answer_em - b.answer_em),
            "f1_delta": float(r.answer_f1 - b.answer_f1),
            "acc_delta": float(r.answer_acc - b.answer_acc),
        }
        if not any(value != 0 for value in changes.values()):
            continue
        cases.append({
            "sample_id": sid, "query": score_map.loc[sid, "query"],
            "read_score": int(score_map.loc[sid, "read_score"]),
            "rule_budget": chosen, **changes,
            "total_token_delta": float(r.total_llm_tokens - b.total_llm_tokens),
            "search_calls_rule": float(r.search_calls), "search_calls_600": float(b.search_calls),
            "visible_tokens_rule": float(r.visible_evidence_tokens_total), "visible_tokens_600": float(b.visible_evidence_tokens_total),
            "retrieved_title_recall_rule": float(r.cumulative_retrieved_title_recall),
            "retrieved_title_recall_600": float(b.cumulative_retrieved_title_recall),
        })
    return pd.DataFrame(rows), pd.DataFrame(cases)


def plot_curves(curves: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.7), constrained_layout=True)
    palette = {1: "#4C78A8", 2: "#59A14F", 3: "#F28E2B", 4: "#E15759"}
    for score, group in curves.groupby("read_score"):
        axes[0].plot(group.budget, group.answer_f1, marker="o", lw=2.2,
                     label=f"READ={score} (n={int(group.n.iloc[0])})", color=palette.get(score))
        axes[1].plot(group.budget, group.total_llm_tokens, marker="o", lw=2.2,
                     label=f"READ={score}", color=palette.get(score))
    axes[0].set(title="Held-out task quality by READ-score group", xlabel="Evidence cap per search (tokens)", ylabel="Mean answer F1")
    axes[1].set(title="Actual end-to-end token use", xlabel="Evidence cap per search (tokens)", ylabel="Mean LLM input + output tokens")
    axes[0].legend(frameon=True)
    axes[1].legend(frameon=True)
    fig.suptitle("Query-only READ score: demand signal or budget selector?", fontsize=17, weight="bold")
    fig.savefig(OUT / "read_score_budget_curves.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "read_score_budget_curves.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_policy(summary: pd.DataFrame, deltas: pd.DataFrame):
    sub = summary[summary.optimized_metric == "answer_f1"].copy()
    order = ["fixed_500", "fixed_600", "loo_global", "loo_exact_score", "loo_coarse_score", "loo_shrunk_score_a20", "loo_monotone_coarse"]
    sub = sub.set_index("policy").loc[order].reset_index()
    labels = {
        "fixed_500": "Fixed 500", "fixed_600": "Fixed 600", "loo_global": "LOO global",
        "loo_exact_score": "READ lookup", "loo_coarse_score": "Coarse READ",
        "loo_shrunk_score_a20": "Shrunk READ", "loo_monotone_coarse": "Monotone READ",
    }
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    axes[0].scatter(sub.total_llm_tokens, sub.answer_f1, s=85, color="#276FBF")
    offsets = {
        "fixed_500": (7, -13), "fixed_600": (7, 8), "loo_global": (7, 22),
        "loo_exact_score": (7, 16), "loo_coarse_score": (7, -17),
        "loo_shrunk_score_a20": (7, -2), "loo_monotone_coarse": (7, 5),
    }
    for _, row in sub.iterrows():
        axes[0].annotate(labels[row.policy], (row.total_llm_tokens, row.answer_f1), xytext=offsets[row.policy], textcoords="offset points", fontsize=9)
    axes[0].set(title="Quality-cost outcome", xlabel="Mean total LLM tokens", ylabel="Held-out F1")
    d = deltas[(deltas.optimized_metric == "answer_f1") & (deltas.outcome == "answer_f1")].copy()
    d = d[d.policy.isin(order)].set_index("policy").loc[[x for x in order if x != "loo_global"]].reset_index()
    y = np.arange(len(d))
    axes[1].errorbar(d["mean"], y, xerr=[d["mean"] - d.ci95_low, d.ci95_high - d["mean"]], fmt="o", capsize=4, color="#D1495B")
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_yticks(y, [labels[x] for x in d.policy])
    axes[1].set(title="Paired F1 change vs LOO global", xlabel="Mean difference (bootstrap 95% CI)")
    fig.suptitle("Can READ-score rules beat a fixed/global budget?", fontsize=17, weight="bold")
    fig.savefig(OUT / "policy_quality_cost.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "policy_quality_cost.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_oracle_signal(oracle: pd.DataFrame, process: pd.DataFrame):
    grouped = oracle.groupby("read_score").agg(
        n=("sample_id", "size"),
        mean_oracle_budget=("oracle_budget_answer_f1", "mean"),
        high_budget_rate=("oracle_budget_answer_f1", lambda x: float((x >= 500).mean())),
    ).reset_index()
    proc = process.groupby("read_score").agg(mean_total_tokens=("mean_total_tokens", "mean")).reset_index()
    cross = pd.crosstab(oracle.read_score, oracle.oracle_budget_answer_f1, normalize="index").reindex(columns=BUDGETS, fill_value=0)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    bars = axes[0].bar(grouped.read_score.astype(str), grouped.high_budget_rate, color="#4C78A8")
    axes[0].bar_label(bars, labels=[f"{v:.1%}\n(n={n})" for v, n in zip(grouped.high_budget_rate, grouped.n)], padding=3, fontsize=9)
    axes[0].set(title="Oracle chooses 500–600 tokens", xlabel="Query-only READ score", ylabel="Proportion of questions", ylim=(0, max(.55, grouped.high_budget_rate.max() + .12)))
    bars = axes[1].bar(proc.read_score.astype(str), proc.mean_total_tokens, color="#F28E2B")
    axes[1].bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    axes[1].set(title="Observed end-to-end token demand", xlabel="Query-only READ score", ylabel="Mean input + output tokens")
    bottom = np.zeros(len(cross))
    colors = plt.cm.viridis(np.linspace(.15, .9, len(BUDGETS)))
    for color, budget in zip(colors, BUDGETS):
        vals = cross[budget].to_numpy(float)
        axes[2].bar(cross.index.astype(str), vals, bottom=bottom, label=str(budget), color=color)
        bottom += vals
    axes[2].set(title="Quality-then-cost oracle distribution", xlabel="Query-only READ score", ylabel="Within-group proportion", ylim=(0, 1))
    axes[2].legend(title="Oracle cap", ncol=2, fontsize=8)
    fig.suptitle("READ score predicts demand better than the exact optimal cap", fontsize=17, weight="bold")
    fig.savefig(OUT / "oracle_mapping_signal.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "oracle_mapping_signal.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(jsonl(SCORES_PATH)).rename(columns={"question_id": "sample_id"})
    traj_all = pd.DataFrame(jsonl(TRAJ_PATH, gz=True))
    traj_all["budget_numeric"] = pd.to_numeric(traj_all["condition_group"], errors="coerce")
    df = traj_all[traj_all.budget_numeric.isin(BUDGETS)].copy()
    df["budget"] = df.budget_numeric.astype(int)
    df["total_llm_tokens"] = df.standalone_llm_input_tokens + df.standalone_llm_output_tokens

    counts = df.groupby("sample_id").budget.nunique()
    audit = json.loads(SCORES_AUDIT.read_text(encoding="utf-8"))
    config = json.loads(SCORES_CONFIG.read_text(encoding="utf-8"))
    state_hashes = Counter()
    for path in SCORES_STATES.glob("*.json"):
        state = json.loads(path.read_text(encoding="utf-8"))
        for key in ["prompt_sha256", "read_prompt_sha256"]:
            if key in state:
                state_hashes[state[key]] += 1
    integrity = {
        "score_records": len(scores), "score_unique_ids": scores.sample_id.nunique(),
        "trajectory_records_all": len(traj_all), "trajectory_unique_ids": traj_all.sample_id.nunique(),
        "common_budget_grid": BUDGETS, "common_grid_records": len(df),
        "common_grid_unique_ids": df.sample_id.nunique(), "common_grid_complete_questions": int((counts == len(BUDGETS)).sum()),
        "id_overlap": len(set(scores.sample_id) & set(df.sample_id)),
        "score_distribution": {str(k): int(v) for k, v in scores.read_score.value_counts().sort_index().items()},
        "read_rating_recovery": {
            "final_accepted": audit["data_integrity"]["accepted_read_scores"],
            "physical_attempts": audit["transport"]["physical_attempts"],
            "failed_transport_attempts": audit["transport"]["failed_attempts"],
            "strict_one_shot": audit["transport"]["strict_one_shot"],
            "runtime_prompt_hash_from_run_config": (config.get("prompt_sha256") or {}).get("read"),
            "state_prompt_hashes_if_present": dict(state_hashes),
            "note": "The final 500 accepted ratings are complete, but recovery after TLS failures used 1132 physical attempts. The audit's prompt-file hash warning compares raw file bytes with runtime stripped text; run_config and accepted states use one frozen runtime hash.",
        },
    }
    required = [len(scores) == 500, scores.sample_id.nunique() == 500, len(df) == 3000, df.sample_id.nunique() == 500, (counts == 6).all(), integrity["id_overlap"] == 500]
    integrity["passed"] = bool(all(required))
    (OUT / "integrity.json").write_text(json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8")
    if not integrity["passed"]:
        raise RuntimeError(f"Integrity check failed: {integrity}")

    # Curves by exact score.
    merged = df.merge(scores[["sample_id", "read_score", "read_confidence"]], on="sample_id", validate="many_to_one")
    curves = merged.groupby(["read_score", "budget"]).agg(
        n=("sample_id", "nunique"), answer_em=("answer_em", "mean"), answer_f1=("answer_f1", "mean"), answer_acc=("answer_acc", "mean"),
        input_tokens=("standalone_llm_input_tokens", "mean"), output_tokens=("standalone_llm_output_tokens", "mean"),
        total_llm_tokens=("total_llm_tokens", "mean"), visible_evidence_tokens=("visible_evidence_tokens_total", "mean"), search_calls=("search_calls", "mean"),
    ).reset_index()
    curves.to_csv(OUT / "read_score_budget_curves.csv", index=False, encoding="utf-8-sig")

    oracle = make_oracles(df, scores)
    oracle.to_csv(OUT / "per_question_oracles.csv", index=False, encoding="utf-8-sig")
    process, correlations, groups = association_tables(df, scores, oracle)
    process.to_csv(OUT / "per_question_process_means.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(OUT / "associations.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(OUT / "read_score_group_summary.csv", index=False, encoding="utf-8-sig")

    # Kruskal-Wallis tests are descriptive because score=4 has n=2.
    tests = []
    for metric in METRICS:
        arrays = [g[f"oracle_budget_{metric}"].to_numpy(float) for _, g in oracle.groupby("read_score")]
        stat = kruskal(*arrays)
        tests.append({"outcome": f"oracle_budget_{metric}", "test": "Kruskal-Wallis", "statistic": float(stat.statistic), "p": float(stat.pvalue), "warning": "score=4 has n=2"})
    pd.DataFrame(tests).to_csv(OUT / "group_tests.csv", index=False, encoding="utf-8-sig")

    predictions = pd.concat([loo_predictions(df, scores, metric) for metric in METRICS], ignore_index=True)
    predictions.to_csv(OUT / "loo_policy_predictions.csv", index=False, encoding="utf-8-sig")
    policy = policy_summary(predictions)
    policy.to_csv(OUT / "policy_summary.csv", index=False, encoding="utf-8-sig")
    deltas = paired_policy_deltas(predictions)
    deltas.to_csv(OUT / "paired_policy_deltas.csv", index=False, encoding="utf-8-sig")
    stability = bootstrap_budget_selection(df, scores)
    stability.to_csv(OUT / "bootstrap_budget_selection_stability.csv", index=False, encoding="utf-8-sig")
    em_rule_delta, em_rule_cases = compare_em_rule_to_fixed600(predictions, df, scores)
    em_rule_delta.to_csv(OUT / "em_rule_vs_fixed600.csv", index=False, encoding="utf-8-sig")
    em_rule_cases.to_csv(OUT / "em_rule_flip_cases.csv", index=False, encoding="utf-8-sig")

    plot_curves(curves)
    plot_policy(policy, deltas)
    plot_oracle_signal(oracle, process)

    # Compact machine-readable summary.
    f1_policy = policy[policy.optimized_metric == "answer_f1"].set_index("policy")
    f1_delta = deltas[(deltas.optimized_metric == "answer_f1") & (deltas.outcome == "answer_f1")].set_index("policy")
    summary = {
        "integrity": integrity,
        "primary_scope": "500 questions x budgets 100..600; F1 primary; total cost=input+output tokens",
        "correlations": correlations.to_dict("records"),
        "read_score_groups": groups.to_dict("records"),
        "primary_policy_summary": f1_policy.reset_index().to_dict("records"),
        "primary_f1_deltas_vs_loo_global": f1_delta.reset_index().to_dict("records"),
        "em_rule_vs_fixed600": em_rule_delta.to_dict("records"),
        "bootstrap_budget_selection_stability": stability.to_dict("records"),
        "warnings": [
            "READ=4 has only 2 questions; do not interpret it as a stable category.",
            "Quality-then-cost oracle is an upper bound using realized outcomes, not a deployable query-only controller.",
            "The query-only policy is evaluated with leave-one-out training; in-sample score-to-budget lookup is intentionally not reported as causal improvement.",
            "Budgets above 600 are non-rectangular and excluded from primary paired inference; FULL is a separate endpoint, not a fixed numeric cap.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "integrity_passed": integrity["passed"],
        "records": len(df),
        "scores": len(scores),
        "f1_policy_summary": f1_policy[["mean_selected_budget", "answer_f1", "answer_em", "answer_acc", "total_llm_tokens"]].to_dict("index"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
