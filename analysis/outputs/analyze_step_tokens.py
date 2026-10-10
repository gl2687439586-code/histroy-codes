from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"C:\Users\RESEARCH_USER\Documents\Codex\2026-08-11\referenced-chatgpt-conversation-this-is-an-3")
INPUT = ROOT / "outputs/qwen3_only_bamboogle_v6_review/02_src_tree/analysis_primary_natural_roots/sample_sequence_values.csv"
OUT = ROOT / "outputs/step_token_analysis"
OUT.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(INPUT)
d["step"] = d["path"].str.len()
d["answer_correct_em"] = np.where(d["em_mean"].eq(1), "Correct", "Incorrect")
d["answer_correct_acc"] = np.where(d["acc_mean"].eq(1), "Correct", "Incorrect")
d["f1_quality_band"] = np.select(
    [d["f1_mean"].eq(0), d["f1_mean"].eq(1)],
    ["Zero (F1=0)", "Exact-quality (F1=1)"],
    default="Partial (0<F1<1)",
)

token_cols = {
    "total_cost_tokens_mean": "total_cost_tokens",
    "llm_tokens_total_mean": "llm_tokens_total",
    "total_llm_input_tokens_mean": "llm_input_tokens",
    "total_llm_output_tokens_mean": "llm_output_tokens",
    "retrieved_tokens_total_mean": "retrieved_tokens",
    "new_evidence_tokens_mean": "new_evidence_tokens",
    "read_input_tokens_mean": "read_input_tokens",
    "read_output_tokens_mean": "read_output_tokens",
    "reasoning_tokens_mean": "reasoning_tokens",
    "answer_tokens_mean": "answer_tokens",
}


def cluster_bootstrap_mean_ci(frame, value_col, seed):
    if len(frame) == 0:
        return np.nan, np.nan
    clusters = frame["sample_id"].unique()
    if len(clusters) == 1:
        value = frame[value_col].astype(float).mean()
        return float(value), float(value)
    rng = np.random.default_rng(seed)
    grouped = frame.groupby("sample_id")[value_col].agg(["sum", "count"]).reindex(clusters)
    cluster_sums = grouped["sum"].to_numpy(float)
    cluster_counts = grouped["count"].to_numpy(float)
    ix = rng.integers(0, len(clusters), size=(10000, len(clusters)))
    means = cluster_sums[ix].sum(axis=1) / cluster_counts[ix].sum(axis=1)
    return tuple(np.quantile(means, [0.025, 0.975]))


def summarize(group_cols, label):
    rows = []
    for keys, g in d.groupby(group_cols, sort=True, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            analysis=label,
            trajectories=len(g),
            unique_questions=g["sample_id"].nunique(),
            mean_f1=g["f1_mean"].mean(),
            em_rate=g["em_mean"].mean(),
            acc_rate=g["acc_mean"].mean(),
            mean_search_calls=g["search_calls_mean"].mean(),
            mean_read_calls=g["read_calls_mean"].mean(),
        )
        seed = 2026 + int(row["step"]) * 100 + len(rows)
        for src, dst in token_cols.items():
            vals = g[src].astype(float).to_numpy()
            lo, hi = cluster_bootstrap_mean_ci(g, src, seed)
            row[f"mean_{dst}"] = vals.mean()
            row[f"median_{dst}"] = np.median(vals)
            row[f"ci95_low_{dst}"] = lo
            row[f"ci95_high_{dst}"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


overall = summarize(["step"], "overall_by_step")
by_em = summarize(["step", "answer_correct_em"], "by_step_and_em")
by_acc = summarize(["step", "answer_correct_acc"], "by_step_and_acc")
by_f1_band = summarize(["step", "f1_quality_band"], "by_step_and_f1_band")
combined = pd.concat([overall, by_em, by_acc, by_f1_band], ignore_index=True, sort=False)

# Descriptive correct-vs-incorrect total-token gaps at each step. These are not causal step effects.
gap_rows = []
for metric, group_col in [("EM", "answer_correct_em"), ("ACC", "answer_correct_acc")]:
    for step, g in d.groupby("step"):
        correct = g.loc[g[group_col].eq("Correct"), "total_cost_tokens_mean"].to_numpy(float)
        incorrect = g.loc[g[group_col].eq("Incorrect"), "total_cost_tokens_mean"].to_numpy(float)
        if not len(correct) or not len(incorrect):
            continue
        rng = np.random.default_rng(2026 + step + (0 if metric == "EM" else 1000))
        eligible = g.groupby("sample_id")[group_col].nunique()
        paired_ids = eligible[eligible.eq(2)].index.to_numpy()
        paired_diffs = np.array([
            g.loc[g["sample_id"].eq(sid) & g[group_col].eq("Correct"), "total_cost_tokens_mean"].mean()
            - g.loc[g["sample_id"].eq(sid) & g[group_col].eq("Incorrect"), "total_cost_tokens_mean"].mean()
            for sid in paired_ids
        ], dtype=float)
        if len(paired_diffs):
            ix = rng.integers(0, len(paired_diffs), size=(10000, len(paired_diffs)))
            diffs = paired_diffs[ix].mean(axis=1)
            paired_mean = paired_diffs.mean()
            lo, hi = np.quantile(diffs, [0.025, 0.975])
        else:
            paired_mean = np.nan
            lo, hi = np.nan, np.nan
        gap_rows.append(
            {
                "correctness_metric": metric,
                "step": step,
                "correct_n": len(correct),
                "incorrect_n": len(incorrect),
                "correct_mean_total_cost_tokens": correct.mean(),
                "incorrect_mean_total_cost_tokens": incorrect.mean(),
                "unpaired_difference_correct_minus_incorrect": correct.mean() - incorrect.mean(),
                "questions_with_both_outcomes": len(paired_diffs),
                "paired_question_mean_difference": paired_mean,
                "paired_question_ci95_low": lo,
                "paired_question_ci95_high": hi,
            }
        )
gaps = pd.DataFrame(gap_rows)

# Step 3 is the only well-powered counterfactual layer: every natural-root question
# contributes alternative terminal paths at the same decision depth. Estimate a
# question-fixed-effect association between quality and tokens by demeaning within question.
association_rows = []
step3 = d[d["step"].eq(3)].copy()
for src, dst in token_cols.items():
    x = step3["f1_mean"] - step3.groupby("sample_id")["f1_mean"].transform("mean")
    y = step3[src] - step3.groupby("sample_id")[src].transform("mean")
    denominator = float(np.dot(x, x))
    slope = float(np.dot(x, y) / denominator) if denominator else np.nan
    clusters = step3["sample_id"].unique()
    rng = np.random.default_rng(3026 + len(association_rows))
    cluster_num, cluster_den = [], []
    for sid in clusters:
        part = step3[step3["sample_id"].eq(sid)]
        bx = (part["f1_mean"] - part["f1_mean"].mean()).to_numpy(float)
        by = (part[src] - part[src].mean()).to_numpy(float)
        cluster_num.append(float(np.dot(bx, by)))
        cluster_den.append(float(np.dot(bx, bx)))
    cluster_num = np.asarray(cluster_num)
    cluster_den = np.asarray(cluster_den)
    ix = rng.integers(0, len(clusters), size=(10000, len(clusters)))
    bootstrap = cluster_num[ix].sum(axis=1) / cluster_den[ix].sum(axis=1)
    association_rows.append(
        {
            "step": 3,
            "token_metric": dst,
            "questions": len(clusters),
            "trajectories": len(step3),
            "tokens_per_plus_one_f1_with_question_fixed_effects": slope,
            "ci95_low": np.nanquantile(bootstrap, 0.025),
            "ci95_high": np.nanquantile(bootstrap, 0.975),
        }
    )
associations = pd.DataFrame(association_rows)

raw_cols = [
    "sample_id", "path", "horizon", "step", "em_mean", "acc_mean", "f1_mean",
    "answer_correct_em", "answer_correct_acc", "f1_quality_band",
] + list(token_cols)
d[raw_cols].to_csv(OUT / "trajectory_step_tokens.csv", index=False)
combined.to_csv(OUT / "step_token_summary.csv", index=False)
gaps.to_csv(OUT / "correctness_token_gaps.csv", index=False)
associations.to_csv(OUT / "step3_quality_token_association.csv", index=False)

def markdown_table(frame):
    frame = frame.copy()
    headers = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


with (OUT / "report.md").open("w", encoding="utf-8") as f:
    f.write("# Token cost by terminal step and answer quality\n\n")
    f.write("Primary subset: 107 natural-root questions, 917 terminal trajectories. Step is path length and includes the fixed initial Search. Values are cumulative through termination, not marginal per-action costs.\n\n")
    f.write("## Overall by step\n\n")
    f.write(markdown_table(overall[["step", "trajectories", "unique_questions", "mean_f1", "em_rate", "mean_total_cost_tokens", "mean_llm_tokens_total", "mean_retrieved_tokens"]].round(3)))
    f.write("\n\n## By step and strict EM correctness\n\n")
    f.write(markdown_table(by_em[["step", "answer_correct_em", "trajectories", "unique_questions", "mean_f1", "mean_total_cost_tokens", "median_total_cost_tokens", "ci95_low_total_cost_tokens", "ci95_high_total_cost_tokens", "mean_llm_input_tokens", "mean_llm_output_tokens", "mean_retrieved_tokens", "mean_read_input_tokens", "mean_read_output_tokens"]].round(3)))
    f.write("\n\n## Correct-minus-incorrect total-token gaps\n\n")
    f.write(markdown_table(gaps.round(3)))
    f.write("\n\n## By step and F1 quality band\n\n")
    f.write(markdown_table(by_f1_band[["step", "f1_quality_band", "trajectories", "unique_questions", "mean_f1", "mean_total_cost_tokens", "ci95_low_total_cost_tokens", "ci95_high_total_cost_tokens", "mean_llm_input_tokens", "mean_llm_output_tokens", "mean_retrieved_tokens"]].round(3)))
    f.write("\n\n## Step 3 within-question quality-token association\n\n")
    f.write(markdown_table(associations.round(3)))
    f.write("\n")

print(overall[["step", "trajectories", "unique_questions", "mean_f1", "em_rate", "mean_total_cost_tokens", "mean_llm_tokens_total", "mean_retrieved_tokens"]].round(3).to_string(index=False))
print("\nBY EM\n", by_em[["step", "answer_correct_em", "trajectories", "mean_total_cost_tokens", "ci95_low_total_cost_tokens", "ci95_high_total_cost_tokens", "mean_f1"]].round(3).to_string(index=False))
print("\nGAPS\n", gaps.round(3).to_string(index=False))
print("\nBY F1 BAND\n", by_f1_band[["step", "f1_quality_band", "trajectories", "mean_total_cost_tokens", "mean_f1"]].round(3).to_string(index=False))
print("\nSTEP 3 WITHIN-QUESTION ASSOCIATION\n", associations.round(3).to_string(index=False))
