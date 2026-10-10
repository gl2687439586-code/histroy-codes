from __future__ import annotations

import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
WORKSPACE = BASE.parent
INPUT = BASE / "inputs"
OUT = BASE / "results"
COMMON = [100, 200, 300, 400, 500, 600]
SEED = 2026

HOTPOT_SCORES = WORKSPACE / "analysis_readscore_oracle_mapping_20260915/read/hotpot_random500_openrouter_gpt56sol_read_v2_fresh/results/read_scores_combined.jsonl"
HOTPOT_TRAJ = WORKSPACE / "analysis_readscore_oracle_mapping_20260915/traj/analysis/scored_records.jsonl.gz"


def read_jsonl(path: Path, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def coarse(score):
    score = int(score)
    if score <= 1:
        return "low_1"
    if score == 2:
        return "mid_2"
    return "high_3plus"


def load_data():
    hs = pd.DataFrame(read_jsonl(HOTPOT_SCORES)).rename(columns={"question_id": "sample_id"})
    h_all = pd.DataFrame(read_jsonl(HOTPOT_TRAJ, True))
    h_all["budget_numeric"] = pd.to_numeric(h_all.condition_group, errors="coerce")
    h = h_all[h_all.budget_numeric.isin(COMMON)].copy()
    h["budget"] = h.budget_numeric.astype(int)
    h["total_llm_tokens"] = h.standalone_llm_input_tokens + h.standalone_llm_output_tokens
    h_full = h_all[h_all.condition_group.astype(str) == "FULL"].copy()
    h_full["total_llm_tokens"] = h_full.standalone_llm_input_tokens + h_full.standalone_llm_output_tokens

    fs = pd.DataFrame(read_jsonl(INPUT / "fanout_query_scores.jsonl")).rename(columns={"question_id": "sample_id"})
    f_all = pd.DataFrame(read_jsonl(INPUT / "fanout_budget_scored.jsonl.gz", True))
    f_all["budget_numeric"] = pd.to_numeric(f_all.condition_label, errors="coerce")
    f = f_all[f_all.budget_numeric.isin(COMMON)].copy()
    f["budget"] = f.budget_numeric.astype(int)
    f["total_llm_tokens"] = f.standalone_llm_input_tokens + f.standalone_llm_output_tokens
    f_blank = pd.DataFrame(read_jsonl(INPUT / "fanout_blank_scored.jsonl.gz", True))
    f_blank["total_llm_tokens"] = f_blank.standalone_llm_input_tokens + f_blank.standalone_llm_output_tokens
    return hs, h, h_full, fs, f, f_blank, f_all


def validate(scores, data, expected):
    counts = data.groupby("sample_id").budget.nunique()
    return {
        "score_records": len(scores),
        "score_unique_ids": int(scores.sample_id.nunique()),
        "trajectory_records_common": len(data),
        "trajectory_unique_ids": int(data.sample_id.nunique()),
        "id_overlap": len(set(scores.sample_id) & set(data.sample_id)),
        "complete_common_grid": int((counts == len(COMMON)).sum()),
        "read_score_distribution": {str(k): int(v) for k, v in scores.read_score.value_counts().sort_index().items()},
        "passed": bool(len(scores) == expected and scores.sample_id.nunique() == expected and len(data) == expected * len(COMMON) and (counts == len(COMMON)).all() and set(scores.sample_id) == set(data.sample_id)),
    }


def learn_mapping(data, scores, metric, grouping="coarse"):
    meta = scores[["sample_id", "read_score"]].copy()
    meta["group"] = meta.read_score.map(coarse) if grouping == "coarse" else meta.read_score.astype(int)
    merged = data.merge(meta[["sample_id", "group"]], on="sample_id", validate="many_to_one")
    global_table = merged.groupby("budget").agg(quality=(metric, "mean"), cost=("total_llm_tokens", "mean")).reset_index()
    global_budget = int(global_table.sort_values(["quality", "cost", "budget"], ascending=[False, True, True]).iloc[0].budget)
    mapping = {}
    table_rows = []
    for group, rows in merged.groupby("group"):
        table = rows.groupby("budget").agg(n=("sample_id", "nunique"), quality=(metric, "mean"), cost=("total_llm_tokens", "mean")).reset_index()
        selected = table.sort_values(["quality", "cost", "budget"], ascending=[False, True, True]).iloc[0]
        mapping[str(group)] = int(selected.budget)
        for _, row in table.iterrows():
            table_rows.append({"group": str(group), "budget": int(row.budget), "n": int(row.n), "quality": float(row.quality), "cost": float(row.cost), "selected": int(row.budget) == int(selected.budget)})
    return mapping, global_budget, pd.DataFrame(table_rows)


def apply_mapping(target, scores, mapping, fallback_budget, grouping, policy, direction, source_metric):
    meta = scores[["sample_id", "read_score"]].copy()
    meta["group"] = meta.read_score.map(coarse) if grouping == "coarse" else meta.read_score.astype(int)
    meta["selected_budget"] = meta.group.map(lambda x: mapping.get(str(x), fallback_budget)).astype(int)
    chosen = target.merge(meta, on="sample_id", validate="many_to_one")
    chosen = chosen[chosen.budget == chosen.selected_budget].copy()
    if len(chosen) != scores.sample_id.nunique() or chosen.sample_id.duplicated().any():
        raise RuntimeError(f"Mapping failed for {policy}")
    chosen["policy"] = policy
    chosen["direction"] = direction
    chosen["source_metric"] = source_metric
    chosen["grouping"] = grouping
    return chosen


def fixed_policy(data, budget, name, dataset):
    rows = data[data.budget == budget].copy()
    rows["selected_budget"] = budget
    rows["policy"] = name
    rows["direction"] = dataset
    rows["source_metric"] = "target comparator"
    rows["grouping"] = "none"
    return rows


def oracle_policy(data, metric, name, dataset):
    rows = []
    for _, group in data.groupby("sample_id"):
        rows.append(group.sort_values([metric, "total_llm_tokens", "budget"], ascending=[False, True, True]).iloc[0])
    out = pd.DataFrame(rows)
    out["selected_budget"] = out.budget.astype(int)
    out["policy"] = name
    out["direction"] = dataset
    out["source_metric"] = "target oracle"
    out["grouping"] = "per_question"
    return out


def loo_policy(data, scores, metric, grouping, name, dataset):
    rows = []
    meta = scores.set_index("sample_id")
    ids = list(meta.index)
    for held in ids:
        train_ids = [sid for sid in ids if sid != held]
        train_scores = scores[scores.sample_id.isin(train_ids)]
        mapping, fallback, _ = learn_mapping(data[data.sample_id.isin(train_ids)], train_scores, metric, grouping)
        score = int(meta.loc[held, "read_score"])
        group = coarse(score) if grouping == "coarse" else score
        budget = mapping.get(str(group), fallback)
        row = data[(data.sample_id == held) & (data.budget == budget)].iloc[0].copy()
        row["selected_budget"] = budget
        row["policy"] = name
        row["direction"] = dataset
        row["source_metric"] = metric
        row["grouping"] = grouping
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(rows, dataset):
    metric_cols = [
        "official_loose_accuracy", "official_strict_accuracy", "rougeL_f1",
        "answer_f1", "answer_em", "answer_acc",
    ]
    aggs = {
        "n": ("sample_id", "nunique"),
        "mean_selected_budget": ("selected_budget", "mean"),
        "total_llm_tokens": ("total_llm_tokens", "mean"),
        "visible_evidence_tokens": ("visible_evidence_tokens_total", "mean"),
        "search_calls": ("search_calls", "mean"),
    }
    for col in metric_cols:
        if col in rows.columns:
            aggs[col] = (col, "mean")
    result = rows.groupby(["policy", "direction", "source_metric", "grouping"], dropna=False).agg(**aggs).reset_index()
    result["target_dataset"] = dataset
    return result


def paired_delta(candidate, reference, metric, seed, n_boot=20000):
    c = candidate.set_index("sample_id")
    r = reference.set_index("sample_id").loc[c.index]
    delta = c[metric].to_numpy(float) - r[metric].to_numpy(float)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(20):
        idx = rng.integers(0, len(delta), size=(n_boot // 20, len(delta)))
        means.append(delta[idx].mean(axis=1))
    means = np.concatenate(means)
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(np.quantile(means, .025)),
        "ci95_high": float(np.quantile(means, .975)),
        "probability_gt_zero": float(np.mean(means > 0)),
        "wins": int(np.sum(delta > 0)), "ties": int(np.sum(delta == 0)), "losses": int(np.sum(delta < 0)),
    }


def bootstrap_mapping_stability(data, scores, metric, dataset, grouping="coarse", n_boot=10000):
    rng = np.random.default_rng(SEED + (0 if dataset == "HotpotQA" else 1000))
    meta = scores[["sample_id", "read_score"]].copy()
    meta["group"] = meta.read_score.map(coarse) if grouping == "coarse" else meta.read_score.astype(int)
    q = data.pivot(index="sample_id", columns="budget", values=metric)[COMMON]
    c = data.pivot(index="sample_id", columns="budget", values="total_llm_tokens")[COMMON]
    meta = meta.set_index("sample_id").loc[q.index]
    result = []
    for group, group_meta in meta.groupby("group"):
        ids = list(group_meta.index)
        qa = q.loc[ids].to_numpy(float)
        ca = c.loc[ids].to_numpy(float)
        counts = Counter()
        for _ in range(n_boot):
            idx = rng.integers(0, len(ids), len(ids))
            qm, cm = qa[idx].mean(axis=0), ca[idx].mean(axis=0)
            chosen = sorted(range(len(COMMON)), key=lambda j: (-qm[j], cm[j], COMMON[j]))[0]
            counts[COMMON[chosen]] += 1
        for budget in COMMON:
            result.append({"dataset": dataset, "metric": metric, "grouping": grouping, "group": str(group), "group_n": len(ids), "budget": budget, "selection_probability": counts[budget] / n_boot})
    return pd.DataFrame(result)


def plot_transfer(summary, comparisons):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    configs = [
        ("FanOutQA", "official_loose_accuracy", "FanOutQA official Loose"),
        ("HotpotQA", "answer_f1", "HotpotQA answer F1"),
    ]
    for ax, (dataset, metric, ylabel) in zip(axes, configs):
        sub = summary[summary.target_dataset == dataset].dropna(subset=[metric]).copy()
        transfer_names = [x for x in sub.policy if x.startswith("transfer_")]
        keep = [p for p in sub.policy if p in transfer_names or p in ("BLANK", "FULL", "fixed_500", "fixed_600", "common_oracle")]
        sub = sub[sub.policy.isin(keep)].drop_duplicates("policy")
        colors = ["#D1495B" if p.startswith("transfer_") else ("#59A14F" if p == "common_oracle" else "#4C78A8") for p in sub.policy]
        ax.scatter(sub.total_llm_tokens, sub[metric], s=95, c=colors)
        offsets = defaultdict(lambda: (5, 5), {"fixed_500": (5, -13), "fixed_600": (5, 7), "FULL": (5, 7), "BLANK": (5, 7)})
        for _, row in sub.iterrows():
            ax.annotate(row.policy.replace("transfer_", ""), (row.total_llm_tokens, row[metric]), xytext=offsets[row.policy], textcoords="offset points", fontsize=8)
        ax.set(title=f"Target: {dataset}", xlabel="Mean total LLM tokens", ylabel=ylabel)
    fig.suptitle("Cross-dataset transfer of READ-score budget mappings", fontsize=17, weight="bold")
    fig.savefig(OUT / "cross_dataset_transfer.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "cross_dataset_transfer.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hs, h, h_full, fs, f, f_blank, f_all = load_data()
    audit = {"hotpot": validate(hs, h, 500), "fanout": validate(fs, f, 100)}
    audit["fanout_blank_records"] = len(f_blank)
    audit["hotpot_full_records"] = len(h_full)
    audit["passed"] = audit["hotpot"]["passed"] and audit["fanout"]["passed"] and len(f_blank) == 100 and len(h_full) == 500
    (OUT / "integrity.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if not audit["passed"]:
        raise RuntimeError(audit)

    mapping_specs = [
        ("HotpotQA", h, hs, "answer_f1", "coarse", "hotpot_f1_coarse"),
        ("HotpotQA", h, hs, "answer_f1", "exact", "hotpot_f1_exact"),
        ("HotpotQA", h, hs, "answer_em", "coarse", "hotpot_em_coarse"),
        ("FanOutQA", f, fs, "official_loose_accuracy", "coarse", "fanout_loose_coarse"),
        ("FanOutQA", f, fs, "official_loose_accuracy", "exact", "fanout_loose_exact"),
        ("FanOutQA", f, fs, "rougeL_f1", "coarse", "fanout_rougel_coarse"),
    ]
    mappings = {}
    mapping_tables = []
    learned = {}
    for dataset, data, scores, metric, grouping, name in mapping_specs:
        mapping, fallback, table = learn_mapping(data, scores, metric, grouping)
        learned[name] = (mapping, fallback, grouping, metric)
        mappings[name] = {"source_dataset": dataset, "source_metric": metric, "grouping": grouping, "mapping": mapping, "unseen_group_fallback": fallback}
        table["mapping_name"] = name
        table["source_dataset"] = dataset
        table["source_metric"] = metric
        table["grouping"] = grouping
        mapping_tables.append(table)
    (OUT / "learned_mappings.json").write_text(json.dumps(mappings, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.concat(mapping_tables, ignore_index=True).to_csv(OUT / "source_group_curves.csv", index=False, encoding="utf-8-sig")

    fanout_policies = []
    hotpot_policies = []
    # Source-frozen transfer policies.
    for name in ["hotpot_f1_coarse", "hotpot_f1_exact", "hotpot_em_coarse"]:
        mapping, fallback, grouping, metric = learned[name]
        fanout_policies.append(apply_mapping(f, fs, mapping, fallback, grouping, f"transfer_{name}", "HotpotQA->FanOutQA", metric))
    for name in ["fanout_loose_coarse", "fanout_loose_exact", "fanout_rougel_coarse"]:
        mapping, fallback, grouping, metric = learned[name]
        hotpot_policies.append(apply_mapping(h, hs, mapping, fallback, grouping, f"transfer_{name}", "FanOutQA->HotpotQA", metric))

    # Target comparators.
    fanout_fixed = [fixed_policy(f, b, f"fixed_{b}", "FanOutQA") for b in COMMON]
    hotpot_fixed = [fixed_policy(h, b, f"fixed_{b}", "HotpotQA") for b in COMMON]
    fanout_oracle = oracle_policy(f, "official_loose_accuracy", "common_oracle", "FanOutQA")
    hotpot_oracle = oracle_policy(h, "answer_f1", "common_oracle", "HotpotQA")
    fanout_loo = loo_policy(f, fs, "official_loose_accuracy", "coarse", "in_domain_loo_coarse", "FanOutQA")
    hotpot_loo = loo_policy(h, hs, "answer_f1", "coarse", "in_domain_loo_coarse", "HotpotQA")
    f_blank = f_blank.copy()
    f_blank["selected_budget"] = np.nan
    f_blank["policy"] = "BLANK"
    f_blank["direction"] = "FanOutQA"
    f_blank["source_metric"] = "target comparator"
    f_blank["grouping"] = "none"
    h_full = h_full.copy()
    h_full["selected_budget"] = np.nan
    h_full["policy"] = "FULL"
    h_full["direction"] = "HotpotQA"
    h_full["source_metric"] = "target comparator"
    h_full["grouping"] = "none"

    fanout_all = pd.concat(fanout_policies + fanout_fixed + [fanout_oracle, fanout_loo, f_blank], ignore_index=True, sort=False)
    hotpot_all = pd.concat(hotpot_policies + hotpot_fixed + [hotpot_oracle, hotpot_loo, h_full], ignore_index=True, sort=False)
    fanout_all.to_csv(OUT / "fanout_per_question_policies.csv", index=False, encoding="utf-8-sig")
    hotpot_all.to_csv(OUT / "hotpot_per_question_policies.csv", index=False, encoding="utf-8-sig")
    summary = pd.concat([summarize(fanout_all, "FanOutQA"), summarize(hotpot_all, "HotpotQA")], ignore_index=True, sort=False)
    summary.to_csv(OUT / "policy_summary.csv", index=False, encoding="utf-8-sig")

    comparisons = []
    fanout_refs = {p: fanout_all[fanout_all.policy == p] for p in ["BLANK", "fixed_500", "common_oracle"]}
    hotpot_refs = {p: hotpot_all[hotpot_all.policy == p] for p in ["fixed_600", "FULL", "common_oracle"]}
    counter = 0
    for cand in fanout_policies:
        for ref_name, ref in fanout_refs.items():
            for metric in ["official_loose_accuracy", "official_strict_accuracy", "rougeL_f1", "total_llm_tokens"]:
                comparisons.append({"target_dataset": "FanOutQA", "candidate": cand.policy.iloc[0], "reference": ref_name, "metric": metric, **paired_delta(cand, ref, metric, SEED + counter)})
                counter += 1
    for cand in hotpot_policies:
        for ref_name, ref in hotpot_refs.items():
            for metric in ["answer_f1", "answer_em", "answer_acc", "total_llm_tokens"]:
                comparisons.append({"target_dataset": "HotpotQA", "candidate": cand.policy.iloc[0], "reference": ref_name, "metric": metric, **paired_delta(cand, ref, metric, SEED + counter)})
                counter += 1
    comp = pd.DataFrame(comparisons)
    comp.to_csv(OUT / "paired_transfer_comparisons.csv", index=False, encoding="utf-8-sig")

    stability = pd.concat([
        bootstrap_mapping_stability(h, hs, "answer_f1", "HotpotQA", "coarse"),
        bootstrap_mapping_stability(h, hs, "answer_em", "HotpotQA", "coarse"),
        bootstrap_mapping_stability(f, fs, "official_loose_accuracy", "FanOutQA", "coarse"),
        bootstrap_mapping_stability(f, fs, "rougeL_f1", "FanOutQA", "coarse"),
    ], ignore_index=True)
    stability.to_csv(OUT / "mapping_stability.csv", index=False, encoding="utf-8-sig")

    # Native FanOutQA oracle (includes 700+ and FULL) for the requested reference.
    native = pd.read_csv(INPUT / "fanout_native_oracle.csv")
    native_summary = {
        "fanout_blank_loose": float(native.blank_loose.mean()),
        "fanout_native_budget_oracle_loose": float(native.oracle_loose.mean()),
        "fanout_oracle_over_blank_and_budget": float(native[["blank_loose", "oracle_loose"]].max(axis=1).mean()),
        "fanout_native_oracle_visible_evidence_tokens": float(native.oracle_visible_evidence_tokens.mean()),
        "fanout_blank_visible_evidence_tokens": float(native.blank_visible_evidence_tokens.mean()),
    }

    # Oracle-gap recovery, using primary target metrics.
    sidx = summary.set_index(["target_dataset", "policy"])
    gap_rows = []
    for dataset, metric, base_name in [("FanOutQA", "official_loose_accuracy", "BLANK"), ("HotpotQA", "answer_f1", "fixed_600")]:
        base_value = float(sidx.loc[(dataset, base_name), metric])
        oracle_value = float(sidx.loc[(dataset, "common_oracle"), metric])
        candidates = [p for p in summary[summary.target_dataset == dataset].policy if p.startswith("transfer_")]
        for policy in candidates:
            value = float(sidx.loc[(dataset, policy), metric])
            gap_rows.append({"target_dataset": dataset, "policy": policy, "primary_metric": metric, "baseline": base_name, "baseline_quality": base_value, "policy_quality": value, "common_oracle_quality": oracle_value, "absolute_gain_vs_baseline": value - base_value, "fraction_of_oracle_gap_recovered": (value - base_value) / (oracle_value - base_value) if oracle_value != base_value else None})
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(OUT / "oracle_gap_recovery.csv", index=False, encoding="utf-8-sig")

    result = {"integrity": audit, "mappings": mappings, "native_references": native_summary, "oracle_gap_recovery": gaps.to_dict("records")}
    (OUT / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_transfer(summary, comp)
    print(json.dumps({"integrity": audit, "mappings": mappings, "primary_transfer": gaps.to_dict("records")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
