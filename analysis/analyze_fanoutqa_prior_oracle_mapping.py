import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 2026
COMMON_CAPS = [100, 200, 300, 400, 500, 600, 700]


def read_jsonl(path: Path, compressed: bool = False):
    opener = gzip.open if compressed else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def tokenize(text):
    return re.findall(r"[a-z0-9']+", str(text).lower())


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "twenty": 20,
}


def query_features(text):
    lower = text.lower()
    words = tokenize(text)
    numbers = [int(value) for value in re.findall(r"\b\d+\b", lower)]
    numbers.extend(
        value
        for word, value in NUMBER_WORDS.items()
        if re.search(rf"\b{word}\b", lower)
    )
    return {
        "query_word_count": len(words),
        "query_char_count": len(text),
        "explicit_cardinality": max(numbers) if numbers else 0,
        "number_mention_count": len(numbers),
        "comma_count": text.count(","),
        "conjunction_count": len(re.findall(r"\b(?:and|or|as well as)\b", lower)),
        "quoted_span_count": len(re.findall(r'[\"“][^\"”]+[\"”]', text)),
        "has_ranking": int(bool(re.search(
            r"\b(?:largest|smallest|highest|lowest|most|least|top|bottom|"
            r"rank|best|worst|oldest|youngest|longest|shortest)\b",
            lower,
        ))),
        "has_aggregation": int(bool(re.search(
            r"\b(?:all|each|every|list|nominees|members|authors|actors|"
            r"countries|states|works|songs|films|books|companies|universities|awards)\b",
            lower,
        ))),
        "has_comparison": int(bool(re.search(
            r"\b(?:compare|comparison|difference|same|both|versus|vs\.?|"
            r"respectively|more than|less than)\b",
            lower,
        ))),
        "has_temporal_constraint": int(bool(re.search(
            r"\b(?:before|after|between|during|since|until|prior|year|date|"
            r"century|currently|current|latest)\b|\b(?:19|20)\d{2}\b",
            lower,
        ))),
        "has_relational_constraint": int(bool(re.search(
            r"\b(?:whose|which|that|located|based|from|associated|founded|"
            r"written|directed|born|belong)\b|\bof the\b|\bwork at\b",
            lower,
        ))),
        "has_multi_output_form": int(bool(re.search(
            r"\b(?:what are|what were|who were|who are|which are|which were|"
            r"how many|and what|and where|and who)\b",
            lower,
        ))),
    }


def lexical_overlap(left, right):
    left_tokens = set(tokenize(left))
    right_tokens = set(tokenize(right))
    return len(left_tokens & right_tokens) / max(1, len(left_tokens))


def root_features(row):
    documents = row.get("documents") or []
    titles = []
    content_lengths = []
    title_overlaps = []
    for doc in documents:
        title = str(doc.get("title") or "")
        content = str(doc.get("contents") or doc.get("content") or "")
        titles.append(title.strip().lower())
        content_lengths.append(len(tokenize(content)))
        title_overlaps.append(lexical_overlap(row.get("question", ""), title))
    usage = row.get("usage") or {}
    return {
        "root_full_context_tokens": row.get("root_full_context_tokens", 0),
        "root_actual_retrieved_k": row.get("root_actual_retrieved_k", 0),
        "root_query_word_count": len(tokenize(row.get("query", ""))),
        "root_query_question_overlap": lexical_overlap(
            row.get("question", ""), row.get("query", "")
        ),
        "root_unique_title_ratio": len(set(titles)) / max(1, len(titles)),
        "root_mean_document_words": float(np.mean(content_lengths)) if content_lengths else 0.0,
        "root_max_title_overlap": max(title_overlaps) if title_overlaps else 0.0,
        "root_mean_title_overlap": float(np.mean(title_overlaps)) if title_overlaps else 0.0,
        "root_thinking_tokens": usage.get("thinking_tokens", 0) or 0,
        "root_output_tokens": usage.get("llm_output_tokens", 0) or 0,
    }


def bh_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 1.0
    for reverse_rank in range(len(values) - 1, -1, -1):
        index = order[reverse_rank]
        rank = reverse_rank + 1
        running = min(running, values[index] * len(values) / rank)
        adjusted[index] = running
    return adjusted


def bootstrap_mean_difference(values, repetitions=50000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(values), size=(repetitions, len(values)))
    means = values[samples].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def cv_oracle_bin(features, labels):
    splitter = RepeatedStratifiedKFold(
        n_splits=5, n_repeats=20, random_state=SEED
    )
    fold_metrics = []
    for train, test in splitter.split(features, labels):
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                max_iter=5000,
                class_weight="balanced",
                random_state=SEED,
            ),
        )
        model.fit(features.iloc[train], labels[train])
        prediction = model.predict(features.iloc[test])
        fold_metrics.append({
            "accuracy": accuracy_score(labels[test], prediction),
            "balanced_accuracy": balanced_accuracy_score(labels[test], prediction),
            "macro_f1": f1_score(labels[test], prediction, average="macro"),
        })
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in fold_metrics])),
            "sd_across_folds": float(np.std([row[metric] for row in fold_metrics])),
        }
        for metric in ("accuracy", "balanced_accuracy", "macro_f1")
    }


def loo_group_policy(scores, quality, cost, columns, alpha=5.0):
    outcomes = []
    ids = quality.index
    for sample_id in ids:
        training_ids = ids[ids != sample_id]
        key = tuple(scores.loc[sample_id, column] for column in columns)
        group_ids = [
            candidate
            for candidate in training_ids
            if tuple(scores.loc[candidate, column] for column in columns) == key
        ]
        predicted = {}
        for cap in COMMON_CAPS:
            global_mean = quality.loc[training_ids, cap].mean()
            group_sum = quality.loc[group_ids, cap].sum() if group_ids else 0.0
            predicted[cap] = (group_sum + alpha * global_mean) / (len(group_ids) + alpha)
        selected = max(
            COMMON_CAPS,
            key=lambda cap: (predicted[cap], -cost.loc[training_ids, cap].mean()),
        )
        outcomes.append({
            "sample_id": sample_id,
            "selected_cap": selected,
            "quality": quality.loc[sample_id, selected],
            "visible_tokens": cost.loc[sample_id, selected],
        })
    return pd.DataFrame(outcomes).set_index("sample_id")


def main():
    source = Path("fanout_query_prior_mapping_input")
    output = Path("fanoutqa_query_prior_oracle_mapping_20260911")
    output.mkdir(parents=True, exist_ok=True)

    scores = pd.read_json(source / "query_prior_scores.jsonl", lines=True)
    scores = scores.rename(columns={"question_id": "sample_id"})
    questions = pd.read_json(source / "questions.jsonl", lines=True)
    budget = pd.DataFrame(read_jsonl(source / "budget_scored_records.jsonl.gz", True))
    blank = pd.DataFrame(read_jsonl(source / "blank_scored_records.jsonl.gz", True))
    roots = pd.DataFrame(read_jsonl(source / "roots.jsonl.gz", True))
    oracle = pd.read_csv(
        source / "quality_then_cost_oracle.csv",
        dtype={"sample_id": str, "oracle_condition": str},
    )
    for column in (
        "blank_loose",
        "blank_visible_evidence_tokens",
        "oracle_loose",
        "oracle_minus_blank",
        "oracle_visible_evidence_tokens",
    ):
        oracle[column] = pd.to_numeric(oracle[column])

    assert len(scores) == len(questions) == len(blank) == len(roots) == len(oracle) == 100
    expected_ids = set(questions["sample_id"])
    assert set(scores["sample_id"]) == expected_ids
    assert set(blank["sample_id"]) == expected_ids
    assert set(roots["sample_id"]) == expected_ids
    assert set(oracle["sample_id"]) == expected_ids
    query_by_id = questions.set_index("sample_id")["question"]
    assert all(
        scores.set_index("sample_id").loc[sample_id, "query"] == query_by_id[sample_id]
        for sample_id in expected_ids
    )

    selected = budget.merge(
        oracle[["sample_id", "oracle_condition"]],
        left_on=["sample_id", "condition_label"],
        right_on=["sample_id", "oracle_condition"],
    )
    assert len(selected) == 100

    sample_static = (
        budget.sort_values(["sample_id", "condition_label"])
        .groupby("sample_id", as_index=False)
        .first()
    )
    frame = scores.merge(oracle, on="sample_id")
    frame = frame.merge(
        selected[[
            "sample_id",
            "budget",
            "visible_evidence_tokens_total",
            "search_calls",
            "standalone_llm_input_tokens",
            "standalone_llm_output_tokens",
            "necessary_title_retrieval_recall",
            "necessary_title_visibility_recall",
        ]],
        on="sample_id",
    )
    frame = frame.rename(columns={
        "budget": "oracle_cap_per_search",
        "visible_evidence_tokens_total": "oracle_visible_tokens_recomputed",
        "search_calls": "oracle_search_calls",
        "standalone_llm_input_tokens": "oracle_llm_input_tokens",
        "standalone_llm_output_tokens": "oracle_llm_output_tokens",
        "necessary_title_retrieval_recall": "oracle_retrieval_recall",
        "necessary_title_visibility_recall": "oracle_visibility_recall",
    })
    frame = frame.merge(
        sample_static[[
            "sample_id",
            "categories",
            "decomposition_leaf_count",
            "necessary_evidence_count",
        ]],
        on="sample_id",
    )
    frame = frame.merge(
        blank[[
            "sample_id",
            "search_calls",
            "retrieved_docs_total",
            "necessary_title_retrieval_recall",
            "necessary_title_visibility_recall",
            "answer_component_evidence_coverage",
            "official_loose_accuracy",
            "visible_evidence_tokens_total",
        ]],
        on="sample_id",
    )
    frame = frame.rename(columns={
        "search_calls": "blank_search_calls",
        "retrieved_docs_total": "blank_retrieved_docs",
        "necessary_title_retrieval_recall": "blank_retrieval_recall",
        "necessary_title_visibility_recall": "blank_visibility_recall",
        "answer_component_evidence_coverage": "blank_answer_evidence_coverage",
        "official_loose_accuracy": "blank_loose_recomputed",
        "visible_evidence_tokens_total": "blank_visible_tokens_recomputed",
    })

    lexical = pd.DataFrame([query_features(text) for text in frame["query"]])
    root_feature_frame = pd.DataFrame([
        root_features(row)
        for row in roots.set_index("sample_id").loc[frame["sample_id"]].to_dict("records")
    ])
    frame = pd.concat(
        [frame.reset_index(drop=True), lexical, root_feature_frame], axis=1
    )
    category_counts = Counter(
        category
        for categories in frame["categories"]
        for category in categories
    )
    frequent_categories = sorted(
        category for category, count in category_counts.items() if count >= 5
    )
    category_columns = []
    for category in frequent_categories:
        column = "topic_" + re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")
        frame[column] = frame["categories"].apply(lambda values: int(category in values))
        category_columns.append(column)
    frame["oracle_budget_bin"] = np.where(
        frame["oracle_cap_per_search"] <= 200,
        "low_100_200",
        np.where(
            frame["oracle_cap_per_search"] <= 500,
            "mid_300_500",
            "high_600_plus",
        ),
    )

    common = budget[budget["condition_label"].isin([str(cap) for cap in COMMON_CAPS])].copy()
    common["cap"] = common["condition_label"].astype(int)
    quality = common.pivot(
        index="sample_id", columns="cap", values="official_loose_accuracy"
    )[COMMON_CAPS]
    costs = common.pivot(
        index="sample_id", columns="cap", values="visible_evidence_tokens_total"
    )[COMMON_CAPS]
    assert not quality.isna().any().any()
    assert not costs.isna().any().any()
    quality = quality.loc[frame["sample_id"]]
    costs = costs.loc[frame["sample_id"]]
    score_indexed = frame.set_index("sample_id")

    association_predictors = [
        "read_score",
        "search_score",
        "read_confidence",
        "search_confidence",
        "query_word_count",
        "explicit_cardinality",
        "has_temporal_constraint",
        "has_ranking",
        "has_multi_output_form",
        "root_full_context_tokens",
        "root_query_question_overlap",
        "root_max_title_overlap",
    ]
    association_targets = [
        "oracle_cap_per_search",
        "oracle_visible_evidence_tokens",
        "oracle_search_calls",
        "oracle_loose",
        "oracle_minus_blank",
        "decomposition_leaf_count",
        "necessary_evidence_count",
        "blank_search_calls",
        "blank_retrieval_recall",
    ]
    associations = []
    for predictor in association_predictors:
        for target in association_targets:
            rho, p_value = spearmanr(frame[predictor], frame[target])
            associations.append({
                "predictor": predictor,
                "target": target,
                "spearman_rho": float(rho),
                "p_value": float(p_value),
            })
    adjusted = bh_adjust([row["p_value"] for row in associations])
    for row, q_value in zip(associations, adjusted):
        row["bh_q_value_across_reported_associations"] = float(q_value)
    association_frame = pd.DataFrame(associations).sort_values(
        ["bh_q_value_across_reported_associations", "p_value"]
    )
    association_frame.to_csv(output / "associations.csv", index=False)

    score_columns = [
        "read_score",
        "search_score",
        "read_confidence",
        "search_confidence",
    ]
    lexical_columns = list(lexical.columns)
    root_columns = list(root_feature_frame.columns)
    gold_columns = [
        "decomposition_leaf_count",
        "necessary_evidence_count",
        "blank_retrieval_recall",
        "blank_visibility_recall",
    ]
    feature_sets = {
        "scalar_scores": score_columns,
        "query_lexical_structure": lexical_columns,
        "scores_plus_query_structure": score_columns + lexical_columns,
        "benchmark_topic_metadata_not_directly_deployable": category_columns,
        "scores_plus_topic_metadata": score_columns + category_columns,
        "post_first_search_state": root_columns,
        "scores_query_plus_root_state": score_columns + lexical_columns + root_columns,
        "gold_diagnostic_not_deployable": gold_columns,
    }
    labels = frame["oracle_budget_bin"].map({
        "low_100_200": 0,
        "mid_300_500": 1,
        "high_600_plus": 2,
    }).to_numpy()
    cv_results = {
        name: cv_oracle_bin(frame[columns], labels)
        for name, columns in feature_sets.items()
    }
    majority_prediction = np.zeros_like(labels)
    cv_results["majority_low_baseline"] = {
        "accuracy": {"mean": float(accuracy_score(labels, majority_prediction)), "sd_across_folds": 0.0},
        "balanced_accuracy": {"mean": float(balanced_accuracy_score(labels, majority_prediction)), "sd_across_folds": 0.0},
        "macro_f1": {"mean": float(f1_score(labels, majority_prediction, average="macro")), "sd_across_folds": 0.0},
    }

    policy_rows = []
    for cap in COMMON_CAPS:
        policy_rows.append({
            "policy": f"fixed_{cap}",
            "mean_loose_accuracy": quality[cap].mean(),
            "mean_visible_tokens": costs[cap].mean(),
        })
    policy_rows.append({
        "policy": "quality_then_cost_oracle",
        "mean_loose_accuracy": frame["oracle_loose"].mean(),
        "mean_visible_tokens": frame["oracle_visible_evidence_tokens"].mean(),
    })

    learned_policies = {}
    for name, columns in {
        "loo_read_score_group": ["read_score"],
        "loo_search_score_group": ["search_score"],
        "loo_read_search_pair_group": ["read_score", "search_score"],
    }.items():
        result = loo_group_policy(score_indexed, quality, costs, columns)
        learned_policies[name] = result
        policy_rows.append({
            "policy": name,
            "mean_loose_accuracy": result["quality"].mean(),
            "mean_visible_tokens": result["visible_tokens"].mean(),
        })
        result.reset_index().to_csv(output / f"{name}.csv", index=False)

    policies = pd.DataFrame(policy_rows)
    policies.to_csv(output / "policy_comparison.csv", index=False)
    read_policy = learned_policies["loo_read_score_group"]
    fixed_700 = quality[700]
    delta = read_policy["quality"] - fixed_700
    read_policy_test = {
        "quality_difference_vs_fixed_700": float(delta.mean()),
        "quality_difference_bootstrap_95ci": bootstrap_mean_difference(delta.to_numpy()),
        "wins": int((delta > 0).sum()),
        "ties": int((delta == 0).sum()),
        "losses": int((delta < 0).sum()),
        "wilcoxon_two_sided_p": float(wilcoxon(delta).pvalue),
        "visible_token_difference_vs_fixed_700": float(
            read_policy["visible_tokens"].mean() - costs[700].mean()
        ),
        "selected_cap_counts": {
            str(key): int(value)
            for key, value in read_policy["selected_cap"].value_counts().sort_index().items()
        },
    }

    score_group_table = (
        frame.groupby(["read_score", "search_score"], as_index=False)
        .agg(
            questions=("sample_id", "size"),
            oracle_cap_median=("oracle_cap_per_search", "median"),
            oracle_cap_mean=("oracle_cap_per_search", "mean"),
            oracle_visible_tokens_mean=("oracle_visible_evidence_tokens", "mean"),
            oracle_quality_mean=("oracle_loose", "mean"),
            oracle_gain_over_blank_mean=("oracle_minus_blank", "mean"),
            necessary_evidence_mean=("necessary_evidence_count", "mean"),
        )
    )
    score_group_table.to_csv(output / "score_group_oracle_summary.csv", index=False)

    category_rows = []
    for category in frequent_categories:
        subset = frame[frame["categories"].apply(lambda values: category in values)]
        category_rows.append({
            "category": category,
            "questions": len(subset),
            "oracle_cap_mean": subset["oracle_cap_per_search"].mean(),
            "oracle_cap_median": subset["oracle_cap_per_search"].median(),
            "oracle_quality_mean": subset["oracle_loose"].mean(),
            "oracle_gain_over_blank_mean": subset["oracle_minus_blank"].mean(),
        })
    category_frame = pd.DataFrame(category_rows).sort_values(
        "oracle_cap_mean", ascending=False
    )
    category_frame.to_csv(output / "category_oracle_summary_exploratory.csv", index=False)

    coverage_by_sample = (
        budget.groupby("sample_id", as_index=False)
        .agg(
            maximum_necessary_title_visibility_recall=(
                "necessary_title_visibility_recall", "max"
            ),
            ever_all_necessary_titles_retrieved=(
                "necessary_title_all_retrieved", "max"
            ),
            ever_all_necessary_titles_visible=(
                "necessary_title_all_visible", "max"
            ),
        )
    )

    frame.to_csv(output / "aligned_per_question.csv", index=False)
    report = {
        "integrity": {
            "aligned_questions": len(frame),
            "exact_id_sets": True,
            "exact_query_text_match": True,
            "common_budget_grid": COMMON_CAPS,
            "budget_records": len(budget),
        },
        "oracle": {
            "definition": "maximize official loose accuracy, then minimize visible evidence tokens, then output tokens",
            "condition_counts": frame["oracle_condition"].value_counts().sort_index().to_dict(),
            "budget_bin_counts": frame["oracle_budget_bin"].value_counts().to_dict(),
            "mean_loose_accuracy": float(frame["oracle_loose"].mean()),
            "mean_visible_evidence_tokens": float(frame["oracle_visible_evidence_tokens"].mean()),
            "mean_gain_over_blank": float(frame["oracle_minus_blank"].mean()),
        },
        "targeted_associations": {
            f"{predictor}__{target}": next(
                row for row in associations
                if row["predictor"] == predictor and row["target"] == target
            )
            for predictor, target in [
                ("read_score", "oracle_cap_per_search"),
                ("read_score", "oracle_visible_evidence_tokens"),
                ("search_score", "oracle_search_calls"),
                ("search_score", "oracle_cap_per_search"),
                ("read_score", "necessary_evidence_count"),
                ("search_score", "necessary_evidence_count"),
                ("read_score", "oracle_loose"),
                ("search_score", "oracle_loose"),
                ("has_temporal_constraint", "oracle_cap_per_search"),
            ]
        },
        "oracle_bin_prediction_repeated_5fold": cv_results,
        "necessary_evidence_coverage_diagnostic": {
            "mean_per_sample_maximum_title_visibility_recall": float(
                coverage_by_sample["maximum_necessary_title_visibility_recall"].mean()
            ),
            "samples_ever_all_titles_retrieved": int(
                coverage_by_sample["ever_all_necessary_titles_retrieved"].sum()
            ),
            "samples_ever_all_titles_visible": int(
                coverage_by_sample["ever_all_necessary_titles_visible"].sum()
            ),
            "record_level_all_titles_retrieved_rate": float(
                budget["necessary_title_all_retrieved"].mean()
            ),
            "record_level_all_titles_visible_rate": float(
                budget["necessary_title_all_visible"].mean()
            ),
        },
        "exploratory_topic_summary": category_frame.to_dict("records"),
        "policy_comparison": policies.to_dict("records"),
        "read_score_policy_test": read_policy_test,
        "interpretation_guardrails": [
            "Oracle labels are selected from one deterministic rollout per budget and contain answer-instability noise.",
            "Quality-then-cost selects the cheapest observed maximizer, so a low oracle cap can mean either evidence sufficiency or failure of larger budgets to improve quality.",
            "The query scores are concentrated at 3-4, limiting their resolution.",
            "Gold diagnostic features are not deployable and are reported only for construct validation.",
            "Any learned policy result on these same 100 questions is exploratory until tested on a disjoint split or benchmark.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
