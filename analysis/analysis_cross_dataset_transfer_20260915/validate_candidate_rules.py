from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import analyze_transfer as base
import analyze_quantile_transfer as qcal


BASE = Path(__file__).resolve().parent
OUT = BASE / "rule_validation"
BUDGETS = base.COMMON
SEED = base.SEED


NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "twenty": 20,
}


def midranks(scores: pd.DataFrame) -> dict[int, float]:
    return qcal.score_midranks(scores)


def query_feature_frame(scores: pd.DataFrame) -> pd.DataFrame:
    ranks = midranks(scores)
    rows = []
    for record in scores.itertuples(index=False):
        text = str(record.query)
        lower = text.lower()
        words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)
        numeric = [int(x) for x in re.findall(r"\b\d+\b", lower) if int(x) <= 100]
        word_numbers = [value for word, value in NUMBER_WORDS.items() if re.search(rf"\b{word}\b", lower)]
        requested_n = max(numeric + word_numbers + [1])
        rows.append({
            "sample_id": str(record.sample_id),
            "score_quantile": ranks[int(record.read_score)],
            "read_confidence": float(getattr(record, "read_confidence", 0.0)),
            "log_word_count": math.log1p(len(words)),
            "log_char_count": math.log1p(len(text)),
            "requested_n_log": math.log1p(requested_n),
            "comma_count": text.count(","),
            "quote_count": text.count('"') + text.count("'"),
            "and_count": len(re.findall(r"\band\b", lower)),
            "flag_each": float(bool(re.search(r"\beach\b", lower))),
            "flag_list": float(bool(re.search(r"\b(list|name|identify)\b", lower))),
            "flag_top": float(bool(re.search(r"\btop\b", lower))),
            "flag_all_both": float(bool(re.search(r"\b(all|both|every)\b", lower))),
            "flag_rank": float(bool(re.search(r"\b(largest|smallest|highest|lowest|most|least|rank|ranking)\b", lower))),
            "flag_compare": float(bool(re.search(r"\b(compare|difference|same|older|younger|more than|less than)\b", lower))),
            "flag_time": float(bool(re.search(r"\b(year|date|when|current|currently|latest|since|before|after)\b", lower))),
            "flag_count": float(bool(re.search(r"\b(how many|number of|percentage|percent|population)\b", lower))),
            "flag_where": float(bool(re.search(r"\b(where|which university|which company|located)\b", lower))),
            "flag_who": float(bool(re.search(r"\b(who|whose)\b", lower))),
            "flag_what": float(bool(re.search(r"\bwhat\b", lower))),
            "flag_which": float(bool(re.search(r"\bwhich\b", lower))),
        })
    return pd.DataFrame(rows).set_index("sample_id")


def design_matrix(data: pd.DataFrame, qfeatures: pd.DataFrame) -> np.ndarray:
    q = qfeatures.loc[data.sample_id.astype(str)].to_numpy(float)
    budget = data.budget.to_numpy(float) / max(BUDGETS)
    onehot = np.column_stack([(data.budget.to_numpy(int) == b).astype(float) for b in BUDGETS])
    interactions = np.concatenate([q * onehot[:, j:j + 1] for j in range(len(BUDGETS))], axis=1)
    return np.column_stack([q, budget, budget ** 2, onehot, interactions])


def select_rows(data: pd.DataFrame, prediction: np.ndarray, epsilon: float, policy: str) -> pd.DataFrame:
    work = data.copy()
    work["predicted_quality"] = prediction
    selected = []
    for _, group in work.groupby("sample_id", sort=False):
        threshold = float(group.predicted_quality.max()) - epsilon
        eligible = group[group.predicted_quality >= threshold]
        selected.append(eligible.sort_values("budget").iloc[0])
    out = pd.DataFrame(selected)
    out["selected_budget"] = out.budget.astype(int)
    out["policy"] = policy
    return out


def make_model(kind: str, parameter: float):
    if kind == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=parameter))
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=240,
            max_depth=int(parameter),
            min_samples_leaf=8,
            max_features=.7,
            n_jobs=-1,
            random_state=SEED,
        )
    raise ValueError(kind)


def choose_feature_model(data: pd.DataFrame, scores: pd.DataFrame, metric: str):
    qfeatures = query_feature_frame(scores)
    x = design_matrix(data, qfeatures)
    y = data[metric].to_numpy(float)
    groups = data.sample_id.to_numpy(str)
    candidates = [
        *(('ridge', alpha) for alpha in (.1, 1.0, 10.0, 100.0)),
        *(('rf', depth) for depth in (2, 3, 4, 5, 6)),
    ]
    epsilons = (0.0, .0025, .005, .01, .02)
    folds = GroupKFold(n_splits=5)
    cv_rows = []
    prediction_cache = {}
    for kind, parameter in candidates:
        prediction = np.empty(len(data), dtype=float)
        for train, valid in folds.split(x, y, groups):
            model = make_model(kind, parameter)
            model.fit(x[train], y[train])
            prediction[valid] = model.predict(x[valid])
        prediction_cache[(kind, parameter)] = prediction
        for epsilon in epsilons:
            chosen = select_rows(data, prediction, epsilon, "cv")
            cv_rows.append({
                "kind": kind,
                "parameter": parameter,
                "epsilon": epsilon,
                "source_cv_quality": float(chosen[metric].mean()),
                "source_cv_total_llm_tokens": float(chosen.total_llm_tokens.mean()),
                "mean_selected_budget": float(chosen.selected_budget.mean()),
            })
    cv = pd.DataFrame(cv_rows)
    # Source-only lexicographic selection. A 1e-4 quality tolerance prevents
    # floating noise from selecting a materially more expensive model.
    max_quality = cv.source_cv_quality.max()
    eligible = cv[cv.source_cv_quality >= max_quality - 1e-4]
    best = eligible.sort_values(
        ["source_cv_total_llm_tokens", "mean_selected_budget", "kind", "parameter", "epsilon"]
    ).iloc[0]
    model = make_model(str(best.kind), float(best.parameter))
    model.fit(x, y)
    return model, float(best.epsilon), qfeatures, cv, best.to_dict()


def apply_feature_model(model, epsilon, target, target_scores, policy):
    qfeatures = query_feature_frame(target_scores)
    prediction = model.predict(design_matrix(target, qfeatures))
    return select_rows(target, prediction, epsilon, policy)


def softmax(values, temperature):
    values = np.asarray(values, dtype=float)
    logits = (values - values.max()) / temperature
    weights = np.exp(logits)
    return weights / weights.sum()


def source_curves(data, scores, metric):
    merged = data.merge(scores[["sample_id", "read_score"]], on="sample_id", validate="many_to_one")
    curves = {}
    for score, group in merged.groupby("read_score"):
        curves[int(score)] = group.groupby("budget")[metric].mean().reindex(BUDGETS).to_numpy(float)
    global_curve = merged.groupby("budget")[metric].mean().reindex(BUDGETS).to_numpy(float)
    return curves, global_curve


def choose_soft_temperature(data, scores, metric):
    temperatures = (.0025, .005, .01, .02, .05, .1, .2, .5)
    ids = scores.sample_id.astype(str).to_numpy()
    folds = GroupKFold(n_splits=5)
    records = []
    for temperature in temperatures:
        qualities, costs, budgets = [], [], []
        for train_idx, valid_idx in folds.split(ids, groups=ids):
            train_ids = set(ids[train_idx])
            valid_ids = set(ids[valid_idx])
            curves, global_curve = source_curves(
                data[data.sample_id.isin(train_ids)], scores[scores.sample_id.isin(train_ids)], metric
            )
            valid_scores = scores[scores.sample_id.isin(valid_ids)].set_index("sample_id")
            for sample_id, group in data[data.sample_id.isin(valid_ids)].groupby("sample_id"):
                score = int(valid_scores.loc[sample_id, "read_score"])
                probability = softmax(curves.get(score, global_curve), temperature)
                ordered = group.set_index("budget").loc[BUDGETS]
                qualities.append(float(np.dot(probability, ordered[metric])))
                costs.append(float(np.dot(probability, ordered.total_llm_tokens)))
                budgets.append(float(np.dot(probability, BUDGETS)))
        records.append({
            "temperature": temperature,
            "source_cv_expected_quality": float(np.mean(qualities)),
            "source_cv_expected_total_llm_tokens": float(np.mean(costs)),
            "source_cv_expected_budget": float(np.mean(budgets)),
        })
    cv = pd.DataFrame(records)
    max_quality = cv.source_cv_expected_quality.max()
    best = cv[cv.source_cv_expected_quality >= max_quality - 1e-4].sort_values(
        ["source_cv_expected_total_llm_tokens", "source_cv_expected_budget", "temperature"]
    ).iloc[0]
    return float(best.temperature), cv, best.to_dict()


def apply_soft_policy(source, source_scores, target, target_scores, metric, temperature, policy):
    curves, global_curve = source_curves(source, source_scores, metric)
    bridge = qcal.quantile_bridge(source_scores, target_scores)
    score_by_id = target_scores.set_index("sample_id").read_score.astype(int)
    metric_columns = [
        column for column in [
            "official_loose_accuracy", "official_strict_accuracy", "rougeL_f1",
            "answer_f1", "answer_em", "answer_acc", "total_llm_tokens",
            "visible_evidence_tokens_total", "search_calls",
        ] if column in target.columns
    ]
    rows = []
    for sample_id, group in target.groupby("sample_id"):
        target_score = int(score_by_id.loc[sample_id])
        source_score = bridge[target_score]
        probability = softmax(curves.get(source_score, global_curve), temperature)
        ordered = group.set_index("budget").loc[BUDGETS]
        row = {
            "sample_id": sample_id,
            "read_score": target_score,
            "source_equivalent_score": source_score,
            "selected_budget": float(np.dot(probability, BUDGETS)),
            "budget_probability": json.dumps(
                {str(b): float(p) for b, p in zip(BUDGETS, probability)}, sort_keys=True
            ),
            "policy": policy,
        }
        for column in metric_columns:
            row[column] = float(np.dot(probability, ordered[column].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def prepare_baseline(rows, name):
    out = rows.copy()
    out["policy"] = name
    out["selected_budget"] = np.nan
    return out


def summary(rows, dataset):
    metrics = [
        column for column in [
            "official_loose_accuracy", "official_strict_accuracy", "rougeL_f1",
            "answer_f1", "answer_em", "answer_acc", "total_llm_tokens",
            "visible_evidence_tokens_total", "search_calls",
        ] if column in rows.columns
    ]
    result = rows.groupby("policy").agg(
        n=("sample_id", "nunique"),
        mean_selected_budget=("selected_budget", "mean"),
        **{column: (column, "mean") for column in metrics},
    ).reset_index()
    result["target_dataset"] = dataset
    return result


def paired(candidate, baseline, metrics, target):
    records = []
    for metric in metrics:
        records.append({
            "target_dataset": target,
            "candidate": candidate.policy.iloc[0],
            "baseline": baseline.policy.iloc[0],
            "metric": metric,
            **base.paired_delta(candidate, baseline, metric, SEED + len(records)),
        })
    return records


def run_direction(source_name, source, source_scores, target_name, target, target_scores, baseline):
    model, epsilon, _, feature_cv, feature_choice = choose_feature_model(
        source, source_scores, "answer_f1"
    )
    feature_policy = apply_feature_model(
        model, epsilon, target, target_scores, f"query_features_{source_name}_to_{target_name}"
    )
    temperature, soft_cv, soft_choice = choose_soft_temperature(
        source, source_scores, "answer_f1"
    )
    soft_policy = apply_soft_policy(
        source, source_scores, target, target_scores, "answer_f1", temperature,
        f"soft_probability_{source_name}_to_{target_name}",
    )

    metrics = ["answer_f1", "total_llm_tokens"]
    if target_name == "FanOutQA":
        metrics = ["official_loose_accuracy", "official_strict_accuracy", "rougeL_f1", *metrics]
    else:
        metrics = ["answer_em", "answer_acc", *metrics]
    comparisons = paired(feature_policy, baseline, metrics, target_name)
    comparisons += paired(soft_policy, baseline, metrics, target_name)
    return {
        "feature_policy": feature_policy,
        "soft_policy": soft_policy,
        "feature_cv": feature_cv,
        "soft_cv": soft_cv,
        "feature_choice": feature_choice,
        "soft_choice": soft_choice,
        "comparisons": comparisons,
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hs, h, h_full, fs, f, f_blank, _ = base.load_data()
    f_blank = prepare_baseline(f_blank, "BLANK")
    h_full = prepare_baseline(h_full, "FULL")

    h_to_f = run_direction("HotpotQA", h, hs, "FanOutQA", f, fs, f_blank)
    f_to_h = run_direction("FanOutQA", f, fs, "HotpotQA", h, hs, h_full)

    policies = pd.concat([
        h_to_f["feature_policy"], h_to_f["soft_policy"], f_blank,
        f_to_h["feature_policy"], f_to_h["soft_policy"], h_full,
    ], ignore_index=True, sort=False)
    summaries = pd.concat([
        summary(pd.concat([h_to_f["feature_policy"], h_to_f["soft_policy"], f_blank]), "FanOutQA"),
        summary(pd.concat([f_to_h["feature_policy"], f_to_h["soft_policy"], h_full]), "HotpotQA"),
    ], ignore_index=True, sort=False)
    comparisons = pd.DataFrame(h_to_f["comparisons"] + f_to_h["comparisons"])

    policies.to_csv(OUT / "per_question_rule_policies.csv", index=False, encoding="utf-8-sig")
    summaries.to_csv(OUT / "rule_summary.csv", index=False, encoding="utf-8-sig")
    comparisons.to_csv(OUT / "paired_vs_unbudgeted_baseline.csv", index=False, encoding="utf-8-sig")
    h_to_f["feature_cv"].to_csv(OUT / "hotpot_source_feature_cv.csv", index=False, encoding="utf-8-sig")
    f_to_h["feature_cv"].to_csv(OUT / "fanout_source_feature_cv.csv", index=False, encoding="utf-8-sig")
    h_to_f["soft_cv"].to_csv(OUT / "hotpot_source_soft_cv.csv", index=False, encoding="utf-8-sig")
    f_to_h["soft_cv"].to_csv(OUT / "fanout_source_soft_cv.csv", index=False, encoding="utf-8-sig")

    manifest = {
        "protocol": {
            "source_selection": "5-fold GroupKFold by question on source only",
            "training_metric": "answer_f1 on both datasets",
            "target_labels_used_for_selection": False,
            "target_unsupervised_information": "READ-score empirical midrank distribution",
            "budgets": BUDGETS,
            "primary_unbudgeted_baselines": {"FanOutQA": "BLANK", "HotpotQA": "FULL"},
        },
        "source_choices": {
            "HotpotQA_feature": h_to_f["feature_choice"],
            "HotpotQA_soft": h_to_f["soft_choice"],
            "FanOutQA_feature": f_to_h["feature_choice"],
            "FanOutQA_soft": f_to_h["soft_choice"],
        },
        "summary": summaries.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
