from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_transfer as base


OUT = Path(__file__).resolve().parent / "results"


def score_midranks(scores: pd.DataFrame) -> dict[int, float]:
    counts = scores["read_score"].astype(int).value_counts().sort_index()
    total = float(counts.sum())
    below = 0
    result = {}
    for score, count in counts.items():
        result[int(score)] = (below + count / 2.0) / total
        below += int(count)
    return result


def quantile_bridge(source_scores: pd.DataFrame, target_scores: pd.DataFrame) -> dict[int, int]:
    source = score_midranks(source_scores)
    target = score_midranks(target_scores)
    bridge = {}
    for target_score, target_q in target.items():
        # Deterministic tie break: smaller quantile distance, then smaller score.
        source_score = min(source, key=lambda s: (abs(source[s] - target_q), s))
        bridge[target_score] = source_score
    return bridge


def apply_quantile_mapping(
    target: pd.DataFrame,
    target_scores: pd.DataFrame,
    bridge: dict[int, int],
    source_budget_mapping: dict[str, int],
    fallback: int,
    policy: str,
    direction: str,
    source_metric: str,
) -> pd.DataFrame:
    meta = target_scores[["sample_id", "read_score"]].copy()
    meta["source_equivalent_score"] = meta.read_score.astype(int).map(bridge)
    meta["selected_budget"] = meta.source_equivalent_score.map(
        lambda score: source_budget_mapping.get(str(int(score)), fallback)
    ).astype(int)
    chosen = target.merge(meta, on="sample_id", validate="many_to_one")
    chosen = chosen[chosen.budget == chosen.selected_budget].copy()
    if len(chosen) != target_scores.sample_id.nunique() or chosen.sample_id.duplicated().any():
        raise RuntimeError(f"Quantile mapping failed: {policy}")
    chosen["policy"] = policy
    chosen["direction"] = direction
    chosen["source_metric"] = source_metric
    chosen["grouping"] = "target-unlabeled quantile calibration"
    return chosen


def main():
    hs, h, h_full, fs, f, f_blank, _ = base.load_data()

    hotpot_exact, hotpot_fallback, _ = base.learn_mapping(h, hs, "answer_f1", "exact")
    fanout_exact, fanout_fallback, _ = base.learn_mapping(
        f, fs, "official_loose_accuracy", "exact"
    )

    h_to_f_bridge = quantile_bridge(hs, fs)
    f_to_h_bridge = quantile_bridge(fs, hs)

    h_to_f = apply_quantile_mapping(
        f,
        fs,
        h_to_f_bridge,
        hotpot_exact,
        hotpot_fallback,
        "transfer_hotpot_f1_quantile_calibrated",
        "HotpotQA->FanOutQA",
        "answer_f1",
    )
    f_to_h = apply_quantile_mapping(
        h,
        hs,
        f_to_h_bridge,
        fanout_exact,
        fanout_fallback,
        "transfer_fanout_loose_quantile_calibrated",
        "FanOutQA->HotpotQA",
        "official_loose_accuracy",
    )

    f_blank = f_blank.copy()
    f_blank["selected_budget"] = float("nan")
    f_blank["policy"] = "BLANK"
    f_blank["direction"] = "FanOutQA"
    f_blank["source_metric"] = "target comparator"
    f_blank["grouping"] = "none"
    f_fixed = base.fixed_policy(f, 500, "fixed_500", "FanOutQA")
    f_oracle = base.oracle_policy(
        f, "official_loose_accuracy", "common_oracle", "FanOutQA"
    )

    h_full = h_full.copy()
    h_full["selected_budget"] = float("nan")
    h_full["policy"] = "FULL"
    h_full["direction"] = "HotpotQA"
    h_full["source_metric"] = "target comparator"
    h_full["grouping"] = "none"
    h_fixed = base.fixed_policy(h, 600, "fixed_600", "HotpotQA")
    h_oracle = base.oracle_policy(h, "answer_f1", "common_oracle", "HotpotQA")

    summary = pd.concat(
        [
            base.summarize(pd.concat([h_to_f, f_blank, f_fixed, f_oracle]), "FanOutQA"),
            base.summarize(pd.concat([f_to_h, h_full, h_fixed, h_oracle]), "HotpotQA"),
        ],
        ignore_index=True,
        sort=False,
    )
    summary.to_csv(OUT / "quantile_calibrated_summary.csv", index=False, encoding="utf-8-sig")

    comparisons = []
    for candidate, references, metrics, target in [
        (
            h_to_f,
            {"BLANK": f_blank, "fixed_500": f_fixed, "common_oracle": f_oracle},
            ["official_loose_accuracy", "rougeL_f1", "total_llm_tokens"],
            "FanOutQA",
        ),
        (
            f_to_h,
            {"FULL": h_full, "fixed_600": h_fixed, "common_oracle": h_oracle},
            ["answer_f1", "answer_em", "total_llm_tokens"],
            "HotpotQA",
        ),
    ]:
        for reference_name, reference in references.items():
            for offset, metric in enumerate(metrics):
                comparisons.append(
                    {
                        "target_dataset": target,
                        "candidate": candidate.policy.iloc[0],
                        "reference": reference_name,
                        "metric": metric,
                        **base.paired_delta(
                            candidate,
                            reference,
                            metric,
                            base.SEED + len(comparisons) + offset,
                        ),
                    }
                )
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(
        OUT / "quantile_calibrated_comparisons.csv", index=False, encoding="utf-8-sig"
    )

    fanout_cases = h_to_f[[
        "sample_id", "question", "read_score", "source_equivalent_score",
        "selected_budget", "official_loose_accuracy", "official_strict_accuracy",
        "rougeL_f1", "total_llm_tokens",
    ]].merge(
        f_blank[[
            "sample_id", "official_loose_accuracy", "official_strict_accuracy",
            "rougeL_f1", "total_llm_tokens",
        ]],
        on="sample_id",
        suffixes=("_transfer", "_blank"),
        validate="one_to_one",
    )
    for metric in [
        "official_loose_accuracy", "official_strict_accuracy", "rougeL_f1",
        "total_llm_tokens",
    ]:
        fanout_cases[f"delta_{metric}"] = (
            fanout_cases[f"{metric}_transfer"] - fanout_cases[f"{metric}_blank"]
        )
    fanout_cases.to_csv(
        OUT / "quantile_hotpot_to_fanout_cases.csv", index=False, encoding="utf-8-sig"
    )

    hotpot_queries = hs[["sample_id", "query"]].drop_duplicates("sample_id")
    hotpot_cases = f_to_h[[
        "sample_id", "read_score", "source_equivalent_score", "selected_budget",
        "answer_f1", "answer_em", "total_llm_tokens",
    ]].merge(hotpot_queries, on="sample_id", how="left", validate="one_to_one").merge(
        h_full[["sample_id", "answer_f1", "answer_em", "total_llm_tokens"]],
        on="sample_id",
        suffixes=("_transfer", "_full"),
        validate="one_to_one",
    )
    for metric in ["answer_f1", "answer_em", "total_llm_tokens"]:
        hotpot_cases[f"delta_{metric}"] = (
            hotpot_cases[f"{metric}_transfer"] - hotpot_cases[f"{metric}_full"]
        )
    hotpot_cases.to_csv(
        OUT / "quantile_fanout_to_hotpot_cases.csv", index=False, encoding="utf-8-sig"
    )

    output = {
        "method": (
            "Map each target READ-score category to the source category with the nearest "
            "empirical midrank quantile. Target answers and trajectories are not used."
        ),
        "hotpot_score_midranks": score_midranks(hs),
        "fanout_score_midranks": score_midranks(fs),
        "hotpot_to_fanout_score_bridge": h_to_f_bridge,
        "fanout_to_hotpot_score_bridge": f_to_h_bridge,
        "hotpot_source_budget_mapping": hotpot_exact,
        "fanout_source_budget_mapping": fanout_exact,
        "summary": summary.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
    }
    (OUT / "quantile_calibrated_transfer.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
