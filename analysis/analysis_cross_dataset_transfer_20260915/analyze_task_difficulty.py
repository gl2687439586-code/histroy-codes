from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

import analyze_transfer as base


OUT = Path(__file__).resolve().parent / "results"


def describe(frame, columns):
    result = {}
    for column in columns:
        result[column] = {
            "mean": float(frame[column].mean()),
            "median": float(frame[column].median()),
            "min": float(frame[column].min()),
            "max": float(frame[column].max()),
        }
    return result


def main():
    hs, _, h_full, fs, _, f_blank, _ = base.load_data()
    hotpot_scores = hs.read_score.to_numpy(int)
    fanout_scores = fs.read_score.to_numpy(int)
    test = mannwhitneyu(fanout_scores, hotpot_scores, alternative="two-sided")
    cliffs_delta = float(
        (fanout_scores[:, None] > hotpot_scores).mean()
        - (fanout_scores[:, None] < hotpot_scores).mean()
    )

    workload_columns = [
        "search_calls",
        "visible_evidence_tokens_total",
        "standalone_llm_input_tokens",
        "standalone_llm_output_tokens",
        "standalone_model_calls",
        "total_llm_tokens",
    ]
    hotpot_workload = describe(h_full, workload_columns)
    fanout_workload = describe(f_blank, workload_columns)
    ratios = {
        column: fanout_workload[column]["mean"] / hotpot_workload[column]["mean"]
        for column in workload_columns
    }
    result = {
        "interpretation_boundary": (
            "The READ prompt/model are shared, but retriever backend, Wikipedia snapshot, "
            "answer structure, and official metrics differ. Results are consistent with "
            "higher FanOutQA demand, not a causal proof of intrinsic benchmark difficulty."
        ),
        "read_score": {
            "hotpot_mean": float(hotpot_scores.mean()),
            "fanout_mean": float(fanout_scores.mean()),
            "mann_whitney_u": float(test.statistic),
            "p_value": float(test.pvalue),
            "cliffs_delta_fanout_minus_hotpot": cliffs_delta,
        },
        "unbudgeted_baselines": {
            "HotpotQA_FULL": hotpot_workload,
            "FanOutQA_BLANK": fanout_workload,
            "fanout_over_hotpot_mean_ratios": ratios,
        },
        "task_structure": {
            "HotpotQA_FULL": describe(
                h_full, ["gold_supporting_title_count", "gold_supporting_fact_count"]
            ),
            "FanOutQA_BLANK": describe(
                f_blank, ["decomposition_leaf_count", "necessary_evidence_count"]
            ),
        },
        "retrieval_and_coverage": {
            "HotpotQA_FULL": describe(
                h_full,
                ["cumulative_retrieved_title_recall", "cumulative_visible_title_recall"],
            ),
            "FanOutQA_BLANK": describe(
                f_blank,
                ["necessary_title_retrieval_recall", "answer_component_evidence_coverage"],
            ),
        },
    }
    (OUT / "task_difficulty_baseline_comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
