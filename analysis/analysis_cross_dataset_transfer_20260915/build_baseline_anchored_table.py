from __future__ import annotations

from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "results"


def one(frame, dataset, policy):
    rows = frame[(frame.target_dataset == dataset) & (frame.policy == policy)]
    if len(rows) != 1:
        raise RuntimeError((dataset, policy, len(rows)))
    return rows.iloc[0]


def main():
    direct = pd.read_csv(OUT / "policy_summary.csv")
    quantile = pd.read_csv(OUT / "quantile_calibrated_summary.csv")
    harmonized = pd.read_csv(OUT / "harmonized_metric_summary.csv")

    specs = [
        ("FanOutQA", direct, "transfer_hotpot_f1_coarse", "Hotpot F1 direct"),
        ("FanOutQA", direct, "transfer_hotpot_em_coarse", "Hotpot EM direct"),
        (
            "FanOutQA", quantile, "transfer_hotpot_f1_quantile_calibrated",
            "Hotpot F1 quantile-calibrated",
        ),
        ("HotpotQA", direct, "transfer_fanout_loose_coarse", "FanOut Loose direct"),
        (
            "HotpotQA", quantile, "transfer_fanout_loose_quantile_calibrated",
            "FanOut Loose quantile-calibrated",
        ),
        (
            "HotpotQA", harmonized, "transfer_fanout_answerf1_coarse",
            "FanOut answer-F1 metric-matched",
        ),
        (
            "HotpotQA", harmonized,
            "transfer_fanout_answerf1_quantile_calibrated",
            "FanOut answer-F1 metric-matched + quantile",
        ),
    ]

    rows = []
    for dataset, frame, policy, label in specs:
        if dataset == "FanOutQA":
            metric = "official_loose_accuracy"
            baseline_name = "BLANK"
            baseline = one(direct, dataset, baseline_name)
            fixed = one(direct, dataset, "fixed_500")
        else:
            metric = "answer_f1"
            baseline_name = "FULL"
            baseline = one(direct, dataset, baseline_name)
            fixed = one(direct, dataset, "fixed_600")
        oracle = one(direct, dataset, "common_oracle")
        candidate = one(frame, dataset, policy)
        q = float(candidate[metric])
        bq = float(baseline[metric])
        oq = float(oracle[metric])
        rows.append({
            "target_dataset": dataset,
            "policy": policy,
            "policy_label": label,
            "primary_metric": metric,
            "policy_quality": q,
            "unbudgeted_baseline": baseline_name,
            "unbudgeted_baseline_quality": bq,
            "delta_vs_unbudgeted_baseline": q - bq,
            "policy_mean_total_llm_tokens": float(candidate.total_llm_tokens),
            "unbudgeted_baseline_mean_total_llm_tokens": float(baseline.total_llm_tokens),
            "token_saving_vs_unbudgeted_baseline_fraction": (
                1.0 - float(candidate.total_llm_tokens) / float(baseline.total_llm_tokens)
            ),
            "best_fixed_common_budget": int(fixed.mean_selected_budget),
            "best_fixed_quality": float(fixed[metric]),
            "delta_vs_best_fixed": q - float(fixed[metric]),
            "common_oracle_quality": oq,
            "fraction_of_unbudgeted_to_oracle_gap_recovered": (
                (q - bq) / (oq - bq) if oq != bq else float("nan")
            ),
        })
    pd.DataFrame(rows).to_csv(
        OUT / "baseline_anchored_comparisons.csv", index=False, encoding="utf-8-sig"
    )


if __name__ == "__main__":
    main()
