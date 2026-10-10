from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import analyze_transfer as base
import analyze_quantile_transfer as quantile


OUT = Path(__file__).resolve().parent / "results"


def prep_full(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    rows["selected_budget"] = float("nan")
    rows["policy"] = "FULL"
    rows["direction"] = "HotpotQA"
    rows["source_metric"] = "target comparator"
    rows["grouping"] = "none"
    return rows


def main():
    hs, h, h_full, fs, f, _, _ = base.load_data()
    direct_mapping, direct_fallback, _ = base.learn_mapping(
        f, fs, "answer_f1", "coarse"
    )
    direct = base.apply_mapping(
        h,
        hs,
        direct_mapping,
        direct_fallback,
        "coarse",
        "transfer_fanout_answerf1_coarse",
        "FanOutQA->HotpotQA",
        "answer_f1",
    )

    exact_mapping, exact_fallback, _ = base.learn_mapping(
        f, fs, "answer_f1", "exact"
    )
    bridge = quantile.quantile_bridge(fs, hs)
    calibrated = quantile.apply_quantile_mapping(
        h,
        hs,
        bridge,
        exact_mapping,
        exact_fallback,
        "transfer_fanout_answerf1_quantile_calibrated",
        "FanOutQA->HotpotQA",
        "answer_f1",
    )

    fixed = base.fixed_policy(h, 600, "fixed_600", "HotpotQA")
    full = prep_full(h_full)
    oracle = base.oracle_policy(h, "answer_f1", "common_oracle", "HotpotQA")
    all_rows = pd.concat([direct, calibrated, fixed, full, oracle], ignore_index=True)
    summary = base.summarize(all_rows, "HotpotQA")
    summary.to_csv(OUT / "harmonized_metric_summary.csv", index=False, encoding="utf-8-sig")

    comparisons = []
    for candidate in [direct, calibrated]:
        for reference_name, reference in {
            "fixed_600": fixed,
            "FULL": full,
            "common_oracle": oracle,
        }.items():
            for metric in ["answer_f1", "answer_em", "answer_acc", "total_llm_tokens"]:
                comparisons.append({
                    "candidate": candidate.policy.iloc[0],
                    "reference": reference_name,
                    "metric": metric,
                    **base.paired_delta(
                        candidate, reference, metric, base.SEED + len(comparisons)
                    ),
                })
    comparisons = pd.DataFrame(comparisons)
    comparisons.to_csv(
        OUT / "harmonized_metric_comparisons.csv", index=False, encoding="utf-8-sig"
    )
    result = {
        "purpose": "FanOutQA-to-HotpotQA robustness check using answer F1 on both datasets",
        "direct_mapping": direct_mapping,
        "direct_unseen_group_fallback": direct_fallback,
        "quantile_exact_mapping": exact_mapping,
        "quantile_score_bridge": bridge,
        "summary": summary.to_dict("records"),
        "comparisons": comparisons.to_dict("records"),
    }
    (OUT / "harmonized_metric_transfer.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
