from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "results"


def main():
    integrity = json.loads((OUT / "integrity.json").read_text(encoding="utf-8"))
    assert integrity["passed"] is True
    assert integrity["hotpot"]["complete_common_grid"] == 500
    assert integrity["fanout"]["complete_common_grid"] == 100

    direct = pd.read_csv(OUT / "policy_summary.csv")
    assert (direct.loc[direct.target_dataset == "HotpotQA", "n"] == 500).all()
    assert (direct.loc[direct.target_dataset == "FanOutQA", "n"] == 100).all()

    quantile = pd.read_csv(OUT / "quantile_calibrated_summary.csv")
    fanout = quantile[
        quantile.policy == "transfer_hotpot_f1_quantile_calibrated"
    ].iloc[0]
    assert fanout.n == 100
    assert abs(fanout.official_loose_accuracy - 0.4140297619047619) < 1e-12

    harmonized = pd.read_csv(OUT / "harmonized_metric_summary.csv")
    hotpot = harmonized[
        harmonized.policy == "transfer_fanout_answerf1_coarse"
    ].iloc[0]
    assert hotpot.n == 500
    assert abs(hotpot.answer_f1 - 0.4939306377082417) < 1e-12

    fanout_cases = pd.read_csv(OUT / "quantile_hotpot_to_fanout_cases.csv")
    hotpot_cases = pd.read_csv(OUT / "quantile_fanout_to_hotpot_cases.csv")
    assert len(fanout_cases) == fanout_cases.sample_id.nunique() == 100
    assert len(hotpot_cases) == hotpot_cases.sample_id.nunique() == 500

    anchored = pd.read_csv(OUT / "baseline_anchored_comparisons.csv")
    assert len(anchored) == 7
    assert set(anchored.loc[anchored.target_dataset == "FanOutQA", "unbudgeted_baseline"]) == {"BLANK"}
    assert set(anchored.loc[anchored.target_dataset == "HotpotQA", "unbudgeted_baseline"]) == {"FULL"}

    for name in [
        "cross_dataset_transfer_with_calibration.png",
        "cross_dataset_transfer_with_calibration.pdf",
        "quantile_calibrated_transfer.json",
        "harmonized_metric_transfer.json",
        "mapping_stability.csv",
        "oracle_budget_correlations.json",
        "task_difficulty_baseline_comparison.json",
    ]:
        assert (OUT / name).stat().st_size > 0

    report = {
        "passed": True,
        "hotpot_questions": 500,
        "fanout_questions": 100,
        "common_budget_grid": [100, 200, 300, 400, 500, 600],
        "fanout_quantile_transfer_loose": float(fanout.official_loose_accuracy),
        "hotpot_metric_matched_transfer_f1": float(hotpot.answer_f1),
        "artifacts_checked": 8,
    }
    (OUT / "FINAL_AUDIT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
