from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "rule_validation"


def main():
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["protocol"]["target_labels_used_for_selection"] is False
    assert manifest["protocol"]["training_metric"] == "answer_f1 on both datasets"
    rows = pd.read_csv(OUT / "per_question_rule_policies.csv")
    assert len(rows) == 1800
    expected = {
        "BLANK": 100,
        "query_features_HotpotQA_to_FanOutQA": 100,
        "soft_probability_HotpotQA_to_FanOutQA": 100,
        "FULL": 500,
        "query_features_FanOutQA_to_HotpotQA": 500,
        "soft_probability_FanOutQA_to_HotpotQA": 500,
    }
    assert rows.policy.value_counts().to_dict() == expected
    summary = pd.read_csv(OUT / "rule_summary.csv")
    assert len(summary) == 6
    for filename in [
        "paired_vs_unbudgeted_baseline.csv",
        "hotpot_source_feature_cv.csv",
        "fanout_source_feature_cv.csv",
        "hotpot_source_soft_cv.csv",
        "fanout_source_soft_cv.csv",
        "rule_validation_vs_unbudgeted_baselines.png",
        "rule_validation_vs_unbudgeted_baselines.pdf",
    ]:
        assert (OUT / filename).stat().st_size > 0
    result = {
        "passed": True,
        "target_labels_used_for_selection": False,
        "source_cv": "5-fold GroupKFold by question",
        "per_question_policy_rows": len(rows),
        "policies": expected,
        "artifacts_checked": 7,
    }
    (OUT / "FINAL_AUDIT.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
