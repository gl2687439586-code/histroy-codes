from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
RULES = BASE / "rule_validation"


def main():
    direct = pd.read_csv(RESULTS / "policy_summary.csv")
    quantile = pd.read_csv(RESULTS / "quantile_calibrated_summary.csv")
    harmonized = pd.read_csv(RESULTS / "harmonized_metric_summary.csv")
    new = pd.read_csv(RULES / "rule_summary.csv")

    specs = {
        "FanOutQA": [
            ("Blank", direct, "BLANK"),
            ("Direct", direct, "transfer_hotpot_f1_coarse"),
            ("Quantile", quantile, "transfer_hotpot_f1_quantile_calibrated"),
            ("Soft P(b|s)", new, "soft_probability_HotpotQA_to_FanOutQA"),
            ("Query features", new, "query_features_HotpotQA_to_FanOutQA"),
            ("Oracle", direct, "common_oracle"),
        ],
        "HotpotQA": [
            ("FULL", direct, "FULL"),
            ("Direct", direct, "transfer_fanout_loose_coarse"),
            ("Metric matched", harmonized, "transfer_fanout_answerf1_coarse"),
            ("Soft P(b|s)", new, "soft_probability_FanOutQA_to_HotpotQA"),
            ("Query features", new, "query_features_FanOutQA_to_HotpotQA"),
            ("Oracle", direct, "common_oracle"),
        ],
    }
    config = {
        "FanOutQA": ("official_loose_accuracy", "Blank"),
        "HotpotQA": ("answer_f1", "FULL"),
    }
    colors = {
        "Blank": "#2E6F9E", "FULL": "#2E6F9E", "Direct": "#E68136",
        "Quantile": "#168F83", "Metric matched": "#7B61A8",
        "Soft P(b|s)": "#4FA3A5", "Query features": "#D95F76", "Oracle": "#4C9A4A",
    }
    markers = {"Blank": "s", "FULL": "s", "Oracle": "*", "Quantile": "D", "Metric matched": "P"}

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), constrained_layout=True)
    for ax, dataset in zip(axes, ["FanOutQA", "HotpotQA"]):
        metric, baseline_label = config[dataset]
        records = []
        for label, frame, policy in specs[dataset]:
            row = frame[(frame.target_dataset == dataset) & (frame.policy == policy)].iloc[0]
            records.append((label, row))
        baseline = next(row for label, row in records if label == baseline_label)
        for label, row in records:
            saving = 100 * (1 - row.total_llm_tokens / baseline.total_llm_tokens)
            delta = row[metric] - baseline[metric]
            ax.scatter(
                saving, delta, s=175 if label == "Oracle" else 95,
                color=colors[label], marker=markers.get(label, "o"),
                edgecolor="white", linewidth=.8, zorder=4,
            )
            offset = (6, 7)
            if dataset == "FanOutQA" and label == "Soft P(b|s)": offset = (6, -15)
            if dataset == "HotpotQA" and label == "Direct": offset = (6, -15)
            ax.annotate(label, (saving, delta), xytext=offset, textcoords="offset points", fontsize=8.5, color=colors[label], weight="semibold")
        ax.axhline(0, color="#65717C", linewidth=1.1, linestyle="--")
        ax.axvline(0, color="#B1B8BF", linewidth=.9)
        ax.fill_between([0, 55], 0, ax.get_ylim()[1], color="#EAF7F3", alpha=.35, zorder=0)
        ax.set_title(f"Target: {dataset}", fontsize=13, weight="bold")
        ax.set_xlabel(f"Total-token saving vs {baseline_label} (%)")
        ax.set_ylabel(f"Quality change vs {baseline_label}\n({metric})")
        ax.grid(color="#E6EAEE", linewidth=.8)
    fig.suptitle("Source-only validation of query-prior budget rules", fontsize=16, weight="bold")
    fig.savefig(RULES / "rule_validation_vs_unbudgeted_baselines.png", dpi=240, bbox_inches="tight")
    fig.savefig(RULES / "rule_validation_vs_unbudgeted_baselines.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
