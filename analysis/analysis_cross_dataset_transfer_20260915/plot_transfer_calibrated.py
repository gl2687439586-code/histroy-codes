from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BASE = Path(__file__).resolve().parent
OUT = BASE / "results"


def point(ax, row, metric, label, color, marker="o", size=100, dx=7, dy=6):
    ax.scatter(
        row.total_llm_tokens,
        row[metric],
        s=size,
        color=color,
        marker=marker,
        edgecolor="white",
        linewidth=0.9,
        zorder=5,
    )
    ax.annotate(
        label,
        (row.total_llm_tokens, row[metric]),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.3,
        color=color,
        weight="semibold",
    )


def get(summary, dataset, policy):
    rows = summary[(summary.target_dataset == dataset) & (summary.policy == policy)]
    if len(rows) != 1:
        raise RuntimeError((dataset, policy, len(rows)))
    return rows.iloc[0]


def main():
    direct = pd.read_csv(OUT / "policy_summary.csv")
    calibrated = pd.read_csv(OUT / "quantile_calibrated_summary.csv")
    harmonized = pd.read_csv(OUT / "harmonized_metric_summary.csv")

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig = plt.figure(figsize=(15.5, 10.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.15, .85])
    ax_f = fig.add_subplot(grid[0, 0])
    ax_h = fig.add_subplot(grid[0, 1])
    ax_d = fig.add_subplot(grid[1, 0])
    ax_m = fig.add_subplot(grid[1, 1])

    gray = "#8B95A1"
    blue = "#2E6F9E"
    orange = "#E68136"
    teal = "#168F83"
    green = "#4C9A4A"
    purple = "#7B61A8"

    for ax, dataset, metric, baseline, fixed_best, direct_policy, cal_policy, ylabel in [
        (
            ax_f, "FanOutQA", "official_loose_accuracy", "BLANK", "fixed_500",
            "transfer_hotpot_em_coarse", "transfer_hotpot_f1_quantile_calibrated",
            "Official Loose Accuracy",
        ),
        (
            ax_h, "HotpotQA", "answer_f1", "FULL", "fixed_600",
            "transfer_fanout_loose_coarse", "transfer_fanout_loose_quantile_calibrated",
            "Answer F1",
        ),
    ]:
        fixed = direct[
            (direct.target_dataset == dataset)
            & direct.policy.str.match(r"fixed_(100|200|300|400|500|600)$")
        ].sort_values("mean_selected_budget")
        ax.plot(
            fixed.total_llm_tokens,
            fixed[metric],
            "o-",
            color=gray,
            linewidth=1.5,
            markersize=5,
            alpha=.9,
            label="Fixed budgets (100–600)",
        )
        for _, row in fixed.iterrows():
            ax.annotate(
                str(int(row.mean_selected_budget)),
                (row.total_llm_tokens, row[metric]),
                xytext=(0, -13),
                textcoords="offset points",
                ha="center",
                fontsize=7.2,
                color="#66707A",
            )

        base = get(direct, dataset, baseline)
        fixed_row = get(direct, dataset, fixed_best)
        direct_row = get(direct, dataset, direct_policy)
        cal = get(calibrated, dataset, cal_policy)
        oracle = get(direct, dataset, "common_oracle")
        point(ax, base, metric, baseline, blue, "s", 110)
        direct_dy = -18 if dataset == "HotpotQA" else 6
        point(ax, direct_row, metric, "Direct score transfer", orange, "^", 115, 7, direct_dy)
        point(ax, cal, metric, "Quantile-calibrated transfer", teal, "D", 115)
        if dataset == "HotpotQA":
            matched = get(harmonized, dataset, "transfer_fanout_answerf1_coarse")
            point(ax, matched, metric, "Metric-matched transfer", purple, "P", 125, 7, 12)
        point(ax, oracle, metric, "Common-grid oracle", green, "*", 180)

        ax.annotate(
            "",
            xy=(cal.total_llm_tokens, cal[metric]),
            xytext=(base.total_llm_tokens, base[metric]),
            arrowprops=dict(arrowstyle="->", color=teal, linewidth=1.7, alpha=.75),
        )
        ax.set_title(f"Target: {dataset}", fontsize=13.5)
        ax.set_xlabel("Mean total LLM tokens per question")
        ax.set_ylabel(ylabel)
        ax.grid(axis="both", color="#E7EBEF", linewidth=.8)

    # Query-only READ-score distribution shift.
    score_values = np.arange(1, 5)
    hotpot_pct = np.array([62, 286, 150, 2]) / 500 * 100
    fanout_pct = np.array([0, 6, 56, 38]) / 100 * 100
    width = .34
    ax_d.bar(score_values - width / 2, hotpot_pct, width, color=blue, label="HotpotQA (n=500)")
    ax_d.bar(score_values + width / 2, fanout_pct, width, color=orange, label="FanOutQA (n=100)")
    for x, y in zip(score_values - width / 2, hotpot_pct):
        ax_d.text(x, y + 1.2, f"{y:.1f}%", ha="center", fontsize=7.5)
    for x, y in zip(score_values + width / 2, fanout_pct):
        ax_d.text(x, y + 1.2, f"{y:.1f}%", ha="center", fontsize=7.5)
    ax_d.set_title("READ-score distribution shift", fontsize=13.5)
    ax_d.set_xlabel("Query-only READ score")
    ax_d.set_ylabel("Questions (%)")
    ax_d.set_xticks(score_values)
    ax_d.set_ylim(0, 66)
    ax_d.legend(frameon=False, ncol=2, fontsize=8)
    ax_d.grid(axis="y", color="#E7EBEF", linewidth=.8)

    # Compact mapping diagram.
    ax_m.axis("off")
    ax_m.set_title("Transferred policies (common budget grid)", fontsize=13.5, pad=12)
    cells = [
        ["Direction", "Rule", "Target score → budget"],
        ["Hotpot → FanOut", "Direct", "2→300, 3–4→500"],
        ["Hotpot → FanOut", "Quantile calibrated", "2→500, 3→300, 4→500"],
        ["FanOut → Hotpot", "Direct", "1→500*, 2→400, 3–4→300"],
        ["FanOut → Hotpot", "Metric matched", "1→500*, 2→400, 3–4→500"],
        ["FanOut → Hotpot", "Quantile calibrated", "1→400, 2→100, 3–4→400"],
    ]
    table = ax_m.table(
        cellText=cells[1:],
        colLabels=cells[0],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[.29, .31, .40],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.3)
    table.scale(1, 1.65)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D7DDE3")
        if r == 0:
            cell.set_facecolor("#EAF1F7")
            cell.set_text_props(weight="bold", color="#173A52")
        elif r in (2, 4):
            cell.set_facecolor("#EAF7F5")
        else:
            cell.set_facecolor("#FAFBFC")
    ax_m.text(
        .0,
        .04,
        "* Score 1 is absent in FanOutQA; direct transfer uses the source global fallback budget.",
        transform=ax_m.transAxes,
        fontsize=7.7,
        color="#5D6770",
    )

    fig.suptitle(
        "Bidirectional transfer of query-only READ-score budget policies",
        fontsize=17,
        weight="bold",
    )
    fig.text(
        .5,
        .966,
        "Point estimates; target answers are never used by transferred policies. Oracle is hindsight-only.",
        ha="center",
        fontsize=9.2,
        color="#59636D",
    )
    fig.savefig(OUT / "cross_dataset_transfer_with_calibration.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT / "cross_dataset_transfer_with_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
