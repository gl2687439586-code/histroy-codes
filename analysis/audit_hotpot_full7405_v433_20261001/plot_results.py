"""Paper-ready descriptive figures; no raw data or scientific code is modified."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
PDF = HERE.parent / "output" / "pdf" / "hotpot_full7405_v433_audit_20261001"
PDF.mkdir(parents=True, exist_ok=True)
DATA = json.loads((OUT / "plot_data.json").read_text(encoding="utf-8"))
ROWS = {str(x["condition_group"]): x for x in DATA["budget"]}
CONDITIONS = ["100", "200", "300", "400", "500", "600", "FULL"]
NAVY, TEAL, ORANGE, PURPLE, GREY = "#18354A", "#008A8C", "#D76B36", "#7161A7", "#9DA9B2"
COLORS = {"100": GREY, "200": "#52B6AC", "300": "#1DA4A4", "400": "#078B95", "500": "#057482", "600": "#075668", "FULL": ORANGE}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": NAVY,
    "axes.labelcolor": NAVY, "axes.edgecolor": "#CBD3D8", "axes.titleweight": "bold",
    "axes.titlesize": 12, "axes.labelsize": 10, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.color": NAVY, "ytick.color": NAVY,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "svg.fonttype": "none", "axes.unicode_minus": False,
})


def save(fig, stem):
    fig.savefig(OUT / (stem + ".png"), dpi=220)
    fig.savefig(OUT / (stem + ".svg"))
    fig.savefig(PDF / (stem + ".pdf"))
    plt.close(fig)


def clean(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E9EEF1", linewidth=.75)
    ax.tick_params(length=0, pad=7)


def pareto():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.6))
    fig.subplots_adjust(left=.075, right=.972, bottom=.27, top=.79, wspace=.23)
    fig.text(.075, .955, "Evidence exposure: quality-cost frontier", fontsize=18, fontweight="bold")
    fig.text(.075, .893, "HotpotQA dev  |  7,405 paired questions  |  natural search, maximum four rounds", fontsize=10, color="#5C6F7C")
    for ax, axis, title, xlabel in zip(
            axs, ["llm_total_tokens", "visible_evidence_tokens_total"],
            ["A  Total model-token cost", "B  Cumulative exposed evidence"],
            ["Mean LLM input + output tokens / question", "Mean evidence tokens / question"]):
        clean(ax)
        entries = [x for x in DATA["pareto"] if x["cost_axis"] == axis]
        frontier = sorted([x for x in entries if x["on_frontier"]], key=lambda x: x["cost"])
        ax.plot([x["cost"] for x in frontier], [x["f1"]*100 for x in frontier], color=TEAL, lw=2.0, zorder=2)
        for label in CONDITIONS:
            row = ROWS[label]
            x, y = row[axis], row["answer_f1"]*100
            ax.scatter(x, y, s=66 if label != "FULL" else 88, marker="o" if label != "FULL" else "D", color=COLORS[label], edgecolor="white", linewidth=1.0, zorder=4)
            offsets = {"100": (5,-15), "200": (-10,-20), "300": (-14,8), "400": (-15,8), "500": (-17,-22), "600": (-13,9), "FULL": (-16,9)}
            ax.annotate(label, (x,y), xytext=offsets[label], textcoords="offset points", fontsize=9, fontweight="bold", color=COLORS[label])
        ax.set_title(title, loc="left", pad=17)
        ax.set_ylabel("Current token-overlap F1 (%)")
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylim(40, 52.4)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x,p: f"{x:,.0f}"))
        margin = 90 if axis == "llm_total_tokens" else 70
        ax.set_xlim(min(r[axis] for r in ROWS.values())-margin, max(r[axis] for r in ROWS.values())+margin)
    fig.text(.075, .112, "600 vs FULL: nearly equal F1; 4.99% fewer model tokens and 20.01% less exposed evidence.", fontsize=10, fontweight="bold")
    fig.text(.075, .064, "Lines show empirical mean frontiers, not statistically proven dominance. FULL is a complete Top-5 exposure policy.", fontsize=8.5, color="#5C6F7C")
    fig.text(.075, .032, "F1 preserves the existing overlap scorer; official yes/no/noanswer correction is pending. Root cost is included.", fontsize=8.5, color="#5C6F7C")
    save(fig, "pareto_frontier_full7405")


def joint():
    fig, axs = plt.subplots(1, 2, figsize=(11.8, 7.2))
    fig.subplots_adjust(left=.10, right=.94, bottom=.22, top=.79, wspace=.32)
    fig.text(.10, .96, "Read budget and natural search count - jointly", fontsize=17, fontweight="bold")
    fig.text(.10, .911, "Each cell is a post-hoc group; n denotes the number of questions ending at that search count.", fontsize=10, color="#5C6F7C")
    lookup = {(str(x["condition_group"]), int(x["search_calls"])): x for x in DATA["joint"]}
    for ax, metric, title in zip(axs, ["answer_f1", "llm_total_tokens"], ["A  Current overlap F1 (%)", "B  Model tokens / question"]):
        matrix = np.array([[lookup[(b,r)][metric] for r in range(1,5)] for b in CONDITIONS])
        if metric == "answer_f1":
            matrix *= 100
        im = ax.imshow(matrix, cmap="YlGnBu" if metric == "answer_f1" else "PuBu", aspect="auto", vmin=0 if metric == "answer_f1" else None)
        ax.set_xticks(range(4), ["1", "2", "3", "4"])
        ax.set_yticks(range(7), CONDITIONS)
        ax.set_xlabel("Final number of natural searches", labelpad=12)
        ax.set_ylabel("Per-search exposure cap", labelpad=10)
        ax.set_title(title, loc="left", pad=15)
        ax.tick_params(length=0, pad=7)
        for i,b in enumerate(CONDITIONS):
            for j,r in enumerate(range(1,5)):
                value, count = matrix[i,j], lookup[(b,r)]["n"]
                color = "white" if im.norm(value) > .64 else NAVY
                label = f"{value:.1f}%" if metric == "answer_f1" else f"{value:,.0f}"
                ax.text(j, i-.12, label, ha="center", va="center", color=color, fontsize=10, fontweight="bold")
                ax.text(j, i+.17, f"n={count:,}", ha="center", va="center", color=color, fontsize=8)
        ax.set_xticks(np.arange(-.5,4,1), minor=True)
        ax.set_yticks(np.arange(-.5,7,1), minor=True)
        ax.grid(which="minor", color="white", linewidth=2)
        ax.tick_params(which="minor", bottom=False, left=False)
        cbar = fig.colorbar(im, ax=ax, fraction=.037, pad=.04)
        cbar.outline.set_visible(False)
        cbar.ax.tick_params(length=0, labelsize=8)
    fig.text(.10, .088, "Important: these are natural-stopping groups, not randomized search-round interventions.", fontsize=10, fontweight="bold")
    fig.text(.10, .052, "Harder questions can both search more and score worse. Cell differences do not identify the causal effect of another search.", fontsize=8.5, color="#5C6F7C")
    fig.text(.10, .022, "Current F1 awaits official yes/no/noanswer correction; model cost includes root input and output.", fontsize=8.5, color="#5C6F7C")
    save(fig, "budget_search_joint_full7405")


def flips():
    rows = DATA["adjacent"]
    x = np.arange(len(rows))
    gain = np.array([r["gain"] for r in rows])*100
    harm = np.array([r["harm"] for r in rows])*100
    net = np.array([r["net_gain"] for r in rows])*100
    low = np.array([r["net_gain_ci_low"] for r in rows])*100
    high = np.array([r["net_gain_ci_high"] for r in rows])*100
    fig, ax = plt.subplots(figsize=(11.8, 5.6))
    fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=.29)
    fig.text(.08, .955, "More evidence can help - or undo a correct answer", fontsize=18, fontweight="bold")
    fig.text(.08, .895, "Adjacent exposure caps, paired on all 7,405 questions  |  correctness defined by strict EM", fontsize=10, color="#5C6F7C")
    clean(ax)
    ax.axhline(0, color="#AAB7BF", lw=.8)
    ax.bar(x-.25, gain, width=.23, color=TEAL, label="Wrong to Correct (Gain)", zorder=3)
    ax.bar(x, harm, width=.23, color=ORANGE, label="Correct to Wrong (Harm)", zorder=3)
    ax.bar(x+.25, net, width=.23, color=PURPLE, label="NetGain, 95% paired CI", zorder=3)
    ax.errorbar(x+.25, net, yerr=np.vstack([net-low,high-net]), fmt="none", ecolor=NAVY, elinewidth=1.1, capsize=3, zorder=4)
    for i,(g,h,n,r) in enumerate(zip(gain,harm,net,rows)):
        ax.text(i-.25, g+.17, f"{g:.2f}", ha="center", fontsize=8.5, color=TEAL)
        ax.text(i, h+.17, f"{h:.2f}", ha="center", fontsize=8.5, color=ORANGE)
        tag = " *" if r["mcnemar_significant_bh_0_05"] else ""
        ax.text(i+.25, high[i]+.18, f"{n:+.2f}"+tag, ha="center", fontsize=8.5, color=PURPLE, fontweight="bold")
    ax.set_xticks(x, [f"{r['budget_from']} to {r['budget_to']}" for r in rows])
    ax.set_ylim(-1.2, 10.5)
    ax.set_ylabel("Share of all questions (%) / net change (pp)")
    ax.set_xlabel("Per-search evidence exposure cap (tokens)", labelpad=10)
    ax.legend(ncol=3, loc="upper right", frameon=False, fontsize=9)
    fig.text(.08, .125, "25.64% of questions flip correctness at least once; 16.92% have at least one Correct-to-Wrong transition.", fontsize=10, fontweight="bold")
    fig.text(.08, .078, "* Exact McNemar significant after BH-FDR correction at 0.05. NetGain intervals use question-level paired bootstrap.", fontsize=8.5, color="#5C6F7C")
    fig.text(.08, .043, "The stability analysis is limited to the common numeric grid 100-600; FULL and higher, incomplete-coverage caps are excluded.", fontsize=8.5, color="#5C6F7C")
    save(fig, "answer_flip_full7405")


if __name__ == "__main__":
    pareto()
    joint()
    flips()
    print("Created three PNG/SVG figures and three one-page vector PDFs.")
    print(PDF)
