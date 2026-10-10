from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(r"C:\Users\RESEARCH_USER\Documents\Codex\2026-08-11\referenced-chatgpt-conversation-this-is-an-3")
OUT = ROOT / "outputs"

points = {
    "SC":  (2491.144, 0.378342, 2439.980, 2545.522, 0.299219, 0.459890, True),
    "SRC": (4915.510, 0.506136, 4822.391, 5008.203, 0.419596, 0.593003, True),
    "SSC": (4938.750, 0.570009, 4863.979, 5014.856, 0.481545, 0.656961, True),
    "SSS": (7521.346, 0.565522, 7392.304, 7651.765, 0.478886, 0.651099, False),
    "SRR": (7668.154, 0.521337, 7485.854, 7846.810, 0.433654, 0.607051, False),
    "SSR": (7718.875, 0.621154, 7583.767, 7854.617, 0.534295, 0.706098, True),
    "SRS": (8050.452, 0.588272, 7907.096, 8191.088, 0.499942, 0.677214, False),
}
frontier = ["SC", "SRC", "SSC", "SSR"]
label_offsets = {
    "SC": (85, -0.010), "SRC": (-520, -0.020), "SSC": (-555, 0.017),
    "SSS": (-510, 0.017), "SRR": (-500, -0.024), "SSR": (-490, 0.020), "SRS": (80, -0.022),
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.titlesize": 17,
    "axes.labelsize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
fig, ax = plt.subplots(figsize=(10.4, 6.5), dpi=180)
fig.patch.set_facecolor("white")
ax.set_facecolor("#fbfcfe")

for name, (x, y, xlo, xhi, ylo, yhi, optimal) in points.items():
    color = "#1769aa" if optimal else "#7b8794"
    marker = "o" if optimal else "X"
    ax.errorbar(
        x, y, xerr=[[x-xlo], [xhi-x]], yerr=[[y-ylo], [yhi-y]], fmt=marker,
        ms=9 if optimal else 8, mfc=color, mec="white", mew=1.2,
        ecolor=color, elinewidth=1.1, capsize=3, alpha=0.96, zorder=4 if optimal else 3,
    )
    dx, dy = label_offsets[name]
    ax.text(x+dx, y+dy, f"{name}  {y:.3f}", color="#17212b", weight="semibold", zorder=6)

ax.plot([points[n][0] for n in frontier], [points[n][1] for n in frontier],
        color="#1769aa", lw=2.3, zorder=2)

sss = points["SSS"]
ssc = points["SSC"]
ax.annotate(
    "", xy=(ssc[0]+55, ssc[1]+0.001), xytext=(sss[0]-70, sss[1]-0.002),
    arrowprops=dict(arrowstyle="-|>", color="#d4553f", lw=2.0,
                    connectionstyle="arc3,rad=-0.18"), zorder=5,
)
ax.text(6245, 0.516, "SSC dominates SSS", color="#c23e2d", weight="semibold", ha="center")
ax.text(6245, 0.493, "−2,583 tokens; +0.004 mean F1", color="#4f5d6a", ha="center", fontsize=9.5)

ax.set_xlim(2000, 8500)
ax.set_ylim(0.27, 0.74)
ax.set_xticks([2500, 3500, 4500, 5500, 6500, 7500, 8500])
ax.set_xticklabels([f"{v:,}" for v in [2500, 3500, 4500, 5500, 6500, 7500, 8500]])
ax.set_yticks([0.30, 0.40, 0.50, 0.60, 0.70])
ax.grid(axis="both", color="#d9e0e7", lw=0.8, alpha=0.75)
ax.set_axisbelow(True)
ax.set_xlabel("Total cost (tokens)", labelpad=10)
ax.set_ylabel("Answer quality (F1)", labelpad=10)
ax.set_title("Search–Read–Commit Pareto Frontier", loc="left", pad=24, weight="semibold")
ax.text(0, 1.035, "H = 3 natural roots · n = 104 · mean ± 95% bootstrap CI",
        transform=ax.transAxes, color="#566574", fontsize=10.5)

legend = [
    Line2D([0], [0], color="#1769aa", lw=2.3, marker="o", ms=8,
           markerfacecolor="#1769aa", markeredgecolor="white", label="Pareto-optimal"),
    Line2D([0], [0], color="none", marker="X", ms=8,
           markerfacecolor="#7b8794", markeredgecolor="white", label="Dominated"),
]
ax.legend(handles=legend, loc="lower right", frameon=False, ncol=2, bbox_to_anchor=(1, -0.18))

ax.text(0.0, -0.205,
        "S = Search · R = Evidence Deepening · C = Commit   |   Pareto status is based on aggregate mean quality and cost.",
        transform=ax.transAxes, color="#566574", fontsize=9.2)
fig.subplots_adjust(left=0.11, right=0.97, top=0.86, bottom=0.23)

for suffix in ("pdf", "png", "svg"):
    target = OUT / f"pareto_frontier_h3_paper.{suffix}"
    fig.savefig(target, bbox_inches="tight", facecolor="white", dpi=300)
    print(target)
plt.close(fig)
