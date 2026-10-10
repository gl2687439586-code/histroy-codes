from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


OUT = Path("figures")
OUT.mkdir(exist_ok=True)

font_candidates = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
available = {f.name for f in font_manager.fontManager.ttflist}
font = next((name for name in font_candidates if name in available), "DejaVu Sans")

plt.rcParams.update({
    "font.family": font,
    "axes.unicode_minus": False,
    "font.size": 12,
})

x = [100, 200, 300, 400, 500, 600, 700]
y = [0.3682, 0.3801, 0.3915, 0.3858, 0.3924, 0.3757, 0.4052]
blank = 0.4102

fig, ax = plt.subplots(figsize=(11.4, 6.4), dpi=180)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

blue = "#2F91F5"
orange = "#F08243"
red = "#D95D5D"
gray = "#667085"

ax.plot(
    x,
    y,
    color=blue,
    linewidth=2.7,
    marker="o",
    markersize=7.2,
    markerfacecolor=blue,
    markeredgecolor="white",
    markeredgewidth=1.4,
    zorder=3,
)

ax.scatter([300, 700], [0.3915, 0.4052], s=110, color=orange,
           edgecolor="white", linewidth=1.5, zorder=5)

ax.axhline(blank, color=red, linewidth=1.7, linestyle=(0, (6, 4)), zorder=1)
ax.text(
    690,
    blank + 0.0017,
    "Blank  0.4102  ·  总 token 14,179.0",
    ha="right",
    va="bottom",
    color=red,
    fontweight="semibold",
)

ax.annotate(
    "低成本局部高点  0.3915\n总 token 8,650.7",
    xy=(300, 0.3915),
    xytext=(255, 0.4000),
    ha="center",
    va="bottom",
    color="#222222",
    fontweight="semibold",
    arrowprops=dict(arrowstyle="-", color=orange, lw=1.2),
)

ax.annotate(
    "固定预算最高  0.4052\n总 token 11,562.1",
    xy=(700, 0.4052),
    xytext=(648, 0.4183),
    ha="center",
    va="bottom",
    color="#222222",
    fontweight="semibold",
    arrowprops=dict(arrowstyle="-", color=orange, lw=1.2),
)

ax.set_xlim(75, 725)
ax.set_ylim(0.35, 0.425)
ax.set_xticks(x)
ax.set_yticks([0.36, 0.38, 0.40, 0.42])
ax.set_xlabel("每轮证据预算（tokens）", labelpad=16)
ax.set_ylabel("Loose Accuracy", labelpad=16)

ax.grid(axis="y", color="#E9EDF2", linewidth=0.8)
ax.grid(axis="x", visible=False)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color("#D5DAE0")
    ax.spines[side].set_linewidth(1.0)
ax.tick_params(colors="#344054", length=0, pad=8)

fig.text(
    0.5,
    0.018,
    "注：Oracle = 0.5639（总 token 7,931.4）是逐题事后选择上界，不与固定预算曲线连接。",
    ha="center",
    va="bottom",
    color=gray,
    fontsize=10.5,
)

fig.subplots_adjust(left=0.10, right=0.97, top=0.93, bottom=0.20)

for suffix in ("png", "pdf", "svg"):
    fig.savefig(OUT / f"fanoutqa_loose_vs_read_budget.{suffix}", bbox_inches="tight")

plt.close(fig)
