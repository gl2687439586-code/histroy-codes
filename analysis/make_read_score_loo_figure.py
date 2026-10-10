from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path("outputs/read_score_loo_process_figure")
OUT.mkdir(parents=True, exist_ok=True)

FONT = r"C:\Windows\Fonts\msyh.ttc"
FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont(FONT_BOLD)
FONT_NAME = font_manager.FontProperties(fname=FONT).get_name()

plt.rcParams.update({
    "font.family": FONT_NAME,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

NAVY = "#173B57"
BLUE = "#2F6B91"
TEAL = "#2A9D8F"
TEAL_LIGHT = "#E6F4F1"
BLUE_LIGHT = "#EAF1F7"
ORANGE = "#E8863A"
ORANGE_LIGHT = "#FFF1E6"
GRAY = "#52616B"
LIGHT = "#F5F7F9"
LINE = "#B7C2CA"
WHITE = "#FFFFFF"


def box(ax, xy, w, h, face, edge, title, body=None, title_color=NAVY,
        lw=1.2, radius=0.025, title_size=12.5, body_size=10.5):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        linewidth=lw, edgecolor=edge, facecolor=face,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h * (0.64 if body else 0.50), title,
            ha="center", va="center", fontsize=title_size,
            color=title_color, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + h * 0.31, body,
                ha="center", va="center", fontsize=body_size, color=GRAY,
                linespacing=1.35)
    return patch


def arrow(ax, start, end, color=BLUE, lw=1.8):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
        linewidth=lw, color=color, shrinkA=2, shrinkB=2,
        connectionstyle="arc3,rad=0",
    ))


def step(ax, x, y, number, label):
    ax.text(x, y, str(number), ha="center", va="center", fontsize=10.5,
            color=WHITE, fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.28", facecolor=NAVY,
                      edgecolor=NAVY, linewidth=0))
    ax.text(x + 0.021, y, label, ha="left", va="center", fontsize=10.5,
            color=NAVY, fontweight="bold")


fig = plt.figure(figsize=(15.2, 6.2), facecolor=WHITE)
ax = fig.add_axes([0.025, 0.08, 0.95, 0.84])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

ax.text(0.5, 1.005, "基于 READ 评分的逐题留一预算分配",
        ha="center", va="top", fontsize=18, fontweight="bold", color=NAVY)
ax.text(0.5, 0.955, "用相似问题的历史经验为当前问题选择阅读预算；选择时不查看当前问题的答案结果",
        ha="center", va="top", fontsize=10.5, color=GRAY)

# Step headers
step(ax, 0.035, 0.865, 1, "只看当前问题")
step(ax, 0.245, 0.865, 2, "预测阅读需求")
step(ax, 0.455, 0.865, 3, "参考同分旧题")
step(ax, 0.745, 0.865, 4, "选择预算")
step(ax, 0.895, 0.865, 5, "评估当前题")

# Main flow boxes
box(ax, (0.025, 0.48), 0.16, 0.25, BLUE_LIGHT, BLUE,
    "当前问题  i",
    "原始 Query\n（不看检索结果与答案）")

box(ax, (0.225, 0.48), 0.16, 0.25, ORANGE_LIGHT, ORANGE,
    "READ 评分",
    "GPT-5.6-Sol\n例如：READ = 3")

# Training pool panel
pool = FancyBboxPatch((0.425, 0.30), 0.285, 0.43,
                      boxstyle="round,pad=0.012,rounding_size=0.022",
                      linewidth=1.2, edgecolor=TEAL, facecolor=TEAL_LIGHT)
ax.add_patch(pool)
ax.text(0.5675, 0.685, "其余 99 题中的同分问题",
        ha="center", va="center", fontsize=12.5, color=NAVY,
        fontweight="bold")
ax.text(0.5675, 0.645, "当前问题先被留出，不参与预算统计",
        ha="center", va="center", fontsize=9.7, color=GRAY)

# Mini empirical quality plot
chart_x, chart_y, chart_w, chart_h = 0.455, 0.405, 0.225, 0.18
ax.add_patch(Rectangle((chart_x, chart_y), chart_w, chart_h,
                       facecolor=WHITE, edgecolor=LINE, linewidth=0.8))
budgets = [100, 200, 300, 400, 500, 600, 700]
qualities = [0.38, 0.40, 0.41, 0.36, 0.39, 0.40, 0.45]
qmin, qmax = 0.33, 0.47
xs = [chart_x + 0.018 + j * (chart_w - 0.036) / 6 for j in range(7)]
ys = [chart_y + 0.03 + (q - qmin) / (qmax - qmin) * (chart_h - 0.06)
      for q in qualities]
ax.plot(xs, ys, color=TEAL, linewidth=1.8, zorder=4)
for j, (x, y) in enumerate(zip(xs, ys)):
    color = ORANGE if budgets[j] == 700 else TEAL
    size = 48 if budgets[j] == 700 else 25
    ax.scatter([x], [y], s=size, color=color, edgecolor=WHITE,
               linewidth=0.7, zorder=5)
    ax.text(x, chart_y + 0.012, str(budgets[j]), ha="center", va="bottom",
            fontsize=7.5, color=GRAY)
ax.text(chart_x - 0.008, chart_y + chart_h / 2, "平均\nLoose",
        ha="right", va="center", fontsize=8, color=GRAY)
ax.text(chart_x + chart_w / 2, chart_y - 0.022, "候选预算（tokens / 次搜索）",
        ha="center", va="top", fontsize=8.2, color=GRAY)
ax.text(0.5675, 0.335, "同类题平均表现  +  少量全局平均平滑",
        ha="center", va="center", fontsize=9.2, color=GRAY)

box(ax, (0.75, 0.48), 0.12, 0.25, ORANGE_LIGHT, ORANGE,
    "预算  b*",
    "选择预测质量最高者\n例如：700 tokens")

box(ax, (0.895, 0.48), 0.09, 0.25, BLUE_LIGHT, BLUE,
    "真实评估",
    "答案质量\n总 token 成本",
    title_size=11.7, body_size=10.2)

# Arrows in main flow
arrow(ax, (0.188, 0.605), (0.222, 0.605))
arrow(ax, (0.388, 0.605), (0.422, 0.605))
arrow(ax, (0.713, 0.605), (0.747, 0.605))
arrow(ax, (0.873, 0.605), (0.892, 0.605))

# Guardrail strip
ax.add_patch(FancyBboxPatch((0.025, 0.145), 0.96, 0.09,
                            boxstyle="round,pad=0.01,rounding_size=0.018",
                            linewidth=0.8, edgecolor=LINE, facecolor=LIGHT))
ax.text(0.175, 0.19, "不使用问题 i 自己的轨迹结果",
        ha="center", va="center", fontsize=10.2, color=NAVY, fontweight="bold")
ax.text(0.49, 0.19, "每题轮流留出一次",
        ha="center", va="center", fontsize=10.2, color=NAVY, fontweight="bold")
ax.text(0.79, 0.19, "最终得到 100 个样本外预测",
        ha="center", va="center", fontsize=10.2, color=NAVY, fontweight="bold")
arrow(ax, (0.315, 0.19), (0.395, 0.19), color=LINE, lw=1.4)
arrow(ax, (0.605, 0.19), (0.685, 0.19), color=LINE, lw=1.4)

ax.text(0.5, 0.075,
        "核心思想：用“相似旧题”的经验为“当前新题”分配预算，而不是事后挑选当前题的最佳结果。",
        ha="center", va="center", fontsize=10.7, color=NAVY)

png = OUT / "read-score-loo-budget-allocation-zh.png"
pdf = OUT / "read-score-loo-budget-allocation-zh.pdf"
svg = OUT / "read-score-loo-budget-allocation-zh.svg"
fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=WHITE)
fig.savefig(pdf, bbox_inches="tight", facecolor=WHITE)
fig.savefig(svg, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)

print(png.resolve())
print(pdf.resolve())
print(svg.resolve())
