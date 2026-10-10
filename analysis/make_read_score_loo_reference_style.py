from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


OUT = Path("outputs/read_score_loo_process_figure")
OUT.mkdir(parents=True, exist_ok=True)

ZH = r"C:\Windows\Fonts\simsun.ttc"
ZH_BOLD = r"C:\Windows\Fonts\simsunb.ttf"
for path in (ZH, ZH_BOLD):
    font_manager.fontManager.addfont(path)
FONT = font_manager.FontProperties(fname=ZH).get_name()
plt.rcParams.update({
    "font.family": FONT,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PURPLE = "#3E0B74"
PURPLE_FILL = "#E8E5FF"
BLUE = "#005296"
BLUE_FILL = "#D7E9FA"
GRAY = "#656565"
GRAY_FILL = "#E5E5E5"
GRAY_HEAD = "#D1D1D1"
RED = "#B20B20"
RED_FILL = "#FBE2E2"
ORANGE = "#CF651B"
ORANGE_FILL = "#FFE9D7"
TEXT = "#222222"
WHITE = "#FFFFFF"


def module(ax, x, y, w, h, title, lines, head_fill, head_edge, body_edge=GRAY):
    head_h = 0.29 * h
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor=WHITE, edgecolor=body_edge, linewidth=1.35,
    ))
    ax.add_patch(FancyBboxPatch(
        (x, y + h - head_h), w, head_h,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        facecolor=head_fill, edgecolor=head_edge, linewidth=1.35,
    ))
    # Mask the rounded lower corners of the header so it reads as a band.
    ax.add_patch(Rectangle(
        (x + 0.002, y + h - head_h), w - 0.004, 0.014,
        facecolor=head_fill, edgecolor="none",
    ))
    ax.text(x + w / 2, y + h - head_h / 2 + 0.002, title,
            ha="center", va="center", fontsize=12.0, color=TEXT,
            fontweight="bold")
    ax.text(x + 0.012, y + (h - head_h) / 2 - 0.003, lines,
            ha="left", va="center", fontsize=9.0, color=GRAY,
            linespacing=1.28)
    return x + w / 2


def small_box(ax, x, y, w, h, label, face, edge, hatch=None,
              text_color=TEXT, lw=1.2, bold=False):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.002,rounding_size=0.006",
        facecolor=face, edgecolor=edge, linewidth=lw, hatch=hatch,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=8.5, color=text_color,
            fontweight="bold" if bold else "normal")
    return (x + w / 2, y + h / 2)


def arrow(ax, a, b, color=GRAY, lw=1.25, style="-|>"):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=12,
        color=color, linewidth=lw, shrinkA=1, shrinkB=1,
        connectionstyle="arc3,rad=0",
    ))


def leader(ax, a, b, color=GRAY):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=color, linewidth=1.1)


fig = plt.figure(figsize=(15.4, 6.2), facecolor=WHITE)
ax = fig.add_axes([0.02, 0.06, 0.96, 0.90])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

# Title
ax.text(0.60, 0.985, "Query-only READ 评分的逐题留一预算分配",
        ha="center", va="top", fontsize=17, color=TEXT,
        fontweight="bold")

# Top explanation modules, aligned with the episode strip.
centers = []
centers.append(module(ax, 0.225, 0.735, 0.135, 0.205,
                      "问题", "只输入原始 Query\n不查看检索与答案",
                      GRAY_HEAD, GRAY))
centers.append(module(ax, 0.385, 0.735, 0.135, 0.205,
                      "READ 评分", "GPT-5.6-Sol\n例如：READ = 3",
                      PURPLE_FILL, PURPLE))
centers.append(module(ax, 0.545, 0.735, 0.155, 0.205,
                      "同分参考组", "从其余 99 题中\n找 READ = 3 的题",
                      PURPLE_FILL, PURPLE))
centers.append(module(ax, 0.725, 0.735, 0.135, 0.205,
                      "预算选择", "比较 100–700\n预测质量最高者胜出",
                      PURPLE_FILL, PURPLE))
centers.append(module(ax, 0.885, 0.735, 0.105, 0.205,
                      "真实评估", "仅此时读取\n质量与总成本",
                      GRAY_FILL, GRAY))

# Left controller.
cx, cy, cw, ch = 0.02, 0.205, 0.18, 0.525
ax.add_patch(FancyBboxPatch(
    (cx, cy), cw, ch, boxstyle="round,pad=0.004,rounding_size=0.008",
    facecolor=WHITE, edgecolor=RED, linewidth=1.8,
))
header_h = 0.115
ax.add_patch(FancyBboxPatch(
    (cx, cy + ch - header_h), cw, header_h,
    boxstyle="round,pad=0.004,rounding_size=0.008",
    facecolor=RED_FILL, edgecolor=RED, linewidth=1.8,
))
ax.add_patch(Rectangle((cx + 0.002, cy + ch - header_h), cw - 0.004, 0.012,
                       facecolor=RED_FILL, edgecolor="none"))
ax.text(cx + cw / 2, cy + ch - header_h / 2, "Leave-One-Out",
        ha="center", va="center", fontsize=13.2, color=TEXT,
        fontweight="bold")
controller_text = (
    "1）留出当前问题\n"
    "     不用它自己的结果\n\n"
    "2）参考同分旧题\n"
    "     估计各预算表现\n\n"
    "3）选定预算后\n"
    "     才评估当前问题"
)
ax.text(cx + 0.017, cy + 0.205, controller_text,
        ha="left", va="center", fontsize=10.2, color=GRAY,
        linespacing=1.35)

# Episode 1 dashed container.
e1x, e1y, e1w, e1h = 0.225, 0.455, 0.765, 0.205
ax.add_patch(FancyBboxPatch(
    (e1x, e1y), e1w, e1h,
    boxstyle="round,pad=0.005,rounding_size=0.010",
    facecolor="none", edgecolor=PURPLE, linewidth=1.5,
    linestyle=(0, (8, 5)),
))

y, h = 0.515, 0.082
x = 0.238
p_q = small_box(ax, x, y, 0.043, h, "Q1", WHITE, GRAY, bold=True); x += 0.058
p_r = small_box(ax, x, y, 0.050, h, "r=3", PURPLE_FILL, PURPLE, bold=True); x += 0.065
for lab in ("旧题", "旧题", "…", "旧题"):
    face = PURPLE_FILL if lab != "…" else WHITE
    hatch = "///" if lab == "…" else None
    small_box(ax, x, y, 0.043, h, lab, face, PURPLE if lab != "…" else GRAY,
              hatch=hatch)
    x += 0.051
ax.text(x + 0.002, y + h / 2, "→", ha="center", va="center",
        fontsize=13, color=GRAY); x += 0.018

budget_centers = []
for b in (100, 200, 300, 400, 500, 600, 700):
    chosen = b == 700
    budget_centers.append(small_box(
        ax, x, y, 0.038, h, str(b),
        ORANGE_FILL if chosen else PURPLE_FILL,
        ORANGE if chosen else PURPLE,
        bold=chosen,
    ))
    x += 0.044
ax.text(x + 0.001, y + h / 2, "→", ha="center", va="center",
        fontsize=13, color=GRAY); x += 0.016
p_eval = small_box(ax, x, y, 0.052, h, "Q, C", GRAY_FILL, GRAY, bold=True)

# Leaders from modules to relevant episode tokens.
targets = [p_q, p_r, (0.405, y + h / 2), budget_centers[4], p_eval]
for center, target in zip(centers, targets):
    leader(ax, (center, 0.735), (target[0], e1y + e1h),
           PURPLE if center in centers[1:4] else GRAY)

# Episode label and span.
arrow(ax, (e1x, 0.425), (e1x + e1w, 0.425), color=PURPLE,
      lw=1.1, style="<->")
ax.text(e1x + e1w / 2, 0.425, "  逐题留一轮次 1：留出 Q1  ",
        ha="center", va="center", fontsize=10.5, color=TEXT,
        fontstyle="italic", backgroundcolor=WHITE)

# Episode i: compact second pass to show repetition.
e2x, e2y, e2w, e2h = 0.225, 0.115, 0.725, 0.205
ax.add_patch(FancyBboxPatch(
    (e2x, e2y), e2w, e2h,
    boxstyle="round,pad=0.005,rounding_size=0.010",
    facecolor="none", edgecolor=BLUE, linewidth=1.5,
    linestyle=(0, (8, 5)),
))

y2 = 0.175
x = 0.238
small_box(ax, x, y2, 0.043, h, "Qi", WHITE, GRAY, bold=True); x += 0.058
small_box(ax, x, y2, 0.050, h, "r=4", BLUE_FILL, BLUE, bold=True); x += 0.065
for lab in ("旧题", "旧题", "…", "旧题"):
    face = BLUE_FILL if lab != "…" else WHITE
    hatch = "///" if lab == "…" else None
    small_box(ax, x, y2, 0.043, h, lab, face, BLUE if lab != "…" else GRAY,
              hatch=hatch)
    x += 0.051
ax.text(x + 0.002, y2 + h / 2, "→", ha="center", va="center",
        fontsize=13, color=GRAY); x += 0.018
for b in (100, 200, 300, 400, 500, 600, 700):
    chosen = b == 400
    small_box(ax, x, y2, 0.038, h, str(b),
              ORANGE_FILL if chosen else BLUE_FILL,
              ORANGE if chosen else BLUE, bold=chosen)
    x += 0.044
ax.text(x + 0.001, y2 + h / 2, "→", ha="center", va="center",
        fontsize=13, color=GRAY); x += 0.016
small_box(ax, x, y2, 0.052, h, "Q, C", GRAY_FILL, GRAY, bold=True)

arrow(ax, (e2x, 0.085), (e2x + e2w, 0.085), color=BLUE,
      lw=1.1, style="<->")
ax.text(e2x + e2w / 2, 0.085, "  逐题留一轮次 i：每道题轮流留出一次  ",
        ha="center", va="center", fontsize=10.5, color=TEXT,
        fontstyle="italic", backgroundcolor=WHITE)

# Controller arrows and loop, echoing the reference design.
arrow(ax, (cx + cw, 0.56), (e1x, 0.56), color=GRAY, lw=1.4)
arrow(ax, (cx + cw, 0.215), (e2x, 0.215), color=GRAY, lw=1.4)
arrow(ax, (e2x + e2w, 0.215), (0.982, 0.215), color=GRAY, lw=1.4)
ax.text(0.973, 0.155, "重复", ha="center", va="center",
        fontsize=9.0, color=GRAY)
ax.plot([0.982, 0.982, 0.11, 0.11], [0.185, 0.035, 0.035, cy],
        color=GRAY, linewidth=1.25)
arrow(ax, (0.11, 0.035), (0.11, cy), color=GRAY, lw=1.25)

# Legend.
legend_y = 0.69
ax.add_patch(Rectangle((0.395, legend_y), 0.016, 0.024,
                       facecolor=PURPLE_FILL, edgecolor=PURPLE, linewidth=1))
ax.text(0.416, legend_y + 0.012, "同分旧题 / 候选预算", va="center",
        fontsize=8.7, color=GRAY)
ax.add_patch(Rectangle((0.535, legend_y), 0.016, 0.024,
                       facecolor=ORANGE_FILL, edgecolor=ORANGE, linewidth=1))
ax.text(0.556, legend_y + 0.012, "最终选择", va="center",
        fontsize=8.7, color=GRAY)
ax.add_patch(Rectangle((0.625, legend_y), 0.016, 0.024,
                       facecolor=GRAY_FILL, edgecolor=GRAY, linewidth=1))
ax.text(0.646, legend_y + 0.012, "仅用于评估", va="center",
        fontsize=8.7, color=GRAY)

png = OUT / "read-score-loo-budget-allocation-reference-style-zh.png"
pdf = OUT / "read-score-loo-budget-allocation-reference-style-zh.pdf"
svg = OUT / "read-score-loo-budget-allocation-reference-style-zh.svg"
fig.savefig(png, dpi=320, bbox_inches="tight", facecolor=WHITE)
fig.savefig(pdf, bbox_inches="tight", facecolor=WHITE)
fig.savefig(svg, bbox_inches="tight", facecolor=WHITE)
plt.close(fig)

print(png.resolve())
print(pdf.resolve())
print(svg.resolve())
