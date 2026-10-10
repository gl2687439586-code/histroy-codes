import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
RUN = (
    ROOT
    / "audit_fanoutqa_random100_v31_20260910"
    / "outputs"
    / "fanoutqa_concat100_full310_v2_wiki2023"
    / "random100_v31_upto5_fallback_marked_seed2026"
)
OUT = ROOT / "fanoutqa_random100_v31_budget_table_zh.png"

FONT_REGULAR = Path(r"C:\Windows\Fonts\msyh.ttc")
FONT_BOLD = Path(r"C:\Windows\Fonts\msyhbd.ttc")


def font(size, bold=False):
    return ImageFont.truetype(str(FONT_BOLD if bold else FONT_REGULAR), size)


def text_center(draw, box, text, used_font, fill):
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=used_font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bbox[1]),
        text,
        font=used_font,
        fill=fill,
    )


budget_summary = json.loads(
    (RUN / "analysis" / "budget" / "summary.json").read_text(encoding="utf-8")
)
blank_summary = json.loads(
    (RUN / "analysis" / "blank" / "summary.json").read_text(encoding="utf-8")
)

conditions = {str(x["condition_label"]): x for x in budget_summary["by_condition"]}
rows = []
for label in ["100", "200", "300", "400", "500", "600", "700", "FULL"]:
    item = conditions[label]
    rows.append(
        {
            "label": "FULL" if label == "FULL" else label,
            "evidence": item["visible_evidence_tokens_total"],
            "loose": item["official_loose_accuracy"],
            "strict": item["official_strict_accuracy"],
            "rouge": item["rougeL_f1"],
            "total": item["standalone_llm_input_tokens"]
            + item["standalone_llm_output_tokens"],
            "kind": "full" if label == "FULL" else "budget",
        }
    )

item = blank_summary["overall"]
rows.append(
    {
        "label": "Blank",
        "evidence": item["visible_evidence_tokens_total"],
        "loose": item["official_loose_accuracy"],
        "strict": item["official_strict_accuracy"],
        "rouge": item["rougeL_f1"],
        "total": item["standalone_llm_input_tokens"]
        + item["standalone_llm_output_tokens"],
        "kind": "blank",
    }
)

columns = [
    ("证据预算 / token", 370),
    ("实际可见证据 / token", 470),
    ("Loose Accuracy", 365),
    ("Strict Accuracy", 365),
    ("ROUGE-L F1", 345),
    ("总 LLM 预算 / token", 485),
]

canvas_w, canvas_h = 2600, 1600
image = Image.new("RGB", (canvas_w, canvas_h), "#F5F7FB")
draw = ImageDraw.Draw(image)

navy = "#172554"
blue = "#315A8C"
teal = "#087F8C"
coral = "#D94841"
ink = "#18212F"
muted = "#5C6878"
grid = "#CBD3DF"

draw.rounded_rectangle((55, 50, canvas_w - 55, canvas_h - 50), radius=28, fill="white")
draw.rounded_rectangle((55, 50, canvas_w - 55, 235), radius=28, fill=navy)
draw.rectangle((55, 195, canvas_w - 55, 235), fill=navy)

draw.text((115, 78), "FanOutQA Random-100｜证据预算与回答质量", font=font(56, True), fill="white")
draw.text(
    (115, 158),
    "Qwen3-8B · Wikipedia 2023-09 · 自然搜索最多 4 轮 · 每轮自然返回 0–5 篇 · Seed 2026",
    font=font(27),
    fill="#D9E5F5",
)

x0, y0 = 105, 290
table_w = sum(width for _, width in columns)
header_h, row_h = 92, 92

draw.rounded_rectangle(
    (x0, y0, x0 + table_w, y0 + header_h + row_h * len(rows)),
    radius=15,
    fill="white",
    outline=grid,
    width=2,
)
draw.rounded_rectangle((x0, y0, x0 + table_w, y0 + header_h), radius=15, fill=blue)
draw.rectangle((x0, y0 + 35, x0 + table_w, y0 + header_h), fill=blue)

x = x0
header_font = font(29, True)
for index, (label, width) in enumerate(columns):
    text_center(draw, (x, y0, x + width, y0 + header_h), label, header_font, "white")
    if index:
        draw.line((x, y0, x, y0 + header_h + row_h * len(rows)), fill=grid, width=2)
    x += width

best_loose = max(row["loose"] for row in rows)
best_strict = max(row["strict"] for row in rows)
best_rouge = max(row["rouge"] for row in rows)
best_cost = min(row["total"] for row in rows)

body_font = font(31)
body_bold = font(31, True)
for row_index, row in enumerate(rows):
    top = y0 + header_h + row_index * row_h
    bottom = top + row_h
    if row["kind"] == "full":
        fill = "#E7F5F6"
    elif row["kind"] == "blank":
        fill = "#FFF2E8"
    else:
        fill = "#F8FAFD" if row_index % 2 else "white"
    draw.rectangle((x0, top, x0 + table_w, bottom), fill=fill)
    draw.line((x0, top, x0 + table_w, top), fill=grid, width=2)

    values = [
        row["label"],
        f'{row["evidence"]:,.1f}',
        f'{row["loose"]:.4f}',
        f'{row["strict"]:.4f}',
        f'{row["rouge"]:.4f}',
        f'{row["total"]:,.1f}',
    ]
    highlights = [
        False,
        False,
        row["loose"] == best_loose,
        row["strict"] == best_strict,
        row["rouge"] == best_rouge,
        row["total"] == best_cost,
    ]
    x = x0
    for col_index, ((_, width), value) in enumerate(zip(columns, values)):
        is_label = col_index == 0
        special_label = is_label and row["kind"] in {"full", "blank"}
        used_font = body_bold if highlights[col_index] or special_label else body_font
        color = coral if highlights[col_index] else (teal if special_label else ink)
        text_center(draw, (x, top, x + width, bottom), value, used_font, color)
        x += width

bottom = y0 + header_h + row_h * len(rows)
draw.line((x0, bottom, x0 + table_w, bottom), fill=grid, width=2)

note_y = bottom + 50
draw.rounded_rectangle((105, note_y, 2495, note_y + 210), radius=16, fill="#F2F6FB")
draw.text((140, note_y + 22), "口径说明", font=font(29, True), fill=navy)
notes = [
    "• 100–700 是所有 100 题共有的配对预算网格；FULL 为预算实验中的完整证据条件，Blank 为不做预算处理的空白组。",
    "• “实际可见证据”是一次轨迹内所有搜索轮次实际暴露证据 token 的累计均值。",
    "• “总 LLM 预算”= 该轨迹全部模型调用的输入 tokens + 输出 tokens；红色表示该列最优值（成本列为最低）。",
]
for idx, note in enumerate(notes):
    draw.text((140, note_y + 72 + idx * 42), note, font=font(23), fill=muted)

draw.text(
    (105, canvas_h - 92),
    "主指标：FanOutQA 官方 Loose / Strict Accuracy　｜　完整性：100 题、1,126 条 Budget、100 条 Blank",
    font=font(24, True),
    fill=navy,
)

image.save(OUT, format="PNG", optimize=True, dpi=(300, 300))
print(OUT)
