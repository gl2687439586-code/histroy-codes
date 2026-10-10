import json
from collections import defaultdict
from pathlib import Path


BASE = Path(__file__).resolve().parent / "full500_r2_t06_v11"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


metrics = read_jsonl(BASE / "analysis" / "per_record_metrics.jsonl")
labels = {
    row["sample_id"]: row
    for row in read_jsonl(BASE / "dataset" / "gold_labels.sealed.jsonl")
}

groups = defaultdict(list)
for row in metrics:
    groups[labels[row["sample_id"]]["type"]].append(row)


def summarize(rows):
    n = len(rows)
    completion = sorted(row["completion_tokens"] for row in rows)
    return {
        "records": n,
        "em": sum(row["em"] for row in rows) / n,
        "f1": sum(row["f1"] for row in rows) / n,
        "subem": sum(row["subem"] for row in rows) / n,
        "subem_but_not_em": sum(
            bool(row["subem"]) and not bool(row["em"]) for row in rows
        ),
        "mean_completion_tokens": sum(completion) / n,
    }


result = {
    "overall": summarize(metrics),
    "by_type": {name: summarize(rows) for name, rows in sorted(groups.items())},
    "yes_no_gold_questions": sum(
        label["answer"].strip().lower() in {"yes", "no"}
        for label in labels.values()
    ),
    "comparison_subem_not_em_examples": [
        {
            "sample_id": row["sample_id"],
            "gold_answer": row["gold_answer"],
            "final_answer": row["final_answer"][:500],
        }
        for row in groups.get("comparison", [])
        if row["subem"] and not row["em"]
    ][:5],
}

rendered = json.dumps(result, ensure_ascii=False, indent=2)
(BASE / "analysis" / "EXTRA_DIAGNOSTICS.json").write_text(
    rendered + "\n", encoding="utf-8"
)
print(rendered)
