import json
from collections import Counter, defaultdict
from pathlib import Path


RUN = Path(__file__).resolve().parent / "full500_r2_t06_v11"
SOURCE = RUN / "analysis" / "subem_em_gap_for_semantic_audit.jsonl"

# Manual semantic review of every SubEM=1, EM=0 output. The index is the stable
# audit_index emitted by classify_subem_gap.py. All unlisted records are class 1.
CLASS_2 = {
    41, 55, 58, 59, 101, 111, 112, 177, 196, 259, 268, 270, 271,
    283, 296, 297, 340, 341, 357, 358, 360, 363,
}
CLASS_3 = {
    4, 5, 11, 12, 29, 30, 46, 47, 56, 57, 60, 90, 94, 95, 144,
    158, 159, 160, 161, 178, 179, 180, 181, 182, 183, 191, 192,
    333, 334, 342, 344, 347, 348, 364, 365,
}

LABELS = {
    1: "correct_but_not_canonical",
    2: "gold_found_but_not_uniquely_integrated",
    3: "gold_mentioned_but_semantically_wrong",
}


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


rows = read_jsonl(SOURCE)
indices = {row["audit_index"] for row in rows}
assert len(rows) == 372
assert indices == set(range(372))
assert not (CLASS_2 & CLASS_3)
assert CLASS_2 <= indices
assert CLASS_3 <= indices

for row in rows:
    index = row["audit_index"]
    category = 2 if index in CLASS_2 else 3 if index in CLASS_3 else 1
    row["semantic_category"] = category
    row["semantic_category_name"] = LABELS[category]

counts = Counter(row["semantic_category"] for row in rows)
by_type = {
    hotpot_type: Counter(
        row["semantic_category"]
        for row in rows
        if row["hotpot_type"] == hotpot_type
    )
    for hotpot_type in sorted({row["hotpot_type"] for row in rows})
}

by_sample = defaultdict(list)
for row in rows:
    by_sample[row["sample_id"]].append(row["semantic_category"])

question_patterns = Counter(tuple(sorted(categories)) for categories in by_sample.values())
questions_all_same = sum(len(set(categories)) == 1 for categories in by_sample.values())

examples = {
    str(category): [
        {
            "audit_index": row["audit_index"],
            "sample_id": row["sample_id"],
            "rollout_id": row["rollout_id"],
            "gold_answer": row["gold_answer"],
            "question": row["question"],
            "final_answer": row["final_answer"],
        }
        for row in rows
        if row["semantic_category"] == category
    ][:5]
    for category in (1, 2, 3)
}

report = {
    "scope": {
        "all_generation_records": 1000,
        "subem_positive_records": 493,
        "strict_em_positive_records": 121,
        "audited_subem_1_em_0_records": 372,
        "unique_questions_in_gap": len(by_sample),
        "unit": "generation record; two rollouts are separate outputs",
    },
    "manual_semantic_categories": {
        str(category): {
            "name": LABELS[category],
            "records": counts[category],
            "percentage_of_subem_em_gap": counts[category] / len(rows),
            "percentage_of_all_subem_positive": counts[category] / 493,
        }
        for category in (1, 2, 3)
    },
    "strict_em_positive_share_of_all_subem_positive": 121 / 493,
    "by_hotpot_type": {
        hotpot_type: {
            str(category): {
                "records": counter[category],
                "percentage": counter[category] / sum(counter.values()),
            }
            for category in (1, 2, 3)
        }
        for hotpot_type, counter in by_type.items()
    },
    "rollout_pairing": {
        "unique_questions_in_gap": len(by_sample),
        "questions_with_all_gap_outputs_same_category": questions_all_same,
        "question_category_patterns": {
            "+".join(map(str, pattern)): count
            for pattern, count in sorted(question_patterns.items())
        },
    },
    "method_note": (
        "Single-reviewer manual semantic audit of frozen final_answer text. "
        "Class 1 includes semantically correct answers with verbosity, aliases, "
        "or harmless extra explanation. Class 2 includes hedging, multiple candidates, "
        "over-inclusive lists, or insufficiently committed conclusions. Class 3 includes "
        "explicitly wrong/opposite conclusions, negation of the gold, or incidental gold mentions."
    ),
    "examples": examples,
}

classified_path = RUN / "analysis" / "subem_em_gap_semantic_labels.jsonl"
with classified_path.open("w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")

report["classified_records"] = str(classified_path)
report_path = RUN / "analysis" / "SUBEM_GAP_SEMANTIC_AUDIT.json"
report_path.write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(report, ensure_ascii=False, indent=2))
