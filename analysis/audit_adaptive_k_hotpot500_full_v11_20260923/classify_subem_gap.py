import json
import re
import string
import unicodedata
from collections import Counter
from pathlib import Path


RUN = Path(__file__).resolve().parent / "full500_r2_t06_v11"


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = "".join(ch if ch not in string.punctuation else " " for ch in text)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


metrics = read_jsonl(RUN / "analysis" / "per_record_metrics.jsonl")
questions = {
    row["sample_id"]: row
    for row in read_jsonl(RUN / "dataset" / "questions.no_labels.jsonl")
}
labels = {
    row["sample_id"]: row
    for row in read_jsonl(RUN / "dataset" / "gold_labels.sealed.jsonl")
}

gap = [row for row in metrics if row["subem"] and not row["em"]]


def gold_occurrences(answer, gold):
    a = normalize(answer)
    g = normalize(gold)
    return [match.start() for match in re.finditer(r"(?<!\w)" + re.escape(g) + r"(?!\w)", a)]


def features(row):
    answer = row["final_answer"]
    gold = row["gold_answer"]
    answer_n = normalize(answer)
    gold_n = normalize(gold)
    occurrences = gold_occurrences(answer, gold)
    starts_gold = answer_n == gold_n or answer_n.startswith(gold_n + " ")
    answer_tokens = answer_n.split()
    gold_tokens = gold_n.split()
    prefix_distance = occurrences[0] if occurrences else None
    is_yes_no = gold_n in {"yes", "no"}
    uncertainty = bool(re.search(
        r"\b(insufficient|unclear|cannot determine|can't determine|could be|might be|"
        r"possibly|probably|appears to|seems to|not enough|no information|"
        r"not specified|not provided|not mentioned|unknown)\b",
        answer_n,
    ))
    contradiction = bool(re.search(
        r"\b(neither|not|instead|rather than|however|but|although|yet)\b",
        answer_n,
    ))
    alternative_cues = bool(re.search(
        r"\b(either|or|both|followed by|while|whereas|respectively|first|second|"
        r"earlier|later|compared|same|different)\b",
        answer_n,
    ))
    return {
        "starts_with_gold": starts_gold,
        "is_yes_no": is_yes_no,
        "answer_token_count": len(answer_tokens),
        "gold_token_count": len(gold_tokens),
        "gold_occurrence_count": len(occurrences),
        "gold_first_char_offset_normalized": prefix_distance,
        "uncertainty_cue": uncertainty,
        "contradiction_cue": contradiction,
        "alternative_cue": alternative_cues,
    }


records = []
for row in gap:
    item = {
        "sample_id": row["sample_id"],
        "rollout_id": row["rollout_id"],
        "question": questions[row["sample_id"]]["question"],
        "gold_answer": row["gold_answer"],
        "final_answer": row["final_answer"],
        "hotpot_type": labels[row["sample_id"]]["type"],
        "features": features(row),
    }
    records.append(item)

for index, item in enumerate(records):
    item["audit_index"] = index

out = RUN / "analysis" / "subem_em_gap_for_semantic_audit.jsonl"
with out.open("w", encoding="utf-8") as handle:
    for item in records:
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")

review_path = RUN / "analysis" / "subem_em_gap_review.tsv"
with review_path.open("w", encoding="utf-8") as handle:
    handle.write("index\ttype\tstarts_gold\tgold\tquestion\tanswer\n")
    for item in records:
        clean_question = item["question"].replace("\t", " ").replace("\n", " ")
        clean_answer = item["final_answer"].replace("\t", " ").replace("\n", " ")
        handle.write(
            f'{item["audit_index"]}\t{item["hotpot_type"]}\t'
            f'{int(item["features"]["starts_with_gold"])}\t'
            f'{item["gold_answer"]}\t{clean_question}\t{clean_answer}\n'
        )

summary = {
    "denominator_subem_1_em_0": len(records),
    "strict_em_records_excluded": sum(row["em"] for row in metrics),
    "feature_counts": {
        key: sum(bool(item["features"][key]) for item in records)
        for key in [
            "starts_with_gold",
            "is_yes_no",
            "uncertainty_cue",
            "contradiction_cue",
            "alternative_cue",
        ]
    },
    "by_hotpot_type": dict(Counter(item["hotpot_type"] for item in records)),
    "output": str(out),
    "review_tsv": str(review_path),
}

summary_path = RUN / "analysis" / "subem_em_gap_extraction_summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2))
