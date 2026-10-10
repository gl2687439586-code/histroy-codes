import hashlib
import json
import math
import random
import re
import statistics
import string
import sys
from collections import Counter
from pathlib import Path


root = Path(sys.argv[1])


def load(path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def normalize(value):
    text = str(value or "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def contains(text, gold):
    gold = normalize(gold)
    return bool(gold) and gold in normalize(text)


hash_mismatches = []
hash_count = 0
for line in (root / "FILES.sha256").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    expected, relative = line.split(None, 1)
    relative = relative.lstrip("*")
    hash_count += 1
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        hash_mismatches.append(relative)

records = load(root / "cosine/records.jsonl")
scored_rows = load(root / "audit/cosine.scored.jsonl")
scores = {str(row["sample_id"]): row for row in scored_rows}
labels = {
    str(row["sample_id"]): str(row["answer"])
    for row in load(root / "inputs/gold_labels.sealed.jsonl")
}

flow = Counter()
answer_statuses = Counter()
finish_types = Counter()
document_counts = Counter()
length_ids = set()
repeated_query_ids = set()
duplicate_query_rounds = 0
rows = []

for record in records:
    sample_id = str(record["sample_id"])
    gold = labels[sample_id]
    score = scores[sample_id]
    evidence_hits = []
    subanswer_hits = []
    queries = []

    for step in record.get("steps", []):
        evidence = "\n".join(
            str(document.get("contents") or "")
            for search_round in step.get("search_rounds", [])
            for document in search_round.get("documents", [])
        )
        subanswer = (
            str(step.get("subanswer") or "")
            + "\n"
            + str(step.get("final_answer") or "")
        )
        evidence_hits.append(contains(evidence, gold))
        subanswer_hits.append(contains(subanswer, gold))
        answer_statuses[str(step.get("answer_status"))] += 1

        for search_round in step.get("search_rounds", []):
            document_counts[len(search_round.get("documents", []))] += 1
            query = re.sub(
                r"[^a-z0-9 ]+", " ", str(search_round.get("query") or "").lower()
            )
            queries.append(" ".join(query.split()))

        for turn in step.get("model_turns", []):
            finish = turn.get("finish_reason")
            if isinstance(finish, dict):
                finish = finish.get("type")
            finish_types[str(finish)] += 1
            if finish == "length":
                length_ids.add(sample_id)

    if len(queries) != len(set(queries)):
        repeated_query_ids.add(sample_id)
        duplicate_query_rounds += len(queries) - len(set(queries))

    answer_acc = float(contains(record.get("prediction"), gold))
    evidence_hit = any(evidence_hits)
    subanswer_hit = any(subanswer_hits)
    flow[(evidence_hit, subanswer_hit, bool(answer_acc))] += 1
    rows.append(
        {
            "sample_id": sample_id,
            "em": float(score["em"]),
            "f1": float(score["f1"]),
            "acc": answer_acc,
            "budget": int(score["official_budget"]),
            "completion": int(score["completion_tokens"]),
            "reasoner_total": int(score["server_total_tokens"]),
            "planner_total": int(score["planner_total_tokens"]),
            "search": int(score["search_calls"]),
            "model_calls": int(score["model_calls"]),
            "fallback": bool(score["has_fallback"]),
            "evidence_hit": evidence_hit,
            "subanswer_hit": subanswer_hit,
        }
    )


def aggregate(items):
    if not items:
        return {"n": 0}
    return {
        "n": len(items),
        "em": sum(row["em"] for row in items) / len(items),
        "f1": sum(row["f1"] for row in items) / len(items),
        "acc": sum(row["acc"] for row in items) / len(items),
        "mean_completion": sum(row["completion"] for row in items) / len(items),
        "mean_reasoner_total": sum(row["reasoner_total"] for row in items) / len(items),
        "mean_search": sum(row["search"] for row in items) / len(items),
    }


ratios = sorted(row["completion"] / row["budget"] for row in rows)


def quantile(values, probability):
    position = (len(values) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    return values[lower] * (upper - position) + values[upper] * (position - lower)


random_generator = random.Random(2026)


def bootstrap_ci(field, repetitions=20000):
    values = [row[field] for row in rows]
    count = len(values)
    estimates = []
    for _ in range(repetitions):
        estimates.append(
            sum(values[random_generator.randrange(count)] for _ in range(count)) / count
        )
    estimates.sort()
    return [estimates[int(0.025 * repetitions)], estimates[int(0.975 * repetitions)]]


result = {
    "hashes": {"files": hash_count, "mismatches": hash_mismatches},
    "structure": {
        "records": len(records),
        "unique_ids": len({str(row["sample_id"]) for row in records}),
        "schedulers": dict(Counter(row.get("scheduler") for row in records)),
        "rollouts": dict(Counter(str(row.get("rollout")) for row in records)),
        "model_paths": dict(
            Counter(
                row.get("reasoner_model_info", {}).get("model_path") for row in records
            )
        ),
        "gold_loaded": dict(
            Counter(str(row.get("gold_labels_loaded_during_generation")) for row in records)
        ),
    },
    "overall": aggregate(rows),
    "bootstrap_95": {
        "em": bootstrap_ci("em"),
        "f1": bootstrap_ci("f1"),
        "acc": bootstrap_ci("acc"),
    },
    "flow": {str(key): value for key, value in sorted(flow.items())},
    "runtime": {
        "answer_statuses": dict(answer_statuses),
        "finish_types": dict(finish_types),
        "search_document_counts": dict(document_counts),
        "length_questions": len(length_ids),
        "repeated_query_questions": len(repeated_query_ids),
        "duplicate_query_rounds": duplicate_query_rounds,
    },
    "fallback": {
        "no": aggregate([row for row in rows if not row["fallback"]]),
        "yes": aggregate([row for row in rows if row["fallback"]]),
    },
    "evidence": {
        "hit": aggregate([row for row in rows if row["evidence_hit"]]),
        "miss": aggregate([row for row in rows if not row["evidence_hit"]]),
    },
    "by_budget": {
        str(budget): aggregate([row for row in rows if row["budget"] == budget])
        for budget in sorted({row["budget"] for row in rows})
    },
    "actual_to_soft_budget_ratio": {
        "mean": statistics.fmean(ratios),
        "median": statistics.median(ratios),
        "p10": quantile(ratios, 0.1),
        "p90": quantile(ratios, 0.9),
        "min": min(ratios),
        "max": max(ratios),
    },
    "totals": {
        "official_budget": sum(row["budget"] for row in rows),
        "completion_tokens": sum(row["completion"] for row in rows),
        "reasoner_total_tokens": sum(row["reasoner_total"] for row in rows),
        "planner_total_tokens": sum(row["planner_total"] for row in rows),
        "end_to_end_model_tokens": sum(
            row["reasoner_total"] + row["planner_total"] for row in rows
        ),
        "search_calls": sum(row["search"] for row in rows),
        "model_calls": sum(row["model_calls"] for row in rows),
    },
}

print(json.dumps(result, ensure_ascii=False, indent=2))
