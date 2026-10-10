import json
import math
import random
import re
import string
import sys
from collections import Counter
from pathlib import Path


poly_root = Path(sys.argv[1])
cos_root = Path(sys.argv[2])


def load_jsonl(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize(value):
    text = str(value or "").lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def acc(prediction, gold):
    normalized_gold = normalize(gold)
    return int(bool(normalized_gold) and normalized_gold in normalize(prediction))


def exact_mcnemar(left_only, right_only):
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) * (0.5 ** discordant)
        for k in range(min(left_only, right_only) + 1)
    )
    return min(1.0, 2.0 * tail)


def percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


poly_scores = {
    str(row["sample_id"]): row
    for row in load_jsonl(poly_root / "audit/polynomial.scored.jsonl")
}
cos_scores = {
    str(row["sample_id"]): row
    for row in load_jsonl(cos_root / "audit/cosine.scored.jsonl")
}
poly_records = {
    str(row["sample_id"]): row
    for row in load_jsonl(poly_root / "polynomial/records.jsonl")
}
cos_records = {
    str(row["sample_id"]): row
    for row in load_jsonl(cos_root / "cosine/records.jsonl")
}
labels = {
    str(row["sample_id"]): str(row["answer"])
    for row in load_jsonl(cos_root / "inputs/gold_labels.sealed.jsonl")
}

ids = sorted(set(poly_scores) & set(cos_scores) & set(labels))
if len(ids) != 500:
    raise RuntimeError(f"Expected 500 paired records, got {len(ids)}")

rows = []
for sample_id in ids:
    p = poly_scores[sample_id]
    c = cos_scores[sample_id]
    gold = labels[sample_id]
    p_acc = acc(poly_records[sample_id].get("prediction"), gold)
    c_acc = acc(cos_records[sample_id].get("prediction"), gold)
    rows.append(
        {
            "sample_id": sample_id,
            "p_em": float(p["em"]),
            "c_em": float(c["em"]),
            "p_f1": float(p["f1"]),
            "c_f1": float(c["f1"]),
            "p_acc": p_acc,
            "c_acc": c_acc,
            "p_completion": int(p["completion_tokens"]),
            "c_completion": int(c["completion_tokens"]),
            "p_total": int(p["server_total_tokens"]),
            "c_total": int(c["server_total_tokens"]),
            "planner_total": int(c["planner_total_tokens"]),
            "p_search": int(p["search_calls"]),
            "c_search": int(c["search_calls"]),
            "p_calls": int(p["model_calls"]),
            "c_calls": int(c["model_calls"]),
            "p_fallback": bool(p["has_fallback"]),
            "c_fallback": bool(c["has_fallback"]),
        }
    )


def mean(field):
    return sum(row[field] for row in rows) / len(rows)


def paired_bootstrap(left_field, right_field, repetitions=30000):
    differences = [row[right_field] - row[left_field] for row in rows]
    generator = random.Random(2026)
    count = len(differences)
    estimates = []
    for _ in range(repetitions):
        estimates.append(
            sum(differences[generator.randrange(count)] for _ in range(count)) / count
        )
    estimates.sort()
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def binary_pair(left_field, right_field):
    counts = Counter((int(row[left_field]), int(row[right_field])) for row in rows)
    left_only = counts[(1, 0)]
    right_only = counts[(0, 1)]
    return {
        "both_wrong": counts[(0, 0)],
        "polynomial_only_correct": left_only,
        "cosine_only_correct": right_only,
        "both_correct": counts[(1, 1)],
        "discordant": left_only + right_only,
        "exact_mcnemar_p": exact_mcnemar(left_only, right_only),
    }


result = {
    "paired_integrity": {
        "records": len(rows),
        "same_ids": set(poly_scores) == set(cos_scores) == set(labels),
        "same_official_budgets": all(
            int(poly_scores[sample_id]["official_budget"])
            == int(cos_scores[sample_id]["official_budget"])
            for sample_id in ids
        ),
        "same_planner_tokens": all(
            int(poly_scores[sample_id]["planner_total_tokens"])
            == int(cos_scores[sample_id]["planner_total_tokens"])
            for sample_id in ids
        ),
    },
    "means": {
        "polynomial": {
            "em": mean("p_em"),
            "f1": mean("p_f1"),
            "acc": mean("p_acc"),
            "completion_tokens": mean("p_completion"),
            "reasoner_total_tokens": mean("p_total"),
            "end_to_end_tokens": mean("p_total") + mean("planner_total"),
            "search_calls": mean("p_search"),
            "model_calls": mean("p_calls"),
            "fallback_rate": mean("p_fallback"),
        },
        "cosine": {
            "em": mean("c_em"),
            "f1": mean("c_f1"),
            "acc": mean("c_acc"),
            "completion_tokens": mean("c_completion"),
            "reasoner_total_tokens": mean("c_total"),
            "end_to_end_tokens": mean("c_total") + mean("planner_total"),
            "search_calls": mean("c_search"),
            "model_calls": mean("c_calls"),
            "fallback_rate": mean("c_fallback"),
        },
    },
    "cosine_minus_polynomial": {
        "em": mean("c_em") - mean("p_em"),
        "f1": mean("c_f1") - mean("p_f1"),
        "acc": mean("c_acc") - mean("p_acc"),
        "completion_tokens": mean("c_completion") - mean("p_completion"),
        "reasoner_total_tokens": mean("c_total") - mean("p_total"),
        "end_to_end_tokens": mean("c_total") - mean("p_total"),
        "search_calls": mean("c_search") - mean("p_search"),
        "model_calls": mean("c_calls") - mean("p_calls"),
        "fallback_rate": mean("c_fallback") - mean("p_fallback"),
    },
    "paired_95_ci_cosine_minus_polynomial": {
        "em": paired_bootstrap("p_em", "c_em"),
        "f1": paired_bootstrap("p_f1", "c_f1"),
        "acc": paired_bootstrap("p_acc", "c_acc"),
        "completion_tokens": paired_bootstrap("p_completion", "c_completion"),
        "reasoner_total_tokens": paired_bootstrap("p_total", "c_total"),
        "search_calls": paired_bootstrap("p_search", "c_search"),
    },
    "paired_binary": {
        "em": binary_pair("p_em", "c_em"),
        "acc": binary_pair("p_acc", "c_acc"),
    },
    "fallback_transition": {
        str(key): value
        for key, value in sorted(
            Counter(
                (row["p_fallback"], row["c_fallback"])
                for row in rows
            ).items()
        )
    },
    "aggregate_totals": {
        "polynomial_reasoner": sum(row["p_total"] for row in rows),
        "cosine_reasoner": sum(row["c_total"] for row in rows),
        "shared_planner": sum(row["planner_total"] for row in rows),
        "polynomial_end_to_end_if_deployed_alone": sum(
            row["p_total"] + row["planner_total"] for row in rows
        ),
        "cosine_end_to_end_if_deployed_alone": sum(
            row["c_total"] + row["planner_total"] for row in rows
        ),
    },
}

print(json.dumps(result, ensure_ascii=False, indent=2))
