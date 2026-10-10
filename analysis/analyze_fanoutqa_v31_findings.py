import gzip
import json
import math
import random
from pathlib import Path


ROOT = Path("audit_fanoutqa_random100_v31_20260910")
RUN = next(ROOT.rglob("random100_v31_upto5_fallback_marked_seed2026"))


def read_gzip_jsonl(path):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


budget = read_gzip_jsonl(RUN / "analysis" / "budget" / "scored_records.jsonl.gz")
blank = read_gzip_jsonl(RUN / "analysis" / "blank" / "scored_records.jsonl.gz")
budget_by = {(str(x["sample_id"]), str(x["condition_label"])): x for x in budget}
blank_by = {str(x["sample_id"]): x for x in blank}
sample_ids = sorted(blank_by)


def exact_mcnemar_p(wins, losses):
    n = wins + losses
    if n == 0:
        return 1.0
    smaller = min(wins, losses)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2**n)
    return min(1.0, 2 * tail)


def paired(label_a, label_b, key, repetitions=10000):
    def get(sample_id, label):
        return blank_by[sample_id] if label == "BLANK" else budget_by[(sample_id, label)]

    differences = [
        float(get(sample_id, label_a)[key]) - float(get(sample_id, label_b)[key])
        for sample_id in sample_ids
    ]
    rng = random.Random(2026)
    boot = []
    n = len(differences)
    for _ in range(repetitions):
        boot.append(sum(differences[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()
    low = boot[int(0.025 * repetitions)]
    high = boot[int(0.975 * repetitions) - 1]
    wins = sum(x > 0 for x in differences)
    losses = sum(x < 0 for x in differences)
    return {
        "a": label_a,
        "b": label_b,
        "metric": key,
        "a_mean": sum(float(get(s, label_a)[key]) for s in sample_ids) / n,
        "b_mean": sum(float(get(s, label_b)[key]) for s in sample_ids) / n,
        "difference": sum(differences) / n,
        "bootstrap_95_ci": [low, high],
        "wins": wins,
        "ties": n - wins - losses,
        "losses": losses,
        "exact_mcnemar_p": exact_mcnemar_p(wins, losses)
        if key == "official_strict_accuracy"
        else None,
    }


metrics = [
    "official_loose_accuracy",
    "official_strict_accuracy",
    "rougeL_f1",
    "compat_f1",
]
results = []
for pair in [("700", "100"), ("700", "FULL"), ("700", "BLANK")]:
    for metric in metrics:
        results.append(paired(*pair, metric))


def mean_cost(label):
    values = []
    for sample_id in sample_ids:
        row = blank_by[sample_id] if label == "BLANK" else budget_by[(sample_id, label)]
        values.append(
            float(row["standalone_llm_input_tokens"])
            + float(row["standalone_llm_output_tokens"])
        )
    return sum(values) / len(values)


costs = {label: mean_cost(label) for label in ["700", "FULL", "BLANK"]}
output = {
    "paired_results": results,
    "mean_total_llm_tokens": costs,
    "cost_savings_700_vs_full": (costs["FULL"] - costs["700"]) / costs["FULL"],
    "cost_savings_700_vs_blank": (costs["BLANK"] - costs["700"]) / costs["BLANK"],
}
print(json.dumps(output, ensure_ascii=False, indent=2))
