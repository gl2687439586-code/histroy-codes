#!/usr/bin/env python3
"""Audit whether GPT-5.6 HotpotQA errors exhibit observable uncertainty."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter
from pathlib import Path


# Conservative manual audit after comparing question, gold answer, final answer,
# raw answer generation, and API reasoning summary.
FACTUAL_ERRORS = {"test_34", "test_38", "test_42", "test_58", "test_63", "test_80"}
ANSWER_SELECTION_ERRORS = {"test_94"}
OPERATIONAL_FAILURES = {"test_20"}

PATTERNS = {
    "any_uncertainty": re.compile(
        r"\b(?:not sure|unsure|uncertain|unclear|ambig(?:uity|uous)|assuming|"
        r"likely|probably|perhaps|maybe|appears?|seems?|might|could refer|"
        r"may refer|i wonder|i think|i believe|double[- ]check|verify|"
        r"need to check|need to clarify)\b",
        re.I,
    ),
    "ambiguity": re.compile(
        r"\b(?:ambig(?:uity|uous)|assuming|could refer|may refer|either|"
        r"on the other hand|depends on|need to clarify)\b",
        re.I,
    ),
    "epistemic_hedge": re.compile(
        r"\b(?:likely|probably|perhaps|maybe|appears?|seems?|might|"
        r"i wonder|i think|i believe|not sure|unsure|uncertain|unclear)\b",
        re.I,
    ),
    "verification_desire": re.compile(
        r"\b(?:double[- ]check|verify|need to check|check(?:ing)? (?:the|some) "
        r"sources?|confirm(?:ing)?|ensure accuracy)\b",
        re.I,
    ),
    "approximation": re.compile(r"\b(?:approximately|approx\.?|about|roughly)\b", re.I),
}


def mean(xs):
    return statistics.fmean(xs) if xs else float("nan")


def median(xs):
    return statistics.median(xs) if xs else float("nan")


def percentile(xs, p):
    if not xs:
        return float("nan")
    ys = sorted(xs)
    pos = (len(ys) - 1) * p
    lo, hi = math.floor(pos), math.ceil(pos)
    return ys[lo] if lo == hi else ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


def bootstrap_mean_diff(a, b, seed=2026, n=20000):
    """Return mean(a)-mean(b), percentile CI, and two-sided sign probability."""
    rng = random.Random(seed)
    observed = mean(a) - mean(b)
    draws = []
    for _ in range(n):
        aa = [a[rng.randrange(len(a))] for _ in a]
        bb = [b[rng.randrange(len(b))] for _ in b]
        draws.append(mean(aa) - mean(bb))
    ci = [percentile(draws, 0.025), percentile(draws, 0.975)]
    p = 2 * min(sum(x <= 0 for x in draws) / n, sum(x >= 0 for x in draws) / n)
    return observed, ci, min(1.0, p)


def fisher_two_sided(a, b, c, d):
    """Two-sided Fisher exact p for [[a,b],[c,d]], stdlib only."""
    n1, n2, k = a + b, c + d, a + c
    total = n1 + n2

    def prob(x):
        return math.comb(k, x) * math.comb(total - k, n1 - x) / math.comb(total, n1)

    obs = prob(a)
    lo, hi = max(0, n1 - (total - k)), min(n1, k)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= obs + 1e-15))


def label(sample_id):
    if sample_id in FACTUAL_ERRORS:
        return "factual_error"
    if sample_id in ANSWER_SELECTION_ERRORS:
        return "answer_selection_error"
    if sample_id in OPERATIONAL_FAILURES:
        return "operational_failure"
    return "semantically_correct_or_acceptable"


def get_event(row):
    return row.get("events", [{}])[0]


def analyze(input_path: Path, output_dir: Path):
    rows = [json.loads(line) for line in input_path.open(encoding="utf-8") if line.strip()]
    audits = []
    for row in rows:
        event = get_event(row)
        raw = event.get("raw_model_response") or row.get("raw_answer_generation") or ""
        summary = event.get("api_reasoning") or ""
        text = f"{raw}\n{summary}"
        flags = {name: bool(rx.search(text)) for name, rx in PATTERNS.items()}
        audits.append(
            {
                "sample_id": row["sample_id"],
                "label": label(row["sample_id"]),
                "question": row.get("question", ""),
                "gold": " | ".join(map(str, row.get("gold_answers", []))),
                "answer": row.get("final_answer", ""),
                "em": float(row.get("em", row.get("metrics", {}).get("em", 0))),
                "f1": float(row.get("f1", row.get("metrics", {}).get("f1", 0))),
                "acc": float(row.get("acc", row.get("metrics", {}).get("acc", 0))),
                "reasoning_tokens": int(row.get("reasoning_tokens") or 0),
                "output_tokens": int(row.get("total_llm_output_tokens") or 0),
                "total_tokens": int(row.get("total_cost_tokens") or 0),
                "latency_seconds": float(row.get("latency_seconds") or 0),
                **{f"flag_{k}": int(v) for k, v in flags.items()},
                "raw_response": raw,
                "reasoning_summary": summary,
            }
        )

    groups = {}
    for group_name, selected in {
        "semantic_correct": [x for x in audits if x["label"] == "semantically_correct_or_acceptable"],
        "task_error": [x for x in audits if x["label"] in {"factual_error", "answer_selection_error"}],
        "factual_error_only": [x for x in audits if x["label"] == "factual_error"],
        "operational_failure": [x for x in audits if x["label"] == "operational_failure"],
    }.items():
        groups[group_name] = {
            "n": len(selected),
            "mean_em": mean([x["em"] for x in selected]),
            "mean_f1": mean([x["f1"] for x in selected]),
            "mean_reasoning_tokens": mean([x["reasoning_tokens"] for x in selected]),
            "median_reasoning_tokens": median([x["reasoning_tokens"] for x in selected]),
            "mean_output_tokens": mean([x["output_tokens"] for x in selected]),
            "mean_latency_seconds": mean([x["latency_seconds"] for x in selected]),
            **{
                f"{name}_rate": mean([x[f"flag_{name}"] for x in selected])
                for name in PATTERNS
            },
        }

    correct = [x for x in audits if x["label"] == "semantically_correct_or_acceptable"]
    wrong = [x for x in audits if x["label"] in {"factual_error", "answer_selection_error"}]
    comparisons = {}
    for metric in ["reasoning_tokens", "output_tokens", "latency_seconds"]:
        obs, ci, p = bootstrap_mean_diff(
            [x[metric] for x in wrong], [x[metric] for x in correct]
        )
        comparisons[f"wrong_minus_correct_{metric}"] = {
            "mean_difference": obs,
            "bootstrap_95pct_ci": ci,
            "bootstrap_two_sided_p": p,
        }
    for name in PATTERNS:
        a = sum(x[f"flag_{name}"] for x in wrong)
        b = len(wrong) - a
        c = sum(x[f"flag_{name}"] for x in correct)
        d = len(correct) - c
        comparisons[f"wrong_vs_correct_{name}"] = {
            "wrong_flagged": a,
            "wrong_n": len(wrong),
            "correct_flagged": c,
            "correct_n": len(correct),
            "risk_difference": a / len(wrong) - c / len(correct),
            "fisher_two_sided_p": fisher_two_sided(a, b, c, d),
        }

    report = {
        "input": str(input_path),
        "n": len(audits),
        "manual_label_counts": Counter(x["label"] for x in audits),
        "strict_metric_counts": {
            "em_correct": sum(x["em"] == 1 for x in audits),
            "em_incorrect": sum(x["em"] != 1 for x in audits),
            "acc_correct": sum(x["acc"] == 1 for x in audits),
            "acc_incorrect": sum(x["acc"] != 1 for x in audits),
            "f1_zero": sum(x["f1"] == 0 for x in audits),
        },
        "confidence_observability": {
            "token_logprobs_present": any(
                get_event(row).get("model_meta", {}).get("choices", [{}])[0].get("logprobs")
                is not None
                for row in rows
            ),
            "explicit_scalar_confidence_present": False,
            "note": "Only weak behavioral/linguistic proxies are observable; they are not calibrated probabilities.",
        },
        "groups": groups,
        "comparisons": comparisons,
        "uncertainty_flagged_task_errors": [
            x["sample_id"] for x in wrong if x["flag_any_uncertainty"]
        ],
        "unflagged_task_errors": [
            x["sample_id"] for x in wrong if not x["flag_any_uncertainty"]
        ],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "confidence_audit.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=list)
    fields = [k for k in audits[0] if k not in {"raw_response", "reasoning_summary"}]
    with (output_dir / "per_sample_confidence_audit.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in audits)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    report = analyze(args.input, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, default=list))


if __name__ == "__main__":
    main()
