#!/usr/bin/env python3
"""Matched three-condition statistics for the OpenRouter GPT-5.6 Sol experiment."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common import answer_metrics, read_jsonl, utc_stamp  # noqa: E402


ACTION_PRIORITY = {"S": 1, "R": 2, "C": 3}


def mean(rows: Iterable[Dict[str, Any]], field: str) -> float:
    values = [float(row.get(field, 0.0)) for row in rows]
    return statistics.mean(values) if values else 0.0


def paired_bootstrap(
    left: Dict[str, Dict[str, Any]],
    right: Dict[str, Dict[str, Any]],
    field: str,
    samples: int,
    seed: int,
) -> Dict[str, Any]:
    ids = sorted(set(left) & set(right))
    diffs = [float(right[sid].get(field, 0.0)) - float(left[sid].get(field, 0.0)) for sid in ids]
    if not diffs:
        return {"n": 0, "mean_difference_right_minus_left": 0.0, "ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        draws.append(statistics.mean(rng.choice(diffs) for _ in diffs))
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return {
        "n": len(ids),
        "mean_difference_right_minus_left": statistics.mean(diffs),
        "ci95": [lo, hi],
        "win_tie_loss_right_vs_left": {
            "win": sum(diff > 0 for diff in diffs),
            "tie": sum(diff == 0 for diff in diffs),
            "loss": sum(diff < 0 for diff in diffs),
        },
    }


def path_tie_key(path: str) -> Tuple[int, ...]:
    return tuple(ACTION_PRIORITY.get(action, 0) for action in str(path)[1:])


def best_tree_record(
    rows: Sequence[Dict[str, Any]], quality: str, *, quality_then_cost: bool = False
) -> Dict[str, Any]:
    """Select one terminal record under an explicitly named Oracle.

    Quality-Oracle uses the experiment's fixed Commit > Read > Search tie-break.
    Quality-then-Cost Oracle instead minimizes total tokens among quality ties.
    Keeping both prevents one summary from silently changing the DP policy.
    """
    if quality_then_cost:
        return min(
            rows,
            key=lambda row: (
                -float(row.get(quality, 0.0)),
                float(row.get("total_cost_tokens", math.inf)),
                tuple(-value for value in path_tie_key(str(row.get("path", "")))),
                str(row.get("path", "")),
            ),
        )
    return max(
        rows,
        key=lambda row: (
            float(row.get(quality, 0.0)),
            path_tie_key(str(row.get("path", ""))),
            str(row.get("path", "")),
        ),
    )


def condition_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "samples": len(rows),
        "em": mean(rows, "em"),
        "f1": mean(rows, "f1"),
        "acc": mean(rows, "acc"),
        "normalized_exact": mean(rows, "normalized_exact"),
        "normalized_contains": mean(rows, "normalized_contains"),
        "search_calls": mean(rows, "search_calls"),
        "retrieved_docs_total": mean(rows, "retrieved_docs_total"),
        "retrieved_tokens_total": mean(rows, "retrieved_tokens_total"),
        "llm_tokens_total": mean(rows, "llm_tokens_total"),
        "total_cost_tokens": mean(rows, "total_cost_tokens"),
        "api_cost_usd": mean(rows, "api_cost_usd"),
        "api_cost_usd_sum": sum(float(row.get("api_cost_usd", 0.0)) for row in rows),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blank", required=True)
    parser.add_argument("--roots", required=True)
    parser.add_argument("--tree", required=True)
    parser.add_argument("--neutral", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--quality_metric", default="f1")
    parser.add_argument("--bootstrap_samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    blank = read_jsonl(args.blank)
    roots = read_jsonl(args.roots)
    tree = read_jsonl(args.tree)
    neutral = read_jsonl(args.neutral)
    blank_by = {str(row["sample_id"]): row for row in blank}
    neutral_by = {str(row["sample_id"]): row for row in neutral}
    paper_by = {}
    missing_paper_prediction_ids = []
    for sid, row in blank_by.items():
        paper_prediction = str(row.get("paper_prediction", "")).strip()
        if not paper_prediction:
            missing_paper_prediction_ids.append(sid)
            continue
        metrics = answer_metrics(paper_prediction, row.get("gold_answers", []))
        paper_by[sid] = {
            "sample_id": sid,
            **metrics,
            "search_calls": 0,
            "retrieved_docs_total": 0,
            "retrieved_tokens_total": 0,
            "llm_tokens_total": 0,
            "total_cost_tokens": 0,
        }
    root_by = {str(row["sample_id"]): row for row in roots}
    tree_by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in tree:
        tree_by[str(row["sample_id"])].append(row)

    oracle_all = {sid: best_tree_record(rows, args.quality_metric) for sid, rows in tree_by.items() if rows}
    cost_oracle_all = {
        sid: best_tree_record(rows, args.quality_metric, quality_then_cost=True)
        for sid, rows in tree_by.items()
        if rows
    }
    natural_ids = {sid for sid, root in root_by.items() if root.get("root_origin") == "openrouter_natural_root"}
    oracle_natural = {sid: row for sid, row in oracle_all.items() if sid in natural_ids}
    cost_oracle_natural = {sid: row for sid, row in cost_oracle_all.items() if sid in natural_ids}
    blank_natural = {sid: row for sid, row in blank_by.items() if sid in natural_ids}
    neutral_natural = {sid: row for sid, row in neutral_by.items() if sid in natural_ids}
    primary_estimand_available = bool(natural_ids)

    paired_fields = ("f1", "em", "acc", "search_calls", "retrieved_tokens_total", "total_cost_tokens")
    blank_vs_neutral_primary = {
        field: paired_bootstrap(blank_natural, neutral_natural, field, args.bootstrap_samples, args.seed + index)
        for index, field in enumerate(paired_fields)
    }
    blank_vs_neutral_all_roots = {
        field: paired_bootstrap(blank_by, neutral_by, field, args.bootstrap_samples, args.seed + 25 + index)
        for index, field in enumerate(paired_fields)
    }
    paper_vs_blank = {
        field: paired_bootstrap(paper_by, blank_by, field, args.bootstrap_samples, args.seed + 50 + index)
        for index, field in enumerate(("f1", "em", "acc"))
    }
    blank_vs_oracle_primary = {
        field: paired_bootstrap(blank_natural, oracle_natural, field, args.bootstrap_samples, args.seed + 100 + index)
        for index, field in enumerate(paired_fields)
    }

    condition_summaries = {
        "blank": condition_summary(list(blank_by.values())),
        "neutral": condition_summary(list(neutral_by.values())),
        "blank_primary_natural_roots": condition_summary(list(blank_natural.values())),
        "neutral_primary_natural_roots": condition_summary(list(neutral_natural.values())),
        "tree_oracle_primary_natural_roots": condition_summary(list(oracle_natural.values())),
        "tree_oracle_sensitivity_all_roots": condition_summary(list(oracle_all.values())),
        "tree_quality_then_cost_oracle_primary_natural_roots": condition_summary(
            list(cost_oracle_natural.values())
        ),
        "tree_quality_then_cost_oracle_sensitivity_all_roots": condition_summary(
            list(cost_oracle_all.values())
        ),
    }
    if paper_by:
        condition_summaries["original_paper_saved_predictions"] = condition_summary(list(paper_by.values()))

    report = {
        "created_at": utc_stamp(),
        "quality_metric": args.quality_metric,
        "causal_contrasts": {
            "blank_vs_neutral_primary_natural_roots": (
                "Prompt/scaffold effect among samples whose Blank trajectory naturally produced the reused first Search root."
            ),
            "blank_vs_neutral_all_roots": (
                "Sensitivity contrast only. For forced-root samples it combines the first-Search evidence intervention "
                "with the scaffold/prompt effect and must not be described as a pure prompt effect."
            ),
            "within_tree_actions": "Action-intervention effect at a shared counterfactual parent state.",
            "blank_vs_tree_oracle": "Attainable counterfactual upper bound, not deployed-policy accuracy.",
        },
        "primary_natural_root_estimand": {
            "available": primary_estimand_available,
            "sample_count": len(natural_ids),
            "interpretation": (
                "Available only when Blank naturally searched. If unavailable, all tree results are conditional "
                "on a controller-forced first Search and are reported as sensitivity analysis."
            ),
        },
        "condition_summaries": condition_summaries,
        "paper_baseline_availability": {
            "available_predictions": len(paper_by),
            "required_predictions": len(blank_by),
            "paired_comparison_available": len(paper_by) == len(blank_by),
            "missing_sample_ids": missing_paper_prediction_ids,
            "interpretation": (
                "The paper comparison is valid only when paired_comparison_available is true. "
                "Missing saved predictions must not be scored as empty answers."
            ),
        },
        "paired_blank_vs_neutral_primary_natural_roots": blank_vs_neutral_primary,
        "paired_blank_vs_neutral_sensitivity_all_roots": blank_vs_neutral_all_roots,
        "paired_original_paper_vs_gpt56sol_blank": paper_vs_blank,
        "paired_blank_vs_tree_oracle_primary": blank_vs_oracle_primary,
        "oracle_path_distribution_primary": dict(Counter(row.get("path", "") for row in oracle_natural.values())),
        "oracle_path_distribution_all_roots": dict(Counter(row.get("path", "") for row in oracle_all.values())),
        "quality_then_cost_oracle_path_distribution_primary": dict(
            Counter(row.get("path", "") for row in cost_oracle_natural.values())
        ),
        "quality_then_cost_oracle_path_distribution_all_roots": dict(
            Counter(row.get("path", "") for row in cost_oracle_all.values())
        ),
        "oracle_definitions": {
            "tree_oracle": "maximize quality with fixed Commit > Read > Search tie-break",
            "tree_quality_then_cost_oracle": "maximize quality, then minimize total_cost_tokens",
        },
        "sample_counts": {
            "blank": len(blank_by),
            "paper_saved_predictions": len(paper_by),
            "neutral": len(neutral_by),
            "tree": len(tree_by),
            "tree_natural_root_primary": len(oracle_natural),
            "tree_forced_root_sensitivity": len(oracle_all) - len(oracle_natural),
        },
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "three_condition_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# OpenRouter GPT-5.6 Sol three-condition analysis",
        "",
        "## Interpretation boundary",
        "",
        "- Blank vs Neutral estimates a prompt/scaffold effect only on natural-root samples.",
        "- Search vs Read vs Commit inside the tree estimates assigned action effects from a shared state.",
        "- The tree oracle is an attainable counterfactual upper bound, not the performance of a learned controller.",
        "- The primary tree analysis includes only natural GPT-5.6 Sol roots; forced roots are sensitivity analysis.",
        (
            "- Primary natural-root estimand available: "
            f"{str(primary_estimand_available).lower()} (N={len(natural_ids)})."
        ),
        "- On forced-root samples, Blank vs Neutral is not a pure prompt contrast because the first Search is also intervened on.",
        "",
        "## Condition means",
        "",
        "| Condition | N | F1 | EM | Searches | Retrieved tokens | Total cost tokens | Mean API USD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, summary in report["condition_summaries"].items():
        markdown.append(
            f"| {label} | {summary['samples']} | {summary['f1']:.4f} | {summary['em']:.4f} | "
            f"{summary['search_calls']:.3f} | {summary['retrieved_tokens_total']:.1f} | "
            f"{summary['total_cost_tokens']:.1f} | ${summary['api_cost_usd']:.6f} |"
        )
    markdown += [
        "",
        "## Paired Blank vs Neutral: primary natural-root estimand",
        "",
        "Positive differences mean Neutral minus Blank.",
        "",
    ]
    if primary_estimand_available:
        for field, value in blank_vs_neutral_primary.items():
            markdown.append(
                f"- {field} (N={value['n']}): "
                f"{value['mean_difference_right_minus_left']:.4f}, 95% CI {value['ci95']}"
            )
    else:
        markdown.append(
            "- Unavailable: no Blank trajectory naturally produced the reused first Search root."
        )
    markdown += [
        "",
        "## Paired Blank vs Neutral: all-root sensitivity",
        "",
        (
            "This is not a pure prompt effect on forced-root samples: it also includes the "
            "controller-forced first Search and its evidence."
        ),
        "",
    ]
    for field, value in blank_vs_neutral_all_roots.items():
        markdown.append(
            f"- {field} (N={value['n']}): "
            f"{value['mean_difference_right_minus_left']:.4f}, 95% CI {value['ci95']}"
        )
    (output_dir / "report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
