#!/usr/bin/env python3
"""Create a concise, human-readable initial audit for the five-question Gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import read_jsonl, utc_stamp, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()
    run = Path(args.run_dir)

    dataset = json.loads((run / "dataset/dataset_manifest.json").read_text(encoding="utf-8"))
    merge = json.loads(
        (run / "generations/generations.merged.jsonl.manifest.json").read_text(encoding="utf-8")
    )
    final = json.loads((run / "FINAL_AUDIT.json").read_text(encoding="utf-8"))
    summary = json.loads((run / "analysis/summary.json").read_text(encoding="utf-8"))
    roots = read_jsonl(run / "roots/roots.jsonl")
    metrics = read_jsonl(run / "analysis/per_record_metrics.jsonl")
    by_key = {(str(row["sample_id"]), int(row["rollout_id"])): row for row in metrics}

    errors = []
    if dataset.get("records") != 5 or dataset.get("selection") != "frozen_prefix_5":
        errors.append(f"Gate dataset mismatch: {dataset.get('records')}, {dataset.get('selection')}")
    if len(roots) != 5:
        errors.append(f"Expected 5 roots; got {len(roots)}")
    if merge.get("records") != 10 or not merge.get("passed"):
        errors.append(f"Expected a passed 10-record merge; got {merge}")
    if not final.get("passed"):
        errors.append("FINAL_AUDIT did not pass")

    rows = []
    for root in roots:
        sample_id = str(root["sample_id"])
        item = {
            "sample_id": sample_id,
            "adaptive_k": int(root["selection"]["selected_k"]),
            "largest_gap_after_rank": int(root["selection"]["largest_gap_after_rank_1based"]),
            "selected_context_tokens": int(root["selected_context_token_count"]),
            "retrieval_latency_seconds": float(root["retrieval_latency_seconds"]),
            "rollouts": [],
        }
        for rollout_id in (0, 1):
            metric = by_key.get((sample_id, rollout_id))
            if metric is None:
                errors.append(f"Missing metric {(sample_id, rollout_id)}")
                continue
            item["rollouts"].append({
                "rollout_id": rollout_id,
                "em": float(metric["em"]),
                "f1": float(metric["f1"]),
                "answer_status": metric["answer_status"],
                "finish_type": metric["finish_type"],
                "prompt_tokens": int(metric["prompt_tokens"]),
                "completion_tokens": int(metric["completion_tokens"]),
                "selected_title_recall": float(metric["selected_title_recall"]),
            })
        rows.append(item)

    report = {
        "created_at": utc_stamp(),
        "passed": not errors,
        "errors": errors,
        "dataset": {
            "records": dataset.get("records"),
            "source_records": dataset.get("source_records"),
            "selection": dataset.get("selection"),
            "queries_sha256": dataset.get("queries_sha256"),
            "gold_visible_during_generation": False,
        },
        "protocol_integrity": {
            "roots": final.get("roots"),
            "generation_records": final.get("generation_records"),
            "retrieval_calls_recorded": final.get("retrieval_calls_recorded"),
            "model_calls_expected": final.get("model_calls_expected"),
            "adaptive_k_recomputed_for_all_roots": final.get(
                "adaptive_k_recomputed_for_all_roots"
            ),
            "no_iterative_search": final.get("no_iterative_search"),
            "rollouts_share_frozen_root": final.get("rollouts_share_frozen_root"),
            "capacity_adjusted_generation_count": final.get(
                "capacity_adjusted_generation_count"
            ),
            "warning_count": final.get("warning_count"),
            "warnings": final.get("warnings"),
        },
        "quality": summary.get("overall"),
        "adaptive_k": summary.get("adaptive_k"),
        "rollout_analysis": summary.get("rollout_analysis"),
        "per_question": rows,
    }
    write_json(run / "analysis/GATE_INITIAL_AUDIT.json", report)

    lines = [
        "# Adaptive-k 五题 Gate 初步审计",
        "",
        f"- 结论：{'PASSED' if report['passed'] else 'FAILED'}。",
        f"- 数据：{dataset.get('records')} 题；检索根：{len(roots)}；生成记录：{merge.get('records')}。",
        f"- 实际检索调用：{final.get('retrieval_calls_recorded')}；模型调用：{final.get('model_calls_expected')}。",
        f"- EM/F1：{summary['overall']['em']:.4f} / {summary['overall']['f1']:.4f}。",
        f"- 格式有效率：{summary['overall']['valid_answer_rate']:.4f}；长度终止率：{summary['overall']['length_termination_rate']:.4f}。",
        f"- Adaptive-k 均值/中位数/范围：{summary['adaptive_k']['mean']:.2f} / "
        f"{summary['adaptive_k']['median']:.1f} / "
        f"[{summary['adaptive_k']['min']}, {summary['adaptive_k']['max']}]。",
        f"- 两次 rollout 答案一致率：{summary['rollout_analysis']['same_normalized_answer_rate']:.4f}。",
        "",
        "| sample_id | k | gap后rank | 上下文token | r0 EM/F1/状态 | r1 EM/F1/状态 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in rows:
        rollouts = row["rollouts"]
        cells = []
        for rollout in rollouts:
            cells.append(
                f"{rollout['em']:.0f}/{rollout['f1']:.3f}/{rollout['answer_status']}"
            )
        while len(cells) < 2:
            cells.append("MISSING")
        lines.append(
            f"| {row['sample_id']} | {row['adaptive_k']} | {row['largest_gap_after_rank']} | "
            f"{row['selected_context_tokens']} | {cells[0]} | {cells[1]} |"
        )
    if errors:
        lines.extend(["", "## Errors", ""] + [f"- {error}" for error in errors])
    (run / "analysis/GATE_INITIAL_AUDIT_ZH.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
