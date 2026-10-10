#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT=/home/researcher/smartsearch_rebuild
REPO_ROOT="$REBUILD_ROOT/SmartSearch"
CORE_PY="$REBUILD_ROOT/envs/src-core/bin/python"
RUN_DIR="${RUN_DIR:-$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/hotpot500_qwen3_llama31_soft_v2_full_v1}"
PREPARED_DIR="$REBUILD_ROOT/data/plan_and_budget_hotpot_agent_v2_soft/hotpot_random500_seed2026"
QUESTIONS="$PREPARED_DIR/questions.no_labels.jsonl"
LABELS="$PREPARED_DIR/gold_labels.sealed.jsonl"
PLANS="$RUN_DIR/plans/plans.jsonl"
POLY_DIR="$RUN_DIR/trajectories/polynomial"
STAMP="$(date -u +%Y%m%d_%H%M%S)"
NAME="plan_and_budget_hotpot_polynomial500_review_${STAMP}"
EXPORT_ROOT="$RUN_DIR/exports"
STAGE="$EXPORT_ROOT/$NAME"
ARCHIVE="$EXPORT_ROOT/$NAME.tar.gz"

required=(
  "$CORE_PY"
  "$QUESTIONS"
  "$LABELS"
  "$PLANS"
  "$POLY_DIR/records.jsonl"
  "$POLY_DIR/manifest.json"
  "$POLY_DIR/merged_manifest.json"
  "$POLY_DIR/shards/shard_0/manifest.json"
  "$POLY_DIR/shards/shard_1/manifest.json"
)
for file in "${required[@]}"; do
  [[ -s "$file" ]] || { echo "MISSING: $file" >&2; exit 1; }
done

mkdir -p \
  "$STAGE/inputs" \
  "$STAGE/polynomial/shards/shard_0" \
  "$STAGE/polynomial/shards/shard_1" \
  "$STAGE/logs" \
  "$STAGE/audit"

# Polynomial is already merged and immutable. These are read-only copies; the
# active Cosine directories and processes are deliberately not accessed.
cp "$QUESTIONS" "$STAGE/inputs/questions.no_labels.jsonl"
cp "$LABELS" "$STAGE/inputs/gold_labels.sealed.jsonl"
cp "$PLANS" "$STAGE/inputs/plans.jsonl"
cp "$POLY_DIR/records.jsonl" "$STAGE/polynomial/records.jsonl"
cp "$POLY_DIR/manifest.json" "$STAGE/polynomial/manifest.json"
cp "$POLY_DIR/merged_manifest.json" "$STAGE/polynomial/merged_manifest.json"

for shard in 0 1; do
  cp "$POLY_DIR/shards/shard_$shard/manifest.json" \
    "$STAGE/polynomial/shards/shard_$shard/manifest.json"
  if [[ -s "$POLY_DIR/shards/shard_$shard/failures.jsonl" ]]; then
    cp "$POLY_DIR/shards/shard_$shard/failures.jsonl" \
      "$STAGE/polynomial/shards/shard_$shard/failures.history.jsonl"
  fi
done

for log in \
  "$RUN_DIR/tmux_logs/dual_qwen/polynomial_shard_0.log" \
  "$RUN_DIR/tmux_logs/dual_qwen/polynomial_shard_1.log" \
  "$RUN_DIR/tmux_logs/dual_qwen/polynomial_merge.log"
do
  [[ -s "$log" ]] && cp "$log" "$STAGE/logs/$(basename "$log")"
done

export PYTHONPATH="$REPO_ROOT/scripts/experiments${PYTHONPATH:+:$PYTHONPATH}"
"$CORE_PY" - \
  "$STAGE/inputs/questions.no_labels.jsonl" \
  "$STAGE/inputs/gold_labels.sealed.jsonl" \
  "$STAGE/inputs/plans.jsonl" \
  "$STAGE/polynomial/records.jsonl" \
  "$STAGE/polynomial/manifest.json" \
  "$STAGE/polynomial/shards" \
  "$STAGE/audit" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

from plan_and_budget_hotpot_agent_v2_soft.analyze import (
    summarize,
    validate_trajectory,
)
from plan_and_budget_hotpot_agent_v2_soft.common import (
    pick_answer,
    pick_id,
    read_jsonl,
    sha256_file,
    utc_now,
)

questions_path = Path(sys.argv[1])
labels_path = Path(sys.argv[2])
plans_path = Path(sys.argv[3])
records_path = Path(sys.argv[4])
manifest_path = Path(sys.argv[5])
shards_path = Path(sys.argv[6])
audit_dir = Path(sys.argv[7])

def unique_rows(path, key="sample_id"):
    rows = read_jsonl(path)
    result = {}
    duplicates = []
    for row in rows:
        item_id = str(row[key])
        if item_id in result:
            duplicates.append(item_id)
        result[item_id] = row
    return rows, result, duplicates

question_rows, questions, question_dupes = unique_rows(questions_path)
plan_rows, plans, plan_dupes = unique_rows(plans_path)
record_rows, records, record_dupes = unique_rows(records_path)
label_rows = read_jsonl(labels_path)
labels = {str(pick_id(row)): pick_answer(row) for row in label_rows}
merged_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

errors = []
for name, rows, mapping, dupes in (
    ("questions", question_rows, questions, question_dupes),
    ("plans", plan_rows, plans, plan_dupes),
    ("polynomial records", record_rows, records, record_dupes),
):
    if len(rows) != 500:
        errors.append(f"{name}: expected 500 rows, found {len(rows)}")
    if len(mapping) != 500:
        errors.append(f"{name}: expected 500 unique IDs, found {len(mapping)}")
    if dupes:
        errors.append(f"{name}: duplicate IDs {dupes[:20]}")

expected_ids = set(questions)
for name, mapping in (("plans", plans), ("records", records), ("labels", labels)):
    missing = sorted(expected_ids - set(mapping))
    extra = sorted(set(mapping) - expected_ids)
    if missing or extra:
        errors.append(
            f"{name}: ID mismatch; missing={missing[:20]} extra={extra[:20]}"
        )

if merged_manifest.get("passed") is not True:
    errors.append("Polynomial merged manifest is not passed=true")
if int(merged_manifest.get("records", -1)) != 500:
    errors.append("Polynomial merged manifest does not report 500 records")

for sample_id, row in records.items():
    errors.extend(validate_trajectory(row, plans[sample_id], "polynomial"))

historical_failures = []
for shard in (0, 1):
    path = shards_path / f"shard_{shard}" / "failures.history.jsonl"
    if path.is_file():
        historical_failures.extend(read_jsonl(path))
historical_failure_ids = sorted(
    {str(row.get("sample_id")) for row in historical_failures}
)
unresolved_historical_ids = sorted(set(historical_failure_ids) - set(records))
if unresolved_historical_ids:
    errors.append(
        f"Historical failures not recovered: {unresolved_historical_ids}"
    )

summary = summarize(records, labels, plans) if not errors else {}
per_sample = summary.pop("per_sample", []) if summary else []

answer_statuses = Counter()
finish_types = Counter()
technical_caps = Counter()
model_paths = Counter()
budget_counts = Counter()
planner_levels = Counter()
length_finished_questions = set()
for sample_id, row in records.items():
    technical_caps[str(row.get("technical_generation_max_tokens_per_turn"))] += 1
    model_info = row.get("reasoner_model_info", {})
    model_paths[str(model_info.get("model_path"))] += 1
    budget_counts[str(row.get("official_total_output_budget"))] += 1
    planner_levels[str(row.get("level"))] += 1
    for step in row.get("steps", []):
        answer_statuses[str(step.get("answer_status"))] += 1
        for turn in step.get("model_turns", []):
            finish = turn.get("finish_reason")
            if isinstance(finish, dict):
                finish = finish.get("type") or json.dumps(finish, sort_keys=True)
            finish = str(finish)
            finish_types[finish] += 1
            if finish == "length":
                length_finished_questions.add(sample_id)

scored_path = audit_dir / "polynomial.scored.jsonl"
with scored_path.open("w", encoding="utf-8", newline="\n") as stream:
    for row in per_sample:
        stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

audit = {
    "created_at": utc_now(),
    "passed": not errors,
    "scope": "Polynomial only; read-only snapshot while Cosine continued running",
    "expected_records": 500,
    "records": len(record_rows),
    "unique_record_ids": len(records),
    "errors": errors,
    "error_count": len(errors),
    "historical_failure_records": len(historical_failures),
    "historical_failure_unique_ids": len(historical_failure_ids),
    "historical_failure_ids": historical_failure_ids,
    "historical_failures_recovered": len(unresolved_historical_ids) == 0,
    "unresolved_historical_failure_ids": unresolved_historical_ids,
    "metrics": summary,
    "descriptive": {
        "budget_distribution": dict(sorted(budget_counts.items())),
        "planner_level_distribution": dict(sorted(planner_levels.items())),
        "technical_generation_caps": dict(sorted(technical_caps.items())),
        "reasoner_model_paths": dict(sorted(model_paths.items())),
        "answer_statuses": dict(sorted(answer_statuses.items())),
        "finish_types": dict(sorted(finish_types.items())),
        "questions_with_at_least_one_length_finish": len(length_finished_questions),
    },
    "source_sha256": {
        "questions": sha256_file(questions_path),
        "labels": sha256_file(labels_path),
        "plans": sha256_file(plans_path),
        "polynomial_records": sha256_file(records_path),
        "polynomial_manifest": sha256_file(manifest_path),
    },
    "policy_notes": {
        "budget": "P&B budget is a prompt-level soft target; overruns are recorded but not enforced.",
        "technical_length": "Length-finished/fallback outputs remain valid experimental outcomes and are not zeroed solely for truncation.",
        "planner": "Planner decomposition/credit behavior is retained as part of the reproduced method.",
        "citations": "Hallucinated or imperfect citations are retained as model/method behavior.",
        "historical_failures": "Failure JSONL is append-only history; final resolution is determined by the complete merged 500-record set.",
    },
}
(audit_dir / "summary.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)

if summary:
    report = [
        "# Plan-and-Budget × HotpotQA：Polynomial 500 题中期审计",
        "",
        "## 审计结论",
        "",
        f"- 完整性：{'PASSED' if not errors else 'FAILED'}。合并轨迹 {len(record_rows)}/500，唯一题号 {len(records)}/500。",
        f"- 历史失败：{len(historical_failures)} 条、{len(historical_failure_ids)} 个题号；均已由最终合并轨迹恢复：{not unresolved_historical_ids}。",
        "- 本报告只审计已经冻结的 Polynomial；生成时未加载金标。正在运行的 Cosine 未被读取、停止或修改。",
        "",
        "## Polynomial 主结果",
        "",
        "| 指标 | 结果 |",
        "|---|---:|",
        f"| EM | {summary['em']:.6f} |",
        f"| F1 | {summary['f1']:.6f} |",
        f"| 平均官方软预算 | {summary['mean_official_budget']:.2f} |",
        f"| 平均 Reasoner completion token | {summary['mean_reasoner_completion_tokens']:.2f} |",
        f"| 平均软预算超额 token | {summary['mean_soft_budget_overrun_tokens']:.2f} |",
        f"| 软预算遵从率 | {summary['soft_budget_compliance_rate']:.4%} |",
        f"| 平均 Search 次数 | {summary['mean_search_calls']:.4f} |",
        f"| 平均模型调用次数 | {summary['mean_model_calls']:.4f} |",
        f"| 平均 Reasoner 总 token | {summary['mean_reasoner_total_tokens']:.2f} |",
        f"| 平均 Planner 总 token | {summary['mean_planner_total_tokens']:.2f} |",
        f"| 平均端到端模型 token | {summary['mean_end_to_end_model_tokens']:.2f} |",
        f"| fallback 题目率 | {summary['fallback_question_rate']:.4%} |",
        "",
        "## 描述性可靠性信息",
        "",
        f"- 技术生成上限分布：`{dict(sorted(technical_caps.items()))}`。",
        f"- 至少一次 length finish 的题目：{len(length_finished_questions)}/500；仅标记，不统一记错。",
        f"- Answer status：`{dict(sorted(answer_statuses.items()))}`。",
        f"- Finish type：`{dict(sorted(finish_types.items()))}`。",
        f"- 官方预算分布：`{dict(sorted(budget_counts.items()))}`。",
        f"- Planner level 分布：`{dict(sorted(planner_levels.items()))}`。",
        "",
        "## 协议解释",
        "",
        "- 官方预算是 Prompt 软目标，未用于截断、终止或强制后处理。",
        "- 达到技术长度上限的输出保留为模型实际结果；不会只因截断而统一置零。",
        "- Planner 缺陷、引用幻觉与预算不遵从均作为复现方法的自然行为保留。",
        "- `failures.history.jsonl` 是追加式历史日志；其中记录不代表最终仍失败。",
        "",
    ]
else:
    report = [
        "# Polynomial 500 题中期审计",
        "",
        "完整性检查失败，未计算结果指标。",
        "",
        "```json",
        json.dumps(errors, ensure_ascii=False, indent=2),
        "```",
    ]
(audit_dir / "REPORT_ZH.md").write_text("\n".join(report) + "\n", encoding="utf-8")

print(json.dumps({
    "passed": not errors,
    "records": len(record_rows),
    "unique_ids": len(records),
    "historical_failure_ids": len(historical_failure_ids),
    "historical_failures_recovered": not unresolved_historical_ids,
    "EM": summary.get("em") if summary else None,
    "F1": summary.get("f1") if summary else None,
    "mean_end_to_end_model_tokens": summary.get("mean_end_to_end_model_tokens") if summary else None,
    "mean_search_calls": summary.get("mean_search_calls") if summary else None,
    "soft_budget_compliance_rate": summary.get("soft_budget_compliance_rate") if summary else None,
    "questions_with_length_finish": len(length_finished_questions),
}, ensure_ascii=False, indent=2))

if errors:
    raise SystemExit(1)
PY

(cd "$STAGE" && \
  find . -type f ! -name FILES.sha256 -print0 \
  | sort -z \
  | xargs -0 sha256sum > FILES.sha256)

tar -C "$EXPORT_ROOT" -czf "$ARCHIVE" "$NAME"
sha256sum "$ARCHIVE" | tee "$ARCHIVE.sha256"

echo
echo "POLYNOMIAL SNAPSHOT AUDIT PASSED"
echo "Report:  $STAGE/audit/REPORT_ZH.md"
echo "Summary: $STAGE/audit/summary.json"
echo "Archive: $ARCHIVE"
echo "SHA256:  $ARCHIVE.sha256"
