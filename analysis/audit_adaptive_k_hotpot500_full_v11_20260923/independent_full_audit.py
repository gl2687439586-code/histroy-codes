from __future__ import annotations

import hashlib
import json
import math
import re
import string
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


RUN = Path(__file__).parent / "full500_r2_t06_v11"


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def canonical_hash(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_hash(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def title_legacy(value):
    return re.sub(r"\s+", " ", str(value or "").replace("_", " ").strip().lower())


def title_robust(value):
    text = unicodedata.normalize("NFKC", str(value or "")).replace("_", " ").strip().lower()
    return re.sub(r"\s+", " ", text)


def normalized_text(value):
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = "".join(ch if ch.isalnum() else " " for ch in text)
    return " ".join(text.split())


def normalized_answer(value):
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def em_f1(prediction, golds):
    prediction = normalized_answer(prediction)
    refs = [normalized_answer(gold) for gold in golds]
    em = float(any(prediction == ref for ref in refs))
    scores = []
    for ref in refs:
        pred_tokens, ref_tokens = prediction.split(), ref.split()
        if not pred_tokens or not ref_tokens:
            scores.append(float(pred_tokens == ref_tokens))
            continue
        common = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
        if not common:
            scores.append(0.0)
            continue
        precision, recall = common / len(pred_tokens), common / len(ref_tokens)
        scores.append(2 * precision * recall / (precision + recall))
    return em, max(scores)


def mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return 0.0
    point = (len(values) - 1) * fraction
    left, right = math.floor(point), math.ceil(point)
    if left == right:
        return float(values[left])
    return values[left] * (right - point) + values[right] * (point - left)


questions = read_jsonl(RUN / "dataset/questions.no_labels.jsonl")
labels = {str(row["sample_id"]): row for row in read_jsonl(RUN / "dataset/gold_labels.sealed.jsonl")}
roots = read_jsonl(RUN / "roots/roots.jsonl")
generations = read_jsonl(RUN / "generations/generations.merged.jsonl")
metrics = read_jsonl(RUN / "analysis/per_record_metrics.jsonl")
root_by_id = {str(row["sample_id"]): row for row in roots}
generation_by_key = {(str(row["sample_id"]), int(row["rollout_id"])): row for row in generations}
metric_by_key = {(str(row["sample_id"]), int(row["rollout_id"])): row for row in metrics}
question_ids = [str(row["sample_id"]) for row in questions]
errors = []

if not (
    len(questions) == len(labels) == len(roots) == len(root_by_id) == 500
    and len(generations) == len(generation_by_key) == len(metrics) == len(metric_by_key) == 1000
):
    errors.append("500/1000 cardinality or uniqueness check failed")

per_question = []
metric_mismatches = 0
legacy_full_selected = robust_full_selected = 0

for sample_id in question_ids:
    root, label = root_by_id[sample_id], labels[sample_id]
    candidates, selected = root["candidates"], root["selected_documents"]
    scores = [float(row["score"]) for row in candidates]
    if len(candidates) != 1000 or any(scores[i] < scores[i + 1] for i in range(999)):
        errors.append(f"candidate score order/count failed: {sample_id}")
    drops = [scores[i] - scores[i + 1] for i in range(999)]
    eligible = drops[: 999 - int(999 * 0.1)]
    gap_index = max(range(len(eligible)), key=eligible.__getitem__)
    expected_k = min(1000, gap_index + 1 + 5)
    if expected_k != int(root["selection"]["selected_k"]):
        errors.append(f"Adaptive-k recomputation failed: {sample_id}")
    projection = [
        {key: row.get(key) for key in ("rank", "id", "title", "score")} for row in selected
    ]
    prefix = [
        {key: row.get(key) for key in ("rank", "id", "title", "score")}
        for row in candidates[:expected_k]
    ]
    if projection != prefix:
        errors.append(f"selected documents are not the candidate prefix: {sample_id}")
    context = "\n\n".join(row["contents"] for row in selected)
    if hashlib.sha256(context.encode()).hexdigest() != root["selected_context_sha256"]:
        errors.append(f"selected context hash failed: {sample_id}")
    root_copy = dict(root)
    root_hash = root_copy.pop("root_hash")
    if canonical_hash(root_copy) != root_hash:
        errors.append(f"root record hash failed: {sample_id}")

    gold_legacy = {title_legacy(value) for value in label.get("supporting_titles") or []}
    candidate_legacy = {title_legacy(row["title"]) for row in candidates}
    selected_legacy = {title_legacy(row["title"]) for row in selected}
    legacy_full_selected += int(gold_legacy <= selected_legacy)

    gold_titles = {title_robust(value) for value in label.get("supporting_titles") or []}
    candidate_titles = {title_robust(row["title"]) for row in candidates}
    selected_titles = {title_robust(row["title"]) for row in selected}
    candidate_complete = bool(gold_titles <= candidate_titles)
    selected_complete = bool(gold_titles <= selected_titles)
    robust_full_selected += int(selected_complete)
    first_rank = {}
    for row in candidates:
        title = title_robust(row["title"])
        if title in gold_titles and title not in first_rank:
            first_rank[title] = int(row["rank"])
    max_support_rank = max(first_rank.values()) if candidate_complete and first_rank else None
    deficit = max(0, max_support_rank - expected_k) if max_support_rank is not None else None

    context_normalized = normalized_text(context)
    supporting_facts = label.get("supporting_facts") or []
    fact_hits = [
        int(bool(normalized_text(item.get("sentence"))) and normalized_text(item.get("sentence")) in context_normalized)
        for item in supporting_facts
    ]
    fact_recall = mean(fact_hits) if fact_hits else 0.0
    facts_complete = bool(fact_hits and all(fact_hits))
    gold_answers = label.get("gold_answers") or [label.get("answer") or ""]
    answer_norms = [normalized_text(value) for value in gold_answers]
    answer_present = any(
        answer and answer not in {"yes", "no"} and answer in context_normalized for answer in answer_norms
    )

    rollout_rows = []
    for rollout_id in (0, 1):
        key = (sample_id, rollout_id)
        generation, recorded_metric = generation_by_key[key], metric_by_key[key]
        if (
            generation["root_hash"] != root_hash
            or generation["selected_document_ids"] != [row["id"] for row in selected]
            or generation["adaptive_k"] != expected_k
            or generation["retrieval_calls"] != 1
            or generation["candidate_count"] != 1000
            or generation["temperature"] != 0.6
            or generation["top_p"] != 0.95
            or generation["effective_max_new_tokens"] != 16384
            or generation["model_path_verified"] != "/home/researcher/smartsearch_rebuild/models/Qwen3-8B"
        ):
            errors.append(f"generation protocol failed: {key}")
        generation_copy = dict(generation)
        generation_hash = generation_copy.pop("record_hash")
        if canonical_hash(generation_copy) != generation_hash:
            errors.append(f"generation record hash failed: {key}")
        em, f1 = em_f1(generation["final_answer"], gold_answers)
        if abs(em - float(recorded_metric["em"])) > 1e-12 or abs(f1 - float(recorded_metric["f1"])) > 1e-12:
            metric_mismatches += 1
        rollout_rows.append({
            "rollout": rollout_id,
            "em": em,
            "f1": f1,
            "completion_tokens": int(generation["model_meta"]["completion_tokens"]),
            "total_tokens": int(generation["model_meta"]["prompt_tokens"])
            + int(generation["model_meta"]["completion_tokens"]),
            "answer_status": generation["answer_status"],
            "finish_type": generation["finish_type"],
            "answer": generation["final_answer"],
        })
    if metric_mismatches:
        pass
    any_em = any(row["em"] for row in rollout_rows)
    both_em = all(row["em"] for row in rollout_rows)
    if not candidate_complete:
        evidence_class = "retrieval_title_miss"
    elif not selected_complete:
        evidence_class = "adaptive_k_title_cut"
    else:
        evidence_class = "selected_titles_complete"
    per_question.append({
        "sample_id": sample_id,
        "type": label.get("type"),
        "level": label.get("level"),
        "k": expected_k,
        "base_k": int(root["selection"]["base_k_before_buffer"]),
        "context_tokens": int(root["selected_context_token_count"]),
        "candidate_complete": candidate_complete,
        "selected_complete": selected_complete,
        "max_support_rank": max_support_rank,
        "deficit": deficit,
        "fact_recall": fact_recall,
        "facts_complete": facts_complete,
        "answer_present": answer_present,
        "evidence_class": evidence_class,
        "any_em": any_em,
        "both_em": both_em,
        "mean_em": mean(row["em"] for row in rollout_rows),
        "mean_f1": mean(row["f1"] for row in rollout_rows),
        "mean_total_tokens": mean(row["total_tokens"] for row in rollout_rows),
        "same_answer": normalized_answer(rollout_rows[0]["answer"]) == normalized_answer(rollout_rows[1]["answer"]),
    })

if metric_mismatches:
    errors.append(f"metric recomputation mismatches: {metric_mismatches}")

root_path = RUN / "roots/roots.jsonl"
generation_path = RUN / "generations/generations.merged.jsonl"
merge = json.loads((RUN / "generations/generations.merged.jsonl.manifest.json").read_text(encoding="utf-8"))
summary = json.loads((RUN / "analysis/summary.json").read_text(encoding="utf-8"))
final = json.loads((RUN / "FINAL_AUDIT.json").read_text(encoding="utf-8"))
if merge["roots_sha256"] != file_hash(root_path):
    errors.append("roots file SHA mismatch")
if summary["input_hashes"]["generations"] != file_hash(generation_path):
    errors.append("generation file SHA mismatch")
if merge["content_hash"] != canonical_hash(generations):
    errors.append("merged content hash mismatch")
if final["content_hashes"]["roots"] != canonical_hash(roots):
    errors.append("final roots content hash mismatch")
if final["content_hashes"]["generations"] != canonical_hash(generations):
    errors.append("final generations content hash mismatch")


def group_stats(items):
    return {
        "questions": len(items),
        "percentage": len(items) / 500,
        "em": mean(row["mean_em"] for row in items),
        "f1": mean(row["mean_f1"] for row in items),
        "pass_at_2": mean(row["any_em"] for row in items),
        "both_em": mean(row["both_em"] for row in items),
        "mean_k": mean(row["k"] for row in items),
        "mean_context_tokens": mean(row["context_tokens"] for row in items),
        "mean_total_tokens": mean(row["mean_total_tokens"] for row in items),
        "mean_fact_recall": mean(row["fact_recall"] for row in items),
        "facts_complete_rate": mean(row["facts_complete"] for row in items),
        "answer_present_rate": mean(row["answer_present"] for row in items),
    }


grouped = {}
for name in ("retrieval_title_miss", "adaptive_k_title_cut", "selected_titles_complete"):
    grouped[name] = group_stats([row for row in per_question if row["evidence_class"] == name])

candidate_complete_rows = [row for row in per_question if row["candidate_complete"]]
cut_rows = [row for row in candidate_complete_rows if not row["selected_complete"]]
deficits = [row["deficit"] for row in cut_rows if row["deficit"] is not None]
buffer_coverage = {}
for extra in (0, 5, 10, 20, 50, 100):
    buffer_coverage[str(extra)] = mean(
        (row["max_support_rank"] is not None and row["max_support_rank"] <= row["k"] + extra)
        for row in candidate_complete_rows
    )

facts_complete = [row for row in per_question if row["facts_complete"]]
facts_incomplete = [row for row in per_question if not row["facts_complete"]]
answer_present = [row for row in per_question if row["answer_present"]]
answer_absent = [row for row in per_question if not row["answer_present"]]

format_stats = {
    "valid": sum(row["answer_status"] == "valid" for row in generations),
    "stop": sum(row["finish_type"] == "stop" for row in generations),
    "length": sum(row["finish_type"] == "length" for row in generations),
    "think_closed": sum(row.get("had_think_open") and row.get("had_think_close") for row in generations),
    "leading_markdown_marker": sum(str(row["final_answer"]).lstrip().startswith("**") for row in generations),
    "completion_tokens": {
        "mean": mean(int(row["model_meta"]["completion_tokens"]) for row in generations),
        "p50": percentile([int(row["model_meta"]["completion_tokens"]) for row in generations], 0.5),
        "p90": percentile([int(row["model_meta"]["completion_tokens"]) for row in generations], 0.9),
        "p99": percentile([int(row["model_meta"]["completion_tokens"]) for row in generations], 0.99),
        "max": max(int(row["model_meta"]["completion_tokens"]) for row in generations),
    },
}

report = {
    "passed": not errors,
    "errors": errors,
    "integrity": {
        "questions": len(questions),
        "labels": len(labels),
        "roots": len(roots),
        "generations": len(generations),
        "metric_mismatches": metric_mismatches,
    },
    "quality": {
        "em": mean(row["mean_em"] for row in per_question),
        "f1": mean(row["mean_f1"] for row in per_question),
        "pass_at_2": mean(row["any_em"] for row in per_question),
        "both_em": mean(row["both_em"] for row in per_question),
        "same_answer_rate": mean(row["same_answer"] for row in per_question),
    },
    "title_normalization_audit": {
        "legacy_full_selected": legacy_full_selected,
        "unicode_robust_full_selected": robust_full_selected,
        "legacy_under_count": robust_full_selected - legacy_full_selected,
    },
    "evidence_groups": grouped,
    "lexical_fact_groups": {
        "facts_complete": group_stats(facts_complete),
        "facts_incomplete": group_stats(facts_incomplete),
        "gold_answer_present": group_stats(answer_present),
        "gold_answer_absent_or_yes_no": group_stats(answer_absent),
    },
    "cutoff_diagnostics": {
        "candidate_titles_complete_questions": len(candidate_complete_rows),
        "cutoff_lost_at_least_one_title_questions": len(cut_rows),
        "deficit_median": percentile(deficits, 0.5),
        "deficit_p75": percentile(deficits, 0.75),
        "deficit_p90": percentile(deficits, 0.9),
        "deficit_max": max(deficits) if deficits else 0,
        "candidate_complete_coverage_if_extra_docs_added": buffer_coverage,
    },
    "format": format_stats,
    "cost": {
        "total_input_tokens": sum(int(row["model_meta"]["prompt_tokens"]) for row in generations),
        "total_output_tokens": sum(int(row["model_meta"]["completion_tokens"]) for row in generations),
        "total_llm_tokens": sum(
            int(row["model_meta"]["prompt_tokens"]) + int(row["model_meta"]["completion_tokens"])
            for row in generations
        ),
        "mean_retrieval_latency_seconds": mean(row["retrieval_latency_seconds"] for row in roots),
        "sum_retrieval_latency_seconds": sum(row["retrieval_latency_seconds"] for row in roots),
        "sum_generation_latency_seconds": sum(row["model_latency_seconds"] for row in generations),
    },
    "by_type": {
        name: group_stats([row for row in per_question if row["type"] == name])
        for name in sorted({row["type"] for row in per_question})
    },
}
rendered = json.dumps(report, ensure_ascii=False, indent=2)
(RUN / "analysis" / "INDEPENDENT_AUDIT.json").write_text(
    rendered + "\n", encoding="utf-8"
)
print(rendered)
raise SystemExit(0 if not errors else 1)
