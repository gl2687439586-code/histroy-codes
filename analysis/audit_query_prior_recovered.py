import hashlib
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def pearson(xs, ys):
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    den = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    return sum(x * y for x, y in zip(dx, dy)) / den if den else None


def rank(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    pos = 0
    while pos < len(order):
        end = pos + 1
        while end < len(order) and values[order[end]] == values[order[pos]]:
            end += 1
        mean_rank = (pos + 1 + end) / 2
        for j in range(pos, end):
            ranks[order[j]] = mean_rank
        pos = end
    return ranks


def main():
    run = Path(sys.argv[1])
    manifest = read_json(run / "manifest.json")
    catalogue = read_json(run / "model_catalogue.json")
    combined = read_jsonl(run / "results" / "combined_scores.jsonl")
    read_rows = read_jsonl(run / "results" / "read_scores.jsonl")
    search_rows = read_jsonl(run / "results" / "search_scores.jsonl")
    raw_rows = read_jsonl(run / "raw" / "successful_responses.jsonl")
    attempts = read_jsonl(run / "api_attempts.jsonl")

    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    check(manifest.get("question_count") == 100, "manifest question_count != 100")
    check(manifest.get("expected_logical_ratings") == 200, "expected ratings != 200")
    check(manifest.get("completed_logical_ratings") == 200, "completed ratings != 200")
    check(manifest.get("complete_question_pairs") == 100, "complete pairs != 100")
    check(manifest.get("missing_tasks") == [], "missing tasks remain")
    check(len(read_rows) == 100, "read rows != 100")
    check(len(search_rows) == 100, "search rows != 100")
    check(len(combined) == 100, "combined rows != 100")
    check(len(raw_rows) == 200, "raw successful rows != 200")
    check(len(attempts) == 215, "physical attempts != 215")

    ids = [row.get("question_id") for row in combined]
    check(len(set(ids)) == 100, "combined question IDs are not unique")
    read_ids = [row.get("question_id") for row in read_rows]
    search_ids = [row.get("question_id") for row in search_rows]
    check(set(read_ids) == set(ids), "read ID set differs from combined")
    check(set(search_ids) == set(ids), "search ID set differs from combined")

    successful_task_keys = [row.get("task_key") for row in raw_rows]
    check(len(set(successful_task_keys)) == 200, "accepted logical task keys are duplicated")
    expected_keys = {f"{i:04d}:{d}" for i in range(100) for d in ("read", "search")}
    check(set(successful_task_keys) == expected_keys, "accepted task-key set is incomplete")

    models = sorted({row.get("response_model") for row in raw_rows})
    check(models == ["openai/gpt-5.6-sol"], f"unexpected response models: {models}")
    check(manifest.get("returned_models") == models, "manifest returned_models mismatch")
    check(catalogue.get("id") == "openai/gpt-5.6-sol", "catalogue model mismatch")

    bad_values = []
    bad_query_pairs = []
    combined_by_id = {row["question_id"]: row for row in combined}
    for row in combined:
        for dimension in ("read", "search"):
            score = row.get(f"{dimension}_score")
            confidence = row.get(f"{dimension}_confidence")
            basis = row.get(f"{dimension}_basis")
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
                bad_values.append((row["question_id"], dimension, "score", score))
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
                bad_values.append((row["question_id"], dimension, "confidence", confidence))
            if not isinstance(basis, str) or not basis.strip() or len(basis.split()) >= 25:
                bad_values.append((row["question_id"], dimension, "basis", basis))
    check(not bad_values, f"invalid rating fields: {bad_values[:5]}")

    for rows, dimension in ((read_rows, "read"), (search_rows, "search")):
        for row in rows:
            parent = combined_by_id.get(row["question_id"])
            if not parent or row.get("query") != parent.get("query"):
                bad_query_pairs.append((row.get("question_id"), dimension))
    check(not bad_query_pairs, f"query text mismatch: {bad_query_pairs[:5]}")

    attempt_ok = Counter(bool(row.get("ok")) for row in attempts)
    attempt_errors = Counter()
    failed_elapsed = []
    for row in attempts:
        if not row.get("ok"):
            err = str(row.get("error", ""))
            if "SSLError" in err or "UNEXPECTED_EOF_WHILE_READING" in err:
                attempt_errors["tls_ssl_eof"] += 1
            else:
                attempt_errors["other"] += 1
            if isinstance(row.get("elapsed_seconds"), (int, float)):
                failed_elapsed.append(row["elapsed_seconds"])
    check(attempt_ok[True] == 200, f"successful physical attempts != 200: {attempt_ok[True]}")
    check(attempt_ok[False] == 15, f"failed physical attempts != 15: {attempt_ok[False]}")
    check(attempt_errors.get("other", 0) == 0, f"non-TLS attempt failures: {attempt_errors}")

    request_hashes = defaultdict(set)
    prompt_hashes = defaultdict(set)
    for row in raw_rows:
        request_hashes[row["task_key"]].add(row.get("request_sha256"))
        prompt_hashes[row["task_key"]].add(row.get("prompt_sha256"))
    check(all(len(x) == 1 for x in request_hashes.values()), "accepted request hash ambiguity")
    check(all(len(x) == 1 for x in prompt_hashes.values()), "accepted prompt hash ambiguity")

    rs = [row["read_score"] for row in combined]
    ss = [row["search_score"] for row in combined]
    read_dist = Counter(rs)
    search_dist = Counter(ss)
    joint = Counter(zip(rs, ss))
    paired = {
        "read_gt_search": sum(r > s for r, s in zip(rs, ss)),
        "read_eq_search": sum(r == s for r, s in zip(rs, ss)),
        "read_lt_search": sum(r < s for r, s in zip(rs, ss)),
        "pearson": pearson(rs, ss),
        "spearman": pearson(rank(rs), rank(ss)),
    }

    usage = manifest.get("usage", {}).get("all", {})
    validation_errors = manifest.get("validation_errors", [])
    strict_only_error = (
        len(validation_errors) == 1
        and "logged 215 physical generation requests" in validation_errors[0]
        and "strict target is 200" in validation_errors[0]
    )
    check(strict_only_error, f"unexpected manifest validation errors: {validation_errors}")

    report = {
        "archive_run": str(run),
        "sha256_manifest": hashlib.sha256((run / "manifest.json").read_bytes()).hexdigest(),
        "passed_data_integrity": not errors,
        "strict_one_shot_protocol_passed": False,
        "strict_one_shot_success_rate": 185 / 200,
        "recovered_dataset_complete": len(raw_rows) == 200 and len(combined) == 100,
        "recovery": {
            "physical_attempts": len(attempts),
            "successful_attempts": attempt_ok[True],
            "failed_attempts": attempt_ok[False],
            "recovery_requests": len(attempts) - 200,
            "overhead_rate": (len(attempts) - 200) / 200,
            "failed_error_classes": dict(attempt_errors),
            "mean_failed_elapsed_seconds": statistics.mean(failed_elapsed) if failed_elapsed else None,
        },
        "model": {
            "requested": manifest.get("requested_model"),
            "returned": models,
            "catalogue_id": catalogue.get("id"),
            "canonical_slug": catalogue.get("canonical_slug"),
        },
        "protocol": {
            "reasoning_effort": manifest.get("reasoning_effort"),
            "seed": manifest.get("seed"),
            "one_query_per_request": manifest.get("one_query_per_request"),
            "dimensions_per_question": manifest.get("dimensions_per_question"),
        },
        "ratings": {
            "questions": len(combined),
            "logical_ratings": len(raw_rows),
            "read_mean": statistics.mean(rs),
            "search_mean": statistics.mean(ss),
            "read_distribution": {str(k): read_dist.get(k, 0) for k in range(5)},
            "search_distribution": {str(k): search_dist.get(k, 0) for k in range(5)},
            "mean_read_confidence": statistics.mean(row["read_confidence"] for row in combined),
            "mean_search_confidence": statistics.mean(row["search_confidence"] for row in combined),
            "paired": paired,
            "joint_distribution": {f"read_{r}_search_{s}": joint.get((r, s), 0) for r in range(5) for s in range(5)},
        },
        "usage_successful_responses_only": usage,
        "manifest_passed": manifest.get("passed"),
        "manifest_validation_errors": validation_errors,
        "errors": errors,
    }
    output = run.parent / "independent_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
