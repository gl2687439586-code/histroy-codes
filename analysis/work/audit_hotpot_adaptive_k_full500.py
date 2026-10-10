"""Independent read-only audit of the uploaded full500 archive."""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import math
import shutil
import statistics
import sys
import tarfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = Path("C:/Users/RESEARCH_USER/Desktop/过程结果/full500_uploaded_r1_t0_review.tar.gz")
DEST = ROOT / "outputs/hotpot_adaptive_k_full500_audit_20261002"
RUN = DEST / "full500_uploaded_r1_t0"
DELIVERED = ROOT / "hotpot_adaptive_k_budget_full500_v1/scripts/experiments/hotpot_adaptive_k_budget_full500_v1"
CODE = DEST / "experiment_code"


def sha(path):
    with Path(path).open("rb") as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def iter_rows(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def save(name, value):
    (DEST / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract():
    DEST.mkdir(parents=True, exist_ok=True)
    marker = DEST / "archive_inventory.json"
    archive_hash = sha(ARCHIVE)
    if marker.exists() and read_json(marker)["archive_sha256"] == archive_hash:
        print("Using already verified local extraction", flush=True)
        return
    inventory = []
    total = 0
    seen = set()
    with tarfile.open(ARCHIVE, "r|gz") as tar:
        for member in tar:
            name = PurePosixPath(member.name)
            if name.is_absolute() or ".." in name.parts or "\\" in member.name or member.name in seen:
                raise ValueError("Unsafe/duplicate archive path: " + member.name)
            seen.add(member.name)
            if not member.isfile():
                if member.isdir():
                    continue
                raise ValueError("Links/devices not permitted in audit archive")
            total += member.size
            if total > 2 * 1024**3:
                raise ValueError("Archive exceeds 2GiB extraction guard")
            target = DEST.joinpath(*name.parts)
            if not target.resolve().is_relative_to(DEST.resolve()):
                raise ValueError("Archive path escaped intended audit directory")
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open("wb") as handle:
                shutil.copyfileobj(source, handle, length=1024 * 1024)
            inventory.append({"name": member.name, "bytes": member.size})
    save("archive_inventory.json", {"archive_sha256": archive_hash, "archive_bytes": ARCHIVE.stat().st_size,
        "extracted_bytes": total, "files": inventory})
    print(f"Extracted {len(inventory)} regular files, {total:,} bytes", flush=True)


def describe(values):
    values = sorted(values)
    def quantile(p):
        pos = (len(values) - 1) * p
        low, high = math.floor(pos), math.ceil(pos)
        return values[low] + (values[high] - values[low]) * (pos - low)
    return {"n": len(values), "mean": statistics.mean(values), "median": quantile(.5),
        "p90": quantile(.9), "p95": quantile(.95), "min": values[0], "max": values[-1]}


def main():
    extract()
    # Never execute code from the attachment: use only the locally delivered auditor.
    sys.path.insert(0, str(DELIVERED))
    sys.path.insert(0, str(ROOT / "work/_deps"))
    from tokenizers import Tokenizer
    from audit_gate import audit_records
    from actions import parse_action
    from hp_common import answer_metrics, canonical_hash, normalize_text, sha256_text, SafeSGLangClient
    from adaptive_k import select_adaptive_k
    import bootstrap
    questions = list(iter_rows(DELIVERED / "data/questions.no_labels.jsonl"))
    labels = {r["sample_id"]: r for r in iter_rows(DELIVERED / "data/gold_labels.sealed.jsonl")}
    qmap = {q["sample_id"]: q for q in questions}
    errors, warnings, checks = [], [], Counter()
    def check(value, name):
        checks["checked"] += 1
        if not value:
            errors.append(name)
    archived_frozen = read_json(CODE / "data/sample_manifest.json")
    check(archived_frozen == read_json(DELIVERED / "data/sample_manifest.json"), "Frozen selection manifest changed")
    identity = {}
    for file in DELIVERED.rglob("*"):
        if file.is_file() and "__pycache__" not in file.parts:
            relative = file.relative_to(DELIVERED)
            other = CODE / relative
            match = other.exists() and sha(file) == sha(other)
            check(match, "Delivered file differs: " + str(relative))
            identity[str(relative)] = match
    print("Frozen data/code identity checked", flush=True)
    service = read_json(RUN / "service_preflight.json")
    check(service["passed"], "Service preflight failed")
    model_settings = []
    for spec in service["models"]:
        model_settings.append({"gpu": spec["gpu"], "port": spec["port"], "safe_pool": spec["safe_pool"],
            "model_info": spec["model_info"], "cmdline": spec["cmdline"]})
        check(spec["generation_probe_passed"], "Missing generation probe")
        check("qwen3-8b" in json.dumps(spec["model_info"]).lower(), "Wrong model identity")
    check([(s["gpu"], s["port"]) for s in service["models"]] == [(2, 9103), (3, 9104)], "GPU/port mapping changed")
    check(service["server_max_total_tokens"] == 24497, "Token pool guard differs")
    retriever = service["retriever"]
    check(retriever["port"] == 6009 and retriever["config"]["retrieval_topk"] == 1000,
          "Wrong retrieval endpoint or candidate count")
    check(not retriever["faiss_gpu"] and "wiki18_100w" in retriever["corpus_path"], "Wrong index/corpus setup")
    manifests, statuses, physical = [], [], []
    record_hashes, root_hashes = {}, {}
    for shard in range(2):
        d = RUN / "trajectories/shards" / f"shard_{shard}"
        status = read_json(d / "run_status.json")
        statuses.append(status)
        check(status["complete"] and status["records"] == status["expected_records"] == 250 and status["unresolved_failures"] == 0,
            f"Incomplete shard {shard}")
        manifest = read_json(d / "manifest.json")
        manifests.append(manifest)
        check(manifest["signature_hash"] == canonical_hash(manifest["signature"]), f"Signature hash mismatch shard {shard}")
        check(manifest["signature"]["questions_sha256"] == sha(DELIVERED / "data/questions.no_labels.jsonl"),
            f"Wrong question file hash shard {shard}")
        for name, digest in manifest["signature"]["code_sha256"].items():
            check(sha(CODE / name) == digest, f"Recorded code hash mismatch {name}")
        for name, digest in manifest["signature"]["vendor_sha256"].items():
            check(sha(CODE / "vendor_v433" / name) == digest, f"Recorded vendor hash mismatch {name}")
        expected_ids = {q["sample_id"] for q in questions[shard::2]}
        observed_ids = []
        for row in iter_rows(d / "records.jsonl.gz"):
            sid = row["sample_id"]
            observed_ids.append(sid)
            check(sid not in record_hashes, f"Duplicate shard record {sid}")
            record_hashes[sid] = row["record_sha256"]
        check(len(observed_ids) == 250 and set(observed_ids) == expected_ids, f"Wrong shard allocation {shard}")
        observed_ids = []
        for root in iter_rows(RUN / "roots/shards" / f"shard_{shard}" / "records.jsonl.gz"):
            sid = root["sample_id"]
            observed_ids.append(sid)
            check(sid not in root_hashes, f"Duplicate shard root {sid}")
            root_hashes[sid] = root["root_checkpoint_sha256"]
        check(len(observed_ids) == 250 and set(observed_ids) == expected_ids, f"Wrong roots shard allocation {shard}")
        physical.extend(iter_rows(d / "physical_model_requests.jsonl"))
    config = manifests[0]["signature"]["settings"]
    check({k:v for k,v in config.items() if k != "shard_id"} ==
        {k:v for k,v in manifests[1]["signature"]["settings"].items() if k != "shard_id"}, "Shard configuration mismatch")
    check({k:config[k] for k in ("candidate_count", "ignore_extreme_tail", "retrieve_more", "max_searches",
        "context_length", "server_max_total_tokens", "action_max_new_tokens", "answer_max_new_tokens", "num_shards")} == {
        "candidate_count": 1000, "ignore_extreme_tail": .1, "retrieve_more": 5, "max_searches": 4,
        "context_length": 40960, "server_max_total_tokens": 24497,
        "action_max_new_tokens": 8192, "answer_max_new_tokens": 8192, "num_shards": 2}, "Settings differ from approved Gate")
    tok_path = ROOT / "outputs/hotpot_adaptive_k_gate5_audit_20261002/tokenizer.json"
    rust = Tokenizer.from_file(str(tok_path))
    class LocalCounter:
        def count(self, text):
            return len(rust.encode(str(text or ""), add_special_tokens=False).ids)
    tokenizer = LocalCounter()
    # Join roots and trajectories one at a time, avoiding retention of duplicated 1000-candidate arrays.
    roots_iter = iter_rows(RUN / "roots/roots.merged.jsonl.gz")
    rows_iter = iter_rows(RUN / "trajectories/trajectories.merged.jsonl.gz")
    per_question, selection_rows, length_events = [], [], []
    required, ids = Counter(), []
    search_hist, k_hist, finish_hist, status_hist = Counter(), Counter(), Counter(), Counter()
    root_finish_hist, final_finish_hist = Counter(), Counter()
    delimiter_repairs, terminal_parse_forms = Counter(), Counter()
    sums = Counter()
    for index in range(500):
        try:
            root, row = next(roots_iter), next(rows_iter)
        except StopIteration:
            errors.append("Merged file has fewer than 500 records")
            break
        sid = row["sample_id"]
        ids.append(sid)
        check(sid == questions[index]["sample_id"] == root["sample_id"], f"Merged question order/ID mismatch {index}")
        check(row["record_sha256"] == record_hashes.get(sid), f"Merged record differs from shard {sid}")
        check(root["root_checkpoint_sha256"] == root_hashes.get(sid), f"Merged root differs from shard {sid}")
        err, warn = audit_records([row], [root], [qmap[sid]], tokenizer, config)
        checks["trajectory_replay_audited"] += 1
        errors.extend(err); warnings.extend(warn)
        check(row["sample_index"] == index == root["sample_index"], f"Wrong sample index {sid}")
        check(root["question"] == qmap[sid]["question"], f"Wrong root question {sid}")
        check(row["max_searches"] == 4 and row["evidence_exposure_protocol"] == "per_search_adaptive_k_complete_passages_once",
              f"Wrong protocol {sid}")
        check(row["usage"]["format_repair_calls"] == 0 and row["usage"]["standalone_model_calls"] == row["search_calls"] + 1,
              f"Unexpected generation/repair calls {sid}")
        label = labels[sid]
        extracted = answer_metrics(row["final_answer"], label["gold_answers"])
        metric = {key: 0.0 if row["answer_status"] == "local_fallback" else extracted[key] for key in ("em", "f1", "acc")}
        attempts = row["root_action_attempts"] + [a for e in row["decision_events"] for a in e["attempts"]]
        root_finish_hist.update([attempts[0]["finish_type"]])
        final_finish_hist.update([attempts[-1]["finish_type"]])
        local_total = 0
        for stage, attempt in enumerate(attempts):
            required.update([canonical_hash(attempt["prompt"])])
            finish_hist.update([attempt["finish_type"]])
            meta = attempt["server_meta_info"]
            event = None if stage == 0 else row["decision_events"][stage - 1]
            allowed = ["search"] if stage == 0 else (["answer"] if event["forced_answer"] else ["search", "answer"])
            rebuilt_output, rebuilt_repair = SafeSGLangClient._complete_stop(
                attempt["raw_output"], attempt["finish_reason"], [f"</{name}>" for name in allowed])
            check(rebuilt_output == attempt["output"] and rebuilt_repair == attempt["stop_boundary_repair"],
                  f"Stop-delimiter reconstruction mismatch {sid}/{stage}")
            check(sha256_text(attempt["prompt"]) == attempt["prompt_sha256"] and
                  sha256_text(attempt["output"]) == attempt["output_sha256"], f"Prompt/output raw hash mismatch {sid}/{stage}")
            parsed = parse_action(attempt["output"], allowed)
            if stage == 0:
                wanted_query = parsed["payload"] if parsed else row["question"]
                check(root["query"] == row["search_rounds"][0]["query"] == wanted_query,
                    f"Root query parsing/fallback mismatch {sid}")
            else:
                check(event["action"] == (parsed["action"] if parsed else "") and
                      event["payload"] == (parsed["payload"] if parsed else ""),
                      f"Decision parsing mismatch {sid}/{stage}")
            delimiter_repairs.update([rebuilt_repair])
            if stage == len(attempts) - 1:
                terminal_parse_forms.update(["closed_answer" if "</answer>" in attempt["output"].lower()
                    else "no_closed_answer"])
            check(attempt["input_tokens"] == meta["prompt_tokens"] and attempt["output_tokens"] == meta["completion_tokens"],
                f"Server token accounting mismatch {sid}/{stage}")
            check(attempt["input_tokens_local"] == tokenizer.count(attempt["prompt"]), f"Local prompt token mismatch {sid}/{stage}")
            local_total += attempt["input_tokens_local"] + attempt["output_tokens_local_with_stop_completion"]
            if attempt["finish_type"] == "length":
                length_events.append({"sample_id":sid, "stage":"root" if stage == 0 else "decision",
                    "decision_index":stage, "output_tokens":attempt["output_tokens"],
                    "action":attempt.get("action"), "answer_status":row["answer_status"],
                    "raw_tail":attempt["raw_output"][-500:]})
        titles = set()
        for rnd in row["search_rounds"]:
            ev = rnd["evidence"]
            scores = ev["candidate_scores"]
            check(all(math.isfinite(x) for x in scores) and all(a >= b for a,b in zip(scores,scores[1:])),
                f"Invalid/non-sorted candidate scores {sid}/{rnd['search_round']}")
            check(len({m["id"] for m in ev["candidate_metadata"]}) == 1000,
                f"Duplicate retrieved candidate IDs {sid}/{rnd['search_round']}")
            k_hist.update([ev["selected_k"]])
            titles.update(normalize_text(d["title"]) for d in ev["retrieved_docs"])
            selection_rows.append({"sample_id":sid,"round":rnd["search_round"],"k":ev["selected_k"],
                "evidence_tokens":ev["visible_evidence_tokens"],"top_score":scores[0],
                "selection":ev["selection"],"query":rnd["query"]})
            sums["retrieval_cache_hit_rounds"] += rnd["retrieval_cache_hit"]
        gold_titles = {normalize_text(t) for t in label["supporting_titles"]}
        title_recall = len(titles & gold_titles) / len(gold_titles)
        status_hist.update([row["answer_status"]]); search_hist.update([row["search_calls"]])
        for key, value in metric.items():
            sums[key] += value
        sums["input_tokens"] += row["usage"]["standalone_llm_input_tokens"]
        sums["output_tokens"] += row["usage"]["standalone_llm_output_tokens"]
        sums["root_input_tokens"] += attempts[0]["input_tokens"]
        sums["root_output_tokens"] += attempts[0]["output_tokens"]
        sums["local_total_tokens"] += local_total
        sums["context_compaction_records"] += row["context_compactions"] > 0
        sums["repeated_query_records"] += row["repeated_query_count"] > 0
        root_fallback = row["root_query_parse_mode"] == "fallback_question_as_query"
        sums["root_query_fallback_records"] += root_fallback
        per_question.append({"sample_id":sid,"question":row["question"],"gold_answers":label["gold_answers"],
            "final_answer":row["final_answer"],"answer_status":row["answer_status"],
            "em":metric["em"],"f1":metric["f1"],"acc_contains":metric["acc"],
            "recovered_answer_metrics":extracted,"search_calls":row["search_calls"],
            "k_by_round":row["selected_k_by_round"],"evidence_tokens":row["visible_evidence_tokens_total"],
            "total_tokens":row["usage"]["standalone_total_tokens"],"local_total_tokens":local_total,
            "root_query_fallback":root_fallback,"truncated_generation_count":row["truncated_generation_count"],
            "final_finish_type":attempts[-1]["finish_type"],"title_recall":title_recall})
        if (index + 1) % 100 == 0:
            print(f"Independently replay-audited {index + 1}/500 trajectories", flush=True)
    check(next(roots_iter, None) is None and next(rows_iter, None) is None, "Merged files have extra rows")
    check(len(ids) == len(set(ids)) == 500 and set(ids) == set(qmap), "Merged IDs/count mismatch")
    observed = Counter(e["prompt_sha256"] for e in physical if e["success"])
    check(not (required - observed), "Missing physical request logs for successful generations")
    primary = {"answer_em":sums["em"] / 500,"answer_f1":sums["f1"] / 500,"answer_acc_contains":sums["acc"] / 500,
        "mean_total_tokens":(sums["input_tokens"] + sums["output_tokens"]) / 500,
        "mean_search_calls":sum(q["search_calls"] for q in per_question) / 500}
    archived_summary = read_json(RUN / "analysis/summary.json")
    for key, value in primary.items():
        check(abs(archived_summary["overall"][key] - value) < 1e-10, "Archived metric mismatch " + key)
    archived_scores = {r["sample_id"]:r for r in iter_rows(RUN / "analysis/scored_records.jsonl.gz")}
    check(len(archived_scores) == 500, "Wrong archived scored-record count")
    for row in per_question:
        scored = archived_scores[row["sample_id"]]
        check(abs(scored["EM"] - row["em"]) < 1e-10 and abs(scored["F1"] - row["f1"]) < 1e-10 and
              abs(scored["ACC_contains"] - row["acc_contains"]) < 1e-10 and scored["total_tokens"] == row["total_tokens"] and
              abs(scored["selected_support_title_recall"] - row["title_recall"]) < 1e-10,
              "Archived question metrics mismatch " + row["sample_id"])
    for name in ("roots/roots.merged.jsonl.gz.manifest.json", "trajectories/trajectories.merged.jsonl.gz.manifest.json"):
        manifest = read_json(RUN / name)
        check(manifest["passed"] and not manifest["errors"], "Merged manifest reports failure " + name)
    groups = {}
    for dimension in ("search_calls", "answer_status", "root_query_fallback", "final_finish_type"):
        buckets = defaultdict(list)
        for row in per_question:
            buckets[str(row[dimension])].append(row)
        groups[dimension] = {key:{"n":len(rows),"em":statistics.mean(r["em"] for r in rows),
            "f1":statistics.mean(r["f1"] for r in rows),"acc_contains":statistics.mean(r["acc_contains"] for r in rows),
            "mean_total_tokens":statistics.mean(r["total_tokens"] for r in rows),
            "mean_evidence_tokens":statistics.mean(r["evidence_tokens"] for r in rows),
            "mean_title_recall":statistics.mean(r["title_recall"] for r in rows)} for key,rows in buckets.items()}
    report = {"passed":not errors,"errors":errors,"warnings":warnings,"checks":dict(checks),
        "records":len(ids),"unique_samples":len(set(ids)),"settings":config,"model_services":model_settings,
        "retriever":retriever,"code_and_data_identity":identity,"statuses":statuses,
        "overall":primary,"search_distribution":dict(search_hist),"k_distribution":dict(k_hist),
        "per_search_k":describe([r["k"] for r in selection_rows]),
        "per_search_evidence_tokens":describe([r["evidence_tokens"] for r in selection_rows]),
        "per_question_evidence_tokens":describe([r["evidence_tokens"] for r in per_question]),
        "per_question_total_tokens":describe([r["total_tokens"] for r in per_question]),
        "answer_status_distribution":dict(status_hist),"generation_finish_distribution":dict(finish_hist),
        "root_finish_distribution":dict(root_finish_hist),"final_finish_distribution":dict(final_finish_hist),
        "stop_delimiter_completion_counts":dict(delimiter_repairs),
        "terminal_answer_tag_forms":dict(terminal_parse_forms),
        "length_records":sum(r["truncated_generation_count"] > 0 for r in per_question),
        "sums":dict(sums),"groups":groups,
        "physical_requests":{"all":len(physical),"successful":sum(e["success"] for e in physical),
            "failed":sum(not e["success"] for e in physical),"expected_successful_calls":sum(required.values()),
            "orphan_successes":sum((observed - required).values()),
            "successful_input_tokens":sum(e.get("meta_info",{}).get("prompt_tokens",0) for e in physical if e["success"]),
            "successful_output_tokens":sum(e.get("meta_info",{}).get("completion_tokens",0) for e in physical if e["success"])},
        "evidence_metric_scope":"title recall only; not proof of supporting sentence/token coverage",
        "limitations":["Full model weight/index digests are not present; identity checked via model path/process/config, not full binary hashes.",
            "ACC is normalized gold containment, not semantic correctness or an LLM judge.",
            "No per-question cumulative hard budget was imposed; this is an adaptive evidence-selection baseline.",
            "New first-search roots were generated; old FULL is not a strict shared-root causal comparison."]}
    save("INDEPENDENT_AUDIT.json", report)
    save("per_question.json", per_question)
    save("per_search_selection.json", selection_rows)
    save("length_events.json", length_events)
    print(json.dumps({"passed":report["passed"],"error_count":len(errors),"overall":primary,
        "search_distribution":dict(search_hist),"k_distribution":dict(k_hist),
        "answer_status":dict(status_hist),"root_fallback":sums["root_query_fallback_records"],
        "generation_finish":dict(finish_hist),"physical_requests":report["physical_requests"],
        "groups":groups},ensure_ascii=False,indent=2),flush=True)
    if errors:
        raise RuntimeError("Independent audit failed; see INDEPENDENT_AUDIT.json")


if __name__ == "__main__":
    main()
