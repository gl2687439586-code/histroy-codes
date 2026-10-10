"""Three fresh full-pipeline repetitions around unchanged v4.3.3 generation."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import full7405 as infrastructure
from prepare_repeat500 import load_queries

HERE = Path(__file__).resolve().parent
REPETITIONS = 3
QUESTIONS = 500
PROTOCOL = {**infrastructure.PROTOCOL, "questions": QUESTIONS,
            "repetitions": REPETITIONS, "repeat_mode": "fresh_greedy_full_pipeline",
            "between_repeats": "sequential; separate roots, records and retrieval caches",
            "within_repeat": "all budgets of one question share only that repeat's first-search root"}
CODE_FILES = infrastructure.CORE_FILES + [
    "repeat500.py", "repeat500.sh", "analyze_repeats.py", "prepare_repeat500.py",
    "export_repeats.py",
]


def settings():
    cfg = infrastructure.settings()
    workspace = cfg["workspace"]
    cfg.update({
        "batch_dir": Path(os.environ.get("BATCH_DIR", str(workspace / "outputs/hotpot_budget_repeat500_v433/repeat500_20261004"))).resolve(),
        "service_dir": Path(os.environ.get("SERVICE_RUN_DIR", str(workspace / "outputs/hotpot_budget_repeat500_v433/services_g2g3"))).resolve(),
        "session": os.environ.get("SESSION_NAME", "hotpot_repeat500_v433"),
        "model_session_prefix": "hotpot_repeat500_model",
        "retriever_session": "hotpot_repeat500_retriever",
        "query_file": Path(os.environ.get("QUERY_FILE", str(HERE / "data/hotpot_random500_queries.jsonl"))).resolve(),
        "questions": HERE / "data/questions.no_labels.jsonl",
        "labels": HERE / "data/gold_labels.sealed.jsonl",
    })
    return cfg


def code_hashes():
    return {name: infrastructure.digest(HERE / name) for name in CODE_FILES}


def validate_inputs(cfg):
    queries = load_queries(cfg["query_file"])
    infrastructure.read_dataset(cfg["questions"], cfg["labels"], expected=QUESTIONS)
    rows = [json.loads(line) for line in cfg["questions"].read_text(encoding="utf-8").splitlines() if line.strip()]
    wanted = [(str(row["query_id"]), row["query"]) for row in queries]
    actual = [(str(row["sample_id"]), row["question"]) for row in rows]
    if wanted != actual:
        raise RuntimeError("Uploaded 500 query IDs, texts or order differ from bundled dataset")
    manifest = infrastructure.read_json(HERE / "data/INPUT_MANIFEST.json")
    for key in ("questions", "labels"):
        if infrastructure.digest(cfg[key]) != manifest["files"][cfg[key].name]:
            raise RuntimeError(f"Bundled {key} changed after official-data verification")
    return rows


def validate_paths(cfg):
    probe = dict(cfg, run_dir=cfg["batch_dir"])
    infrastructure.bound_run_dir(probe)
    output_root = (cfg["workspace"] / "outputs").resolve()
    if not cfg["service_dir"].is_relative_to(output_root):
        raise RuntimeError("SERVICE_RUN_DIR must stay below smartsearch_rebuild/outputs")


def plan(cfg):
    validate_paths(cfg)
    rows = validate_inputs(cfg)
    for key in ("core_python", "model_python", "retriever_config"):
        if not cfg[key].is_file():
            raise RuntimeError(f"Missing {key}: {cfg[key]}")
    model_config = infrastructure.read_json(cfg["model_path"] / "config.json")
    if model_config.get("model_type") != "qwen3" or int(model_config.get("max_position_embeddings", 0)) < 40960:
        raise RuntimeError("Expected Qwen3 model supporting 40960 positions")
    from hp_common import ExactTokenizer
    from prompts import initial_search_prompt
    tokenizer = ExactTokenizer(str(cfg["model_path"]))
    maximum = max(tokenizer.count(initial_search_prompt(row["question"])) for row in rows)
    if maximum + 8192 > PROTOCOL["client_token_limit"]:
        raise RuntimeError("Root prompt plus output allowance exceeds preserved token capacity")
    return {
        "created_at": infrastructure.stamp(), "protocol": PROTOCOL,
        "query_file_sha256": infrastructure.digest(cfg["query_file"]),
        "questions_sha256": infrastructure.digest(cfg["questions"]),
        "labels_sha256": infrastructure.digest(cfg["labels"]),
        "package_hashes": code_hashes(), "model_path": str(cfg["model_path"]),
        "model_config_sha256": infrastructure.digest(cfg["model_path"] / "config.json"),
        "retriever_config_sha256": infrastructure.digest(cfg["retriever_config"]),
        "core_library_versions": {name: importlib.metadata.version(name) for name in ("transformers", "numpy", "requests")},
        "batch_dir": str(cfg["batch_dir"]), "max_root_prompt_tokens": maximum,
        "first_search_roots": 1500, "estimated_terminal_records_from_old500": 12540,
        "estimate_note": "Exact counts are determined independently from each repeat's generated roots",
        "acc_definition": "original normalized contains; not LLM Judge",
        "determinism_note": "temperature=0; repeats are fresh executions, not independent random samples",
    }


def run_cfg(cfg, repetition):
    return dict(cfg, run_dir=cfg["batch_dir"] / f"repeat_{repetition}")


def compare_plan(original, current):
    fields = ("protocol", "query_file_sha256", "questions_sha256", "labels_sha256", "package_hashes",
              "model_path", "model_config_sha256", "retriever_config_sha256", "core_library_versions")
    changes = [key for key in fields if original.get(key) != current.get(key)]
    if changes:
        raise RuntimeError(f"Resume configuration differs: {changes}; existing results preserved")


def verify_frozen(cfg):
    original = infrastructure.read_json(cfg["batch_dir"] / "BATCH_CONFIG.json")
    if original["package_hashes"] != code_hashes():
        raise RuntimeError("Installed experiment code changed after batch started")
    for repetition in range(1, REPETITIONS + 1):
        run = run_cfg(cfg, repetition)["run_dir"]
        record = infrastructure.read_json(run / "REPEAT_CONFIG.json")
        if record["repetition"] != repetition or record["batch_config"] != original:
            raise RuntimeError(f"Unexpected repeat configuration: {run}")
        for name, field in (("questions.no_labels.jsonl", "questions_sha256"),
                            ("gold_labels.sealed.jsonl", "labels_sha256")):
            if infrastructure.digest(run / "dataset" / name) != original[field]:
                raise RuntimeError(f"Frozen dataset changed: {run / 'dataset' / name}")
        if (run / "retrieval_cache.sqlite").is_symlink():
            raise RuntimeError(f"Repeat cache must not be a shared symlink: {run}")
    return original


def explicit_env(cfg):
    # The tmux server can predate the new terminal. Propagate only runtime values.
    return {"WORKSPACE_ROOT": str(cfg["workspace"]), "BATCH_DIR": str(cfg["batch_dir"]),
            "SERVICE_RUN_DIR": str(cfg["service_dir"]), "SESSION_NAME": cfg["session"],
            "QUERY_FILE": str(cfg["query_file"]), "PYTHONUNBUFFERED": "1"}


def launch(cfg, resume):
    validate_paths(cfg)
    if infrastructure.tmux_alive(cfg["session"]):
        raise RuntimeError(f"Batch tmux {cfg['session']} is already running")
    if not resume:
        infrastructure.require_empty_run(cfg["batch_dir"])
    current = plan(cfg)
    if resume:
        compare_plan(infrastructure.read_json(cfg["batch_dir"] / "BATCH_CONFIG.json"), current)
        verify_frozen(cfg)
    print(json.dumps(current, ensure_ascii=False, indent=2), flush=True)
    checks = infrastructure.services(cfg, create=True)
    if not resume:
        infrastructure.require_empty_run(cfg["batch_dir"])
        cfg["batch_dir"].mkdir(parents=True, exist_ok=True)
        infrastructure.write_json(cfg["batch_dir"] / "BATCH_CONFIG.json", current)
        for repetition in range(1, REPETITIONS + 1):
            run = run_cfg(cfg, repetition)["run_dir"]
            (run / "dataset").mkdir(parents=True)
            for key in ("questions", "labels"):
                shutil.copyfile(cfg[key], run / "dataset" / cfg[key].name)
            infrastructure.write_json(run / "REPEAT_CONFIG.json", {"repetition": repetition, "batch_config": current})
    infrastructure.write_json(cfg["batch_dir"] / "last_service_check.json", checks)
    args = [cfg["core_python"], "-u", HERE / "repeat500.py", "worker"]
    if resume:
        args.append("--retry-failures")
    infrastructure.tmux_start(cfg["session"], args, cfg["batch_dir"] / "tmux_logs/coordinator.log",
                              cfg["repo"], explicit_env(cfg))
    print(f"Three fresh repetitions dispatched sequentially: {cfg['batch_dir']}", flush=True)


def completed(run):
    status = run / "PIPELINE_STATUS.json"
    return status.is_file() and infrastructure.read_json(status).get("stage") == "complete"


def worker(cfg, retry):
    verify_frozen(cfg)
    started = time.time()
    active = None
    stage = "failed"
    try:
        for repetition in range(1, REPETITIONS + 1):
            active = repetition
            single = run_cfg(cfg, repetition)
            run = single["run_dir"]
            if completed(run):
                print(f"[repeat-skip] repeat_{repetition} already complete", flush=True)
                continue
            # Refuse to proceed if services have died or changed between runs.
            checks = infrastructure.services(cfg, create=False)
            infrastructure.write_json(run / "last_service_check.json", checks)
            infrastructure.write_json(cfg["batch_dir"] / "BATCH_STATUS.json", {
                "stage": "running", "active_repetition": repetition,
                "started_at": infrastructure.stamp(), "retry_failures": retry})
            infrastructure.write_json(run / "PIPELINE_STATUS.json", {
                "stage": "running", "started_at": infrastructure.stamp()})
            env = infrastructure.env_for_run(single)
            env["RETRY_FAILURES"] = "1" if retry else "0"
            print(f"[repeat-start] {repetition}/{REPETITIONS} run={run}; fresh run-local roots and cache", flush=True)
            repetition_started = time.time()
            code = 1
            try:
                code = subprocess.call(["bash", str(HERE / "orchestrate.sh"), "all"], cwd=cfg["repo"], env=env)
                if code:
                    raise RuntimeError(f"repeat_{repetition} pipeline exited {code}; later repeats were not started")
                code = 1
                infrastructure.write_json(run / "PIPELINE_STATUS.json", {"stage": "token_summary"})
                code = subprocess.call([str(cfg["core_python"]), str(HERE / "summarize_full_tokens.py"),
                                        "--run-dir", str(run), "--expected-questions", str(QUESTIONS)],
                                       cwd=cfg["repo"], env=env)
                if code:
                    raise RuntimeError(f"repeat_{repetition} token summary exited {code}")
            finally:
                infrastructure.write_json(run / "PIPELINE_STATUS.json", {
                    "stage": "complete" if code == 0 else "failed", "exit_code": code,
                    "finished_at": infrastructure.stamp(),
                    "elapsed_seconds_this_invocation": time.time() - repetition_started})
            print(f"[repeat-complete] {repetition}/{REPETITIONS}", flush=True)
        infrastructure.command([cfg["core_python"], HERE / "analyze_repeats.py", "--batch-dir", cfg["batch_dir"]])
        stage = "complete"
        return 0
    finally:
        infrastructure.write_json(cfg["batch_dir"] / "BATCH_STATUS.json", {
            "stage": stage, "active_repetition": active,
            "finished_at": infrastructure.stamp(), "elapsed_seconds_this_invocation": time.time() - started})


def status(cfg):
    runs = []
    # snapshot's ETA stage decision uses this global; no generation code reads it.
    infrastructure.EXPECTED_QUESTIONS = QUESTIONS
    for repetition in range(1, REPETITIONS + 1):
        snap = infrastructure.snapshot(run_cfg(cfg, repetition))
        snap.update({"repetition": repetition, "expected_roots": QUESTIONS})
        runs.append(snap)
    path = cfg["batch_dir"] / "BATCH_STATUS.json"
    return {"batch_dir": str(cfg["batch_dir"]),
            "batch_status": infrastructure.read_json(path) if path.exists() else {"stage": "not_started"},
            "repetitions": runs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "self-test", "services", "check", "start", "resume", "status", "worker", "analyze", "export"])
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    cfg = settings()
    if args.action == "plan":
        print(json.dumps(plan(cfg), ensure_ascii=False, indent=2))
    elif args.action == "self-test":
        validate_inputs(cfg)
        env = dict(os.environ, TOP_K="5")
        for name in ("self_test.py", "test_repeat500.py"):
            infrastructure.command([sys.executable, HERE / name], env=env)
    elif args.action in ("services", "check"):
        print(json.dumps(infrastructure.services(cfg, create=args.action == "services"), ensure_ascii=False, indent=2))
    elif args.action in ("start", "resume"):
        launch(cfg, resume=args.action == "resume")
    elif args.action == "status":
        print(json.dumps(status(cfg), ensure_ascii=False, indent=2))
    elif args.action == "worker":
        return worker(cfg, args.retry_failures)
    elif args.action == "analyze":
        verify_frozen(cfg)
        infrastructure.command([cfg["core_python"], HERE / "analyze_repeats.py", "--batch-dir", cfg["batch_dir"]])
    elif args.action == "export":
        # A finalized review archive; never mutate ongoing record stores.
        if infrastructure.tmux_alive(cfg["session"]):
            raise RuntimeError("Export only after the batch tmux finishes")
        if any(not completed(run_cfg(cfg, repetition)["run_dir"]) for repetition in range(1, REPETITIONS + 1)):
            raise RuntimeError("All three repetitions must finish before final export")
        from export_repeats import export_review
        print(export_review(cfg["batch_dir"]))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[repeat500-error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
