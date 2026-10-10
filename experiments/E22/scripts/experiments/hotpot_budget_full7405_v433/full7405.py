#!/usr/bin/env python3
"""Full dev-set operations around the preserved v4.3.3 scientific protocol.

No SSH or local credential discovery. This script runs on the experiment server.
It does not stop processes, reuse previous records, or change model prompts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import socket
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPECTED_QUESTIONS = 7405
PROTOCOL = {
    "source_protocol": "v4.3.3",
    "split": "HotpotQA dev-distractor questions; open-domain wiki retrieval",
    "questions": EXPECTED_QUESTIONS,
    "fresh_generation": True,
    "top_k": 5,
    "exposure": "100-token global prefix caps below first-search length, plus FULL per search",
    "max_searches_including_root": 4,
    "temperature": 0.0,
    "native_thinking": True,
    "action_max_new_tokens": 8192,
    "answer_max_new_tokens": 8192,
    "repair_max_new_tokens": 0,
    "context_length": 40960,
    "client_token_limit": 24497,
    "model_mem_fraction": 0.84,
    "max_running_requests": 2,
    "gpu_ids": [2, 3],
    "model_ports": [9103, 9104],
    "retriever_port": 6008,
    "workers_per_gpu": 2,
    "shards": 4,
}
CORE_FILES = [
    "actions.py", "analyze_answer_flips.py", "analyze_hotpotqa.py", "build_roots.py",
    "hp_common.py", "locate_supporting_evidence.py", "merge_roots.py",
    "merge_trajectories.py", "orchestrate.sh", "prompts.py", "retrieval.py",
    "run_trajectories.py", "full7405.py", "summarize_full_tokens.py",
]


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def settings():
    workspace = Path(os.environ.get("WORKSPACE_ROOT", "/home/researcher/smartsearch_rebuild")).resolve()
    repo = workspace / "SmartSearch"
    return {
        "workspace": workspace,
        "repo": repo,
        "core_python": workspace / "envs/src-core/bin/python",
        "model_python": workspace / "envs/src-model/bin/python",
        "model_path": workspace / "models/Qwen3-8B",
        "questions": workspace / "data/hotpotqa/prepared/questions.no_labels.jsonl",
        "labels": workspace / "data/hotpotqa/prepared/gold_labels.sealed.jsonl",
        "retriever_config": repo / "scripts/serving/retriever_config_src_action_tree.yaml",
        "service_dir": Path(os.environ.get("SERVICE_RUN_DIR", str(workspace / "outputs/hotpot_budget_full7405_v433/services_g2g3"))).resolve(),
        "run_dir": Path(os.environ.get("RUN_DIR", str(workspace / "outputs/hotpot_budget_full7405_v433/full7405_fresh_v433_20260924"))).resolve(),
        "session": os.environ.get("SESSION_NAME", "hotpot_budget_full7405_v433"),
    }


def command(args, **kwargs):
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def env_for_run(cfg):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    # Ignore stale experiment variables in the user's newly opened terminal.
    env.update({
        "PYTHONUNBUFFERED": "1", "TOP_K": "5", "LIMIT": "0", "EXPOSURES": "",
        "MAX_SEARCHES": "4", "TEMPERATURE": "0.0", "CONTEXT_LENGTH": "40960",
        "SERVER_MAX_TOTAL_TOKENS": "24497", "ACTION_MAX_NEW_TOKENS": "8192",
        "ANSWER_MAX_NEW_TOKENS": "8192", "REPAIR_MAX_NEW_TOKENS": "0",
        "NUM_GPUS": "2", "GPU_IDS": "2,3", "BASE_PORT": "9103",
        "WORKERS_PER_GPU": "2", "MAX_RUNNING_REQUESTS": "2", "TIMEOUT": "900",
        "RESOURCE_INTERVAL_SECONDS": "300", "PYTHON_BIN": str(cfg["core_python"]),
        "MODEL_PATH": str(cfg["model_path"]), "RUN_DIR": str(cfg["run_dir"]),
        "QUESTIONS": str(cfg["run_dir"] / "dataset/questions.no_labels.jsonl"),
        "LABELS": str(cfg["run_dir"] / "dataset/gold_labels.sealed.jsonl"),
        "RETRIEVER_URL": "http://127.0.0.1:6008",
        "RETRIEVAL_CACHE": str(cfg["run_dir"] / "retrieval_cache.sqlite"),
        "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1",
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    env["PATH"] = str(cfg["model_python"].parent) + os.pathsep + env.get("PATH", "")
    return env


def read_dataset(questions, labels, expected=EXPECTED_QUESTIONS):
    qrows = [json.loads(x) for x in Path(questions).read_text(encoding="utf-8").splitlines() if x.strip()]
    lrows = [json.loads(x) for x in Path(labels).read_text(encoding="utf-8").splitlines() if x.strip()]
    ids = [str(x["sample_id"]) for x in qrows]
    label_ids = [str(x["sample_id"]) for x in lrows]
    if len(ids) != expected or len(set(ids)) != expected:
        raise RuntimeError(f"Expected {expected} unique full-dev questions; records={len(ids)}, unique={len(set(ids))}. Refusing a 500-question subset.")
    if len(label_ids) != expected or set(label_ids) != set(ids):
        raise RuntimeError("Question and sealed-label ID sets differ or labels contain duplicates")
    if any(not str(x.get("question", "")).strip() for x in qrows):
        raise RuntimeError("Empty question in input")
    forbidden = {"answer", "gold_answers", "supporting_facts", "supporting_titles", "context"}
    if any(forbidden.intersection(x) for x in qrows):
        raise RuntimeError("Generation questions contain evidence or gold-label fields")
    if len({int(x["sample_index"]) for x in qrows}) != expected:
        raise RuntimeError("Duplicate sample_index in questions")
    return qrows


def package_hashes():
    return {name: digest(HERE / name) for name in CORE_FILES}


def local_plan(cfg):
    for key in ("questions", "labels", "core_python", "model_python", "retriever_config"):
        if not cfg[key].is_file():
            raise RuntimeError(f"Missing {key}: {cfg[key]}")
    if not cfg["model_path"].is_dir():
        raise RuntimeError(f"Missing model: {cfg['model_path']}")
    rows = read_dataset(cfg["questions"], cfg["labels"])
    model_cfg = read_json(cfg["model_path"] / "config.json")
    if model_cfg.get("model_type") != "qwen3":
        raise RuntimeError("Local model config is not qwen3")
    if int(model_cfg.get("max_position_embeddings", 0)) < 40960:
        raise RuntimeError("Local model max_position_embeddings is below 40960")
    from hp_common import ExactTokenizer
    from prompts import initial_search_prompt
    tokenizer = ExactTokenizer(str(cfg["model_path"]))
    longest_root_prompt = max(tokenizer.count(initial_search_prompt(row["question"])) for row in rows)
    if longest_root_prompt + 8192 > PROTOCOL["client_token_limit"]:
        raise RuntimeError("An initial prompt plus 8192 exceeds the preserved token limit")
    versions = {}
    for name in ("transformers", "numpy", "requests"):
        versions[name] = importlib.metadata.version(name)
    parent = cfg["run_dir"].parent
    while not parent.exists():
        parent = parent.parent
    return {
        "created_at": stamp(), "protocol": PROTOCOL,
        "run_dir": str(cfg["run_dir"]), "package_hashes": package_hashes(),
        "questions_sha256": digest(cfg["questions"]), "labels_sha256": digest(cfg["labels"]),
        "model_path": str(cfg["model_path"]), "model_config_sha256": digest(cfg["model_path"] / "config.json"),
        "retriever_config_sha256": digest(cfg["retriever_config"]),
        "retriever_config_path": str(cfg["retriever_config"]),
        "core_library_versions": versions, "max_root_prompt_tokens": longest_root_prompt,
        "estimated_trajectories_from_old500": round(4180 * EXPECTED_QUESTIONS / 500),
        "exact_trajectory_count": "available after all first-search roots finish",
        "free_disk_gib": round(shutil.disk_usage(parent).free / (1024 ** 3), 2),
        "data_freshness": "all 7405 roots and continuations regenerated in a new run-local cache",
    }


def bound_run_dir(cfg):
    run = cfg["run_dir"]
    outputs = cfg["workspace"] / "outputs"
    if outputs not in run.parents or run == outputs:
        raise RuntimeError(f"RUN_DIR must be a dedicated subdirectory of {outputs}")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", cfg["session"]):
        raise RuntimeError("SESSION_NAME may contain only letters, digits, underscore and hyphen")


def require_empty_run(run):
    if run.exists() and any(run.iterdir()):
        raise RuntimeError(f"RUN_DIR is not empty: {run}. Use resume for this full run, or choose a new RUN_DIR. Old records are never imported.")


def tmux_alive(name):
    return subprocess.run(["tmux", "has-session", "-t", "=" + name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def tmux_start(name, args, log, cwd, env):
    if tmux_alive(name):
        raise RuntimeError(f"tmux session {name} already exists; inspect it before restarting")
    log.parent.mkdir(parents=True, exist_ok=True)
    # Pass a fully quoted env command: tmux server may predate this terminal.
    assignments = [f"{k}={v}" for k, v in env.items()]
    shell = "exec " + shlex.join(["env", "-u", "PYTHONPATH"] + assignments + [str(x) for x in args])
    shell += " >>" + shlex.quote(str(log)) + " 2>&1"
    command(["tmux", "new-session", "-d", "-s", name, "-c", cwd, shell])
    print(f"Started tmux {name}; log={log}", flush=True)


def http(port, endpoint, payload=None, timeout=20):
    import requests
    with requests.Session() as session:
        session.trust_env = False
        url = f"http://127.0.0.1:{port}{endpoint}"
        response = session.get(url, timeout=(3, timeout)) if payload is None else session.post(url, json=payload, timeout=(3, timeout))
        response.raise_for_status()
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.text


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def process_for_port(port, module):
    matches = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            argv = path.read_bytes().decode().rstrip("\0").split("\0")
            if not any(module in arg for arg in argv):
                continue
            opts = {}
            for index, arg in enumerate(argv):
                if arg.startswith("--"):
                    if "=" in arg:
                        key, value = arg.split("=", 1)
                        opts[key] = value
                    else:
                        opts[arg] = argv[index + 1] if index + 1 < len(argv) and not argv[index + 1].startswith("--") else True
            if str(opts.get("--port")) != str(port):
                continue
            proc_env = dict(item.split("=", 1) for item in (path.parent / "environ").read_bytes().decode().split("\0") if "=" in item)
            matches.append((int(path.parent.name), argv, opts, proc_env))
        except (PermissionError, FileNotFoundError, UnicodeError):
            continue
    if len(matches) != 1:
        raise RuntimeError(f"Expected one visible {module} process on {port}; found {len(matches)}")
    return matches[0]


def tail_text(path, max_bytes=256000):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(0, size - max_bytes))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def recursive_pools(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "max_total_num_tokens" and isinstance(item, (int, float)) and item > 0:
                yield int(item)
            else:
                yield from recursive_pools(item)
    elif isinstance(value, list):
        for item in value:
            yield from recursive_pools(item)


def model_probe(cfg, port, gpu):
    http(port, "/health", timeout=10)
    info = http(port, "/get_model_info", timeout=10)
    if not isinstance(info, dict) or str(info.get("model_path", "")).rstrip("/") != str(cfg["model_path"]):
        raise RuntimeError(f"Port {port}: model identity differs: {info}")
    pid, argv, opts, penv = process_for_port(port, "sglang.launch_server")
    required = {"--context-length": "40960", "--dtype": "bfloat16", "--tp": "1", "--dp": "1",
                "--attention-backend": "triton", "--sampling-backend": "pytorch",
                "--chunked-prefill-size": "4096", "--max-running-requests": "2"}
    for key, wanted in required.items():
        if str(opts.get(key)) != wanted:
            raise RuntimeError(f"Port {port} PID {pid}: {key}={opts.get(key)!r}; expected {wanted}. Existing process retained. Command: {shlex.join(argv)}")
    for flag in ("--disable-cuda-graph", "--disable-overlap", "--trust-remote-code"):
        if flag not in opts:
            raise RuntimeError(f"Port {port} is missing {flag}; existing process retained")
    if "--disable-radix-cache" in opts:
        raise RuntimeError(f"Port {port}: radix cache disabled; v4.3.3 used radix cache")
    if "--enable-torch-compile" in opts:
        raise RuntimeError(f"Port {port}: torch.compile was not enabled in the preserved launch configuration")
    if abs(float(opts.get("--mem-fraction-static", 0)) - 0.84) > 1e-6:
        raise RuntimeError(f"Port {port}: expected --mem-fraction-static 0.84")
    if penv.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError(f"Port {port}: physical GPU mapping is {penv.get('CUDA_VISIBLE_DEVICES')}, expected {gpu}")
    pools = []
    pool_source = ""
    try:
        pools = list(recursive_pools(http(port, "/get_server_info", timeout=10)))
        if pools:
            pool_source = "live get_server_info"
    except Exception:
        pass
    if not pools:
        # Read only the current PID's regular stdout/stderr file, never a pipe or TTY.
        for fd in ("1", "2"):
            try:
                stdout_path = Path(os.readlink(f"/proc/{pid}/fd/{fd}"))
                if stdout_path.is_file():
                    matches = re.findall(r"max_total_num_tokens\s*=\s*(\d+)", tail_text(stdout_path, 4 * 1024 * 1024))
                    if matches:
                        pools = [int(matches[-1])]
                        pool_source = str(stdout_path)
                        break
            except OSError:
                pass
    if not pools:
        explicit = os.environ.get(f"OBSERVED_POOL_{port}")
        if explicit:
            pools = [int(explicit)]
            pool_source = f"explicit OBSERVED_POOL_{port} supplied from current server logs"
    if not pools:
        raise RuntimeError(f"Cannot verify current token pool on {port}. Read its startup max_total_num_tokens and set OBSERVED_POOL_{port}=that_value. Do not assume the historical 24497 value.")
    if min(pools) < PROTOCOL["client_token_limit"]:
        raise RuntimeError(f"Port {port}: observed pool {min(pools)} < preserved client limit 24497. No experiment started.")
    print(f"Port {port}: GPU{gpu}, context=40960, observed pool={min(pools)}. Running real generation probe...", flush=True)
    probe = http(port, "/generate", {"text": "<|im_start|>user\nReply with OK.<|im_end|>\n<|im_start|>assistant\n",
                                    "sampling_params": {"temperature": 0.0, "max_new_tokens": 16}}, timeout=90)
    if not isinstance(probe, dict) or not isinstance(probe.get("text"), str) or not probe["text"].strip():
        raise RuntimeError(f"Port {port}: real generation produced no text")
    return {"port": port, "gpu": gpu, "pid": pid, "model_info": info,
            "observed_token_pool": min(pools), "pool_source": pool_source,
            "probe_finish_reason": probe.get("meta_info", {}).get("finish_reason"), "passed": True}


def retriever_probe(cfg):
    http(6008, "/health", timeout=10)
    pid, _, opts, _ = process_for_port(6008, "retriever_serving.py")
    supplied = Path(str(opts.get("--config", "")))
    if not supplied.is_absolute():
        supplied = Path(os.readlink(f"/proc/{pid}/cwd")) / supplied
    if not supplied.is_file() or digest(supplied) != digest(cfg["retriever_config"]):
        raise RuntimeError("Retriever on 6008 is not using the expected configuration")
    docs = http(6008, "/search", {"query": "HotpotQA evidence retrieval", "top_n": 5, "return_score": False}, timeout=120)
    if isinstance(docs, dict):
        docs = docs.get("documents") or docs.get("result") or docs.get("data")
    if not isinstance(docs, list) or len(docs) != 5 or any(not x.get("contents") for x in docs):
        raise RuntimeError("Retriever actual search did not return five nonempty contents")
    print("Retriever 6008: health and real Top-5 search passed", flush=True)
    return {"port": 6008, "pid": pid, "passed": True, "documents": 5, "config_sha256": digest(supplied)}


def wait_for_health(port, session, log, seconds):
    deadline = time.monotonic() + seconds
    last_message = 0.0
    while time.monotonic() < deadline:
        if not tmux_alive(session):
            raise RuntimeError(f"Session {session} exited during initialization. Log {log}:\n{tail_text(log, 8000)}")
        try:
            http(port, "/health", timeout=3)
            return
        except Exception:
            pass
        if time.monotonic() - last_message >= 30:
            print(f"Waiting for {session} port={port}; log={log}", flush=True)
            last_message = time.monotonic()
        time.sleep(5)
    raise RuntimeError(f"Timed out waiting for {session}; inspect {log}")


def service_env(cfg, gpu):
    # Do not copy secrets into tmux command strings; pass only explicit runtime values.
    env = {"CUDA_VISIBLE_DEVICES": str(gpu), "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "PATH": str(cfg["model_python"].parent) + os.pathsep + os.environ.get("PATH", ""),
           "PYTHONUNBUFFERED": "1", "NO_PROXY": "127.0.0.1,localhost,::1", "no_proxy": "127.0.0.1,localhost,::1"}
    for key in ("HF_HOME", "TMPDIR", "TORCH_EXTENSIONS_DIR", "TRITON_CACHE_DIR"):
        location = cfg["workspace"] / "cache" / key.lower()
        location.mkdir(parents=True, exist_ok=True)
        env[key] = str(location)
    return env


def services(cfg, create):
    if sys.platform != "linux":
        raise RuntimeError("Service operations must be run on the Ubuntu experiment server")
    if not shutil.which("tmux"):
        raise RuntimeError("tmux is not installed")
    logs = cfg["service_dir"] / "tmux_logs"
    results = []
    for gpu, port in zip(PROTOCOL["gpu_ids"], PROTOCOL["model_ports"]):
        if not listening(port):
            if not create:
                raise RuntimeError(f"No service listening on {port}; run full7405.sh services")
            free = int(command(["nvidia-smi", "-i", gpu, "--query-gpu=memory.free", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip())
            if free < 22000:
                raise RuntimeError(f"GPU{gpu} has only {free} MiB free and port {port} is unavailable. Existing GPU processes were not stopped.")
            name = f"hotpot_full7405_model_g{gpu}"
            log = logs / f"model_g{gpu}.log"
            args = [cfg["model_python"], "-u", "-m", "sglang.launch_server", "--served-model-name", "Qwen3-8B",
                    "--model-path", cfg["model_path"], "--tp", "1", "--dp", "1", "--context-length", "40960",
                    "--dtype", "bfloat16", "--host", "127.0.0.1", "--port", str(port), "--trust-remote-code",
                    "--attention-backend", "triton", "--sampling-backend", "pytorch", "--disable-cuda-graph",
                    "--disable-overlap", "--chunked-prefill-size", "4096", "--max-running-requests", "2",
                    "--mem-fraction-static", "0.84"]
            tmux_start(name, args, log, cfg["repo"], service_env(cfg, gpu))
            wait_for_health(port, name, log, 1200)
        # An occupied but unhealthy or incompatible endpoint is never replaced.
        results.append(model_probe(cfg, port, gpu))
    if not listening(6008):
        if not create:
            raise RuntimeError("Retriever 6008 unavailable; run full7405.sh services")
        name = "hotpot_full7405_retriever"
        log = logs / "retriever.log"
        args = [cfg["core_python"], "-u", cfg["repo"] / "scripts/serving/retriever_serving.py",
                "--config", cfg["retriever_config"], "--num_retriever", "1", "--port", "6008"]
        tmux_start(name, args, log, cfg["repo"], service_env(cfg, 2))
        wait_for_health(6008, name, log, 2400)
    report = {"checked_at": stamp(), "models": results, "retriever": retriever_probe(cfg)}
    write_json(cfg["service_dir"] / "service_check.json", report)
    return report


def assert_resume(cfg, plan):
    manifest = cfg["run_dir"] / "FULL7405_CONFIG.json"
    if not manifest.is_file():
        raise RuntimeError(f"No full-run configuration at {manifest}; refusing to resume another experiment")
    original = read_json(manifest)
    keys = ["protocol", "package_hashes", "questions_sha256", "labels_sha256", "model_path",
            "model_config_sha256", "retriever_config_sha256", "core_library_versions"]
    differences = [key for key in keys if original.get(key) != plan.get(key)]
    if differences:
        raise RuntimeError(f"Resume configuration changed: {differences}. Do not mix settings in this RUN_DIR.")
    for name, key in (("questions.no_labels.jsonl", "questions_sha256"), ("gold_labels.sealed.jsonl", "labels_sha256")):
        if digest(cfg["run_dir"] / "dataset" / name) != original[key]:
            raise RuntimeError(f"Frozen dataset changed: {name}")


def launch(cfg, resume=False):
    bound_run_dir(cfg)
    if shutil.which("tmux") and tmux_alive(cfg["session"]):
        raise RuntimeError(f"Experiment session {cfg['session']} is already running")
    if not resume:
        require_empty_run(cfg["run_dir"])
    plan = local_plan(cfg)
    if resume:
        assert_resume(cfg, plan)
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    checks = services(cfg, create=True)
    run = cfg["run_dir"]
    if not resume:
        require_empty_run(run)
        (run / "dataset").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cfg["questions"], run / "dataset/questions.no_labels.jsonl")
        shutil.copyfile(cfg["labels"], run / "dataset/gold_labels.sealed.jsonl")
        write_json(run / "FULL7405_CONFIG.json", plan)
    write_json(run / "last_service_check.json", checks)
    args = [cfg["core_python"], "-u", HERE / "full7405.py", "worker"]
    if resume:
        args.append("--retry-failures")
    explicit = {"WORKSPACE_ROOT": str(cfg["workspace"]), "RUN_DIR": str(run),
                "SERVICE_RUN_DIR": str(cfg["service_dir"]), "SESSION_NAME": cfg["session"], "PYTHONUNBUFFERED": "1"}
    tmux_start(cfg["session"], args, run / "tmux_logs/coordinator.log", cfg["repo"], explicit)
    print(f"Full 7405 run dispatched. Progress: bash {HERE / 'full7405.sh'} status", flush=True)


def snapshot(cfg):
    run = cfg["run_dir"]
    result = {"run_dir": str(run), "roots": 0, "trajectories": 0,
              "root_failures": 0, "trajectory_failures": 0, "expected_roots": EXPECTED_QUESTIONS,
              "expected_trajectories": None, "stage": "not_started"}
    for stage, count_key, failure_key in (("roots", "roots", "root_failures"), ("trajectories", "trajectories", "trajectory_failures")):
        for db in sorted((run / stage / "shards").glob("shard_*/state.sqlite")):
            try:
                with sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=2) as conn:
                    result[count_key] += conn.execute("SELECT COUNT(*) FROM records").fetchone()[0]
                    result[failure_key] += conn.execute("SELECT COUNT(*) FROM failures").fetchone()[0]
            except sqlite3.Error as exc:
                result.setdefault("read_errors", []).append(str(exc))
    manifests = list((run / "trajectories/shards").glob("shard_*/manifest.json"))
    if len(manifests) == 4:
        result["expected_trajectories"] = sum(read_json(p)["expected_records"] for p in manifests)
    stage_file = run / "PIPELINE_STATUS.json"
    if stage_file.exists():
        result.update({"stage": read_json(stage_file).get("stage"), "last_pipeline_status": read_json(stage_file)})
    if shutil.which("tmux"):
        result["tmux_running"] = tmux_alive(cfg["session"])
    etas = []
    current = "root" if result["roots"] < EXPECTED_QUESTIONS else "trajectory"
    for log in (run / "tmux_logs").glob(current + "_worker_*.log"):
        values = re.findall(r"eta=([0-9.]+)h", tail_text(log, 16000))
        if values:
            etas.append(float(values[-1]))
    if etas:
        result["stage_eta_hours_last_worker_estimates"] = max(etas)
        result["eta_note"] = "Last worker log estimate; may be stale if workers have stopped. Analysis time excluded."
    return result


def worker(cfg, retry):
    run = cfg["run_dir"]
    plan = read_json(run / "FULL7405_CONFIG.json")
    if package_hashes() != plan["package_hashes"]:
        raise RuntimeError("Installed package changed after plan was frozen")
    for name, key in (("questions.no_labels.jsonl", "questions_sha256"), ("gold_labels.sealed.jsonl", "labels_sha256")):
        if digest(run / "dataset" / name) != plan[key]:
            raise RuntimeError(f"Frozen dataset changed before execution: {name}")
    env = env_for_run(cfg)
    env["RETRY_FAILURES"] = "1" if retry else "0"
    write_json(run / "PIPELINE_STATUS.json", {"stage": "running", "started_at": stamp(), "retry_failures": retry})
    started = time.time()
    code = 1
    try:
        code = subprocess.call(["bash", str(HERE / "orchestrate.sh"), "all"], cwd=cfg["repo"], env=env)
        if code:
            return code
        code = 1  # An exception while starting offline analysis must not mark completion.
        write_json(run / "PIPELINE_STATUS.json", {"stage": "token_summary", "updated_at": stamp()})
        code = subprocess.call([str(cfg["core_python"]), str(HERE / "summarize_full_tokens.py"), "--run-dir", str(run), "--expected-questions", str(EXPECTED_QUESTIONS)], cwd=cfg["repo"], env=env)
        return code
    finally:
        write_json(run / "PIPELINE_STATUS.json", {"stage": "complete" if code == 0 else "failed",
                   "finished_at": stamp(), "exit_code": code, "elapsed_seconds_this_invocation": time.time() - started})
        print(f"Pipeline exit={code}. Results: {run}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["plan", "self-test", "prepare", "check", "services", "start", "resume", "status", "worker"])
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()
    cfg = settings()
    if args.action == "self-test":
        env = dict(os.environ, TOP_K="5")
        command([sys.executable, HERE / "self_test.py"], env=env)
        command([sys.executable, HERE / "test_full7405.py"], env=env)
    elif args.action == "plan":
        print(json.dumps(local_plan(cfg), ensure_ascii=False, indent=2))
    elif args.action == "prepare":
        command(["bash", HERE / "download_hotpotqa.sh"], env=dict(os.environ, PYTHON_BIN=str(cfg["core_python"]), DATA_DIR=str(cfg["workspace"] / "data/hotpotqa")))
    elif args.action in ("services", "check"):
        print(json.dumps(services(cfg, create=args.action == "services"), ensure_ascii=False, indent=2))
    elif args.action in ("start", "resume"):
        launch(cfg, resume=args.action == "resume")
    elif args.action == "status":
        print(json.dumps(snapshot(cfg), ensure_ascii=False, indent=2))
    elif args.action == "worker":
        return worker(cfg, args.retry_failures)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"[full7405-error] {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
