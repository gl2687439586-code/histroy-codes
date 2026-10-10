#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT=/home/researcher/smartsearch_rebuild
REPO_ROOT="$REBUILD_ROOT/SmartSearch"
EXP="$REPO_ROOT/scripts/experiments/plan_and_budget_hotpot_agent_v2_soft"
CORE_PY="$REBUILD_ROOT/envs/src-core/bin/python"
MODEL_PY="$REBUILD_ROOT/envs/src-model/bin/python"
QWEN_MODEL="$REBUILD_ROOT/models/Qwen3-8B"
TOKENIZER_PATH="$QWEN_MODEL"

RUN_DIR="$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/hotpot500_qwen3_llama31_soft_v2_full_v1"
PREPARED_DIR="$REBUILD_ROOT/data/plan_and_budget_hotpot_agent_v2_soft/hotpot_random500_seed2026"
QUESTIONS="$PREPARED_DIR/questions.no_labels.jsonl"
LABELS="$PREPARED_DIR/gold_labels.sealed.jsonl"
PLANS="$RUN_DIR/plans/plans.jsonl"

SERVICE_RUN_DIR="$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/dual_qwen_services_g2g3_v1"
RETRIEVER_CONFIG="$REPO_ROOT/scripts/serving/retriever_config_src_action_tree.yaml"
MODEL_MEM_FRACTION=0.80
CONTEXT_LENGTH=40960
GENERATION_MAX_TOKENS=12288
MAX_SEARCHES=4
TOP_K=5

PORT_G2=9103
PORT_G3=9104
RETRIEVER_PORT=6008
COORDINATOR_SESSION=pandb_hotpot500_dual_qwen_v1
MODEL_SESSION_G2=pandb_qwen_g2_hotpot500
MODEL_SESSION_G3=pandb_qwen_g3_hotpot500
RETRIEVER_SESSION=pandb_retriever_g2_hotpot500

LOG_DIR="$SERVICE_RUN_DIR/tmux_logs"
RUN_LOG_DIR="$RUN_DIR/tmux_logs/dual_qwen"
mkdir -p "$LOG_DIR" "$RUN_LOG_DIR"

export PYTHONPATH="$REPO_ROOT/scripts/experiments${PYTHONPATH:+:$PYTHONPATH}"

wait_http() {
  local url="$1" timeout="${2:-1200}" start now
  start=$(date +%s)
  while true; do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      return 0
    fi
    now=$(date +%s)
    if (( now - start >= timeout )); then
      echo "Timed out waiting for $url" >&2
      return 1
    fi
    sleep 3
  done
}

port_pids() {
  fuser -n tcp "$1" 2>/dev/null || true
}

stop_known_port() {
  local port="$1" kind="$2" pids pid cmd
  pids=$(port_pids "$port")
  [[ -z "$pids" ]] && return 0
  for pid in $pids; do
    cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
    case "$kind" in
      model)
        [[ "$cmd" == *"sglang.launch_server"* ]] || {
          echo "Refusing to stop unknown process on model port $port: PID=$pid CMD=$cmd" >&2
          exit 1
        }
        ;;
      retriever)
        [[ "$cmd" == *"retriever_serving.py"* || "$cmd" == *"wiki2023_retriever_service.py"* ]] || {
          echo "Refusing to stop unknown process on retriever port $port: PID=$pid CMD=$cmd" >&2
          exit 1
        }
        ;;
      *) echo "Unknown service kind: $kind" >&2; exit 1 ;;
    esac
  done
  echo "Stopping known $kind service on port $port: $pids"
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 30); do
    [[ -z "$(port_pids "$port")" ]] && return 0
    sleep 1
  done
  pids=$(port_pids "$port")
  [[ -z "$pids" ]] || kill -KILL $pids 2>/dev/null || true
  for _ in $(seq 1 10); do
    [[ -z "$(port_pids "$port")" ]] && return 0
    sleep 1
  done
  echo "Port $port did not become free" >&2
  exit 1
}

verify_inputs() {
  [[ -x "$CORE_PY" ]] || { echo "Missing core Python: $CORE_PY" >&2; exit 1; }
  [[ -x "$MODEL_PY" ]] || { echo "Missing model Python: $MODEL_PY" >&2; exit 1; }
  [[ -s "$QWEN_MODEL/config.json" ]] || { echo "Missing Qwen model: $QWEN_MODEL" >&2; exit 1; }
  [[ -s "$QUESTIONS" ]] || { echo "Missing questions: $QUESTIONS" >&2; exit 1; }
  [[ -s "$LABELS" ]] || { echo "Missing labels: $LABELS" >&2; exit 1; }
  [[ -s "$PLANS" ]] || { echo "Missing plans: $PLANS" >&2; exit 1; }
  [[ -s "$RETRIEVER_CONFIG" ]] || { echo "Missing retriever config: $RETRIEVER_CONFIG" >&2; exit 1; }

  "$CORE_PY" - "$QUESTIONS" "$PLANS" <<'PY'
import json
import sys
from pathlib import Path

def load(path):
    return [
        json.loads(line)
        for line in Path(path).open(encoding="utf-8")
        if line.strip()
    ]

questions = load(sys.argv[1])
plans = load(sys.argv[2])
question_ids = [str(row["sample_id"]) for row in questions]
plan_ids = [str(row["sample_id"]) for row in plans]
if len(question_ids) != 500 or len(set(question_ids)) != 500:
    raise SystemExit(f"questions are not 500 unique records: {len(question_ids)}")
if len(plan_ids) != 500 or len(set(plan_ids)) != 500:
    raise SystemExit(f"plans are not 500 unique records: {len(plan_ids)}")
if set(question_ids) != set(plan_ids):
    raise SystemExit("question IDs and plan IDs differ")
for row in plans:
    credits = [int(item["credit"]) for item in row["subquestions"]]
    if not credits or any(value <= 0 for value in credits) or sum(credits) != 100:
        raise SystemExit(f"invalid credits after normalization: {row['sample_id']} {credits}")
print("INPUT INTEGRITY PASSED: 500 questions, 500 plans, credits sum to 100")
PY

  grep -qF 'wiki18_100w.jsonl' "$RETRIEVER_CONFIG" || {
    echo "Retriever config does not point to wiki18_100w.jsonl" >&2
    exit 1
  }
  grep -Eq '^[[:space:]]*retrieval_topk:[[:space:]]*5[[:space:]]*$' "$RETRIEVER_CONFIG" || {
    echo "Retriever config is not Top-5" >&2
    exit 1
  }
}

start_retriever() {
  local health count log="$LOG_DIR/retriever.log"
  health=$(curl -fsS --max-time 5 "http://127.0.0.1:$RETRIEVER_PORT/health" 2>/dev/null || true)
  if [[ "$health" == *'"retrievers"'* && "$health" != *"fanoutqa"* ]]; then
    echo "Reusing verified wiki18 FlashRAG Retriever on port $RETRIEVER_PORT"
  else
    stop_known_port "$RETRIEVER_PORT" retriever
    tmux kill-session -t "$RETRIEVER_SESSION" 2>/dev/null || true
    tmux new-session -d -s "$RETRIEVER_SESSION" \
      "cd '$REPO_ROOT' && env \
CUDA_VISIBLE_DEVICES=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
'$CORE_PY' -u scripts/serving/retriever_serving.py \
--config '$RETRIEVER_CONFIG' \
--num_retriever 1 \
--port '$RETRIEVER_PORT' \
>>'$log' 2>&1"
    echo "Started wiki18 Retriever on GPU2; waiting for readiness"
    if ! wait_http "http://127.0.0.1:$RETRIEVER_PORT/health" 1800; then
      tail -n 200 "$log" >&2 || true
      exit 1
    fi
  fi

  health=$(curl -fsS --max-time 5 "http://127.0.0.1:$RETRIEVER_PORT/health")
  [[ "$health" == *'"retrievers"'* && "$health" != *"fanoutqa"* ]] || {
    echo "Port $RETRIEVER_PORT is not the expected FlashRAG Retriever: $health" >&2
    exit 1
  }
  count=$(curl -fsS --max-time 300 \
    -X POST "http://127.0.0.1:$RETRIEVER_PORT/search" \
    -H 'Content-Type: application/json' \
    -d '{"query":"Albert Einstein birthplace","top_n":5,"return_score":false}' \
    | "$CORE_PY" -c 'import json,sys; print(len(json.load(sys.stdin)))')
  [[ "$count" == 5 ]] || { echo "Retriever returned $count documents, expected 5" >&2; exit 1; }
  echo "Retriever Top-5 passed"
}

start_qwen() {
  local gpu="$1" port="$2" session="$3" log="$4" model_path
  stop_known_port "$port" model
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && env \
CUDA_VISIBLE_DEVICES='$gpu' \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
'$MODEL_PY' -u -m sglang.launch_server \
--served-model-name Qwen3-8B \
--model-path '$QWEN_MODEL' \
--tp 1 --dp 1 \
--context-length '$CONTEXT_LENGTH' \
--dtype bfloat16 \
--host 127.0.0.1 \
--port '$port' \
--trust-remote-code \
--attention-backend triton \
--sampling-backend pytorch \
--disable-cuda-graph \
--disable-overlap \
--disable-radix-cache \
--chunked-prefill-size 4096 \
--max-running-requests 1 \
--mem-fraction-static '$MODEL_MEM_FRACTION' \
>>'$log' 2>&1"
  echo "Started Qwen3-8B on physical GPU$gpu port=$port"
  if ! wait_http "http://127.0.0.1:$port/health" 1200; then
    tail -n 200 "$log" >&2 || true
    exit 1
  fi
  model_path=$(curl -fsS --max-time 10 "http://127.0.0.1:$port/get_model_info" \
    | "$CORE_PY" -c 'import json,sys; print(json.load(sys.stdin).get("model_path", ""))')
  [[ "$model_path" == "$QWEN_MODEL" ]] || {
    echo "Wrong model on port $port: $model_path" >&2
    exit 1
  }
}

run_worker() {
  local scheduler="$1" shard="$2" port="$3"
  local output="$RUN_DIR/trajectories/$scheduler/shards/shard_$shard"
  "$CORE_PY" -m plan_and_budget_hotpot_agent_v2_soft.run_trajectories \
    --questions "$QUESTIONS" \
    --plans "$PLANS" \
    --output-dir "$output" \
    --scheduler "$scheduler" \
    --model-url "http://127.0.0.1:$port" \
    --model-name Qwen3-8B \
    --expected-model-fragment Qwen3-8B \
    --retriever-url "http://127.0.0.1:$RETRIEVER_PORT" \
    --tokenizer-path "$TOKENIZER_PATH" \
    --max-searches "$MAX_SEARCHES" \
    --top-k "$TOP_K" \
    --generation-max-tokens "$GENERATION_MAX_TOKENS" \
    --rollout 0 \
    --limit 500 \
    --shard-id "$shard" \
    --num-shards 2
}

merge_scheduler() {
  local scheduler="$1"
  "$CORE_PY" - "$QUESTIONS" "$RUN_DIR" "$scheduler" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

questions_path = Path(sys.argv[1])
run_dir = Path(sys.argv[2])
scheduler = sys.argv[3]
parent = run_dir / "trajectories" / scheduler
shard_dirs = [parent / "shards" / f"shard_{i}" for i in range(2)]

def read_jsonl(path):
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8")
        if line.strip()
    ]

questions = read_jsonl(questions_path)
ordered_ids = [str(row["sample_id"]) for row in questions[:500]]
records = {}
errors = []
shard_counts = {}

for shard_id, shard_dir in enumerate(shard_dirs):
    manifest_path = shard_dir / "manifest.json"
    records_path = shard_dir / "records.jsonl"
    if not manifest_path.is_file() or not records_path.is_file():
        errors.append(f"missing shard {shard_id} output")
        continue
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(records_path)
    shard_counts[str(shard_id)] = len(rows)
    if not manifest.get("complete"):
        errors.append(f"shard {shard_id} manifest is incomplete")
    if int(manifest.get("records_total", -1)) != len(rows):
        errors.append(f"shard {shard_id} manifest count mismatch")
    for row in rows:
        sample_id = str(row["sample_id"])
        if row.get("scheduler") != scheduler:
            errors.append(f"wrong scheduler for {sample_id}")
        if sample_id in records:
            errors.append(f"duplicate sample_id {sample_id}")
        records[sample_id] = row

missing = [sample_id for sample_id in ordered_ids if sample_id not in records]
extra = sorted(set(records) - set(ordered_ids))
if missing:
    errors.append(f"missing {len(missing)} IDs: {missing[:10]}")
if extra:
    errors.append(f"extra {len(extra)} IDs: {extra[:10]}")
if len(records) != 500:
    errors.append(f"expected 500 unique records, got {len(records)}")

manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "scheduler": scheduler,
    "num_shards": 2,
    "shard_counts": shard_counts,
    "expected_records": 500,
    "records": len(records),
    "unique_ids": len(records),
    "errors": errors,
    "passed": not errors,
}
manifest_text = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
(parent / "manifest.json").write_text(manifest_text, encoding="utf-8")
(parent / "merged_manifest.json").write_text(manifest_text, encoding="utf-8")
if errors:
    raise SystemExit(json.dumps(manifest, ensure_ascii=False, indent=2))

parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix="records.", suffix=".tmp", dir=parent)
try:
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        for sample_id in ordered_ids:
            stream.write(json.dumps(records[sample_id], ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, parent / "records.jsonl")
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(json.dumps(manifest, ensure_ascii=False, indent=2))
PY
}

run_scheduler_parallel() {
  local scheduler="$1" pid0 pid1 rc0 rc1
  echo "[$(date -Is)] Starting $scheduler: two 250-question shards"
  run_worker "$scheduler" 0 "$PORT_G2" \
    >>"$RUN_LOG_DIR/${scheduler}_shard_0.log" 2>&1 &
  pid0=$!
  run_worker "$scheduler" 1 "$PORT_G3" \
    >>"$RUN_LOG_DIR/${scheduler}_shard_1.log" 2>&1 &
  pid1=$!
  echo "$pid0" >"$RUN_LOG_DIR/${scheduler}_shard_0.pid"
  echo "$pid1" >"$RUN_LOG_DIR/${scheduler}_shard_1.pid"

  set +e
  wait "$pid0"; rc0=$?
  wait "$pid1"; rc1=$?
  set -e
  if [[ "$rc0" != 0 || "$rc1" != 0 ]]; then
    echo "$scheduler worker failure: shard0=$rc0 shard1=$rc1" >&2
    tail -n 100 "$RUN_LOG_DIR/${scheduler}_shard_0.log" >&2 || true
    tail -n 100 "$RUN_LOG_DIR/${scheduler}_shard_1.log" >&2 || true
    exit 1
  fi
  merge_scheduler "$scheduler" \
    >>"$RUN_LOG_DIR/${scheduler}_merge.log" 2>&1
  echo "[$(date -Is)] Completed and merged $scheduler"
}

coordinator() {
  verify_inputs
  run_scheduler_parallel polynomial
  run_scheduler_parallel cosine
  "$CORE_PY" -m plan_and_budget_hotpot_agent_v2_soft.analyze \
    --run-dir "$RUN_DIR" \
    --labels "$LABELS" \
    --plans "$PLANS" \
    --expected 500 \
    >>"$RUN_LOG_DIR/analyze.log" 2>&1
  echo "[$(date -Is)] FULL DUAL-QWEN EXPERIMENT COMPLETED"
  echo "Report: $RUN_DIR/analysis/REPORT_ZH.md"
}

case "${1:-start}" in
  coordinator)
    coordinator
    ;;
  start)
    verify_inputs
    tmux has-session -t "$COORDINATOR_SESSION" 2>/dev/null && {
      echo "Coordinator session already exists: $COORDINATOR_SESSION" >&2
      exit 1
    }

    # Planning is complete. Retire the planner/task sessions before model replacement.
    tmux kill-session -t pandb_hotpot500_plan_only_v1 2>/dev/null || true
    tmux kill-session -t pandb_hotpot500_plan_retry7 2>/dev/null || true

    start_retriever
    start_qwen 2 "$PORT_G2" "$MODEL_SESSION_G2" "$LOG_DIR/qwen_g2.log"
    start_qwen 3 "$PORT_G3" "$MODEL_SESSION_G3" "$LOG_DIR/qwen_g3.log"

    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader
    tmux new-session -d -s "$COORDINATOR_SESSION" \
      "cd '$REPO_ROOT' && bash '$REPO_ROOT/start_pandb_hotpot500_dual_qwen.sh' coordinator >>'$RUN_LOG_DIR/coordinator.log' 2>&1"
    echo "Started dual-Qwen 500-question experiment"
    echo "Coordinator: $COORDINATOR_SESSION"
    echo "Run dir:     $RUN_DIR"
    echo "Follow: tail -f '$RUN_LOG_DIR/coordinator.log'"
    ;;
  status)
    for port in "$PORT_G2" "$PORT_G3" "$RETRIEVER_PORT"; do
      printf 'port %s: ' "$port"
      curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
        && echo healthy || echo unavailable
    done
    for scheduler in polynomial cosine; do
      for shard in 0 1; do
        file="$RUN_DIR/trajectories/$scheduler/shards/shard_$shard/records.jsonl"
        if [[ -f "$file" ]]; then
          printf '%-10s shard%s: %s/250\n' "$scheduler" "$shard" "$(wc -l <"$file")"
        else
          printf '%-10s shard%s: 0/250\n' "$scheduler" "$shard"
        fi
      done
    done
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
      --format=csv,noheader
    ;;
  *)
    echo "Usage: $0 {start|coordinator|status}" >&2
    exit 2
    ;;
esac
