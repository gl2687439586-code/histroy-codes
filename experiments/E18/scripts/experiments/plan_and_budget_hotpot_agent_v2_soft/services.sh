#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT="${REBUILD_ROOT:-/home/researcher/smartsearch_rebuild}"
REPO_ROOT="${REPO_ROOT:-$REBUILD_ROOT/SmartSearch}"
CORE_PY="${CORE_PY:-$REBUILD_ROOT/envs/src-core/bin/python}"
MODEL_PY="${MODEL_PY:-$REBUILD_ROOT/envs/src-model/bin/python}"
PLANNER_MODEL_PATH="${PLANNER_MODEL_PATH:-$REBUILD_ROOT/models/Llama-3.1-8B-Instruct}"
REASONER_MODEL_PATH="${REASONER_MODEL_PATH:-$REBUILD_ROOT/models/Qwen3-8B}"
RETRIEVER_CONFIG="${RETRIEVER_CONFIG:-$REPO_ROOT/scripts/serving/retriever_config_src_action_tree.yaml}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/services_g2g3}"

PLANNER_PORT="${PLANNER_PORT:-9103}"
REASONER_PORT="${REASONER_PORT:-9104}"
RETRIEVER_PORT="${RETRIEVER_PORT:-6008}"
PLANNER_GPU="${PLANNER_GPU:-2}"
REASONER_GPU="${REASONER_GPU:-3}"
PLANNER_CONTEXT_LENGTH="${PLANNER_CONTEXT_LENGTH:-16384}"
REASONER_CONTEXT_LENGTH="${REASONER_CONTEXT_LENGTH:-40960}"
PLANNER_MEM_FRACTION="${PLANNER_MEM_FRACTION:-0.68}"
REASONER_MEM_FRACTION="${REASONER_MEM_FRACTION:-0.84}"
TAKEOVER_KNOWN_SERVICES="${TAKEOVER_KNOWN_SERVICES:-1}"
START_FRESH="${START_FRESH:-0}"

LOG_DIR="$SERVICE_RUN_DIR/tmux_logs"
mkdir -p "$LOG_DIR"

wait_http() {
  local url="$1" timeout="${2:-900}" start now
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

known_service_pid() {
  local pid="$1" cmd
  cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
  [[ "$cmd" == *"sglang.launch_server"* || "$cmd" == *"retriever_serving.py"* ]]
}

stop_known_port() {
  local port="$1" pids pid
  pids=$(port_pids "$port")
  [[ -z "$pids" ]] && return 0
  for pid in $pids; do
    if ! known_service_pid "$pid"; then
      echo "Refusing to stop unknown PID $pid on port $port:" >&2
      tr '\0' ' ' <"/proc/$pid/cmdline" >&2 || true
      exit 1
    fi
  done
  [[ "$TAKEOVER_KNOWN_SERVICES" == "1" ]] || {
    echo "Known old service occupies port $port; set TAKEOVER_KNOWN_SERVICES=1." >&2
    exit 1
  }
  echo "Stopping known old service on port $port: $pids"
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 20); do
    [[ -z "$(port_pids "$port")" ]] && return 0
    sleep 1
  done
  pids=$(port_pids "$port")
  for pid in $pids; do
    known_service_pid "$pid" && kill -KILL "$pid" 2>/dev/null || true
  done
}

model_path_on_port() {
  curl -fsS --max-time 5 "http://127.0.0.1:$1/get_model_info" 2>/dev/null \
    | "$CORE_PY" -c 'import json,sys; print(json.load(sys.stdin).get("model_path", ""))' \
    2>/dev/null || true
}

start_model() {
  local session="$1" gpu="$2" port="$3" model_path="$4" served_name="$5" context="$6" fraction="$7" log="$8"
  local existing
  existing=$(model_path_on_port "$port")
  if [[ "$START_FRESH" != "1" && "$existing" == "$model_path" ]]; then
    echo "Reusing verified $served_name on port $port"
    return 0
  fi
  stop_known_port "$port"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && env CUDA_VISIBLE_DEVICES='$gpu' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$MODEL_PY' -u -m sglang.launch_server --served-model-name '$served_name' --model-path '$model_path' --tp 1 --dp 1 --context-length '$context' --dtype bfloat16 --host 127.0.0.1 --port '$port' --trust-remote-code --attention-backend triton --sampling-backend pytorch --disable-cuda-graph --disable-overlap --disable-radix-cache --chunked-prefill-size 4096 --max-running-requests 1 --mem-fraction-static '$fraction' >>'$log' 2>&1"
  echo "Started $served_name: tmux=$session GPU=$gpu port=$port log=$log"
  if ! wait_http "http://127.0.0.1:$port/health" 1200; then
    tail -n 200 "$log" >&2 || true
    exit 1
  fi
  existing=$(model_path_on_port "$port")
  [[ "$existing" == "$model_path" ]] || {
    echo "Model identity mismatch on port $port: $existing" >&2
    exit 1
  }
}

retriever_is_correct() {
  local body
  body=$(curl -fsS --max-time 5 "http://127.0.0.1:$RETRIEVER_PORT/health" 2>/dev/null || true)
  [[ "$body" == *'"retrievers"'* && "$body" != *"fanoutqa"* ]]
}

start_retriever() {
  local session="pandb_retriever_g2" log="$LOG_DIR/retriever.log"
  if [[ "$START_FRESH" != "1" ]] && retriever_is_correct; then
    echo "Reusing verified FlashRAG retriever on port $RETRIEVER_PORT"
    return 0
  fi
  stop_known_port "$RETRIEVER_PORT"
  tmux kill-session -t "$session" 2>/dev/null || true
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && env CUDA_VISIBLE_DEVICES='$PLANNER_GPU' PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$CORE_PY' -u scripts/serving/retriever_serving.py --config '$RETRIEVER_CONFIG' --num_retriever 1 --port '$RETRIEVER_PORT' >>'$log' 2>&1"
  echo "Started FlashRAG retriever: tmux=$session GPU=$PLANNER_GPU port=$RETRIEVER_PORT log=$log"
  if ! wait_http "http://127.0.0.1:$RETRIEVER_PORT/health" 1800; then
    tail -n 200 "$log" >&2 || true
    exit 1
  fi
  retriever_is_correct || {
    echo "Port $RETRIEVER_PORT is healthy but is not the expected FlashRAG retriever." >&2
    exit 1
  }
  local count
  count=$(curl -fsS --max-time 300 -X POST "http://127.0.0.1:$RETRIEVER_PORT/search" \
    -H 'Content-Type: application/json' \
    -d '{"query":"Albert Einstein birthplace","top_n":5,"return_score":false}' \
    | "$CORE_PY" -c 'import json,sys; print(len(json.load(sys.stdin)))')
  [[ "$count" == "5" ]] || { echo "Retriever Top-5 probe returned $count" >&2; exit 1; }
}

status() {
  for port in "$PLANNER_PORT" "$REASONER_PORT" "$RETRIEVER_PORT"; do
    printf 'port %s: ' "$port"
    curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && echo healthy || echo unavailable
  done
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader
  tmux ls 2>/dev/null | grep -E 'pandb_(planner|reasoner|retriever)' || true
}

case "${1:-start}" in
  start)
    [[ -s "$PLANNER_MODEL_PATH/config.json" ]] || { echo "Missing planner: $PLANNER_MODEL_PATH" >&2; exit 1; }
    [[ -s "$REASONER_MODEL_PATH/config.json" ]] || { echo "Missing reasoner: $REASONER_MODEL_PATH" >&2; exit 1; }
    [[ -s "$RETRIEVER_CONFIG" ]] || { echo "Missing retriever config: $RETRIEVER_CONFIG" >&2; exit 1; }
    start_model pandb_planner_g2 "$PLANNER_GPU" "$PLANNER_PORT" "$PLANNER_MODEL_PATH" Llama-3.1-8B-Instruct "$PLANNER_CONTEXT_LENGTH" "$PLANNER_MEM_FRACTION" "$LOG_DIR/planner_g2.log"
    start_model pandb_reasoner_g3 "$REASONER_GPU" "$REASONER_PORT" "$REASONER_MODEL_PATH" Qwen3-8B "$REASONER_CONTEXT_LENGTH" "$REASONER_MEM_FRACTION" "$LOG_DIR/reasoner_g3.log"
    start_retriever
    status
    ;;
  status) status ;;
  stop)
    for session in pandb_planner_g2 pandb_reasoner_g3 pandb_retriever_g2; do
      tmux kill-session -t "$session" 2>/dev/null || true
    done
    ;;
  *) echo "Usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac

