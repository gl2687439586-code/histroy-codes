#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REBUILD_ROOT="${REBUILD_ROOT:-/home/researcher/smartsearch_rebuild}"
REPO_ROOT="${REPO_ROOT:-$REBUILD_ROOT/SmartSearch}"
CORE_PY="${PYTHON_BIN:-$REBUILD_ROOT/envs/src-core/bin/python}"
MODEL_PY="${MODEL_PYTHON:-$REBUILD_ROOT/envs/src-model/bin/python}"
MODEL_PATH="${MODEL_PATH:-$REBUILD_ROOT/models/Qwen3-8B}"
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-$REBUILD_ROOT/models/e5-base-v2}"
CORPUS_PATH="${CORPUS_PATH:-$REBUILD_ROOT/retrieval/wiki18_100w.jsonl}"
INDEX_PATH="${INDEX_PATH:-$REBUILD_ROOT/retrieval/wiki18_100w_e5_index/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index}"
RETRIEVER_CONFIG="${RETRIEVER_CONFIG:-$REPO_ROOT/scripts/serving/retriever_config_tale_ep.yaml}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-$REBUILD_ROOT/outputs/tale_ep_hotpot500_qwen3_v3_original/services_g2g3}"
GPU_IDS="${GPU_IDS:-2,3}"
BASE_PORT="${BASE_PORT:-9103}"
RETRIEVER_PORT="${RETRIEVER_PORT:-6008}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-40960}"
MODEL_MEM_FRACTION="${MODEL_MEM_FRACTION:-0.84}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-24497}"
TAKEOVER_KNOWN_SERVICES="${TAKEOVER_KNOWN_SERVICES:-1}"

LOG_DIR="$SERVICE_RUN_DIR/tmux_logs"
mkdir -p "$LOG_DIR"
IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ "${#GPUS[@]}" == 2 && "${GPUS[0]}" == 2 && "${GPUS[1]}" == 3 ]] || {
  echo "Registered launcher requires GPU_IDS=2,3" >&2; exit 2;
}

port_pids() { fuser -n tcp "$1" 2>/dev/null || true; }

known_owned_service_pid() {
  local pid="$1" owner cmd
  owner="$(ps -o user= -p "$pid" 2>/dev/null | xargs || true)"
  cmd="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$owner" == "$USER" ]] || return 1
  [[ "$cmd" == *"sglang.launch_server"* || "$cmd" == *"retriever_serving.py"* ]]
}

stop_known_port() {
  local port="$1" pids pid
  pids="$(port_pids "$port")"
  [[ -z "$pids" ]] && return 0
  for pid in $pids; do
    if ! known_owned_service_pid "$pid"; then
      echo "Refusing to stop unknown or foreign PID=$pid on port=$port" >&2
      ps -fp "$pid" >&2 || true
      ss -ltnp "sport = :$port" >&2 || true
      return 1
    fi
  done
  [[ "$TAKEOVER_KNOWN_SERVICES" == "1" ]] || {
    echo "Known old service occupies port $port; set TAKEOVER_KNOWN_SERVICES=1" >&2
    return 1
  }
  echo "Stopping known old service(s) on port $port: $pids"
  kill -TERM $pids 2>/dev/null || true
  for _ in $(seq 1 30); do
    [[ -z "$(port_pids "$port")" ]] && return 0
    sleep 2
  done
  pids="$(port_pids "$port")"
  for pid in $pids; do
    known_owned_service_pid "$pid" && kill -KILL "$pid" 2>/dev/null || true
  done
  for _ in $(seq 1 15); do
    [[ -z "$(port_pids "$port")" ]] && return 0
    sleep 2
  done
  echo "Port $port remains occupied" >&2
  return 1
}

model_line() {
  ps -eo pid=,user=,args= | grep '[s]glang.launch_server' \
    | grep -E -- "--port[ =]+$1([[:space:]]|$)" | head -n 1 || true
}

verify_model() {
  local port="$1" gpu="$2" body line pid visible
  body="$(curl -fsS --max-time 10 "http://127.0.0.1:$port/get_model_info")" || return 1
  [[ "$body" == *"Qwen3-8B"* ]] || return 1
  line="$(model_line "$port")"; [[ -n "$line" ]] || return 1
  [[ "$line" == *"--context-length $CONTEXT_LENGTH"* ]] || return 1
  [[ "$line" == *"--max-running-requests $MAX_RUNNING_REQUESTS"* ]] || return 1
  pid="$(awk '{print $1}' <<<"$line")"
  visible="$(tr '\0' '\n' <"/proc/$pid/environ" 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | tail -n 1)"
  [[ "$visible" == "$gpu" ]]
}

wait_model() {
  local port="$1" gpu="$2"
  for _ in $(seq 1 300); do
    if verify_model "$port" "$gpu"; then
      echo "Qwen3-8B ready: physical GPU$gpu port=$port"; return 0
    fi
    sleep 4
  done
  echo "Timed out waiting for Qwen3-8B on port $port" >&2; return 1
}

start_models() {
  [[ -s "$MODEL_PATH/config.json" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 1; }
  local index gpu port session log command
  for index in 0 1; do
    gpu="${GPUS[$index]}"; port=$((BASE_PORT + index))
    session="tale_ep_v3_original_model_g$gpu"; log="$LOG_DIR/model_g$gpu.log"
    if verify_model "$port" "$gpu"; then
      echo "Reusing verified Qwen3-8B: GPU$gpu port=$port"
      continue
    fi
    [[ -z "$(port_pids "$port")" ]] || stop_known_port "$port"
    tmux kill-session -t "$session" 2>/dev/null || true
    command="env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NO_PROXY=127.0.0.1,localhost \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$MODEL_PY' -u -m sglang.launch_server \
--served-model-name Qwen3-8B --model-path '$MODEL_PATH' --tp 1 --dp 1 \
--context-length '$CONTEXT_LENGTH' --dtype bfloat16 --host 127.0.0.1 --port '$port' \
--trust-remote-code --attention-backend triton --sampling-backend pytorch \
--disable-cuda-graph --disable-overlap --disable-radix-cache --chunked-prefill-size 4096 \
--max-running-requests '$MAX_RUNNING_REQUESTS' --mem-fraction-static '$MODEL_MEM_FRACTION'"
    tmux new-session -d -s "$session" \
      "cd '$REPO_ROOT' && $command >>'$log' 2>&1"
    echo "Started $session; log=$log"
    wait_model "$port" "$gpu"
  done
  local values=() value
  for gpu in "${GPUS[@]}"; do
    value="$(grep -oE 'max_total_num_tokens=[0-9]+' "$LOG_DIR/model_g$gpu.log" 2>/dev/null \
      | tail -n 1 | cut -d= -f2 || true)"
    [[ -n "$value" ]] && values+=("$value")
  done
  if [[ "${#values[@]}" == 2 ]]; then
    SERVER_MAX_TOTAL_TOKENS="$(printf '%s\n' "${values[@]}" | sort -n | head -n 1)"
  fi
  printf '%s\n' "$SERVER_MAX_TOTAL_TOKENS" >"$SERVICE_RUN_DIR/model_token_pool.txt"
  echo "Safe common model token pool: $SERVER_MAX_TOTAL_TOKENS"
}

probe_retriever() {
  local file count
  curl -fsS --max-time 10 "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null || return 1
  file="$(mktemp /tmp/tale-ep-retriever.XXXXXX.json)"
  if ! curl -fsS --max-time 900 -X POST "http://127.0.0.1:$RETRIEVER_PORT/search" \
    -H 'Content-Type: application/json' \
    -d '{"query":"Albert Einstein birthplace","top_n":5,"return_score":false}' \
    -o "$file"; then rm -f "$file"; return 1; fi
  count="$($CORE_PY - "$file" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
print(len(value) if isinstance(value, list) else -1)
PY
)"
  rm -f "$file"
  [[ "$count" == "5" ]] || return 1
  echo "FlashRAG E5/wiki18 Top-5 verified: port=$RETRIEVER_PORT"
}

verify_retriever() {
  local line
  line="$(ps -eo pid=,user=,args= | grep '[r]etriever_serving.py' \
    | grep -E -- "--port[ =]+$RETRIEVER_PORT([[:space:]]|$)" | head -n 1 || true)"
  [[ -n "$line" && "$line" == *"--config $RETRIEVER_CONFIG"* ]] || return 1
  probe_retriever
}

start_retriever() {
  local session="tale_ep_v3_original_retriever" log="$LOG_DIR/retriever.log" command
  "$CORE_PY" "$SCRIPT_DIR/write_retriever_config.py" \
    --model "$RETRIEVER_MODEL_PATH" --index "$INDEX_PATH" --corpus "$CORPUS_PATH" \
    --output "$RETRIEVER_CONFIG" >/dev/null
  if verify_retriever; then return 0; fi
  [[ -z "$(port_pids "$RETRIEVER_PORT")" ]] || stop_known_port "$RETRIEVER_PORT"
  tmux kill-session -t "$session" 2>/dev/null || true
  command="env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NO_PROXY=127.0.0.1,localhost \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True '$CORE_PY' -u \
'$REPO_ROOT/scripts/serving/retriever_serving.py' --config '$RETRIEVER_CONFIG' \
--num_retriever 1 --port '$RETRIEVER_PORT'"
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && $command >>'$log' 2>&1"
  echo "Started $session; loading the CPU FAISS index may take several minutes; log=$log"
  for _ in $(seq 1 480); do
    curl -fsS --max-time 5 "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null 2>&1 && break
    sleep 5
  done
  probe_retriever || { tail -n 200 "$log" >&2 || true; return 1; }
}

status() {
  local port
  for port in "$BASE_PORT" "$((BASE_PORT + 1))" "$RETRIEVER_PORT"; do
    printf 'port %s: ' "$port"
    curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
      && echo healthy || echo unavailable
  done
  nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader
  tmux ls 2>/dev/null | grep -E 'tale_ep_v3_original_(model|retriever)' || true
}

case "${1:-start}" in
  start) start_models; start_retriever; status ;;
  status) status ;;
  stop)
    for session in tale_ep_v3_original_model_g2 tale_ep_v3_original_model_g3 tale_ep_v3_original_retriever; do
      tmux kill-session -t "$session" 2>/dev/null || true
    done
    for port in "$BASE_PORT" "$((BASE_PORT + 1))" "$RETRIEVER_PORT"; do
      [[ -z "$(port_pids "$port")" ]] || stop_known_port "$port"
    done
    ;;
  *) echo "Usage: $0 {start|status|stop}" >&2; exit 2 ;;
esac
