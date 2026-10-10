#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
MODEL_PYTHON="${MODEL_PYTHON:-/home/researcher/smartsearch_rebuild/envs/src-model/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
RETRIEVER_MODEL_PATH="${RETRIEVER_MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/e5-base-v2}"
CORPUS_PATH="${CORPUS_PATH:-/home/researcher/smartsearch_rebuild/retrieval/wiki18_100w.jsonl}"
INDEX_PATH="${INDEX_PATH:-/home/researcher/smartsearch_rebuild/retrieval/wiki18_100w_e5_index/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index}"
RETRIEVER_CONFIG="${RETRIEVER_CONFIG:-$REPO_ROOT/scripts/serving/retriever_config_adaptive_k.yaml}"
RETRIEVER_PORT="${RETRIEVER_PORT:-6008}"
GPU_IDS="${GPU_IDS:-2,3}"
BASE_PORT="${BASE_PORT:-9103}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-40960}"
MODEL_MEM_FRACTION="${MODEL_MEM_FRACTION:-0.84}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-24497}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-1000}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/adaptive_k_hotpot500_qwen3_e5_v1/services_g2g3}"
RUN_DIR_WAS_SET="${RUN_DIR+x}"
SESSION_NAME_WAS_SET="${SESSION_NAME+x}"
RUN_DIR="${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/adaptive_k_hotpot500_qwen3_e5_v1/full500_r2_t06}"
SESSION_NAME="${SESSION_NAME:-adaptivek_hotpot500_full}"
mkdir -p "$SERVICE_RUN_DIR/tmux_logs" "$RUN_DIR/tmux_logs"

IFS=',' read -r -a GPUS <<< "$GPU_IDS"
[[ "${#GPUS[@]}" == 2 && "${GPUS[0]}" == 2 && "${GPUS[1]}" == 3 ]] || {
  echo "This registered launcher requires GPU_IDS=2,3" >&2; exit 2;
}

port_listening() { ss -ltn "sport = :$1" 2>/dev/null | grep -q LISTEN; }
model_cmdline() {
  ps -eo pid=,user=,args= | grep '[s]glang.launch_server' \
    | grep -E -- "--port[ =]+$1([[:space:]]|$)" | head -n 1 || true
}

stop_owned_matching_port() {
  local port="$1" pattern="$2" label="$3" line pid owner cmd
  port_listening "$port" || return 0
  line="$(ps -eo pid=,user=,args= | grep -E "$pattern" \
    | grep -E -- "--port([ =]+)$port([[:space:]]|$)" | head -n 1 || true)"
  if [[ -z "$line" ]]; then
    echo "Port $port is occupied, but no safely identifiable $label process was found." >&2
    ss -ltnp "sport = :$port" >&2 || true
    return 1
  fi
  pid="$(awk '{print $1}' <<<"$line")"; owner="$(awk '{print $2}' <<<"$line")"
  cmd="$(cut -d' ' -f3- <<<"$line")"
  [[ "$owner" == "$USER" ]] || { echo "Refusing to stop $label PID $pid owned by $owner" >&2; return 1; }
  echo "Stopping stale/mismatched $label PID=$pid on port=$port"
  kill -TERM "$pid"
  for _ in $(seq 1 30); do port_listening "$port" || return 0; sleep 2; done
  if kill -0 "$pid" 2>/dev/null; then kill -KILL "$pid"; fi
  for _ in $(seq 1 15); do port_listening "$port" || return 0; sleep 2; done
  echo "Port $port remained occupied after stopping: $cmd" >&2; return 1
}

verify_model() {
  local port="$1" gpu="$2" body line pid visible
  body="$(curl -fsS --max-time 10 "http://127.0.0.1:$port/get_model_info")" || return 1
  [[ "$body" == *"Qwen3-8B"* ]] || return 1
  line="$(model_cmdline "$port")"; [[ -n "$line" ]] || return 1
  [[ "$line" == *"--context-length $CONTEXT_LENGTH"* ]] || return 1
  [[ "$line" == *"--max-running-requests $MAX_RUNNING_REQUESTS"* ]] || return 1
  pid="$(awk '{print $1}' <<<"$line")"
  visible="$(tr '\0' '\n' <"/proc/$pid/environ" | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | tail -n 1)"
  [[ "$visible" == "$gpu" ]] || return 1
}

wait_model() {
  local port="$1" gpu="$2"
  for _ in $(seq 1 300); do
    if verify_model "$port" "$gpu"; then echo "Qwen3-8B ready: GPU$gpu port=$port"; return 0; fi
    sleep 4
  done
  echo "Timed out waiting for Qwen model on port $port" >&2; return 1
}

start_models() {
  local slot gpu port session free command value
  for slot in 0 1; do
    gpu="${GPUS[$slot]}"; port="$((BASE_PORT + slot))"; session="adaptivek_v1_model_g$gpu"
    if verify_model "$port" "$gpu"; then
      echo "Reusing verified Qwen3-8B: GPU$gpu port=$port"; continue
    fi
    if port_listening "$port"; then
      stop_owned_matching_port "$port" '[s]glang.launch_server' "SGLang model"
    fi
    tmux kill-session -t "$session" 2>/dev/null || true
    free="$(nvidia-smi -i "$gpu" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')"
    (( free >= 22000 )) || { echo "GPU$gpu has only ${free}MiB free" >&2; exit 1; }
    command="env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NO_PROXY=127.0.0.1,localhost \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu \
'$MODEL_PYTHON' -u -m sglang.launch_server --served-model-name Qwen3-8B \
--model-path '$MODEL_PATH' --tp 1 --dp 1 --context-length '$CONTEXT_LENGTH' \
--dtype bfloat16 --host 127.0.0.1 --port '$port' --trust-remote-code \
--attention-backend triton --sampling-backend pytorch --disable-cuda-graph \
--disable-overlap --chunked-prefill-size 4096 --max-running-requests '$MAX_RUNNING_REQUESTS' \
--mem-fraction-static '$MODEL_MEM_FRACTION'"
    tmux new-session -d -s "$session" \
      "cd '$REPO_ROOT' && $command >>'$SERVICE_RUN_DIR/tmux_logs/model_g$gpu.log' 2>&1"
    echo "Started $session"
    wait_model "$port" "$gpu"
  done
  values=()
  for gpu in "${GPUS[@]}"; do
    value="$(grep -oE 'max_total_num_tokens=[0-9]+' \
      "$SERVICE_RUN_DIR/tmux_logs/model_g$gpu.log" 2>/dev/null \
      | tail -n 1 | cut -d= -f2 || true)"
    [[ -n "$value" ]] && values+=("$value")
  done
  if [[ "${#values[@]}" == 2 ]]; then
    SERVER_MAX_TOTAL_TOKENS="$(printf '%s\n' "${values[@]}" | sort -n | head -n 1)"
    printf '%s\n' "$SERVER_MAX_TOTAL_TOKENS" >"$SERVICE_RUN_DIR/model_token_pool.txt"
    echo "Detected safe common model token pool: $SERVER_MAX_TOTAL_TOKENS"
  elif [[ -s "$SERVICE_RUN_DIR/model_token_pool.txt" ]]; then
    SERVER_MAX_TOTAL_TOKENS="$(cat "$SERVICE_RUN_DIR/model_token_pool.txt")"
    echo "Reusing recorded safe model token pool: $SERVER_MAX_TOTAL_TOKENS"
  else
    echo "Model startup logs are unavailable; using verified prior pool: $SERVER_MAX_TOTAL_TOKENS"
  fi
}

probe_retriever() {
  local probe
  curl -fsS --max-time 10 "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null || return 1
  probe="$(mktemp /tmp/adaptivek-retriever-probe.XXXXXX.json)"
  if ! curl -fsS --max-time 900 -X POST "http://127.0.0.1:$RETRIEVER_PORT/search" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"HotpotQA adaptive retrieval probe\",\"top_n\":$CANDIDATE_COUNT,\"return_score\":true}" \
    -o "$probe"; then
    rm -f -- "$probe"; return 1
  fi
  if ! "$PYTHON_BIN" - "$probe" "$CANDIDATE_COUNT" <<'PY'
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
expected = int(sys.argv[2])
assert isinstance(value, list) and len(value) == 2, type(value)
assert len(value[0]) == expected and len(value[1]) == expected, (len(value[0]), len(value[1]))
assert all(value[1][i] >= value[1][i+1] for i in range(expected-1))
PY
  then
    rm -f -- "$probe"; return 1
  fi
  rm -f -- "$probe"
  echo "Scored FlashRAG Top-$CANDIDATE_COUNT verified"
}

verify_retriever() {
  local line
  line="$(ps -eo pid=,user=,args= | grep '[r]etriever_serving.py' \
    | grep -E -- "--port[ =]+$RETRIEVER_PORT([[:space:]]|$)" | head -n 1 || true)"
  [[ -n "$line" && "$line" == *"--config $RETRIEVER_CONFIG"* ]] || return 1
  probe_retriever
}

start_retriever() {
  local session="adaptivek_v1_retriever" command
  "$PYTHON_BIN" "$SCRIPT_DIR/write_retriever_config.py" \
    --model "$RETRIEVER_MODEL_PATH" --index "$INDEX_PATH" --corpus "$CORPUS_PATH" \
    --candidate_count "$CANDIDATE_COUNT" --output "$RETRIEVER_CONFIG" >/dev/null
  if verify_retriever; then
    echo "Reusing verified Adaptive-k FlashRAG retriever: port=$RETRIEVER_PORT"
    return 0
  fi
  if port_listening "$RETRIEVER_PORT"; then
    stop_owned_matching_port "$RETRIEVER_PORT" '[r]etriever_serving.py' "FlashRAG retriever"
  fi
  tmux kill-session -t "$session" 2>/dev/null || true
  command="env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY NO_PROXY=127.0.0.1,localhost \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=2 \
'$PYTHON_BIN' -u '$REPO_ROOT/scripts/serving/retriever_serving.py' \
--config '$RETRIEVER_CONFIG' --num_retriever 1 --port '$RETRIEVER_PORT'"
  tmux new-session -d -s "$session" \
    "cd '$REPO_ROOT' && $command >>'$SERVICE_RUN_DIR/tmux_logs/retriever.log' 2>&1"
  echo "Started $session; loading the 61GB CPU FAISS index may take several minutes."
  for _ in $(seq 1 480); do
    curl -fsS --max-time 5 "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null 2>&1 && break
    sleep 5
  done
  curl -fsS --max-time 10 "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null
  probe_retriever
}

services() { start_models; start_retriever; }

start_experiment() {
  local limit="$1" default_session="$2"
  services
  if [[ -s "$SERVICE_RUN_DIR/model_token_pool.txt" ]]; then
    SERVER_MAX_TOTAL_TOKENS="$(cat "$SERVICE_RUN_DIR/model_token_pool.txt")"
  fi
  [[ -n "$SESSION_NAME_WAS_SET" ]] || SESSION_NAME="$default_session"
  mkdir -p "$RUN_DIR/tmux_logs"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session already exists: $SESSION_NAME"; return 0
  fi
  tmux new-session -d -s "$SESSION_NAME" \
    "cd '$REPO_ROOT' && env RUN_DIR='$RUN_DIR' LIMIT='$limit' \
PYTHON_BIN='$PYTHON_BIN' MODEL_PATH='$MODEL_PATH' CORPUS_PATH='$CORPUS_PATH' INDEX_PATH='$INDEX_PATH' \
RETRIEVER_URL='http://127.0.0.1:$RETRIEVER_PORT' \
MODEL_URLS='http://127.0.0.1:$BASE_PORT,http://127.0.0.1:$((BASE_PORT+1))' \
SERVER_MAX_TOTAL_TOKENS='$SERVER_MAX_TOTAL_TOKENS' \
bash '$SCRIPT_DIR/run.sh' full >>'$RUN_DIR/tmux_logs/coordinator.log' 2>&1"
  echo "Started $SESSION_NAME"
  echo "Follow: tail -f $RUN_DIR/tmux_logs/coordinator.log"
}

status() {
  for name in adaptivek_v1_model_g2 adaptivek_v1_model_g3 adaptivek_v1_retriever "$SESSION_NAME"; do
    tmux has-session -t "$name" 2>/dev/null && echo "running: $name" || echo "absent:  $name"
  done
  for port in "$BASE_PORT" "$((BASE_PORT+1))" "$RETRIEVER_PORT"; do
    curl -fsS --max-time 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1 \
      && echo "healthy port=$port" || echo "unavailable port=$port"
  done
  for log in "$RUN_DIR/tmux_logs/coordinator.log" "$RUN_DIR/tmux_logs/generation_shard_0.log" "$RUN_DIR/tmux_logs/generation_shard_1.log"; do
    [[ -s "$log" ]] && { echo "===== $log ====="; tail -n 30 "$log"; }
  done
  find "$RUN_DIR" -type f -name '*.jsonl' -exec wc -l {} \; 2>/dev/null || true
}

case "${1:-}" in
  services) services ;;
  plan) env RUN_DIR="$RUN_DIR" LIMIT=0 bash "$SCRIPT_DIR/run.sh" plan ;;
  probe5)
    [[ -n "$RUN_DIR_WAS_SET" ]] || RUN_DIR="/home/researcher/smartsearch_rebuild/outputs/adaptive_k_hotpot500_qwen3_e5_v1/probe5_r2_t06"
    start_experiment 5 adaptivek_hotpot_probe5 ;;
  full) start_experiment 0 adaptivek_hotpot500_full ;;
  status) status ;;
  stop-experiment) tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true ;;
  stop-services)
    for name in adaptivek_v1_model_g2 adaptivek_v1_model_g3 adaptivek_v1_retriever; do
      tmux kill-session -t "$name" 2>/dev/null || true
    done ;;
  *) echo "Usage: $0 {services|plan|probe5|full|status|stop-experiment|stop-services}" >&2; exit 2 ;;
esac
