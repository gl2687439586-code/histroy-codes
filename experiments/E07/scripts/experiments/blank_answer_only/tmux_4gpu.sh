#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RUN_DIR="${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/blank_answer_only/bamboogle_qwen3_v5}"
LOG_DIR="$RUN_DIR/tmux_logs"
mkdir -p "$LOG_DIR"

MODEL_PYTHON="${MODEL_PYTHON:-/home/researcher/smartsearch_rebuild/envs/src-model/bin/python}"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
ROOT_CACHE="${ROOT_CACHE:-/home/researcher/smartsearch_rebuild/outputs/read_budget_surface/bamboogle_qwen3_fresh_v1/root_cache.jsonl}"
PILOT_RECORDS="${PILOT_RECORDS:-/home/researcher/smartsearch_rebuild/outputs/blank_evidence_budget/bamboogle_qwen3_v4_gate5_allbudgets_answer3072/probe5/records.jsonl}"
BASE_PORT="${BASE_PORT:-9101}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-10086}"
MODEL_MEM_FRACTION="${MODEL_MEM_FRACTION:-0.72}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6144}"
TIMEOUT="${TIMEOUT:-900}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-2026}"
ROOT_SCOPE="${ROOT_SCOPE:-all}"
CONDITION_IDS="${CONDITION_IDS:-}"
RETRY_FAILURES="${RETRY_FAILURES:-0}"
ALLOW_UNVERIFIED_SERVER="${ALLOW_UNVERIFIED_SERVER:-0}"
RESOURCE_INTERVAL_SECONDS="${RESOURCE_INTERVAL_SECONDS:-300}"
ESTIMATE_SAFETY_MULTIPLIER="${ESTIMATE_SAFETY_MULTIPLIER:-1.25}"
SESSION_PREFIX="${SESSION_PREFIX:-blankanswer_v5}"
COORDINATOR_SESSION="${COORDINATOR_SESSION:-${SESSION_PREFIX}_experiment}"

start_session() {
  local name="$1" command="$2" log="$3"
  if tmux has-session -t "$name" 2>/dev/null; then
    echo "Session already exists: $name"
    return
  fi
  tmux new-session -d -s "$name" \
    "cd '$REPO_ROOT' && $command >>'$log' 2>&1"
  echo "Started $name; log=$log"
}

verify_qwen_port() {
  local port="$1" body
  body="$(curl -fsS "http://127.0.0.1:$port/get_model_info" || true)"
  [[ "$body" == *"Qwen3-8B"* ]]
}

wait_model() {
  local port="$1"
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      verify_qwen_port "$port" || { echo "Port $port is not Qwen3-8B." >&2; return 1; }
      echo "Qwen3-8B ready: http://127.0.0.1:$port"
      return 0
    fi
    sleep 5
  done
  echo "Model timeout on port $port" >&2
  return 1
}

check_gpu_capacity() {
  local gpu="$1" free_mib
  free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
  [[ "$free_mib" -ge 22000 ]] || {
    echo "GPU $gpu has only ${free_mib} MiB free." >&2; return 1;
  }
}

start_models() {
  local gpu port session command
  for gpu in $(seq 0 3); do
    port="$((BASE_PORT + gpu))"
    session="${SESSION_PREFIX}_model_g$gpu"
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      verify_qwen_port "$port" || { echo "Non-Qwen service on $port" >&2; exit 1; }
      echo "Reusing verified Qwen3-8B on port $port"
      continue
    fi
    check_gpu_capacity "$gpu"
    command="env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
NO_PROXY=localhost,127.0.0.1,0.0.0.0,::1 no_proxy=localhost,127.0.0.1,0.0.0.0,::1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu \
'$MODEL_PYTHON' -u -m sglang.launch_server --served-model-name Qwen3-8B \
--model-path '$MODEL_PATH' --tp 1 --dp 1 --context-length '$CONTEXT_LENGTH' \
--dtype bfloat16 --host 127.0.0.1 --port '$port' --trust-remote-code \
--attention-backend triton --sampling-backend pytorch --disable-cuda-graph \
--disable-overlap --disable-radix-cache --chunked-prefill-size 4096 \
--max-running-requests 1 --mem-fraction-static '$MODEL_MEM_FRACTION'"
    start_session "$session" "$command" "$LOG_DIR/model_g$gpu.log"
    wait_model "$port"
  done
}

capture_service_snapshot() {
  {
    echo "captured_at=$(date --iso-8601=seconds)"
    echo "expected_context_length=$CONTEXT_LENGTH"
    echo "expected_server_max_total_tokens=$SERVER_MAX_TOTAL_TOKENS"
    echo "expected_model_mem_fraction=$MODEL_MEM_FRACTION"
    echo "=== process command lines ==="
    ps -eo pid,args | grep '[s]glang.launch_server.*--port 910[1-4]' || true
    echo "=== model identities ==="
    for port in $(seq "$BASE_PORT" $((BASE_PORT + 3))); do
      printf 'port=%s ' "$port"
      curl -fsS "http://127.0.0.1:$port/get_model_info" || true
      echo
    done
    echo "=== GPUs ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total \
      --format=csv,noheader
  } >"$RUN_DIR/service_snapshot.txt"
}

start_experiment() {
  local command
  capture_service_snapshot
  command="export RUN_DIR='$RUN_DIR' PYTHON_BIN='$PYTHON_BIN' MODEL_PATH='$MODEL_PATH' \
ROOT_CACHE='$ROOT_CACHE' PILOT_RECORDS='$PILOT_RECORDS' BASE_PORT='$BASE_PORT' NUM_SHARDS=4 \
CONTEXT_LENGTH='$CONTEXT_LENGTH' SERVER_MAX_TOTAL_TOKENS='$SERVER_MAX_TOTAL_TOKENS' \
MAX_NEW_TOKENS='$MAX_NEW_TOKENS' TIMEOUT='$TIMEOUT' TEMPERATURE='$TEMPERATURE' \
SEED='$SEED' ROOT_SCOPE='$ROOT_SCOPE' CONDITION_IDS='$CONDITION_IDS' \
RETRY_FAILURES='$RETRY_FAILURES' ALLOW_UNVERIFIED_SERVER='$ALLOW_UNVERIFIED_SERVER' \
RESOURCE_INTERVAL_SECONDS='$RESOURCE_INTERVAL_SECONDS' \
ESTIMATE_SAFETY_MULTIPLIER='$ESTIMATE_SAFETY_MULTIPLIER'; \
bash '$SCRIPT_DIR/parallel_coordinator.sh'"
  start_session "$COORDINATOR_SESSION" "$command" "$LOG_DIR/coordinator.log"
}

case "${1:-all}" in
  models) start_models ;;
  experiment)
    for gpu in $(seq 0 3); do wait_model "$((BASE_PORT + gpu))"; done
    start_experiment
    ;;
  all) start_models; start_experiment ;;
  status)
    tmux ls 2>/dev/null || true
    for log in "$LOG_DIR"/coordinator.log "$LOG_DIR"/worker_*.log \
               "$LOG_DIR"/resources_and_eta.log "$LOG_DIR"/finalize.log; do
      [[ -e "$log" ]] && { echo "===== $log ====="; tail -n 20 "$log"; }
    done
    ;;
  stop-experiment)
    tmux kill-session -t "$COORDINATOR_SESSION" 2>/dev/null || true
    echo "Stopped coordinator only; records and model services retained."
    ;;
  stop)
    tmux kill-session -t "$COORDINATOR_SESSION" 2>/dev/null || true
    for gpu in $(seq 0 3); do
      tmux kill-session -t "${SESSION_PREFIX}_model_g$gpu" 2>/dev/null || true
    done
    echo "Stopped $SESSION_PREFIX sessions; experiment files retained."
    ;;
  *) echo "Usage: $0 {models|experiment|all|status|stop-experiment|stop}" >&2; exit 2 ;;
esac
