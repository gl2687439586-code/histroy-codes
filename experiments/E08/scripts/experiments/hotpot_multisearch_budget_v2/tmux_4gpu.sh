#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
MODEL_PYTHON="${MODEL_PYTHON:-/home/researcher/smartsearch_rebuild/envs/src-model/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
BASE_PORT="${BASE_PORT:-9101}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
MODEL_MEM_FRACTION="${MODEL_MEM_FRACTION:-0.75}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
RETRIEVER_PORT="${RETRIEVER_PORT:-6008}"
RETRIEVER_CONFIG="${RETRIEVER_CONFIG:-$REPO_ROOT/scripts/serving/retriever_config_src_action_tree.yaml}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/hotpot_multisearch_budget_v2/services}"
DATA_DIR="${DATA_DIR:-/home/researcher/smartsearch_rebuild/data/hotpotqa}"
mkdir -p "$SERVICE_RUN_DIR/tmux_logs"

start_session() {
  local name="$1" command="$2" log="$3"
  tmux has-session -t "$name" 2>/dev/null && {
    echo "Session already exists: $name"; return 0;
  }
  tmux new-session -d -s "$name" \
    "cd '$REPO_ROOT' && $command >>'$log' 2>&1"
  echo "Started $name; log=$log"
}

model_cmdline() {
  local port="$1"
  ps -eo args= | grep '[s]glang.launch_server' \
    | grep -E -- "--port[ =]+${port}([[:space:]]|$)" | head -n 1 || true
}

verify_model_config() {
  local port="$1" cmdline
  cmdline="$(model_cmdline "$port")"
  [[ -n "$cmdline" ]] || { echo "Cannot find model process for port $port" >&2; return 1; }
  [[ "$cmdline" == *"--context-length $CONTEXT_LENGTH"* ]] || {
    echo "Port $port is not using context-length=$CONTEXT_LENGTH" >&2
    echo "$cmdline" >&2
    return 1
  }
}

check_models() {
  for port in $(seq "$BASE_PORT" $((BASE_PORT + 3))); do
    body="$(curl -fsS "http://127.0.0.1:$port/get_model_info" || true)"
    [[ "$body" == *"Qwen3-8B"* ]] || {
      echo "Model port $port is unavailable or not Qwen3-8B" >&2; return 1;
    }
    verify_model_config "$port" || return 1
    echo "Qwen3-8B verified: $port"
  done
}

wait_model() {
  local port="$1"
  for _ in $(seq 1 240); do
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && {
      verify_model_config "$port"
      echo "Qwen3-8B ready: $port"
      return 0
    }
    sleep 5
  done
  echo "Timed out waiting for model port $port" >&2
  return 1
}

start_models() {
  local gpu port session command free_mib
  for gpu in $(seq 0 3); do
    port="$((BASE_PORT + gpu))"
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "Port $port is occupied; verifying its model configuration."
      body="$(curl -fsS "http://127.0.0.1:$port/get_model_info" || true)"
      [[ "$body" == *"Qwen3-8B"* ]] || {
        echo "Port $port is not Qwen3-8B" >&2; return 1;
      }
      verify_model_config "$port"
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      -i "$gpu" | tr -d ' ')"
    [[ "$free_mib" -ge 22000 ]] || {
      echo "GPU$gpu has only ${free_mib}MiB free; stop old services first." >&2
      return 1
    }
    session="hotpot_v2_model_g$gpu"
    command="env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=$gpu \
'$MODEL_PYTHON' -u -m sglang.launch_server --served-model-name Qwen3-8B \
--model-path '$MODEL_PATH' --tp 1 --dp 1 --context-length '$CONTEXT_LENGTH' \
--dtype bfloat16 --host 127.0.0.1 --port '$port' --trust-remote-code \
--attention-backend triton --sampling-backend pytorch --disable-cuda-graph \
--disable-overlap --chunked-prefill-size 4096 \
--max-running-requests '$MAX_RUNNING_REQUESTS' \
--mem-fraction-static '$MODEL_MEM_FRACTION'"
    start_session "$session" "$command" "$SERVICE_RUN_DIR/tmux_logs/model_g$gpu.log"
    wait_model "$port"
  done
  check_models
  values=()
  for gpu in $(seq 0 3); do
    value="$(grep -oE 'max_total_num_tokens=[0-9]+' \
      "$SERVICE_RUN_DIR/tmux_logs/model_g$gpu.log" 2>/dev/null \
      | tail -n 1 | cut -d= -f2 || true)"
    [[ -n "$value" ]] && values+=("$value")
  done
  if [[ "${#values[@]}" == "4" ]]; then
    printf '%s\n' "${values[@]}" | sort -n | head -n 1 \
      >"$SERVICE_RUN_DIR/model_token_pool.txt"
    echo "Detected safe per-server token pool: $(cat "$SERVICE_RUN_DIR/model_token_pool.txt")"
  else
    echo "Token-pool logs unavailable; set SERVER_MAX_TOTAL_TOKENS explicitly." >&2
  fi
}

check_retriever() {
  curl -fsS "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null
  count="$(curl -fsS -X POST "http://127.0.0.1:$RETRIEVER_PORT/search" \
    -H 'Content-Type: application/json' \
    -d '{"query":"HotpotQA evidence retrieval","top_n":10,"return_score":false}' \
    | "$PYTHON_BIN" -c 'import json,sys; print(len(json.load(sys.stdin)))')"
  [[ "$count" == "10" ]] || { echo "Retriever did not return Top-10" >&2; return 1; }
  echo "Retriever Top-10 verified: $RETRIEVER_PORT"
}

start_services() {
  start_models

  if ! check_retriever; then
    session=hotpot_retriever
    tmux kill-session -t "$session" 2>/dev/null || true
    [[ -f "$RETRIEVER_CONFIG" ]] || { echo "Missing $RETRIEVER_CONFIG" >&2; exit 2; }
    command="env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 \
'$PYTHON_BIN' -u '$REPO_ROOT/scripts/serving/retriever_serving.py' \
--config '$RETRIEVER_CONFIG' --num_retriever 1 --port '$RETRIEVER_PORT'"
    tmux new-session -d -s "$session" \
      "cd '$REPO_ROOT' && $command >>'$SERVICE_RUN_DIR/tmux_logs/retriever.log' 2>&1"
    echo "Started retriever session $session"
    for _ in $(seq 1 360); do
      curl -fsS "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null 2>&1 && break
      sleep 5
    done
  fi
  check_retriever
}

start_experiment() {
  local limit="$1" default_name="$2"
  start_services
  [[ -s "$DATA_DIR/prepared/questions.no_labels.jsonl" ]] || {
    echo "HotpotQA is not prepared. Run: $0 prepare" >&2; exit 2;
  }
  RUN_DIR="${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/hotpot_multisearch_budget/$default_name}"
  SESSION_NAME="${SESSION_NAME:-${default_name}_experiment}"
  LOG_DIR="$RUN_DIR/tmux_logs"; mkdir -p "$LOG_DIR"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session already exists: $SESSION_NAME"; return
  fi
  detected_pool="${SERVER_MAX_TOTAL_TOKENS:-}"
  if [[ -z "$detected_pool" && -s "$SERVICE_RUN_DIR/model_token_pool.txt" ]]; then
    detected_pool="$(cat "$SERVICE_RUN_DIR/model_token_pool.txt")"
  fi
  [[ -n "$detected_pool" ]] || {
    echo "Set SERVER_MAX_TOTAL_TOKENS to the minimum max_total_num_tokens in model logs." >&2
    exit 2
  }
  [[ "$detected_pool" -ge 9000 ]] || {
    echo "Detected token pool $detected_pool is too small for native-thinking max_new_tokens=8192." >&2
    echo "Increase MODEL_MEM_FRACTION or lower ACTION/ANSWER_MAX_NEW_TOKENS deliberately." >&2
    exit 2
  }
  command="export RUN_DIR='$RUN_DIR' LIMIT='$limit' DATA_DIR='$DATA_DIR' \
PYTHON_BIN='$PYTHON_BIN' MODEL_PATH='$MODEL_PATH' BASE_PORT='$BASE_PORT' \
RETRIEVER_URL='http://127.0.0.1:$RETRIEVER_PORT' \
CONTEXT_LENGTH='$CONTEXT_LENGTH' SERVER_MAX_TOTAL_TOKENS='$detected_pool' \
WORKERS_PER_GPU='$WORKERS_PER_GPU'"
  for name in BUDGETS MAX_SEARCHES \
              ACTION_MAX_NEW_TOKENS ANSWER_MAX_NEW_TOKENS \
              TIMEOUT RETRY_FAILURES \
              RESOURCE_INTERVAL_SECONDS RETRIEVAL_CACHE; do
    value="${!name:-}"
    [[ -n "$value" ]] && command+=" $name='$value'"
  done
  command+="; bash '$SCRIPT_DIR/orchestrate.sh' all"
  tmux new-session -d -s "$SESSION_NAME" \
    "cd '$REPO_ROOT' && $command >>'$LOG_DIR/coordinator.log' 2>&1"
  echo "Started $SESSION_NAME"
  echo "Run directory: $RUN_DIR"
  echo "Follow: tail -f $LOG_DIR/coordinator.log"
}

case "${1:-status}" in
  prepare)
    DATA_DIR="$DATA_DIR" PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/download_hotpotqa.sh"
    ;;
  services) start_services ;;
  check) check_models; check_retriever ;;
  probe5) start_experiment 5 hotpotqa_probe5_v2 ;;
  pilot100) start_experiment 100 hotpotqa_pilot100_v2 ;;
  full) start_experiment 0 hotpotqa_full7405_v2 ;;
  status)
    tmux ls 2>/dev/null || true
    if [[ -n "${RUN_DIR:-}" ]]; then
      for log in "$RUN_DIR"/tmux_logs/coordinator.log \
                 "$RUN_DIR"/tmux_logs/root_worker_*.log \
                 "$RUN_DIR"/tmux_logs/trajectory_worker_*.log \
                 "$RUN_DIR"/tmux_logs/resources.log; do
        [[ -f "$log" ]] && { echo "===== $log ====="; tail -n 12 "$log"; }
      done
    else
      echo "Set RUN_DIR to display experiment logs."
    fi
    ;;
  stop-experiment)
    [[ -n "${SESSION_NAME:-}" ]] || { echo "Set SESSION_NAME" >&2; exit 2; }
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    echo "Stopped $SESSION_NAME; model/retriever services retained."
    ;;
  stop-services)
    for gpu in $(seq 0 3); do
      tmux kill-session -t "hotpot_v2_model_g$gpu" 2>/dev/null || true
    done
    tmux kill-session -t hotpot_retriever 2>/dev/null || true
    echo "Stopped only hotpot_v2 service sessions. Experiment files were retained."
    ;;
  *) echo "Usage: $0 {prepare|services|check|probe5|pilot100|full|status|stop-experiment|stop-services}" >&2; exit 2 ;;
esac
