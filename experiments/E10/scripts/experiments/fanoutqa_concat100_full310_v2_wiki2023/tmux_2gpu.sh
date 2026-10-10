#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
MODEL_PYTHON="${MODEL_PYTHON:-/home/researcher/smartsearch_rebuild/envs/src-model/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
BASE_PORT="${BASE_PORT:-9103}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-40960}"
MODEL_MEM_FRACTION="${MODEL_MEM_FRACTION:-0.84}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-2}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
TOP_K="${TOP_K:-5}"
GPU_IDS="${GPU_IDS:-2,3}"
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
[[ "$NUM_GPUS" == "2" && "${GPU_ARRAY[0]}" == "2" && "${GPU_ARRAY[1]}" == "3" ]] || {
  echo "Formal two-GPU protocol requires GPU_IDS=2,3; got $GPU_IDS" >&2
  exit 2
}
RETRIEVER_PORT="${RETRIEVER_PORT:-6008}"
RETRIEVER_BACKEND="${RETRIEVER_BACKEND:-fanoutqa_wikipedia_en_all_nopic_2023-09_kiwix_passage100_upto5}"
KIWIX_PORT="${KIWIX_PORT:-8888}"
WIKI2023_ASSET_DIR="${WIKI2023_ASSET_DIR:-/home/researcher/smartsearch_rebuild/retrieval/fanoutqa_wiki2023}"
WIKI2023_ZIM_PATH="${WIKI2023_ZIM_PATH:-$WIKI2023_ASSET_DIR/wikipedia_en_all_nopic_2023-09.zim}"
KIWIX_SERVE="${KIWIX_SERVE:-$WIKI2023_ASSET_DIR/kiwix-tools/kiwix-serve}"
KIWIX_ZIMNAME="${KIWIX_ZIMNAME:-wikipedia_en_all_nopic_2023-09}"
SERVICE_RETRIEVAL_CACHE="${SERVICE_RETRIEVAL_CACHE:-$WIKI2023_ASSET_DIR/retriever_cache.sqlite}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/fanoutqa_concat100_full310_v2_wiki2023/services_g2g3}"
DATA_DIR="${DATA_DIR:-/home/researcher/smartsearch_rebuild/data/fanoutqa}"
QUESTIONS="${QUESTIONS:-$DATA_DIR/prepared/questions.no_labels.jsonl}"
LABELS="${LABELS:-$DATA_DIR/prepared/gold_labels.sealed.jsonl}"
SAMPLE_SEED="${SAMPLE_SEED:-2026}"
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

model_pid() {
  local port="$1"
  ps -eo pid=,args= | grep '[s]glang.launch_server' \
    | grep -E -- "--port[ =]+${port}([[:space:]]|$)" \
    | awk 'NR==1 {print $1}' || true
}

verify_model_config() {
  local port="$1" expected_gpu="$2" cmdline pid visible
  cmdline="$(model_cmdline "$port")"
  [[ -n "$cmdline" ]] || { echo "Cannot find model process for port $port" >&2; return 1; }
  [[ "$cmdline" == *"--context-length $CONTEXT_LENGTH"* ]] || {
    echo "Port $port is not using context-length=$CONTEXT_LENGTH" >&2
    echo "$cmdline" >&2
    return 1
  }
  [[ "$cmdline" == *"--model-path $MODEL_PATH"* ]] || {
    echo "Port $port is not serving the required model path: $MODEL_PATH" >&2
    echo "$cmdline" >&2
    return 1
  }
  [[ "$cmdline" == *"--max-running-requests $MAX_RUNNING_REQUESTS"* ]] || {
    echo "Port $port does not use max-running-requests=$MAX_RUNNING_REQUESTS" >&2
    echo "$cmdline" >&2
    return 1
  }
  pid="$(model_pid "$port")"
  [[ -n "$pid" ]] || { echo "Cannot resolve model PID for port $port" >&2; return 1; }
  visible="$(tr '\0' '\n' < "/proc/$pid/environ" \
    | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | tail -n 1)"
  [[ "$visible" == "$expected_gpu" ]] || {
    echo "Port $port uses CUDA_VISIBLE_DEVICES=${visible:-unset}; expected physical GPU$expected_gpu" >&2
    return 1
  }
}

check_models() {
  local slot port gpu
  for slot in "${!GPU_ARRAY[@]}"; do
    port="$((BASE_PORT + slot))"
    gpu="${GPU_ARRAY[$slot]}"
    body="$(curl -fsS "http://127.0.0.1:$port/get_model_info" || true)"
    [[ "$body" == *"Qwen3-8B"* ]] || {
      echo "Model port $port is unavailable or not Qwen3-8B" >&2; return 1;
    }
    verify_model_config "$port" "$gpu" || return 1
    echo "Qwen3-8B verified: physical GPU$gpu port=$port"
  done
}

wait_model() {
  local port="$1" gpu="$2"
  for _ in $(seq 1 240); do
    curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1 && {
      verify_model_config "$port" "$gpu"
      echo "Qwen3-8B ready: $port"
      return 0
    }
    sleep 5
  done
  echo "Timed out waiting for model port $port" >&2
  return 1
}

start_models() {
  local slot gpu port session command free_mib
  for slot in "${!GPU_ARRAY[@]}"; do
    gpu="${GPU_ARRAY[$slot]}"
    port="$((BASE_PORT + slot))"
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "Port $port is occupied; verifying its model configuration."
      body="$(curl -fsS "http://127.0.0.1:$port/get_model_info" || true)"
      [[ "$body" == *"Qwen3-8B"* ]] || {
        echo "Port $port is not Qwen3-8B" >&2; return 1;
      }
      verify_model_config "$port" "$gpu"
      continue
    fi
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      -i "$gpu" | tr -d ' ')"
    [[ "$free_mib" -ge 22000 ]] || {
      echo "GPU$gpu has only ${free_mib}MiB free; stop old services first." >&2
      return 1
    }
    session="fanoutqa_v2_model_g$gpu"
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
    wait_model "$port" "$gpu"
  done
  check_models
  values=()
  for gpu in "${GPU_ARRAY[@]}"; do
    value="$(grep -oE 'max_total_num_tokens=[0-9]+' \
      "$SERVICE_RUN_DIR/tmux_logs/model_g$gpu.log" 2>/dev/null \
      | tail -n 1 | cut -d= -f2 || true)"
    [[ -n "$value" ]] && values+=("$value")
  done
  if [[ "${#values[@]}" == "$NUM_GPUS" ]]; then
    printf '%s\n' "${values[@]}" | sort -n | head -n 1 \
      >"$SERVICE_RUN_DIR/model_token_pool.txt"
    echo "Detected safe per-server token pool: $(cat "$SERVICE_RUN_DIR/model_token_pool.txt")"
  else
    if [[ -n "${SERVER_MAX_TOTAL_TOKENS:-}" ]]; then
      printf '%s\n' "$SERVER_MAX_TOTAL_TOKENS" \
        >"$SERVICE_RUN_DIR/model_token_pool.txt"
      echo "Reused explicitly supplied safe token pool: $SERVER_MAX_TOTAL_TOKENS"
    else
      echo "Token-pool logs unavailable because existing services were reused." >&2
      echo "Set SERVER_MAX_TOTAL_TOKENS explicitly (previous verified value was 10086)." >&2
    fi
  fi
}

check_retriever() {
  health="$(curl -fsS "http://127.0.0.1:$RETRIEVER_PORT/health")"
  [[ "$health" == *"$RETRIEVER_BACKEND"* ]] || {
    echo "Port $RETRIEVER_PORT is not the required 2023-09 Kiwix retriever." >&2
    echo "$health" >&2
    return 1
  }
  probe_body="$(curl -sS -X POST "http://127.0.0.1:$RETRIEVER_PORT/search" \
    -H 'Content-Type: application/json' \
    -d "{\"query\":\"2023 Nobel Prize in Physics\",\"top_n\":$TOP_K,\"return_score\":false}")" || {
      echo "Retriever functional probe request failed." >&2
      return 1
    }
  if ! count="$(printf '%s' "$probe_body" \
      | "$PYTHON_BIN" -c 'import json,sys; value=json.load(sys.stdin); print(len(value) if isinstance(value,list) else -1)' \
      2>/dev/null)"; then
    echo "Retriever functional probe returned invalid JSON: $probe_body" >&2
    return 1
  fi
  [[ "$count" -ge 0 && "$count" -le "$TOP_K" ]] || {
    echo "Retriever violated K in [0,$TOP_K]; response: $probe_body" >&2
    return 1
  }
  echo "Retriever up-to-$TOP_K verified: returned K=$count; $RETRIEVER_BACKEND port=$RETRIEVER_PORT"
}

check_kiwix() {
  curl -fsS --max-time 20 -G "http://127.0.0.1:$KIWIX_PORT/search" \
    --data-urlencode "pattern=Wikipedia" \
    --data-urlencode "start=0" \
    --data-urlencode "pageLength=1" \
    --data-urlencode "books.name=$KIWIX_ZIMNAME" \
    --data-urlencode "format=xml" \
    | grep -Eq '<(rss|item|feed|entry)([[:space:]>])|<(atom:)?feed'
}

start_kiwix() {
  [[ -s "$WIKI2023_ZIM_PATH" ]] || {
    echo "Missing 2023-09 ZIM: $WIKI2023_ZIM_PATH" >&2
    echo "Run: $0 prepare-wiki" >&2
    return 1
  }
  [[ -x "$KIWIX_SERVE" ]] || {
    echo "Missing kiwix-serve: $KIWIX_SERVE" >&2
    echo "Run: $0 prepare-wiki" >&2
    return 1
  }
  if check_kiwix; then
    echo "Kiwix already healthy: port=$KIWIX_PORT"
    return 0
  fi
  start_session "fanoutqa_v2_kiwix" \
    "'$KIWIX_SERVE' --port '$KIWIX_PORT' '$WIKI2023_ZIM_PATH'" \
    "$SERVICE_RUN_DIR/tmux_logs/kiwix.log"
  for _ in $(seq 1 120); do
    check_kiwix && { echo "Kiwix ready: port=$KIWIX_PORT"; return 0; }
    sleep 5
  done
  echo "Timed out waiting for Kiwix; inspect $SERVICE_RUN_DIR/tmux_logs/kiwix.log" >&2
  return 1
}

start_retriever() {
  if curl -fsS "http://127.0.0.1:$RETRIEVER_PORT/health" >/dev/null 2>&1; then
    check_retriever || {
      echo "The required retriever occupies port $RETRIEVER_PORT, but its functional probe failed." >&2
      echo "Inspect the response above and $SERVICE_RUN_DIR/tmux_logs/wiki2023_retriever.log." >&2
      return 1
    }
    return 0
  fi
  command="env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
-u http_proxy -u https_proxy -u all_proxy \
NO_PROXY='127.0.0.1,localhost' no_proxy='127.0.0.1,localhost' \
FANOUTQA_WIKIPEDIA_TYPE=kiwix \
FANOUTQA_KIWIX_BASE='http://127.0.0.1:$KIWIX_PORT' \
FANOUTQA_KIWIX_ZIMNAME='$KIWIX_ZIMNAME' \
'$PYTHON_BIN' -u '$SCRIPT_DIR/wiki2023_retriever_service.py' \
--port '$RETRIEVER_PORT' --cache '$SERVICE_RETRIEVAL_CACHE'"
  start_session "fanoutqa_v2_wiki2023_retriever" "$command" \
    "$SERVICE_RUN_DIR/tmux_logs/wiki2023_retriever.log"
  for _ in $(seq 1 120); do
    check_retriever && return 0
    sleep 5
  done
  echo "Timed out waiting for the 2023 retriever." >&2
  return 1
}

start_services() {
  start_models
  start_kiwix
  start_retriever
  check_retriever
}

use_random_subset() {
  local size="$1" seed="${2:-$SAMPLE_SEED}" subset_dir
  [[ -s "$DATA_DIR/prepared/questions.no_labels.jsonl" ]] || {
    echo "FanOutQA is not prepared. Run: $0 prepare" >&2; exit 2;
  }
  subset_dir="$DATA_DIR/prepared/random${size}_seed${seed}"
  "$PYTHON_BIN" "$SCRIPT_DIR/sample_fanoutqa.py" \
    --questions "$DATA_DIR/prepared/questions.no_labels.jsonl" \
    --labels "$DATA_DIR/prepared/gold_labels.sealed.jsonl" \
    --output_dir "$subset_dir" --sample_size "$size" --seed "$seed"
  QUESTIONS="$subset_dir/questions.no_labels.jsonl"
  LABELS="$subset_dir/gold_labels.sealed.jsonl"
}

start_experiment() {
  local limit="$1" default_name="$2"
  start_services
  [[ -s "$QUESTIONS" && -s "$LABELS" ]] || {
    echo "FanOutQA is not prepared. Run: $0 prepare" >&2; exit 2;
  }
  RUN_DIR="${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/fanoutqa_concat100_full310_v2_wiki2023/$default_name}"
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
  command="export RUN_DIR='$RUN_DIR' LIMIT='$limit' DATA_DIR='$DATA_DIR' TOP_K='$TOP_K' \
QUESTIONS='$QUESTIONS' LABELS='$LABELS' \
PYTHON_BIN='$PYTHON_BIN' MODEL_PATH='$MODEL_PATH' BASE_PORT='$BASE_PORT' \
RETRIEVER_URL='http://127.0.0.1:$RETRIEVER_PORT' \
RETRIEVER_BACKEND='$RETRIEVER_BACKEND' \
CONTEXT_LENGTH='$CONTEXT_LENGTH' SERVER_MAX_TOTAL_TOKENS='$detected_pool' \
WORKERS_PER_GPU='$WORKERS_PER_GPU' NUM_GPUS='$NUM_GPUS' GPU_IDS='$GPU_IDS'"
  for name in EXPOSURES MAX_SEARCHES \
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
    # The official wheel contains the corrected dev file.  Install it from the
    # domestic PyPI mirror before attempting any Hugging Face URL.
    PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/install_eval_deps.sh"
    DATA_DIR="$DATA_DIR" PYTHON_BIN="$PYTHON_BIN" bash "$SCRIPT_DIR/download_fanoutqa.sh"
    WORK_ROOT=/home/researcher/smartsearch_rebuild \
      WIKI2023_ASSET_DIR="$WIKI2023_ASSET_DIR" \
      WIKI2023_ZIM_PATH="$WIKI2023_ZIM_PATH" \
      bash "$SCRIPT_DIR/download_wiki2023.sh"
    ;;
  prepare-wiki)
    WORK_ROOT=/home/researcher/smartsearch_rebuild \
      WIKI2023_ASSET_DIR="$WIKI2023_ASSET_DIR" \
      WIKI2023_ZIM_PATH="$WIKI2023_ZIM_PATH" \
      bash "$SCRIPT_DIR/download_wiki2023.sh"
    ;;
  services) start_services ;;
  check) check_models; check_kiwix; check_retriever ;;
  probe5) use_random_subset 5; start_experiment 0 fanoutqa_dev5_budget_blank_wiki2023_v2 ;;
  pilot100) use_random_subset 100; start_experiment 0 fanoutqa_dev100_budget_blank_wiki2023_v2 ;;
  full310) start_experiment 0 fanoutqa_dev310_budget_blank_wiki2023_v2 ;;
  status)
    tmux ls 2>/dev/null || true
    if [[ -n "${RUN_DIR:-}" ]]; then
      for log in "$RUN_DIR"/tmux_logs/coordinator.log \
                 "$RUN_DIR"/tmux_logs/root_worker_*.log \
                 "$RUN_DIR"/tmux_logs/budget_worker_*.log \
                 "$RUN_DIR"/tmux_logs/blank_worker_*.log \
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
    for gpu in "${GPU_ARRAY[@]}"; do
      tmux kill-session -t "fanoutqa_v2_model_g$gpu" 2>/dev/null || true
    done
    tmux kill-session -t "fanoutqa_v2_wiki2023_retriever" 2>/dev/null || true
    tmux kill-session -t "fanoutqa_v2_kiwix" 2>/dev/null || true
    echo "Stopped only FanOutQA v2 service sessions. Experiment files were retained."
    ;;
  *) echo "Usage: $0 {prepare|prepare-wiki|services|check|probe5|pilot100|full310|status|stop-experiment|stop-services}" >&2; exit 2 ;;
esac
