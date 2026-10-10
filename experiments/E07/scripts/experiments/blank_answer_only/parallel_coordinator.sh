#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/blank_answer_only/bamboogle_qwen3_v5}"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
ROOT_CACHE="${ROOT_CACHE:-/home/researcher/smartsearch_rebuild/outputs/read_budget_surface/bamboogle_qwen3_fresh_v1/root_cache.jsonl}"
PILOT_RECORDS="${PILOT_RECORDS:-/home/researcher/smartsearch_rebuild/outputs/blank_evidence_budget/bamboogle_qwen3_v4_gate5_allbudgets_answer3072/probe5/records.jsonl}"
BASE_PORT="${BASE_PORT:-9101}"
NUM_SHARDS="${NUM_SHARDS:-4}"
TOTAL_RECORDS="${TOTAL_RECORDS:-1875}"
RESOURCE_INTERVAL_SECONDS="${RESOURCE_INTERVAL_SECONDS:-300}"
ESTIMATE_SAFETY_MULTIPLIER="${ESTIMATE_SAFETY_MULTIPLIER:-1.25}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-10086}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6144}"
TIMEOUT="${TIMEOUT:-900}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-2026}"
ROOT_SCOPE="${ROOT_SCOPE:-all}"
CONDITION_IDS="${CONDITION_IDS:-}"
RETRY_FAILURES="${RETRY_FAILURES:-0}"
ALLOW_UNVERIFIED_SERVER="${ALLOW_UNVERIFIED_SERVER:-0}"
LOG_DIR="$RUN_DIR/tmux_logs"
mkdir -p "$LOG_DIR" "$RUN_DIR/shards"

[[ "$NUM_SHARDS" -eq 4 ]] || { echo "Coordinator requires four shards." >&2; exit 2; }
[[ "$TOTAL_RECORDS" -eq 1875 ]] || {
  echo "Formal Bamboogle run requires TOTAL_RECORDS=1875." >&2; exit 2;
}

exec 9>"$RUN_DIR/coordinator.lock"
flock -n 9 || {
  echo "Another coordinator already owns $RUN_DIR/coordinator.lock" >&2
  exit 2
}

wait_model() {
  local port="$1" body
  for _ in $(seq 1 240); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      body="$(curl -fsS "http://127.0.0.1:$port/get_model_info" || true)"
      [[ "$body" == *"Qwen3-8B"* ]] || {
        echo "Port $port is not verified Qwen3-8B." >&2; return 1;
      }
      return 0
    fi
    sleep 5
  done
  echo "Qwen3 server timeout on port $port" >&2
  return 1
}

for shard in $(seq 0 3); do wait_model "$((BASE_PORT + shard))"; done

started_epoch="$(date +%s)"
printf '%s\n' "$started_epoch" >"$RUN_DIR/started_epoch.txt"

"$PYTHON_BIN" "$SCRIPT_DIR/estimate_runtime.py" initial \
  --pilot_records "$PILOT_RECORDS" --total_records "$TOTAL_RECORDS" \
  --workers 4 --safety_multiplier "$ESTIMATE_SAFETY_MULTIPLIER" \
  --output "$RUN_DIR/runtime_estimate_initial.json" \
  | tee "$LOG_DIR/runtime_estimate_initial.log"

resource_monitor() {
  while true; do
    {
      echo
      echo "[$(date --iso-8601=seconds)] RESOURCE AND ETA CHECK"
      free -h
      df -h /home/researcher
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
      "$PYTHON_BIN" "$SCRIPT_DIR/estimate_runtime.py" progress \
        --run_dir "$RUN_DIR" --total_records "$TOTAL_RECORDS" --workers 4 \
        --started_epoch "$started_epoch" \
        --output "$RUN_DIR/runtime_estimate_latest.json"
    } >>"$LOG_DIR/resources_and_eta.log" 2>&1
    sleep "$RESOURCE_INTERVAL_SECONDS"
  done
}

resource_monitor &
monitor_pid="$!"
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT

pids=()
for shard in $(seq 0 3); do
  port="$((BASE_PORT + shard))"
  (
    export RUN_DIR MODEL_PATH ROOT_CACHE PYTHON_BIN NUM_SHARDS
    export CONTEXT_LENGTH SERVER_MAX_TOTAL_TOKENS MAX_NEW_TOKENS
    export TIMEOUT TEMPERATURE SEED ROOT_SCOPE CONDITION_IDS
    export RETRY_FAILURES ALLOW_UNVERIFIED_SERVER
    export SHARD_ID="$shard"
    export MODEL_URL="http://127.0.0.1:$port"
    export MODEL_SERVER_COMMAND="${MODEL_SERVER_COMMAND:-captured_in_service_snapshot}"
    export SERVICE_SNAPSHOT_PATH="$RUN_DIR/service_snapshot.txt"
    bash "$SCRIPT_DIR/run.sh" shard-full
  ) >"$LOG_DIR/worker_$shard.log" 2>&1 &
  pids+=("$!")
  echo "Started shard $shard pid=$! model_port=$port"
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Shard $index failed; inspect $LOG_DIR/worker_$index.log" >&2
    failed=1
  fi
done
if [[ "$failed" -ne 0 ]]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/estimate_runtime.py" progress \
    --run_dir "$RUN_DIR" --total_records "$TOTAL_RECORDS" --workers 4 \
    --started_epoch "$started_epoch" \
    --output "$RUN_DIR/runtime_estimate_latest.json" || true
  echo "At least one shard failed. Preserve record_parts and resume after diagnosis." >&2
  exit 1
fi

export RUN_DIR MODEL_PATH ROOT_CACHE PYTHON_BIN NUM_SHARDS
export CONTEXT_LENGTH SERVER_MAX_TOTAL_TOKENS MAX_NEW_TOKENS
export TIMEOUT TEMPERATURE SEED ROOT_SCOPE CONDITION_IDS
export RETRY_FAILURES ALLOW_UNVERIFIED_SERVER
bash "$SCRIPT_DIR/run.sh" finalize >"$LOG_DIR/finalize.log" 2>&1
"$PYTHON_BIN" "$SCRIPT_DIR/estimate_runtime.py" progress \
  --run_dir "$RUN_DIR" --total_records "$TOTAL_RECORDS" --workers 4 \
  --started_epoch "$started_epoch" \
  --output "$RUN_DIR/runtime_estimate_final.json" \
  >"$LOG_DIR/runtime_estimate_final.log"
echo "FOUR-GPU ANSWER-ONLY EXPERIMENT COMPLETE: $RUN_DIR"
