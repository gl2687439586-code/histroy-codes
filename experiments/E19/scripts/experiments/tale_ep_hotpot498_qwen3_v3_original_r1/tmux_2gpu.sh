#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REBUILD_ROOT="${REBUILD_ROOT:-/home/researcher/smartsearch_rebuild}"
REPO_ROOT="${REPO_ROOT:-$REBUILD_ROOT/SmartSearch}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-$REBUILD_ROOT/outputs/tale_ep_hotpot500_qwen3_v3_original/services_g2g3}"
RUN_DIR_WAS_SET="${RUN_DIR+x}"
RUN_DIR="${RUN_DIR:-$REBUILD_ROOT/outputs/tale_ep_hotpot500_qwen3_v3_original/full500_r2_t01}"
SESSION_NAME_WAS_SET="${SESSION_NAME+x}"
SESSION_NAME="${SESSION_NAME:-tale_ep_v3_original_hotpot500_full}"
GPU_IDS="${GPU_IDS:-2,3}"
BASE_PORT="${BASE_PORT:-9103}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-40960}"
MODEL_MEM_FRACTION="${MODEL_MEM_FRACTION:-0.84}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-1}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-24497}"
ROLLOUTS="${ROLLOUTS:-1}"
ESTIMATOR_FORMAT_RETRY_PASSES="${ESTIMATOR_FORMAT_RETRY_PASSES:-0}"

service_env=(
  SERVICE_RUN_DIR="$SERVICE_RUN_DIR" GPU_IDS="$GPU_IDS" BASE_PORT="$BASE_PORT"
  CONTEXT_LENGTH="$CONTEXT_LENGTH" MODEL_MEM_FRACTION="$MODEL_MEM_FRACTION"
  MAX_RUNNING_REQUESTS="$MAX_RUNNING_REQUESTS"
  SERVER_MAX_TOTAL_TOKENS="$SERVER_MAX_TOTAL_TOKENS"
)

start_services() {
  env "${service_env[@]}" bash "$SCRIPT_DIR/services.sh" start
  if [[ -s "$SERVICE_RUN_DIR/model_token_pool.txt" ]]; then
    SERVER_MAX_TOTAL_TOKENS="$(cat "$SERVICE_RUN_DIR/model_token_pool.txt")"
  fi
}

start_run() {
  local limit="$1" default_session="$2"
  start_services
  [[ -n "$SESSION_NAME_WAS_SET" ]] || SESSION_NAME="$default_session"
  mkdir -p "$RUN_DIR/tmux_logs"
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session already exists: $SESSION_NAME"; return 0
  fi
  tmux new-session -d -s "$SESSION_NAME" \
    "cd '$REPO_ROOT' && env RUN_DIR='$RUN_DIR' LIMIT='$limit' \
SERVICE_RUN_DIR='$SERVICE_RUN_DIR' SERVER_MAX_TOTAL_TOKENS='$SERVER_MAX_TOTAL_TOKENS' \
ROLLOUTS='$ROLLOUTS' ESTIMATOR_FORMAT_RETRY_PASSES='$ESTIMATOR_FORMAT_RETRY_PASSES' \
MODEL_URL_0='http://127.0.0.1:$BASE_PORT' MODEL_URL_1='http://127.0.0.1:$((BASE_PORT + 1))' \
bash '$SCRIPT_DIR/run.sh' all >>'$RUN_DIR/tmux_logs/coordinator.log' 2>&1"
  echo "Started tmux session: $SESSION_NAME"
  echo "Run directory: $RUN_DIR"
  echo "Follow: tail -f $RUN_DIR/tmux_logs/coordinator.log"
}

case "${1:-status}" in
  services) start_services ;;
  probe5)
    [[ -n "$RUN_DIR_WAS_SET" ]] || RUN_DIR="$REBUILD_ROOT/outputs/tale_ep_hotpot500_qwen3_v3_original/gate5_r2_t01"
    start_run 5 "tale_ep_v3_original_hotpot_probe5"
    ;;
  full|resume)
    [[ -n "$RUN_DIR_WAS_SET" ]] || RUN_DIR="$REBUILD_ROOT/outputs/tale_ep_hotpot500_qwen3_v3_original/full500_r2_t01"
    start_run 500 "tale_ep_v3_original_hotpot500_full"
    ;;
  analyze|audit)
    env RUN_DIR="$RUN_DIR" LIMIT="${LIMIT:-500}" \
      SERVER_MAX_TOTAL_TOKENS="$SERVER_MAX_TOTAL_TOKENS" \
      bash "$SCRIPT_DIR/run.sh" analyze
    ;;
  status)
    env "${service_env[@]}" bash "$SCRIPT_DIR/services.sh" status
    echo "===== experiment ====="
    tmux ls 2>/dev/null | grep -E 'tale_ep_v3_original_hotpot' || true
    [[ -s "$RUN_DIR/tmux_logs/coordinator.log" ]] && tail -n 40 "$RUN_DIR/tmux_logs/coordinator.log" || true
    ;;
  stop-services) env "${service_env[@]}" bash "$SCRIPT_DIR/services.sh" stop ;;
  *) echo "Usage: $0 {services|probe5|full|resume|analyze|audit|status|stop-services}" >&2; exit 2 ;;
esac
