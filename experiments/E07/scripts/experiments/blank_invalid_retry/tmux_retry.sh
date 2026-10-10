#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RETRY_RUN_DIR="${RETRY_RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/blank_answer_only/bamboogle_qwen3_v5_full_6144_invalid_retry_v1}"
SESSION_NAME="${SESSION_NAME:-blankanswer_v5_invalid_retry}"
LOG_DIR="$RETRY_RUN_DIR/tmux_logs"
mkdir -p "$LOG_DIR"

case "${1:-experiment}" in
  check)
    for port in 9101 9102 9103 9104; do
      printf 'port %s: ' "$port"
      curl -fsS "http://127.0.0.1:$port/get_model_info" | grep -q 'Qwen3-8B' \
        && echo verified || { echo unavailable-or-wrong-model; exit 1; }
    done
    ;;
  experiment)
    "$0" check
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
      echo "Session already exists: $SESSION_NAME"
      exit 0
    fi
    command="export RETRY_RUN_DIR='$RETRY_RUN_DIR'"
    for name in PYTHON_BIN MODEL_PATH ORIGINAL_RUN_DIR ORIGINAL_RECORDS \
                BASE_PORT CONTEXT_LENGTH SERVER_MAX_TOTAL_TOKENS \
                MAX_NEW_TOKENS TIMEOUT EXPECTED_INVALID \
                EXPECTED_SOURCE_SHA256 RETRY_FAILURES; do
      value="${!name:-}"
      [[ -n "$value" ]] && command+=" $name='$value'"
    done
    command+="; bash '$SCRIPT_DIR/parallel_retry.sh'"
    tmux new-session -d -s "$SESSION_NAME" \
      "cd '$REPO_ROOT' && $command >>'$LOG_DIR/coordinator.log' 2>&1"
    echo "Started tmux session: $SESSION_NAME"
    echo "Follow: tail -f $LOG_DIR/coordinator.log"
    ;;
  status)
    tmux has-session -t "$SESSION_NAME" 2>/dev/null \
      && echo "tmux session running: $SESSION_NAME" \
      || echo "tmux session not running: $SESSION_NAME"
    for log in "$LOG_DIR"/coordinator.log "$LOG_DIR"/worker_*.log \
               "$LOG_DIR"/merge_and_analyze.log; do
      [[ -f "$log" ]] && { echo "===== $log ====="; tail -n 15 "$log"; }
    done
    ;;
  stop)
    tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true
    echo "Stopped coordinator; model services were not stopped."
    ;;
  *)
    echo "Usage: $0 {check|experiment|status|stop}" >&2
    exit 2
    ;;
esac

