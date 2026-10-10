#!/usr/bin/env bash
set -euo pipefail

REBUILD_ROOT="${REBUILD_ROOT:-/home/researcher/smartsearch_rebuild}"
REPO_ROOT="${REPO_ROOT:-$REBUILD_ROOT/SmartSearch}"
EXP_DIR="${EXP_DIR:-$REPO_ROOT/scripts/experiments/plan_and_budget_hotpot_agent_v2_soft}"
RUN_DIR="${RUN_DIR:-$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/hotpot500_qwen3_llama31_soft_v2}"
SERVICE_RUN_DIR="${SERVICE_RUN_DIR:-$REBUILD_ROOT/outputs/plan_and_budget_hotpot_agent_v2_soft/services_g2g3}"
SESSION_NAME="${SESSION_NAME:-pandb_hotpot500_soft_v2}"
mkdir -p "$RUN_DIR/tmux_logs"

case "${1:-}" in
  services)
    env REBUILD_ROOT="$REBUILD_ROOT" REPO_ROOT="$REPO_ROOT" EXP_DIR="$EXP_DIR" \
      SERVICE_RUN_DIR="$SERVICE_RUN_DIR" bash "$EXP_DIR/services.sh" start
    ;;
  gate5|full)
    mode="$1"
    tmux has-session -t "$SESSION_NAME" 2>/dev/null && {
      echo "Session already exists: $SESSION_NAME" >&2
      exit 1
    }
    tmux new-session -d -s "$SESSION_NAME" \
      "cd '$REPO_ROOT' && env REBUILD_ROOT='$REBUILD_ROOT' REPO_ROOT='$REPO_ROOT' EXP_DIR='$EXP_DIR' RUN_DIR='$RUN_DIR' SERVICE_RUN_DIR='$SERVICE_RUN_DIR' N_ROLLOUTS=1 MAX_SEARCHES_PER_SUBQUESTION=4 REASONER_GENERATION_MAX_TOKENS='${REASONER_GENERATION_MAX_TOKENS:-12288}' bash '$EXP_DIR/orchestrate.sh' '$mode' >>'$RUN_DIR/tmux_logs/coordinator.log' 2>&1"
    echo "Started $mode in tmux: $SESSION_NAME"
    echo "Follow: tail -f $RUN_DIR/tmux_logs/coordinator.log"
    ;;
  all)
    env REBUILD_ROOT="$REBUILD_ROOT" REPO_ROOT="$REPO_ROOT" EXP_DIR="$EXP_DIR" \
      SERVICE_RUN_DIR="$SERVICE_RUN_DIR" bash "$EXP_DIR/services.sh" start
    "$0" full
    ;;
  status)
    env REBUILD_ROOT="$REBUILD_ROOT" REPO_ROOT="$REPO_ROOT" EXP_DIR="$EXP_DIR" \
      SERVICE_RUN_DIR="$SERVICE_RUN_DIR" bash "$EXP_DIR/services.sh" status
    echo "===== coordinator ====="
    tail -n 80 "$RUN_DIR/tmux_logs/coordinator.log" 2>/dev/null || true
    ;;
  *) echo "Usage: $0 {services|gate5|full|all|status}" >&2; exit 2 ;;
esac
