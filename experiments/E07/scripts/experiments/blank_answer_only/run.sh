#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
MODEL_URL="${MODEL_URL:-http://127.0.0.1:9101}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-10086}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6144}"
TEMPERATURE="${TEMPERATURE:-0.0}"
SEED="${SEED:-2026}"
TIMEOUT="${TIMEOUT:-900}"
ROOT_SCOPE="${ROOT_SCOPE:-all}"
CONDITION_IDS="${CONDITION_IDS:-}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_ID="${SHARD_ID:-0}"
RETRY_FAILURES="${RETRY_FAILURES:-0}"
ALLOW_UNVERIFIED_SERVER="${ALLOW_UNVERIFIED_SERVER:-0}"

RUN_DIR="${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/blank_answer_only/bamboogle_qwen3_v5}"
ROOT_CACHE="${ROOT_CACHE:-/home/researcher/smartsearch_rebuild/outputs/read_budget_surface/bamboogle_qwen3_fresh_v1/root_cache.jsonl}"
RECORDS="$RUN_DIR/records.jsonl"
mkdir -p "$RUN_DIR"

run_experiment() {
  local output_dir="$1" limit="$2" shard_id="$3" num_shards="$4" dry_flag="${5:-}"
  local optional_args=(--resume)
  [[ -f "$ROOT_CACHE" ]] || {
    echo "Missing frozen ROOT_CACHE: $ROOT_CACHE" >&2
    exit 2
  }
  [[ "$RETRY_FAILURES" == "1" ]] && optional_args+=(--retry_failures)
  [[ "$ALLOW_UNVERIFIED_SERVER" == "1" ]] && optional_args+=(--allow_unverified_server)
  [[ -n "$dry_flag" ]] && optional_args+=("$dry_flag")
  "$PYTHON_BIN" "$SCRIPT_DIR/run_answer_only.py" \
    --root_cache "$ROOT_CACHE" --output_dir "$output_dir" \
    --condition_ids "$CONDITION_IDS" --root_scope "$ROOT_SCOPE" \
    --model_url "$MODEL_URL" --model_path "$MODEL_PATH" \
    --context_length "$CONTEXT_LENGTH" \
    --server_max_total_tokens "$SERVER_MAX_TOTAL_TOKENS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" --seed "$SEED" --timeout "$TIMEOUT" \
    --limit "$limit" --shard_id "$shard_id" --num_shards "$num_shards" \
    "${optional_args[@]}"
}

merge_shards() {
  local shard_dirs=() shard
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    shard_dirs+=("$RUN_DIR/shards/shard_$shard")
  done
  "$PYTHON_BIN" "$SCRIPT_DIR/merge_shards.py" \
    --shard_dirs "${shard_dirs[@]}" --output_dir "$RUN_DIR"
}

analyze() {
  [[ -f "$RECORDS" ]] || { echo "Missing merged records: $RECORDS" >&2; exit 2; }
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_answer_only.py" \
    --records "$RECORDS" --output_dir "$RUN_DIR/analysis"
}

lexical_hit() {
  [[ -f "$RECORDS" ]] || { echo "Missing merged records: $RECORDS" >&2; exit 2; }
  "$PYTHON_BIN" "$SCRIPT_DIR/offline_lexical_hit.py" \
    --records "$RECORDS" --output_dir "$RUN_DIR/offline_lexical_hit"
}

case "${1:-help}" in
  self-test) "$PYTHON_BIN" "$SCRIPT_DIR/self_test.py" ;;
  check-model) "$PYTHON_BIN" "$SCRIPT_DIR/probe_services.py" --model_url "$MODEL_URL" ;;
  dry-run5) run_experiment "$RUN_DIR/dry_run5" 5 0 1 --dry_run ;;
  dry-run-full|dry-run) run_experiment "$RUN_DIR/dry_run_full" 0 0 1 --dry_run ;;
  probe5) run_experiment "$RUN_DIR/probe5" 5 0 1 ;;
  shard-full) run_experiment "$RUN_DIR/shards/shard_$SHARD_ID" 0 "$SHARD_ID" "$NUM_SHARDS" ;;
  full-serial) run_experiment "$RUN_DIR/serial" 0 0 1 ;;
  merge) merge_shards ;;
  analyze) analyze ;;
  lexical-hit) lexical_hit ;;
  finalize) merge_shards; analyze ;;
  syntax-check) "$PYTHON_BIN" -m compileall -q "$SCRIPT_DIR" ;;
  help|*)
    echo "Usage: bash scripts/experiments/blank_answer_only/run.sh COMMAND"
    echo "Commands: self-test check-model dry-run5 dry-run-full probe5"
    echo "          shard-full full-serial merge analyze lexical-hit finalize syntax-check"
    echo "Design: 125 questions x 15 budgets = 1,875 records/model calls."
    ;;
esac
