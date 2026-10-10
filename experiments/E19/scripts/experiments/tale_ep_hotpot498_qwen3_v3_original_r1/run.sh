#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REBUILD_ROOT="${REBUILD_ROOT:-/home/researcher/smartsearch_rebuild}"
REPO_ROOT="${REPO_ROOT:-$REBUILD_ROOT/SmartSearch}"
PYTHON_BIN="${PYTHON_BIN:-$REBUILD_ROOT/envs/src-core/bin/python}"
MODEL_PATH="${MODEL_PATH:-$REBUILD_ROOT/models/Qwen3-8B}"
RUN_DIR="${RUN_DIR:-$REBUILD_ROOT/outputs/tale_ep_hotpot500_qwen3_v3_original/full500_r2_t01}"
LIMIT="${LIMIT:-500}"
NUM_SHARDS=2
ROLLOUTS="${ROLLOUTS:-1}"
if [[ "$ROLLOUTS" != "1" && "$ROLLOUTS" != "2" ]]; then
  echo "ROLLOUTS must be 1 or 2" >&2
  exit 2
fi
export TALE_EP_ROLLOUTS="$ROLLOUTS"
MAX_SEARCHES=4
TOP_K=5
TEMPERATURE=0.1
TOP_P=1.0
TECHNICAL_MAX_TOKENS="${TECHNICAL_MAX_TOKENS:-8192}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-24497}"
ESTIMATOR_FORMAT_RETRY_PASSES="${ESTIMATOR_FORMAT_RETRY_PASSES:-10}"
MODEL_URL_0="${MODEL_URL_0:-http://127.0.0.1:9103}"
MODEL_URL_1="${MODEL_URL_1:-http://127.0.0.1:9104}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:6008}"
QUERIES="${QUERIES:-$SCRIPT_DIR/data/hotpot_random500_queries.jsonl}"
SOURCE_LABELS="${SOURCE_LABELS:-$SCRIPT_DIR/data/hotpot_random500_gold_labels.sealed.jsonl}"

mkdir -p "$RUN_DIR/tmux_logs"

wait_workers() {
  local label="$1"; shift
  local failed=0 pid
  for pid in "$@"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if [[ "$failed" == "1" ]]; then
    echo "$label worker failed; inspect $RUN_DIR/tmux_logs" >&2
    return 1
  fi
}

prepare() {
  "$PYTHON_BIN" "$SCRIPT_DIR/self_test.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/accounting_integration_test.py"
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_dataset.py" \
    --queries "$QUERIES" --labels "$SOURCE_LABELS" \
    --output-dir "$RUN_DIR/dataset" --expected-source 500 --limit "$LIMIT"
}

estimate() {
  mkdir -p "$RUN_DIR/estimates/shards"
  local pids=() shard url
  for shard in 0 1; do
    if [[ "$shard" == "0" ]]; then url="$MODEL_URL_0"; else url="$MODEL_URL_1"; fi
    "$PYTHON_BIN" "$SCRIPT_DIR/estimate_budgets.py" \
      --questions "$RUN_DIR/dataset/questions.no_labels.jsonl" \
      --output-dir "$RUN_DIR/estimates/shards" \
      --model-url "$url" --model-path "$MODEL_PATH" \
      --shard-id "$shard" --num-shards "$NUM_SHARDS" \
      --temperature "$TEMPERATURE" --top-p "$TOP_P" \
      >"$RUN_DIR/tmux_logs/estimate_$shard.log" 2>&1 &
    pids+=("$!")
    echo "Started TALE estimator shard=$shard pid=$! url=$url"
  done
  wait_workers "Estimator" "${pids[@]}"
  "$PYTHON_BIN" "$SCRIPT_DIR/merge_records.py" \
    --input-dir "$RUN_DIR/estimates/shards" --stage estimates \
    --expected "$LIMIT" --num-shards "$NUM_SHARDS" \
    --output "$RUN_DIR/estimates/estimates.merged.jsonl"
}

estimate_until_complete() {
  local pass
  for ((pass=1; pass<=ESTIMATOR_FORMAT_RETRY_PASSES; pass++)); do
    echo "Estimator completeness pass $pass/$ESTIMATOR_FORMAT_RETRY_PASSES"
    if estimate; then
      echo "Estimator reached all $LIMIT valid budgets on pass $pass."
      return 0
    fi
    echo "Estimator still has unresolved format failures after pass $pass; retrying only missing samples." >&2
  done
  echo "Estimator did not reach $LIMIT valid budgets after $ESTIMATOR_FORMAT_RETRY_PASSES passes." >&2
  echo "Freezing persistent estimator failures without assigning synthetic budgets." >&2
  "$PYTHON_BIN" "$SCRIPT_DIR/merge_records.py" \
    --input-dir "$RUN_DIR/estimates/shards" --stage estimates \
    --expected "$LIMIT" --num-shards "$NUM_SHARDS" --allow-partial \
    --output "$RUN_DIR/estimates/estimates.merged.jsonl"
}

trajectories() {
  mkdir -p "$RUN_DIR/trajectories/shards"
  local pids=() shard url
  for shard in 0 1; do
    if [[ "$shard" == "0" ]]; then url="$MODEL_URL_0"; else url="$MODEL_URL_1"; fi
    "$PYTHON_BIN" "$SCRIPT_DIR/run_trajectories.py" \
      --questions "$RUN_DIR/dataset/questions.no_labels.jsonl" \
      --estimates "$RUN_DIR/estimates/estimates.merged.jsonl" \
      --output-dir "$RUN_DIR/trajectories/shards" \
      --model-url "$url" --model-path "$MODEL_PATH" \
      --retriever-url "$RETRIEVER_URL" \
      --shard-id "$shard" --num-shards "$NUM_SHARDS" \
      --rollouts "$ROLLOUTS" --max-searches "$MAX_SEARCHES" --top-k "$TOP_K" \
      --technical-max-tokens "$TECHNICAL_MAX_TOKENS" \
      --server-max-total-tokens "$SERVER_MAX_TOTAL_TOKENS" \
      >"$RUN_DIR/tmux_logs/trajectory_$shard.log" 2>&1 &
    pids+=("$!")
    echo "Started trajectory shard=$shard pid=$! url=$url"
  done
  wait_workers "Trajectory" "${pids[@]}"
  local estimate_count
  estimate_count="$(wc -l < "$RUN_DIR/estimates/estimates.merged.jsonl")"
  "$PYTHON_BIN" "$SCRIPT_DIR/merge_records.py" \
    --input-dir "$RUN_DIR/trajectories/shards" --stage trajectories \
    --expected "$((estimate_count * ROLLOUTS))" --num-shards "$NUM_SHARDS" \
    --output "$RUN_DIR/trajectories/trajectories.merged.jsonl"
}

analyze() {
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze.py" \
    --questions "$RUN_DIR/dataset/questions.no_labels.jsonl" \
    --labels "$RUN_DIR/dataset/gold_labels.sealed.jsonl" \
    --estimates "$RUN_DIR/estimates/estimates.merged.jsonl" \
    --trajectories "$RUN_DIR/trajectories/trajectories.merged.jsonl" \
    --estimator-ledger-dir "$RUN_DIR/estimates/shards" \
    --solver-ledger-dir "$RUN_DIR/trajectories/shards" \
    --output-dir "$RUN_DIR/analysis"
  "$PYTHON_BIN" "$SCRIPT_DIR/audit.py" \
    --questions "$RUN_DIR/dataset/questions.no_labels.jsonl" \
    --labels "$RUN_DIR/dataset/gold_labels.sealed.jsonl" \
    --estimates "$RUN_DIR/estimates/estimates.merged.jsonl" \
    --trajectories "$RUN_DIR/trajectories/trajectories.merged.jsonl" \
    --estimator-ledger-dir "$RUN_DIR/estimates/shards" \
    --solver-ledger-dir "$RUN_DIR/trajectories/shards" \
    --analysis "$RUN_DIR/analysis/summary.json" \
    --expected-questions "$LIMIT" \
    --output "$RUN_DIR/analysis/AUDIT.json"
}

all() {
  echo "Original-prompt TALE-EP v3: questions=$LIMIT, estimates up to=$LIMIT, rollouts=$ROLLOUTS"
  echo "Temperature=0.1; rollout(s) share one estimate; max searches=4; Top-5."
  echo "Minimum physical model calls=$((LIMIT + LIMIT * ROLLOUTS)); maximum is approximately $((LIMIT + LIMIT * ROLLOUTS * 6))."
  prepare
  estimate_until_complete
  trajectories
  analyze
  date -Is >"$RUN_DIR/COMPLETED"
  echo "Completed TALE-EP run: $RUN_DIR"
}

case "${1:-all}" in
  all) all ;;
  prepare) prepare ;;
  estimate) estimate ;;
  trajectories) trajectories ;;
  analyze|audit) analyze ;;
  *) echo "Usage: $0 {all|prepare|estimate|trajectories|analyze|audit}" >&2; exit 2 ;;
esac
