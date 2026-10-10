#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-/home/researcher/smartsearch_rebuild/data/fanoutqa}"
QUESTIONS="${QUESTIONS:-$DATA_DIR/prepared/questions.no_labels.jsonl}"
LABELS="${LABELS:-$DATA_DIR/prepared/gold_labels.sealed.jsonl}"
RUN_DIR="${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/fanoutqa_concat100_full310_v2_wiki2023/full310}"
ROOT_RUN_DIR="$RUN_DIR/roots"
ROOTS="$ROOT_RUN_DIR/roots.merged.jsonl.gz"
BUDGET_DIR="$RUN_DIR/budget"
BLANK_DIR="$RUN_DIR/blank"
RETRIEVAL_CACHE="${RETRIEVAL_CACHE:-$RUN_DIR/retrieval_cache.sqlite}"
BASE_PORT="${BASE_PORT:-9103}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:6008}"
RETRIEVER_BACKEND="${RETRIEVER_BACKEND:-fanoutqa_wikipedia_en_all_nopic_2023-09_kiwix_passage100_upto5}"
LIMIT="${LIMIT:-0}"
EXPOSURES="${EXPOSURES:-}"
MAX_SEARCHES="${MAX_SEARCHES:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-40960}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-24497}"
ACTION_MAX_NEW_TOKENS="${ACTION_MAX_NEW_TOKENS:-8192}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-8192}"
NUM_GPUS="${NUM_GPUS:-2}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
NUM_WORKERS="$((NUM_GPUS * WORKERS_PER_GPU))"
TIMEOUT="${TIMEOUT:-900}"
RETRY_FAILURES="${RETRY_FAILURES:-0}"
RESOURCE_INTERVAL_SECONDS="${RESOURCE_INTERVAL_SECONDS:-300}"
LOG_DIR="$RUN_DIR/tmux_logs"
mkdir -p "$LOG_DIR" "$ROOT_RUN_DIR/shards" \
  "$BUDGET_DIR/shards" "$BLANK_DIR/shards"

[[ "$NUM_GPUS" == "2" ]] || {
  echo "Formal protocol requires two model GPUs; got $NUM_GPUS" >&2
  exit 2
}
optional=()
[[ "$RETRY_FAILURES" == "1" ]] && optional+=(--retry_failures)

check_inputs() {
  [[ -s "$QUESTIONS" ]] || { echo "Missing $QUESTIONS; run prepare" >&2; exit 2; }
  [[ -s "$LABELS" ]] || { echo "Missing sealed labels $LABELS" >&2; exit 2; }
}

resource_monitor() {
  while true; do
    {
      echo
      echo "[$(date --iso-8601=seconds)] FANOUTQA RESOURCE CHECK"
      free -h
      df -h /home/researcher
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
      for db in "$ROOT_RUN_DIR"/shards/*/state.sqlite \
                "$BUDGET_DIR"/shards/*/state.sqlite \
                "$BLANK_DIR"/shards/*/state.sqlite; do
        [[ -f "$db" ]] || continue
        printf '%s records=' "$db"
        sqlite3 "$db" 'select count(*) from records;' 2>/dev/null || echo unknown
      done
    } >>"$LOG_DIR/resources.log" 2>&1
    sleep "$RESOURCE_INTERVAL_SECONDS"
  done
}

dry_run() {
  check_inputs
  "$PYTHON_BIN" "$SCRIPT_DIR/build_roots.py" \
    --questions "$QUESTIONS" --output_dir "$ROOT_RUN_DIR/dry" \
    --retrieval_cache "$RETRIEVAL_CACHE" \
    --model_url "http://127.0.0.1:$BASE_PORT" \
    --model_path "$MODEL_PATH" --retriever_url "$RETRIEVER_URL" \
    --retriever_backend "$RETRIEVER_BACKEND" \
    --limit "$LIMIT" --shard_id 0 --num_shards "$NUM_WORKERS" --dry_run
  if [[ -s "$ROOTS" ]]; then
    for mode in budget blank; do
      exposure_args=()
      [[ "$mode" == "budget" && -n "$EXPOSURES" ]] && \
        exposure_args=(--exposures "$EXPOSURES")
      "$PYTHON_BIN" "$SCRIPT_DIR/run_trajectories.py" \
        --roots "$ROOTS" --output_dir "$RUN_DIR/$mode/dry" \
        --retrieval_cache "$RETRIEVAL_CACHE" \
        --model_url "http://127.0.0.1:$BASE_PORT" \
        --model_path "$MODEL_PATH" --retriever_url "$RETRIEVER_URL" \
        --retriever_backend "$RETRIEVER_BACKEND" \
        --condition_mode "$mode" "${exposure_args[@]}" \
        --limit "$LIMIT" --max_searches "$MAX_SEARCHES" \
        --shard_id 0 --num_shards "$NUM_WORKERS" --dry_run
    done
  else
    count="$(wc -l < "$QUESTIONS" | tr -d ' ')"
    if [[ "$LIMIT" != "0" && "$LIMIT" -lt "$count" ]]; then count="$LIMIT"; fi
    echo "Questions: $count. Budget count becomes exact after shared roots."
    echo "Blank records: $count; one full Top-5 natural-search trajectory per question."
  fi
}

run_roots() {
  check_inputs
  pids=()
  for shard in $(seq 0 $((NUM_WORKERS - 1))); do
    port="$((BASE_PORT + (shard % NUM_GPUS)))"
    (
      "$PYTHON_BIN" "$SCRIPT_DIR/build_roots.py" \
        --questions "$QUESTIONS" \
        --output_dir "$ROOT_RUN_DIR/shards/shard_$shard" \
        --retrieval_cache "$RETRIEVAL_CACHE" \
        --model_url "http://127.0.0.1:$port" --model_path "$MODEL_PATH" \
        --retriever_url "$RETRIEVER_URL" --limit "$LIMIT" \
        --retriever_backend "$RETRIEVER_BACKEND" \
        --shard_id "$shard" --num_shards "$NUM_WORKERS" \
        --action_max_new_tokens "$ACTION_MAX_NEW_TOKENS" \
        --timeout "$TIMEOUT" "${optional[@]}"
    ) >"$LOG_DIR/root_worker_$shard.log" 2>&1 &
    pids+=("$!")
    echo "Started root shard=$shard pid=$! port=$port"
  done
  failed=0
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || { echo "Root shard $index failed" >&2; failed=1; }
  done
  [[ "$failed" == "0" ]] || exit 1
  shard_dirs=()
  for shard in $(seq 0 $((NUM_WORKERS - 1))); do
    shard_dirs+=("$ROOT_RUN_DIR/shards/shard_$shard")
  done
  "$PYTHON_BIN" "$SCRIPT_DIR/merge_roots.py" \
    --questions "$QUESTIONS" --shard_dirs "${shard_dirs[@]}" \
    --output "$ROOTS" --limit "$LIMIT" \
    >"$LOG_DIR/merge_roots.log" 2>&1
}

run_condition() {
  local mode="$1" out_dir trajectories
  local -a exposure_args=()
  [[ "$mode" == "budget" && -n "$EXPOSURES" ]] && \
    exposure_args=(--exposures "$EXPOSURES")
  [[ -s "$ROOTS" ]] || { echo "Missing merged roots: $ROOTS" >&2; exit 2; }
  out_dir="$RUN_DIR/$mode"
  trajectories="$out_dir/trajectories.merged.jsonl.gz"
  mkdir -p "$out_dir/shards"
  pids=()
  for shard in $(seq 0 $((NUM_WORKERS - 1))); do
    port="$((BASE_PORT + (shard % NUM_GPUS)))"
    (
      "$PYTHON_BIN" "$SCRIPT_DIR/run_trajectories.py" \
        --roots "$ROOTS" --output_dir "$out_dir/shards/shard_$shard" \
        --retrieval_cache "$RETRIEVAL_CACHE" \
        --model_url "http://127.0.0.1:$port" --model_path "$MODEL_PATH" \
        --retriever_url "$RETRIEVER_URL" --condition_mode "$mode" \
        --retriever_backend "$RETRIEVER_BACKEND" \
        "${exposure_args[@]}" --limit "$LIMIT" \
        --max_searches "$MAX_SEARCHES" --context_length "$CONTEXT_LENGTH" \
        --server_max_total_tokens "$SERVER_MAX_TOTAL_TOKENS" \
        --action_max_new_tokens "$ACTION_MAX_NEW_TOKENS" \
        --answer_max_new_tokens "$ANSWER_MAX_NEW_TOKENS" \
        --repair_max_new_tokens 0 --timeout "$TIMEOUT" \
        --shard_id "$shard" --num_shards "$NUM_WORKERS" \
        "${optional[@]}"
    ) >"$LOG_DIR/${mode}_worker_$shard.log" 2>&1 &
    pids+=("$!")
    echo "Started $mode shard=$shard pid=$! port=$port"
  done
  failed=0
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || {
      echo "$mode shard $index failed" >&2
      failed=1
    }
  done
  [[ "$failed" == "0" ]] || exit 1
  shard_dirs=()
  for shard in $(seq 0 $((NUM_WORKERS - 1))); do
    shard_dirs+=("$out_dir/shards/shard_$shard")
  done
  "$PYTHON_BIN" "$SCRIPT_DIR/merge_trajectories.py" \
    --roots "$ROOTS" --shard_dirs "${shard_dirs[@]}" \
    --output "$trajectories" --condition_mode "$mode" \
    "${exposure_args[@]}" --limit "$LIMIT" \
    >"$LOG_DIR/merge_$mode.log" 2>&1
}

analyze_one() {
  local mode="$1" input="$RUN_DIR/$1/trajectories.merged.jsonl.gz"
  local output="$RUN_DIR/analysis/$1"
  [[ -s "$input" ]] || { echo "Missing $input" >&2; exit 2; }
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_fanoutqa.py" \
    --trajectories "$input" --labels "$LABELS" \
    --condition_mode "$mode" --output_dir "$output" \
    >"$LOG_DIR/analyze_$mode.log" 2>&1
  "$PYTHON_BIN" "$SCRIPT_DIR/locate_necessary_evidence.py" \
    --trajectories "$input" --labels "$LABELS" \
    --output_dir "$output/evidence_positions" \
    >"$LOG_DIR/positions_$mode.log" 2>&1
}

analyze_all() {
  analyze_one budget
  analyze_one blank
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_answer_flips.py" \
    --scored_records "$RUN_DIR/analysis/budget/scored_records.jsonl.gz" \
    --output_dir "$RUN_DIR/analysis/budget/answer_flip" \
    --primary_metric official_strict_accuracy \
    --bootstrap_repetitions 10000 --seed 2026 \
    >"$LOG_DIR/analyze_answer_flips.log" 2>&1
  "$PYTHON_BIN" "$SCRIPT_DIR/compare_blank_budget.py" \
    --budget_scored "$RUN_DIR/analysis/budget/scored_records.jsonl.gz" \
    --blank_scored "$RUN_DIR/analysis/blank/scored_records.jsonl.gz" \
    --output_dir "$RUN_DIR/analysis/comparison" \
    >"$LOG_DIR/compare_blank_budget.log" 2>&1
}

case "${1:-all}" in
  dry-run) dry_run ;;
  roots) run_roots ;;
  budget) run_condition budget ;;
  blank) run_condition blank ;;
  analyze) analyze_all ;;
  all)
    check_inputs
    resource_monitor & monitor_pid="$!"
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    dry_run
    run_roots
    run_condition budget
    run_condition blank
    analyze_all
    echo "FANOUTQA BUDGET + BLANK EXPERIMENT COMPLETE: $RUN_DIR"
    ;;
  *) echo "Usage: $0 {dry-run|roots|budget|blank|analyze|all}" >&2; exit 2 ;;
esac
