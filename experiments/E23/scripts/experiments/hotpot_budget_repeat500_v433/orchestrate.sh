#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
DATA_DIR="${DATA_DIR:-/home/researcher/smartsearch_rebuild/data/hotpotqa}"
QUESTIONS="${QUESTIONS:-$DATA_DIR/prepared/questions.no_labels.jsonl}"
LABELS="${LABELS:-$DATA_DIR/prepared/gold_labels.sealed.jsonl}"
RUN_DIR="${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/hotpot_concat100_random500_v4/hotpotqa_top5_random500_2gpu_g2g3_seed2026}"
ROOT_RUN_DIR="$RUN_DIR/roots"
ROOTS="$ROOT_RUN_DIR/roots.merged.jsonl.gz"
TRAJECTORY_RUN_DIR="$RUN_DIR/trajectories"
TRAJECTORIES="$TRAJECTORY_RUN_DIR/trajectories.merged.jsonl.gz"
RETRIEVAL_CACHE="${RETRIEVAL_CACHE:-$RUN_DIR/retrieval_cache.sqlite}"
BASE_PORT="${BASE_PORT:-9103}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:6008}"
LIMIT="${LIMIT:-0}"
EXPOSURES="${EXPOSURES:-}"
MAX_SEARCHES="${MAX_SEARCHES:-4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-32768}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-14336}"
ACTION_MAX_NEW_TOKENS="${ACTION_MAX_NEW_TOKENS:-8192}"
ANSWER_MAX_NEW_TOKENS="${ANSWER_MAX_NEW_TOKENS:-8192}"
REPAIR_MAX_NEW_TOKENS=0
NUM_GPUS="${NUM_GPUS:-2}"
WORKERS_PER_GPU="${WORKERS_PER_GPU:-2}"
NUM_WORKERS="$((NUM_GPUS * WORKERS_PER_GPU))"
[[ "$NUM_GPUS" == "2" ]] || {
  echo "Formal two-GPU run requires NUM_GPUS=2; got $NUM_GPUS" >&2
  exit 2
}
TIMEOUT="${TIMEOUT:-900}"
RETRY_FAILURES="${RETRY_FAILURES:-0}"
RESOURCE_INTERVAL_SECONDS="${RESOURCE_INTERVAL_SECONDS:-300}"
LOG_DIR="$RUN_DIR/tmux_logs"
mkdir -p "$LOG_DIR" "$ROOT_RUN_DIR/shards" "$TRAJECTORY_RUN_DIR/shards"

optional=()
[[ "$RETRY_FAILURES" == "1" ]] && optional+=(--retry_failures)

check_inputs() {
  [[ -s "$QUESTIONS" ]] || { echo "Missing $QUESTIONS; run download_hotpotqa.sh" >&2; exit 2; }
  [[ -s "$LABELS" ]] || { echo "Missing sealed labels $LABELS" >&2; exit 2; }
}

resource_monitor() {
  while true; do
    {
      echo
      echo "[$(date --iso-8601=seconds)] HOTPOT RESOURCE CHECK"
      free -h
      df -h /home/researcher
      nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
        --format=csv,noheader
      for db in "$ROOT_RUN_DIR"/shards/*/state.sqlite \
                "$TRAJECTORY_RUN_DIR"/shards/*/state.sqlite; do
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
    --retrieval_cache "$RETRIEVAL_CACHE" --model_url "http://127.0.0.1:$BASE_PORT" \
    --model_path "$MODEL_PATH" --retriever_url "$RETRIEVER_URL" \
    --limit "$LIMIT" --shard_id 0 --num_shards "$NUM_WORKERS" --dry_run
  if [[ -s "$ROOTS" ]]; then
    "$PYTHON_BIN" "$SCRIPT_DIR/run_trajectories.py" \
      --roots "$ROOTS" --output_dir "$TRAJECTORY_RUN_DIR/dry" \
      --retrieval_cache "$RETRIEVAL_CACHE" --model_url "http://127.0.0.1:$BASE_PORT" \
      --model_path "$MODEL_PATH" --retriever_url "$RETRIEVER_URL" \
      --exposures "$EXPOSURES" --limit "$LIMIT" --max_searches "$MAX_SEARCHES" \
      --shard_id 0 --num_shards "$NUM_WORKERS" --dry_run
  else
    count="$(wc -l < "$QUESTIONS" | tr -d ' ')"
    if [[ "$LIMIT" != "0" && "$LIMIT" -lt "$count" ]]; then count="$LIMIT"; fi
    echo "Questions: $count. Exact trajectory count is root-specific and is reported after roots."
    echo "Each root uses 100-token caps plus one exact FULL endpoint."
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
        --shard_id "$shard" --num_shards "$NUM_WORKERS" --timeout "$TIMEOUT" \
        "${optional[@]}"
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

run_trajectories() {
  [[ -s "$ROOTS" ]] || { echo "Missing merged roots: $ROOTS" >&2; exit 2; }
  pids=()
  for shard in $(seq 0 $((NUM_WORKERS - 1))); do
    port="$((BASE_PORT + (shard % NUM_GPUS)))"
    (
      "$PYTHON_BIN" "$SCRIPT_DIR/run_trajectories.py" \
        --roots "$ROOTS" --output_dir "$TRAJECTORY_RUN_DIR/shards/shard_$shard" \
        --retrieval_cache "$RETRIEVAL_CACHE" \
        --model_url "http://127.0.0.1:$port" --model_path "$MODEL_PATH" \
        --retriever_url "$RETRIEVER_URL" --exposures "$EXPOSURES" --limit "$LIMIT" \
        --max_searches "$MAX_SEARCHES" --context_length "$CONTEXT_LENGTH" \
        --server_max_total_tokens "$SERVER_MAX_TOTAL_TOKENS" \
        --action_max_new_tokens "$ACTION_MAX_NEW_TOKENS" \
        --answer_max_new_tokens "$ANSWER_MAX_NEW_TOKENS" \
        --repair_max_new_tokens "$REPAIR_MAX_NEW_TOKENS" \
        --timeout "$TIMEOUT" --shard_id "$shard" --num_shards "$NUM_WORKERS" \
        "${optional[@]}"
    ) >"$LOG_DIR/trajectory_worker_$shard.log" 2>&1 &
    pids+=("$!")
    echo "Started trajectory shard=$shard pid=$! port=$port"
  done
  failed=0
  for index in "${!pids[@]}"; do
    wait "${pids[$index]}" || { echo "Trajectory shard $index failed" >&2; failed=1; }
  done
  [[ "$failed" == "0" ]] || exit 1
  shard_dirs=()
  for shard in $(seq 0 $((NUM_WORKERS - 1))); do
    shard_dirs+=("$TRAJECTORY_RUN_DIR/shards/shard_$shard")
  done
  "$PYTHON_BIN" "$SCRIPT_DIR/merge_trajectories.py" \
    --roots "$ROOTS" --shard_dirs "${shard_dirs[@]}" \
    --output "$TRAJECTORIES" --exposures "$EXPOSURES" --limit "$LIMIT" \
    >"$LOG_DIR/merge_trajectories.log" 2>&1
}

analyze() {
  [[ -s "$TRAJECTORIES" ]] || { echo "Missing $TRAJECTORIES" >&2; exit 2; }
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_hotpotqa.py" \
    --trajectories "$TRAJECTORIES" --labels "$LABELS" \
    --output_dir "$RUN_DIR/analysis" >"$LOG_DIR/analyze.log" 2>&1
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze_answer_flips.py" \
    --scored_records "$RUN_DIR/analysis/scored_records.jsonl.gz" \
    --output_dir "$RUN_DIR/analysis/answer_flip" \
    --primary_metric answer_em --bootstrap_repetitions 10000 --seed 2026 \
    >"$LOG_DIR/analyze_answer_flips.log" 2>&1
  "$PYTHON_BIN" "$SCRIPT_DIR/locate_supporting_evidence.py" \
    --trajectories "$TRAJECTORIES" --labels "$LABELS" \
    --model_path "$MODEL_PATH" --output_dir "$RUN_DIR/analysis/evidence_positions" \
    >"$LOG_DIR/locate_supporting_evidence.log" 2>&1
}

case "${1:-all}" in
  dry-run) dry_run ;;
  roots) run_roots ;;
  trajectories) run_trajectories ;;
  analyze) analyze ;;
  all)
    check_inputs
    resource_monitor & monitor_pid="$!"
    trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
    dry_run
    run_roots
    run_trajectories
    analyze
    echo "HOTPOT MULTISEARCH BUDGET EXPERIMENT COMPLETE: $RUN_DIR"
    ;;
  *) echo "Usage: $0 {dry-run|roots|trajectories|analyze|all}" >&2; exit 2 ;;
esac
