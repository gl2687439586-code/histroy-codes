#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
ORIGINAL_RUN_DIR="${ORIGINAL_RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/blank_answer_only/bamboogle_qwen3_v5_full_6144}"
ORIGINAL_RECORDS="${ORIGINAL_RECORDS:-$ORIGINAL_RUN_DIR/records.jsonl}"
RETRY_RUN_DIR="${RETRY_RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/blank_answer_only/bamboogle_qwen3_v5_full_6144_invalid_retry_v1}"
BASE_PORT="${BASE_PORT:-9101}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-16384}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-10086}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6144}"
TIMEOUT="${TIMEOUT:-900}"
EXPECTED_INVALID="${EXPECTED_INVALID:-271}"
EXPECTED_SOURCE_SHA256="${EXPECTED_SOURCE_SHA256:-68c5be70bed953053cc9cdcddba8cc08ab6922f1f7aed09319d9b41dbdfc6042}"
RETRY_FAILURES="${RETRY_FAILURES:-0}"
LOG_DIR="$RETRY_RUN_DIR/tmux_logs"
mkdir -p "$LOG_DIR" "$RETRY_RUN_DIR/shards"

[[ -f "$ORIGINAL_RECORDS" ]] || {
  echo "Missing original records: $ORIGINAL_RECORDS" >&2
  exit 2
}

for port in $(seq "$BASE_PORT" $((BASE_PORT + 3))); do
  body="$(curl -fsS "http://127.0.0.1:$port/get_model_info" || true)"
  [[ "$body" == *"Qwen3-8B"* ]] || {
    echo "Port $port is not a verified Qwen3-8B endpoint." >&2
    exit 2
  }
done

"$PYTHON_BIN" "$SCRIPT_DIR/run_invalid_retry.py" \
  --original_records "$ORIGINAL_RECORDS" \
  --output_dir "$RETRY_RUN_DIR/dry_run" \
  --model_url "http://127.0.0.1:$BASE_PORT" \
  --model_path "$MODEL_PATH" \
  --context_length "$CONTEXT_LENGTH" \
  --server_max_total_tokens "$SERVER_MAX_TOTAL_TOKENS" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --expected_invalid "$EXPECTED_INVALID" \
  --expected_source_sha256 "$EXPECTED_SOURCE_SHA256" \
  --shard_id 0 --num_shards 4 --dry_run \
  | tee "$LOG_DIR/retry_plan.json"

started_epoch="$(date +%s)"
pids=()
for shard in 0 1 2 3; do
  port="$((BASE_PORT + shard))"
  optional=()
  [[ "$RETRY_FAILURES" == "1" ]] && optional+=(--retry_failures)
  (
    "$PYTHON_BIN" "$SCRIPT_DIR/run_invalid_retry.py" \
      --original_records "$ORIGINAL_RECORDS" \
      --output_dir "$RETRY_RUN_DIR/shards/shard_$shard" \
      --model_url "http://127.0.0.1:$port" \
      --model_path "$MODEL_PATH" \
      --context_length "$CONTEXT_LENGTH" \
      --server_max_total_tokens "$SERVER_MAX_TOTAL_TOKENS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --temperature 0.0 --seed 2026 --timeout "$TIMEOUT" \
      --expected_invalid "$EXPECTED_INVALID" \
      --expected_source_sha256 "$EXPECTED_SOURCE_SHA256" \
      --shard_id "$shard" --num_shards 4 \
      "${optional[@]}"
  ) >"$LOG_DIR/worker_$shard.log" 2>&1 &
  pids+=("$!")
  echo "Started retry shard=$shard pid=$! port=$port"
done

failed=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "Retry shard $index failed; inspect $LOG_DIR/worker_$index.log" >&2
    failed=1
  fi
done
[[ "$failed" == "0" ]] || exit 1

"$PYTHON_BIN" "$SCRIPT_DIR/merge_and_analyze_retry.py" \
  --original_records "$ORIGINAL_RECORDS" \
  --shard_dirs \
    "$RETRY_RUN_DIR/shards/shard_0" \
    "$RETRY_RUN_DIR/shards/shard_1" \
    "$RETRY_RUN_DIR/shards/shard_2" \
    "$RETRY_RUN_DIR/shards/shard_3" \
  --output_dir "$RETRY_RUN_DIR/analysis" \
  --expected_invalid "$EXPECTED_INVALID" \
  --expected_source_sha256 "$EXPECTED_SOURCE_SHA256" \
  >"$LOG_DIR/merge_and_analyze.log" 2>&1

finished_epoch="$(date +%s)"
echo "Exact-prompt invalid retry completed in $((finished_epoch-started_epoch)) seconds."
echo "Report: $RETRY_RUN_DIR/analysis/RETRY_REPORT_ZH.md"

