#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}"
MODEL_PATH="${MODEL_PATH:-/home/researcher/smartsearch_rebuild/models/Qwen3-8B}"
QUERIES_SOURCE="${QUERIES_SOURCE:-$SCRIPT_DIR/data/hotpot_random500_queries.jsonl}"
LABELS_SOURCE="${LABELS_SOURCE:-/home/researcher/smartsearch_rebuild/data/hotpotqa/prepared/gold_labels.sealed.jsonl}"
CORPUS_PATH="${CORPUS_PATH:-/home/researcher/smartsearch_rebuild/retrieval/wiki18_100w.jsonl}"
INDEX_PATH="${INDEX_PATH:-/home/researcher/smartsearch_rebuild/retrieval/wiki18_100w_e5_index/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/e5_flat_inner.index}"
RUN_DIR="${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/adaptive_k_hotpot500_qwen3_e5_v1/full500_r2_t06}"
RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:6008}"
MODEL_URLS="${MODEL_URLS:-http://127.0.0.1:9103,http://127.0.0.1:9104}"
LIMIT="${LIMIT:-0}"
CANDIDATE_COUNT="${CANDIDATE_COUNT:-1000}"
ROLLOUTS="${ROLLOUTS:-2}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
SERVER_MAX_TOTAL_TOKENS="${SERVER_MAX_TOTAL_TOKENS:-24497}"

DATASET_DIR="$RUN_DIR/dataset"
ROOTS_DIR="$RUN_DIR/roots"
SHARDS_DIR="$RUN_DIR/generations/shards"
MERGED="$RUN_DIR/generations/generations.merged.jsonl"
ANALYSIS_DIR="$RUN_DIR/analysis"
LOG_DIR="$RUN_DIR/tmux_logs"
mkdir -p "$DATASET_DIR" "$ROOTS_DIR" "$SHARDS_DIR" "$ANALYSIS_DIR" "$LOG_DIR"

prepare() {
  "$PYTHON_BIN" "$SCRIPT_DIR/prepare_dataset.py" \
    --queries "$QUERIES_SOURCE" --labels "$LABELS_SOURCE" \
    --output_dir "$DATASET_DIR" --limit "$LIMIT"
}

self_test() {
  "$PYTHON_BIN" "$SCRIPT_DIR/self_test.py"
}

plan() {
  prepare
  self_test
  "$PYTHON_BIN" "$SCRIPT_DIR/plan.py" \
    --questions "$DATASET_DIR/questions.no_labels.jsonl" \
    --labels "$DATASET_DIR/gold_labels.sealed.jsonl" \
    --model_path "$MODEL_PATH" --corpus "$CORPUS_PATH" --index "$INDEX_PATH" \
    --limit "$LIMIT" --output "$RUN_DIR/PLAN.json"
}

roots() {
  prepare
  "$PYTHON_BIN" "$SCRIPT_DIR/build_roots.py" \
    --questions "$DATASET_DIR/questions.no_labels.jsonl" \
    --output_dir "$ROOTS_DIR" --retriever_url "$RETRIEVER_URL" \
    --model_path "$MODEL_PATH" --candidate_count "$CANDIDATE_COUNT" \
    --ignore_extreme_tail 0.1 --retrieve_more 5 --limit "$LIMIT"
}

generate() {
  [[ -s "$ROOTS_DIR/manifest.json" && -s "$ROOTS_DIR/roots.jsonl" ]] || {
    echo "Missing complete roots; run roots stage first." >&2; exit 2;
  }
  "$PYTHON_BIN" - "$ROOTS_DIR/manifest.json" <<'PY'
import json, sys
from pathlib import Path
x = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert x["complete"] and x["unresolved_failures"] == 0, x
PY
  IFS=',' read -r -a URL_ARRAY <<< "$MODEL_URLS"
  [[ "${#URL_ARRAY[@]}" == 2 ]] || { echo "MODEL_URLS must contain exactly two endpoints" >&2; exit 2; }
  pids=()
  for shard in 0 1; do
    "$PYTHON_BIN" "$SCRIPT_DIR/run_generation.py" \
      --roots "$ROOTS_DIR/roots.jsonl" --output_dir "$SHARDS_DIR" \
      --model_url "${URL_ARRAY[$shard]}" --model_path "$MODEL_PATH" \
      --shard_id "$shard" --num_shards 2 --rollouts "$ROLLOUTS" \
      --temperature "$TEMPERATURE" --top_p "$TOP_P" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --server_max_total_tokens "$SERVER_MAX_TOTAL_TOKENS" \
      >"$LOG_DIR/generation_shard_${shard}.log" 2>&1 &
    pids+=("$!")
    echo "Started generation shard=$shard pid=${pids[-1]} url=${URL_ARRAY[$shard]}"
  done
  failed=0
  for index in 0 1; do
    if ! wait "${pids[$index]}"; then
      echo "Generation shard $index failed; inspect $LOG_DIR/generation_shard_${index}.log" >&2
      failed=1
    fi
  done
  (( failed == 0 )) || exit 1
  "$PYTHON_BIN" "$SCRIPT_DIR/merge_results.py" \
    --questions "$DATASET_DIR/questions.no_labels.jsonl" \
    --roots "$ROOTS_DIR/roots.jsonl" --shards_dir "$SHARDS_DIR" \
    --rollouts "$ROLLOUTS" --num_shards 2 --output "$MERGED"
}

analyze() {
  [[ -s "$MERGED" ]] || { echo "Missing $MERGED" >&2; exit 2; }
  "$PYTHON_BIN" "$SCRIPT_DIR/analyze.py" \
    --questions "$DATASET_DIR/questions.no_labels.jsonl" \
    --labels "$DATASET_DIR/gold_labels.sealed.jsonl" \
    --roots "$ROOTS_DIR/roots.jsonl" --generations "$MERGED" \
    --output_dir "$ANALYSIS_DIR"
}

audit() {
  "$PYTHON_BIN" "$SCRIPT_DIR/audit.py" \
    --questions "$DATASET_DIR/questions.no_labels.jsonl" \
    --roots "$ROOTS_DIR/roots.jsonl" --generations "$MERGED" \
    --analysis "$ANALYSIS_DIR/summary.json" --output "$RUN_DIR/FINAL_AUDIT.json"
}

full() {
  self_test
  plan
  roots
  generate
  analyze
  audit
  echo "ADAPTIVE-K RUN COMPLETE: $RUN_DIR"
}

case "${1:-}" in
  prepare) prepare ;;
  self-test) self_test ;;
  plan) plan ;;
  roots) roots ;;
  generate) generate ;;
  analyze) analyze ;;
  audit) audit ;;
  full) full ;;
  *) echo "Usage: $0 {prepare|self-test|plan|roots|generate|analyze|audit|full}" >&2; exit 2 ;;
esac
