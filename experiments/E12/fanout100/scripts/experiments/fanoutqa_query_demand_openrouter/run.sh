#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INPUT="${INPUT:-$REPO_ROOT/data/fanoutqa/query_prior/fanoutqa_random100_seed2026_queries.json}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/outputs/query_prior/fanoutqa_random100_gpt56sol_v1}"
MODEL="${MODEL:-openai/gpt-5.6-sol}"
WORKERS="${WORKERS:-4}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-1}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
REASONING_EFFORT="${REASONING_EFFORT:-medium}"

common=(
  --input "$INPUT"
  --output-dir "$RUN_DIR"
  --model "$MODEL"
  --workers "$WORKERS"
  --max-attempts "$MAX_ATTEMPTS"
  --max-tokens "$MAX_TOKENS"
  --reasoning-effort "$REASONING_EFFORT"
  --expected-samples 100
)

mode="${1:-run}"
case "$mode" in
  self-test)
    "$PYTHON_BIN" "$SCRIPT_DIR/self_test.py"
    ;;
  build-input)
    SOURCE_QUESTIONS="${SOURCE_QUESTIONS:?Set SOURCE_QUESTIONS to questions.no_labels.jsonl}"
    "$PYTHON_BIN" "$SCRIPT_DIR/build_query_file.py" \
      --input "$SOURCE_QUESTIONS" --output "$INPUT" --expected-samples 100
    ;;
  plan)
    "$PYTHON_BIN" "$SCRIPT_DIR/score_queries_openrouter.py" "${common[@]}" --plan
    ;;
  preflight)
    "$PYTHON_BIN" "$SCRIPT_DIR/score_queries_openrouter.py" "${common[@]}" --preflight-only
    ;;
  run|resume)
    "$PYTHON_BIN" "$SCRIPT_DIR/score_queries_openrouter.py" "${common[@]}"
    ;;
  validate)
    "$PYTHON_BIN" "$SCRIPT_DIR/score_queries_openrouter.py" "${common[@]}" --validate-only
    ;;
  analyze)
    "$PYTHON_BIN" "$SCRIPT_DIR/analyze_scores.py" \
      --input "$RUN_DIR/results/combined_scores.jsonl" \
      --output-dir "$RUN_DIR/analysis"
    ;;
  *)
    echo "Usage: $0 {self-test|build-input|plan|preflight|run|resume|validate|analyze}" >&2
    exit 2
    ;;
esac
