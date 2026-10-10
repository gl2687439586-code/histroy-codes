#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/researcher/smartsearch_rebuild/SmartSearch
PYTHON_BIN=${PYTHON_BIN:-/home/researcher/smartsearch_rebuild/envs/src-core/bin/python}
EXP="$ROOT/scripts/experiments/hotpot_read_demand_openrouter"
INPUT=${INPUT:-$ROOT/hotpot_random500_queries.jsonl}
RUN_DIR=${RUN_DIR:-/home/researcher/smartsearch_rebuild/outputs/query_prior/hotpot_random500_openrouter_gpt56sol_read_v1}

MODE=${1:-run}

common=(
  --input "$INPUT"
  --output-dir "$RUN_DIR"
  --expected-samples 500
  --model openai/gpt-5.6-sol
  --workers "${WORKERS:-1}"
  --timeout "${TIMEOUT:-180}"
  --max-attempts "${MAX_ATTEMPTS:-1}"
  --reasoning-effort "${REASONING_EFFORT:-medium}"
  --max-tokens "${MAX_TOKENS:-2048}"
  --seed "${SEED:-2026}"
)

case "$MODE" in
  self-test)
    exec "$PYTHON_BIN" "$EXP/self_test.py"
    ;;
  plan)
    exec "$PYTHON_BIN" "$EXP/score_hotpot_read_openrouter.py" "${common[@]}" --plan
    ;;
  preflight)
    exec "$PYTHON_BIN" "$EXP/score_hotpot_read_openrouter.py" "${common[@]}" --preflight-only
    ;;
  run)
    exec "$PYTHON_BIN" "$EXP/score_hotpot_read_openrouter.py" "${common[@]}"
    ;;
  validate)
    exec "$PYTHON_BIN" "$EXP/score_hotpot_read_openrouter.py" "${common[@]}" --validate-only
    ;;
  analyze)
    exec "$PYTHON_BIN" "$EXP/analyze_read_scores.py" \
      --input "$RUN_DIR/results/read_scores_combined.jsonl" \
      --output-dir "$RUN_DIR/analysis"
    ;;
  *)
    echo "Usage: $0 {self-test|plan|preflight|run|validate|analyze}" >&2
    exit 2
    ;;
esac
